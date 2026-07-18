from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

THRESHOLD = 0.50
TOP_ERROR_PERCENT = 20
RANDOM_STATE = 42
MIN_K = 2
MAX_K = 8
TOP_EFFECT_FEATURES = None
FEATURE_SET = "05_Add_MedTrajectory"


def import_ml_script(script_path: Path) -> Any:
    if not script_path.exists():
        raise FileNotFoundError(f"Model script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("ml_llm_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model script: {script_path}")
    ml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml)
    return ml


def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def select_top_error_patients(
    patient_pred: pd.DataFrame,
    top_percent: float = TOP_ERROR_PERCENT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < top_percent < 100:
        raise ValueError("top_percent must be between 0 and 100.")
    ranked = patient_pred.sort_values(
        ["prediction_error", "Patient_ID"],
        ascending=[False, True],
    ).reset_index(drop=True)
    remove_n = int(np.ceil(len(ranked) * top_percent / 100))
    removed = ranked.iloc[:remove_n].copy()
    retained = ranked.iloc[remove_n:].copy()
    removed["error_rank"] = np.arange(1, len(removed) + 1)
    removed["removed_in_top_percent"] = 1
    retained["error_rank"] = np.arange(len(removed) + 1, len(ranked) + 1)
    retained["removed_in_top_percent"] = 0
    return removed, retained


def rebuild_final_test_predictions(
    ml_script: Path,
    input_path: Path,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    ml = import_ml_script(ml_script)
    df = ml.load_data(input_path)
    df_1yr = ml.build_1yr_patient_dataset(df)
    X, y, feature_sets = ml.build_feature_matrix(df_1yr)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=ml.TEST_SIZE,
        random_state=ml.SEED,
        stratify=y,
    )

    if FEATURE_SET not in feature_sets:
        raise KeyError(f"Feature set not found: {FEATURE_SET}")

    selection_cols = ml.expand_feature_columns(
        X_train,
        feature_sets[FEATURE_SET],
    )

    from sklearn.metrics import roc_auc_score

    models = ml.make_models(y_train)
    model_rows = []
    for model_name, model in models.items():
        pipeline = ml.make_pipeline_for_model(model_name, model)
        pipeline.fit(X_train[selection_cols], y_train)
        probabilities = pipeline.predict_proba(X_test[selection_cols])[:, 1]
        model_rows.append({
            "Model": model_name,
            "AUC": float(roc_auc_score(y_test, probabilities)),
        })

    model_comparison = (
        pd.DataFrame(model_rows)
        .sort_values("AUC", ascending=False)
        .reset_index(drop=True)
    )
    best_model_name = str(model_comparison.loc[0, "Model"])
    best_model = models[best_model_name]

    final_raw_features = feature_sets[FEATURE_SET]
    final_cols = ml.expand_feature_columns(X_train, final_raw_features)

    final_pipeline = ml.make_pipeline_for_model(best_model_name, best_model)
    final_pipeline.fit(X_train[final_cols], y_train)

    X_test_all = X_test.copy().reset_index(drop=True)
    X_test_final = X_test[final_cols].copy().reset_index(drop=True)
    pred_prob = final_pipeline.predict_proba(X_test[final_cols])[:, 1]
    pred_label = (pred_prob >= THRESHOLD).astype(int)

    patient_pred = pd.DataFrame({
        "Patient_ID": df_1yr.loc[X_test.index, "Patient_ID"].astype(str).values,
        "true_msk_1yr": y_test.astype(int).values,
        "pred_prob": pred_prob,
        "pred_label": pred_label,
    })
    patient_pred["prediction_error"] = np.abs(
        patient_pred["true_msk_1yr"] - patient_pred["pred_prob"]
    )
    conditions = [
        (patient_pred["true_msk_1yr"] == 1) & (patient_pred["pred_label"] == 1),
        (patient_pred["true_msk_1yr"] == 0) & (patient_pred["pred_label"] == 0),
        (patient_pred["true_msk_1yr"] == 0) & (patient_pred["pred_label"] == 1),
        (patient_pred["true_msk_1yr"] == 1) & (patient_pred["pred_label"] == 0),
    ]
    patient_pred["prediction_type"] = np.select(
        conditions,
        ["TP", "TN", "FP", "FN"],
        default="Unknown",
    )
    patient_pred["selected_model"] = best_model_name
    patient_pred["final_feature_set"] = FEATURE_SET
    return ml, patient_pred, X_test_all, X_test_final, final_cols


def evaluate_kmeans_k(
    X_scaled: np.ndarray,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
) -> pd.DataFrame:
    upper_k = min(max_k, X_scaled.shape[0] - 1)
    if upper_k < min_k:
        raise ValueError("Too few removed patients to evaluate clustering.")
    rows = []
    for k in range(min_k, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(X_scaled)
        rows.append({
            "K": k,
            "Inertia": float(model.inertia_),
            "Silhouette_Score": float(silhouette_score(X_scaled, labels)),
        })
    return pd.DataFrame(rows)


def choose_optimal_k(k_evaluation: pd.DataFrame) -> int:
    return int(
        k_evaluation.sort_values(
            ["Silhouette_Score", "K"],
            ascending=[False, True],
        ).iloc[0]["K"]
    )


def create_cluster_size_table(clustered_patients: pd.DataFrame) -> pd.DataFrame:
    total_n = len(clustered_patients)
    table = (
        clustered_patients.groupby("KMeans_Cluster", as_index=False)
        .agg(
            N=("Patient_ID", "size"),
            Mean_Prediction_Error=("prediction_error", "mean"),
            Median_Prediction_Error=("prediction_error", "median"),
            Mean_Predicted_Probability=("pred_prob", "mean"),
            MSK_Outcome_Rate=("true_msk_1yr", "mean"),
        )
    )
    table["Percent_of_Removed_Patients"] = 100 * table["N"] / total_n
    return table


def create_cluster_feature_summary(
    clustered_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    total_removed = len(clustered_features)
    cluster_sizes = (
        clustered_features
        .groupby("KMeans_Cluster")
        .size()
        .to_dict()
    )

    for feature in feature_cols:
        values = pd.to_numeric(
            clustered_features[feature],
            errors="coerce",
        )
        unique_values = set(values.dropna().unique().tolist())
        is_binary = (
            len(unique_values) <= 2
            and unique_values.issubset({0, 1})
        )

        for cluster_id, subset in clustered_features.groupby(
            "KMeans_Cluster"
        ):
            cluster_values = pd.to_numeric(
                subset[feature],
                errors="coerce",
            ).dropna()

            if cluster_values.empty:
                continue

            cluster_n = int(cluster_sizes[cluster_id])
            cluster_percent = (
                100 * cluster_n / total_removed
                if total_removed > 0
                else np.nan
            )

            rows.append({
                "Feature": feature,
                "Feature_Type": (
                    "Binary"
                    if is_binary
                    else "Continuous/count"
                ),
                "KMeans_Cluster": int(cluster_id),
                "Cluster_N": cluster_n,
                "Cluster_Percent_of_Removed_Patients": cluster_percent,
                "N_Nonmissing_for_Feature": int(len(cluster_values)),
                "Mean_or_Proportion": float(cluster_values.mean()),
                "SD": (
                    float(cluster_values.std(ddof=1))
                    if len(cluster_values) > 1
                    else 0.0
                ),
                "Median": float(cluster_values.median()),
                "Q1": float(cluster_values.quantile(0.25)),
                "Q3": float(cluster_values.quantile(0.75)),
            })

    return pd.DataFrame(rows)


def create_cluster_feature_tests(
    clustered_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    cluster_ids = sorted(clustered_features["KMeans_Cluster"].unique())
    for feature in feature_cols:
        values = pd.to_numeric(clustered_features[feature], errors="coerce")
        unique_values = set(values.dropna().unique().tolist())
        is_binary = len(unique_values) <= 2 and unique_values.issubset({0, 1})
        if is_binary:
            contingency = []
            for cluster_id in cluster_ids:
                cluster_values = values[
                    clustered_features["KMeans_Cluster"] == cluster_id
                ].dropna()
                contingency.append([
                    int((cluster_values == 0).sum()),
                    int((cluster_values == 1).sum()),
                ])
            try:
                statistic, p_value, _, _ = stats.chi2_contingency(contingency)
            except (ValueError, ZeroDivisionError):
                statistic, p_value = np.nan, np.nan
            test_name = "Chi-square"
        else:
            arrays = [
                values[clustered_features["KMeans_Cluster"] == cluster_id].dropna()
                for cluster_id in cluster_ids
            ]
            arrays = [array for array in arrays if len(array) > 0]
            try:
                statistic, p_value = stats.kruskal(*arrays) if len(arrays) >= 2 else (np.nan, np.nan)
            except ValueError:
                statistic, p_value = np.nan, np.nan
            test_name = "Kruskal-Wallis"
        rows.append({
            "Feature": feature,
            "Feature_Type": "Binary" if is_binary else "Continuous/count",
            "Test": test_name,
            "Statistic": statistic,
            "P_Value": p_value,
            "Significant_P_lt_0_05": bool(p_value < 0.05) if pd.notna(p_value) else False,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["Significant_P_lt_0_05", "P_Value"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
    return result


def classify_feature_domain(feature: str) -> str:
    name = feature.lower()

    if name in {"age", "sex"}:
        return "Demographics"

    if (
        "mental_diagnosis" in name
        or "overall_mental_group" in name
        or "recent_30d_mental_visit" in name
    ):
        return "Mental health"

    if (
        "pre_existing" in name
        or "multimorbidity" in name
    ):
        return "Comorbidity"

    if (
        "n_mental_drugs" in name
        or "n_pain_drugs" in name
        or "med_addition" in name
        or "med_discontinuation" in name
        or "mental_pain_overlap" in name
        or "medclass_" in name
    ):
        return "Medication"

    return "Other"


def rank_biserial_from_mann_whitney(
    removed_values: pd.Series,
    retained_values: pd.Series,
) -> float:
  
    removed = pd.to_numeric(
        removed_values,
        errors="coerce",
    ).dropna()

    retained = pd.to_numeric(
        retained_values,
        errors="coerce",
    ).dropna()

    n_removed = len(removed)
    n_retained = len(retained)

    if n_removed == 0 or n_retained == 0:
        return np.nan

    try:
        u_statistic = stats.mannwhitneyu(
            removed,
            retained,
            alternative="two-sided",
        ).statistic
    except ValueError:
        return np.nan

    return float(
        (2 * u_statistic) / (
            n_removed * n_retained
        )
        - 1
    )


def cramers_v_2x2(
    removed_binary: pd.Series,
    retained_binary: pd.Series,
) -> float:
   
    removed = pd.to_numeric(
        removed_binary,
        errors="coerce",
    ).dropna()

    retained = pd.to_numeric(
        retained_binary,
        errors="coerce",
    ).dropna()

    contingency = np.array([
        [
            int((removed == 0).sum()),
            int((removed == 1).sum()),
        ],
        [
            int((retained == 0).sum()),
            int((retained == 1).sum()),
        ],
    ])

    if contingency.sum() == 0:
        return np.nan

    try:
        chi2, _, _, _ = stats.chi2_contingency(
            contingency
        )
    except (
        ValueError,
        ZeroDivisionError,
    ):
        return np.nan

    total = contingency.sum()
    magnitude = float(
        np.sqrt(
            chi2 / total
        )
    )

    removed_rate = (
        float((removed == 1).mean())
        if len(removed) > 0
        else np.nan
    )
    retained_rate = (
        float((retained == 1).mean())
        if len(retained) > 0
        else np.nan
    )

    if pd.isna(removed_rate) or pd.isna(retained_rate):
        return magnitude

    sign = 1.0 if removed_rate >= retained_rate else -1.0
    return sign * magnitude


def interpret_effect_size(
    value: float,
) -> str:
    if pd.isna(value):
        return ""

    magnitude = abs(value)

    if magnitude < 0.10:
        return "Negligible"
    if magnitude < 0.30:
        return "Small"
    if magnitude < 0.50:
        return "Moderate"
    return "Large"


def create_removed_vs_retained_feature_comparison(
    all_patient_features: pd.DataFrame,
    final_cols: list[str],
) -> pd.DataFrame:
    """
    Compare the top 20% highest-error patients with the retained 80%.

    Binary features:
        Chi-square test + signed Cramer's V.

    Continuous/count features:
        Mann-Whitney U test + rank-biserial correlation.
    """
    rows = []

    for feature in final_cols:
        values = pd.to_numeric(
            all_patient_features[feature],
            errors="coerce",
        )

        unique_values = set(
            values.dropna().unique().tolist()
        )
        is_binary = (
            len(unique_values) <= 2
            and unique_values.issubset({0, 1})
        )

        removed_values = values[
            all_patient_features[
                "Removed_Top20"
            ] == 1
        ].dropna()

        retained_values = values[
            all_patient_features[
                "Removed_Top20"
            ] == 0
        ].dropna()

        total_test_n = len(all_patient_features)
        removed_group_n = int(
            (all_patient_features["Removed_Top20"] == 1).sum()
        )
        retained_group_n = int(
            (all_patient_features["Removed_Top20"] == 0).sum()
        )

        row = {
            "Feature": feature,
            "Clinical_Domain": classify_feature_domain(
                feature
            ),
            "Feature_Type": (
                "Binary"
                if is_binary
                else "Continuous/count"
            ),
            "Removed_Group_N": removed_group_n,
            "Removed_Group_Percent_of_Test": (
                100 * removed_group_n / total_test_n
                if total_test_n > 0
                else np.nan
            ),
            "Retained_Group_N": retained_group_n,
            "Retained_Group_Percent_of_Test": (
                100 * retained_group_n / total_test_n
                if total_test_n > 0
                else np.nan
            ),
            "Removed_N_Nonmissing_for_Feature": len(removed_values),
            "Retained_N_Nonmissing_for_Feature": len(retained_values),
        }

        if is_binary:
            removed_n1 = int(
                (removed_values == 1).sum()
            )
            retained_n1 = int(
                (retained_values == 1).sum()
            )

            removed_pct = (
                100 * removed_n1 / len(removed_values)
                if len(removed_values) > 0
                else np.nan
            )
            retained_pct = (
                100 * retained_n1 / len(retained_values)
                if len(retained_values) > 0
                else np.nan
            )

            contingency = [
                [
                    int((removed_values == 0).sum()),
                    removed_n1,
                ],
                [
                    int((retained_values == 0).sum()),
                    retained_n1,
                ],
            ]

            try:
                statistic, p_value, _, _ = (
                    stats.chi2_contingency(
                        contingency
                    )
                )
            except (
                ValueError,
                ZeroDivisionError,
            ):
                statistic = np.nan
                p_value = np.nan

            effect_size = cramers_v_2x2(
                removed_values,
                retained_values,
            )

            row.update({
                "Removed_Summary": (
                    f"{removed_n1} "
                    f"({removed_pct:.1f}%)"
                ),
                "Retained_Summary": (
                    f"{retained_n1} "
                    f"({retained_pct:.1f}%)"
                ),
                "Removed_Mean_or_Proportion": (
                    removed_n1 / len(removed_values)
                    if len(removed_values) > 0
                    else np.nan
                ),
                "Retained_Mean_or_Proportion": (
                    retained_n1 / len(retained_values)
                    if len(retained_values) > 0
                    else np.nan
                ),
                "Difference_Removed_Minus_Retained": (
                    removed_pct - retained_pct
                ),
                "Test": "Chi-square",
                "Statistic": statistic,
                "P_Value": p_value,
                "Effect_Size_Type": "Signed Cramer's V",
                "Effect_Size": effect_size,
            })

        else:
            try:
                statistic, p_value = (
                    stats.mannwhitneyu(
                        removed_values,
                        retained_values,
                        alternative="two-sided",
                    )
                    if (
                        len(removed_values) > 0
                        and len(retained_values) > 0
                    )
                    else (np.nan, np.nan)
                )
            except ValueError:
                statistic = np.nan
                p_value = np.nan

            effect_size = rank_biserial_from_mann_whitney(
                removed_values,
                retained_values,
            )

            removed_mean = (
                float(removed_values.mean())
                if len(removed_values) > 0
                else np.nan
            )
            retained_mean = (
                float(retained_values.mean())
                if len(retained_values) > 0
                else np.nan
            )

            row.update({
                "Removed_Summary": (
                    f"{removed_mean:.2f} ± "
                    f"{removed_values.std(ddof=1):.2f}"
                    if len(removed_values) > 1
                    else f"{removed_mean:.2f}"
                ),
                "Retained_Summary": (
                    f"{retained_mean:.2f} ± "
                    f"{retained_values.std(ddof=1):.2f}"
                    if len(retained_values) > 1
                    else f"{retained_mean:.2f}"
                ),
                "Removed_Mean_or_Proportion": (
                    removed_mean
                ),
                "Retained_Mean_or_Proportion": (
                    retained_mean
                ),
                "Difference_Removed_Minus_Retained": (
                    removed_mean - retained_mean
                ),
                "Test": "Mann-Whitney U",
                "Statistic": statistic,
                "P_Value": p_value,
                "Effect_Size_Type": (
                    "Rank-biserial correlation"
                ),
                "Effect_Size": effect_size,
            })

        row["Absolute_Effect_Size"] = (
            abs(row["Effect_Size"])
            if pd.notna(row["Effect_Size"])
            else np.nan
        )
        row["Effect_Size_Magnitude"] = (
            interpret_effect_size(
                row["Effect_Size"]
            )
        )
        row["Direction"] = (
            "Higher/more common in removed patients"
            if (
                pd.notna(row["Effect_Size"])
                and row["Effect_Size"] > 0
            )
            else "Higher/more common in retained patients"
            if (
                pd.notna(row["Effect_Size"])
                and row["Effect_Size"] < 0
            )
            else "No directional difference"
        )
        row["Significant_P_lt_0_05"] = (
            bool(row["P_Value"] < 0.05)
            if pd.notna(row["P_Value"])
            else False
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            [
                "Absolute_Effect_Size",
                "P_Value",
            ],
            ascending=[
                False,
                True,
            ],
            na_position="last",
        ).reset_index(drop=True)

    return result


def create_removed_vs_retained_domain_summary(
    feature_comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for domain, subset in feature_comparison.groupby(
        "Clinical_Domain"
    ):
        rows.append({
            "Clinical_Domain": domain,
            "N_Features": len(subset),
            "N_Significant_Features": int(
                subset[
                    "Significant_P_lt_0_05"
                ].sum()
            ),
            "Mean_Absolute_Effect_Size": float(
                subset[
                    "Absolute_Effect_Size"
                ].mean()
            ),
            "Maximum_Absolute_Effect_Size": float(
                subset[
                    "Absolute_Effect_Size"
                ].max()
            ),
            "Top_Feature": str(
                subset.sort_values(
                    "Absolute_Effect_Size",
                    ascending=False,
                ).iloc[0]["Feature"]
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Mean_Absolute_Effect_Size",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def plot_removed_vs_retained_effect_sizes(
    feature_comparison: pd.DataFrame,
    figures_dir: Path,
    top_n: int | None = None,
) -> None:
    """
    Plot signed effect sizes for removed versus retained patients.

    By default, all final-model features are displayed.
    Positive values indicate higher or more common values in removed patients.
    Negative values indicate higher or more common values in retained patients.
    """
    plot_df = (
        feature_comparison
        .dropna(subset=["Effect_Size"])
        .sort_values(
            "Absolute_Effect_Size",
            ascending=False,
        )
        .copy()
    )

    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plot_df = plot_df.sort_values(
        "Effect_Size",
        ascending=True,
    )

    plt.figure(
        figsize=(
            10,
            max(7, 0.42 * len(plot_df)),
        )
    )
    plt.barh(
        plot_df["Feature"],
        plot_df["Effect_Size"],
    )
    plt.axvline(
        0,
        linewidth=1,
    )
    plt.xlabel(
        "Signed effect size "
        "(positive = higher in removed patients)"
    )
    plt.ylabel("")
    plt.title(
        "All Final-Model Features: "
        "Removed Top 20% vs Retained 80%"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "00_removed_vs_retained_all_feature_effect_sizes.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_prediction_type_distribution_by_cluster(
    clustered_patients: pd.DataFrame,
) -> pd.DataFrame:
    """Return TP/TN/FP/FN counts and percentages within each cluster."""
    rows = []
    prediction_types = ["TP", "TN", "FP", "FN"]

    for cluster_id, subset in clustered_patients.groupby("KMeans_Cluster"):
        denominator = len(subset)

        for prediction_type in prediction_types:
            n = int(
                (subset["prediction_type"] == prediction_type).sum()
            )
            rows.append({
                "KMeans_Cluster": int(cluster_id),
                "Prediction_Type": prediction_type,
                "N": n,
                "Percent_Within_Cluster": (
                    100 * n / denominator
                    if denominator > 0
                    else np.nan
                ),
            })

    return pd.DataFrame(rows)


def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def create_cluster_profile_summary_table(
    clustered_features: pd.DataFrame,
    feature_tests: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:

    rows = []
    cluster_ids = sorted(
        clustered_features["KMeans_Cluster"].unique()
    )
    total_removed = len(clustered_features)
    cluster_sizes = (
        clustered_features
        .groupby("KMeans_Cluster")
        .size()
        .to_dict()
    )

    cluster_column_names = {
        cluster_id: (
            f"Cluster_{int(cluster_id)}"
            f"_n{int(cluster_sizes[cluster_id])}"
            f"_{100 * cluster_sizes[cluster_id] / total_removed:.1f}pct"
        )
        for cluster_id in cluster_ids
    }

    for feature in selected_features:
        values = pd.to_numeric(
            clustered_features[feature],
            errors="coerce",
        )
        unique_values = set(values.dropna().unique().tolist())
        is_binary = (
            len(unique_values) <= 2
            and unique_values.issubset({0, 1})
        )

        row = {
            "Feature": feature,
            "Feature_Type": (
                "Binary"
                if is_binary
                else "Continuous/count"
            ),
        }

        for cluster_id in cluster_ids:
            cluster_values = values[
                clustered_features["KMeans_Cluster"]
                == cluster_id
            ].dropna()

            if cluster_values.empty:
                summary = "NA"
            elif is_binary:
                n_positive = int((cluster_values == 1).sum())
                summary = (
                    f"{n_positive} "
                    f"({100 * n_positive / len(cluster_values):.1f}%)"
                )
            else:
                summary = (
                    f"{cluster_values.mean():.2f} ± "
                    f"{cluster_values.std(ddof=1):.2f}"
                    if len(cluster_values) > 1
                    else f"{cluster_values.mean():.2f}"
                )

            row[cluster_column_names[cluster_id]] = summary

        test_row = feature_tests[
            feature_tests["Feature"] == feature
        ]
        p_value = (
            float(test_row["P_Value"].iloc[0])
            if not test_row.empty
            else np.nan
        )

        row["P_Value"] = p_value
        row["P_Value_Formatted"] = format_p_value(p_value)
        rows.append(row)

    return pd.DataFrame(rows)



def find_clinical_composition_columns(
    all_feature_columns: list[str],
) -> dict[str, list[str]]:
    """
    Detect available composition variables from the complete test feature matrix.

    Medication classes are interpreted as whether a class changed during the
    one-year window when the available variable is a *_count_change feature.

    Comorbidity variables are system-specific Pre_* and Multi_* flags.
    """
    columns = list(all_feature_columns)

    medication_columns = sorted([
        column
        for column in columns
        if column.startswith("medclass_")
        and column.endswith("_count_change")
    ])

    preexisting_comorbidity_columns = sorted([
        column
        for column in columns
        if column.startswith("Pre_")
        and column
        not in {
            "Pre_existing_condition_count",
        }
    ])

    incident_comorbidity_columns = sorted([
        column
        for column in columns
        if column.startswith("Multi_")
        and column
        not in {
            "Multimorbidity_count",
        }
    ])

    sex_columns = [
        column
        for column in columns
        if column == "Sex"
    ]

    return {
        "Sex": sex_columns,
        "Medication_Class_Change": medication_columns,
        "Preexisting_Comorbidity": preexisting_comorbidity_columns,
        "Incident_Comorbidity": incident_comorbidity_columns,
    }


def make_binary_composition_indicator(
    values: pd.Series,
    composition_type: str,
) -> pd.Series:
    """
    Convert an available clinical-composition variable to a binary indicator.

    Medication class-change variables:
        1 = any increase or decrease in that medication class.
    Other composition variables:
        1 = feature is present.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    if composition_type == "Medication_Class_Change":
        return (numeric.fillna(0) != 0).astype(int)

    return (numeric.fillna(0) > 0).astype(int)


def signed_cramers_v_from_groups(
    group_a: pd.Series,
    group_b: pd.Series,
) -> float:
    """Signed Cramer's V; positive means the feature is more common in group A."""
    a = pd.to_numeric(group_a, errors="coerce").dropna()
    b = pd.to_numeric(group_b, errors="coerce").dropna()

    contingency = np.array([
        [
            int((a == 0).sum()),
            int((a == 1).sum()),
        ],
        [
            int((b == 0).sum()),
            int((b == 1).sum()),
        ],
    ])

    if contingency.sum() == 0:
        return np.nan

    try:
        chi2, _, _, _ = stats.chi2_contingency(contingency)
    except (ValueError, ZeroDivisionError):
        return np.nan

    magnitude = float(np.sqrt(chi2 / contingency.sum()))
    difference = float(a.mean() - b.mean())

    if difference > 0:
        return magnitude
    if difference < 0:
        return -magnitude
    return 0.0


def compare_binary_composition_removed_vs_retained(
    all_patient_features: pd.DataFrame,
    composition_columns: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Compare sex, medication-class changes, and comorbidity-system composition
    between removed top-20% and retained 80% patients.
    """
    rows = []

    removed_mask = all_patient_features["Removed_Top20"] == 1
    retained_mask = all_patient_features["Removed_Top20"] == 0

    removed_n = int(removed_mask.sum())
    retained_n = int(retained_mask.sum())

    for composition_type, columns in composition_columns.items():
        for feature in columns:
            indicator = make_binary_composition_indicator(
                all_patient_features[feature],
                composition_type,
            )

            removed_values = indicator[removed_mask]
            retained_values = indicator[retained_mask]

            removed_positive = int(removed_values.sum())
            retained_positive = int(retained_values.sum())

            contingency = [
                [
                    removed_n - removed_positive,
                    removed_positive,
                ],
                [
                    retained_n - retained_positive,
                    retained_positive,
                ],
            ]

            try:
                statistic, p_value, _, _ = stats.chi2_contingency(
                    contingency
                )
            except (ValueError, ZeroDivisionError):
                statistic = np.nan
                p_value = np.nan

            effect_size = signed_cramers_v_from_groups(
                removed_values,
                retained_values,
            )

            rows.append({
                "Composition_Type": composition_type,
                "Feature": feature,
                "Removed_N": removed_n,
                "Removed_Positive_N": removed_positive,
                "Removed_Percent": (
                    100 * removed_positive / removed_n
                    if removed_n > 0
                    else np.nan
                ),
                "Retained_N": retained_n,
                "Retained_Positive_N": retained_positive,
                "Retained_Percent": (
                    100 * retained_positive / retained_n
                    if retained_n > 0
                    else np.nan
                ),
                "Percentage_Point_Difference": (
                    100 * removed_positive / removed_n
                    - 100 * retained_positive / retained_n
                    if removed_n > 0 and retained_n > 0
                    else np.nan
                ),
                "Test": "Chi-square",
                "Statistic": statistic,
                "P_Value": p_value,
                "Signed_Cramers_V": effect_size,
                "Direction": (
                    "More common in removed patients"
                    if pd.notna(effect_size) and effect_size > 0
                    else "More common in retained patients"
                    if pd.notna(effect_size) and effect_size < 0
                    else "No directional difference"
                ),
                "Significant_P_lt_0_05": (
                    bool(p_value < 0.05)
                    if pd.notna(p_value)
                    else False
                ),
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["Absolute_Effect_Size"] = (
            result["Signed_Cramers_V"].abs()
        )
        result = result.sort_values(
            [
                "Composition_Type",
                "Absolute_Effect_Size",
                "P_Value",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            na_position="last",
        ).reset_index(drop=True)

    return result


def compare_binary_composition_across_clusters(
    clustered_all_features: pd.DataFrame,
    composition_columns: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Compare sex, medication-class changes, and comorbidity-system composition
    across K-means clusters.

    Each cluster cell is reported as n (%). An overall chi-square test compares
    the distributions across all clusters.
    """
    rows = []
    cluster_ids = sorted(
        clustered_all_features["KMeans_Cluster"].unique()
    )
    total_removed = len(clustered_all_features)

    cluster_sizes = (
        clustered_all_features
        .groupby("KMeans_Cluster")
        .size()
        .to_dict()
    )

    cluster_column_names = {
        cluster_id: (
            f"Cluster_{int(cluster_id)}"
            f"_n{int(cluster_sizes[cluster_id])}"
            f"_{100 * cluster_sizes[cluster_id] / total_removed:.1f}pct"
        )
        for cluster_id in cluster_ids
    }

    for composition_type, columns in composition_columns.items():
        for feature in columns:
            indicator = make_binary_composition_indicator(
                clustered_all_features[feature],
                composition_type,
            )

            contingency = []
            row = {
                "Composition_Type": composition_type,
                "Feature": feature,
            }

            cluster_percentages = []

            for cluster_id in cluster_ids:
                cluster_mask = (
                    clustered_all_features["KMeans_Cluster"]
                    == cluster_id
                )
                cluster_values = indicator[cluster_mask]
                cluster_n = len(cluster_values)
                positive_n = int(cluster_values.sum())
                positive_pct = (
                    100 * positive_n / cluster_n
                    if cluster_n > 0
                    else np.nan
                )

                contingency.append([
                    cluster_n - positive_n,
                    positive_n,
                ])

                row[
                    cluster_column_names[cluster_id]
                ] = (
                    f"{positive_n} ({positive_pct:.1f}%)"
                    if cluster_n > 0
                    else "NA"
                )
                cluster_percentages.append(positive_pct)

            try:
                statistic, p_value, _, _ = stats.chi2_contingency(
                    contingency
                )
            except (ValueError, ZeroDivisionError):
                statistic = np.nan
                p_value = np.nan

            row.update({
                "Test": "Chi-square",
                "Statistic": statistic,
                "P_Value": p_value,
                "Maximum_Percentage_Point_Range": (
                    float(
                        np.nanmax(cluster_percentages)
                        - np.nanmin(cluster_percentages)
                    )
                    if cluster_percentages
                    else np.nan
                ),
                "Significant_P_lt_0_05": (
                    bool(p_value < 0.05)
                    if pd.notna(p_value)
                    else False
                ),
            })
            rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            [
                "Composition_Type",
                "Maximum_Percentage_Point_Range",
                "P_Value",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            na_position="last",
        ).reset_index(drop=True)

    return result


def create_clinical_composition_dictionary(
    composition_columns: dict[str, list[str]],
) -> pd.DataFrame:
    """Document how every clinical-composition variable is interpreted."""
    rows = []

    for composition_type, columns in composition_columns.items():
        for feature in columns:
            if composition_type == "Medication_Class_Change":
                interpretation = (
                    "Any non-zero count change during the 1-year window"
                )
            elif composition_type == "Sex":
                interpretation = (
                    "Binary Sex value coded as 1 in the model matrix"
                )
            elif composition_type == "Preexisting_Comorbidity":
                interpretation = (
                    "Pre-existing comorbidity-system flag is present"
                )
            else:
                interpretation = (
                    "Incident comorbidity-system flag is present"
                )

            rows.append({
                "Composition_Type": composition_type,
                "Feature": feature,
                "Binary_Indicator_Definition": interpretation,
            })

    return pd.DataFrame(rows)



def plot_k_selection(
    k_evaluation: pd.DataFrame,
    optimal_k: int,
    figures_dir: Path,
) -> None:
    """Save silhouette and elbow plots with the selected K highlighted."""
    plt.figure(figsize=(7, 5))
    plt.plot(
        k_evaluation["K"],
        k_evaluation["Silhouette_Score"],
        marker="o",
    )
    best_row = k_evaluation[
        k_evaluation["K"] == optimal_k
    ].iloc[0]
    plt.scatter(
        [optimal_k],
        [best_row["Silhouette_Score"]],
        s=110,
        marker="o",
        zorder=4,
        label=f"Selected K = {optimal_k}",
    )
    plt.axvline(
        optimal_k,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.title("K-means Cluster Selection")
    plt.xticks(k_evaluation["K"])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "01_kmeans_silhouette_by_k.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(
        k_evaluation["K"],
        k_evaluation["Inertia"],
        marker="o",
    )
    plt.scatter(
        [optimal_k],
        [best_row["Inertia"]],
        s=110,
        marker="o",
        zorder=4,
        label=f"Selected K = {optimal_k}",
    )
    plt.axvline(
        optimal_k,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Within-cluster sum of squares")
    plt.title("K-means Elbow Plot")
    plt.xticks(k_evaluation["K"])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "02_kmeans_elbow_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_pca_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    figures_dir: Path,
) -> pd.DataFrame:
    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )
    coordinates = pca.fit_transform(X_scaled)

    pca_table = pd.DataFrame({
        "PCA1": coordinates[:, 0],
        "PCA2": coordinates[:, 1],
        "KMeans_Cluster": labels,
    })

    plt.figure(figsize=(9, 7))

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_x = coordinates[mask, 0]
        cluster_y = coordinates[mask, 1]

        plt.scatter(
            cluster_x,
            cluster_y,
            label=f"Cluster {cluster_id}",
            alpha=0.45,
            s=32,
        )

        plt.scatter(
            [cluster_x.mean()],
            [cluster_y.mean()],
            marker="X",
            s=180,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    plt.axhline(
        0,
        linewidth=0.8,
        alpha=0.35,
    )
    plt.axvline(
        0,
        linewidth=0.8,
        alpha=0.35,
    )
    plt.xlabel(
        "PCA 1 "
        f"({100 * pca.explained_variance_ratio_[0]:.1f}% variance)"
    )
    plt.ylabel(
        "PCA 2 "
        f"({100 * pca.explained_variance_ratio_[1]:.1f}% variance)"
    )
    plt.title(
        "Clinical Profiles of the Top 20% Highest-Error Patients"
    )
    plt.legend(
        title="K-means cluster",
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir / "03_pca_kmeans_clusters.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return pca_table


def run_cluster_analysis(
    ml_script: Path,
    input_path: Path,
    outdir: Path,
    top_error_percent: float = TOP_ERROR_PERCENT,
) -> None:
    results_dir = outdir

    tables_dir = (
        results_dir
        / "tables"
        / "cluster_analysis"
    )
    figures_dir = (
        results_dir
        / "figures"
        / "cluster_analysis"
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        _,
        patient_pred,
        X_test_all,
        X_test_final,
        final_cols,
    ) = rebuild_final_test_predictions(
        ml_script=ml_script,
        input_path=input_path,
    )

    removed_patients, retained_patients = select_top_error_patients(
        patient_pred,
        top_percent=top_error_percent,
    )

    feature_table = pd.concat(
        [
            patient_pred[
                ["Patient_ID"]
            ].reset_index(drop=True),
            X_test_final.reset_index(drop=True),
        ],
        axis=1,
    )

    all_feature_table = pd.concat(
        [
            patient_pred[
                ["Patient_ID"]
            ].reset_index(drop=True),
            X_test_all.reset_index(drop=True),
        ],
        axis=1,
    )

    removed_id_set = set(
        removed_patients[
            "Patient_ID"
        ].astype(str)
    )

    all_patient_features = (
        patient_pred[
            [
                "Patient_ID",
                "true_msk_1yr",
                "pred_prob",
                "pred_label",
                "prediction_error",
                "prediction_type",
            ]
        ]
        .merge(
            feature_table,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )
    )

    all_patient_features[
        "Removed_Top20"
    ] = (
        all_patient_features[
            "Patient_ID"
        ]
        .astype(str)
        .isin(
            removed_id_set
        )
        .astype(int)
    )

    composition_columns = find_clinical_composition_columns(
        X_test_all.columns.tolist()
    )

    composition_feature_names = sorted({
        feature
        for columns in composition_columns.values()
        for feature in columns
    })

    composition_patient_features = all_patient_features[
        [
            "Patient_ID",
            "Removed_Top20",
        ]
    ].copy()

    composition_patient_features = (
        composition_patient_features
        .merge(
            all_feature_table[
                ["Patient_ID"]
                + composition_feature_names
            ],
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )
    )

    removed_vs_retained = (
        create_removed_vs_retained_feature_comparison(
            all_patient_features,
            final_cols,
        )
    )

    removed_vs_retained.to_csv(
        tables_dir
        / "01_removed_vs_retained_feature_comparison.csv",
        index=False,
    )

    create_removed_vs_retained_domain_summary(
        removed_vs_retained
    ).to_csv(
        tables_dir
        / "02_removed_vs_retained_domain_summary.csv",
        index=False,
    )

    all_patient_features.to_csv(
        tables_dir
        / "03_all_test_patients_with_removal_status.csv",
        index=False,
    )

    create_clinical_composition_dictionary(
        composition_columns
    ).to_csv(
        tables_dir
        / "12_clinical_composition_variable_dictionary.csv",
        index=False,
    )

    compare_binary_composition_removed_vs_retained(
        composition_patient_features,
        composition_columns,
    ).to_csv(
        tables_dir
        / "13_clinical_composition_removed_vs_retained.csv",
        index=False,
    )

    plot_removed_vs_retained_effect_sizes(
        removed_vs_retained,
        figures_dir,
        top_n=TOP_EFFECT_FEATURES,
    )

    removed_features = removed_patients[["Patient_ID"]].merge(
        feature_table,
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )

    X_removed = safe_numeric_frame(removed_features[final_cols])
    variances = X_removed.var(axis=0)
    clustering_cols = variances[variances > 0].index.tolist()
    zero_variance_cols = variances[variances <= 0].index.tolist()
    if len(clustering_cols) < 2:
        raise RuntimeError(
            "Fewer than two non-constant features are available for clustering."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_removed[clustering_cols])

    k_evaluation = evaluate_kmeans_k(X_scaled)
    optimal_k = choose_optimal_k(k_evaluation)

    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=RANDOM_STATE,
        n_init=100,
    )
    kmeans_labels = kmeans.fit_predict(X_scaled) + 1

    hierarchical = AgglomerativeClustering(
        n_clusters=optimal_k,
        linkage="ward",
    )
    hierarchical_labels = hierarchical.fit_predict(X_scaled) + 1
    cluster_agreement = adjusted_rand_score(kmeans_labels, hierarchical_labels)

    removed_clustered = removed_patients.reset_index(drop=True).copy()
    removed_clustered["KMeans_Cluster"] = kmeans_labels
    removed_clustered["Hierarchical_Cluster"] = hierarchical_labels

    clustered_features = pd.concat(
        [removed_clustered.reset_index(drop=True), X_removed.reset_index(drop=True)],
        axis=1,
    )

    removed_all_features = (
        removed_patients[
            ["Patient_ID"]
        ]
        .merge(
            all_feature_table,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    clustered_all_features = pd.concat(
        [
            removed_clustered[
                [
                    "Patient_ID",
                    "KMeans_Cluster",
                    "Hierarchical_Cluster",
                ]
            ].reset_index(drop=True),
            removed_all_features[
                composition_feature_names
            ].reset_index(drop=True),
        ],
        axis=1,
    )


    pd.DataFrame([{
        "Feature_Set": FEATURE_SET,
        "Top_Error_Percent": top_error_percent,
        "N_Test_Patients": len(patient_pred),
        "N_Removed_Patients": len(removed_patients),
        "N_Final_Model_Features": len(final_cols),
        "N_Nonconstant_Clustering_Features": len(clustering_cols),
        "Optimal_K": optimal_k,
        "KMeans_Silhouette": float(silhouette_score(X_scaled, kmeans_labels)),
        "KMeans_Hierarchical_Adjusted_Rand_Index": float(cluster_agreement),
    }]).to_csv(tables_dir / "04_cluster_analysis_summary.csv", index=False)
    clustered_features.to_csv(tables_dir / "07_removed_patients_clusters_with_all_features.csv", index=False)

    create_cluster_size_table(removed_clustered).to_csv(
        tables_dir / "05_cluster_sizes_and_outcomes.csv", index=False
    )
    create_cluster_feature_summary(
        clustered_features,
        final_cols,
    ).to_csv(
        tables_dir
        / "09_all_feature_summary_by_cluster.csv",
        index=False,
    )
    cluster_feature_tests = create_cluster_feature_tests(
        clustered_features,
        final_cols,
    )
    cluster_feature_tests.to_csv(
        tables_dir
        / "10_all_feature_tests_across_clusters.csv",
        index=False,
    )

    prediction_type_distribution = (
        create_prediction_type_distribution_by_cluster(
            removed_clustered
        )
    )
    prediction_type_distribution.to_csv(
        tables_dir
        / "06_prediction_type_distribution_by_cluster.csv",
        index=False,
    )

    plot_k_selection(
        k_evaluation,
        optimal_k,
        figures_dir,
    )
    pca_table = plot_pca_clusters(X_scaled, kmeans_labels, figures_dir)
    pd.concat(
        [
            removed_clustered[[
                "Patient_ID",
                "prediction_error",
                "prediction_type",
                "KMeans_Cluster",
                "Hierarchical_Cluster",
            ]].reset_index(drop=True),
            pca_table[["PCA1", "PCA2"]].reset_index(drop=True),
        ],
        axis=1,
    ).to_csv(tables_dir / "08_pca_coordinates_and_clusters.csv", index=False)


    create_cluster_profile_summary_table(
        clustered_features,
        cluster_feature_tests,
        final_cols,
    ).to_csv(
        tables_dir
        / "11_all_feature_cluster_profile_with_n_percent.csv",
        index=False,
    )

    compare_binary_composition_across_clusters(
        clustered_all_features,
        composition_columns,
    ).to_csv(
        tables_dir
        / "14_clinical_composition_by_cluster.csv",
        index=False,
    )

    print("\nCluster analysis completed.")
    print("Top error patients selected:", len(removed_patients))
    print("Final model features:", len(final_cols))
    print("Nonconstant clustering features:", len(clustering_cols))
    print("Zero-variance features excluded:", len(zero_variance_cols))
    print("Optimal K:", optimal_k)
    print(
        "K-means silhouette:",
        round(
            float(
                silhouette_score(
                    X_scaled,
                    kmeans_labels,
                )
            ),
            4,
        ),
    )
    print(
        "K-means vs hierarchical ARI:",
        round(
            float(cluster_agreement),
            4,
        ),
    )
    print("\nSaved tables to:", tables_dir)
    print("Saved figures to:", figures_dir)


def parse_args() -> argparse.Namespace:
    default_root = Path(
        os.environ.get(
            "BIOSTAT_ROOT",
            "/Users/xinyichen/Desktop/BiostatisticsPracticum_Depression",
        )
    )

    parser = argparse.ArgumentParser(
        description=(
            "Cluster analysis of the top 20% highest prediction-error patients "
            "identified in the MSK sensitivity analysis."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            default_root
            / "final_cohort"
            / "results"
            / "tables"
            / "final_analysis_long.csv"
        ),
        help="Path to final_analysis_long.csv.",
    )
    parser.add_argument(
        "--ml_script",
        type=Path,
        default=default_root / "final_cohort" / "ml_llm.py",
        help="Path to the main ml_llm.py script.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=(
            default_root
            / "final_cohort"
            / "results"
        ),
        help=(
            "Path to final_cohort/results. Tables are saved under "
            "results/tables/cluster_analysis and figures under "
            "results/figures/cluster_analysis."
        ),
    )
    parser.add_argument(
        "--top_error_percent",
        type=float,
        default=TOP_ERROR_PERCENT,
        help="Percentage of highest prediction-error test patients to cluster. Default: 20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cluster_analysis(
        ml_script=args.ml_script,
        input_path=args.input,
        outdir=args.outdir,
        top_error_percent=args.top_error_percent,
    )


if __name__ == "__main__":
    main()
