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
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


THRESHOLD = 0.50
REMOVAL_PERCENTAGES = (0, 5, 10, 15, 20)

APPLICABILITY_ORDER = [
    "High applicability",
    "Intermediate applicability",
    "Low applicability",
]

QUARTILE_ORDER = ["Q1", "Q2", "Q3", "Q4"]
PREDICTION_TYPE_ORDER = ["TP", "TN", "FP", "FN"]


def import_ml_script(script_path: Path) -> Any:
    if not script_path.exists():
        raise FileNotFoundError(f"Model script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("ml_llm_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model script: {script_path}")

    ml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml)
    return ml


def safe_auc(y_true, y_prob) -> float:

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_prob))


def get_prediction_metrics(
    y_true,
    y_prob,
    threshold: float = THRESHOLD,
) -> dict[str, float | int]:
    """Return discrimination and threshold-based classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return {
            "AUC": np.nan,
            "Accuracy": np.nan,
            "Sensitivity": np.nan,
            "Specificity": np.nan,
            "PPV": np.nan,
            "NPV": np.nan,
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
        }

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "AUC": safe_auc(y_true, y_prob),
        "Accuracy": float((tp + tn) / len(y_true)),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "PPV": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def add_prediction_type(patient_pred: pd.DataFrame) -> pd.DataFrame:
    """Label every test-set prediction as TP, TN, FP, or FN."""
    df = patient_pred.copy()

    conditions = [
        (df["true_msk_1yr"] == 1) & (df["pred_label"] == 1),
        (df["true_msk_1yr"] == 0) & (df["pred_label"] == 0),
        (df["true_msk_1yr"] == 0) & (df["pred_label"] == 1),
        (df["true_msk_1yr"] == 1) & (df["pred_label"] == 0),
    ]

    df["prediction_type"] = np.select(
        conditions,
        PREDICTION_TYPE_ORDER,
        default="Unknown",
    )
    return df


def assign_error_quartiles_and_applicability(
    patient_pred: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Assign prediction-error quartiles and broader applicability groups.

    Quartile definitions:
        Q1: error <= 25th percentile
        Q2: 25th percentile < error <= median
        Q3: median < error < 75th percentile
        Q4: error >= 75th percentile

    Applicability definitions:
        Q1       -> High applicability
        Q2 + Q3  -> Intermediate applicability
        Q4       -> Low applicability

    Boundary values are retained as patient-level columns for transparency.
    Ties at a boundary may cause quartile counts to differ slightly from 25%.
    """
    df = patient_pred.copy()

    errors = pd.to_numeric(df["prediction_error"], errors="coerce")
    if errors.isna().any():
        raise ValueError("prediction_error contains missing or non-numeric values.")

    q1 = float(errors.quantile(0.25))
    q2 = float(errors.quantile(0.50))
    q3 = float(errors.quantile(0.75))

    df["error_q1_cutoff_25pct"] = q1
    df["error_q2_cutoff_50pct"] = q2
    df["error_q3_cutoff_75pct"] = q3

    quartile_conditions = [
        errors <= q1,
        (errors > q1) & (errors <= q2),
        (errors > q2) & (errors < q3),
        errors >= q3,
    ]

    df["prediction_error_quartile"] = np.select(
        quartile_conditions,
        QUARTILE_ORDER,
        default="Unassigned",
    )

    if (df["prediction_error_quartile"] == "Unassigned").any():
        raise RuntimeError("Some patients could not be assigned to an error quartile.")

    applicability_map = {
        "Q1": "High applicability",
        "Q2": "Intermediate applicability",
        "Q3": "Intermediate applicability",
        "Q4": "Low applicability",
    }
    df["applicability_group"] = df["prediction_error_quartile"].map(
        applicability_map
    )

    cutoffs = {
        "Minimum_error": float(errors.min()),
        "Q1_25th_percentile": q1,
        "Q2_50th_percentile_median": q2,
        "Q3_75th_percentile": q3,
        "Maximum_error": float(errors.max()),
    }

    return df, cutoffs


def create_error_quartile_interval_table(
    patient_pred: pd.DataFrame,
    cutoffs: dict[str, float],
) -> pd.DataFrame:
    """
    Create a transparent Q1-Q4 table containing:
      - percentile-based interval definition,
      - observed minimum and maximum error in each quartile,
      - sample size and percentage,
      - corresponding applicability group.
    """
    total_n = len(patient_pred)

    q1 = cutoffs["Q1_25th_percentile"]
    q2 = cutoffs["Q2_50th_percentile_median"]
    q3 = cutoffs["Q3_75th_percentile"]

    interval_definitions = {
        "Q1": f"prediction_error <= {q1:.6f}",
        "Q2": f"{q1:.6f} < prediction_error <= {q2:.6f}",
        "Q3": f"{q2:.6f} < prediction_error < {q3:.6f}",
        "Q4": f"prediction_error >= {q3:.6f}",
    }

    applicability_map = {
        "Q1": "High applicability",
        "Q2": "Intermediate applicability",
        "Q3": "Intermediate applicability",
        "Q4": "Low applicability",
    }

    rows = []
    for quartile in QUARTILE_ORDER:
        subset = patient_pred[
            patient_pred["prediction_error_quartile"] == quartile
        ]
        n = len(subset)

        rows.append({
            "Prediction_Error_Quartile": quartile,
            "Applicability_Group": applicability_map[quartile],
            "Percentile_Interval_Definition": interval_definitions[quartile],
            "Observed_Min_Prediction_Error": (
                subset["prediction_error"].min() if n > 0 else np.nan
            ),
            "Observed_Max_Prediction_Error": (
                subset["prediction_error"].max() if n > 0 else np.nan
            ),
            "Mean_Prediction_Error": (
                subset["prediction_error"].mean() if n > 0 else np.nan
            ),
            "Median_Prediction_Error": (
                subset["prediction_error"].median() if n > 0 else np.nan
            ),
            "N": n,
            "Percent_of_Test_Set": (
                100 * n / total_n if total_n > 0 else np.nan
            ),
        })

    return pd.DataFrame(rows)


def create_error_cutoff_table(cutoffs: dict[str, float]) -> pd.DataFrame:
    """Create a compact table of the overall prediction-error cutoffs."""
    return pd.DataFrame([
        {
            "Minimum_Error": cutoffs["Minimum_error"],
            "Q1_25th_Percentile": cutoffs["Q1_25th_percentile"],
            "Median_50th_Percentile": cutoffs["Q2_50th_percentile_median"],
            "Q3_75th_Percentile": cutoffs["Q3_75th_percentile"],
            "Maximum_Error": cutoffs["Maximum_error"],
        }
    ])

def add_profile_completeness(
    df_1yr: pd.DataFrame,
    X: pd.DataFrame,
    patient_pred: pd.DataFrame,
    raw_feature_cols: list[str],
    model_feature_cols: list[str],
) -> pd.DataFrame:
    """Attach selected patient characteristics and feature-completeness counts."""
    df_profile = df_1yr.copy().reset_index(drop=True)
    df_profile["Patient_ID"] = df_profile["Patient_ID"].astype(str)

    raw_available = [
        col for col in raw_feature_cols if col in df_profile.columns
    ]

    if raw_available:
        raw_values = df_profile[raw_available].copy()
        missing_mask = raw_values.apply(
            lambda col: col.astype(str).str.strip().isin(
                ["", "nan", "NaN", "None", "NA", "<NA>"]
            )
        )
        df_profile["raw_missing_feature_count"] = missing_mask.sum(axis=1)
        df_profile["raw_available_feature_count"] = (
            len(raw_available) - df_profile["raw_missing_feature_count"]
        )
    else:
        df_profile["raw_missing_feature_count"] = 0
        df_profile["raw_available_feature_count"] = 0

    X_model = X[model_feature_cols].copy().reset_index(drop=True)
    X_model = X_model.apply(pd.to_numeric, errors="coerce").fillna(0)

    df_profile["model_nonzero_feature_count"] = (X_model != 0).sum(axis=1)
    df_profile["model_zero_feature_count"] = (X_model == 0).sum(axis=1)

    keep_cols = [
        "Patient_ID",
        "Age",
        "Sex",
        "window_days",
        "Mental_Diagnosis_count",
        "Recent_30d_mental_visit",
        "Overall_Mental_group",
        "Pre_existing_condition_count",
        "Multimorbidity_count",
        "n_mental_drugs",
        "n_pain_drugs",
        "med_addition_count",
        "med_discontinuation_count",
        "mental_pain_overlap_ratio",
        "raw_missing_feature_count",
        "raw_available_feature_count",
        "model_nonzero_feature_count",
        "model_zero_feature_count",
    ]
    keep_cols = [col for col in keep_cols if col in df_profile.columns]

    return patient_pred.merge(
        df_profile[keep_cols],
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )


def create_group_percentage_table(
    patient_pred: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize applicability-group size and observed prediction-error range."""
    total_n = len(patient_pred)
    rows = []

    for group in APPLICABILITY_ORDER:
        subset = patient_pred[
            patient_pred["applicability_group"] == group
        ]
        n = len(subset)

        rows.append({
            "Applicability_Group": group,
            "Included_Quartiles": {
                "High applicability": "Q1",
                "Intermediate applicability": "Q2 + Q3",
                "Low applicability": "Q4",
            }[group],
            "N": n,
            "Percent_of_Test_Set": (
                100 * n / total_n if total_n > 0 else np.nan
            ),
            "Observed_Min_Prediction_Error": (
                subset["prediction_error"].min() if n > 0 else np.nan
            ),
            "Observed_Max_Prediction_Error": (
                subset["prediction_error"].max() if n > 0 else np.nan
            ),
        })

    return pd.DataFrame(rows)


def create_quartile_performance_table(
    patient_pred: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate performance separately in Q1, Q2, Q3, and Q4."""
    rows = []

    for quartile in QUARTILE_ORDER:
        subset = patient_pred[
            patient_pred["prediction_error_quartile"] == quartile
        ]
        metrics = get_prediction_metrics(
            subset["true_msk_1yr"],
            subset["pred_prob"],
        )

        rows.append({
            "Prediction_Error_Quartile": quartile,
            "Applicability_Group": (
                subset["applicability_group"].iloc[0]
                if len(subset) > 0 else ""
            ),
            "N": len(subset),
            "Mean_Prediction_Error": subset["prediction_error"].mean(),
            "Median_Prediction_Error": subset["prediction_error"].median(),
            **metrics,
        })

    return pd.DataFrame(rows)


def create_group_performance_table(
    patient_pred: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate performance within each broader applicability group."""
    rows = []

    for group in APPLICABILITY_ORDER:
        subset = patient_pred[
            patient_pred["applicability_group"] == group
        ]
        metrics = get_prediction_metrics(
            subset["true_msk_1yr"],
            subset["pred_prob"],
        )

        rows.append({
            "Applicability_Group": group,
            "N": len(subset),
            "Mean_Prediction_Error": subset["prediction_error"].mean(),
            "Median_Prediction_Error": subset["prediction_error"].median(),
            **metrics,
        })

    return pd.DataFrame(rows)


def create_prediction_type_distribution(
    patient_pred: pd.DataFrame,
    group_column: str,
    group_order: list[str],
) -> pd.DataFrame:
    """Return TP/TN/FP/FN counts and percentages within each requested group."""
    rows = []

    for group in group_order:
        subset = patient_pred[patient_pred[group_column] == group]
        denominator = len(subset)

        for prediction_type in PREDICTION_TYPE_ORDER:
            n = int((subset["prediction_type"] == prediction_type).sum())
            rows.append({
                "Grouping_Variable": group_column,
                "Group": group,
                "Prediction_Type": prediction_type,
                "N": n,
                "Percent_Within_Group": (
                    100 * n / denominator if denominator > 0 else np.nan
                ),
            })

    return pd.DataFrame(rows)


def create_exploratory_q4_removal_table(
    patient_pred: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    settings = [
        ("All test patients", []),
        ("Exploratory removal of Q4", ["Q4"]),
    ]

    for analysis_set, quartiles_to_remove in settings:
        retained = patient_pred[
            ~patient_pred["prediction_error_quartile"].isin(
                quartiles_to_remove
            )
        ].copy()

        removed = patient_pred[
            patient_pred["prediction_error_quartile"].isin(
                quartiles_to_remove
            )
        ].copy()

        metrics = get_prediction_metrics(
            retained["true_msk_1yr"],
            retained["pred_prob"],
        )

        rows.append({
            "Analysis_Set": analysis_set,
            "Removed_Quartiles": (
                "; ".join(quartiles_to_remove)
                if quartiles_to_remove else "None"
            ),
            "N_Remaining": len(retained),
            "N_Removed": len(removed),
            "Percent_Removed": (
                100 * len(removed) / len(patient_pred)
                if len(patient_pred) > 0 else np.nan
            ),
            **metrics,
        })

    return pd.DataFrame(rows)


def create_progressive_error_removal_table(
    patient_pred: pd.DataFrame,
) -> pd.DataFrame:
    """

    Progressively remove 5%, 10%, 15%, and 20% of test patients with the
    largest observed prediction errors, then recalculate performance among
    the remaining patients.

    Exact rank-based removal avoids unpredictable sample sizes when several
    patients share the same prediction-error value. This remains a
    retrospective held-out-test-set analysis because the ranking uses the
    observed outcome.
    """
    sorted_df = patient_pred.sort_values(
        ["prediction_error", "Patient_ID"],
        ascending=[False, True],
    ).reset_index(drop=True)

    total_n = len(sorted_df)
    rows = []

    for pct in REMOVAL_PERCENTAGES:
        remove_n = int(np.ceil(total_n * pct / 100)) if pct > 0 else 0

        removed = sorted_df.iloc[:remove_n].copy()
        retained = sorted_df.iloc[remove_n:].copy()

        metrics = get_prediction_metrics(
            retained["true_msk_1yr"],
            retained["pred_prob"],
        )

        rows.append({
            "Requested_Removal_Percent": pct,
            "Actual_Removal_Percent": (
                100 * remove_n / total_n if total_n > 0 else np.nan
            ),
            "N_Remaining": len(retained),
            "N_Removed": remove_n,
            "Lowest_Error_Among_Removed": (
                removed["prediction_error"].min()
                if len(removed) > 0 else np.nan
            ),
            "Highest_Error_Among_Remaining": (
                retained["prediction_error"].max()
                if len(retained) > 0 else np.nan
            ),
            **metrics,
        })

    return pd.DataFrame(rows)


def _format_mean_sd(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return "NA"

    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.2f} ± {sd:.2f}"


def _format_median_iqr(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return "NA"

    q1 = values.quantile(0.25)
    median = values.median()
    q3 = values.quantile(0.75)
    return f"{median:.2f} ({q1:.2f}, {q3:.2f})"


def _format_n_pct(n: int, denominator: int) -> str:
    if denominator <= 0:
        return "0 (0.0%)"
    return f"{int(n)} ({100 * n / denominator:.1f}%)"


def _format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def create_final_feature_group_comparison(
    patient_pred: pd.DataFrame,
    X_test_final: pd.DataFrame,
    final_cols: list[str],
) -> pd.DataFrame:
    """Compare all final-model features across applicability groups."""
    df = patient_pred[["Patient_ID", "applicability_group"]].reset_index(
        drop=True
    )
    X_final = X_test_final[final_cols].copy().reset_index(drop=True)
    X_final = X_final.apply(pd.to_numeric, errors="coerce").fillna(0)

    df = pd.concat([df, X_final], axis=1)
    rows = []

    for feature in final_cols:
        values = pd.to_numeric(df[feature], errors="coerce")
        unique_values = set(values.dropna().unique().tolist())
        is_binary = len(unique_values) <= 2 and unique_values.issubset({0, 1})

        row = {
            "Feature": feature,
            "Feature_Type": "Binary" if is_binary else "Continuous/count",
        }

        if is_binary:
            contingency = []

            for group in APPLICABILITY_ORDER:
                group_values = values[
                    df["applicability_group"] == group
                ].dropna()
                n1 = int((group_values == 1).sum())
                row[group] = _format_n_pct(n1, len(group_values))
                contingency.append([
                    int((group_values == 0).sum()),
                    n1,
                ])

            try:
                statistic, p_value, _, _ = stats.chi2_contingency(
                    contingency
                )
            except (ValueError, ZeroDivisionError):
                statistic, p_value = np.nan, np.nan

            row.update({
                "Summary_Format": "n (%) with value = 1",
                "Test": "Chi-square",
                "Statistic": statistic,
                "P_value": p_value,
                "P_value_formatted": _format_p_value(p_value),
                "Significant_p_lt_0.05": (
                    bool(p_value < 0.05) if pd.notna(p_value) else False
                ),
            })

        else:
            arrays = []

            for group in APPLICABILITY_ORDER:
                group_values = values[
                    df["applicability_group"] == group
                ].dropna()
                arrays.append(group_values)

                row[group] = _format_mean_sd(group_values)
                row[f"{group}_median_IQR"] = _format_median_iqr(
                    group_values
                )

            valid_arrays = [array for array in arrays if len(array) > 0]
            try:
                if len(valid_arrays) >= 2:
                    statistic, p_value = stats.kruskal(*valid_arrays)
                else:
                    statistic, p_value = np.nan, np.nan
            except ValueError:
                statistic, p_value = np.nan, np.nan

            row.update({
                "Summary_Format": "mean ± SD; median (IQR) also provided",
                "Test": "Kruskal-Wallis",
                "Statistic": statistic,
                "P_value": p_value,
                "P_value_formatted": _format_p_value(p_value),
                "Significant_p_lt_0.05": (
                    bool(p_value < 0.05) if pd.notna(p_value) else False
                ),
            })

        rows.append(row)

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.sort_values(
            ["Significant_p_lt_0.05", "P_value"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    return comparison


def compare_numeric_profiles(
    patient_pred: pd.DataFrame,
    group_a_name: str,
    group_a_mask: pd.Series,
    group_b_name: str,
    group_b_mask: pd.Series,
) -> pd.DataFrame:
    """Compare selected numeric characteristics between two patient groups."""
    numeric_cols = [
        "Age",
        "window_days",
        "Mental_Diagnosis_count",
        "Recent_30d_mental_visit",
        "Pre_existing_condition_count",
        "Multimorbidity_count",
        "n_mental_drugs",
        "n_pain_drugs",
        "med_addition_count",
        "med_discontinuation_count",
        "mental_pain_overlap_ratio",
        "raw_missing_feature_count",
        "raw_available_feature_count",
        "model_nonzero_feature_count",
        "model_zero_feature_count",
        "pred_prob",
        "prediction_error",
    ]
    numeric_cols = [
        col for col in numeric_cols if col in patient_pred.columns
    ]

    rows = []

    for col in numeric_cols:
        group_a = pd.to_numeric(
            patient_pred.loc[group_a_mask, col],
            errors="coerce",
        ).dropna()
        group_b = pd.to_numeric(
            patient_pred.loc[group_b_mask, col],
            errors="coerce",
        ).dropna()

        try:
            p_value = (
                stats.mannwhitneyu(
                    group_a,
                    group_b,
                    alternative="two-sided",
                ).pvalue
                if len(group_a) >= 2 and len(group_b) >= 2
                else np.nan
            )
        except ValueError:
            p_value = np.nan

        rows.append({
            "Feature": col,
            f"{group_a_name}_N": len(group_a),
            f"{group_a_name}_Mean": group_a.mean() if len(group_a) else np.nan,
            f"{group_a_name}_SD": (
                group_a.std(ddof=1) if len(group_a) > 1 else np.nan
            ),
            f"{group_b_name}_N": len(group_b),
            f"{group_b_name}_Mean": group_b.mean() if len(group_b) else np.nan,
            f"{group_b_name}_SD": (
                group_b.std(ddof=1) if len(group_b) > 1 else np.nan
            ),
            f"Mean_Difference_{group_a_name}_Minus_{group_b_name}": (
                group_a.mean() - group_b.mean()
                if len(group_a) and len(group_b) else np.nan
            ),
            "Mann_Whitney_U_P_value": p_value,
            "P_value_formatted": _format_p_value(p_value),
        })

    return pd.DataFrame(rows)


def plot_auc_removal_curve(
    progressive_removal: pd.DataFrame,
    figure_dir: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        progressive_removal["Actual_Removal_Percent"],
        progressive_removal["AUC"],
        marker="o",
    )
    plt.xlabel("Patients with highest prediction error removed (%)")
    plt.ylabel("AUC among remaining patients")
    plt.title("AUC After Removing the Highest-Error Patients (0%-20%)")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "20_primary_auc_after_0_to_20pct_error_removal.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_group_percentages(
    group_table: pd.DataFrame,
    figure_dir: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.bar(
        group_table["Applicability_Group"],
        group_table["Percent_of_Test_Set"],
    )
    plt.ylabel("Test patients (%)")
    plt.xlabel("Model applicability group")
    plt.title("Applicability Groups Based on Prediction Error")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "13_applicability_group_percentages.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_prediction_type_distribution(
    distribution_table: pd.DataFrame,
    figure_dir: Path,
) -> None:
    pivot = (
        distribution_table.pivot(
            index="Group",
            columns="Prediction_Type",
            values="Percent_Within_Group",
        )
        .reindex(
            index=APPLICABILITY_ORDER,
            columns=PREDICTION_TYPE_ORDER,
        )
        .fillna(0)
    )

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
    )
    ax.set_xlabel("Model applicability group")
    ax.set_ylabel("Patients within group (%)")
    ax.set_title("Prediction Outcomes Within Applicability Groups")
    ax.legend(
        title="Prediction type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "14_prediction_type_distribution_by_group.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def run_sensitivity_analysis(
    ml_script: Path,
    input_path: Path,
    table_dir: Path,
    figure_dir: Path,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    ml = import_ml_script(ml_script)

    print("Input:", input_path)
    print("Model script:", ml_script)
    print("Table directory:", table_dir)
    print("Figure directory:", figure_dir)
    print("Classification threshold:", THRESHOLD)

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

    algorithm_selection_feature_set = "06_Final_WithInteraction"
    if algorithm_selection_feature_set not in feature_sets:
        raise KeyError(
            f"Feature set not found: {algorithm_selection_feature_set}"
        )

    selection_cols = ml.expand_feature_columns(
        X_train,
        feature_sets[algorithm_selection_feature_set],
    )

    models = ml.make_models(y_train)
    model_rows = []

    for model_name, model in models.items():
        pipeline = ml.make_pipeline_for_model(model_name, model)
        pipeline.fit(X_train[selection_cols], y_train)

        probabilities = pipeline.predict_proba(
            X_test[selection_cols]
        )[:, 1]

        model_rows.append({
            "Model": model_name,
            "AUC": safe_auc(y_test, probabilities),
        })

    model_comparison = (
        pd.DataFrame(model_rows)
        .sort_values("AUC", ascending=False)
        .reset_index(drop=True)
    )
    model_comparison.to_csv(
        table_dir / "00_algorithm_selection_for_sensitivity.csv",
        index=False,
    )

    best_model_name = str(model_comparison.loc[0, "Model"])
    best_model = models[best_model_name]
    print("\nSelected model:", best_model_name)

    final_feature_set = "05_Add_MedTrajectory"
    if final_feature_set not in feature_sets:
        raise KeyError(f"Feature set not found: {final_feature_set}")

    final_raw_features = feature_sets[final_feature_set]
    final_cols = ml.expand_feature_columns(
        X_train,
        final_raw_features,
    )

    pd.DataFrame({
        "Final_Model_Feature": final_cols,
    }).to_csv(
        table_dir / "01_final_model_feature_list.csv",
        index=False,
    )

    final_pipeline = ml.make_pipeline_for_model(
        best_model_name,
        best_model,
    )
    final_pipeline.fit(X_train[final_cols], y_train)

    pred_prob = final_pipeline.predict_proba(
        X_test[final_cols]
    )[:, 1]
    pred_label = (pred_prob >= THRESHOLD).astype(int)

    patient_pred = pd.DataFrame({
        "Patient_ID": (
            df_1yr.loc[X_test.index, "Patient_ID"]
            .astype(str)
            .values
        ),
        "true_msk_1yr": y_test.astype(int).values,
        "pred_prob": pred_prob,
        "pred_label": pred_label,
    })

    patient_pred["prediction_error"] = np.abs(
        patient_pred["true_msk_1yr"] - patient_pred["pred_prob"]
    )
    patient_pred["prediction_margin_from_threshold"] = np.abs(
        patient_pred["pred_prob"] - THRESHOLD
    )
    patient_pred["correct_prediction"] = (
        patient_pred["true_msk_1yr"]
        == patient_pred["pred_label"]
    ).astype(int)

    patient_pred = add_prediction_type(patient_pred)
    patient_pred, error_cutoffs = (
        assign_error_quartiles_and_applicability(patient_pred)
    )

    patient_pred = add_profile_completeness(
        df_1yr=df_1yr.loc[X_test.index].reset_index(drop=True),
        X=X_test[final_cols].reset_index(drop=True),
        patient_pred=patient_pred,
        raw_feature_cols=final_raw_features,
        model_feature_cols=final_cols,
    )


    patient_pred.to_csv(
        table_dir / "02_patient_level_predictions_with_quartiles.csv",
        index=False,
    )

    patient_pred[
        patient_pred["applicability_group"] == "High applicability"
    ].to_csv(
        table_dir / "03_high_applicability_Q1_patients.csv",
        index=False,
    )

    patient_pred[
        patient_pred["applicability_group"]
        == "Intermediate applicability"
    ].to_csv(
        table_dir / "04_intermediate_applicability_Q2_Q3_patients.csv",
        index=False,
    )

    patient_pred[
        patient_pred["applicability_group"] == "Low applicability"
    ].to_csv(
        table_dir / "05_low_applicability_Q4_patients.csv",
        index=False,
    )

    cutoff_table = create_error_cutoff_table(error_cutoffs)
    cutoff_table.to_csv(
        table_dir / "06_prediction_error_cutoffs.csv",
        index=False,
    )

    quartile_intervals = create_error_quartile_interval_table(
        patient_pred,
        error_cutoffs,
    )
    quartile_intervals.to_csv(
        table_dir / "07_Q1_Q4_prediction_error_intervals.csv",
        index=False,
    )

    quartile_performance = create_quartile_performance_table(
        patient_pred
    )
    quartile_performance.to_csv(
        table_dir / "08_Q1_Q4_performance.csv",
        index=False,
    )


    group_summary = create_group_percentage_table(patient_pred)
    group_summary.to_csv(
        table_dir / "09_applicability_group_summary.csv",
        index=False,
    )

    group_performance = create_group_performance_table(patient_pred)
    group_performance.to_csv(
        table_dir / "10_applicability_group_performance.csv",
        index=False,
    )

    group_prediction_types = create_prediction_type_distribution(
        patient_pred,
        group_column="applicability_group",
        group_order=APPLICABILITY_ORDER,
    )
    group_prediction_types.to_csv(
        table_dir / "11_prediction_type_distribution_by_group.csv",
        index=False,
    )

    quartile_prediction_types = create_prediction_type_distribution(
        patient_pred,
        group_column="prediction_error_quartile",
        group_order=QUARTILE_ORDER,
    )
    quartile_prediction_types.to_csv(
        table_dir / "11b_prediction_type_distribution_by_quartile.csv",
        index=False,
    )


    progressive_removal = create_progressive_error_removal_table(
        patient_pred
    )
    progressive_removal.to_csv(
        table_dir / "12_primary_progressive_error_removal_0_to_20pct.csv",
        index=False,
    )

    exploratory_q4_removal = create_exploratory_q4_removal_table(
        patient_pred
    )
    exploratory_q4_removal.to_csv(
        table_dir / "13_exploratory_Q4_removal.csv",
        index=False,
    )

    low_vs_high = compare_numeric_profiles(
        patient_pred=patient_pred,
        group_a_name="Low",
        group_a_mask=(
            patient_pred["applicability_group"] == "Low applicability"
        ),
        group_b_name="High",
        group_b_mask=(
            patient_pred["applicability_group"] == "High applicability"
        ),
    )
    low_vs_high.to_csv(
        table_dir / "14_low_vs_high_characteristics.csv",
        index=False,
    )

    low_vs_others = compare_numeric_profiles(
        patient_pred=patient_pred,
        group_a_name="Low",
        group_a_mask=(
            patient_pred["applicability_group"] == "Low applicability"
        ),
        group_b_name="Others",
        group_b_mask=(
            patient_pred["applicability_group"] != "Low applicability"
        ),
    )
    low_vs_others.to_csv(
        table_dir / "15_low_vs_others_characteristics.csv",
        index=False,
    )

    final_feature_comparison = create_final_feature_group_comparison(
        patient_pred=patient_pred,
        X_test_final=X_test[final_cols].reset_index(drop=True),
        final_cols=final_cols,
    )
    final_feature_comparison.to_csv(
        table_dir / "16_final_model_feature_group_comparison.csv",
        index=False,
    )

    significant_features = final_feature_comparison[
        final_feature_comparison["Significant_p_lt_0.05"]
    ].copy()
    significant_features.to_csv(
        table_dir / "17_significant_final_model_features.csv",
        index=False,
    )


    final_metrics = get_prediction_metrics(
        patient_pred["true_msk_1yr"],
        patient_pred["pred_prob"],
    )

    pd.DataFrame([{
        "Final_Feature_Set": final_feature_set,
        "Final_Model": best_model_name,
        "Classification_Threshold": THRESHOLD,
        "N_Final_Model_Features": len(final_cols),
        **final_metrics,
    }]).to_csv(
        table_dir / "18_final_patient_level_metrics.csv",
        index=False,
    )

    plot_auc_removal_curve(progressive_removal, figure_dir)
    plot_group_percentages(group_summary, figure_dir)
    plot_prediction_type_distribution(
        group_prediction_types,
        figure_dir,
    )


    print("\nPrediction-error cutoffs:")
    print(cutoff_table.to_string(index=False))

    print("\nQ1-Q4 prediction-error intervals:")
    print(quartile_intervals.to_string(index=False))

    print("\nPerformance by Q1-Q4:")
    print(quartile_performance.to_string(index=False))

    print("\nPerformance by applicability group:")
    print(group_performance.to_string(index=False))

    print("\nPrimary sensitivity analysis: progressive 0%-20% removal:")
    print(progressive_removal.to_string(index=False))

    print("\nExploratory benchmark: removal of Q4:")
    print(exploratory_q4_removal.to_string(index=False))

    print("\nSaved tables to:", table_dir)
    print("Saved figures to:", figure_dir)



def parse_args() -> argparse.Namespace:
    default_root = Path(
        os.environ.get(
            "BIOSTAT_ROOT",
            "/Users/xinyichen/Desktop/BiostatisticsPracticum_Depression",
        )
    )

    parser = argparse.ArgumentParser(
        description=(
            "Prediction-error applicability analysis and progressive 0%-20% "
            "sensitivity analysis for the final 1-year MSK model."
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
        "--table-dir",
        type=Path,
        default=(
            default_root
            / "final_cohort"
            / "results"
            / "tables"
            / "sensitivity_analysis"
        ),
        help="Directory in which CSV table outputs will be saved.",
    )

    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=(
            default_root
            / "final_cohort"
            / "results"
            / "figures"
            / "sensitivity_analysis"
        ),
        help="Directory in which figure outputs will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sensitivity_analysis(
        ml_script=args.ml_script,
        input_path=args.input,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
    )


if __name__ == "__main__":
    main()
