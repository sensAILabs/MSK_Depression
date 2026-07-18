from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


SEED = 123
TEST_SIZE = 0.30
THRESHOLD = 0.50
N_SPLITS = 5

AGE_GROUP_BINS = [18, 35, 65, 120]
AGE_GROUP_LABELS = ["18-34", "35-64", "65+"]

CORR_THRESHOLD = 0.70


DEMOGRAPHIC_FEATURES = [
    "Age",
    "Sex",
]

MENTAL_FEATURES = [
    "Mental_Diagnosis_count",
    "Recent_30d_mental_visit",
    "Overall_Mental_group",
]

PRE_CATEGORY_FEATURES = [
    "Pre_cancer",
    "Pre_endocrine",
    "Pre_cardiovascular",
    "Pre_respiratory",
    "Pre_genitourinary",
]

MULTI_CATEGORY_FEATURES = [
    "Multi_cancer",
    "Multi_endocrine",
    "Multi_cardiovascular",
    "Multi_respiratory",
    "Multi_genitourinary",
]

COMORBIDITY_CATEGORY_FEATURES = PRE_CATEGORY_FEATURES + MULTI_CATEGORY_FEATURES

COMORBIDITY_COUNT_FEATURES = [
    "Pre_existing_condition_count",
    "Multimorbidity_count",
]

MEDICATION_COMPLEXITY_FEATURES = [
    "n_mental_drugs",
    "n_pain_drugs",
]

INTERACTION_FEATURES = [
    "Age_Sex_interaction",
]

# Dynamic medication class trajectory features.
KEY_MEDCLASS_BASE_CLASSES = [
    "antidepressant",
    "anti_anxiety",
    "antipsychotic",
    "hypnotic",
    "mood_stabilizer",
    "nsaid",
    "opioid",
    "acetaminophen",
    "muscle_relaxant",
    "neuropathic_pain",
]

KEY_MEDCLASS_CHANGE_FEATURES = [
    f"medclass_{cls}_count_change"
    for cls in KEY_MEDCLASS_BASE_CLASSES
]


TRAJECTORY_FEATURES = [
    "med_addition_count",
    "med_discontinuation_count",
    "mental_pain_overlap_ratio",
]

def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return pd.read_csv(input_path, dtype=str)


def build_1yr_patient_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Patient_ID", "window_days", "msk_1yr"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df = df.copy()
    df["window_days"] = pd.to_numeric(df["window_days"], errors="coerce")
    df["msk_1yr"] = pd.to_numeric(df["msk_1yr"], errors="coerce").fillna(0).astype(int)

    # For 1-year prediction, use the latest available row within 365 days.
    # If patient had MSK before 1yr, this will use the last retained row before/at MSK.
    df_1yr = df[df["window_days"] <= 365].copy()
    df_1yr = (
        df_1yr
        .sort_values(["Patient_ID", "window_days"])
        .groupby("Patient_ID", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    if df_1yr.empty:
        raise ValueError("No patients remain after 1-year filtering.")

    return df_1yr


def get_dynamic_medication_features(df: pd.DataFrame) -> Dict[str, List[str]]:

    medication_complexity = [
        c for c in MEDICATION_COMPLEXITY_FEATURES
        if c in df.columns
    ]

    medication_change = [
        c for c in KEY_MEDCLASS_CHANGE_FEATURES
        if c in df.columns
    ]


    return {
        "medication_complexity": medication_complexity,
        "medication_change": medication_change,
    }

def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Clinically staged feature sets.

    Design logic:
      01 Base: demographics + mental diagnosis information.
      02 Disease burden: add only overall pre-existing/multimorbidity counts.
         Disease-specific Pre_xxx and Multi_xxx indicators are intentionally excluded
         from the main comparison because prior SHAP/Pearson results suggested weak
         contribution and possible sparsity.
      03 Medication burden: add active mental/pain drug/class burden.
      04 Medication class dynamics: add class-specific cumulative change counts.
      05 Medication trajectory: add dynamic treatment trajectory variables.
      06 Final with interaction: add Age × Sex interaction.

    Optional 07 is included only as an ablation/sensitivity check to verify whether
    disease-specific comorbidity categories add anything beyond count burden.
    """

    med = get_dynamic_medication_features(df)

    base = DEMOGRAPHIC_FEATURES + MENTAL_FEATURES
    count_comorbidity = [c for c in COMORBIDITY_COUNT_FEATURES if c in df.columns]
    category_comorbidity = [
        c for c in (PRE_CATEGORY_FEATURES + MULTI_CATEGORY_FEATURES)
        if c in df.columns
    ]

    medication_burden = med["medication_complexity"]
    medication_change = med["medication_change"]
    medication_trajectory = [c for c in TRAJECTORY_FEATURES if c in df.columns]

    feature_sets = {
        "01_Base": (
            base
        ),

        "02_Add_DiseaseBurden_CountOnly": (
            base
            + count_comorbidity
        ),

        "03_Add_MedicationBurden": (
            base
            + count_comorbidity
            + medication_burden
        ),

        "04_Add_MedClassDynamics": (
            base
            + count_comorbidity
            + medication_burden
            + medication_change
        ),

        "05_Add_MedTrajectory": (
            base
            + count_comorbidity
            + medication_burden
            + medication_change
            + medication_trajectory
        ),

        "06_Final_WithInteraction": (
            base
            + INTERACTION_FEATURES
            + count_comorbidity
            + medication_burden
            + medication_change
            + medication_trajectory
        ),

        # Sensitivity check only: do disease-specific comorbidity categories help?
        "07_Check_CategoryComorbidity": (
            base
            + INTERACTION_FEATURES
            + count_comorbidity
            + category_comorbidity
            + medication_burden
            + medication_change
            + medication_trajectory
        ),
    }

    return {
        name: list(dict.fromkeys(cols))
        for name, cols in feature_sets.items()
    }

def get_all_candidate_features(df: pd.DataFrame) -> List[str]:
    feature_sets = build_feature_sets(df)
    all_features = []
    for cols in feature_sets.values():
        all_features.extend(cols)
    return list(dict.fromkeys(all_features))


def build_feature_matrix(df_1yr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    feature_sets = build_feature_sets(df_1yr)
    all_candidate_features = get_all_candidate_features(df_1yr)

    available = [c for c in all_candidate_features if c in df_1yr.columns]
    missing = [c for c in all_candidate_features if c not in df_1yr.columns]

    if not available:
        raise ValueError("None of the requested ML features are available.")

    X = df_1yr[available].copy()
    y = df_1yr["msk_1yr"].astype(int)

    if "Sex" in X.columns:
        X["Sex"] = (
            X["Sex"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"F": 0, "M": 1})
        )

    if "Overall_Mental_group" in X.columns:
        X["Overall_Mental_group"] = (
            X["Overall_Mental_group"]
            .fillna("none")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    for col in X.columns:
        if col != "Overall_Mental_group":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    zero_fill_cols = [
        c for c in X.columns
        if (
            c.endswith("_count")
            or c.endswith("_change")
            or c.startswith("Pre_")
            or c.startswith("Multi_")
            or c.startswith("Mental_Group_")
            or c.startswith("Any_")
            or c.startswith("medclass_")
            or c.startswith("n_")
            or c.endswith("_increase")
            or c.endswith("_decrease")
            or c.endswith("_switch")
            or c.endswith("_days")
            or c.endswith("_ratio")
            or c.startswith("active_")
        )
    ]

    for col in zero_fill_cols:
        X[col] = X[col].fillna(0)

    if "Age" in X.columns and "Sex" in X.columns:
        X["Age_Sex_interaction"] = X["Age"] * X["Sex"]

    if "Overall_Mental_group" in X.columns:
        X = pd.get_dummies(
            X,
            columns=["Overall_Mental_group"],
            drop_first=True,
            dtype=int,
        )

    print("\nAvailable features after encoding:")
    print(X.columns.tolist())

    print("\nMissing candidate features:")
    print(missing)

    return X, y, feature_sets


def expand_feature_columns(X: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    cols: List[str] = []

    for col in feature_cols:
        if col == "Overall_Mental_group":
            cols.extend([c for c in X.columns if c.startswith("Overall_Mental_group_")])
        elif col in X.columns:
            cols.append(col)

    return list(dict.fromkeys(cols))


def make_models(y_train: pd.Series) -> Dict[str, object]:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    models: Dict[str, object] = {
        "LogReg": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            base_score=0.5,
            random_state=SEED,
            n_jobs=-1,
        )

    return models


def make_pipeline_for_model(model_name: str, model: object) -> Pipeline:
    if model_name == "LogReg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", clone(model)),
        ])

    return Pipeline([
        ("model", clone(model)),
    ])


def evaluate_predictions(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = THRESHOLD,
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    return {
        "AUC": roc_auc_score(y_true_arr, y_prob_arr),
        "MAE": mean_absolute_error(y_true_arr, y_prob_arr),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

def bootstrap_auc_difference_pvalue(
    y_a,
    prob_a,
    y_b,
    prob_b,
    n_bootstrap=2000,
    random_state=SEED,
):
    rng = np.random.RandomState(random_state)

    y_a = np.asarray(y_a).astype(int)
    prob_a = np.asarray(prob_a, dtype=float)
    y_b = np.asarray(y_b).astype(int)
    prob_b = np.asarray(prob_b, dtype=float)

    if len(np.unique(y_a)) < 2 or len(np.unique(y_b)) < 2:
        return np.nan, np.nan, np.nan

    auc_a = roc_auc_score(y_a, prob_a)
    auc_b = roc_auc_score(y_b, prob_b)
    observed_diff = auc_a - auc_b

    boot_diffs = []

    for _ in range(n_bootstrap):
        idx_a = rng.randint(0, len(y_a), len(y_a))
        idx_b = rng.randint(0, len(y_b), len(y_b))

        if len(np.unique(y_a[idx_a])) < 2 or len(np.unique(y_b[idx_b])) < 2:
            continue

        boot_auc_a = roc_auc_score(y_a[idx_a], prob_a[idx_a])
        boot_auc_b = roc_auc_score(y_b[idx_b], prob_b[idx_b])

        boot_diffs.append(boot_auc_a - boot_auc_b)

    boot_diffs = np.asarray(boot_diffs)

    p_value = 2 * min(
        np.mean(boot_diffs <= 0),
        np.mean(boot_diffs >= 0),
    )

    p_value = min(p_value, 1.0)

    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)

    return observed_diff, ci_low, ci_high, p_value


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """
    Compute midranks for DeLong's test.
    """
    x = np.asarray(x)
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    n = len(x)
    midranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        # ranks are 1-based
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    output = np.empty(n, dtype=float)
    output[sorted_idx] = midranks
    return output


def _fast_delong(predictions_sorted: np.ndarray, n_positive: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong implementation for comparing correlated ROC AUCs.

    predictions_sorted:
        shape = (n_classifiers, n_examples)
        examples sorted with positive cases first.
    n_positive:
        number of positive cases.
    """
    m = n_positive
    n = predictions_sorted.shape[1] - m

    if m <= 0 or n <= 0:
        raise ValueError("DeLong test requires at least one positive and one negative case.")

    positive_examples = predictions_sorted[:, :m]
    negative_examples = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    covariance = np.cov(v01) / m + np.cov(v10) / n

    # np.cov can return scalar when k=1; force 2D for safety
    covariance = np.atleast_2d(covariance)

    return aucs, covariance


def delong_pvalue(
    y_true: pd.Series,
    y_score_ref: np.ndarray,
    y_score_other: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compare two ROC AUCs on the same test set using DeLong's test.

    Returns:
        auc_ref, auc_other, p_value
    """
    y_true_arr = np.asarray(y_true).astype(int)

    if len(np.unique(y_true_arr)) != 2:
        raise ValueError("DeLong test requires both positive and negative cases.")

    y_score_ref = np.asarray(y_score_ref, dtype=float)
    y_score_other = np.asarray(y_score_other, dtype=float)

    if len(y_true_arr) != len(y_score_ref) or len(y_true_arr) != len(y_score_other):
        raise ValueError("y_true and prediction arrays must have the same length.")

    # Sort so positives are first
    order = (-y_true_arr).argsort()
    n_positive = int(y_true_arr.sum())

    predictions_sorted = np.array([
        y_score_ref[order],
        y_score_other[order],
    ])

    aucs, covariance = _fast_delong(predictions_sorted, n_positive)

    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T).item())

    if variance <= 0 or np.isnan(variance):
        return float(aucs[0]), float(aucs[1]), 1.0

    z_score = (aucs[0] - aucs[1]) / np.sqrt(variance)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return float(aucs[0]), float(aucs[1]), float(p_value)


def add_pvalues_to_results(
    results_df: pd.DataFrame,
    prediction_dict: Dict[str, np.ndarray],
    y_true: pd.Series,
    reference_key: str,
    key_column: str,
) -> pd.DataFrame:

    results_df = results_df.copy()
    ref_pred = prediction_dict[reference_key]

    pvals = []

    for _, row in results_df.iterrows():
        current_key = row[key_column]

        if current_key == reference_key:
            pvals.append("ref")
            continue

        current_pred = prediction_dict[current_key]

        try:
            _, _, p = delong_pvalue(
                y_true=y_true,
                y_score_ref=ref_pred,
                y_score_other=current_pred,
            )

            p = float(p)

            if p < 0.001:
                pvals.append("<0.001")
            else:
                pvals.append(f"{p:.3f}")

        except Exception as e:
            print(f"DeLong test failed for {current_key} vs {reference_key}: {e}")
            pvals.append(np.nan)

    results_df["p-value"] = pvals

    return results_df

def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    n_splits: int = N_SPLITS,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    records = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_tr, y_tr)
        y_prob = fold_pipeline.predict_proba(X_val)[:, 1]

        row = evaluate_predictions(y_val, y_prob)
        row["fold"] = fold_idx
        records.append(row)

    return pd.DataFrame(records)


def summarize_cv(cv_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "CV_AUC_mean": cv_df["AUC"].mean(),
        "CV_AUC_sd": cv_df["AUC"].std(ddof=1),
        "CV_MAE_mean": cv_df["MAE"].mean(),
        "CV_MAE_sd": cv_df["MAE"].std(ddof=1),
        "CV_Sensitivity_mean": cv_df["Sensitivity"].mean(),
        "CV_Sensitivity_sd": cv_df["Sensitivity"].std(ddof=1),
        "CV_Specificity_mean": cv_df["Specificity"].mean(),
        "CV_Specificity_sd": cv_df["Specificity"].std(ddof=1),
    }


def plot_roc_comparison(
    predictions: Dict[str, np.ndarray],
    y_test: pd.Series,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    for label, y_prob in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
def final_feature_correlation_screen(
    X: pd.DataFrame,
    feature_cols: List[str],
    table_dir: Path,
    figure_dir: Path,
    threshold: float = CORR_THRESHOLD,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in feature_cols if c in X.columns and c != "Age_Sex_interaction"]

    if len(cols) < 2:
        print("\nNot enough final features for correlation screen.")
        return

    corr_df = X[cols].copy()

    for col in corr_df.columns:
        corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce").fillna(0)

    non_constant_cols = [
        c for c in cols
        if corr_df[c].nunique(dropna=False) > 1
    ]

    corr = corr_df[non_constant_cols].corr(method="pearson")
    corr.to_csv(table_dir / "00_final_feature_correlation_matrix.csv")

    short_labels = [
        c.replace("Overall_Mental_group_", "OMG_")
         .replace("Mental_Diagnosis_count", "MentalDx_count")
         .replace("medclass_", "")
         .replace("_count_change", "_chg")
         .replace("n_unique_", "n_")
        for c in corr.columns
    ]

    annot_values = (corr * 100).round(0).astype(int)

    fig_size = max(12, 0.45 * len(short_labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="equal"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation coefficient (r)", fontsize=10)

    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)

    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            ax.text(
                j,
                i,
                f"{annot_values.iloc[i, j]}",
                ha="center",
                va="center",
                fontsize=5,
                color="black"
            )

    ax.set_title(
        "Correlation Matrix of Final Model Features (%)",
        fontsize=13,
        pad=12
    )

    plt.tight_layout()
    plt.savefig(
        figure_dir / "00_final_feature_correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    pairs = []
    cols = corr.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "pearson_r": r,
                    "abs_pearson_r": abs(r),
                })

    high_corr = (
        pd.DataFrame(pairs).sort_values("abs_pearson_r", ascending=False)
        if pairs
        else pd.DataFrame(columns=["feature_1", "feature_2", "pearson_r", "abs_pearson_r"])
    )

    high_corr.to_csv(
        table_dir / "00_final_feature_high_correlation_pairs.csv",
        index=False
    )

    print("\nFinal feature correlation screen saved:")
    print(table_dir / "00_final_feature_correlation_matrix.csv")
    print(figure_dir / "00_final_feature_correlation_matrix.png")
    print(table_dir / "00_final_feature_high_correlation_pairs.csv")

def medication_correlation_screen(
    X: pd.DataFrame,
    table_dir: Path,
    figure_dir: Path,
    threshold: float = CORR_THRESHOLD,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    med_cols = [
        c for c in X.columns
        if (
            c in MEDICATION_COMPLEXITY_FEATURES
            or c in KEY_MEDCLASS_CHANGE_FEATURES
            or c in TRAJECTORY_FEATURES
        )
    ]

    med_cols = [c for c in med_cols if c in X.columns]

    if not med_cols:
        print("\nNo medication columns found for correlation screen.")
        return

    med_df = X[med_cols].copy()

    for col in med_df.columns:
        med_df[col] = pd.to_numeric(med_df[col], errors="coerce").fillna(0)

    prevalence = pd.DataFrame({
        "Feature": med_cols,
        "Nonzero_N": [(med_df[c] != 0).sum() for c in med_cols],
        "Nonzero_pct": [(med_df[c] != 0).mean() for c in med_cols],
        "Mean": [med_df[c].mean() for c in med_cols],
        "SD": [med_df[c].std(ddof=1) for c in med_cols],
    }).sort_values(["Nonzero_N", "Mean"], ascending=False)

    prevalence.to_csv(
        table_dir / "00_medication_feature_prevalence.csv",
        index=False
    )

    non_constant_cols = [
        c for c in med_cols
        if med_df[c].nunique(dropna=False) > 1
    ]

    if len(non_constant_cols) < 2:
        print("\nNot enough non-constant medication columns for correlation.")
        return

    corr = med_df[non_constant_cols].corr(method="pearson")
    corr.to_csv(table_dir / "00_medication_correlation_matrix.csv")

    short_labels = [
        c.replace("medclass_", "")
         .replace("_count_change", "_chg")
         .replace("n_unique_", "n_")
        for c in corr.columns
    ]

    annot_values = (corr * 100).round(0).astype(int)

    fig_size = max(10, 0.6 * len(short_labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="equal"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation coefficient (r)", fontsize=10)

    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            value = annot_values.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value}",
                ha="center",
                va="center",
                fontsize=7,
                color="black"
            )

    ax.set_title(
        "Correlation Matrix of Medication Features (%)",
        fontsize=13,
        pad=12
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir / "00_medication_correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    pairs = []
    cols = corr.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "pearson_r": r,
                    "abs_pearson_r": abs(r),
                })

    high_corr = (
        pd.DataFrame(pairs).sort_values("abs_pearson_r", ascending=False)
        if pairs
        else pd.DataFrame(columns=["feature_1", "feature_2", "pearson_r", "abs_pearson_r"])
    )

    high_corr.to_csv(
        table_dir / "00_medication_high_correlation_pairs.csv",
        index=False
    )

    print("\nMedication correlation screen saved:")
    print(table_dir / "00_medication_feature_prevalence.csv")
    print(table_dir / "00_medication_correlation_matrix.csv")
    print(figure_dir / "00_medication_correlation_matrix.png")
    print(table_dir / "00_medication_high_correlation_pairs.csv")

def remove_highly_correlated_features(
    X: pd.DataFrame,
    candidate_cols: List[str],
    threshold: float = CORR_THRESHOLD,
) -> List[str]:
    cols = [c for c in candidate_cols if c in X.columns]

    if len(cols) < 2:
        return cols

    X_num = X[cols].copy()
    for col in X_num.columns:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce").fillna(0)

    non_constant_cols = [
        c for c in cols
        if X_num[c].nunique(dropna=False) > 1
    ]

    if len(non_constant_cols) < 2:
        return non_constant_cols

    corr = X_num[non_constant_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]

    kept = [c for c in cols if c not in to_drop]

    return kept


def run_shap_for_final_model(
    pipeline: Pipeline,
    X_test_final: pd.DataFrame,
    table_dir: Path,
    figure_dir: Path,
) -> None:
    if not HAS_SHAP:
        print("\nSHAP is not installed. Skipping SHAP analysis.")
        return

    X_test_final = X_test_final.copy()

    for col in X_test_final.columns:
        X_test_final[col] = pd.to_numeric(X_test_final[col], errors="coerce")

    X_test_final = X_test_final.fillna(0).astype(float)

    model = pipeline.named_steps["model"]

    try:
        if model.__class__.__name__ == "XGBClassifier":
            import xgboost as xgb

            booster = model.get_booster()
            dtest = xgb.DMatrix(
                X_test_final,
                feature_names=list(X_test_final.columns)
            )
            contribs = booster.predict(dtest, pred_contribs=True)
            shap_values_pos = contribs[:, :-1]

        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_final)

            if isinstance(shap_values, list):
                shap_values_pos = shap_values[1]
            else:
                shap_values_pos = shap_values

        plt.figure()
        shap.summary_plot(
            shap_values_pos,
            X_test_final,
            max_display=len(X_test_final.columns),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(figure_dir / "05_shap_summary_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        shap_importance = (
            pd.DataFrame({
                "Feature": X_test_final.columns,
                "MeanAbsSHAP": np.abs(shap_values_pos).mean(axis=0),
            })
            .sort_values("MeanAbsSHAP", ascending=False)
            .reset_index(drop=True)
        )

        shap_importance.insert(0, "Rank", np.arange(1, len(shap_importance) + 1))

        shap_importance.to_csv(table_dir / "05_shap_feature_importance.csv", index=False)

        top = shap_importance.head(20).iloc[::-1]

        plt.figure(figsize=(8, 6))
        plt.barh(top["Feature"], top["MeanAbsSHAP"])
        plt.xlabel("Mean absolute SHAP value")
        plt.title("Top 20 SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(figure_dir / "05_shap_bar_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"\nSHAP failed: {e}")


def assign_age_group(age_series: pd.Series) -> pd.Series:
    age_numeric = pd.to_numeric(age_series, errors="coerce")

    return pd.cut(
        age_numeric,
        bins=AGE_GROUP_BINS,
        labels=AGE_GROUP_LABELS,
        right=False,
        include_lowest=True,
    )


def run_age_group_generalization_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    final_prob: np.ndarray,
    table_dir: Path,
    figure_dir: Path,
) -> None:
    age_group = assign_age_group(X_test["Age"])

    rows = []

    for group in AGE_GROUP_LABELS:
        mask = age_group == group

        y_group = y_test.loc[mask]
        prob_group = final_prob[mask.to_numpy()]

        row = {
            "Age_Group": group,
            "Test_N": len(y_group),
            "MSK_1yr_N": int(y_group.sum()) if len(y_group) > 0 else 0,
            "MSK_1yr_rate": float(y_group.mean()) if len(y_group) > 0 else np.nan,
        }

        if len(y_group) == 0 or y_group.nunique() < 2:
            row.update({
                "AUC": np.nan,
                "MAE": np.nan,
                "Sensitivity": np.nan,
                "Specificity": np.nan,
                "TP": np.nan,
                "TN": np.nan,
                "FP": np.nan,
                "FN": np.nan,
            })
        else:
            row.update(evaluate_predictions(y_group, prob_group))

        rows.append(row)

    pairwise_rows = []

    for i in range(len(AGE_GROUP_LABELS)):
        for j in range(i + 1, len(AGE_GROUP_LABELS)):
            group_a = AGE_GROUP_LABELS[i]
            group_b = AGE_GROUP_LABELS[j]

            mask_a = age_group == group_a
            mask_b = age_group == group_b

            y_a = y_test.loc[mask_a]
            y_b = y_test.loc[mask_b]

            prob_a = final_prob[mask_a.to_numpy()]
            prob_b = final_prob[mask_b.to_numpy()]

            diff, ci_low, ci_high, p = bootstrap_auc_difference_pvalue(
                y_a,
                prob_a,
                y_b,
                prob_b,
            )

            pairwise_rows.append({
                "Comparison": f"{group_a} vs {group_b}",
                "AUC_difference": diff,
                "AUC_difference_95CI_Lower": ci_low,
                "AUC_difference_95CI_Upper": ci_high,
                "p_value": p,
            })

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(
        table_dir / "08_age_group_auc_pairwise_bootstrap_pvalues.csv",
        index=False,
    )

    age_df = pd.DataFrame(rows)
    age_df.to_csv(table_dir / "08_age_group_generalization_auc.csv", index=False)

    completed = age_df.dropna(subset=["AUC"])

    if not completed.empty:
        plt.figure(figsize=(7, 5))
        plt.bar(completed["Age_Group"], completed["AUC"])
        plt.ylim(0.5, 0.8)
        plt.xlabel("Age Group")
        plt.ylabel("AUC")
        plt.title("Final Model Performance by Age Group")
        plt.tight_layout()
        plt.savefig(figure_dir / "08_age_group_generalization_auc.png", dpi=300, bbox_inches="tight")
        plt.close()


def run_pearson_analysis(
    X_test_final: pd.DataFrame,
    y_test: pd.Series,
    table_dir: Path,
    figure_dir: Path,
) -> None:
    df = X_test_final.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)
    df["msk_1yr"] = y_test.values

    corr = df.corr(method="pearson")
    corr.to_csv(table_dir / "06_pearson_correlation_matrix.csv")

    outcome_corr = (
        corr["msk_1yr"]
        .drop("msk_1yr")
        .reset_index()
        .rename(columns={
            "index": "Feature",
            "msk_1yr": "Pearson_r_with_msk_1yr",
        })
    )

    outcome_corr["Abs_Pearson_r"] = outcome_corr["Pearson_r_with_msk_1yr"].abs()
    outcome_corr = outcome_corr.sort_values("Abs_Pearson_r", ascending=False).reset_index(drop=True)

    outcome_corr.to_csv(table_dir / "06_pearson_with_outcome.csv", index=False)

    plot_corr = outcome_corr.sort_values("Pearson_r_with_msk_1yr")

    plt.figure(figsize=(10, max(6, 0.35 * len(plot_corr))))
    plt.barh(plot_corr["Feature"], plot_corr["Pearson_r_with_msk_1yr"])
    plt.axvline(x=0, linestyle="--", linewidth=1)
    plt.xlabel("Pearson Correlation Coefficient (r)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(figure_dir / "06_pearson_outcome_barplot_all_features.png", dpi=300, bbox_inches="tight")
    plt.close()




def _format_n_pct(n: int, denom: int) -> str:
    """Format categorical values as n (%)."""
    if denom <= 0:
        return "0 (0.0%)"
    return f"{int(n)} ({100 * n / denom:.1f}%)"


def _format_p_value(p_value: float | None) -> str:
    """Format p-values for publication-style tables."""
    if p_value is None or pd.isna(p_value):
        return ""
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def _format_mean_sd_range(series: pd.Series) -> str:
    """
    Format continuous variables similar to a descriptive summary table:
        mean ± SD; range: min–max
    """
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return "NA"

    mean = x.mean()
    sd = x.std(ddof=1)

    if pd.isna(sd):
        sd = 0.0

    return f"{mean:.2f} ± {sd:.2f}"


def _continuous_p_value(msk_values: pd.Series, non_msk_values: pd.Series) -> str:
    """
    Compare MSK vs Non-MSK for continuous variables.

    Mann–Whitney U test is used because most medication and diagnosis count
    variables are skewed and zero-inflated.
    """
    x1 = pd.to_numeric(msk_values, errors="coerce").dropna()
    x0 = pd.to_numeric(non_msk_values, errors="coerce").dropna()

    if len(x1) < 2 or len(x0) < 2:
        return ""

    try:
        p_value = stats.mannwhitneyu(x1, x0, alternative="two-sided").pvalue
        return _format_p_value(p_value)
    except Exception:
        return ""


def _categorical_p_value(df: pd.DataFrame, col: str, group_col: str = "MSK_Group") -> str:
    """
    Compare MSK vs Non-MSK for categorical variables.

    Chi-square is used by default. Fisher exact is used automatically for 2x2
    tables with small expected cell counts.
    """
    try:
        tab = pd.crosstab(df[col], df[group_col])
        if tab.shape[0] < 2 or tab.shape[1] != 2:
            return ""

        chi2, p_value, dof, expected = stats.chi2_contingency(tab)

        if tab.shape == (2, 2) and (expected < 5).any():
            _, p_value = stats.fisher_exact(tab)

        return _format_p_value(p_value)
    except Exception:
        return ""


def create_descriptive_table(
    df_1yr: pd.DataFrame,
    train_index=None,
    test_index=None,
    table_dir: Path | None = None,
) -> None:
    """
    Create a publication-style Table 1 for the 1-year MSK prediction cohort.

    Output:
      00_descriptive_table_1yr.csv

    Design:
      - Uses the exact patient-level rows used for modelling.
      - Compares MSK vs Non-MSK groups, not train vs test.
      - Continuous variables are shown as: mean ± SD; range: min–max.
      - Categorical variables are shown as: n (%).
      - P values compare MSK vs Non-MSK.
    """
    if table_dir is None:
        raise ValueError("table_dir must be provided")

    table_dir.mkdir(parents=True, exist_ok=True)

    df = df_1yr.copy()
    if "msk_1yr" not in df.columns:
        raise ValueError("df_1yr must contain msk_1yr for MSK vs Non-MSK descriptive table")

    df["msk_1yr"] = pd.to_numeric(df["msk_1yr"], errors="coerce").fillna(0).astype(int)
    df["MSK_Group"] = np.where(df["msk_1yr"] == 1, "MSK", "Non-MSK")

    n_all = len(df)
    n_msk = int((df["MSK_Group"] == "MSK").sum())
    n_non_msk = int((df["MSK_Group"] == "Non-MSK").sum())

    rows = []

    def add_section(section_name: str) -> None:
        rows.append({
            "Section": section_name,
            "Characteristic": "",
            "Level_or_summary": "",
            f"Overall (N={n_all})": "",
            f"MSK (N={n_msk})": "",
            f"Non-MSK (N={n_non_msk})": "",
            "P_value": "",
            "Test": "",
        })

    def add_continuous(section: str, col: str, label: str) -> None:
        if col not in df.columns:
            return

        all_values = df[col]
        msk_values = df.loc[df["MSK_Group"] == "MSK", col]
        non_msk_values = df.loc[df["MSK_Group"] == "Non-MSK", col]

        rows.append({
            "Section": section,
            "Characteristic": label,
            "Level_or_summary": "Mean ± SD",
            f"Overall (N={n_all})": _format_mean_sd_range(all_values),
            f"MSK (N={n_msk})": _format_mean_sd_range(msk_values),
            f"Non-MSK (N={n_non_msk})": _format_mean_sd_range(non_msk_values),
            "P_value": _continuous_p_value(msk_values, non_msk_values),
            "Test": "Mann–Whitney U",
        })

    def add_categorical(section: str, col: str, label: str, level_order: list[str] | None = None) -> None:
        if col not in df.columns:
            return

        temp = df[[col, "MSK_Group"]].copy()
        temp[col] = temp[col].fillna("Missing").astype(str).str.strip()

        if level_order is None:
            levels = sorted(temp[col].dropna().unique().tolist())
        else:
            existing = temp[col].dropna().unique().tolist()
            levels = [x for x in level_order if x in existing]
            levels += [x for x in sorted(existing) if x not in set(levels)]

        p_text = _categorical_p_value(temp, col)

        for i, level in enumerate(levels):
            all_n = int((temp[col] == level).sum())
            msk_n = int(((temp[col] == level) & (temp["MSK_Group"] == "MSK")).sum())
            non_msk_n = int(((temp[col] == level) & (temp["MSK_Group"] == "Non-MSK")).sum())

            rows.append({
                "Section": section if i == 0 else "",
                "Characteristic": label if i == 0 else "",
                "Level_or_summary": level,
                f"Overall (N={n_all})": _format_n_pct(all_n, n_all),
                f"MSK (N={n_msk})": _format_n_pct(msk_n, n_msk),
                f"Non-MSK (N={n_non_msk})": _format_n_pct(non_msk_n, n_non_msk),
                "P_value": p_text if i == 0 else "",
                "Test": "Chi-square/Fisher exact" if i == 0 else "",
            })

    # ------------------------------------------------------------------
    # Table content: compact and aligned with model feature-set structure.
    # ------------------------------------------------------------------
    add_section("Demographics")
    add_continuous("Demographics", "Age", "Age, years")
    add_categorical("Demographics", "Sex", "Sex", level_order=["F", "M", "Female", "Male", "0", "1"])

    add_section("Mental health")
    add_continuous("Mental health", "Mental_Diagnosis_count", "Mental diagnosis count")
    add_continuous("Mental health", "Mental_visit_count", "Mental visit count")
    add_continuous("Mental health", "Mental_visit_recency_days", "Mental visit recency, days")
    add_categorical("Mental health", "Recent_30d_mental_visit", "Recent mental visit within 30 days", level_order=["0", "1"])
    add_categorical(
        "Mental health",
        "Overall_Mental_group",
        "Overall mental health group",
        level_order=["none", "acute", "chronic", "both", "symptom"],
    )

    add_section("Disease burden")
    add_continuous("Disease burden", "Pre_existing_condition_count", "Pre-existing condition count")
    add_continuous("Disease burden", "Multimorbidity_count", "New multimorbidity system count")
    add_continuous("Disease burden", "Comorbidity_visit_count", "New comorbidity visit count")

    add_section("Medication burden")
    add_continuous("Medication burden", "n_mental_drugs", "Number of active mental-related drugs")
    add_continuous("Medication burden", "n_pain_drugs", "Number of active pain-related drugs")

    add_section("Medication class dynamics")
    class_names_for_table = {
        "antidepressant": "Antidepressant",
        "anti_anxiety": "Anti-anxiety",
        "antipsychotic": "Antipsychotic",
        "hypnotic": "Hypnotic",
        "mood_stabilizer": "Mood stabilizer",
        "nsaid": "NSAID",
        "opioid": "Opioid",
        "acetaminophen": "Acetaminophen",
        "muscle_relaxant": "Muscle relaxant",
        "neuropathic_pain": "Neuropathic pain medication",
    }
    for cls, label in class_names_for_table.items():
        add_continuous(
            "Medication class dynamics",
            f"medclass_{cls}_count_change",
            f"{label} change count",
        )

    add_section("Medication trajectory")
    trajectory_labels = {
    "med_addition_count": "Medication addition count",
    "med_discontinuation_count": "Medication discontinuation count",
    "cross_category_change_count": "Cross-category medication change count",

    "mental_medication_days": "Mental medication days",
    "pain_medication_days": "Pain medication days",
    "active_medication_days": "Active medication days",

    "mental_pain_overlap_days": "Mental-pain overlap days",
    "mental_pain_overlap_ratio": "Mental-pain medication overlap ratio",
    }
    for col, label in trajectory_labels.items():
        add_continuous("Medication trajectory", col, label)

    desc = pd.DataFrame(rows)
    desc.to_csv(table_dir / "00_descriptive_table_1yr.csv", index=False)

    # A second version without the visual section separator rows is easier for statistics reuse.
    desc_clean = desc[desc["Characteristic"].astype(str).str.len() > 0].copy()
    desc_clean.to_csv(table_dir / "00_descriptive_table_1yr_clean.csv", index=False)

    print("\nDescriptive Table 1 saved:")
    print(table_dir / "00_descriptive_table_1yr.csv")
    print(table_dir / "00_descriptive_table_1yr_clean.csv")

def run_medication_change_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    final_prob: np.ndarray,
    table_dir: Path,
    figure_dir: Path,
) -> None:
    """
    Summarize medication class change features.
    """
    rows = []

    for col in KEY_MEDCLASS_CHANGE_FEATURES:
        if col not in X_test.columns:
            continue

        values = pd.to_numeric(X_test[col], errors="coerce").fillna(0)
        nonzero_mask = values > 0
        zero_mask = values == 0

        row = {
            "Feature": col,
            "N_total": int(values.shape[0]),
            "Mean": float(values.mean()),
            "Median": float(values.median()),
            "Nonzero_N": int(nonzero_mask.sum()),
            "Nonzero_pct": float(nonzero_mask.mean()),
            "MSK_rate_nonzero": float(y_test.loc[nonzero_mask].mean()) if nonzero_mask.sum() > 0 else np.nan,
            "MSK_rate_zero": float(y_test.loc[zero_mask].mean()) if zero_mask.sum() > 0 else np.nan,
            "PredRisk_mean_nonzero": float(np.mean(final_prob[nonzero_mask.to_numpy()])) if nonzero_mask.sum() > 0 else np.nan,
            "PredRisk_mean_zero": float(np.mean(final_prob[zero_mask.to_numpy()])) if zero_mask.sum() > 0 else np.nan,
        }

        if values.nunique(dropna=False) > 1:
            row["Pearson_r_with_msk"] = float(np.corrcoef(values, y_test.astype(float))[0, 1])
            row["Pearson_r_with_predrisk"] = float(np.corrcoef(values, final_prob)[0, 1])
        else:
            row["Pearson_r_with_msk"] = np.nan
            row["Pearson_r_with_predrisk"] = np.nan

        row["MSK_rate_nonzero_minus_zero"] = (
            row["MSK_rate_nonzero"] - row["MSK_rate_zero"]
            if pd.notna(row["MSK_rate_nonzero"]) and pd.notna(row["MSK_rate_zero"])
            else np.nan
        )
        row["PredRisk_nonzero_minus_zero"] = (
            row["PredRisk_mean_nonzero"] - row["PredRisk_mean_zero"]
            if pd.notna(row["PredRisk_mean_nonzero"]) and pd.notna(row["PredRisk_mean_zero"])
            else np.nan
        )

        rows.append(row)

    dynamics_df = pd.DataFrame(rows)

    if dynamics_df.empty:
        print("\nNo medication dynamic features found for direction analysis.")
        return

    dynamics_df = dynamics_df.sort_values(
        "PredRisk_nonzero_minus_zero",
        ascending=False,
        na_position="last",
    )

    dynamics_df.to_csv(table_dir / "07_medication_dynamic_feature_summary.csv", index=False)

    plot_df = dynamics_df.dropna(subset=["PredRisk_nonzero_minus_zero"]).copy()
    if not plot_df.empty:
        plot_df["ShortFeature"] = (
            plot_df["Feature"]
            .str.replace("medclass_", "", regex=False)
            .str.replace("_count_", "_", regex=False)
        )
        plot_df = plot_df.sort_values("PredRisk_nonzero_minus_zero")

        plt.figure(figsize=(8, max(5, 0.35 * len(plot_df))))
        plt.barh(plot_df["ShortFeature"], plot_df["PredRisk_nonzero_minus_zero"])
        plt.axvline(x=0, linestyle="--", linewidth=1)
        plt.xlabel("Mean predicted risk: nonzero - zero")
        plt.ylabel("Medication class dynamic feature")
        plt.title("Medication Class Dynamics and Predicted MSK Risk")
        plt.tight_layout()
        plt.savefig(figure_dir / "07_medication_dynamic_feature_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nMedication dynamic feature analysis saved:")
    print(table_dir / "07_medication_dynamic_feature_summary.csv")
    print(figure_dir / "07_medication_dynamic_feature_plot.png")

def run_experiment(
    df: pd.DataFrame,
    table_dir: Path,
    figure_dir: Path,
) -> None:
    np.random.seed(SEED)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df_1yr = build_1yr_patient_dataset(df)
    X, y, feature_sets = build_feature_matrix(df_1yr)

    medication_correlation_screen(
        X,
        table_dir=table_dir,
        figure_dir=figure_dir,
        threshold=CORR_THRESHOLD,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    cohort_summary = pd.DataFrame([
        {
            "Dataset": "All",
            "N": len(y),
            "N_MSK_1yr": int(y.sum()),
            "MSK_1yr_rate": float(y.mean()),
        },
        {
            "Dataset": "Train",
            "N": len(y_train),
            "N_MSK_1yr": int(y_train.sum()),
            "MSK_1yr_rate": float(y_train.mean()),
        },
        {
            "Dataset": "Test",
            "N": len(y_test),
            "N_MSK_1yr": int(y_test.sum()),
            "MSK_1yr_rate": float(y_test.mean()),
        },
    ])
    cohort_summary.to_csv(table_dir / "00_cohort_split_summary_1yr.csv", index=False)

    create_descriptive_table(
        df_1yr=df_1yr,
        train_index=X_train.index,
        test_index=X_test.index,
        table_dir=table_dir,
    )

    models = make_models(y_train)

    # Step 1: algorithm selection must use the most complete all-variable feature set.
    full_feature_set_name = "06_Final_WithInteraction"
    full_feature_cols = feature_sets[full_feature_set_name]
    full_cols = expand_feature_columns(X_train, full_feature_cols)

    if not full_cols:
        raise ValueError(f"No columns available for {full_feature_set_name}")

    # Keep all variables in the full model. Correlation screen is only for QC/reporting.
    final_feature_correlation_screen(
        X_train,
        full_cols,
        table_dir=table_dir,
        figure_dir=figure_dir,
        threshold=CORR_THRESHOLD,
    )

    pd.DataFrame({"Feature": full_cols}).to_csv(
        table_dir / "01_step1_full_model_features_all_variables.csv",
        index=False,
    )

    print("\nSTEP 1: Model comparison using the most complete full model")
    print("Full model feature set:", full_feature_set_name)
    print("Full model columns:", full_cols)

    model_rows = []
    model_predictions: Dict[str, np.ndarray] = {}
    cv_records = []

    for model_name, model in models.items():
        pipeline = make_pipeline_for_model(model_name, model)

        cv_df = run_cross_validation(X_train[full_cols], y_train, pipeline)
        cv_df["Stage"] = "Step1_ModelComparison_FullAllVariables"
        cv_df["FeatureSet"] = full_feature_set_name
        cv_df["Model"] = model_name
        cv_records.append(cv_df)

        cv_summary = summarize_cv(cv_df)

        pipeline.fit(X_train[full_cols], y_train)
        y_prob = pipeline.predict_proba(X_test[full_cols])[:, 1]

        model_predictions[f"{full_feature_set_name}__{model_name}"] = y_prob
        test_metrics = evaluate_predictions(y_test, y_prob)

        model_rows.append({
            "FeatureSet": full_feature_set_name,
            "Model": model_name,
            "n_features": len(full_cols),
            **test_metrics,
            **cv_summary,
        })

    model_comparison_df = pd.DataFrame(model_rows).sort_values(["AUC", "Sensitivity"], ascending=False)
    best_model_name = model_comparison_df.iloc[0]["Model"]

    step1_prediction_dict = {
        row["Model"]:
        model_predictions[f"{full_feature_set_name}__{row['Model']}"]
        for _, row in model_comparison_df.iterrows()
    }

    model_comparison_df = add_pvalues_to_results(
        results_df=model_comparison_df,
        prediction_dict=step1_prediction_dict,
        y_true=y_test,
        reference_key=best_model_name,
        key_column="Model",
    )
    model_comparison_df.to_csv(table_dir / "01_step1_full_model_algorithm_comparison.csv", index=False)
    best_base_model = models[best_model_name]

    plot_roc_comparison(
        {
            row["Model"]: model_predictions[f"{full_feature_set_name}__{row['Model']}"]
            for _, row in model_comparison_df.iterrows()
        },
        y_test,
        figure_dir / "01_step1_full_model_roc_comparison.png",
        title="Step 1: Full All-Variable Model",
    )

    print("\nSelected best algorithm from Step 1:", best_model_name)

    # Step 2: feature-set comparison using only the best algorithm selected in Step 1.
    print("\nSTEP 2: Feature set comparison using selected algorithm only")

    feature_rows = []
    feature_predictions: Dict[str, np.ndarray] = {}
    feature_column_records = []

    for feature_set_name, feature_cols in feature_sets.items():
        cols = expand_feature_columns(X_train, feature_cols)

        if not cols:
            print(f"Skipping {feature_set_name}: no available columns.")
            continue

        # Keep all variables in each candidate set for transparent comparison.
        print("\nFeatureSet:", feature_set_name)
        print("Columns:", cols)

        for c in cols:
            feature_column_records.append({
                "FeatureSet": feature_set_name,
                "Feature": c,
            })

        pipeline = make_pipeline_for_model(best_model_name, best_base_model)

        cv_df = run_cross_validation(X_train[cols], y_train, pipeline)
        cv_df["Stage"] = "Step2_FeatureSetComparison"
        cv_df["FeatureSet"] = feature_set_name
        cv_df["Model"] = best_model_name
        cv_records.append(cv_df)

        cv_summary = summarize_cv(cv_df)

        pipeline.fit(X_train[cols], y_train)
        y_prob = pipeline.predict_proba(X_test[cols])[:, 1]

        feature_predictions[feature_set_name] = y_prob
        test_metrics = evaluate_predictions(y_test, y_prob)

        feature_rows.append({
            "FeatureSet": feature_set_name,
            "Model": best_model_name,
            "n_features": len(cols),
            **test_metrics,
            **cv_summary,
        })

    feature_comparison_df = pd.DataFrame(feature_rows).sort_values(["AUC", "Sensitivity"], ascending=False)
    best_feature_set = feature_comparison_df.iloc[0]["FeatureSet"]

    feature_comparison_df = add_pvalues_to_results(
        results_df=feature_comparison_df,
        prediction_dict=feature_predictions,
        y_true=y_test,
        reference_key=best_feature_set,
        key_column="FeatureSet",
    )
    feature_comparison_df.to_csv(table_dir / "02_step2_feature_set_comparison.csv", index=False)
    
    pd.DataFrame(feature_column_records).to_csv(
        table_dir / "02_step2_feature_set_columns.csv",
        index=False,
    )

    plot_roc_comparison(
        feature_predictions,
        y_test,
        figure_dir / "02_step2_feature_set_roc_comparison.png",
        title=f"Step 2: Feature Set Comparison ({best_model_name})",
    )

    # Step 3: final model = best feature set from Step 2 + best algorithm from Step 1
    
    # Final model is fixed to the compact, clinically preferred medication trajectory set.
    # 07_Check_CategoryComorbidity remains a sensitivity analysis only and is not used as final.
    preferred_final_feature_set = "05_Add_MedTrajectory"
    final_feature_set = (
        preferred_final_feature_set
        if preferred_final_feature_set in feature_sets
        else best_feature_set
    )
    final_cols = expand_feature_columns(
        X_train,
        feature_sets[final_feature_set]
    )

    pd.DataFrame({"Feature": final_cols}).to_csv(
        table_dir / "03_final_model_features.csv",
        index=False,
    )

    # Overwrite the 00_final_feature_correlation_* outputs so they reflect
    # the actual final compact model, not the Step 1 all-variable model.
    final_feature_correlation_screen(
        X_train,
        final_cols,
        table_dir=table_dir,
        figure_dir=figure_dir,
        threshold=CORR_THRESHOLD,
    )

    final_pipeline = make_pipeline_for_model(best_model_name, best_base_model)
    final_pipeline.fit(X_train[final_cols], y_train)
    final_prob = final_pipeline.predict_proba(X_test[final_cols])[:, 1]

    final_metrics = evaluate_predictions(y_test, final_prob)

    final_df = pd.DataFrame([{
        "Final_FeatureSet": final_feature_set,
        "Final_Model": best_model_name,
        "n_features": len(final_cols),
        **final_metrics,
    }])
    final_df.to_csv(table_dir / "03_final_model_metrics_1yr.csv", index=False)

    plot_roc_comparison(
        {f"{final_feature_set}__{best_model_name}": final_prob},
        y_test,
        figure_dir / "03_final_model_roc_curve_1yr.png",
        title="Final Full Model ROC Curve",
    )

    if best_model_name in ["RandomForest", "XGBoost"]:
        run_shap_for_final_model(
            final_pipeline,
            X_test[final_cols],
            table_dir=table_dir,
            figure_dir=figure_dir,
        )

    run_pearson_analysis(
        X_test[final_cols],
        y_test,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )

    run_medication_change_analysis(
        X_test=X_test,
        y_test=y_test,
        final_prob=final_prob,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )

    run_age_group_generalization_analysis(
        X_test=X_test,
        y_test=y_test,
        final_prob=final_prob,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )

    cv_all = pd.concat(cv_records, ignore_index=True)
    cv_all.to_csv(table_dir / "04_all_cv_per_fold_metrics_1yr.csv", index=False)

    with pd.option_context("display.float_format", "{:0.4f}".format):
        print("\nSTEP 1 TABLE: Full model algorithm comparison")
        print(model_comparison_df.to_string(index=False))

        print("\nSTEP 2 TABLE: Feature set comparison")
        print(feature_comparison_df.to_string(index=False))

        print("\nSTEP 3 FINAL FULL MODEL")
        print(final_df.to_string(index=False))

    print("\nSaved table outputs to:", table_dir)
    print("Saved figure outputs to:", figure_dir)


def parse_args() -> argparse.Namespace:
    default_root = Path(
        os.environ.get(
            "BIOSTAT_ROOT",
            "/Users/xinyichen/Desktop/BiostatisticsPracticum_Depression",
        )
    )

    default_table_dir = default_root / "final_cohort" / "results" / "tables" / "ml_llm"
    default_figure_dir = default_root / "final_cohort" / "results" / "figures" / "ml_llm"
    default_input = default_root / "final_cohort" / "results" / "tables" / "final_analysis_long.csv"

    parser = argparse.ArgumentParser(description="Run 1-year MSK prediction ML pipeline with LLM medication features.")

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to final_analysis_long.csv",
    )

    parser.add_argument(
        "--table-dir",
        type=Path,
        default=default_table_dir,
        help="Directory for CSV table outputs",
    )

    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=default_figure_dir,
        help="Directory for figure outputs",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Input:", args.input)
    print("Table directory:", args.table_dir)
    print("Figure directory:", args.figure_dir)

    df = load_data(args.input)
    run_experiment(
        df=df,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
    )


if __name__ == "__main__":
    main()
