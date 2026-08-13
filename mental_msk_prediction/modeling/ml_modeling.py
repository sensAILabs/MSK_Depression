"""
Machine-learning pipeline for one-year MSK prediction.

Input
-----
results/data/analysis/analysis_df_1year.csv

Main analyses
-------------
1. Training-set correlation analysis for quality control and reporting.
2. Stratified 70/30 train-test split.
3. Five-fold cross-validation comparison of:
   - Logistic Regression
   - Random Forest
   - XGBoost
4. Sequential feature-set comparison using the best algorithm.
5. Independent test-set evaluation and DeLong comparisons.
6. Final-model ROC curve, confusion matrix, and predictions.
7. Univariate Pearson correlation of eight demographic/clinical predictors with the outcome.
8. Medication-focused SHAP interpretation from the full final model.
9. Pearson correlation matrix for predictors retained in the final model.
10. Generalization analysis by age group.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mental_msk_prediction.config.paths import (
    RESULTS_DATA_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    create_project_directories,
)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# Settings
SEED = 123
TEST_SIZE = 0.30
N_SPLITS = 5
CLASSIFICATION_THRESHOLD = 0.50
HIGH_CORRELATION_THRESHOLD = 0.70

AGE_GROUP_BINS = [18, 35, 65, np.inf]
AGE_GROUP_LABELS = ["18-34", "35-64", "65+"]

INPUT_FILE = (RESULTS_DATA_DIR/ "analysis"/ "analysis_df_1year.csv")
MODEL_TABLE_DIR = (TABLES_DIR / "modeling")
MODEL_FIGURE_DIR = (FIGURES_DIR/ "modeling")
OUTCOME_COLUMN = "MSK_within_1year"
ID_COLUMN = "Patient_ID"
SEX_COLUMN = "Sex"


DEMOGRAPHIC_FEATURES = ["Age", "Sex",]
MENTAL_HEALTH_FEATURES = [
    "Mental_diagnosis_count",
    "Recent_30d_mental_visit",
    "Repeated_same_mental_diagnosis",
    "Unique_mental_diagnosis_count",
]
DISEASE_BURDEN_FEATURES = [
    "Preexisting_condition_count",
    "Multimorbidity_count",
]
MEDICATION_BURDEN_FEATURES = [
    "Other_medication_class_count",
]
ATC_INDICATOR_FEATURES = [
    "Any_Antipsychotics",
    "Any_Anxiolytics",
    "Any_Hypnotics_and_sedatives",
    "Any_Antidepressants",
    "Any_Psychostimulants_and_ADHD_medications",
    "Any_Psycholeptic_and_psychoanaleptic_combinations",
    "Any_Anti_inflammatory_and_antirheumatic_products",
    "Any_Topical_products_for_joint_and_muscular_pain",
    "Any_Muscle_relaxants",
    "Any_Opioids",
    "Any_Other_analgesics_and_antipyretics",
    "Any_Antimigraine_preparations",
]

ATC_DISCONTINUATION_FEATURES = [
    "Antipsychotics_discontinuation_count",
    "Anxiolytics_discontinuation_count",
    "Hypnotics_and_sedatives_discontinuation_count",
    "Antidepressants_discontinuation_count",
    "Anti_inflammatory_and_antirheumatic_products_discontinuation_count",
    "Muscle_relaxants_discontinuation_count",
    "Opioids_discontinuation_count",
    "Other_analgesics_and_antipyretics_discontinuation_count",
]

ATC_SWITCH_FEATURES = [
    "Antipsychotics_switch_count",
    "Anxiolytics_switch_count",
    "Hypnotics_and_sedatives_switch_count",
    "Antidepressants_switch_count",
    "Anti_inflammatory_and_antirheumatic_products_switch_count",
    "Muscle_relaxants_switch_count",
    "Opioids_switch_count",
    "Other_analgesics_and_antipyretics_switch_count",
]
TIME_TO_FIRST_MEDICATION_FEATURES = [
    "Time_to_first_mental_medication",
    "Time_to_first_pain_medication",
]

OVERLAP_RATIO_FEATURE = [
    "Mental_pain_overlap_ratio",
]

CLINICAL_PEARSON_FEATURES = (
    DEMOGRAPHIC_FEATURES
    + MENTAL_HEALTH_FEATURES
    + DISEASE_BURDEN_FEATURES
)

MEDICATION_SHAP_FEATURES = (
    MEDICATION_BURDEN_FEATURES
    + ATC_INDICATOR_FEATURES
    + ATC_SWITCH_FEATURES
    + ATC_DISCONTINUATION_FEATURES
    + TIME_TO_FIRST_MEDICATION_FEATURES
    + OVERLAP_RATIO_FEATURE
)

BASE_CLINICAL_FEATURES = (
    DEMOGRAPHIC_FEATURES
    + MENTAL_HEALTH_FEATURES
)

FINAL_PRESPECIFIED_FEATURES = list(
    dict.fromkeys(
        BASE_CLINICAL_FEATURES
        + DISEASE_BURDEN_FEATURES
        + MEDICATION_BURDEN_FEATURES
        + ATC_INDICATOR_FEATURES
        + ATC_DISCONTINUATION_FEATURES
        + ATC_SWITCH_FEATURES
        + TIME_TO_FIRST_MEDICATION_FEATURES
        + OVERLAP_RATIO_FEATURE
    )
)

ALL_ANALYSIS_FEATURES = FINAL_PRESPECIFIED_FEATURES

EXPECTED_PREDICTOR_COUNT = 40

if len(FINAL_PRESPECIFIED_FEATURES) != EXPECTED_PREDICTOR_COUNT:
    raise RuntimeError(
        "The prespecified Any-plus-selected-trajectory design must contain exactly "
        f"{EXPECTED_PREDICTOR_COUNT} predictors; "
        f"observed {len(FINAL_PRESPECIFIED_FEATURES)}."
    )


# General helpers
def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    table_name: str,
) -> None:
    """
    Raise an informative error when expected columns are missing.
    """
    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"{table_name} is missing required columns: "
            + ", ".join(missing)
        )


def save_csv(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a comma-delimited table.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )


def format_p_value(
    p_value: float,
) -> str:
    """
    Format p-values for result tables.
    """
    if pd.isna(p_value):
        return ""

    if p_value < 0.001:
        return "<0.001"

    return f"{p_value:.3f}"




def clean_short_label(
    name: str,
) -> str:
    """
    Create compact labels for large all-predictor figures.
    """
    replacements = {
        "Mental_diagnosis_count": "MentalDx_count",
        "Recent_30d_mental_visit": "Recent30d_mental_visit",
        "Repeated_same_mental_diagnosis": "Repeated_same_dx",
        "Maximum_visits_same_mental_diagnosis": "Max_same_dx_visits",
        "Unique_mental_diagnosis_count": "Unique_mental_dx",
        "Overall_mental_health_visit_count": "Mental_visit_count",
        "Preexisting_condition_count": "Preexisting_count",
        "Multimorbidity_count": "Multimorbidity",
       "Other_medication_class_count": "N_other_med_classes",
        "Medication_additions": "Med_additions",
        "Medication_discontinuations": "Med_discontinuations",
        "Mental_pain_overlap_ratio": "Mental_pain_overlap_ratio",
    }

    label = replacements.get(name, name)

    return (
        label
        .replace("_initiation_count", "_init")
        .replace("_discontinuation_count", "_disc")
        .replace("_active_days", "_days")
        .replace("_exposure_ratio", "_exposure")
        .replace("_switch_count", "_switch")
        .replace("_and_", "_")
        .replace("_medications", "_meds")
        .replace("_products", "")
        .replace("_preparations", "")
    )


def load_analysis_data(
    input_path: Path = INPUT_FILE,
) -> pd.DataFrame:
    """
    Load and validate the model-ready analysis dataframe.
    """
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Analysis dataframe not found: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        dtype={
            ID_COLUMN: "string",
            SEX_COLUMN: "string",
        },
        low_memory=False,
    )

    expected_predictors = ALL_ANALYSIS_FEATURES

    require_columns(
        df,
        [
            ID_COLUMN,
            OUTCOME_COLUMN,
            *expected_predictors,
        ],
        "Analysis dataframe",
    )

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(
            "analysis_df_1year.csv contains duplicated Patient_ID values."
        )

    df[OUTCOME_COLUMN] = (
        pd.to_numeric(
            df[OUTCOME_COLUMN],
            errors="coerce",
        )
        .astype("Int64")
    )

    if df[OUTCOME_COLUMN].isna().any():
        raise ValueError(
            f"{OUTCOME_COLUMN} contains missing or invalid values."
        )

    invalid_outcomes = sorted(
        set(
            df[OUTCOME_COLUMN]
            .astype(int)
            .unique()
        )
        - {0, 1}
    )

    if invalid_outcomes:
        raise ValueError(
            f"{OUTCOME_COLUMN} must contain only 0 and 1. "
            f"Invalid values: {invalid_outcomes}"
        )

    return df


def encode_sex(
    sex: pd.Series,
) -> pd.Series:
    """
    Encode Female as 0 and Male as 1.
    """
    normalized = (
        sex.astype("string")
        .str.strip()
        .str.upper()
    )

    encoded = normalized.map(
        {
            "F": 0,
            "FEMALE": 0,
            "M": 1,
            "MALE": 1,
        }
    )

    if encoded.isna().any():
        invalid = sorted(
            normalized[
                encoded.isna()
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Sex contains values that cannot be encoded: "
            + ", ".join(invalid)
        )

    return encoded.astype(int)



def prepare_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare the frozen numeric predictor matrix.

    Patient_ID is retained separately and is never used as a predictor.
    No interaction terms or additional derived variables are created.
    """
    X = df[
        ALL_ANALYSIS_FEATURES
    ].copy()

    X[SEX_COLUMN] = encode_sex(
        X[SEX_COLUMN]
    )

    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    if X.isna().any().any():
        missing_counts = X.isna().sum()
        missing_counts = missing_counts[
            missing_counts.gt(0)
        ]

        raise ValueError(
            "Predictors contain missing or invalid numeric values:\n"
            + missing_counts.to_string()
        )

    y = (
        df[OUTCOME_COLUMN]
        .astype(int)
        .copy()
    )

    patient_ids = (
        df[ID_COLUMN]
        .astype("string")
        .copy()
    )

    return X, y, patient_ids

def build_feature_sets(
    X: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Build the prespecified incremental Any-plus-selected-trajectory sets.

    The primary medication design contains:
    * medication burden;
    * twelve clinically interpretable Any ATC Level 3 indicators;
    * discontinuation and switching for eight selected ATC groups;
    * time to first mental and pain medication;
    * mental-pain overlap ratio.
    """
    feature_sets = {
        "01_Base_Clinical": (
            BASE_CLINICAL_FEATURES
        ),
        "02_Add_Disease_Burden": (
            BASE_CLINICAL_FEATURES
            + DISEASE_BURDEN_FEATURES
        ),
        "03_Add_Medication_Burden": (
            BASE_CLINICAL_FEATURES
            + DISEASE_BURDEN_FEATURES
            + MEDICATION_BURDEN_FEATURES
        ),
        "04_Add_Static_ATC_Any": (
            BASE_CLINICAL_FEATURES
            + DISEASE_BURDEN_FEATURES
            + MEDICATION_BURDEN_FEATURES
            + ATC_INDICATOR_FEATURES
        ),
        "05_Add_ATC_Switching": (
            BASE_CLINICAL_FEATURES
            + DISEASE_BURDEN_FEATURES
            + MEDICATION_BURDEN_FEATURES
            + ATC_INDICATOR_FEATURES
            + ATC_SWITCH_FEATURES
        ),
        "06_Add_ATC_Discontinuations": (
            BASE_CLINICAL_FEATURES
            + DISEASE_BURDEN_FEATURES
            + MEDICATION_BURDEN_FEATURES
            + ATC_INDICATOR_FEATURES
            + ATC_SWITCH_FEATURES
            + ATC_DISCONTINUATION_FEATURES
        ),
        "07_Add_Time_to_First": (
            BASE_CLINICAL_FEATURES
            + DISEASE_BURDEN_FEATURES
            + MEDICATION_BURDEN_FEATURES
            + ATC_INDICATOR_FEATURES
            + ATC_SWITCH_FEATURES
            + ATC_DISCONTINUATION_FEATURES
            + TIME_TO_FIRST_MEDICATION_FEATURES
        ),
        "08_Full_Any_Plus_Selected_Trajectory": (
            FINAL_PRESPECIFIED_FEATURES
        ),
    }

    output: Dict[str, List[str]] = {}

    for name, columns in feature_sets.items():
        unique_columns = list(
            dict.fromkeys(
                columns
            )
        )

        missing = [
            column
            for column in unique_columns
            if column not in X.columns
        ]

        if missing:
            raise KeyError(
                f"Feature set {name} is missing columns: "
                + ", ".join(missing)
            )

        output[name] = unique_columns

    return output


def identify_high_correlation_pairs(
    X: pd.DataFrame,
    feature_columns: List[str],
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """
    Identify unique predictor pairs with absolute Pearson correlation at or
    above the requested threshold.

    Correlations used for feature selection are calculated in the training
    set only.
    """
    usable_columns = [
        column
        for column in feature_columns
        if (
            column in X.columns
            and X[column].nunique(
                dropna=False
            ) > 1
        )
    ]

    if len(usable_columns) < 2:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "pearson_r",
                "absolute_pearson_r",
            ]
        )

    correlation_matrix = X[
        usable_columns
    ].corr(
        method="pearson"
    )

    rows: list[dict[str, object]] = []

    for first_index in range(
        len(
            usable_columns
        )
    ):
        for second_index in range(
            first_index + 1,
            len(
                usable_columns
            ),
        ):
            correlation = correlation_matrix.iloc[
                first_index,
                second_index,
            ]

            if (
                pd.notna(correlation)
                and abs(correlation) >= threshold
            ):
                rows.append(
                    {
                        "feature_1": usable_columns[
                            first_index
                        ],
                        "feature_2": usable_columns[
                            second_index
                        ],
                        "pearson_r": float(
                            correlation
                        ),
                        "absolute_pearson_r": float(
                            abs(
                                correlation
                            )
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "pearson_r",
                "absolute_pearson_r",
            ]
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "absolute_pearson_r",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def run_all_variable_correlation(
    X: pd.DataFrame,
    table_dir: Path,
    figure_dir: Path,
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> None:
    """
    Save the full predictor correlation matrix and an annotated heatmap.

    Values in the heatmap are displayed to two decimal places.
    """
    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    non_constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(
            dropna=False
        ) > 1
    ]

    if len(non_constant_columns) < 2:
        raise ValueError(
            "At least two non-constant predictors are required "
            "for correlation analysis."
        )

    corr = X[
        non_constant_columns
    ].corr(
        method="pearson"
    )

    corr.to_csv(
        table_dir
        / "01_all_variable_correlation_matrix.csv"
    )

    labels = [
        clean_short_label(column)
        for column in corr.columns
    ]

    n_features = len(labels)
    figure_size = max(
        15,
        0.48 * n_features,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_size,
            figure_size,
        )
    )

    image = ax.imshow(
        corr.to_numpy(),
        vmin=-1,
        vmax=1,
        aspect="equal",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Pearson correlation coefficient (r)"
    )

    ax.set_xticks(
        np.arange(n_features)
    )

    ax.set_yticks(
        np.arange(n_features)
    )

    ax.set_xticklabels(
        labels,
        rotation=90,
        fontsize=7,
    )

    ax.set_yticklabels(
        labels,
        fontsize=7,
    )

    rounded = corr.round(2)

    for row_index in range(n_features):
        for column_index in range(n_features):
            value = rounded.iloc[
                row_index,
                column_index,
            ]

            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=4.5,
            )

    ax.set_title(
        "Correlation Matrix of Candidate Predictors",
        pad=12,
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "01_all_variable_correlation_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    high_pairs: list[dict[str, object]] = []

    columns = corr.columns.tolist()

    for first_index in range(len(columns)):
        for second_index in range(
            first_index + 1,
            len(columns),
        ):
            correlation = corr.iloc[
                first_index,
                second_index,
            ]

            if (
                pd.notna(correlation)
                and abs(correlation) >= threshold
            ):
                high_pairs.append(
                    {
                        "feature_1": columns[
                            first_index
                        ],
                        "feature_2": columns[
                            second_index
                        ],
                        "pearson_r": correlation,
                        "absolute_pearson_r": abs(
                            correlation
                        ),
                    }
                )

    if high_pairs:
        high_pairs_df = (
            pd.DataFrame(high_pairs)
            .sort_values(
                "absolute_pearson_r",
                ascending=False,
            )
            .reset_index(drop=True)
        )
    else:
        high_pairs_df = pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "pearson_r",
                "absolute_pearson_r",
            ]
        )



# Models and metrics
def make_models(
    y_train: pd.Series,
) -> Dict[str, object]:
    """
    Define the three model families.
    """
    positive_n = int(
        y_train.eq(1).sum()
    )

    negative_n = int(
        y_train.eq(0).sum()
    )

    scale_pos_weight = (
        negative_n
        / max(
            positive_n,
            1,
        )
    )

    models: Dict[str, object] = {
        "Logistic_Regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        )
    else:
        print(
            "\nWarning: xgboost is not installed. "
            "XGBoost will be skipped."
        )

    return models


def make_pipeline(
    model_name: str,
    model: object,
) -> Pipeline:
    """
    Standardize predictors only for Logistic Regression.
    """
    if model_name == "Logistic_Regression":
        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    clone(model),
                ),
            ]
        )

    return Pipeline(
        steps=[
            (
                "model",
                clone(model),
            ),
        ]
    )



def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> Dict[str, float]:
    """
    Calculate discrimination and threshold-based classification metrics.
    """
    y_true_array = np.asarray(y_true, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)

    predictions = (
        probability_array >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "AUC": float(
            roc_auc_score(
                y_true_array,
                probability_array,
            )
        ),
        "Sensitivity": float(
            recall_score(
                y_true_array,
                predictions,
                zero_division=0,
            )
        ),
        "Specificity": float(
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        ),
        "Precision": float(
            precision_score(
                y_true_array,
                predictions,
                zero_division=0,
            )
        ),
        "Accuracy": float(
            accuracy_score(
                y_true_array,
                predictions,
            )
        ),
        "F1_score": float(
            f1_score(
                y_true_array,
                predictions,
                zero_division=0,
            )
        ),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Threshold": float(threshold),
    }


def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Run five-fold stratified cross-validation in the training set.
    """
    cross_validator = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    records: list[dict[str, float | int]] = []

    for fold_number, (
        fold_train_index,
        fold_validation_index,
    ) in enumerate(
        cross_validator.split(
            X_train,
            y_train,
        ),
        start=1,
    ):
        fold_pipeline = clone(
            pipeline
        )

        fold_pipeline.fit(
            X_train.iloc[
                fold_train_index
            ],
            y_train.iloc[
                fold_train_index
            ],
        )

        probabilities = (
            fold_pipeline
            .predict_proba(
                X_train.iloc[
                    fold_validation_index
                ]
            )[:, 1]
        )

        metrics = evaluate_predictions(
            y_train.iloc[
                fold_validation_index
            ],
            probabilities,
        )

        metrics["Fold"] = fold_number
        records.append(metrics)

    return pd.DataFrame(records)



def summarize_cross_validation(
    fold_results: pd.DataFrame,
) -> Dict[str, float]:
    """
    Summarize all cross-validation performance metrics.
    """
    summary: Dict[str, float] = {}

    for metric in [
        "AUC",
        "Sensitivity",
        "Specificity",
        "Precision",
        "Accuracy",
        "F1_score",
    ]:
        summary[f"CV_{metric}_mean"] = float(
            fold_results[metric].mean()
        )
        summary[f"CV_{metric}_sd"] = float(
            fold_results[metric].std(ddof=1)
        )

    return summary


def compute_midrank(
    values: np.ndarray,
) -> np.ndarray:
    """
    Compute midranks used by the fast DeLong implementation.
    """
    values = np.asarray(
        values
    )

    order = np.argsort(
        values
    )

    sorted_values = values[
        order
    ]

    n_values = len(
        values
    )

    midranks = np.zeros(
        n_values,
        dtype=float,
    )

    start = 0

    while start < n_values:
        end = start

        while (
            end < n_values
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        midranks[
            start:end
        ] = (
            0.5
            * (
                start
                + end
                - 1
            )
            + 1
        )

        start = end

    output = np.empty(
        n_values,
        dtype=float,
    )

    output[
        order
    ] = midranks

    return output


def fast_delong(
    predictions_sorted: np.ndarray,
    positive_n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate correlated AUC covariance using DeLong's method.
    """
    positive_count = positive_n
    negative_count = (
        predictions_sorted.shape[1]
        - positive_count
    )

    if (
        positive_count <= 0
        or negative_count <= 0
    ):
        raise ValueError(
            "DeLong test requires positive and negative observations."
        )

    classifier_count = (
        predictions_sorted.shape[0]
    )

    positive_predictions = (
        predictions_sorted[
            :,
            :positive_count,
        ]
    )

    negative_predictions = (
        predictions_sorted[
            :,
            positive_count:,
        ]
    )

    positive_ranks = np.empty(
        (
            classifier_count,
            positive_count,
        )
    )

    negative_ranks = np.empty(
        (
            classifier_count,
            negative_count,
        )
    )

    total_ranks = np.empty(
        (
            classifier_count,
            positive_count
            + negative_count,
        )
    )

    for row in range(
        classifier_count
    ):
        positive_ranks[row] = compute_midrank(
            positive_predictions[row]
        )

        negative_ranks[row] = compute_midrank(
            negative_predictions[row]
        )

        total_ranks[row] = compute_midrank(
            predictions_sorted[row]
        )

    aucs = (
        total_ranks[
            :,
            :positive_count,
        ].sum(
            axis=1
        )
        / (
            positive_count
            * negative_count
        )
        - (
            positive_count
            + 1.0
        )
        / (
            2.0
            * negative_count
        )
    )

    v01 = (
        total_ranks[
            :,
            :positive_count,
        ]
        - positive_ranks
    ) / negative_count

    v10 = (
        1.0
        - (
            total_ranks[
                :,
                positive_count:,
            ]
            - negative_ranks
        )
        / positive_count
    )

    covariance = (
        np.cov(v01)
        / positive_count
        + np.cov(v10)
        / negative_count
    )

    covariance = np.atleast_2d(
        covariance
    )

    return aucs, covariance


def delong_test(
    y_true: pd.Series,
    reference_probabilities: np.ndarray,
    comparison_probabilities: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compare two correlated test-set AUCs.
    """
    y_array = np.asarray(
        y_true,
        dtype=int,
    )

    if len(
        np.unique(
            y_array
        )
    ) != 2:
        raise ValueError(
            "DeLong test requires both outcome classes."
        )

    reference = np.asarray(
        reference_probabilities,
        dtype=float,
    )

    comparison = np.asarray(
        comparison_probabilities,
        dtype=float,
    )

    order = (
        -y_array
    ).argsort()

    positive_n = int(
        y_array.sum()
    )

    predictions_sorted = np.vstack(
        [
            reference[
                order
            ],
            comparison[
                order
            ],
        ]
    )

    aucs, covariance = fast_delong(
        predictions_sorted,
        positive_n,
    )

    contrast = np.array(
        [
            [
                1.0,
                -1.0,
            ]
        ]
    )

    variance = float(
        (
            contrast
            @ covariance
            @ contrast.T
        ).item()
    )

    if (
        variance <= 0
        or np.isnan(
            variance
        )
    ):
        return (
            float(
                aucs[0]
            ),
            float(
                aucs[1]
            ),
            1.0,
        )

    z_score = (
        aucs[0]
        - aucs[1]
    ) / math.sqrt(
        variance
    )

    p_value = (
        2.0
        * (
            1.0
            - stats.norm.cdf(
                abs(
                    z_score
                )
            )
        )
    )

    return (
        float(
            aucs[0]
        ),
        float(
            aucs[1]
        ),
        float(
            p_value
        ),
    )


def build_delong_table(
    prediction_dictionary: Dict[str, np.ndarray],
    y_test: pd.Series,
    reference_key: str,
    comparison_label: str,
) -> pd.DataFrame:
    """
    Compare every test-set prediction vector with a reference prediction.
    """
    if reference_key not in prediction_dictionary:
        raise KeyError(
            f"Reference prediction not found: {reference_key}"
        )

    reference_probabilities = (
        prediction_dictionary[
            reference_key
        ]
    )

    rows: list[dict[str, object]] = []

    for key, probabilities in prediction_dictionary.items():
        if key == reference_key:
            rows.append(
                {
                    comparison_label: key,
                    "Reference": reference_key,
                    "Reference_AUC": roc_auc_score(
                        y_test,
                        reference_probabilities,
                    ),
                    "Comparison_AUC": roc_auc_score(
                        y_test,
                        probabilities,
                    ),
                    "AUC_difference": 0.0,
                    "p_value": np.nan,
                    "p_value_formatted": "ref",
                }
            )
            continue

        reference_auc, comparison_auc, p_value = delong_test(
            y_true=y_test,
            reference_probabilities=reference_probabilities,
            comparison_probabilities=probabilities,
        )

        rows.append(
            {
                comparison_label: key,
                "Reference": reference_key,
                "Reference_AUC": reference_auc,
                "Comparison_AUC": comparison_auc,
                "AUC_difference": (
                    reference_auc
                    - comparison_auc
                ),
                "p_value": p_value,
                "p_value_formatted": format_p_value(
                    p_value
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def holm_adjust(
    p_values: Iterable[float],
) -> np.ndarray:
    """
    Holm step-down adjustment without an external dependency.
    """
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)

    valid_index = np.where(np.isfinite(values))[0]
    if valid_index.size == 0:
        return adjusted

    valid_values = values[valid_index]
    order = np.argsort(valid_values)
    ordered_values = valid_values[order]
    n_tests = len(ordered_values)

    ordered_adjusted = np.empty(n_tests, dtype=float)
    running_max = 0.0

    for rank, p_value in enumerate(ordered_values):
        candidate = min(
            (n_tests - rank) * p_value,
            1.0,
        )
        running_max = max(
            running_max,
            candidate,
        )
        ordered_adjusted[rank] = running_max

    reverse_order = np.empty_like(order)
    reverse_order[order] = np.arange(n_tests)

    adjusted_valid = ordered_adjusted[
        reverse_order
    ]
    adjusted[valid_index] = adjusted_valid
    return adjusted


def attach_delong_to_performance(
    performance: pd.DataFrame,
    delong_table: pd.DataFrame,
    key_column: str,
    reference_key: str,
) -> pd.DataFrame:
    """
    Add DeLong p-values and Holm-adjusted p-values to a performance matrix.
    """
    output = performance.copy()

    if delong_table.empty:
        output["Reference"] = reference_key
        output["AUC_difference_vs_reference"] = np.nan
        output["DeLong_p_value"] = np.nan
        output["Holm_adjusted_p_value"] = np.nan
        output["Is_reference"] = output[key_column].eq(reference_key)
        return output

    lookup = delong_table.rename(
        columns={
            "Reference": "DeLong_reference",
            "AUC_difference": "AUC_difference_vs_reference",
            "p_value": "DeLong_p_value",
        }
    )[
        [
            key_column,
            "DeLong_reference",
            "AUC_difference_vs_reference",
            "DeLong_p_value",
        ]
    ]

    output = output.merge(
        lookup,
        on=key_column,
        how="left",
        validate="one_to_one",
    )

    output["Reference"] = reference_key
    output["Is_reference"] = output[key_column].eq(reference_key)

    output.loc[
        output["Is_reference"],
        "AUC_difference_vs_reference",
    ] = 0.0
    output.loc[
        output["Is_reference"],
        "DeLong_p_value",
    ] = np.nan

    non_reference = ~output["Is_reference"]
    output["Holm_adjusted_p_value"] = np.nan
    output.loc[
        non_reference,
        "Holm_adjusted_p_value",
    ] = holm_adjust(
        output.loc[
            non_reference,
            "DeLong_p_value",
        ]
    )

    return output



def build_correlation_variants(
    full_features: List[str],
) -> Dict[str, List[str]]:
    """
    Deprecated compatibility helper.

    The current prespecified model reports high correlations but does not
    automatically delete predictors or compare correlation-pruned variants.
    """
    return {
        "00_Full_Prespecified": list(
            full_features
        )
    }



def plot_roc_curves(
    predictions: Dict[str, np.ndarray],
    y_test: pd.Series,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot several test-set ROC curves.
    """
    fig, ax = plt.subplots(
        figsize=(
            7,
            6,
        )
    )

    for label, probabilities in predictions.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test,
            probabilities,
        )

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        ax.plot(
            false_positive_rate,
            true_positive_rate,
            label=(
                f"{label} "
                f"(AUC={auc:.3f})"
            ),
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "False positive rate"
    )

    ax.set_ylabel(
        "True positive rate"
    )

    ax.set_title(
        title
    )

    ax.legend(
        loc="lower right",
        fontsize=8,
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_confusion_matrix(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot the final model confusion matrix at threshold 0.50.
    """
    predictions = (
        np.asarray(
            probabilities
        )
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(
        figsize=(
            5,
            4,
        )
    )

    image = ax.imshow(
        matrix,
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Predicted non-MSK",
            "Predicted MSK",
        ]
    )

    ax.set_yticklabels(
        [
            "Observed non-MSK",
            "Observed MSK",
        ]
    )

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=12,
            )

    ax.set_title(
        "Final Model Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# Model comparison
def compare_model_families(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    full_feature_columns: List[str],
    table_dir: Path,
    figure_dir: Path,
) -> Tuple[
    pd.DataFrame,
    Dict[str, np.ndarray],
    str,
]:
    """
    Compare Logistic Regression, Random Forest, and XGBoost using the same
    complete predictor set.
    """
    models = make_models(
        y_train
    )

    rows: list[dict[str, object]] = []
    test_predictions: Dict[
        str,
        np.ndarray,
    ] = {}

    fold_tables: list[
        pd.DataFrame
    ] = []

    for model_name, model in models.items():
        pipeline = make_pipeline(
            model_name,
            model,
        )

        fold_results = run_cross_validation(
            X_train[
                full_feature_columns
            ],
            y_train,
            pipeline,
        )

        fold_results.insert(
            0,
            "Model",
            model_name,
        )

        fold_tables.append(
            fold_results
        )

        cv_summary = summarize_cross_validation(
            fold_results
        )

        fitted_pipeline = clone(
            pipeline
        )

        fitted_pipeline.fit(
            X_train[
                full_feature_columns
            ],
            y_train,
        )

        probabilities = (
            fitted_pipeline
            .predict_proba(
                X_test[
                    full_feature_columns
                ]
            )[:, 1]
        )

        test_predictions[
            model_name
        ] = probabilities

        test_metrics = evaluate_predictions(
            y_test,
            probabilities,
        )

        rows.append(
            {
                "Model": model_name,
                "N_features": len(
                    full_feature_columns
                ),
                **cv_summary,
                **{
                    f"Test_{key}": value
                    for key, value
                    in test_metrics.items()
                },
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            "CV_AUC_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_model_name = str(
        results.iloc[0]["Model"]
    )

    # Use Logistic Regression as the prespecified statistical reference for
    # pairwise test-set AUC comparisons. Model selection remains based on
    # training cross-validation and is therefore independent of this reference.
    delong_reference_model = "Logistic_Regression"
    if delong_reference_model not in test_predictions:
        raise KeyError(
            "Logistic_Regression is required as the DeLong reference model."
        )

    delong_table = build_delong_table(
        prediction_dictionary=test_predictions,
        y_test=y_test,
        reference_key=delong_reference_model,
        comparison_label="Model",
    )

    performance_matrix = attach_delong_to_performance(
        performance=results,
        delong_table=delong_table,
        key_column="Model",
        reference_key=delong_reference_model,
    )

    save_csv(
        performance_matrix,
        table_dir
        / "02_model_comparison_performance_matrix.csv",
    )

    plot_roc_curves(
        predictions=test_predictions,
        y_test=y_test,
        output_path=(
            figure_dir
            / "02_model_comparison_roc.png"
        ),
        title=(
            "Test-Set ROC Curves by Model"
        ),
    )

    return (
        results,
        test_predictions,
        best_model_name,
    )


# Feature-set comparison
def compare_feature_sets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_sets: Dict[str, List[str]],
    selected_model_name: str,
    table_dir: Path,
    figure_dir: Path,
) -> Tuple[
    pd.DataFrame,
    Dict[str, np.ndarray],
    str,
]:
    """
    Compare sequential feature sets using the selected algorithm.
    """
    model = make_models(
        y_train
    )[selected_model_name]

    rows: list[dict[str, object]] = []
    predictions: Dict[
        str,
        np.ndarray,
    ] = {}

    fold_tables: list[
        pd.DataFrame
    ] = []

    for feature_set_name, columns in feature_sets.items():
        pipeline = make_pipeline(
            selected_model_name,
            model,
        )

        fold_results = run_cross_validation(
            X_train[
                columns
            ],
            y_train,
            pipeline,
        )

        fold_results.insert(
            0,
            "Feature_Set",
            feature_set_name,
        )

        fold_tables.append(
            fold_results
        )

        cv_summary = summarize_cross_validation(
            fold_results
        )

        fitted_pipeline = clone(
            pipeline
        )

        fitted_pipeline.fit(
            X_train[
                columns
            ],
            y_train,
        )

        probabilities = (
            fitted_pipeline
            .predict_proba(
                X_test[
                    columns
                ]
            )[:, 1]
        )

        predictions[
            feature_set_name
        ] = probabilities

        test_metrics = evaluate_predictions(
            y_test,
            probabilities,
        )

        rows.append(
            {
                "Feature_Set": feature_set_name,
                "Model": selected_model_name,
                "N_features": len(
                    columns
                ),
                **cv_summary,
                **{
                    f"Test_{key}": value
                    for key, value
                    in test_metrics.items()
                },
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            "CV_AUC_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_feature_set = str(
        results.iloc[0][
            "Feature_Set"
        ]
    )

    # Compare every candidate feature set with the prespecified complete model.
    # The highest-CV feature set is still calculated independently above, but
    # the statistical reference is fixed a priori for consistent reporting.
    delong_reference_feature_set = "08_Full_Any_Plus_Selected_Trajectory"
    if delong_reference_feature_set not in predictions:
        raise KeyError(
            "The prespecified full feature set is required as the DeLong reference."
        )

    delong_table = build_delong_table(
        prediction_dictionary=predictions,
        y_test=y_test,
        reference_key=delong_reference_feature_set,
        comparison_label="Feature_Set",
    )

    performance_matrix = attach_delong_to_performance(
        performance=results,
        delong_table=delong_table,
        key_column="Feature_Set",
        reference_key=delong_reference_feature_set,
    )

    save_csv(
        performance_matrix,
        table_dir
        / "03_feature_set_comparison_performance_matrix.csv",
    )

    plot_roc_curves(
        predictions=predictions,
        y_test=y_test,
        output_path=(
            figure_dir
            / "03_feature_set_comparison_roc.png"
        ),
        title=(
            f"Test-Set ROC Curves by Feature Set "
            f"({selected_model_name})"
        ),
    )

    return (
        results,
        predictions,
        best_feature_set,
    )



def fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    patient_ids_test: pd.Series,
    selected_model_name: str,
    final_feature_set_name: str,
    final_feature_columns: List[str],
    table_dir: Path,
    figure_dir: Path,
) -> Tuple[
    Pipeline,
    np.ndarray,
    pd.DataFrame,
]:
    """
    Fit the final model and save final test-set outputs.
    """
    model = make_models(
        y_train
    )[selected_model_name]

    pipeline = make_pipeline(
        selected_model_name,
        model,
    )

    final_pipeline = clone(
        pipeline
    )

    final_pipeline.fit(
        X_train[
            final_feature_columns
        ],
        y_train,
    )

    probabilities = (
        final_pipeline
        .predict_proba(
            X_test[
                final_feature_columns
            ]
        )[:, 1]
    )

    final_metrics = evaluate_predictions(
        y_test,
        probabilities,
    )

    final_results = pd.DataFrame(
        [
            {
                "Model": selected_model_name,
                "Feature_Set": final_feature_set_name,
                "N_features": len(
                    final_feature_columns
                ),
                "Threshold": CLASSIFICATION_THRESHOLD,
                **final_metrics,
            }
        ]
    )

    save_csv(
        final_results,
        table_dir
        / "04_final_model_test_performance.csv",
    )

    predictions = pd.DataFrame(
        {
            ID_COLUMN: (
                patient_ids_test
                .reset_index(
                    drop=True
                )
            ),
            "Observed_MSK_within_1year": (
                y_test
                .reset_index(
                    drop=True
                )
            ),
            "Predicted_probability": probabilities,
            "Predicted_class": (
                probabilities
                >= CLASSIFICATION_THRESHOLD
            ).astype(int),
        }
    )

    save_csv(
        predictions,
        table_dir
        / "04_final_model_test_predictions.csv",
    )

    plot_roc_curves(
        predictions={
            "Final model": probabilities,
        },
        y_test=y_test,
        output_path=(
            figure_dir
            / "04_final_model_roc.png"
        ),
        title=(
            "Final Model Test-Set ROC Curve"
        ),
    )

    plot_confusion_matrix(
        y_true=y_test,
        probabilities=probabilities,
        output_path=(
            figure_dir
            / "04_final_model_confusion_matrix.png"
        ),
    )

    return (
        final_pipeline,
        probabilities,
        final_results,
    )


def run_shap_analysis(
    final_pipeline: Pipeline,
    X_test: pd.DataFrame,
    final_feature_columns: List[str],
    table_dir: Path,
    figure_dir: Path,
) -> None:
    """
    Interpret medication-related predictors using SHAP values from the full
    final model.

    The model is still the full 40-predictor model. SHAP values are calculated
    for all final predictors, then the reporting output is restricted to the
    medication-related predictors defined in MEDICATION_SHAP_FEATURES.

    The CSV contains all medication-related predictors ranked by mean absolute
    SHAP value. The manuscript beeswarm displays the top 15 medication-related
    predictors only.
    """
    if not HAS_SHAP:
        print(
            "\nSHAP is not installed. SHAP analysis was skipped."
        )
        return

    model = final_pipeline.named_steps[
        "model"
    ]

    model_class_name = (
        model.__class__.__name__
    )

    if model_class_name not in {
        "XGBClassifier",
        "RandomForestClassifier",
    }:
        print(
            "\nFinal model is not tree-based. "
            "SHAP tree analysis was skipped."
        )
        return

    X_explain = (
        X_test[
            final_feature_columns
        ]
        .copy()
        .astype(float)
    )

    medication_features = [
        feature
        for feature in MEDICATION_SHAP_FEATURES
        if feature in final_feature_columns
    ]

    if not medication_features:
        raise ValueError(
            "No medication-related predictors were found in the final model."
        )

    missing_medication_features = [
        feature
        for feature in MEDICATION_SHAP_FEATURES
        if feature not in final_feature_columns
    ]
    if missing_medication_features:
        raise ValueError(
            "The final model is missing expected medication predictors: "
            + ", ".join(missing_medication_features)
        )

    try:
        if model_class_name == "XGBClassifier":
            import xgboost as xgb

            booster = model.get_booster()

            matrix = xgb.DMatrix(
                X_explain,
                feature_names=list(
                    X_explain.columns
                ),
            )

            contributions = booster.predict(
                matrix,
                pred_contribs=True,
            )

            shap_values = (
                contributions[
                    :,
                    :-1,
                ]
            )
        else:
            explainer = shap.TreeExplainer(
                model
            )

            raw_values = (
                explainer.shap_values(
                    X_explain
                )
            )

            if isinstance(
                raw_values,
                list,
            ):
                shap_values = raw_values[
                    1
                ]
            elif (
                isinstance(
                    raw_values,
                    np.ndarray,
                )
                and raw_values.ndim == 3
            ):
                shap_values = raw_values[
                    :,
                    :,
                    1,
                ]
            else:
                shap_values = raw_values

        feature_to_index = {
            feature: index
            for index, feature in enumerate(
                X_explain.columns
            )
        }

        medication_indices = [
            feature_to_index[feature]
            for feature in medication_features
        ]

        medication_shap_values = shap_values[
            :,
            medication_indices,
        ]

        X_medication = X_explain[
            medication_features
        ].copy()

        medication_importance = (
            pd.DataFrame(
                {
                    "Feature": medication_features,
                    "Mean_absolute_SHAP": np.abs(
                        medication_shap_values
                    ).mean(
                        axis=0
                    ),
                }
            )
            .sort_values(
                "Mean_absolute_SHAP",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        medication_importance.insert(
            0,
            "Rank",
            np.arange(
                1,
                len(
                    medication_importance
                )
                + 1,
            ),
        )

        save_csv(
            medication_importance,
            table_dir
            / "05_medication_shap_feature_importance.csv",
        )

        # Top 15 medication-related predictors from SHAP values calculated
        # within the full 40-predictor final model.
        shap_plot_features = X_medication.rename(
            columns={
                feature: clean_short_label(feature)
                for feature in X_medication.columns
            }
        )

        plt.figure(
            figsize=(9.5, 6.5)
        )

        shap.summary_plot(
            medication_shap_values,
            shap_plot_features,
            max_display=min(
                15,
                len(
                    medication_features
                ),
            ),
            show=False,
            plot_size=None,
        )

        shap_figure = plt.gcf()
        shap_axis = shap_figure.axes[0]

        shap_axis.grid(
            axis="y",
            linestyle=":",
            linewidth=0.7,
            alpha=0.30,
        )
        shap_axis.set_axisbelow(
            True
        )
        shap_axis.set_xlabel(
            "SHAP value (impact on model output)",
            fontsize=11,
        )
        shap_axis.tick_params(
            axis="both",
            labelsize=9,
        )
        shap_axis.set_title(
            "Top Medication-Related Predictors by SHAP",
            fontsize=12,
            pad=10,
        )

        plt.tight_layout()

        plt.savefig(
            figure_dir
            / "05_medication_shap_summary_plot.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            shap_figure
        )

    except Exception as error:
        print(
            f"\nSHAP analysis failed: {error}"
        )



# Final-model descriptive table
def _format_binary_summary(
    values: pd.Series,
) -> str:
    """Format a binary predictor as n (%), using 1 as the positive category."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()

    if numeric.empty:
        return ""

    n = len(numeric)
    positive = int(numeric.eq(1).sum())
    percentage = 100.0 * positive / n

    return f"{positive:,} ({percentage:.1f}%)"


def _format_continuous_summary(
    values: pd.Series,
) -> str:
    """Format a numeric predictor as median [Q1, Q3]."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()

    if numeric.empty:
        return ""

    median = float(numeric.median())
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))

    return f"{median:.2f} [{q1:.2f}, {q3:.2f}]"



def _fisher_p_value(
    values: pd.Series,
    y_numeric: pd.Series,
) -> float:
    """
    Fisher's exact p-value for a binary 0/1 variable.
    """
    valid = values.notna() & y_numeric.notna()

    binary_values = (
        pd.to_numeric(
            values.loc[valid],
            errors="coerce",
        )
        .astype(int)
    )

    binary_outcome = (
        y_numeric.loc[valid]
        .astype(int)
    )

    contingency = (
        pd.crosstab(
            binary_values,
            binary_outcome,
        )
        .reindex(
            index=[0, 1],
            columns=[0, 1],
            fill_value=0,
        )
    )

    if contingency.to_numpy().sum() == 0:
        return np.nan

    _, p_value = stats.fisher_exact(
        contingency.to_numpy()
    )

    return float(p_value)


def _mannwhitney_p_value(
    values: pd.Series,
    y_numeric: pd.Series,
) -> float:
    """
    Two-sided Mann-Whitney U p-value for a numeric variable.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    no_msk = numeric.loc[
        y_numeric.eq(0)
    ].dropna()

    msk = numeric.loc[
        y_numeric.eq(1)
    ].dropna()

    if no_msk.empty or msk.empty:
        return np.nan

    _, p_value = stats.mannwhitneyu(
        no_msk,
        msk,
        alternative="two-sided",
    )

    return float(p_value)


def _format_event_summary(
    values: pd.Series,
) -> str:
    """
    Format a zero-inflated event count as n (%) with at least one event.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return ""

    event = numeric.gt(0)
    n_event = int(event.sum())
    percentage = 100.0 * n_event / len(numeric)

    return f"{n_event:,} ({percentage:.1f}%)"


def _format_positive_continuous_summary(
    values: pd.Series,
) -> str:
    """
    Median [Q1, Q3] among observations with a value greater than zero.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    numeric = numeric.loc[
        numeric.gt(0)
    ]

    if numeric.empty:
        return ""

    median = float(
        numeric.median()
    )
    q1 = float(
        numeric.quantile(0.25)
    )
    q3 = float(
        numeric.quantile(0.75)
    )

    return (
        f"{median:.2f} "
        f"[{q1:.2f}, {q3:.2f}]"
    )


def _descriptive_section(
    feature: str,
) -> str:
    """
    Assign each final predictor to a manuscript-friendly clinical domain.
    """
    if feature in (
        DEMOGRAPHIC_FEATURES
        + MENTAL_HEALTH_FEATURES
    ):
        return "Demographic and mental health characteristics"

    if feature in DISEASE_BURDEN_FEATURES:
        return "Disease burden"

    if (
        feature in MEDICATION_BURDEN_FEATURES
        or feature in ATC_INDICATOR_FEATURES
    ):
        return "Medication burden and exposure"

    if feature in ATC_DISCONTINUATION_FEATURES:
        return "Medication discontinuation"

    if feature in ATC_SWITCH_FEATURES:
        return "Medication switching"

    if (
        feature in TIME_TO_FIRST_MEDICATION_FEATURES
        or feature in OVERLAP_RATIO_FEATURE
    ):
        return "Medication timing and overlap"

    return "Other"


def build_final_model_descriptive_table_conventional(
    X: pd.DataFrame,
    y: pd.Series,
    final_feature_columns: List[str],
    table_dir: Path,
) -> pd.DataFrame:
    """
    Version 1: conventional manuscript descriptive table.

    Binary predictors are presented as n (%); all other numeric predictors,
    including switching/discontinuation counts and the overlap ratio, are
    presented as median [Q1, Q3].

    Binary comparisons use Fisher's exact test. Numeric comparisons use the
    two-sided Mann-Whitney U test.

    This table changes reporting only; the final model is unchanged.
    """
    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_numeric = (
        pd.to_numeric(
            y,
            errors="coerce",
        )
        .astype(int)
    )

    no_msk_mask = y_numeric.eq(0)
    msk_mask = y_numeric.eq(1)

    rows: list[dict[str, object]] = []

    for order, feature in enumerate(
        final_feature_columns,
        start=1,
    ):
        values = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

        observed = values.dropna()
        unique_values = set(
            observed.unique().tolist()
        )

        is_binary = (
            bool(unique_values)
            and unique_values.issubset(
                {0, 1}
            )
        )

        if is_binary:
            summary_type = "n (%)"

            overall_summary = _format_binary_summary(
                values
            )
            no_msk_summary = _format_binary_summary(
                values.loc[
                    no_msk_mask
                ]
            )
            msk_summary = _format_binary_summary(
                values.loc[
                    msk_mask
                ]
            )

            p_value = _fisher_p_value(
                values,
                y_numeric,
            )

        else:
            summary_type = "Median [IQR]"

            overall_summary = _format_continuous_summary(
                values
            )
            no_msk_summary = _format_continuous_summary(
                values.loc[
                    no_msk_mask
                ]
            )
            msk_summary = _format_continuous_summary(
                values.loc[
                    msk_mask
                ]
            )

            p_value = _mannwhitney_p_value(
                values,
                y_numeric,
            )

        rows.append(
            {
                "Order": order,
                "Section": _descriptive_section(
                    feature
                ),
                "Variable": feature,
                "Summary": summary_type,
                "Overall": overall_summary,
                "No MSK within 1 year": no_msk_summary,
                "MSK within 1 year": msk_summary,
                "P value": format_p_value(
                    p_value
                ),
            }
        )

    descriptive = pd.DataFrame(
        rows
    )

    save_csv(
        descriptive,
        table_dir
        / "05a_final_model_descriptive_table_conventional.csv",
    )

    return descriptive


def build_final_model_descriptive_table_adapted(
    X: pd.DataFrame,
    y: pd.Series,
    final_feature_columns: List[str],
    table_dir: Path,
) -> pd.DataFrame:
    """
    Version 2: study-specific descriptive table adapted for zero-inflated
    longitudinal medication variables.

    Reporting rules:
      * Binary predictors: n (%).
      * Ordinary numeric/count predictors: median [Q1, Q3].
      * Switching/discontinuation counts: n (%) with >=1 event.
      * Mental-pain overlap ratio:
          1) any overlap, n (%);
          2) ratio among patients with overlap, median [Q1, Q3].

    The original count and continuous variables remain unchanged in the model.
    Only their descriptive presentation is modified.
    """
    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_numeric = (
        pd.to_numeric(
            y,
            errors="coerce",
        )
        .astype(int)
    )

    no_msk_mask = y_numeric.eq(0)
    msk_mask = y_numeric.eq(1)

    rows: list[dict[str, object]] = []
    display_order = 0

    for feature in final_feature_columns:
        values = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

        section = _descriptive_section(
            feature
        )

        # ----------------------------------------------------------
        # Highly zero-inflated switching/discontinuation counts:
        # present the proportion with at least one event.
        # ----------------------------------------------------------
        if (
            feature in ATC_SWITCH_FEATURES
            or feature in ATC_DISCONTINUATION_FEATURES
        ):
            display_order += 1

            event_values = (
                values.gt(0)
                .where(
                    values.notna(),
                    np.nan,
                )
            )

            p_value = _fisher_p_value(
                event_values,
                y_numeric,
            )

            if feature in ATC_SWITCH_FEATURES:
                display_name = (
                    feature
                    .replace(
                        "_switch_count",
                        " (>=1 switch)",
                    )
                )
            else:
                display_name = (
                    feature
                    .replace(
                        "_discontinuation_count",
                        " (>=1 discontinuation)",
                    )
                )

            rows.append(
                {
                    "Order": display_order,
                    "Section": section,
                    "Variable": display_name,
                    "Summary": "n (%)",
                    "Overall": _format_event_summary(
                        values
                    ),
                    "No MSK within 1 year": _format_event_summary(
                        values.loc[
                            no_msk_mask
                        ]
                    ),
                    "MSK within 1 year": _format_event_summary(
                        values.loc[
                            msk_mask
                        ]
                    ),
                    "P value": format_p_value(
                        p_value
                    ),
                }
            )

            continue

        # ----------------------------------------------------------
        # Zero-inflated overlap ratio:
        # split descriptive reporting into presence + magnitude.
        # ----------------------------------------------------------
        if feature == "Mental_pain_overlap_ratio":
            display_order += 1

            any_overlap = (
                values.gt(0)
                .where(
                    values.notna(),
                    np.nan,
                )
            )

            any_overlap_p = _fisher_p_value(
                any_overlap,
                y_numeric,
            )

            rows.append(
                {
                    "Order": display_order,
                    "Section": section,
                    "Variable": "Any mental-pain medication overlap",
                    "Summary": "n (%)",
                    "Overall": _format_event_summary(
                        values
                    ),
                    "No MSK within 1 year": _format_event_summary(
                        values.loc[
                            no_msk_mask
                        ]
                    ),
                    "MSK within 1 year": _format_event_summary(
                        values.loc[
                            msk_mask
                        ]
                    ),
                    "P value": format_p_value(
                        any_overlap_p
                    ),
                }
            )

            display_order += 1

            # Compare continuous overlap magnitude only among patients
            # who actually had overlap.
            positive_values = values.where(
                values.gt(0)
            )

            positive_overlap_p = _mannwhitney_p_value(
                positive_values,
                y_numeric,
            )

            rows.append(
                {
                    "Order": display_order,
                    "Section": section,
                    "Variable": (
                        "Mental-pain overlap ratio "
                        "among patients with overlap"
                    ),
                    "Summary": "Median [IQR]",
                    "Overall": _format_positive_continuous_summary(
                        values
                    ),
                    "No MSK within 1 year": _format_positive_continuous_summary(
                        values.loc[
                            no_msk_mask
                        ]
                    ),
                    "MSK within 1 year": _format_positive_continuous_summary(
                        values.loc[
                            msk_mask
                        ]
                    ),
                    "P value": format_p_value(
                        positive_overlap_p
                    ),
                }
            )

            continue

        # ----------------------------------------------------------
        # All other features use their conventional summary.
        # ----------------------------------------------------------
        display_order += 1

        observed = values.dropna()
        unique_values = set(
            observed.unique().tolist()
        )

        is_binary = (
            bool(unique_values)
            and unique_values.issubset(
                {0, 1}
            )
        )

        if is_binary:
            summary_type = "n (%)"

            overall_summary = _format_binary_summary(
                values
            )
            no_msk_summary = _format_binary_summary(
                values.loc[
                    no_msk_mask
                ]
            )
            msk_summary = _format_binary_summary(
                values.loc[
                    msk_mask
                ]
            )

            p_value = _fisher_p_value(
                values,
                y_numeric,
            )

        else:
            summary_type = "Median [IQR]"

            overall_summary = _format_continuous_summary(
                values
            )
            no_msk_summary = _format_continuous_summary(
                values.loc[
                    no_msk_mask
                ]
            )
            msk_summary = _format_continuous_summary(
                values.loc[
                    msk_mask
                ]
            )

            p_value = _mannwhitney_p_value(
                values,
                y_numeric,
            )

        rows.append(
            {
                "Order": display_order,
                "Section": section,
                "Variable": feature,
                "Summary": summary_type,
                "Overall": overall_summary,
                "No MSK within 1 year": no_msk_summary,
                "MSK within 1 year": msk_summary,
                "P value": format_p_value(
                    p_value
                ),
            }
        )

    descriptive = pd.DataFrame(
        rows
    )

    save_csv(
        descriptive,
        table_dir
        / "05b_final_model_descriptive_table_adapted.csv",
    )

    return descriptive

# Generalization analyses
def run_feature_outcome_pearson(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    final_feature_columns: List[str],
    table_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    """
    Calculate univariate Pearson correlations between the eight prespecified
    demographic/clinical predictors and the binary one-year MSK outcome.

    The analysis uses the training set only. Medication predictors are excluded
    from this Pearson reporting figure because medication-related model
    interpretation is presented separately with SHAP.

    This does not alter model training or the final 40-predictor model.
    """
    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clinical_features = [
        feature
        for feature in CLINICAL_PEARSON_FEATURES
        if feature in final_feature_columns
    ]

    missing_clinical_features = [
        feature
        for feature in CLINICAL_PEARSON_FEATURES
        if feature not in final_feature_columns
    ]
    if missing_clinical_features:
        raise ValueError(
            "The final model is missing expected clinical predictors: "
            + ", ".join(missing_clinical_features)
        )

    if len(clinical_features) != 8:
        raise RuntimeError(
            "Clinical Pearson reporting must contain exactly 8 predictors; "
            f"observed {len(clinical_features)}."
        )

    y_numeric = pd.to_numeric(
        y_train,
        errors="coerce",
    ).astype(float)

    rows: list[dict[str, object]] = []

    for feature in clinical_features:
        values = pd.to_numeric(
            X_train[feature],
            errors="coerce",
        ).astype(float)

        valid = (
            values.notna()
            & y_numeric.notna()
        )

        feature_values = values.loc[
            valid
        ]
        outcome_values = y_numeric.loc[
            valid
        ]

        if (
            len(feature_values) < 3
            or feature_values.nunique(
                dropna=True
            ) < 2
            or outcome_values.nunique(
                dropna=True
            ) < 2
        ):
            pearson_r = np.nan
            p_value = np.nan
        else:
            pearson_r, p_value = stats.pearsonr(
                feature_values.to_numpy(),
                outcome_values.to_numpy(),
            )

        rows.append(
            {
                "Feature": feature,
                "N_complete": int(
                    valid.sum()
                ),
                "Pearson_r": float(
                    pearson_r
                ),
                "Absolute_Pearson_r": float(
                    abs(
                        pearson_r
                    )
                ),
                "P_value": float(
                    p_value
                ),
            }
        )

    results = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "Pearson_r",
            ascending=True,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    save_csv(
        results,
        table_dir
        / "07_clinical_feature_outcome_pearson.csv",
    )

    plot_data = results.dropna(
        subset=[
            "Pearson_r",
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8.8, 5.2)
    )

    ax.barh(
        [
            clean_short_label(
                feature
            )
            for feature in plot_data[
                "Feature"
            ]
        ],
        plot_data[
            "Pearson_r"
        ],
    )

    ax.axvline(
        0,
        linestyle="-",
        linewidth=0.9,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.30,
    )
    ax.set_axisbelow(
        True
    )

    ax.set_xlabel(
        "Pearson correlation coefficient (r)",
        fontsize=11,
    )
    ax.set_ylabel(
        "",
    )
    ax.set_title(
        "Clinical Predictors and One-Year MSK Outcome",
        fontsize=12,
        pad=10,
    )
    ax.tick_params(
        axis="both",
        labelsize=9,
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "07_clinical_feature_outcome_pearson.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return results



def run_final_model_correlation(
    X_train: pd.DataFrame,
    final_feature_columns: List[str],
    table_dir: Path,
    figure_dir: Path,
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save a numbered Pearson correlation matrix for predictors retained in the
    final model, together with the corresponding high-correlation-pair table.

    The matrix uses the training set. Every cell is annotated with its Pearson
    correlation coefficient rounded to two decimal places.
    """
    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    non_constant_columns = [
        feature
        for feature in final_feature_columns
        if (
            feature in X_train.columns
            and X_train[
                feature
            ].nunique(
                dropna=False
            ) > 1
        )
    ]

    constant_columns = [
        feature
        for feature in final_feature_columns
        if feature not in non_constant_columns
    ]

    if len(non_constant_columns) < 2:
        raise ValueError(
            "At least two non-constant final predictors are required "
            "for the final-model correlation matrix."
        )

    correlation_matrix = (
        X_train[
            non_constant_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .corr(
            method="pearson"
        )
    )

    high_pairs = identify_high_correlation_pairs(
        X=X_train,
        feature_columns=non_constant_columns,
        threshold=threshold,
    )

    labels = [
        clean_short_label(
            feature
        )
        for feature in correlation_matrix.columns
    ]

    n_features = len(labels)

    # A large square figure is intentional so that all numeric annotations
    # remain readable when the PNG is opened at full resolution.
    figure_size = max(
        20,
        0.72 * n_features,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_size,
            figure_size,
        )
    )

    image = ax.imshow(
        correlation_matrix.to_numpy(),
        vmin=-1,
        vmax=1,
        aspect="equal",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Pearson correlation coefficient (r)"
    )

    ax.set_xticks(
        np.arange(n_features)
    )
    ax.set_yticks(
        np.arange(n_features)
    )

    ax.set_xticklabels(
        labels,
        rotation=90,
        fontsize=8,
    )
    ax.set_yticklabels(
        labels,
        fontsize=8,
    )

    rounded = correlation_matrix.round(2)

    for row_index in range(n_features):
        for column_index in range(n_features):
            value = rounded.iloc[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if abs(value) >= 0.55
                else "black"
            )

            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=text_color,
            )

    ax.set_title(
        "Pearson Correlation Matrix of Final-Model Predictors",
        pad=14,
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "08_final_model_correlation_matrix.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return (
        correlation_matrix,
        high_pairs,
    )


def subgroup_metrics(
    subgroup_name: str,
    subgroup_value: str,
    mask: pd.Series,
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, object]:
    """
    Calculate final-model performance in one subgroup.
    """
    mask_array = (
        mask.to_numpy(
            dtype=bool
        )
    )

    y_group = y_test.loc[
        mask
    ]

    probability_group = probabilities[
        mask_array
    ]

    row: dict[str, object] = {
        "Subgroup": subgroup_name,
        "Level": subgroup_value,
        "N": int(
            len(
                y_group
            )
        ),
        "MSK_N": int(
            y_group.sum()
        ),
        "MSK_rate": (
            float(
                y_group.mean()
            )
            if len(
                y_group
            ) > 0
            else np.nan
        ),
    }

    if (
        len(
            y_group
        ) == 0
        or y_group.nunique() < 2
    ):
        row.update(
            {
                "AUC": np.nan,
                "Sensitivity": np.nan,
                "Specificity": np.nan,
                "Precision": np.nan,
                "Accuracy": np.nan,
                "F1_score": np.nan,
                "TP": np.nan,
                "TN": np.nan,
                "FP": np.nan,
                "FN": np.nan,
            }
        )
    else:
        row.update(
            evaluate_predictions(
                y_group,
                probability_group,
            )
        )

    return row



def run_generalization_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    table_dir: Path,
    figure_dir: Path,
) -> None:
    """
    Evaluate final-model performance across the prespecified age groups only.
    """
    rows: list[
        dict[str, object]
    ] = []

    age_groups = pd.cut(
        X_test["Age"],
        bins=AGE_GROUP_BINS,
        labels=AGE_GROUP_LABELS,
        right=False,
        include_lowest=True,
    )

    for age_group in AGE_GROUP_LABELS:
        rows.append(
            subgroup_metrics(
                subgroup_name="Age group",
                subgroup_value=age_group,
                mask=(
                    age_groups
                    == age_group
                ),
                y_test=y_test,
                probabilities=probabilities,
            )
        )

    results = pd.DataFrame(
        rows
    )

    save_csv(
        results,
        table_dir
        / "07_generalization_by_age_group.csv",
    )

    plot_data = results.dropna(
        subset=["AUC"]
    ).copy()

    if not plot_data.empty:
        fig, ax = plt.subplots(
            figsize=(
                7,
                5,
            )
        )

        ax.bar(
            plot_data[
                "Level"
            ],
            plot_data[
                "AUC"
            ],
        )

        ax.set_ylim(
            0.5,
            min(
                1.0,
                max(
                    0.8,
                    float(
                        plot_data[
                            "AUC"
                        ].max()
                    )
                    + 0.05,
                ),
            ),
        )

        ax.set_ylabel(
            "AUC"
        )

        ax.set_xlabel(
            "Age group"
        )

        ax.set_title(
            "Final Model Generalization by Age Group"
        )

        plt.tight_layout()

        plt.savefig(
            figure_dir
            / "07_generalization_by_age_group.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

def main() -> None:
    """
    Run the prespecified one-year MSK prediction workflow.

    Terminal output is intentionally concise. Detailed model, feature-set,
    DeLong, descriptive, SHAP, correlation, prediction, and subgroup results
    are written to the modeling table/figure directories.
    """
    create_project_directories()
    MODEL_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    analysis = load_analysis_data(INPUT_FILE)
    X, y, patient_ids = prepare_feature_matrix(analysis)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        patient_ids_train,
        patient_ids_test,
    ) = train_test_split(
        X,
        y,
        patient_ids,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    patient_ids_train = patient_ids_train.reset_index(drop=True)
    patient_ids_test = patient_ids_test.reset_index(drop=True)

    save_csv(
        pd.DataFrame(
            [
                {
                    "Dataset": "Training",
                    "N": len(y_train),
                    "MSK_N": int(y_train.sum()),
                    "MSK_rate": float(y_train.mean()),
                },
                {
                    "Dataset": "Testing",
                    "N": len(y_test),
                    "MSK_N": int(y_test.sum()),
                    "MSK_rate": float(y_test.mean()),
                },
            ]
        ),
        MODEL_TABLE_DIR / "00_train_test_split_summary.csv",
    )

    feature_sets = build_feature_sets(X)
    feature_records: list[dict[str, object]] = []
    for feature_set, columns in feature_sets.items():
        for order, feature in enumerate(columns, start=1):
            feature_records.append(
                {
                    "Feature_Set": feature_set,
                    "Order": order,
                    "Feature": feature,
                }
            )

    save_csv(
        pd.DataFrame(feature_records),
        MODEL_TABLE_DIR / "01_feature_set_definitions.csv",
    )

    final_feature_set_name = "08_Full_Any_Plus_Selected_Trajectory"
    final_features = feature_sets[final_feature_set_name]

    # Correlation quality control uses training data only.
    run_all_variable_correlation(
        X=X_train[final_features],
        table_dir=MODEL_TABLE_DIR,
        figure_dir=MODEL_FIGURE_DIR,
    )
    high_pairs = identify_high_correlation_pairs(
        X=X_train,
        feature_columns=final_features,
        threshold=HIGH_CORRELATION_THRESHOLD,
    )
    # Full-model algorithm comparison.
    (
        model_results,
        model_predictions,
        best_model_name,
    ) = compare_model_families(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        full_feature_columns=final_features,
        table_dir=MODEL_TABLE_DIR,
        figure_dir=MODEL_FIGURE_DIR,
    )

    # Incremental feature-set comparison using the selected algorithm.
    (
        feature_set_results,
        feature_set_predictions,
        best_cv_feature_set_name,
    ) = compare_feature_sets(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_sets=feature_sets,
        selected_model_name=best_model_name,
        table_dir=MODEL_TABLE_DIR,
        figure_dir=MODEL_FIGURE_DIR,
    )

    descriptive_table_conventional = (
        build_final_model_descriptive_table_conventional(
            X=X,
            y=y,
            final_feature_columns=final_features,
            table_dir=MODEL_TABLE_DIR,
        )
    )

    descriptive_table_adapted = (
        build_final_model_descriptive_table_adapted(
            X=X,
            y=y,
            final_feature_columns=final_features,
            table_dir=MODEL_TABLE_DIR,
        )
    )

    if (
        descriptive_table_conventional.empty
        or descriptive_table_adapted.empty
    ):
        raise RuntimeError(
            "One or both final-model descriptive tables were not generated."
        )

    (
        final_pipeline,
        final_probabilities,
        final_results,
    ) = fit_final_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        patient_ids_test=patient_ids_test,
        selected_model_name=best_model_name,
        final_feature_set_name=final_feature_set_name,
        final_feature_columns=final_features,
        table_dir=MODEL_TABLE_DIR,
        figure_dir=MODEL_FIGURE_DIR,
    )

    # Final-model interpretation and generalization outputs.
    run_feature_outcome_pearson(
        X_train=X_train,
        y_train=y_train,
        final_feature_columns=final_features,
        table_dir=MODEL_TABLE_DIR,
        figure_dir=MODEL_FIGURE_DIR,
    )

    run_shap_analysis(
        final_pipeline=final_pipeline,
        X_test=X_test,
        final_feature_columns=final_features,
        table_dir=MODEL_TABLE_DIR,
        figure_dir=MODEL_FIGURE_DIR,
    )

    run_final_model_correlation(
        X_train=X_train,
        final_feature_columns=final_features,
        table_dir=MODEL_TABLE_DIR,
        figure_dir=MODEL_FIGURE_DIR,
    )

    run_generalization_analysis(
        X_test=X_test,
        y_test=y_test,
        probabilities=final_probabilities,
        table_dir=MODEL_TABLE_DIR,
        figure_dir=MODEL_FIGURE_DIR,
    )
    # Minimal terminal summary; detailed results are saved to CSV/PNG files.
    print("\n" + "=" * 72)
    print("MENTAL → MSK MODELING SUMMARY")
    print("=" * 72)
    print(f"Patients            : {len(analysis):,}")
    print(f"Outcome prevalence  : {y.mean():.1%}")
    print(f"Train / test        : {len(y_train):,} / {len(y_test):,}")
    print(f"Final predictors    : {len(final_features)}")
    print(
        f"High-corr pairs     : {len(high_pairs)}  "
        f"(|r| >= {HIGH_CORRELATION_THRESHOLD:.2f})"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
