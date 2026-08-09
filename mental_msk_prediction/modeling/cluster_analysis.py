from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


# Settings
TOP_ERROR_PERCENT = 20.0
RANDOM_STATE = 42

MIN_K = 2
MAX_K = 8

# K-selection rule:
# 1) keep K values whose silhouette is within 0.02 of the best silhouette;
# 2) require every cluster to contain at least 5% of the difficult-patient group;
# 3) among eligible solutions, choose the highest K-means vs Ward-hierarchical ARI;
# 4) ties: higher silhouette, then smaller K.
SILHOUETTE_TOLERANCE = 0.02
MIN_CLUSTER_PERCENT = 5.0

FINAL_FEATURE_SET = "08_Full_Any_Plus_Selected_Trajectory"

PREDICTION_TYPE_ORDER = ["TP", "TN", "FP", "FN"]


def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    table_name: str,
) -> None:
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


def safe_numeric_frame(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return (
        dataframe
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def format_p_value(
    p_value: float,
) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def prettify_feature_name(
    feature: str,
) -> str:
    labels = {
        "Mental_diagnosis_count": "Mental Diagnosis Count",
        "Recent_30d_mental_visit": "Recent 30-Day Mental Visit",
        "Repeated_same_mental_diagnosis": "Repeated Same Mental Diagnosis",
        "Unique_mental_diagnosis_count": "Unique Mental Diagnosis Count",
        "Preexisting_condition_count": "Pre-existing Condition Count",
        "Multimorbidity_count": "Multimorbidity Count",
        "Mental_medication_count": "Mental Medication Count",
        "Pain_medication_count": "Pain Medication Count",
        "Other_medication_count": "Other Medication Count",
        "Time_to_first_mental_medication": "Time to First Mental Medication",
        "Time_to_first_pain_medication": "Time to First Pain Medication",
        "Mental_pain_overlap_ratio": "Mental-Pain Medication Overlap Ratio",
    }

    if feature in labels:
        return labels[feature]

    if feature.startswith("Any_"):
        return "Any " + feature.removeprefix("Any_").replace("_", " ")

    if feature.endswith("_discontinuation_count"):
        return (
            feature
            .removesuffix("_discontinuation_count")
            .replace("_", " ")
            + " Discontinuations"
        )

    if feature.endswith("_switch_count"):
        return (
            feature
            .removesuffix("_switch_count")
            .replace("_", " ")
            + " Switches"
        )

    return feature.replace("_", " ").title()


def load_cluster_inputs(
    ml_table_dir: Path,
    sensitivity_table_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    pd.DataFrame,
]:
    """
    Load the frozen final-model outputs needed for cluster analysis.

    Required modeling outputs
    -------------------------
    04_final_model_test_performance.csv
    05_final_model_features.csv

    Required sensitivity outputs
    ----------------------------
    00_reconstructed_final_test_feature_matrix.csv
    02_patient_level_predictions_with_quartiles.csv

    No model is refitted in this script.
    """
    performance_path = ( ml_table_dir/ "04_final_model_test_performance.csv")
    feature_list_path = (ml_table_dir/ "05_final_model_features.csv")
    feature_matrix_path = (sensitivity_table_dir/ "00_reconstructed_final_test_feature_matrix.csv")
    prediction_path = (sensitivity_table_dir/ "02_patient_level_predictions_with_quartiles.csv")

    required_paths = [
        performance_path,
        feature_list_path,
        feature_matrix_path,
        prediction_path,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Run the final modeling and sensitivity-analysis pipelines first. "
            "Missing files: "
            + ", ".join(str(path) for path in missing_paths)
        )

    performance = pd.read_csv(performance_path)
    feature_list = pd.read_csv(feature_list_path)
    feature_matrix = pd.read_csv(
        feature_matrix_path,
        dtype={"Patient_ID": "string"},
    )
    predictions = pd.read_csv(
        prediction_path,
        dtype={"Patient_ID": "string"},
    )

    if performance.empty:
        raise ValueError(
            "04_final_model_test_performance.csv is empty."
        )

    require_columns(
        performance,
        ["Model", "Feature_Set", "N_features"],
        "Final model performance",
    )

    require_columns(
        feature_list,
        ["Feature"],
        "Final model feature list",
    )

    final_features = (
        feature_list["Feature"]
        .astype(str)
        .tolist()
    )

    if not final_features:
        raise ValueError(
            "05_final_model_features.csv contains no predictors."
        )

    require_columns(
        feature_matrix,
        ["Patient_ID", *final_features],
        "Reconstructed final test feature matrix",
    )

    require_columns(
        predictions,
        [
            "Patient_ID",
            "true_msk_1yr",
            "pred_prob",
            "pred_label",
            "prediction_error",
            "prediction_type",
        ],
        "Sensitivity patient-level prediction table",
    )

    for name, dataframe in {
        "feature matrix": feature_matrix,
        "patient predictions": predictions,
    }.items():
        if dataframe["Patient_ID"].duplicated().any():
            raise ValueError(
                f"Patient_ID is duplicated in {name}."
            )

    feature_ids = set(
        feature_matrix["Patient_ID"].astype(str)
    )
    prediction_ids = set(
        predictions["Patient_ID"].astype(str)
    )

    if feature_ids != prediction_ids:
        raise ValueError(
            "Patient IDs differ between the reconstructed test feature "
            "matrix and the sensitivity prediction table."
        )

    order = predictions[["Patient_ID"]].copy()

    feature_matrix = order.merge(
        feature_matrix[
            ["Patient_ID", *final_features]
        ],
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )

    metadata = pd.DataFrame(
        [
            {
                "Final_Model": str(
                    performance.iloc[0]["Model"]
                ),
                "Final_Feature_Set": str(
                    performance.iloc[0]["Feature_Set"]
                ),
                "N_Final_Model_Features": len(
                    final_features
                ),
            }
        ]
    )

    if (
        metadata.loc[0, "Final_Feature_Set"]
        != FINAL_FEATURE_SET
    ):
        raise ValueError(
            "Cluster analysis expected final feature set "
            f"{FINAL_FEATURE_SET!r}, but modeling output reports "
            f"{metadata.loc[0, 'Final_Feature_Set']!r}."
        )

    if int(
        metadata.loc[0, "N_Final_Model_Features"]
    ) != len(final_features):
        raise ValueError(
            "Final predictor count is inconsistent between modeling outputs."
        )

    return (
        predictions,
        feature_matrix,
        final_features,
        metadata,
    )


# Difficult-patient selection
def select_highest_error_patients(
    predictions: pd.DataFrame,
    top_percent: float = TOP_ERROR_PERCENT,
) -> pd.DataFrame:
    """
    Select the requested percentage of held-out test patients with the
    largest observed absolute prediction error.
    """
    if not 0 < top_percent < 100:
        raise ValueError(
            "top_percent must be between 0 and 100."
        )

    ranked = (
        predictions
        .sort_values(
            ["prediction_error", "Patient_ID"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    selected_n = int(
        np.ceil(
            len(ranked)
            * top_percent
            / 100
        )
    )

    difficult = (
        ranked
        .iloc[:selected_n]
        .copy()
        .reset_index(drop=True)
    )

    difficult["Highest_Error_Rank"] = (
        np.arange(
            1,
            len(difficult) + 1,
        )
    )

    return difficult


# K selection
def evaluate_candidate_k(
    X_scaled: np.ndarray,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
) -> pd.DataFrame:
    """
    Evaluate candidate K values using:
    - K-means inertia;
    - silhouette score;
    - agreement with Ward hierarchical clustering (ARI);
    - minimum cluster size.
    """
    upper_k = min(
        max_k,
        X_scaled.shape[0] - 1,
    )

    if upper_k < min_k:
        raise ValueError(
            "Too few difficult patients to evaluate clustering."
        )

    rows: list[dict[str, float | int]] = []

    for k in range(
        min_k,
        upper_k + 1,
    ):
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=100,
        )

        kmeans_labels = (
            kmeans.fit_predict(
                X_scaled
            )
        )

        hierarchical = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        )

        hierarchical_labels = (
            hierarchical.fit_predict(
                X_scaled
            )
        )

        cluster_sizes = np.bincount(
            kmeans_labels,
            minlength=k,
        )

        rows.append(
            {
                "K": k,
                "Inertia": float(
                    kmeans.inertia_
                ),
                "Silhouette_Score": float(
                    silhouette_score(
                        X_scaled,
                        kmeans_labels,
                    )
                ),
                "KMeans_Hierarchical_ARI": float(
                    adjusted_rand_score(
                        kmeans_labels,
                        hierarchical_labels,
                    )
                ),
                "Minimum_Cluster_N": int(
                    cluster_sizes.min()
                ),
                "Minimum_Cluster_Percent": float(
                    100
                    * cluster_sizes.min()
                    / len(kmeans_labels)
                ),
                "Maximum_Cluster_N": int(
                    cluster_sizes.max()
                ),
            }
        )

    return pd.DataFrame(rows)


def choose_optimal_k(
    evaluation: pd.DataFrame,
) -> tuple[int, pd.DataFrame]:
    """
    Select K using the prespecified balanced rule.
    """
    output = evaluation.copy()

    best_silhouette = float(
        output[
            "Silhouette_Score"
        ].max()
    )

    output[
        "Silhouette_Difference_From_Best"
    ] = (
        best_silhouette
        - output["Silhouette_Score"]
    )

    output[
        "Within_Silhouette_Tolerance"
    ] = (
        output[
            "Silhouette_Difference_From_Best"
        ]
        <= SILHOUETTE_TOLERANCE
    )

    output[
        "Meets_Minimum_Cluster_Size"
    ] = (
        output[
            "Minimum_Cluster_Percent"
        ]
        >= MIN_CLUSTER_PERCENT
    )

    output[
        "Eligible_For_Selection"
    ] = (
        output[
            "Within_Silhouette_Tolerance"
        ]
        & output[
            "Meets_Minimum_Cluster_Size"
        ]
    )

    eligible = output[
        output[
            "Eligible_For_Selection"
        ]
    ].copy()

    if eligible.empty:
        selected_row = (
            output
            .sort_values(
                [
                    "Silhouette_Score",
                    "K",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[0]
        )
    else:
        selected_row = (
            eligible
            .sort_values(
                [
                    "KMeans_Hierarchical_ARI",
                    "Silhouette_Score",
                    "K",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

    optimal_k = int(
        selected_row["K"]
    )

    output["Selected_K"] = (
        output["K"]
        == optimal_k
    )

    return optimal_k, output


# Final clustering
def fit_difficult_patient_clusters(
    difficult_patients: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    final_features: list[str],
    optimal_k: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
    np.ndarray,
    float,
    float,
]:
    """
    Fit the final K-means solution in the highest-error patient group only.
    """
    difficult_features = (
        difficult_patients[
            ["Patient_ID"]
        ]
        .merge(
            feature_matrix,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )
    )

    X = safe_numeric_frame(
        difficult_features[
            final_features
        ]
    )

    variances = X.var(axis=0)

    clustering_features = (
        variances[
            variances > 0
        ]
        .index
        .tolist()
    )

    constant_features = (
        variances[
            variances <= 0
        ]
        .index
        .tolist()
    )

    if len(clustering_features) < 2:
        raise RuntimeError(
            "Fewer than two non-constant final predictors are available "
            "for clustering."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[clustering_features])
    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=RANDOM_STATE,
        n_init=100,
    )

    labels_zero_based = (kmeans.fit_predict(X_scaled))
    labels = (labels_zero_based + 1)
    hierarchical = AgglomerativeClustering(
        n_clusters=optimal_k,
        linkage="ward",
    )

    hierarchical_labels = (hierarchical.fit_predict(X_scaled))
    silhouette = float(
        silhouette_score(
            X_scaled,
            labels_zero_based,
        )
    )

    ari = float(
        adjusted_rand_score(
            labels_zero_based,
            hierarchical_labels,
        )
    )

    clustered_patients = (
        difficult_patients
        .reset_index(drop=True)
        .copy()
    )

    clustered_patients[
        "KMeans_Cluster"
    ] = labels

    # Build the analysis table from two explicit sources:
    #   1) patient-level clustering metadata from the difficult-patient table;
    #   2) the frozen final-model feature matrix, merged by Patient_ID.
    #
    # This guarantees that each of the final predictors appears exactly once
    # and that cluster profiling always uses the same 42 predictors as the
    # final prediction model.
    metadata_columns = [
        "Patient_ID",
        "true_msk_1yr",
        "pred_prob",
        "pred_label",
        "prediction_error",
        "prediction_type",
        "Highest_Error_Rank",
        "KMeans_Cluster",
    ]

    available_metadata_columns = [
        column
        for column in metadata_columns
        if column in clustered_patients.columns
    ]

    clustered_features = (
        clustered_patients[
            available_metadata_columns
        ]
        .merge(
            pd.concat(
                [
                    difficult_features[
                        ["Patient_ID"]
                    ].reset_index(
                        drop=True
                    ),
                    X[
                        final_features
                    ].reset_index(
                        drop=True
                    ),
                ],
                axis=1,
            ),
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )
    )

    if clustered_features.columns.duplicated().any():
        duplicates = (
            clustered_features.columns[
                clustered_features.columns.duplicated()
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate columns remain in clustered feature table: "
            + ", ".join(
                duplicates
            )
        )

    return (
        clustered_patients,
        clustered_features,
        clustering_features,
        constant_features,
        X_scaled,
        silhouette,
        ari,
    )


# Cluster summaries
def create_cluster_size_summary(
    clustered_patients: pd.DataFrame,
) -> pd.DataFrame:
    total_n = len(
        clustered_patients
    )

    table = (
        clustered_patients
        .groupby(
            "KMeans_Cluster",
            as_index=False,
        )
        .agg(
            N=(
                "Patient_ID",
                "size",
            ),
            Mean_Prediction_Error=(
                "prediction_error",
                "mean",
            ),
            Median_Prediction_Error=(
                "prediction_error",
                "median",
            ),
            Mean_Predicted_Probability=(
                "pred_prob",
                "mean",
            ),
            MSK_Outcome_Rate=(
                "true_msk_1yr",
                "mean",
            ),
        )
    )

    table[
        "Percent_of_Difficult_Patients"
    ] = (
        100
        * table["N"]
        / total_n
    )

    return table


def create_prediction_type_distribution(
    clustered_patients: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for cluster_id, subset in (
        clustered_patients
        .groupby(
            "KMeans_Cluster"
        )
    ):
        denominator = len(
            subset
        )

        for prediction_type in (
            PREDICTION_TYPE_ORDER
        ):
            n = int(
                (
                    subset[
                        "prediction_type"
                    ]
                    == prediction_type
                ).sum()
            )

            rows.append(
                {
                    "KMeans_Cluster": int(
                        cluster_id
                    ),
                    "Prediction_Type": prediction_type,
                    "N": n,
                    "Percent_Within_Cluster": (
                        100
                        * n
                        / denominator
                        if denominator > 0
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def create_cluster_profile_table(
    clustered_features: pd.DataFrame,
    final_features: list[str],
) -> pd.DataFrame:
    """
    Create the main cluster phenotype table.

    Binary variables:
        n (%) with value = 1; overall cluster comparison by chi-square.

    Continuous/count variables:
        median [Q1, Q3]; overall cluster comparison by Kruskal-Wallis.

    P-values are descriptive/exploratory because the same features were used
    to form the clusters.
    """
    cluster_ids = sorted(
        clustered_features[
            "KMeans_Cluster"
        ].unique()
    )

    rows: list[
        dict[str, object]
    ] = []

    for feature in final_features:
        values = pd.to_numeric(
            clustered_features[
                feature
            ],
            errors="coerce",
        )

        unique_values = set(
            values
            .dropna()
            .unique()
            .tolist()
        )

        is_binary = (
            len(unique_values) <= 2
            and unique_values.issubset(
                {0, 1}
            )
        )

        row: dict[
            str,
            object,
        ] = {
            "Feature": feature,
            "Display_Feature": (
                prettify_feature_name(
                    feature
                )
            ),
            "Feature_Type": (
                "Binary"
                if is_binary
                else "Continuous/count"
            ),
        }

        if is_binary:
            contingency: list[
                list[int]
            ] = []

            for cluster_id in cluster_ids:
                cluster_values = (
                    values[
                        clustered_features[
                            "KMeans_Cluster"
                        ]
                        == cluster_id
                    ]
                    .dropna()
                )

                n_positive = int(
                    (
                        cluster_values
                        == 1
                    ).sum()
                )

                row[
                    f"Cluster_{int(cluster_id)}"
                ] = (
                    f"{n_positive} "
                    f"({100 * n_positive / len(cluster_values):.1f}%)"
                    if len(cluster_values) > 0
                    else "NA"
                )

                contingency.append(
                    [
                        int(
                            (
                                cluster_values
                                == 0
                            ).sum()
                        ),
                        n_positive,
                    ]
                )

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

            test_name = "Chi-square"

        else:
            arrays: list[
                pd.Series
            ] = []

            for cluster_id in cluster_ids:
                cluster_values = (
                    values[
                        clustered_features[
                            "KMeans_Cluster"
                        ]
                        == cluster_id
                    ]
                    .dropna()
                )

                arrays.append(
                    cluster_values
                )

                if len(
                    cluster_values
                ) == 0:
                    summary = "NA"
                else:
                    q1 = float(
                        cluster_values.quantile(
                            0.25
                        )
                    )
                    median = float(
                        cluster_values.median()
                    )
                    q3 = float(
                        cluster_values.quantile(
                            0.75
                        )
                    )

                    summary = (
                        f"{median:.2f} "
                        f"[{q1:.2f}, {q3:.2f}]"
                    )

                row[
                    f"Cluster_{int(cluster_id)}"
                ] = summary

            valid_arrays = [
                array
                for array in arrays
                if len(array) > 0
            ]

            try:
                if len(
                    valid_arrays
                ) >= 2:
                    statistic, p_value = (
                        stats.kruskal(
                            *valid_arrays
                        )
                    )
                else:
                    statistic = np.nan
                    p_value = np.nan
            except ValueError:
                statistic = np.nan
                p_value = np.nan

            test_name = "Kruskal-Wallis"

        row["Test"] = test_name
        row["Statistic"] = statistic
        row["P_value"] = p_value
        row[
            "P_value_formatted"
        ] = format_p_value(
            p_value
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# Figures
def plot_elbow(
    k_evaluation: pd.DataFrame,
    optimal_k: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        k_evaluation["K"],
        k_evaluation["Inertia"],
        marker="o",
    )

    ax.axvline(
        optimal_k,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Number of clusters (K)"
    )
    ax.set_ylabel(
        "K-means inertia"
    )
    ax.set_title(
        "Elbow Plot for Difficult-Patient Clustering"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_silhouette(
    k_evaluation: pd.DataFrame,
    optimal_k: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        k_evaluation["K"],
        k_evaluation[
            "Silhouette_Score"
        ],
        marker="o",
    )

    ax.axvline(
        optimal_k,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Number of clusters (K)"
    )
    ax.set_ylabel(
        "Silhouette score"
    )
    ax.set_title(
        "Silhouette Score by Number of Clusters"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_ari(
    k_evaluation: pd.DataFrame,
    optimal_k: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        k_evaluation["K"],
        k_evaluation[
            "KMeans_Hierarchical_ARI"
        ],
        marker="o",
    )

    ax.axvline(
        optimal_k,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Number of clusters (K)"
    )
    ax.set_ylabel(
        "Adjusted Rand index"
    )
    ax.set_title(
        "K-means Agreement with Ward Hierarchical Clustering"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pca_clusters(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    """
    PCA is used only for two-dimensional visualization.
    Clustering is performed in the complete standardized predictor space.
    """
    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    coordinates = (
        pca.fit_transform(
            X_scaled
        )
    )

    pca_table = pd.DataFrame(
        {
            "PC1": coordinates[:, 0],
            "PC2": coordinates[:, 1],
            "KMeans_Cluster": (
                cluster_labels
            ),
        }
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    for cluster_id in sorted(
        pca_table[
            "KMeans_Cluster"
        ].unique()
    ):
        subset = pca_table[
            pca_table[
                "KMeans_Cluster"
            ]
            == cluster_id
        ]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=18,
            alpha=0.65,
            label=(
                f"Cluster {int(cluster_id)}"
            ),
        )

    explained = (
        100
        * pca.explained_variance_ratio_
    )

    ax.set_xlabel(
        f"PC1 ({explained[0]:.1f}% variance)"
    )
    ax.set_ylabel(
        f"PC2 ({explained[1]:.1f}% variance)"
    )
    ax.set_title(
        "PCA Visualization of Highest-Error Patient Clusters"
    )

    ax.legend(
        title="Cluster"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    pca_table[
        "PC1_Explained_Variance_Percent"
    ] = explained[0]

    pca_table[
        "PC2_Explained_Variance_Percent"
    ] = explained[1]

    return pca_table


def run_cluster_analysis(
    ml_table_dir: Path,
    sensitivity_table_dir: Path,
    outdir: Path,
    top_error_percent: float = TOP_ERROR_PERCENT,
) -> None:
    tables_dir = (outdir/ "tables"/ "cluster_analysis")
    figures_dir = ( outdir/ "figures"/ "cluster_analysis")
    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove only stale outputs created by this final cluster-analysis script.
    for old_csv in tables_dir.glob(
        "*.csv"
    ):
        old_csv.unlink()

    for old_png in figures_dir.glob(
        "*.png"
    ):
        old_png.unlink()

    (
        predictions,
        feature_matrix,
        final_features,
        metadata,
    ) = load_cluster_inputs(
        ml_table_dir=ml_table_dir,
        sensitivity_table_dir=sensitivity_table_dir,
    )

    difficult_patients = (
        select_highest_error_patients(
            predictions,
            top_percent=top_error_percent,
        )
    )

    difficult_features = (
        difficult_patients[
            ["Patient_ID"]
        ]
        .merge(
            feature_matrix,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )
    )

    X_difficult = safe_numeric_frame(
        difficult_features[
            final_features
        ]
    )

    variances = (
        X_difficult.var(
            axis=0
        )
    )

    clustering_features = (
        variances[
            variances > 0
        ]
        .index
        .tolist()
    )

    if len(
        clustering_features
    ) < 2:
        raise RuntimeError(
            "Fewer than two non-constant predictors are available "
            "for clustering."
        )

    scaler = StandardScaler()

    X_scaled_for_k = (
        scaler.fit_transform(
            X_difficult[
                clustering_features
            ]
        )
    )

    k_evaluation = (
        evaluate_candidate_k(
            X_scaled_for_k
        )
    )

    (
        optimal_k,
        k_evaluation,
    ) = choose_optimal_k(
        k_evaluation
    )

    (
        clustered_patients,
        clustered_features,
        final_clustering_features,
        constant_features,
        X_scaled,
        final_silhouette,
        final_ari,
    ) = fit_difficult_patient_clusters(
        difficult_patients=difficult_patients,
        feature_matrix=feature_matrix,
        final_features=final_features,
        optimal_k=optimal_k,
    )

    # 01 — K selection
    k_evaluation.to_csv(
        tables_dir
        / "01_optimal_k_selection.csv",
        index=False,
    )

    # 02 — Cluster size / prediction summary
    cluster_sizes = (
        create_cluster_size_summary(
            clustered_patients
        )
    )

    cluster_sizes[
        "Selected_K"
    ] = optimal_k

    cluster_sizes[
        "Final_Silhouette"
    ] = final_silhouette

    cluster_sizes[
        "KMeans_Hierarchical_ARI"
    ] = final_ari

    cluster_sizes[
        "N_Final_Model_Features"
    ] = len(
        final_features
    )

    cluster_sizes[
        "N_Clustering_Features"
    ] = len(
        final_clustering_features
    )

    cluster_sizes.to_csv(
        tables_dir
        / "02_cluster_size_summary.csv",
        index=False,
    )

    # 03 — Main cluster phenotype table
    profile_table = (
        create_cluster_profile_table(
            clustered_features,
            final_features,
        )
    )

    profile_table.to_csv(tables_dir/ "03_cluster_profile_table.csv",index=False,)
    # 04 — TP/TN/FP/FN composition
    prediction_types = (
        create_prediction_type_distribution(
            clustered_patients
        )
    )

    prediction_types.to_csv(tables_dir/ "04_prediction_type_distribution.csv",index=False,)
    # 05 — Patient-level assignments for reproducibility
    clustered_patients.to_csv(
        tables_dir
        / "05_difficult_patient_cluster_assignments.csv",
        index=False,
    )

    # 06 — Clustering metadata / feature QC
    pd.DataFrame(
        [
            {
                "Final_Model": metadata.loc[
                    0,
                    "Final_Model",
                ],
                "Final_Feature_Set": metadata.loc[
                    0,
                    "Final_Feature_Set",
                ],
                "N_Test_Patients": len(
                    predictions
                ),
                "Top_Error_Percent_Clustered": (
                    top_error_percent
                ),
                "N_Difficult_Patients": len(
                    difficult_patients
                ),
                "N_Final_Model_Features": len(
                    final_features
                ),
                "N_Nonconstant_Clustering_Features": len(
                    final_clustering_features
                ),
                "Selected_K": optimal_k,
                "Silhouette_Score": final_silhouette,
                "KMeans_Hierarchical_ARI": final_ari,
                "K_Selection_Rule": (
                    "Highest ARI among K values within "
                    f"{SILHOUETTE_TOLERANCE:.2f} of the best silhouette "
                    "and with minimum cluster size >= "
                    f"{MIN_CLUSTER_PERCENT:.1f}%"
                ),
                "Interpretation_Note": (
                    "Clusters represent distinct clinical phenotypes among "
                    "highest-error patients identified in the independent test set. "
                    "They are intended to characterize heterogeneity rather than "
                    "define prospective patient subgroups."
                ),
            }
        ]
    ).to_csv(
        tables_dir
        / "06_cluster_analysis_metadata.csv",
        index=False,
    )

    # Figures
    plot_elbow(k_evaluation,optimal_k,figures_dir/ "01_elbow_plot.png",)
    plot_silhouette(k_evaluation,optimal_k,figures_dir/ "02_silhouette_by_k.png",)
    plot_ari(k_evaluation,optimal_k,figures_dir / "03_ari_by_k.png",)

    pca_table = (
        plot_pca_clusters(
            X_scaled,
            clustered_patients[
                "KMeans_Cluster"
            ].to_numpy(),
            figures_dir
            / "04_pca_clusters.png",
        )
    )

    pca_table.insert(
        0,
        "Patient_ID",
        clustered_patients[
            "Patient_ID"
        ].reset_index(
            drop=True
        ),
    )

    pca_table.to_csv(tables_dir/ "07_pca_coordinates.csv",index=False,)

    # Concise terminal output.
    selected_row = (
        k_evaluation[
            k_evaluation[
                "Selected_K"
            ]
        ]
        .iloc[0]
    )

    print( "\n"+ "=" * 72)
    print("MENTAL → MSK CLUSTER ANALYSIS SUMMARY")
    print( "=" * 72)
    print(f"Test patients         : {len(predictions):,}")
    print(
        f"Highest-error patients: {len(difficult_patients):,} "
        f"({top_error_percent:.1f}%)"
    )
    print(f"Final predictors      : {len(final_features)}")
    print(f"Clustering predictors : {len(final_clustering_features)}")
    print(f"Selected K            : {optimal_k}")
    print(f"Silhouette            : {final_silhouette:.3f}")
    print(f"K-means vs Ward ARI   : {final_ari:.3f}")

    for row in (
        cluster_sizes
        .sort_values(
            "KMeans_Cluster"
        )
        .itertuples(
            index=False
        )
    ):
        print(
            f"Cluster {int(row.KMeans_Cluster):>2} "
            f"            : {int(row.N):,} "
            f"({row.Percent_of_Difficult_Patients:.1f}%)"
        )

    print(
        "=" * 72
    )


def parse_args() -> argparse.Namespace:
    package_dir = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    parser = argparse.ArgumentParser(
        description=(
                        "Characterize clinical phenotypes among the highest-error "
                        "patients identified from the frozen final one-year MSK "
                        "prediction model."
    )
        )

    parser.add_argument(
        "--ml-table-dir",
        type=Path,
        default=(
            package_dir
            / "results"
            / "tables"
            / "modeling"
        ),
    )

    parser.add_argument(
        "--sensitivity-table-dir",
        type=Path,
        default=(
            package_dir
            / "results"
            / "tables"
            / "sensitivity_analysis"
        ),
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=(
            package_dir
            / "results"
        ),
    )

    parser.add_argument(
        "--top-error-percent",
        "--top_error_percent",
        dest="top_error_percent",
        type=float,
        default=TOP_ERROR_PERCENT,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_cluster_analysis(
        ml_table_dir=args.ml_table_dir,
        sensitivity_table_dir=args.sensitivity_table_dir,
        outdir=args.outdir,
        top_error_percent=args.top_error_percent,
    )


if __name__ == "__main__":
    main()
