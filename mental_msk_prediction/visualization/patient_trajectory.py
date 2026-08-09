"""
Create the final representative one-year patient-trajectory figure for the
mental-to-MSK prediction study.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from mental_msk_prediction.config.paths import (
    CLEANED_DIAGNOSIS_FILE,
    CLEANED_MEDICATION_FILE,
    FIGURES_DIR,
    RESULTS_DATA_DIR,
    TABLES_DIR,
    create_project_directories,
)

from mental_msk_prediction.features.build_analysis_df import (
    prepare_cohort,
    prepare_diagnosis,
    prepare_medication,
)


# Paths
ANALYSIS_DATA_DIR = RESULTS_DATA_DIR / "analysis"
ANALYSIS_DF_FILE = ANALYSIS_DATA_DIR / "analysis_df_1year.csv"
FINAL_COHORT_FILE = (RESULTS_DATA_DIR/ "cohort"/ "final_cohort.csv")
SENSITIVITY_TABLE_DIR = TABLES_DIR / "sensitivity_analysis"
OUTPUT_TABLE_DIR = TABLES_DIR / "patient_trajectory"
OUTPUT_FIGURE_DIR = FIGURES_DIR / "patient_trajectory"
OUTPUT_FIGURE_FILE = (OUTPUT_FIGURE_DIR/ "representative_patient_trajectories.png")


# Study / display settings
FOLLOW_UP_DAYS = 365
DISPLAY_MAX_DAYS = 365

TRAJECTORY_TYPES = [
    "Good example",
    "Normal example",
    "Bad example 1",
    "Bad example 2",
]

LINESTYLES = {
    "Good example": "-",
    "Normal example": "--",
    "Bad example 1": "-.",
    "Bad example 2": ":",
}

MARKER_X_OFFSETS = {
    "Good example": 0.0,
    "Normal example": 2.5,
    "Bad example 1": 5.0,
    "Bad example 2": 7.5,
}

PATIENT_COLORS = {
    "Good example": "#2ca02c",      # green
    "Normal example": "#1f77b4",    # blue
    "Bad example 1": "#ff7f0e",     # orange
    "Bad example 2": "#d62728",     # red
}

# ordinary snapshot = small triangle; frequency/count change = large triangle.
SMALL_TRIANGLE_SIZE = 45
LARGE_TRIANGLE_SIZE = 150


# General helpers
def require_columns(
    dataframe: pd.DataFrame,
    required: Iterable[str],
    table_name: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"{table_name} is missing required columns: "
            + ", ".join(missing)
        )


def normalize_patient_id(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
    )


def safe_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(default)
    )


def first_existing_file(
    directory: Path,
    pattern: str,
) -> Path:
    matches = sorted(
        directory.glob(pattern)
    )

    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern!r} was found in:\n"
            f"  {directory}"
        )

    return matches[0]


def resolve_sensitivity_group_files(
) -> dict[str, Path]:
    """
    Resolve the current sensitivity CSV names without depending on the exact
    suffix used by the sensitivity-analysis script.
    """
    return {
        "High applicability": first_existing_file(
            SENSITIVITY_TABLE_DIR,
            "03_high_applicability*.csv",
        ),
        "Intermediate applicability": first_existing_file(
            SENSITIVITY_TABLE_DIR,
            "04_intermediate_applicability*.csv",
        ),
        "Low applicability": first_existing_file(
            SENSITIVITY_TABLE_DIR,
            "05_low_applicability*.csv",
        ),
    }


# Load current pipeline outputs
def validate_inputs() -> dict[str, Path]:
    group_files = resolve_sensitivity_group_files()

    required_files = [
        ANALYSIS_DF_FILE,
        FINAL_COHORT_FILE,
        CLEANED_DIAGNOSIS_FILE,
        CLEANED_MEDICATION_FILE,
        *group_files.values(),
    ]

    missing = [
        path
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Patient-trajectory inputs are missing:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            )
        )

    return group_files


def load_current_cohort() -> pd.DataFrame:
    """
    Load and prepare the exact final cohort using the same helper as the
    frozen build_analysis_df.py pipeline.
    """
    cohort = pd.read_csv(
        FINAL_COHORT_FILE,
        dtype=str,
        low_memory=False,
    )

    cohort = prepare_cohort(
        cohort
    )

    cohort["Patient_ID"] = normalize_patient_id(
        cohort["Patient_ID"]
    )

    return cohort


def load_analysis_df(
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the frozen patient-level analysis dataframe and attach only the
    cohort timing fields required for visualization.
    """
    dataframe = pd.read_csv(
        ANALYSIS_DF_FILE,
        low_memory=False,
        dtype={"Patient_ID": "string"},
    )

    require_columns(
        dataframe,
        [
            "Patient_ID",
            "Age",
            "Sex",
            "MSK_within_1year",
        ],
        "analysis_df_1year.csv",
    )

    dataframe["Patient_ID"] = normalize_patient_id(
        dataframe["Patient_ID"]
    )

    if dataframe["Patient_ID"].duplicated().any():
        raise ValueError(
            "analysis_df_1year.csv must contain one row per patient."
        )

    timing_columns = [
        "Patient_ID",
        "first_mental_date",
        "first_msk_date",
        "time_to_msk_days",
        "feature_end_date",
    ]

    dataframe = dataframe.merge(
        cohort[
            timing_columns
        ],
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )

    return dataframe


def build_current_longitudinal_tables(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct diagnosis and medication timelines in memory using the exact
    preparation functions used by the final build_analysis_df.py.
    """
    encounter = pd.read_csv(
        CLEANED_DIAGNOSIS_FILE,
        dtype=str,
        low_memory=False,
    )

    medication = pd.read_csv(
        CLEANED_MEDICATION_FILE,
        dtype=str,
        low_memory=False,
    )

  
    # script and should print only the final trajectory summary.
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.INFO)

    try:
        diagnosis = prepare_diagnosis(
            encounter,
            cohort,
        )

        medication = prepare_medication(
            medication,
            cohort,
        )
    finally:
        logging.disable(previous_disable_level)

    diagnosis["Patient_ID"] = normalize_patient_id(
        diagnosis["Patient_ID"]
    )

    medication["Patient_ID"] = normalize_patient_id(
        medication["Patient_ID"]
    )

    # prepare_diagnosis retains first_mental_date and event_date.
    diagnosis["days_from_first_mental"] = (
        diagnosis["event_date"]
        - diagnosis["first_mental_date"]
    ).dt.days

    diagnosis["system"] = (
        diagnosis["system"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Restrict diagnosis events to the actual feature window.
    diagnosis = diagnosis.loc[
        diagnosis["event_date"].ge(
            diagnosis["first_mental_date"]
        )
        & diagnosis["event_date"].le(
            diagnosis["feature_end_date"]
        )
    ].copy()

    medication["days_from_first_mental"] = pd.to_numeric(
        medication["active_start_day"],
        errors="coerce",
    )

    medication["study_group"] = (
        medication["study_group"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    diagnosis = (
        diagnosis
        .dropna(
            subset=[
                "Patient_ID",
                "days_from_first_mental",
            ]
        )
        .loc[
            lambda x:
            x["days_from_first_mental"].between(
                0,
                FOLLOW_UP_DAYS,
                inclusive="both",
            )
        ]
        .copy()
    )

    medication = (
        medication
        .dropna(
            subset=[
                "Patient_ID",
                "days_from_first_mental",
            ]
        )
        .loc[
            lambda x:
            x["days_from_first_mental"].between(
                0,
                FOLLOW_UP_DAYS,
                inclusive="both",
            )
        ]
        .copy()
    )

    return diagnosis, medication


def load_sensitivity_groups(
    group_files: dict[str, Path],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for group_name, path in group_files.items():
        dataframe = pd.read_csv(
            path,
            low_memory=False,
            dtype={"Patient_ID": "string"},
        )

        require_columns(
            dataframe,
            [
                "Patient_ID",
                "prediction_type",
                "true_msk_1yr",
                "pred_prob",
                "prediction_error",
            ],
            path.name,
        )

        dataframe["Patient_ID"] = normalize_patient_id(
            dataframe["Patient_ID"]
        )

        dataframe["source_group"] = group_name
        frames.append(dataframe)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined["Patient_ID"].duplicated().any():
        duplicates = (
            combined.loc[
                combined["Patient_ID"].duplicated(
                    keep=False
                ),
                "Patient_ID",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "A patient appears in more than one sensitivity group: "
            + ", ".join(
                duplicates[:10]
            )
        )

    return combined


# Build patient-level trajectory-selection metrics
def patient_diagnosis_summary(
    diagnosis_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize diagnosis information observed after the index mental diagnosis.

    mental_record_count:
        number of mental diagnosis records.

    nonmental_system_count:
        number of distinct non-mental, non-MSK disease systems observed during
        follow-up.
    """
    if diagnosis_long.empty:
        return pd.DataFrame(
            columns=[
                "Patient_ID",
                "mental_record_count",
                "mental_record_days",
                "nonmental_system_count",
                "nonmental_record_count",
            ]
        )

    mental = diagnosis_long[
        diagnosis_long["system"].eq("mental")
    ]

    nonmental = diagnosis_long[
        ~diagnosis_long["system"].isin(
            ["mental", "msk"]
        )
    ]

    mental_summary = (
        mental.groupby(
            "Patient_ID",
            observed=True,
        )
        .agg(
            mental_record_count=(
                "days_from_first_mental",
                "size",
            ),
            mental_record_days=(
                "days_from_first_mental",
                "nunique",
            ),
        )
        .reset_index()
    )

    nonmental_summary = (
        nonmental.groupby(
            "Patient_ID",
            observed=True,
        )
        .agg(
            nonmental_system_count=(
                "system",
                "nunique",
            ),
            nonmental_record_count=(
                "days_from_first_mental",
                "size",
            ),
        )
        .reset_index()
    )

    return mental_summary.merge(
        nonmental_summary,
        on="Patient_ID",
        how="outer",
        validate="one_to_one",
    ).fillna(0)


def patient_medication_summary(
    medication_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize only mental- and pain-related medication information using
    distinct ATC Level 3 classes.

    Other medication groups are intentionally excluded from the patient
    trajectory figure and from representative-patient selection so that the
    displayed medication trajectory remains aligned with the study question.
    """
    output_columns = [
        "Patient_ID",
        "medication_record_count",
        "total_unique_medication_classes",
        "mental_medication_count",
        "pain_medication_count",
        "medication_start_days",
    ]

    if medication_long.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    relevant = medication_long[
        medication_long["study_group"].isin(
            ["mental", "pain"]
        )
    ].copy()

    rows: list[dict[str, object]] = []

    all_patient_ids = (
        medication_long["Patient_ID"]
        .dropna()
        .astype("string")
        .unique()
    )

    for patient_id in all_patient_ids:
        patient = relevant[
            relevant["Patient_ID"].eq(
                patient_id
            )
        ].copy()

        valid = patient.dropna(
            subset=["ATC_Level3"]
        ).copy()

        def group_class_count(
            group_name: str,
        ) -> int:
            return int(
                valid.loc[
                    valid["study_group"].eq(
                        group_name
                    ),
                    "ATC_Level3",
                ]
                .astype("string")
                .nunique()
            )

        rows.append(
            {
                "Patient_ID": patient_id,
                "medication_record_count": int(
                    len(patient)
                ),
                "total_unique_medication_classes": int(
                    valid["ATC_Level3"]
                    .astype("string")
                    .nunique()
                ),
                "mental_medication_count": (
                    group_class_count("mental")
                ),
                "pain_medication_count": (
                    group_class_count("pain")
                ),
                "medication_start_days": int(
                    patient[
                        "days_from_first_mental"
                    ].nunique()
                )
                if not patient.empty
                else 0,
            }
        )

    return pd.DataFrame(rows)


def build_selection_table(
    groups: pd.DataFrame,
    analysis_df: pd.DataFrame,
    diagnosis_long: pd.DataFrame,
    medication_long: pd.DataFrame,
) -> pd.DataFrame:
    diagnosis_summary = patient_diagnosis_summary(
        diagnosis_long
    )

    medication_summary = patient_medication_summary(
        medication_long
    )

    table = (
        groups.merge(
            analysis_df,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
            suffixes=("", "_analysis"),
        )
        .merge(
            diagnosis_summary,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )
        .merge(
            medication_summary,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )
    )

    summary_columns = [
        "mental_record_count",
        "mental_record_days",
        "nonmental_system_count",
        "nonmental_record_count",
        "medication_record_count",
        "total_unique_medication_classes",
        "mental_medication_count",
        "pain_medication_count",
        "medication_start_days",
    ]

    for column in summary_columns:
        if column not in table.columns:
            table[column] = 0

        table[column] = safe_numeric(
            table[column],
            default=0,
        )

    table["trajectory_information_score"] = (
        table["mental_record_days"]
        + table["medication_start_days"]
        + table["nonmental_system_count"]
    )

    table["medication_information_score"] = (
        table["mental_medication_count"]
        + table["pain_medication_count"]
    )

    return table


def add_trajectory_selection_metrics(
    selection_table: pd.DataFrame,
    diagnosis_long: pd.DataFrame,
    medication_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add simple trajectory-level metrics used only to choose illustrative cases.

    The selection metrics use the same three displayed domains:
      - mental diagnosis count on each exact recorded day,
      - active mental/pain ATC Level 3 medication-class count,
      - cumulative non-mental/non-MSK comorbidity-system count.

    These metrics do not enter model fitting.
    """
    result = selection_table.copy()

    diagnosis_groups = {
        str(patient_id): frame
        for patient_id, frame in diagnosis_long.groupby(
            "Patient_ID",
            observed=True,
        )
    }

    medication_groups = {
        str(patient_id): frame
        for patient_id, frame in medication_long.groupby(
            "Patient_ID",
            observed=True,
        )
    }

    trajectory_change_events: list[int] = []
    mental_change_events: list[int] = []
    medication_change_events: list[int] = []
    comorbidity_change_events: list[int] = []
    displayed_domain_count: list[int] = []

    empty_diagnosis = diagnosis_long.iloc[0:0].copy()
    empty_medication = medication_long.iloc[0:0].copy()

    for row in result.itertuples(index=False):
        patient_id = str(row.Patient_ID)

        diagnosis = diagnosis_groups.get(
            patient_id,
            empty_diagnosis,
        )

        medication = medication_groups.get(
            patient_id,
            empty_medication,
        )

        true_outcome = int(
            pd.to_numeric(
                pd.Series(
                    [
                        getattr(
                            row,
                            "true_msk_1yr",
                            getattr(
                                row,
                                "MSK_within_1year",
                                0,
                            ),
                        )
                    ]
                ),
                errors="coerce",
            )
            .fillna(0)
            .iloc[0]
        )

        msk_day = pd.to_numeric(
            pd.Series(
                [
                    getattr(
                        row,
                        "time_to_msk_days",
                        np.nan,
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if (
            true_outcome == 1
            and pd.notna(msk_day)
            and 0 <= float(msk_day) <= FOLLOW_UP_DAYS
        ):
            end_day = float(msk_day)
        else:
            end_day = FOLLOW_UP_DAYS

        mental = build_mental_trajectory(
            diagnosis
        )

        medication_trajectory = build_medication_trajectory(
            medication,
            end_day,
        )

        comorbidity = build_comorbidity_trajectory(
            diagnosis,
            end_day,
        )

        def n_changes(
            trajectory: pd.DataFrame,
        ) -> int:
            values = pd.to_numeric(
                trajectory["value"],
                errors="coerce",
            ).fillna(0).to_numpy(dtype=float)

            if len(values) <= 1:
                return 0

            return int(
                np.sum(
                    ~np.isclose(
                        values[1:],
                        values[:-1],
                    )
                )
            )

        mental_changes = n_changes(mental)
        medication_changes = n_changes(
            medication_trajectory
        )
        comorbidity_changes = n_changes(
            comorbidity
        )

        mental_change_events.append(
            mental_changes
        )
        medication_change_events.append(
            medication_changes
        )
        comorbidity_change_events.append(
            comorbidity_changes
        )
        trajectory_change_events.append(
            mental_changes
            + medication_changes
            + comorbidity_changes
        )

        displayed_domain_count.append(
            int(
                pd.to_numeric(
                    mental["value"],
                    errors="coerce",
                ).fillna(0).max()
                > 0
            )
            + int(
                pd.to_numeric(
                    medication_trajectory["value"],
                    errors="coerce",
                ).fillna(0).max()
                > 0
            )
            + int(
                pd.to_numeric(
                    comorbidity["value"],
                    errors="coerce",
                ).fillna(0).max()
                > 0
            )
        )

    result["mental_change_events"] = (
        mental_change_events
    )
    result["medication_change_events"] = (
        medication_change_events
    )
    result["comorbidity_change_events"] = (
        comorbidity_change_events
    )
    result["trajectory_change_events"] = (
        trajectory_change_events
    )
    result["displayed_domain_count"] = (
        displayed_domain_count
    )

    return result


# Deterministic representative-patient selection
def choose_first_available(
    candidates: pd.DataFrame,
    sort_columns: list[str],
    ascending: list[bool],
    used_ids: set[str],
) -> pd.Series:
    available = candidates[
        ~candidates["Patient_ID"].isin(
            used_ids
        )
    ].copy()

    if available.empty:
        raise RuntimeError(
            "No unused patient satisfies the representative-patient rule."
        )

    available = available.sort_values(
        [
            *sort_columns,
            "Patient_ID",
        ],
        ascending=[
            *ascending,
            True,
        ],
        na_position="last",
    )

    return available.iloc[0].copy()


def select_representative_patients(
    selection_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select four representative trajectories using the final clinical story.

    Good example
        High-applicability TP with rich information and multiple trajectory
        changes.

    Normal example
        Intermediate-applicability patient with information in all three
        displayed domains, but only a small number of trajectory changes.

    Bad example 1
        Low-applicability FN with no observed medication information before MSK,
        while retaining mental-health information.

    Bad example 2
        Low-applicability FN with meaningful observed information, but MSK occurs
        very soon after the first mental-health diagnosis, leaving little time
        for a longitudinal trajectory to develop.
    """
    used_ids: set[str] = set()
    rows: list[pd.Series] = []

    # GOOD: rich, dynamic, correctly predicted
    good = selection_table[
        selection_table["source_group"].eq(
            "High applicability"
        )
        & selection_table[
            "prediction_type"
        ].astype(str).eq("TP")
        & (
            safe_numeric(
                selection_table["true_msk_1yr"],
                default=0,
            )
            == 1
        )
        & (
            selection_table[
                "displayed_domain_count"
            ]
            >= 2
        )
    ].copy()

    if good.empty:
        raise RuntimeError(
            "No High-applicability TP patient is available."
        )

    chosen = choose_first_available(
        good,
        sort_columns=[
            "trajectory_change_events",
            "trajectory_information_score",
            "mental_record_days",
            "medication_information_score",
            "nonmental_system_count",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
        used_ids=used_ids,
    )

    chosen["trajectory_example"] = "Good example"
    chosen["selection_rationale"] = (
        "High-applicability true positive with rich longitudinal information "
        "and multiple changes across the displayed clinical domains."
    )
    rows.append(chosen)
    used_ids.add(str(chosen["Patient_ID"]))

    # NORMAL: all three domains present, with modest but non-trivial change
    normal = selection_table[
        selection_table["source_group"].eq(
            "Intermediate applicability"
        )
        & (
            safe_numeric(
                selection_table["true_msk_1yr"],
                default=0,
            )
            == 1
        )
        & (
            selection_table[
                "displayed_domain_count"
            ]
            == 3
        )
        & (
            selection_table[
                "mental_record_count"
            ]
            >= 4
        )
        & (
            selection_table[
                "medication_information_score"
            ]
            >= 2
        )
        & (
            selection_table[
                "nonmental_system_count"
            ]
            >= 1
        )
        & (
            selection_table[
                "trajectory_change_events"
            ]
            >= 2
        )
    ].copy()

    if normal.empty:
        # Conservative fallback: still require information in all three domains
        # and more than one observed trajectory change.
        normal = selection_table[
            selection_table["source_group"].eq(
                "Intermediate applicability"
            )
            & (
                safe_numeric(
                    selection_table["true_msk_1yr"],
                    default=0,
                )
                == 1
            )
            & (
                selection_table[
                    "displayed_domain_count"
                ]
                == 3
            )
            & (
                selection_table[
                    "trajectory_change_events"
                ]
                >= 2
            )
        ].copy()

    if normal.empty:
        raise RuntimeError(
            "No suitable Intermediate-applicability patient is available."
        )

    # Pick a typical, not extreme, intermediate trajectory.
    target_changes = float(
        normal[
            "trajectory_change_events"
        ].median()
    )
    target_information = float(
        normal[
            "trajectory_information_score"
        ].median()
    )

    normal[
        "distance_to_typical_change"
    ] = (
        normal[
            "trajectory_change_events"
        ]
        - target_changes
    ).abs()

    normal[
        "distance_to_typical_information"
    ] = (
        normal[
            "trajectory_information_score"
        ]
        - target_information
    ).abs()

    chosen = choose_first_available(
        normal,
        sort_columns=[
            "distance_to_typical_change",
            "distance_to_typical_information",
            "displayed_domain_count",
            "time_to_msk_days",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
        used_ids=used_ids,
    )

    chosen["trajectory_example"] = "Normal example"
    chosen["selection_rationale"] = (
        "Intermediate-applicability patient with mental, medication, and "
        "comorbidity information and a modest, non-trivial amount of "
        "longitudinal change before MSK."
    )
    rows.append(chosen)
    used_ids.add(str(chosen["Patient_ID"]))

    # LOW-APPLICABILITY FALSE NEGATIVES
    low_fn = selection_table[
        selection_table["source_group"].eq(
            "Low applicability"
        )
        & selection_table[
            "prediction_type"
        ].astype(str).eq("FN")
        & (
            safe_numeric(
                selection_table["true_msk_1yr"],
                default=0,
            )
            == 1
        )
    ].copy()

    if low_fn.empty:
        raise RuntimeError(
            "No Low-applicability FN patient is available."
        )

    # BAD 1: no mental/pain medication information
    bad1 = low_fn[
        (
            low_fn[
                "medication_information_score"
            ]
            == 0
        )
        & (
            low_fn[
                "mental_record_days"
            ]
            > 0
        )
    ].copy()

    # Prefer patients with some non-medication information so that the example
    # specifically illustrates the absence of medication information rather
    # than an entirely empty record.
    bad1_with_other_info = bad1[
        (
            bad1[
                "mental_record_days"
            ]
            + bad1[
                "nonmental_system_count"
            ]
        )
        >= 2
    ].copy()

    if not bad1_with_other_info.empty:
        bad1 = bad1_with_other_info

    if bad1.empty:
        # Fallback: minimal medication information rather than exactly zero.
        minimum_medication = float(
            low_fn[
                "medication_information_score"
            ].min()
        )

        bad1 = low_fn[
            low_fn[
                "medication_information_score"
            ].eq(
                minimum_medication
            )
        ].copy()

    chosen = choose_first_available(
        bad1,
        sort_columns=[
            "medication_information_score",
            "mental_record_days",
            "nonmental_system_count",
            "trajectory_information_score",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
        used_ids=used_ids,
    )

    chosen["trajectory_example"] = "Bad example 1"
    chosen["selection_rationale"] = (
        "Low-applicability false negative with no observable mental- or pain-related "
        "medication information before MSK despite available non-medication information."
    )
    rows.append(chosen)
    used_ids.add(str(chosen["Patient_ID"]))

    # BAD 2: meaningful information, but very rapid MSK
    bad2 = low_fn[
        (
            low_fn[
                "Patient_ID"
            ].astype(str)
            != str(chosen["Patient_ID"])
        )
        & (
            low_fn[
                "mental_record_days"
            ]
            > 0
        )
        & (
            low_fn[
                "medication_information_score"
            ]
            > 0
        )
        & (
            safe_numeric(
                low_fn["time_to_msk_days"],
                default=np.nan,
            )
            > 0
        )
    ].copy()

    bad2_all_three = bad2[
        bad2[
            "displayed_domain_count"
        ]
        == 3
    ].copy()

    if not bad2_all_three.empty:
        bad2 = bad2_all_three

    if bad2.empty:
        raise RuntimeError(
            "No information-containing rapid-MSK Low-applicability FN patient "
            "is available."
        )

    chosen = choose_first_available(
        bad2,
        sort_columns=[
            "time_to_msk_days",
            "displayed_domain_count",
            "trajectory_information_score",
            "trajectory_change_events",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
        used_ids=used_ids,
    )

    chosen["trajectory_example"] = "Bad example 2"
    chosen["selection_rationale"] = (
        "Low-applicability false negative with observable clinical information "
        "but very rapid progression from the first mental-health diagnosis to "
        "MSK, limiting the available longitudinal history."
    )
    rows.append(chosen)
    used_ids.add(str(chosen["Patient_ID"]))

    selected = pd.DataFrame(
        rows
    ).reset_index(drop=True)

    selected["trajectory_order"] = (
        selected["trajectory_example"]
        .map(
            {
                name: index
                for index, name
                in enumerate(
                    TRAJECTORY_TYPES,
                    start=1,
                )
            }
        )
    )

    return selected.sort_values(
        "trajectory_order"
    ).reset_index(drop=True)


# Construct event trajectories
def derive_first_msk_day(
    selected_patient: pd.Series,
) -> float:
    if (
        "time_to_msk_days"
        in selected_patient.index
        and pd.notna(
            selected_patient[
                "time_to_msk_days"
            ]
        )
    ):
        value = pd.to_numeric(
            pd.Series(
                [
                    selected_patient[
                        "time_to_msk_days"
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if pd.notna(value):
            return float(value)

    if (
        "first_mental_date"
        in selected_patient.index
        and "first_msk_date"
        in selected_patient.index
    ):
        first_mental = pd.to_datetime(
            selected_patient[
                "first_mental_date"
            ],
            errors="coerce",
        )

        first_msk = pd.to_datetime(
            selected_patient[
                "first_msk_date"
            ],
            errors="coerce",
        )

        if (
            pd.notna(first_mental)
            and pd.notna(first_msk)
        ):
            return float(
                (
                    first_msk
                    - first_mental
                ).days
            )

    return np.nan


def build_mental_trajectory(
    patient_diagnosis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mental diagnosis count on the EXACT recorded day.
    """
    mental = patient_diagnosis[
        patient_diagnosis[
            "system"
        ].eq("mental")
    ].copy()

    if mental.empty:
        return pd.DataFrame(
            {
                "day": [0.0],
                "value": [1.0],
            }
        )

    mental["day"] = pd.to_numeric(
        mental["days_from_first_mental"],
        errors="coerce",
    )

    mental = mental.dropna(
        subset=["day"]
    )

    trajectory = (
        mental.groupby(
            "day",
            as_index=False,
            observed=True,
        )
        .size()
        .rename(
            columns={
                "size": "value"
            }
        )
        .sort_values("day")
        .reset_index(drop=True)
    )

    # By cohort definition, day 0 is the index mental diagnosis.
    if not np.isclose(
        trajectory["day"].to_numpy(dtype=float),
        0.0,
    ).any():
        trajectory = pd.concat(
            [
                pd.DataFrame(
                    {
                        "day": [0.0],
                        "value": [1.0],
                    }
                ),
                trajectory,
            ],
            ignore_index=True,
        ).sort_values("day").reset_index(drop=True)
    else:
        trajectory.loc[
            np.isclose(
                trajectory["day"].to_numpy(dtype=float),
                0.0,
            ),
            "value",
        ] = trajectory.loc[
            np.isclose(
                trajectory["day"].to_numpy(dtype=float),
                0.0,
            ),
            "value",
        ].clip(lower=1)

    return trajectory


def build_medication_trajectory(
    patient_medication: pd.DataFrame,
    end_day: float,
) -> pd.DataFrame:
    """
    Number of distinct ACTIVE mental- or pain-related ATC Level 3 medication
    classes at each actual medication-change date.
    """
    end_day = float(
        np.clip(
            end_day,
            0,
            FOLLOW_UP_DAYS,
        )
    )

    working = patient_medication[
        patient_medication["study_group"].isin(
            ["mental", "pain"]
        )
    ].copy()

    if working.empty:
        return pd.DataFrame(
            {
                "day": [0.0, end_day]
                if end_day > 0
                else [0.0],
                "value": [0.0, 0.0]
                if end_day > 0
                else [0.0],
            }
        )

    working = working.dropna(
        subset=[
            "ATC_Level3",
            "active_start_day",
            "active_stop_day",
        ]
    ).copy()

    working["active_start_day"] = pd.to_numeric(
        working["active_start_day"],
        errors="coerce",
    )
    working["active_stop_day"] = pd.to_numeric(
        working["active_stop_day"],
        errors="coerce",
    )

    working = working.dropna(
        subset=[
            "active_start_day",
            "active_stop_day",
        ]
    )

    if working.empty:
        return pd.DataFrame(
            {
                "day": [0.0, end_day]
                if end_day > 0
                else [0.0],
                "value": [0.0, 0.0]
                if end_day > 0
                else [0.0],
            }
        )

    event_days = np.concatenate(
        [
            working["active_start_day"].to_numpy(
                dtype=float
            ),
            working["active_stop_day"].to_numpy(
                dtype=float
            ),
            np.array(
                [0.0, end_day],
                dtype=float,
            ),
        ]
    )

    event_days = np.unique(
        event_days[
            (event_days >= 0)
            & (event_days <= end_day)
        ]
    )

    values: list[int] = []

    for day in event_days:
        active = working[
            working["active_start_day"].le(day)
            & working["active_stop_day"].ge(day)
        ]

        values.append(
            int(
                active["ATC_Level3"]
                .astype("string")
                .nunique()
            )
        )

    return pd.DataFrame(
        {
            "day": event_days,
            "value": values,
        }
    ).sort_values(
        "day"
    ).reset_index(drop=True)


def build_comorbidity_trajectory(
    patient_diagnosis: pd.DataFrame,
    end_day: float,
) -> pd.DataFrame:
    """
    Cumulative number of distinct non-mental/non-MSK disease systems, updated
    only on actual diagnosis-event dates.
    """
    end_day = float(
        np.clip(
            end_day,
            0,
            FOLLOW_UP_DAYS,
        )
    )

    nonmental = patient_diagnosis[
        ~patient_diagnosis[
            "system"
        ].isin(
            ["mental", "msk"]
        )
    ].copy()

    nonmental["day"] = pd.to_numeric(
        nonmental["days_from_first_mental"],
        errors="coerce",
    )

    nonmental = nonmental.dropna(
        subset=["day"]
    )

    nonmental = nonmental[
        nonmental["day"].between(
            0,
            end_day,
            inclusive="both",
        )
    ].copy()

    event_days = np.unique(
        np.concatenate(
            [
                nonmental["day"].to_numpy(dtype=float)
                if not nonmental.empty
                else np.array([], dtype=float),
                np.array([0.0, end_day], dtype=float),
            ]
        )
    )

    values: list[int] = []

    for day in event_days:
        observed = nonmental[
            nonmental["day"].le(day)
        ]

        values.append(
            int(
                observed["system"]
                .dropna()
                .astype(str)
                .nunique()
            )
        )

    return pd.DataFrame(
        {
            "day": event_days,
            "value": values,
        }
    ).sort_values("day").reset_index(drop=True)


def add_msk_event_to_comorbidity(
    trajectory: pd.DataFrame,
    msk_day: float,
) -> pd.DataFrame:
    if (
        pd.isna(msk_day)
        or msk_day < 0
        or msk_day > FOLLOW_UP_DAYS
    ):
        return trajectory

    result = trajectory.copy()

    prior = result[
        result["day"] < float(msk_day)
    ]

    previous_value = (
        float(
            prior["value"].iloc[-1]
        )
        if not prior.empty
        else 0.0
    )

    result = result[
        ~np.isclose(
            result["day"],
            float(msk_day),
        )
    ].copy()

    result = pd.concat(
        [
            result,
            pd.DataFrame(
                {
                    "day": [
                        float(msk_day)
                    ],
                    "value": [
                        previous_value + 1
                    ],
                }
            ),
        ],
        ignore_index=True,
    )

    return result.sort_values(
        "day"
    ).reset_index(drop=True)


def patient_plot_data(
    selected_patient: pd.Series,
    diagnosis_long: pd.DataFrame,
    medication_long: pd.DataFrame,
) -> dict[str, pd.DataFrame | float]:
    patient_id = str(
        selected_patient["Patient_ID"]
    )

    diagnosis = diagnosis_long[
        diagnosis_long["Patient_ID"].eq(
            patient_id
        )
    ].copy()

    medication = medication_long[
        medication_long["Patient_ID"].eq(
            patient_id
        )
    ].copy()

    msk_day = derive_first_msk_day(
        selected_patient
    )

    true_outcome = int(
        pd.to_numeric(
            pd.Series(
                [
                    selected_patient.get(
                        "true_msk_1yr",
                        selected_patient.get(
                            "MSK_within_1year",
                            0,
                        ),
                    )
                ]
            ),
            errors="coerce",
        )
        .fillna(0)
        .iloc[0]
    )

    if (
        true_outcome == 1
        and pd.notna(msk_day)
        and 0 <= float(msk_day) <= FOLLOW_UP_DAYS
    ):
        end_day = float(msk_day)
    else:
        end_day = FOLLOW_UP_DAYS

    mental = build_mental_trajectory(
        diagnosis
    )

    mental = mental[
        mental["day"].le(end_day)
    ].copy()

    medications = build_medication_trajectory(
        medication,
        end_day,
    )

    comorbidity = build_comorbidity_trajectory(
        diagnosis,
        end_day,
    )

    # First MSK is displayed as the final connected comorbidity change.
    if (
        true_outcome == 1
        and pd.notna(msk_day)
        and 0 <= float(msk_day) <= FOLLOW_UP_DAYS
    ):
        comorbidity = add_msk_event_to_comorbidity(
            comorbidity,
            float(msk_day),
        )

    return {
        "mental": mental,
        "medication": medications,
        "comorbidity": comorbidity,
        "msk_day": msk_day,
    }


# Complete clinical information for selected patients
def build_selected_patient_clinical_timeline(
    selected: pd.DataFrame,
    diagnosis_long: pd.DataFrame,
    medication_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine the complete prepared diagnosis and medication rows for the table.

    Diagnosis and medication records have different schemas, so the output keeps
    the union of all columns. Non-applicable fields are blank.
    """
    selected_key = selected[
        [
            "trajectory_order",
            "trajectory_example",
            "Patient_ID",
        ]
    ].copy()

    selected_key["Patient_ID"] = normalize_patient_id(
        selected_key["Patient_ID"]
    )

    selected_ids = set(
        selected_key["Patient_ID"]
        .dropna()
        .astype(str)
    )

    diagnosis = diagnosis_long[
        diagnosis_long["Patient_ID"].isin(
            selected_ids
        )
    ].copy()

    medication = medication_long[
        medication_long["Patient_ID"].isin(
            selected_ids
        )
    ].copy()

    diagnosis = diagnosis.merge(
        selected_key,
        on="Patient_ID",
        how="left",
        validate="many_to_one",
    )
    medication = medication.merge(
        selected_key,
        on="Patient_ID",
        how="left",
        validate="many_to_one",
    )

    diagnosis["record_type"] = "diagnosis"
    medication["record_type"] = "medication"

    diagnosis["timeline_day"] = pd.to_numeric(
        diagnosis.get(
            "days_from_first_mental",
            np.nan,
        ),
        errors="coerce",
    )

    medication["timeline_day"] = pd.to_numeric(
        medication.get(
            "active_start_day",
            medication.get(
                "days_from_first_mental",
                np.nan,
            ),
        ),
        errors="coerce",
    )

    if "event_date" in diagnosis.columns:
        diagnosis["timeline_date"] = pd.to_datetime(
            diagnosis["event_date"],
            errors="coerce",
        )
    else:
        diagnosis["timeline_date"] = pd.NaT

    medication_date_column = next(
        (
            column
            for column in [
                "active_start_date",
                "med_start_date",
                "StartDate",
                "med_date",
                "DateCreated",
            ]
            if column in medication.columns
        ),
        None,
    )

    if medication_date_column is None:
        medication["timeline_date"] = pd.NaT
    else:
        medication["timeline_date"] = pd.to_datetime(
            medication[
                medication_date_column
            ],
            errors="coerce",
        )

    timeline = pd.concat(
        [
            diagnosis,
            medication,
        ],
        ignore_index=True,
        sort=False,
    )

    timeline["_record_type_order"] = (
        timeline["record_type"]
        .map(
            {
                "diagnosis": 0,
                "medication": 1,
            }
        )
        .fillna(9)
    )

    timeline = timeline.sort_values(
        [
            "trajectory_order",
            "timeline_day",
            "timeline_date",
            "_record_type_order",
        ],
        na_position="last",
    ).reset_index(drop=True)

    preferred = [
        "trajectory_order",
        "trajectory_example",
        "Patient_ID",
        "timeline_day",
        "timeline_date",
        "record_type",
    ]

    remaining = [
        column
        for column in timeline.columns
        if column not in preferred
        and column != "_record_type_order"
    ]

    return timeline[
        preferred + remaining
    ]

# Plotting
def change_mask(
    values: pd.Series,
) -> np.ndarray:
    """
    A point is a change event when its plotted value differs from the previous
    observed event point. Marker size does NOT scale with change magnitude.
    """
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    mask = np.zeros(
        len(numeric_values),
        dtype=bool,
    )

    if len(numeric_values) > 1:
        mask[1:] = ~np.isclose(
            numeric_values[1:],
            numeric_values[:-1],
        )

    return mask


def plot_step_trajectory(
    axis: plt.Axes,
    trajectory: pd.DataFrame,
    label: str,
    linestyle: str,
    marker_x_offset: float,
    color: str,
) -> None:
    if trajectory.empty:
        return

    measure_data = trajectory[
        [
            "day",
            "value",
        ]
    ].dropna().copy()

    if measure_data.empty:
        return

    x = safe_numeric(
        measure_data["day"]
    ).to_numpy(
        dtype=float
    )

    y = safe_numeric(
        measure_data["value"]
    ).to_numpy(
        dtype=float
    )

    axis.step(
        x,
        y,
        where="post",
        linewidth=2,
        linestyle=linestyle,
        color=color,
        label=label,
        zorder=2,
    )

    marker_x = np.clip(
        x + marker_x_offset,
        0,
        DISPLAY_MAX_DAYS,
    )

    changed = change_mask(
        pd.Series(y)
    )
    unchanged = ~changed

    # Ordinary snapshot: small triangle.
    axis.scatter(
        marker_x[unchanged],
        y[unchanged],
        marker="^",
        s=SMALL_TRIANGLE_SIZE,
        color=color,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    # Increase/decrease/change event: one fixed large triangle.
    axis.scatter(
        marker_x[changed],
        y[changed],
        marker="^",
        s=LARGE_TRIANGLE_SIZE,
        color=color,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )


def shared_y_max(
    patient_records: list[
        dict[str, pd.DataFrame | float]
    ],
) -> int:
    values: list[float] = []

    for record in patient_records:
        for key in [
            "mental",
            "medication",
            "comorbidity",
        ]:
            dataframe = record[key]

            if (
                isinstance(
                    dataframe,
                    pd.DataFrame,
                )
                and not dataframe.empty
            ):
                values.extend(
                    safe_numeric(
                        dataframe["value"]
                    ).tolist()
                )

    maximum = (
        max(values)
        if values
        else 1
    )

    return max(
        1,
        int(
            np.ceil(
                maximum
            )
        )
        + 1,
    )


def plot_representative_trajectories(
    selected: pd.DataFrame,
    diagnosis_long: pd.DataFrame,
    medication_long: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    selected = (
        selected.set_index(
            "trajectory_example",
            drop=False,
        )
        .reindex(
            TRAJECTORY_TYPES
        )
        .dropna(
            subset=["Patient_ID"]
        )
    )

    plot_records: list[
        tuple[
            pd.Series,
            dict[str, pd.DataFrame | float],
        ]
    ] = []

    summary_rows: list[
        dict[str, object]
    ] = []

    for _, patient in selected.iterrows():
        trajectories = patient_plot_data(
            patient,
            diagnosis_long,
            medication_long,
        )

        plot_records.append(
            (
                patient,
                trajectories,
            )
        )

        summary_rows.append(
            {
                "trajectory_example": (
                    patient[
                        "trajectory_example"
                    ]
                ),
                "Patient_ID": patient[
                    "Patient_ID"
                ],
                "source_group": patient[
                    "source_group"
                ],
                "prediction_type": patient[
                    "prediction_type"
                ],
                "true_msk_1yr": patient[
                    "true_msk_1yr"
                ],
                "pred_prob": patient[
                    "pred_prob"
                ],
                "prediction_error": patient[
                    "prediction_error"
                ],
                "Age": patient.get(
                    "Age",
                    np.nan,
                ),
                "Sex": patient.get(
                    "Sex",
                    pd.NA,
                ),
                "first_msk_day": trajectories[
                    "msk_day"
                ],
                "mental_final_count": (
                    trajectories[
                        "mental"
                    ]["value"].max()
                    if not trajectories[
                        "mental"
                    ].empty
                    else 0
                ),
                "medication_final_count": (
                    trajectories[
                        "medication"
                    ]["value"].max()
                    if not trajectories[
                        "medication"
                    ].empty
                    else 0
                ),
                "comorbidity_final_count": (
                    trajectories[
                        "comorbidity"
                    ]["value"].max()
                    if not trajectories[
                        "comorbidity"
                    ].empty
                    else 0
                ),
                "selection_rationale": patient[
                    "selection_rationale"
                ],
            }
        )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 6.5),
        sharex=True,
        sharey=True,
    )

    panels = [
        (
            "mental",
            "Mental diagnosis count",
        ),
        (
            "medication",
            "Mental/Pain medication count (ATC Level 3)",
        ),
        (
            "comorbidity",
            "Comorbidity count",
        ),
    ]

    y_max = shared_y_max(
        [
            record
            for _, record
            in plot_records
        ]
    )

    for axis, (_, title) in zip(
        axes,
        panels,
    ):
        axis.set_title(
            title,
            fontweight="bold",
        )

        axis.set_xlabel(
            "Days since first mental diagnosis"
        )

        axis.set_xlim(
            0,
            DISPLAY_MAX_DAYS,
        )

        axis.set_xticks(
            np.arange(
                0,
                DISPLAY_MAX_DAYS + 1,
                50,
            )
        )

        axis.set_ylim(
            0,
            y_max,
        )

        axis.yaxis.set_major_locator(
            MaxNLocator(
                integer=True
            )
        )

        axis.grid(
            alpha=0.25
        )

    axes[0].set_ylabel(
        "Frequency / count"
    )

    for patient, trajectories in plot_records:
        trajectory_type = str(
            patient[
                "trajectory_example"
            ]
        )

        linestyle = LINESTYLES[
            trajectory_type
        ]

        color = PATIENT_COLORS[
            trajectory_type
        ]

        marker_offset = (
            MARKER_X_OFFSETS[
                trajectory_type
            ]
        )

        for axis, (
            trajectory_key,
            _,
        ) in zip(
            axes,
            panels,
        ):
            plot_step_trajectory(
                axis=axis,
                trajectory=trajectories[
                    trajectory_key
                ],
                label=trajectory_type,
                linestyle=linestyle,
                marker_x_offset=marker_offset,
                color=color,
            )

            msk_day = trajectories[
                "msk_day"
            ]

            true_outcome = int(
                pd.to_numeric(
                    pd.Series(
                        [
                            patient.get(
                                "true_msk_1yr",
                                0,
                            )
                        ]
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .iloc[0]
            )

            if (
                true_outcome == 1
                and pd.notna(msk_day)
                and 0
                <= float(msk_day)
                <= FOLLOW_UP_DAYS
            ):
                axis.axvline(
                    float(msk_day),
                    linestyle=linestyle,
                    linewidth=1.6,
                    color=color,
                    alpha=0.80,
                    zorder=1,
                )

    figure.suptitle(
        "Representative 1-Year Patient Trajectories",
        fontsize=16,
        fontweight="bold",
    )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    unique_legend: dict[
        str,
        object,
    ] = {}

    for handle, label in zip(
        handles,
        labels,
    ):
        if label not in unique_legend:
            unique_legend[label] = handle

    if unique_legend:
        figure.legend(
            unique_legend.values(),
            unique_legend.keys(),
            loc="lower center",
            bbox_to_anchor=(
                0.5,
                0.045,
            ),
            ncol=2,
            frameon=False,
        )

    figure.text(
        0.5,
        0.012,
        (
            "Small triangles indicate unchanged consecutive event values; large triangles "
            "indicate a frequency/count change from the previous event. "
            "Colored vertical lines indicate first MSK diagnosis."
        ),
        ha="center",
        fontsize=9.5,
    )

    plt.tight_layout(
        rect=[
            0,
            0.16,
            1,
            0.92,
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return pd.DataFrame(
        summary_rows
    )


# Main
def main() -> None:
    create_project_directories()

    OUTPUT_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_files = validate_inputs()

    cohort = load_current_cohort()

    analysis_df = load_analysis_df(
        cohort
    )

    (
        diagnosis_long,
        medication_long,
    ) = build_current_longitudinal_tables(
        cohort
    )

    sensitivity_groups = (
        load_sensitivity_groups(
            group_files
        )
    )

    selection_table = (
        build_selection_table(
            groups=sensitivity_groups,
            analysis_df=analysis_df,
            diagnosis_long=diagnosis_long,
            medication_long=medication_long,
        )
    )

    selection_table = add_trajectory_selection_metrics(
        selection_table=selection_table,
        diagnosis_long=diagnosis_long,
        medication_long=medication_long,
    )

    selected = (
        select_representative_patients(
            selection_table
        )
    )

    selected_output_columns = [
        "trajectory_order",
        "trajectory_example",
        "Patient_ID",
        "source_group",
        "prediction_type",
        "true_msk_1yr",
        "pred_prob",
        "prediction_error",
        "Age",
        "Sex",
        "mental_record_count",
        "mental_record_days",
        "nonmental_system_count",
        "medication_record_count",
        "total_unique_medication_classes",
        "mental_medication_count",
        "pain_medication_count",
        "medication_start_days",
        "trajectory_information_score",
        "displayed_domain_count",
        "mental_change_events",
        "medication_change_events",
        "comorbidity_change_events",
        "trajectory_change_events",
        "time_to_msk_days",
        "selection_rationale",
    ]

    selected_output_columns = [
        column
        for column in selected_output_columns
        if column in selected.columns
    ]

    selected[
        selected_output_columns
    ].to_csv(
        OUTPUT_TABLE_DIR
        / "selected_patients.csv",
        index=False,
    )

    clinical_timeline = (
        build_selected_patient_clinical_timeline(
            selected=selected,
            diagnosis_long=diagnosis_long,
            medication_long=medication_long,
        )
    )

    clinical_timeline.to_csv(
        OUTPUT_TABLE_DIR
        / "selected_patient_clinical_timeline.csv",
        index=False,
    )

    plot_representative_trajectories(
        selected=selected,
        diagnosis_long=diagnosis_long,
        medication_long=medication_long,
        output_path=OUTPUT_FIGURE_FILE,
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "MENTAL → MSK PATIENT TRAJECTORY SUMMARY"
    )
    print(
        "=" * 72
    )

    for row in selected[
        [
            "trajectory_example",
            "Patient_ID",
        ]
    ].itertuples(
        index=False
    ):
        print(
            f"{row.trajectory_example:<30}: "
            f"Patient {row.Patient_ID}"
        )

    print(
        f"Selected patients    : "
        f"selected_patients.csv"
    )
    print(
        f"Clinical timeline    : "
        f"selected_patient_clinical_timeline.csv"
    )
    print(
        f"Figure saved         : "
        f"{OUTPUT_FIGURE_FILE.name}"
    )
    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
