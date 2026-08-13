"""
Build the final one-year patient-level analysis dataframe.

1. The feature window begins at the first recorded mental diagnosis.
2. The feature window ends at the earlier of:
       - first MSK diagnosis; or
       - 365 days after the first mental diagnosis.
3. ATC Level 3 defines medication classes.
4. Complete ATC Level 7 identifies distinct medications.
"""

from __future__ import annotations

from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mental_msk_prediction.config.paths import (
    CLEANED_DIAGNOSIS_FILE,
    CLEANED_MEDICATION_FILE,
    RESULTS_DATA_DIR,
    TABLES_DIR,
    create_project_directories,
)

from mental_msk_prediction.config.clinical_definitions import (
    ATC_LEVEL3_DEFINITIONS,
    COMORBIDITY_SYSTEMS_MODEL,
    MENTAL_ATC_LEVEL3,
    PAIN_ATC_LEVEL3,
    classify_category,
    classify_system_broad,
    extract_icd9_3,
    get_atc_study_group,
    normalize_atc_level3,
)


def locate_final_cohort_file() -> Path:
    cohort_file = (
        RESULTS_DATA_DIR
        / "cohort"
        / "final_cohort.csv"
    )

    if not cohort_file.is_file():
        raise FileNotFoundError(
            "Final cohort CSV was not found:\n"
            f"  - {cohort_file}\n\n"
            "Run cohort construction first:\n"
            "python -m mental_msk_prediction.cohort.build_cohort"
        )

    return cohort_file


COHORT_CSV = locate_final_cohort_file()
ENCOUNTER_CSV = CLEANED_DIAGNOSIS_FILE
MEDICATION_CSV = CLEANED_MEDICATION_FILE

OUTPUT_DIR = (RESULTS_DATA_DIR / "analysis")
ANALYSIS_TABLE_DIR = (TABLES_DIR/ "analysis_feature")
OUTPUT_FILE = (OUTPUT_DIR/ "analysis_df_1year.csv")
FEATURE_SUMMARY_FILE = (ANALYSIS_TABLE_DIR/ "feature_summary.csv")
MEDICATION_TIMELINE_FILE = (ANALYSIS_TABLE_DIR/ "medication_timeline.csv")
ANALYSIS_MISSINGNESS_FILE = (ANALYSIS_TABLE_DIR/ "analysis_df_missingness.csv")
DYNAMIC_MEDICATION_FILE = (ANALYSIS_TABLE_DIR/ "dynamic_medication_feature.csv")



# Logging and constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(
    __name__
)

FOLLOW_UP_DAYS = 365

MAX_SWITCH_GAP_DAYS = 14
MAX_SWITCH_OVERLAP_DAYS = 7
NO_MEDICATION_TIME_SENTINEL = FOLLOW_UP_DAYS + 1

SELECTED_DISEASE_SYSTEMS = list(COMORBIDITY_SYSTEMS_MODEL)
# Final medication feature design.
# Static exposure (Any_) is retained for all 12 prespecified ATC Level 3
# Discontinuation and switching are retained only for the eight sufficiently
# represented classes selected for longitudinal trajectory modeling.
ANY_ATC_CODES = ["N05A", "N05B", "N05C","N06A","N06B", "N06C","M01A", "M02A", "M03B",
    "N02A", "N02B", "N02C",]

TRAJECTORY_ATC_CODES = ["N05A", "N05B", "N05C", "N06A",
    "M01A","M03B", "N02A","N02B",]

ATC_FEATURE_SUFFIX = {
    "N05A": "Antipsychotics",
    "N05B": "Anxiolytics",
    "N05C": "Hypnotics_and_sedatives",
    "N06A": "Antidepressants",
    "N06B": "Psychostimulants_and_ADHD_medications",
    "N06C": "Psycholeptic_and_psychoanaleptic_combinations",
    "M01A": "Anti_inflammatory_and_antirheumatic_products",
    "M02A": "Topical_products_for_joint_and_muscular_pain",
    "M03B": "Muscle_relaxants",
    "N02A": "Opioids",
    "N02B": "Other_analgesics_and_antipyretics",
    "N02C": "Antimigraine_preparations",
}

MENTAL_ATC_CODES = set( MENTAL_ATC_LEVEL3)
PAIN_ATC_CODES = set(PAIN_ATC_LEVEL3)

def normalize_text(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize a text column while preserving missing values.
    """
    output = (
        series.astype("string")
        .str.strip()
    )

    output = output.mask(
        output.str.lower().isin(
            {
                "",
                "nan",
                "none",
                "<na>",
            }
        )
    )

    return output


def first_available_column(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
    required: bool = True,
) -> str | None:
    """
    Return the first available column from a candidate list.
    """
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise KeyError(
            f"Could not find {label}. "
            f"Expected one of: {candidates}"
        )

    return None


def official_name_to_suffix(
    official_name: str,
) -> str:
    """
    Convert a readable ATC name to a stable feature suffix.
    """
    return (
        official_name
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(" ", "_")
    )


def normalize_atc_code(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize an ATC code to uppercase alphanumeric text.
    """
    return (
        normalize_text(series)
        .str.upper()
        .str.replace(
            r"[^A-Z0-9]",
            "",
            regex=True,
        )
    )


def merged_intervals(
    intervals: list[
        tuple[pd.Timestamp, pd.Timestamp]
    ],
) -> list[
    tuple[pd.Timestamp, pd.Timestamp]
]:
    """
    Merge overlapping or adjacent inclusive calendar-date intervals.
    """
    cleaned = sorted(
        (
            pd.Timestamp(start).normalize(),
            pd.Timestamp(stop).normalize(),
        )
        for start, stop in intervals
        if pd.notna(start)
        and pd.notna(stop)
        and pd.Timestamp(stop) >= pd.Timestamp(start)
    )

    if not cleaned:
        return []

    merged: list[
        list[pd.Timestamp]
    ] = [
        [
            cleaned[0][0],
            cleaned[0][1],
        ]
    ]

    one_day = pd.Timedelta(
        days=1
    )

    for start, stop in cleaned[1:]:
        if start <= (
            merged[-1][1]
            + one_day
        ):
            merged[-1][1] = max(
                merged[-1][1],
                stop,
            )
        else:
            merged.append(
                [
                    start,
                    stop,
                ]
            )

    return [
        (
            start,
            stop,
        )
        for start, stop in merged
    ]


def interval_days(
    intervals: list[
        tuple[pd.Timestamp, pd.Timestamp]
    ],
) -> set[pd.Timestamp]:
    """
    Expand inclusive date intervals to a set of active calendar dates.
    """
    active_days: set[
        pd.Timestamp
    ] = set()

    for start, stop in intervals:
        active_days.update(
            pd.date_range(
                start=start,
                end=stop,
                freq="D",
            )
        )

    return active_days


# Cohort
def prepare_cohort(
    cohort_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare one row per final-cohort patient.
    """
    cohort = cohort_raw.copy()

    patient_column = first_available_column(
        cohort,
        [
            "Patient_ID",
            "patient_id",
        ],
        "Patient_ID",
    )

    sex_column = first_available_column(
        cohort,
        [
            "Sex",
            "sex",
        ],
        "sex",
    )

    age_column = first_available_column(
        cohort,
        [
            "Age_at_first_mental",
            "Age_at_mental",
            "Age",
        ],
        "age at first mental diagnosis",
    )

    first_mental_column = first_available_column(
        cohort,
        [
            "first_mental_date",
            "first_mental",
        ],
        "first mental diagnosis date",
    )

    first_msk_column = first_available_column(
        cohort,
        [
            "first_msk_date",
            "first_msk",
        ],
        "first MSK diagnosis date",
    )

    cohort = cohort.rename(
        columns={
            patient_column: "Patient_ID",
            sex_column: "Sex",
            age_column: "Age",
            first_mental_column: "first_mental_date",
            first_msk_column: "first_msk_date",
        }
    )

    cohort["Patient_ID"] = normalize_text(
        cohort["Patient_ID"]
    )

    cohort["Sex"] = normalize_text(
        cohort["Sex"]
    )

    cohort["Age"] = pd.to_numeric(
        cohort["Age"],
        errors="coerce",
    )

    cohort["first_mental_date"] = pd.to_datetime(
        cohort["first_mental_date"],
        errors="coerce",
    )

    cohort["first_msk_date"] = pd.to_datetime(
        cohort["first_msk_date"],
        errors="coerce",
    )

    cohort["time_to_msk_days"] = (
        cohort["first_msk_date"]
        - cohort["first_mental_date"]
    ).dt.days

    cohort["MSK_within_1year"] = (
        cohort[
            "time_to_msk_days"
        ]
        .between(
            1,
            FOLLOW_UP_DAYS,
            inclusive="both",
        )
        .fillna(False)
        .astype(int)
    )

    cohort["one_year_end_date"] = (
        cohort["first_mental_date"]
        + pd.to_timedelta(
            FOLLOW_UP_DAYS,
            unit="D",
        )
    )

    cohort["feature_end_date"] = (
        cohort["one_year_end_date"]
    )

    positive_mask = cohort[
        "MSK_within_1year"
    ].eq(1)

    cohort.loc[
        positive_mask,
        "feature_end_date",
    ] = cohort.loc[
        positive_mask,
        "first_msk_date",
    ]

    required = [
        "Patient_ID",
        "Sex",
        "Age",
        "first_mental_date",
        "first_msk_date",
        "feature_end_date",
    ]

    missing_counts = (
        cohort[
            required
        ]
        .isna()
        .sum()
    )

    if missing_counts.gt(0).any():
        raise ValueError(
            "Final cohort contains missing required fields:\n"
            + missing_counts[
                missing_counts.gt(0)
            ].to_string()
        )

    if cohort[
        "Patient_ID"
    ].duplicated().any():
        raise ValueError(
            "Final cohort contains duplicated Patient_ID values."
        )

    if not (
        cohort["first_mental_date"]
        < cohort["first_msk_date"]
    ).all():
        raise ValueError(
            "All final-cohort patients must have first mental before first MSK."
        )

    return (
        cohort[
            [
                "Patient_ID",
                "Age",
                "Sex",
                "first_mental_date",
                "first_msk_date",
                "time_to_msk_days",
                "MSK_within_1year",
                "one_year_end_date",
                "feature_end_date",
            ]
        ]
        .sort_values(
            "Patient_ID"
        )
        .reset_index(
            drop=True
        )
    )


# Diagnosis data and mental-health features
def prepare_diagnosis(
    encounter_raw: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare cleaned diagnosis records using the same schema and clinical
    definitions as cohort/build_cohort.py.

    Required cleaned columns
    ------------------------
    Patient_ID
    DateCreated
    DiagnosisCodeType_calc
    DiagnosisCode_calc

    Only calculated ICD-9 records are retained.
    """
    required_columns = [
        "Patient_ID",
        "DateCreated",
        "DiagnosisCodeType_calc",
        "DiagnosisCode_calc",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in encounter_raw.columns
    ]

    if missing_columns:
        raise KeyError(
            "Cleaned diagnosis table is missing required columns: "
            + ", ".join(
                missing_columns
            )
            + "\nAvailable columns include:\n  - "
            + "\n  - ".join(
                map(
                    str,
                    encounter_raw.columns.tolist(),
                )
            )
        )

    encounter = encounter_raw.copy()

    encounter["Patient_ID"] = normalize_text(
        encounter["Patient_ID"]
    )

    encounter[
        "DiagnosisCodeType_calc"
    ] = (
        normalize_text(
            encounter[
                "DiagnosisCodeType_calc"
            ]
        )
        .str.upper()
    )

    # Match build_cohort.py exactly: use calculated ICD-9 records only.
    encounter = encounter[
        encounter[
            "DiagnosisCodeType_calc"
        ].eq(
            "ICD9"
        )
    ].copy()

    encounter["event_date"] = pd.to_datetime(
        encounter["DateCreated"],
        errors="coerce",
    )

    encounter["icd9"] = normalize_text(
        encounter[
            "DiagnosisCode_calc"
        ]
    )

    if "Encounter_ID" in encounter.columns:
        encounter[
            "Encounter_ID"
        ] = normalize_text(
            encounter[
                "Encounter_ID"
            ]
        )
    else:
        encounter[
            "Encounter_ID"
        ] = pd.Series(
            pd.NA,
            index=encounter.index,
            dtype="string",
        )

    encounter["icd9_3"] = (
        encounter["icd9"]
        .apply(
            extract_icd9_3
        )
        .astype(
            "Int64"
        )
    )

    encounter["system"] = (
        encounter["icd9"]
        .apply(
            classify_system_broad
        )
        .astype(
            "string"
        )
        .str.lower()
    )

    encounter["category"] = (
        encounter["icd9"]
        .apply(
            classify_category
        )
        .astype(
            "string"
        )
    )

    encounter = encounter.dropna(
        subset=[
            "Patient_ID",
            "event_date",
            "icd9",
            "icd9_3",
            "system",
        ]
    ).copy()

    encounter = encounter.merge(
        cohort[
            [
                "Patient_ID",
                "first_mental_date",
                "feature_end_date",
            ]
        ],
        on="Patient_ID",
        how="inner",
        validate="many_to_one",
    )

    keep_columns = [
        column
        for column in [
            "EncounterDiagnosis_ID",
            "Encounter_ID",
            "Patient_ID",
            "event_date",
            "icd9",
            "icd9_3",
            "system",
            "category",
            "DiagnosisText_calc",
            "DiagnosisText_orig",
            "first_mental_date",
            "feature_end_date",
        ]
        if column in encounter.columns
    ]

    return (
        encounter[
            keep_columns
        ]
        .sort_values(
            [
                "Patient_ID",
                "event_date",
                "icd9_3",
                "icd9",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def make_visit_key(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Build a visit identifier.

    Encounter_ID is preferred. A patient-specific calendar date is the
    fallback when Encounter_ID is missing.
    """
    encounter_key = normalize_text(
        df["Encounter_ID"]
    )

    date_key = (
        "DATE_"
        + df["event_date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .astype("string")
    )

    return encounter_key.where(
        encounter_key.notna(),
        date_key,
    )



def build_mental_features(
    diagnosis: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the four final mental-health predictors.
    """
    mental = diagnosis[
        diagnosis["system"].eq("mental")
        & diagnosis["event_date"].ge(
            diagnosis["first_mental_date"]
        )
        & diagnosis["event_date"].le(
            diagnosis["feature_end_date"]
        )
    ].copy()

    output = cohort[
        [
            "Patient_ID",
            "feature_end_date",
        ]
    ].copy()

    if mental.empty:
        output["Mental_diagnosis_count"] = 0
        output["Recent_30d_mental_visit"] = 0
        output["Repeated_same_mental_diagnosis"] = 0
        output["Unique_mental_diagnosis_count"] = 0

        return output.drop(
            columns=["feature_end_date"]
        )

    mental["mental_visit_key"] = make_visit_key(
        mental
    )

    diagnosis_count = (
        mental.groupby(
            "Patient_ID",
            observed=True,
        )
        .size()
        .rename(
            "Mental_diagnosis_count"
        )
        .reset_index()
    )

    diagnosis_visit_count = (
        mental[
            [
                "Patient_ID",
                "icd9",
                "mental_visit_key",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "Patient_ID",
                "icd9",
            ],
            observed=True,
        )
        .size()
        .rename(
            "n_visits_same_mental_diagnosis"
        )
        .reset_index()
    )

    repeated_same = (
        diagnosis_visit_count.groupby(
            "Patient_ID",
            observed=True,
        )[
            "n_visits_same_mental_diagnosis"
        ]
        .max()
        .ge(2)
        .astype(int)
        .rename(
            "Repeated_same_mental_diagnosis"
        )
        .reset_index()
    )

    unique_diagnosis = (
        mental.groupby(
            "Patient_ID",
            observed=True,
        )[
            "icd9"
        ]
        .nunique()
        .rename(
            "Unique_mental_diagnosis_count"
        )
        .reset_index()
    )

    recent = mental.merge(
        cohort[
            [
                "Patient_ID",
                "feature_end_date",
            ]
        ],
        on="Patient_ID",
        how="left",
        suffixes=(
            "",
            "_cohort",
        ),
        validate="many_to_one",
    )

    if "feature_end_date_cohort" in recent.columns:
        recent["feature_end_date"] = recent[
            "feature_end_date_cohort"
        ]

    recent_start = (
        recent["feature_end_date"]
        - pd.to_timedelta(
            29,
            unit="D",
        )
    )

    recent_ids = (
        recent.loc[
            recent["event_date"].between(
                recent_start,
                recent["feature_end_date"],
                inclusive="both",
            ),
            "Patient_ID",
        ]
        .drop_duplicates()
    )

    recent_flag = pd.DataFrame(
        {
            "Patient_ID": recent_ids,
            "Recent_30d_mental_visit": 1,
        }
    )

    for table in [
        diagnosis_count,
        repeated_same,
        unique_diagnosis,
        recent_flag,
    ]:
        output = output.merge(
            table,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )

    mental_columns = [
        "Mental_diagnosis_count",
        "Recent_30d_mental_visit",
        "Repeated_same_mental_diagnosis",
        "Unique_mental_diagnosis_count",
    ]

    output[mental_columns] = (
        output[mental_columns]
        .fillna(0)
        .astype(int)
    )

    return output[
        [
            "Patient_ID",
            *mental_columns,
        ]
    ]



def build_disease_burden_features(
    diagnosis: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the two final disease-burden predictors.
    """
    selected = diagnosis[
        diagnosis["system"].isin(
            SELECTED_DISEASE_SYSTEMS
        )
    ].copy()

    baseline = selected[
        selected["event_date"].lt(
            selected["first_mental_date"]
        )
    ]

    follow_up = selected[
        selected["event_date"].ge(
            selected["first_mental_date"]
        )
        & selected["event_date"].le(
            selected["feature_end_date"]
        )
    ]

    baseline_count = (
        baseline.groupby(
            "Patient_ID",
            observed=True,
        )[
            "system"
        ]
        .nunique()
        .rename(
            "Preexisting_condition_count"
        )
        .reset_index()
    )

    follow_up_count = (
        follow_up.groupby(
            "Patient_ID",
            observed=True,
        )[
            "system"
        ]
        .nunique()
        .rename(
            "Multimorbidity_count"
        )
        .reset_index()
    )

    output = cohort[
        [
            "Patient_ID",
        ]
    ].copy()

    output = (
        output
        .merge(
            baseline_count,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )
        .merge(
            follow_up_count,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )
    )

    disease_columns = [
        "Preexisting_condition_count",
        "Multimorbidity_count",
    ]

    output[disease_columns] = (
        output[disease_columns]
        .fillna(0)
        .astype(int)
    )

    return output


def prepare_medication(
    medication_raw: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the cleaned CPCSSN medication table for ATC-based features.

    Input schema follows preprocessing/clean_data.py and
    preprocessing/prepare_atc_definition.py exactly:

    - Patient_ID identifies the patient;
    - StartDate and StopDate define the medication interval;
    - CodeType_calc must equal ``ATC``;
    - Code_calc is the calculated ATC source.

    Rules
    -----
    1. Missing StartDate rows are excluded.
    2. StopDate earlier than StartDate is excluded.
    3. Missing StopDate is treated as active through the patient's feature end.
    4. Medication rows are restricted to starts from first mental diagnosis
       through the patient-specific one-year feature end, matching the original
       medication-long inclusion rule.
    5. ATC Level 3 defines medication class; a complete seven-character ATC
       code identifies a distinct medication.
    """
    required_columns = [
        "Patient_ID",
        "StartDate",
        "StopDate",
        "CodeType_calc",
        "Code_calc",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in medication_raw.columns
    ]

    if missing_columns:
        raise KeyError(
            "Cleaned medication table is missing required columns: "
            + ", ".join(missing_columns)
            + "\nAvailable columns include:\n  - "
            + "\n  - ".join(map(str, medication_raw.columns.tolist()))
        )

    medication = medication_raw.copy()
    before_rows = len(medication)

    medication["Patient_ID"] = normalize_text(
        medication["Patient_ID"]
    )

    medication["CodeType_calc_normalized"] = (
        normalize_text(medication["CodeType_calc"])
        .str.upper()
    )

    medication["med_start_date"] = pd.to_datetime(
        medication["StartDate"],
        errors="coerce",
    )

    medication["med_stop_date"] = pd.to_datetime(
        medication["StopDate"],
        errors="coerce",
    )

    if "DateCreated" in medication.columns:
        medication["med_created_date"] = pd.to_datetime(
            medication["DateCreated"],
            errors="coerce",
        )
    else:
        medication["med_created_date"] = pd.NaT

    medication["ATC_code"] = normalize_atc_code(
        medication["Code_calc"]
    )

    # prepare_atc_definition.py uses only calculated codes explicitly marked ATC.
    is_calculated_atc = medication[
        "CodeType_calc_normalized"
    ].eq("ATC")

    medication.loc[
        ~is_calculated_atc,
        "ATC_code",
    ] = pd.NA

    medication["ATC_Level3"] = (
        medication["ATC_code"]
        .apply(normalize_atc_level3)
        .astype("string")
    )

    complete_level7 = medication[
        "ATC_code"
    ].str.fullmatch(
        r"[A-Z]\d{2}[A-Z]{2}\d{2}",
        na=False,
    )

    medication["ATC_Level7"] = medication[
        "ATC_code"
    ].where(complete_level7)

    missing_start = medication[
        "med_start_date"
    ].isna()

    invalid_stop = (
        medication["med_start_date"].notna()
        & medication["med_stop_date"].notna()
        & medication["med_stop_date"].lt(
            medication["med_start_date"]
        )
    )

    medication = medication.loc[
        ~missing_start
        & ~invalid_stop
    ].copy()

    medication = medication.merge(
        cohort[
            [
                "Patient_ID",
                "first_mental_date",
                "feature_end_date",
            ]
        ],
        on="Patient_ID",
        how="inner",
        validate="many_to_one",
    )

    # Original medication-long rule: medication must start within the window.
    medication = medication.loc[
        medication["med_start_date"].ge(
            medication["first_mental_date"]
        )
        & medication["med_start_date"].le(
            medication["feature_end_date"]
        )
    ].copy()

    medication["effective_stop_date"] = (
        medication["med_stop_date"]
        .fillna(medication["feature_end_date"])
        .clip(upper=medication["feature_end_date"])
    )

    medication["active_start_day"] = (
        medication["med_start_date"]
        - medication["first_mental_date"]
    ).dt.days.astype("Int64")

    medication["active_stop_day"] = (
        medication["effective_stop_date"]
        - medication["first_mental_date"]
    ).dt.days.astype("Int64")

    medication["study_group"] = (
        medication["ATC_Level3"]
        .apply(get_atc_study_group)
        .astype("string")
        .fillna("other")
    )

    medication_timeline = pd.DataFrame(
        [
            {
                "metric": "Original medication rows",
                "value": before_rows,
            },
            {
                "metric": "Rows with CodeType_calc equal to ATC",
                "value": int(is_calculated_atc.sum()),
            },
            {
                "metric": "Missing StartDate removed",
                "value": int(missing_start.sum()),
            },
            {
                "metric": "StopDate earlier than StartDate removed",
                "value": int(invalid_stop.sum()),
            },
            {
                "metric": "Valid medication rows in cohort feature windows",
                "value": len(medication),
            },
            {
                "metric": "Missing StopDate treated as active",
                "value": int(medication["med_stop_date"].isna().sum()),
            },
            {
                "metric": "Patients with retained medication records",
                "value": int(medication["Patient_ID"].nunique()),
            },
            {
                "metric": "Retained rows with valid ATC Level 3",
                "value": int(medication["ATC_Level3"].notna().sum()),
            },
            {
                "metric": "Retained rows with complete ATC Level 7",
                "value": int(medication["ATC_Level7"].notna().sum()),
            },
        ]
    )

    ANALYSIS_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    medication_timeline.to_csv(
        MEDICATION_TIMELINE_FILE,
        index=False,
    )

    log.info("Medication timeline:")
    for record in medication_timeline.itertuples(index=False):
        log.info("  %-48s %s", record.metric, f"{int(record.value):,}")

    log.info("Medication rows retained: %s", f"{len(medication):,}")

    keep_columns = [
        column
        for column in [
            "Medication_ID",
            "Encounter_ID",
            "Patient_ID",
            "Name_calc",
            "StartDate",
            "StopDate",
            "DateCreated",
            "med_start_date",
            "med_stop_date",
            "med_created_date",
            "effective_stop_date",
            "active_start_day",
            "active_stop_day",
            "CodeType_calc_normalized",
            "ATC_code",
            "ATC_Level3",
            "ATC_Level7",
            "study_group",
            "first_mental_date",
            "feature_end_date",
        ]
        if column in medication.columns
    ]

    return (
        medication[keep_columns]
        .sort_values(
            [
                "Patient_ID",
                "med_start_date",
                "ATC_Level3",
                "ATC_Level7",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def active_medications_at_day(
    patient_medication: pd.DataFrame,
    day: int,
) -> pd.DataFrame:
    """
    Return medication records active at a cumulative day.
    """
    return patient_medication[
        patient_medication[
            "active_start_day"
        ].le(day)
        & patient_medication[
            "active_stop_day"
        ].ge(day)
    ].copy()





def build_patient_medication_features(
    medication: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build only the final medication predictors.

    Medication burden uses distinct complete ATC Level 7 medications observed
    anywhere in the patient-specific feature window.

    Static exposure uses 12 ATC Level 3 Any_* indicators, including N06C.

    Longitudinal trajectory uses discontinuation and switching counts for
    eight selected ATC Level 3 classes.

    A medication switch is defined within the same ATC Level 3 class and
    requires a different complete ATC Level 7 medication. The subsequent
    medication may start up to 14 days after the previous medication stops,
    or may overlap the previous medication by no more than 7 calendar days.

    Timing and overlap retain only:
      * Time_to_first_mental_medication
      * Time_to_first_pain_medication
      * Mental_pain_overlap_ratio
    """
    patient_groups = {
        patient_id: group.copy()
        for patient_id, group in medication.groupby(
            "Patient_ID",
            observed=True,
            sort=False,
        )
    }

    empty_medication = medication.iloc[0:0].copy()
    rows: list[dict[str, object]] = []

    for cohort_row in cohort.itertuples(index=False):
        patient_id = cohort_row.Patient_ID

        index_date = pd.Timestamp(
            cohort_row.first_mental_date
        ).normalize()

        feature_end_date = pd.Timestamp(
            cohort_row.feature_end_date
        ).normalize()

        patient_medication = patient_groups.get(
            patient_id,
            empty_medication,
        ).copy()

        patient_output: dict[str, object] = {
            "Patient_ID": patient_id,
        }

        # Distinct ATC Level 7 medication burden across the feature window.
        valid_atc7 = patient_medication[
            patient_medication["ATC_Level7"].notna()
        ].copy()

        patient_output["Mental_medication_count"] = int(
            valid_atc7.loc[
                valid_atc7["study_group"].eq("mental"),
                "ATC_Level7",
            ].nunique()
        )

        patient_output["Pain_medication_count"] = int(
            valid_atc7.loc[
                valid_atc7["study_group"].eq("pain"),
                "ATC_Level7",
            ].nunique()
        )

        patient_output["Other_medication_count"] = int(
            valid_atc7.loc[
                valid_atc7["study_group"].eq("other"),
                "ATC_Level7",
            ].nunique()
        )

        # Number of non-mental/non-pain ATC Level 3 medication classes
        valid_atc3 = patient_medication[
            patient_medication["ATC_Level3"].notna()
        ].copy()

        patient_output["Other_medication_class_count"] = int(
            valid_atc3.loc[
                valid_atc3["study_group"].eq("other"),
                "ATC_Level3",
            ].nunique()
        )

        # Any exposure to each selected ATC Level 3 group during the window.
        for code in ANY_ATC_CODES:
            suffix = ATC_FEATURE_SUFFIX[code]

            patient_output[
                f"Any_{suffix}"
            ] = int(
                patient_medication[
                    "ATC_Level3"
                ].eq(code).any()
            )

        def time_to_first(
            records: pd.DataFrame,
        ) -> int:
            if records.empty:
                return NO_MEDICATION_TIME_SENTINEL

            first_date = records[
                "med_start_date"
            ].min()

            if pd.isna(first_date):
                return NO_MEDICATION_TIME_SENTINEL

            return int(
                (
                    pd.Timestamp(first_date).normalize()
                    - index_date
                ).days
            )

        patient_output[
            "Time_to_first_mental_medication"
        ] = time_to_first(
            patient_medication[
                patient_medication["study_group"].eq("mental")
            ]
        )

        patient_output[
            "Time_to_first_pain_medication"
        ] = time_to_first(
            patient_medication[
                patient_medication["study_group"].eq("pain")
            ]
        )

        # Selected class-specific discontinuation and switching.
        for code in TRAJECTORY_ATC_CODES:
            suffix = ATC_FEATURE_SUFFIX[code]

            class_records = patient_medication[
                patient_medication["ATC_Level3"].eq(code)
            ].copy()

            valid_class_atc7 = class_records[
                class_records["ATC_Level7"].notna()
            ].copy()

            discontinuation_events = 0
            drug_intervals: dict[
                str,
                list[
                    tuple[
                        pd.Timestamp,
                        pd.Timestamp,
                    ]
                ],
            ] = {}

            for atc7, drug_records in valid_class_atc7.groupby(
                "ATC_Level7",
                observed=True,
                sort=False,
            ):
                intervals = merged_intervals(
                    [
                        (
                            max(
                                pd.Timestamp(start).normalize(),
                                index_date,
                            ),
                            min(
                                pd.Timestamp(stop).normalize(),
                                feature_end_date,
                            ),
                        )
                        for start, stop in zip(
                            drug_records["med_start_date"],
                            drug_records["effective_stop_date"],
                        )
                        if pd.notna(start)
                        and pd.notna(stop)
                    ]
                )

                intervals = [
                    interval
                    for interval in intervals
                    if interval[1] >= interval[0]
                ]

                drug_intervals[str(atc7)] = intervals

                for _, interval_stop in intervals:
                    real_stop_match = drug_records[
                        drug_records["med_stop_date"].notna()
                        & drug_records["med_stop_date"]
                        .dt.normalize()
                        .eq(interval_stop)
                    ]

                    if (
                        index_date
                        <= interval_stop
                        <= feature_end_date
                        and not real_stop_match.empty
                    ):
                        discontinuation_events += 1

            patient_output[
                f"{suffix}_discontinuation_count"
            ] = int(
                discontinuation_events
            )

            # Switching is assessed only within this ATC Level 3 class.
            # Different ATC Level 7 codes are required.
            # Operational temporal rule:
            #   - maximum gap after previous stop: 14 days
            #   - maximum overlap: 7 calendar days
            switch_events: set[
                tuple[
                    str,
                    str,
                    pd.Timestamp,
                ]
            ] = set()

            interval_records: list[
                tuple[
                    str,
                    pd.Timestamp,
                    pd.Timestamp,
                ]
            ] = []

            for atc7, intervals in drug_intervals.items():
                for start, stop in intervals:
                    interval_records.append(
                        (
                            atc7,
                            start,
                            stop,
                        )
                    )

            interval_records.sort(
                key=lambda item: (
                    item[1],
                    item[2],
                    item[0],
                )
            )

            for previous_index, (
                previous_drug,
                previous_start,
                previous_stop,
            ) in enumerate(
                interval_records
            ):
                for (
                    next_drug,
                    next_start,
                    next_stop,
                ) in interval_records[
                    previous_index + 1:
                ]:
                    if next_drug == previous_drug:
                        continue

                    gap_days = (
                        next_start
                        - previous_stop
                    ).days

                    if gap_days > MAX_SWITCH_GAP_DAYS:
                        break

                    overlap_days = max(
                        0,
                        (
                            previous_stop
                            - next_start
                        ).days
                        + 1,
                    )

                    if (
                        gap_days >= -MAX_SWITCH_OVERLAP_DAYS
                        and overlap_days <= MAX_SWITCH_OVERLAP_DAYS
                    ):
                        switch_events.add(
                            (
                                previous_drug,
                                next_drug,
                                next_start,
                            )
                        )
                        break

            patient_output[
                f"{suffix}_switch_count"
            ] = int(
                len(switch_events)
            )

        # Mental-pain medication overlap ratio:
        # overlap active days / union of mental-or-pain active days.
        mental_intervals: list[
            tuple[
                pd.Timestamp,
                pd.Timestamp,
            ]
        ] = []

        pain_intervals: list[
            tuple[
                pd.Timestamp,
                pd.Timestamp,
            ]
        ] = []

        for medication_row in patient_medication.itertuples(
            index=False
        ):
            start = max(
                pd.Timestamp(
                    medication_row.med_start_date
                ).normalize(),
                index_date,
            )

            stop = min(
                pd.Timestamp(
                    medication_row.effective_stop_date
                ).normalize(),
                feature_end_date,
            )

            if stop < start:
                continue

            interval = (
                start,
                stop,
            )

            if medication_row.study_group == "mental":
                mental_intervals.append(
                    interval
                )
            elif medication_row.study_group == "pain":
                pain_intervals.append(
                    interval
                )

        mental_days = interval_days(
            merged_intervals(
                mental_intervals
            )
        )

        pain_days = interval_days(
            merged_intervals(
                pain_intervals
            )
        )

        union_days = mental_days | pain_days
        overlap_days = mental_days & pain_days

        patient_output[
            "Mental_pain_overlap_ratio"
        ] = float(
            len(overlap_days)
            / len(union_days)
            if union_days
            else 0.0
        )

        rows.append(
            patient_output
        )

    output = pd.DataFrame(
        rows
    )

    ratio_columns = [
        "Mental_pain_overlap_ratio",
    ]

    integer_columns = [
        column
        for column in output.columns
        if column not in {
            "Patient_ID",
            *ratio_columns,
        }
    ]

    output[integer_columns] = (
        output[integer_columns]
        .fillna(0)
        .astype(int)
    )

    output[ratio_columns] = (
        output[ratio_columns]
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
        .astype(float)
    )

    return output



def build_final_analysis_dataframe(
    cohort: pd.DataFrame,
    mental_features: pd.DataFrame,
    disease_features: pd.DataFrame,
    medication_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the final one-year analysis dataframe.
    """
    analysis = cohort[
        [
            "Patient_ID",
            "MSK_within_1year",
            "Age",
            "Sex",
        ]
    ].copy()

    for feature_table in [
        mental_features,
        disease_features,
        medication_features,
    ]:
        analysis = analysis.merge(
            feature_table,
            on="Patient_ID",
            how="left",
            validate="one_to_one",
        )

    mental_columns = [
        "Mental_diagnosis_count",
        "Recent_30d_mental_visit",
        "Repeated_same_mental_diagnosis",
        "Unique_mental_diagnosis_count",
    ]

    disease_columns = [
        "Preexisting_condition_count",
        "Multimorbidity_count",
    ]

    burden_columns = [
        "Mental_medication_count",
        "Pain_medication_count",
        "Other_medication_count",
        "Other_medication_class_count",
    ]

    any_columns = [
        f"Any_{ATC_FEATURE_SUFFIX[code]}"
        for code in ANY_ATC_CODES
    ]

    discontinuation_columns = [
        (
            f"{ATC_FEATURE_SUFFIX[code]}"
            "_discontinuation_count"
        )
        for code in TRAJECTORY_ATC_CODES
    ]

    switching_columns = [
        (
            f"{ATC_FEATURE_SUFFIX[code]}"
            "_switch_count"
        )
        for code in TRAJECTORY_ATC_CODES
    ]

    timing_columns = [
        "Time_to_first_mental_medication",
        "Time_to_first_pain_medication",
    ]

    overlap_columns = [
        "Mental_pain_overlap_ratio",
    ]

    predictor_columns = [
        "Age",
        "Sex",
        *mental_columns,
        *disease_columns,
        *burden_columns,
        *any_columns,
        *discontinuation_columns,
        *switching_columns,
        *timing_columns,
        *overlap_columns,
    ]

    if len(predictor_columns) != 43:
        raise RuntimeError(
            "Final predictor definition must contain exactly 43 features; "
            f"found {len(predictor_columns)}."
        )

    if len(set(predictor_columns)) != len(
        predictor_columns
    ):
        raise RuntimeError(
            "Duplicate names were found in the final predictor list."
        )

    ordered_columns = [
        "Patient_ID",
        "MSK_within_1year",
        *predictor_columns,
    ]

    missing_columns = [
        column
        for column in ordered_columns
        if column not in analysis.columns
    ]

    if missing_columns:
        raise KeyError(
            "Final analysis dataframe is missing columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    analysis = analysis[
        ordered_columns
    ].copy()

    numeric_columns = [
        column
        for column in analysis.columns
        if column not in {
            "Patient_ID",
            "Sex",
        }
    ]

    for column in numeric_columns:
        analysis[column] = pd.to_numeric(
            analysis[column],
            errors="coerce",
        )

    missing_numeric = (
        analysis[
            numeric_columns
        ]
        .isna()
        .sum()
    )

    if missing_numeric.gt(0).any():
        raise ValueError(
            "Numeric final features contain missing values:\n"
            + missing_numeric[
                missing_numeric.gt(0)
            ].to_string()
        )

    ratio_columns = [
        "Mental_pain_overlap_ratio",
    ]

    integer_columns = [
        column
        for column in numeric_columns
        if column not in ratio_columns
    ]

    analysis[integer_columns] = (
        analysis[integer_columns]
        .astype(int)
    )

    analysis[ratio_columns] = (
        analysis[ratio_columns]
        .astype(float)
    )

    if analysis[
        "Patient_ID"
    ].duplicated().any():
        raise ValueError(
            "Final analysis dataframe contains duplicated Patient_ID values."
        )

    if len(analysis) != len(
        cohort
    ):
        raise ValueError(
            "Final analysis row count does not match the cohort row count."
        )

    return (
        analysis.sort_values(
            "Patient_ID"
        )
        .reset_index(
            drop=True
        )
    )


def build_feature_summary(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact predictor summary.
    """
    rows: list[
        dict[str, object]
    ] = []

    for column in analysis.columns:
        if column in {
            "Patient_ID",
            "Sex",
        }:
            continue

        values = pd.to_numeric(
            analysis[column],
            errors="coerce",
        )

        rows.append(
            {
                "feature": column,
                "n": int(
                    values.notna().sum()
                ),
                "n_missing": int(
                    values.isna().sum()
                ),
                "n_unique": int(
                    values.nunique(
                        dropna=True
                    )
                ),
                "mean": float(
                    values.mean()
                ),
                "median": float(
                    values.median()
                ),
                "minimum": float(
                    values.min()
                ),
                "p99": float(
                    values.quantile(
                        0.99
                    )
                ),
                "maximum": float(
                    values.max()
                ),
                "zero_percent": float(
                    values.eq(0).mean()
                    * 100
                ),
            }
        )

    return pd.DataFrame(
        rows
    )



def build_analysis_missingness(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize missingness for every final analysis column.
    """
    missingness = pd.DataFrame(
        {
            "column": analysis.columns,
            "n_missing": [
                int(analysis[column].isna().sum())
                for column in analysis.columns
            ],
        }
    )

    missingness["percent_missing"] = (
        missingness["n_missing"]
        / len(analysis)
        * 100
    )

    return missingness



def build_dynamic_medication(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize only final medication trajectory/timing predictors.
    """
    dynamic_columns = [
        column
        for column in analysis.columns
        if column.endswith(
            "_discontinuation_count"
        )
        or column.endswith(
            "_switch_count"
        )
        or column.startswith(
            "Time_to_first_"
        )
        or column == "Mental_pain_overlap_ratio"
    ]

    rows: list[dict[str, object]] = []

    for column in dynamic_columns:
        values = pd.to_numeric(
            analysis[column],
            errors="coerce",
        ).fillna(0)

        rows.append(
            {
                "feature": column,
                "n_negative": int(values.lt(0).sum()),
                "n_zero": int(values.eq(0).sum()),
                "n_positive": int(values.gt(0).sum()),
                "percent_zero": float(values.eq(0).mean() * 100),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "n_unique": int(values.nunique()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """
    Build and save the final one-year analysis dataframe.
    """
    create_project_directories()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log.info(
        "=" * 80
    )

    log.info(
        "STEP 1 — LOAD INPUT DATA"
    )

    log.info(
        "=" * 80
    )

    cohort_raw = pd.read_csv(
        COHORT_CSV,
        dtype=str,
        low_memory=False,
    )

    encounter_raw = pd.read_csv(
        ENCOUNTER_CSV,
        dtype=str,
        low_memory=False,
    )

    medication_raw = pd.read_csv(
        MEDICATION_CSV,
        dtype=str,
        low_memory=False,
    )

    log.info(
        f"Cohort rows: {len(cohort_raw):,}"
    )

    log.info(
        f"Encounter rows: {len(encounter_raw):,}"
    )

    log.info(
        f"Medication rows: {len(medication_raw):,}"
    )

    log.info(
        "=" * 80
    )

    log.info(
        "STEP 2 — PREPARE COHORT AND DIAGNOSES"
    )

    log.info(
        "=" * 80
    )

    cohort = prepare_cohort(
        cohort_raw
    )

    diagnosis = prepare_diagnosis(
        encounter_raw,
        cohort,
    )

    mental_features = build_mental_features(
        diagnosis,
        cohort,
    )

    disease_features = build_disease_burden_features(
        diagnosis,
        cohort,
    )

    log.info(
        "=" * 80
    )

    log.info(
        "STEP 3 — PREPARE ATC MEDICATION TIMELINE"
    )

    log.info(
        "=" * 80
    )

    medication = prepare_medication(
        medication_raw,
        cohort,
    )

    medication_features = build_patient_medication_features(
        medication,
        cohort,
    )

    log.info(
        "=" * 80
    )

    log.info(
        "STEP 4 — BUILD FINAL ANALYSIS DATAFRAME"
    )

    log.info(
        "=" * 80
    )

    analysis = build_final_analysis_dataframe(
        cohort=cohort,
        mental_features=mental_features,
        disease_features=disease_features,
        medication_features=medication_features,
    )

    feature_summary = build_feature_summary(
        analysis
    )

    analysis_missingness = build_analysis_missingness(
        analysis
    )

    dynamic_medication = build_dynamic_medication(
        analysis
    )

    analysis.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    feature_summary.to_csv(
        FEATURE_SUMMARY_FILE,
        index=False,
    )

    analysis_missingness.to_csv(
        ANALYSIS_MISSINGNESS_FILE,
        index=False,
    )

    dynamic_medication.to_csv(
        DYNAMIC_MEDICATION_FILE,
        index=False,
    )

    log.info(
        "=" * 80
    )

    log.info(
        "BUILD COMPLETED"
    )

    log.info(
        "=" * 80
    )

    log.info(
        f"Final patients: {len(analysis):,}"
    )

    log.info(
        f"Final columns: {analysis.shape[1]:,}"
    )

    log.info(
        f"MSK within one year: "
        f"{analysis['MSK_within_1year'].sum():,} "
        f"({analysis['MSK_within_1year'].mean():.2%})"
    )

    log.info(
        f"Saved analysis dataframe: {OUTPUT_FILE}"
    )

    log.info(
        f"Saved medication timeline: {MEDICATION_TIMELINE_FILE}"
    )

    log.info(
        f"Saved feature summary: {FEATURE_SUMMARY_FILE}"
    )

    log.info(
        f"Saved analysis missingness: {ANALYSIS_MISSINGNESS_FILE}"
    )

    log.info(
        f"Saved dynamic medication: {DYNAMIC_MEDICATION_FILE}"
    )


if __name__ == "__main__":
    main()
