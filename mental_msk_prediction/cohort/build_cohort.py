"""
Build the mental-to-MSK study cohort from cleaned CPCSSN data.

This module combines the previous:
- build_pre_msk_tables.py
- pre_msk_processing.py
- final_cohort_processing.py

It performs cohort construction only. It does not generate figures, build
medication features, construct model variables, or train models.

Outputs
-------
1. mental_to_msk_index.csv
   One row per patient whose first recorded mental diagnosis precedes the
   first recorded MSK diagnosis.

2. pre_msk_diagnosis_long.csv
   Classified diagnosis records before the first MSK diagnosis for eligible
   patients.

3. pre_msk_patient_summary.csv
   One row per eligible patient summarizing pre-MSK ICD-9 categories and
   broad disease systems.

4. final_cohort.csv
   One row per adult patient with valid demographics.

5. cohort_flow.csv
   Counts after each cohort-selection step.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pandas as pd

from mental_msk_prediction.config.clinical_definitions import (
    COMORBIDITY_SYSTEMS_EDA,
    classify_category,
    classify_system_broad,
    extract_icd9_3,
    get_icd9_display_label,
)
from mental_msk_prediction.config.paths import (
    CLEANED_DIAGNOSIS_FILE,
    CLEANED_PATIENT_FILE,
    RESULTS_DATA_DIR,
    TABLES_DIR,
    create_project_directories,
)

# Study settings and output paths
MINIMUM_AGE = 18

COHORT_DATA_DIR = RESULTS_DATA_DIR / "cohort"
COHORT_TABLE_DIR = TABLES_DIR / "cohort"
MENTAL_TO_MSK_INDEX_FILE = (COHORT_DATA_DIR / "mental_to_msk_index.csv")
PRE_MSK_LONG_FILE = (COHORT_DATA_DIR / "pre_msk_diagnosis_long.csv")
PRE_MSK_PATIENT_FILE = (COHORT_DATA_DIR / "pre_msk_patient_summary.csv")
FINAL_COHORT_FILE = (COHORT_DATA_DIR / "final_cohort.csv")
COHORT_FLOW_FILE = (COHORT_TABLE_DIR / "cohort_flow.csv")

def validate_input_files() -> None:
    """
    Confirm that cleaned patient and diagnosis files are available.
    """
    required_files = [
        CLEANED_PATIENT_FILE,
        CLEANED_DIAGNOSIS_FILE,
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.is_file()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {file_path}"
            for file_path in missing_files
        )

        raise FileNotFoundError(
            "Required cleaned input files were not found:\n"
            f"{missing_text}\n\n"
            "Run data cleaning first:\n"
            "python -m mental_msk_prediction.preprocessing.clean_data"
        )


def load_cleaned_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a cleaned comma-delimited CPCSSN table.
    """
    return pd.read_csv(
        file_path,
        dtype=str,
        low_memory=False,
        encoding="utf-8-sig",
    )


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """
    Raise an informative error when required columns are absent.
    """
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{table_name} is missing required columns: "
            + ", ".join(missing_columns)
        )



def prepare_diagnosis_table(
    diagnosis: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        diagnosis,
        [
            "Patient_ID",
            "DateCreated",
            "DiagnosisCodeType_calc",
            "DiagnosisCode_calc",
        ],
        "Cleaned diagnosis table",
    )

    df = diagnosis.copy()

    df["Patient_ID"] = (
        df["Patient_ID"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    df["DiagnosisCodeType_calc"] = (
        df["DiagnosisCodeType_calc"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Use calculated ICD-9 diagnoses only, matching the original pipeline.
    df = df[
        df["DiagnosisCodeType_calc"].eq("ICD9")
    ].copy()

    df["diagnosis_date"] = pd.to_datetime(
        df["DateCreated"],
        errors="coerce",
    )

    df["icd9"] = (
        df["DiagnosisCode_calc"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    df["icd9_3"] = (
        df["icd9"]
        .apply(extract_icd9_3)
        .astype("Int64")
    )

    df["system"] = (
        df["icd9"]
        .apply(classify_system_broad)
        .astype("string")
    )

    df["category"] = (
        df["icd9"]
        .apply(classify_category)
        .astype("string")
    )

    df["display_label"] = (
        df["icd9"]
        .apply(get_icd9_display_label)
        .astype("string")
    )

    usable = df.dropna(
        subset=[
            "Patient_ID",
            "diagnosis_date",
            "icd9",
            "icd9_3",
            "system",
        ]
    ).copy()

    columns = [
        column
        for column in [
            "EncounterDiagnosis_ID",
            "Encounter_ID",
            "Patient_ID",
            "diagnosis_date",
            "icd9",
            "icd9_3",
            "system",
            "category",
            "display_label",
            "DiagnosisText_calc",
            "DiagnosisText_orig",
        ]
        if column in usable.columns
    ]

    return (
        usable[columns]
        .sort_values(
            [
                "Patient_ID",
                "diagnosis_date",
                "icd9_3",
                "icd9",
            ]
        )
        .reset_index(drop=True)
    )


# Mental-to-MSK index
def build_mental_to_msk_index(
    diagnosis_prepared: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build first mental and first MSK dates for each patient.

    The returned index contains patients with:
        first_mental_date < first_msk_date

    Patients with same-day first mental and first MSK diagnoses are excluded.
    """
    condition_records = diagnosis_prepared[
        diagnosis_prepared["system"].isin(
            ["mental", "msk"]
        )
    ].copy()

    first_dates = (
        condition_records
        .groupby(
            ["Patient_ID", "system"],
            observed=True,
        )["diagnosis_date"]
        .min()
        .unstack("system")
        .reset_index()
        .rename(
            columns={
                "mental": "first_mental_date",
                "msk": "first_msk_date",
            }
        )
    )

    for column in [
        "first_mental_date",
        "first_msk_date",
    ]:
        if column not in first_dates.columns:
            first_dates[column] = pd.NaT

    has_mental = first_dates[
        "first_mental_date"
    ].notna()

    has_mental_and_msk = (
        has_mental
        & first_dates["first_msk_date"].notna()
    )

    mental_before_msk = (
        has_mental_and_msk
        & (
            first_dates["first_mental_date"]
            < first_dates["first_msk_date"]
        )
    )

    same_day = (
        has_mental_and_msk
        & (
            first_dates["first_mental_date"]
            == first_dates["first_msk_date"]
        )
    )

    msk_before_mental = (
        has_mental_and_msk
        & (
            first_dates["first_msk_date"]
            < first_dates["first_mental_date"]
        )
    )

    index = (
        first_dates.loc[
            mental_before_msk,
            [
                "Patient_ID",
                "first_mental_date",
                "first_msk_date",
            ],
        ]
        .copy()
        .sort_values("Patient_ID")
        .reset_index(drop=True)
    )

    index["time_to_msk_days"] = (
        index["first_msk_date"]
        - index["first_mental_date"]
    ).dt.days

    flow = pd.DataFrame(
        [
            {
                "step": "Patients with at least one mental diagnosis",
                "N_patients": int(has_mental.sum()),
            },
            {
                "step": (
                    "Patients with both mental and MSK diagnoses"
                ),
                "N_patients": int(
                    has_mental_and_msk.sum()
                ),
            },
            {
                "step": (
                    "Excluded: first MSK diagnosis before "
                    "first mental diagnosis"
                ),
                "N_patients": int(
                    msk_before_mental.sum()
                ),
            },
            {
                "step": (
                    "Excluded: first mental and first MSK "
                    "diagnoses on the same day"
                ),
                "N_patients": int(
                    same_day.sum()
                ),
            },
            {
                "step": (
                    "Patients with first mental diagnosis "
                    "before first MSK diagnosis"
                ),
                "N_patients": int(
                    mental_before_msk.sum()
                ),
            },
        ]
    )

    return index, flow


# First-date diagnosis characteristics
def build_first_date_characteristics(
    diagnosis_prepared: pd.DataFrame,
    cohort_index: pd.DataFrame,
    condition: str,
    date_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract diagnoses recorded on each patient's first condition date.

    Returns
    -------
    selected_characteristic:
        One deterministic representative diagnosis per patient for the
        one-row-per-patient final cohort.

    all_first_date_diagnoses:
        Every distinct diagnosis category recorded on the first condition
        date. This table is retained for later patient-profile EDA.
    """
    if condition not in {"mental", "msk"}:
        raise ValueError(
            "condition must be 'mental' or 'msk'."
        )

    require_columns(
        cohort_index,
        ["Patient_ID", date_column],
        "Cohort index",
    )

    index = cohort_index[
        ["Patient_ID", date_column]
    ].copy()

    records = diagnosis_prepared.merge(
        index,
        on="Patient_ID",
        how="inner",
        validate="many_to_one",
    )

    records = records[
        records["system"].eq(condition)
        & records["diagnosis_date"].eq(
            records[date_column]
        )
    ].copy()

    prefix = f"first_{condition}"

    all_columns = [
        "Patient_ID",
        "diagnosis_date",
        "icd9",
        "icd9_3",
        "category",
        "display_label",
    ]

    all_first_date = (
        records[all_columns]
        .drop_duplicates(
            [
                "Patient_ID",
                "icd9",
            ]
        )
        .sort_values(
            [
                "Patient_ID",
                "icd9_3",
                "icd9",
            ]
        )
        .rename(
            columns={
                "diagnosis_date": f"{prefix}_date",
                "icd9": f"{prefix}_icd9",
                "icd9_3": f"{prefix}_icd9_3",
                "category": f"{prefix}_category",
                "display_label": (
                    f"{prefix}_display_label"
                ),
            }
        )
        .reset_index(drop=True)
    )

    selected = (
        all_first_date
        .drop_duplicates(
            "Patient_ID",
            keep="first",
        )
        .reset_index(drop=True)
    )

    return selected, all_first_date


# Pre-MSK tables
def build_pre_msk_long(
    diagnosis_prepared: pd.DataFrame,
    cohort_index: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retain classified diagnosis records occurring before first MSK.

    The first mental record is included. The first MSK record itself is not
    included because this table describes the period before MSK onset.
    """
    index = cohort_index[
        [
            "Patient_ID",
            "first_mental_date",
            "first_msk_date",
        ]
    ].copy()

    pre_msk = diagnosis_prepared.merge(
        index,
        on="Patient_ID",
        how="inner",
        validate="many_to_one",
    )

    pre_msk = pre_msk[
        pre_msk["diagnosis_date"]
        < pre_msk["first_msk_date"]
    ].copy()

    pre_msk["first_date_for_icd9_3"] = (
        pre_msk
        .groupby(
            ["Patient_ID", "icd9_3"],
            observed=True,
        )["diagnosis_date"]
        .transform("min")
    )

    return (
        pre_msk
        .sort_values(
            [
                "Patient_ID",
                "diagnosis_date",
                "icd9_3",
                "icd9",
            ]
        )
        .reset_index(drop=True)
    )


def join_unique_strings(
    values: pd.Series,
) -> str:
    """
    Join unique non-missing values in sorted order.
    """
    clean_values = {
        str(value).strip()
        for value in values.dropna()
        if str(value).strip()
    }

    return "; ".join(
        sorted(clean_values)
    )


def count_distinct_allowed_systems(
    values: pd.Series,
    allowed_systems: list[str],
) -> int:
    """
    Count distinct systems appearing in an allowed system list.
    """
    observed = {
        str(value).strip()
        for value in values.dropna()
        if str(value).strip()
    }

    return len(
        observed.intersection(
            set(allowed_systems)
        )
    )


def build_pre_msk_patient_summary(
    pre_msk_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a one-row-per-patient summary of pre-MSK diagnosis history.
    """
    grouped = (
        pre_msk_long
        .groupby(
            "Patient_ID",
            observed=True,
        )
        .agg(
            first_mental_date=(
                "first_mental_date",
                "first",
            ),
            first_msk_date=(
                "first_msk_date",
                "first",
            ),
            pre_msk_icd9_3_list=(
                "icd9_3",
                join_unique_strings,
            ),
            pre_msk_category_list=(
                "category",
                join_unique_strings,
            ),
            pre_msk_system_list=(
                "system",
                join_unique_strings,
            ),
            n_pre_msk_diagnosis_records=(
                "icd9",
                "size",
            ),
            n_unique_pre_msk_icd9_categories=(
                "icd9_3",
                "nunique",
            ),
        )
        .reset_index()
    )

    system_counts = (
        pre_msk_long
        .groupby(
            "Patient_ID",
            observed=True,
        )["system"]
        .apply(
            lambda values: (
                count_distinct_allowed_systems(
                    values,
                    COMORBIDITY_SYSTEMS_EDA,
                )
            )
        )
        .rename(
            "n_pre_msk_comorbidity_systems_eda"
        )
        .reset_index()
    )

    summary = grouped.merge(
        system_counts,
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )

    summary["time_to_msk_days"] = (
        summary["first_msk_date"]
        - summary["first_mental_date"]
    ).dt.days

    return (
        summary
        .sort_values("Patient_ID")
        .reset_index(drop=True)
    )


# Patient demographics and final cohort
def prepare_patient_demographics(
    patient: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare one demographic row per patient.
    """
    require_columns(
        patient,
        [
            "Patient_ID",
            "Sex",
            "BirthYear",
        ],
        "Cleaned patient table",
    )

    demographics = patient[
        [
            "Patient_ID",
            "Sex",
            "BirthYear",
        ]
    ].copy()

    demographics["Patient_ID"] = (
        demographics["Patient_ID"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    sex = (
        demographics["Sex"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    demographics["Sex"] = sex.map(
        {
            "F": "Female",
            "FE": "Female",
            "FEMALE": "Female",
            "M": "Male",
            "MA": "Male",
            "MALE": "Male",
        }
    )

    demographics["BirthYear"] = (
        pd.to_numeric(
            demographics["BirthYear"],
            errors="coerce",
        )
        .astype("Int64")
    )

    duplicated_patient_ids = (
        demographics["Patient_ID"]
        .notna()
        & demographics["Patient_ID"]
        .duplicated(keep=False)
    )

    if duplicated_patient_ids.any():
        conflicting = (
            demographics.loc[
                duplicated_patient_ids
            ]
            .groupby(
                "Patient_ID",
                observed=True,
            )
            .agg(
                n_sex_values=(
                    "Sex",
                    lambda values: (
                        values.dropna().nunique()
                    ),
                ),
                n_birth_year_values=(
                    "BirthYear",
                    lambda values: (
                        values.dropna().nunique()
                    ),
                ),
            )
        )

        conflicting = conflicting[
            (conflicting["n_sex_values"] > 1)
            | (
                conflicting[
                    "n_birth_year_values"
                ] > 1
            )
        ]

        if not conflicting.empty:
            raise ValueError(
                "Conflicting demographic records were found "
                f"for {len(conflicting):,} Patient_ID values."
            )

    return (
        demographics
        .drop_duplicates(
            "Patient_ID",
            keep="first",
        )
        .reset_index(drop=True)
    )


def build_final_cohort(
    cohort_index: pd.DataFrame,
    demographics: pd.DataFrame,
    first_mental_selected: pd.DataFrame,
    first_msk_selected: pd.DataFrame,
    minimum_age: int = MINIMUM_AGE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the final adult mental-to-MSK cohort.

    Inclusion criteria
    ------------------
    - first mental diagnosis occurs before first MSK diagnosis;
    - Sex is Female or Male;
    - BirthYear is available;
    - age at first mental diagnosis is at least minimum_age.

    Age is anchored at the first mental diagnosis year.
    """
    cohort = cohort_index.merge(
        demographics,
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )

    cohort = cohort.merge(
        first_mental_selected.drop(
            columns=["first_mental_date"],
            errors="ignore",
        ),
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )

    cohort = cohort.merge(
        first_msk_selected.drop(
            columns=["first_msk_date"],
            errors="ignore",
        ),
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )

    cohort["Age_at_first_mental"] = (
        cohort["first_mental_date"].dt.year
        - cohort["BirthYear"]
    ).astype("Int64")

    valid_sex = cohort["Sex"].isin(
        ["Female", "Male"]
    )
    valid_birth_year = (
        cohort["BirthYear"].notna()
    )
    valid_demographics = (
        valid_sex
        & valid_birth_year
    )

    valid_age = (
        cohort["Age_at_first_mental"].notna()
        & (
            cohort["Age_at_first_mental"]
            >= minimum_age
        )
    )

    valid_time_order = (
        cohort["first_mental_date"].notna()
        & cohort["first_msk_date"].notna()
        & (
            cohort["first_mental_date"]
            < cohort["first_msk_date"]
        )
    )

    final_mask = (
        valid_time_order
        & valid_demographics
        & valid_age
    )

    flow = pd.DataFrame(
        [
            {
                "step": (
                    "Patients with first mental diagnosis "
                    "before first MSK diagnosis"
                ),
                "N_patients": int(
                    valid_time_order.sum()
                ),
            },
            {
                "step": (
                    "Patients with valid sex and birth year"
                ),
                "N_patients": int(
                    (
                        valid_time_order
                        & valid_demographics
                    ).sum()
                ),
            },
            {
                "step": (
                    f"Final adult cohort "
                    f"(age ≥ {minimum_age} at first mental diagnosis)"
                ),
                "N_patients": int(
                    final_mask.sum()
                ),
            },
        ]
    )

    final_columns = [
        "Patient_ID",
        "Sex",
        "BirthYear",
        "Age_at_first_mental",
        "first_mental_date",
        "first_mental_icd9",
        "first_mental_icd9_3",
        "first_mental_category",
        "first_mental_display_label",
        "first_msk_date",
        "time_to_msk_days",
        "first_msk_icd9",
        "first_msk_icd9_3",
        "first_msk_category",
        "first_msk_display_label",
    ]

    final = (
        cohort.loc[
            final_mask,
            final_columns,
        ]
        .sort_values("Patient_ID")
        .reset_index(drop=True)
    )

    if final.empty:
        raise ValueError(
            "The final cohort is empty after applying "
            "the inclusion criteria."
        )

    if final["Patient_ID"].duplicated().any():
        raise ValueError(
            "The final cohort contains duplicated Patient_ID values."
        )

    return final, flow

def save_csv(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a dataframe and create its parent directory.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )


def validate_outputs(
    cohort_index: pd.DataFrame,
    pre_msk_long: pd.DataFrame,
    pre_msk_patient: pd.DataFrame,
    final_cohort: pd.DataFrame,
) -> None:
    """
    Perform basic consistency checks before completing the pipeline.
    """
    if cohort_index.empty:
        raise ValueError(
            "The mental-to-MSK index is empty."
        )

    if pre_msk_long.empty:
        raise ValueError(
            "The pre-MSK diagnosis table is empty."
        )

    if pre_msk_patient.empty:
        raise ValueError(
            "The pre-MSK patient summary is empty."
        )

    if final_cohort.empty:
        raise ValueError(
            "The final cohort is empty."
        )

    if (
        pre_msk_patient["Patient_ID"].nunique()
        != len(pre_msk_patient)
    ):
        raise ValueError(
            "The pre-MSK patient summary is not "
            "one row per patient."
        )

    if (
        final_cohort["Patient_ID"].nunique()
        != len(final_cohort)
    ):
        raise ValueError(
            "The final cohort is not one row per patient."
        )

    final_ids = set(
        final_cohort["Patient_ID"]
    )
    index_ids = set(
        cohort_index["Patient_ID"]
    )

    if not final_ids.issubset(index_ids):
        raise ValueError(
            "The final cohort contains patients not present "
            "in the mental-to-MSK index."
        )


def main() -> None:
    """
    Run complete cohort construction.
    """
    create_project_directories()
    validate_input_files()

    COHORT_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    COHORT_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("STEP 1 — LOAD CLEANED DATA")
    print("=" * 80)

    diagnosis_raw = load_cleaned_csv(
        CLEANED_DIAGNOSIS_FILE
    )
    patient_raw = load_cleaned_csv(
        CLEANED_PATIENT_FILE
    )

    print(
        f"Diagnosis rows loaded: {len(diagnosis_raw):,}"
    )
    print(
        f"Patient rows loaded:   {len(patient_raw):,}"
    )

    print("\n" + "=" * 80)
    print("STEP 2 — PREPARE DIAGNOSIS RECORDS")
    print("=" * 80)

    diagnosis = prepare_diagnosis_table(
        diagnosis_raw
    )

    print(
        f"Usable classified diagnosis rows: {len(diagnosis):,}"
    )
    print(
        "Unique patients with classified diagnoses:",
        f"{diagnosis['Patient_ID'].nunique():,}",
    )

    del diagnosis_raw
    gc.collect()

    print("\n" + "=" * 80)
    print("STEP 3 — BUILD MENTAL-TO-MSK INDEX")
    print("=" * 80)

    cohort_index, temporal_flow = (
        build_mental_to_msk_index(
            diagnosis
        )
    )

    print(
        "Patients with first mental before first MSK:",
        f"{len(cohort_index):,}",
    )

    print("\n" + "=" * 80)
    print("STEP 4 — BUILD FIRST-DATE CHARACTERISTICS")
    print("=" * 80)

    (
        first_mental_selected,
        first_mental_all,
    ) = build_first_date_characteristics(
        diagnosis_prepared=diagnosis,
        cohort_index=cohort_index,
        condition="mental",
        date_column="first_mental_date",
    )

    (
        first_msk_selected,
        first_msk_all,
    ) = build_first_date_characteristics(
        diagnosis_prepared=diagnosis,
        cohort_index=cohort_index,
        condition="msk",
        date_column="first_msk_date",
    )

    print(
        "Patients with first-date mental characteristics:",
        f"{first_mental_selected['Patient_ID'].nunique():,}",
    )
    print(
        "Patients with first-date MSK characteristics:",
        f"{first_msk_selected['Patient_ID'].nunique():,}",
    )

    print("\n" + "=" * 80)
    print("STEP 5 — BUILD PRE-MSK TABLES")
    print("=" * 80)

    pre_msk_long = build_pre_msk_long(
        diagnosis_prepared=diagnosis,
        cohort_index=cohort_index,
    )

    pre_msk_patient = (
        build_pre_msk_patient_summary(
            pre_msk_long
        )
    )

    print(
        f"Pre-MSK diagnosis rows: {len(pre_msk_long):,}"
    )
    print(
        f"Pre-MSK patients:       {len(pre_msk_patient):,}"
    )

    print("\n" + "=" * 80)
    print("STEP 6 — BUILD FINAL ADULT COHORT")
    print("=" * 80)

    demographics = prepare_patient_demographics(
        patient_raw
    )

    final_cohort, final_flow = build_final_cohort(
        cohort_index=cohort_index,
        demographics=demographics,
        first_mental_selected=first_mental_selected,
        first_msk_selected=first_msk_selected,
        minimum_age=MINIMUM_AGE,
    )

    cohort_flow = pd.concat(
        [
            temporal_flow,
            final_flow,
        ],
        ignore_index=True,
    )

    print(
        f"Final cohort patients: {len(final_cohort):,}"
    )

    validate_outputs(
        cohort_index=cohort_index,
        pre_msk_long=pre_msk_long,
        pre_msk_patient=pre_msk_patient,
        final_cohort=final_cohort,
    )

    print("\n" + "=" * 80)
    print("STEP 7 — SAVE OUTPUTS")
    print("=" * 80)

    save_csv(
        cohort_index,
        MENTAL_TO_MSK_INDEX_FILE,
    )
    save_csv(
        pre_msk_long,
        PRE_MSK_LONG_FILE,
    )
    save_csv(
        pre_msk_patient,
        PRE_MSK_PATIENT_FILE,
    )
    save_csv(
        final_cohort,
        FINAL_COHORT_FILE,
    )
    save_csv(
        cohort_flow,
        COHORT_FLOW_FILE,
    )

    # These two tables preserve all distinct diagnoses recorded on the first
    # mental/MSK dates and are used later by cohort EDA.
    save_csv(
        first_mental_all,
        COHORT_DATA_DIR
        / "first_mental_date_diagnoses.csv",
    )
    save_csv(
        first_msk_all,
        COHORT_DATA_DIR
        / "first_msk_date_diagnoses.csv",
    )

    print("\n" + "=" * 80)
    print("COHORT CONSTRUCTION COMPLETED")
    print("=" * 80)
    print(
        f"Mental-to-MSK index: {MENTAL_TO_MSK_INDEX_FILE}"
    )
    print(
        f"Pre-MSK long table:  {PRE_MSK_LONG_FILE}"
    )
    print(
        f"Pre-MSK summary:     {PRE_MSK_PATIENT_FILE}"
    )
    print(
        f"Final cohort:        {FINAL_COHORT_FILE}"
    )
    print(
        f"Cohort flow:         {COHORT_FLOW_FILE}"
    )


if __name__ == "__main__":
    main()
