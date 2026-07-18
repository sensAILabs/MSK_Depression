from __future__ import annotations

import numpy as np
import pandas as pd

from eda.clinical_definitions import (
    _icd9_3_int,
    classify_system,
    classify_subgroup,
    classify_acute_chronic,
)


def normalize_sex(series: pd.Series) -> pd.Series:
    """Standardize sex values to F/M."""
    values = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({
            "FEMALE": "F",
            "MALE": "M",
        })
    )

    return values.where(values.isin(["F", "M"]))


def build_first_system_dates_from_pre_long(
    pre_long: pd.DataFrame,
    patient_col: str = "Patient_ID",
    date_col: str = "DateCreated",
    type_col: str = "_type",
) -> pd.DataFrame:
    """Get the first recorded date for every disease system per patient."""
    df = pre_long[
        [patient_col, date_col, type_col]
    ].copy()

    df[patient_col] = (
        df[patient_col]
        .astype(str)
        .str.strip()
    )

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    df[type_col] = (
        df[type_col]
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=[
            patient_col,
            date_col,
            type_col,
        ]
    )

    return (
        df.groupby(
            [patient_col, type_col],
            as_index=False,
        )[date_col]
        .min()
        .rename(
            columns={
                date_col: "first_date",
                type_col: "system",
            }
        )
    )


def pivot_first_dates(
    first_system_dates: pd.DataFrame,
    patient_col: str = "Patient_ID",
) -> pd.DataFrame:
    """Convert first-system dates into one row per patient."""
    wide = (
        first_system_dates
        .pivot_table(
            index=patient_col,
            columns="system",
            values="first_date",
            aggfunc="min",
        )
        .reset_index()
    )

    rename_map = {
        "respiratory": "first_resp",
        "mental": "first_mental",
        "msk": "first_msk",
    }

    return wide.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in wide.columns
        }
    )


def add_first_msk_from_pre_patient_table(
    wide_dates: pd.DataFrame,
    pre_patient: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add first MSK date from the patient-level pre-MSK table.

    The pre-MSK encounter table excludes MSK records, so first MSK must be
    restored from analysis_df2_pre_msk_named.csv.
    """
    wide = wide_dates.copy()
    patient_table = pre_patient.copy()

    wide["Patient_ID"] = (
        wide["Patient_ID"]
        .astype(str)
        .str.strip()
    )

    patient_table["Patient_ID"] = (
        patient_table["Patient_ID"]
        .astype(str)
        .str.strip()
    )

    first_msk = (
        patient_table[
            ["Patient_ID", "first_msk_date"]
        ]
        .copy()
        .rename(
            columns={
                "first_msk_date": "first_msk"
            }
        )
    )

    first_msk["first_msk"] = pd.to_datetime(
        first_msk["first_msk"],
        errors="coerce",
    )

    # Drop an existing first_msk column before merging, if present.
    wide = wide.drop(
        columns=["first_msk"],
        errors="ignore",
    )

    return wide.merge(
        first_msk,
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )


def build_first_msk_subgroup_from_encounter(
    encounter: pd.DataFrame,
    pre_patient: pd.DataFrame,
) -> pd.DataFrame:
    """Identify the first MSK ICD-9 diagnosis and subgroup per patient."""
    enc = encounter.copy()
    patient_table = pre_patient.copy()

    enc["Patient_ID"] = (
        enc["Patient_ID"]
        .astype(str)
        .str.strip()
    )

    patient_table["Patient_ID"] = (
        patient_table["Patient_ID"]
        .astype(str)
        .str.strip()
    )

    if "DiagnosisCodeType_calc" in enc.columns:
        enc = enc[
            enc["DiagnosisCodeType_calc"] == "ICD9"
        ].copy()

    if (
        "DiagnosisCode_calc" in enc.columns
        and "icd9" not in enc.columns
    ):
        enc = enc.rename(
            columns={
                "DiagnosisCode_calc": "icd9"
            }
        )

    if "icd9" not in enc.columns:
        raise ValueError(
            "Encounter table must contain either "
            "'icd9' or 'DiagnosisCode_calc'."
        )

    enc["DateCreated"] = pd.to_datetime(
        enc["DateCreated"],
        errors="coerce",
    )

    enc["icd9"] = (
        enc["icd9"]
        .astype(str)
        .str.strip()
    )

    enc["icd9_3"] = enc["icd9"].apply(
        _icd9_3_int
    )

    enc["system"] = enc["icd9"].apply(
        classify_system
    )

    enc["subgroup"] = enc["icd9"].apply(
        classify_subgroup
    )

    enc["acute_chronic"] = enc["icd9"].apply(
        classify_acute_chronic
    )

    patient_first_msk = (
        patient_table[
            ["Patient_ID", "first_msk_date"]
        ]
        .copy()
    )

    patient_first_msk["first_msk_date"] = pd.to_datetime(
        patient_first_msk["first_msk_date"],
        errors="coerce",
    )

    df = enc.merge(
        patient_first_msk,
        on="Patient_ID",
        how="inner",
        validate="many_to_one",
    )

    df = df[
        (df["system"] == "msk")
        & (
            df["DateCreated"]
            == df["first_msk_date"]
        )
    ].copy()

    df = df.dropna(
        subset=[
            "Patient_ID",
            "icd9_3",
            "subgroup",
        ]
    )

    df = df.sort_values(
        [
            "Patient_ID",
            "DateCreated",
            "icd9_3",
            "icd9",
        ]
    )

    return (
        df.drop_duplicates(
            subset=["Patient_ID"],
            keep="first",
        )
        [
            [
                "Patient_ID",
                "icd9",
                "icd9_3",
                "subgroup",
                "acute_chronic",
            ]
        ]
        .rename(
            columns={
                "icd9": "first_msk_icd9",
                "icd9_3": "first_msk_icd9_3",
                "subgroup": "first_msk_subgroup",
                "acute_chronic": "first_msk_course",
            }
        )
        .copy()
    )


def add_msk_exposure_features(
    wide_dates: pd.DataFrame,
    patient: pd.DataFrame,
) -> pd.DataFrame:
    """Add demographics, exposure dates and mental/respiratory groups."""
    df = wide_dates.copy()
    patient = patient.copy()

    df["Patient_ID"] = (
        df["Patient_ID"]
        .astype(str)
        .str.strip()
    )

    patient["Patient_ID"] = (
        patient["Patient_ID"]
        .astype(str)
        .str.strip()
    )

    demographic_columns = [
        column
        for column in [
            "Patient_ID",
            "Sex",
            "BirthYear",
        ]
        if column in patient.columns
    ]

    df = df.merge(
        patient[demographic_columns]
        .drop_duplicates("Patient_ID"),
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )

    if "first_msk" not in df.columns:
        raise ValueError(
            "The first-event table is missing first_msk."
        )

    for column in [
        "first_resp",
        "first_mental",
        "first_msk",
    ]:
        if column not in df.columns:
            df[column] = pd.NaT

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=["first_msk"]
    ).copy()

    if "Sex" in df.columns:
        df["Sex"] = normalize_sex(
            df["Sex"]
        )

    if "BirthYear" in df.columns:
        df["BirthYear"] = pd.to_numeric(
            df["BirthYear"],
            errors="coerce",
        )

        df["Age_at_MSK"] = (
            df["first_msk"].dt.year
            - df["BirthYear"]
        )

        df["Age_at_mental"] = (
            df["first_mental"].dt.year
            - df["BirthYear"]
        )

        df["Age_at_MSK"] = df["Age_at_MSK"].clip(
            lower=0,
            upper=110,
        )

        df["Age_at_mental"] = df["Age_at_mental"].clip(
            lower=0,
            upper=110,
        )

    # Respiratory records only count when they occur after first mental
    # and before first MSK.
    df["resp_before_msk"] = df["first_resp"].where(
        (df["first_resp"] >= df["first_mental"])
        & (df["first_resp"] < df["first_msk"])
    )

    df["mental_before_msk"] = df["first_mental"].where(
        df["first_mental"] < df["first_msk"]
    )

    has_resp = df["resp_before_msk"].notna()
    has_mental = df["mental_before_msk"].notna()

    group_order = [
        "None",
        "Resp only",
        "Mental only",
        "Both",
    ]

    df["group"] = np.select(
        [
            (~has_resp) & (~has_mental),
            has_resp & (~has_mental),
            (~has_resp) & has_mental,
            has_resp & has_mental,
        ],
        group_order,
        default="None",
    )

    df["group"] = pd.Categorical(
        df["group"],
        categories=group_order,
        ordered=True,
    )

    df["has_resp"] = df["group"].isin(
        ["Resp only", "Both"]
    )

    df["has_mental"] = df["group"].isin(
        ["Mental only", "Both"]
    )

    df["exposure_start"] = pd.NaT

    resp_only = df["group"] == "Resp only"
    mental_only = df["group"] == "Mental only"
    both = df["group"] == "Both"

    df.loc[
        resp_only,
        "exposure_start",
    ] = df.loc[
        resp_only,
        "resp_before_msk",
    ]

    df.loc[
        mental_only,
        "exposure_start",
    ] = df.loc[
        mental_only,
        "mental_before_msk",
    ]

    df.loc[
        both,
        "exposure_start",
    ] = df.loc[
        both,
        [
            "resp_before_msk",
            "mental_before_msk",
        ],
    ].min(axis=1)

    df["time_to_msk_days"] = (
        df["first_msk"]
        - df["exposure_start"]
    ).dt.days

    df.loc[
        df["time_to_msk_days"] < 0,
        "time_to_msk_days",
    ] = np.nan

    return df


def build_analysis_cohorts(
    df_msk_all: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Return:
      1. Mental only/Both, all ages
      2. Mental only/Both, age >= 18
      3. Final cohort, age 18-102
    """
    analysis_all = df_msk_all[
        df_msk_all["group"].isin(
            ["Mental only", "Both"]
        )
    ].copy()

    age = pd.to_numeric(
        analysis_all["Age_at_mental"],
        errors="coerce",
    )

    analysis_18plus = analysis_all[
        age >= 18
    ].copy()

    analysis_adult = analysis_all[
        age.between(
            18,
            102,
            inclusive="both",
        )
    ].copy()

    return (
        analysis_all,
        analysis_18plus,
        analysis_adult,
    )


def build_cohort_flow_table(
    pre_patient: pd.DataFrame,
    wide: pd.DataFrame,
    df_msk_all: pd.DataFrame,
    analysis_all: pd.DataFrame,
    analysis_adult: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final cohort flow table."""
    return pd.DataFrame([
        {
            "step": "Pre-MSK patient table",
            "cohort": "all_ages",
            "N_patients": (
                pre_patient["Patient_ID"].nunique()
            ),
        },
        {
            "step": "Wide first-date table",
            "cohort": "all_ages",
            "N_patients": (
                wide["Patient_ID"].nunique()
            ),
        },
        {
            "step": "Patients with first MSK",
            "cohort": "all_ages",
            "N_patients": (
                df_msk_all["Patient_ID"].nunique()
            ),
        },
        {
            "step": "Analysis cohort (Mental only/Both)",
            "cohort": "all_ages",
            "N_patients": (
                analysis_all["Patient_ID"].nunique()
            ),
        },
        {
            "step": "Final cohort (Mental only/Both, age 18-102)",
            "cohort": "age_18_102",
            "N_patients": (
                analysis_adult["Patient_ID"].nunique()
            ),
        },
    ])


def build_group_count_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize Mental only/Both patient counts."""
    counts = df["group"].value_counts(
        dropna=False
    )

    percentages = (
        df["group"].value_counts(
            normalize=True,
            dropna=False,
        )
        * 100
    )

    return pd.DataFrame({
        "group": counts.index.astype(str),
        "N_patients": counts.values,
        "Percent": percentages.round(2).values,
    })