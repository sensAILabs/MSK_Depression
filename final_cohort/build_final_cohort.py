from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from final_cohort.cohort_preparation.final_cohort_processing import (
    build_first_system_dates_from_pre_long,
    pivot_first_dates,
    add_first_msk_from_pre_patient_table,
    build_first_msk_subgroup_from_encounter,
    add_msk_exposure_features,
    build_analysis_cohorts,
    build_cohort_flow_table,
    build_group_count_table,
)


ANALYSIS_COLUMNS = [
    "Patient_ID",
    "Sex",
    "BirthYear",
    "Age_at_mental",
    "first_resp",
    "first_mental",
    "first_msk",
    "first_msk_icd9",
    "first_msk_icd9_3",
    "first_msk_subgroup",
    "first_msk_course",
    "resp_before_msk",
    "mental_before_msk",
    "exposure_start",
    "time_to_msk_days",
    "group",
    "has_resp",
    "has_mental",
]


def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


def main() -> None:
    project_root = Path(
        os.environ.get(
            "BIOSTAT_ROOT",
            str(Path(__file__).resolve().parents[1]),
        )
    )

    preparation_dir = (
        project_root
        / "final_cohort"
        / "results"
        / "tables"
        / "cohort_preparation"
    )

    pre_long_path = (
        preparation_dir
        / "analysis_long_pre_msk_only.csv"
    )

    pre_patient_path = (
        preparation_dir
        / "analysis_df2_pre_msk_named.csv"
    )

    # Keep the same patient input used by the old pipeline during validation.
    patient_path = (
        project_root
        / "data"
        / "C4MPatient.csv"
    )

    encounter_path = (
        project_root
        / "data"
        / "output"
        / "C4MEncounterDiagnosis_clean_1_31.csv"
    )

    output_dir = (
        project_root
        / "final_cohort"
        / "results"
        / "tables"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        pre_long_path,
        pre_patient_path,
        patient_path,
        encounter_path,
    ]:
        check_file(path)

    print("=" * 70)
    print("BUILD FINAL COHORT")
    print("=" * 70)

    pre_long = pd.read_csv(
        pre_long_path,
        dtype=str,
        low_memory=False,
    )

    pre_patient = pd.read_csv(
        pre_patient_path,
        dtype=str,
        low_memory=False,
    )

    patient = pd.read_csv(
        patient_path,
        dtype=str,
        sep="|",
        low_memory=False,
    )

    encounter = pd.read_csv(
        encounter_path,
        dtype=str,
        low_memory=False,
    )

    for frame in [
        pre_long,
        pre_patient,
        patient,
        encounter,
    ]:
        frame["Patient_ID"] = (
            frame["Patient_ID"]
            .astype(str)
            .str.strip()
        )

    cohort_ids = set(
        pre_patient["Patient_ID"].unique()
    )

    patient = patient[
        patient["Patient_ID"].isin(cohort_ids)
    ].copy()

    encounter = encounter[
        encounter["Patient_ID"].isin(cohort_ids)
    ].copy()

    # ---------------------------------------------------------
    # Step 1: first disease-system dates
    # ---------------------------------------------------------
    first_system_dates = (
        build_first_system_dates_from_pre_long(
            pre_long
        )
    )

    wide = pivot_first_dates(
        first_system_dates
    )

    wide = add_first_msk_from_pre_patient_table(
        wide,
        pre_patient,
    )

    # ---------------------------------------------------------
    # Step 2: first MSK subgroup
    # ---------------------------------------------------------
    first_msk_subgroup = (
        build_first_msk_subgroup_from_encounter(
            encounter,
            pre_patient,
        )
    )

    wide = wide.merge(
        first_msk_subgroup,
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Step 3: demographics and exposure dates
    # ---------------------------------------------------------
    df_msk_all = add_msk_exposure_features(
        wide,
        patient,
    )

    # ---------------------------------------------------------
    # Step 4: final cohort inclusion criteria
    # ---------------------------------------------------------
    (
        analysis_all,
        analysis_18plus,
        analysis_adult,
    ) = build_analysis_cohorts(
        df_msk_all
    )

    print("\nCohort counts")
    print(
        "Pre-MSK patient table:",
        pre_patient["Patient_ID"].nunique(),
    )
    print(
        "Wide first-date table:",
        wide["Patient_ID"].nunique(),
    )
    print(
        "Patients with first MSK:",
        df_msk_all["Patient_ID"].nunique(),
    )
    print(
        "Mental only/Both, all ages:",
        analysis_all["Patient_ID"].nunique(),
    )
    print(
        "Mental only/Both, age >=18:",
        analysis_18plus["Patient_ID"].nunique(),
    )
    print(
        "Final cohort, age 18-102:",
        analysis_adult["Patient_ID"].nunique(),
    )

    if analysis_adult.empty:
        raise ValueError(
            "Final cohort is empty."
        )

    if analysis_adult["Patient_ID"].duplicated().any():
        raise ValueError(
            "Final cohort contains duplicate Patient_ID values."
        )

    # ---------------------------------------------------------
    # Step 5: export final cohort
    # ---------------------------------------------------------
    available_columns = [
        column
        for column in ANALYSIS_COLUMNS
        if column in analysis_adult.columns
    ]

    final_cohort = (
        analysis_adult[
            available_columns
        ]
        .copy()
        .sort_values("Patient_ID")
        .reset_index(drop=True)
    )

    final_cohort_path = (
        output_dir
        / "analysis_cohort_final.csv"
    )

    final_cohort.to_csv(
        final_cohort_path,
        index=False,
    )

    cohort_flow = build_cohort_flow_table(
        pre_patient=pre_patient,
        wide=wide,
        df_msk_all=df_msk_all,
        analysis_all=analysis_all,
        analysis_adult=analysis_adult,
    )

    cohort_flow_path = (
        output_dir
        / "cohort_flow.csv"
    )

    cohort_flow.to_csv(
        cohort_flow_path,
        index=False,
    )

    group_counts = build_group_count_table(
        analysis_adult
    )

    group_count_path = (
        output_dir
        / "group_count_final_cohort.csv"
    )

    group_counts.to_csv(
        group_count_path,
        index=False,
    )

    print("\nSaved:")
    print(final_cohort_path)
    print(cohort_flow_path)
    print(group_count_path)

    print("\nFinal cohort shape:", final_cohort.shape)
    print(
        "Unique patients:",
        final_cohort["Patient_ID"].nunique(),
    )


if __name__ == "__main__":
    main()