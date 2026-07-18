from __future__ import annotations

from pathlib import Path

import pandas as pd

from final_cohort.cohort_preparation.icd_naming import (
    build_icd9_name_map_from_encounter,
    build_icd3_name_map_from_icd9_map,
    format_icd9_list_with_names,
)
from final_cohort.cohort_preparation.pre_msk_processing import (
    filter_cohort,
    build_pre_msk_encounter,
    rebuild_pre_msk_patient_table,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    data_dir = root / "data" / "output"

    # Temporary input from the original EDA.
    # This dependency will be removed only after validation.
    initial_eda_table = (
        root
        / "eda"
        / "results"
        / "tables"
        / "analysis_df.csv"
    )

    encounter_path = (
        data_dir
        / "C4MEncounterDiagnosis_clean_1_31.csv"
    )

    output_dir = (
        root
        / "final_cohort"
        / "results"
        / "tables"
        / "cohort_preparation"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("Building pre-MSK cohort preparation tables")
    print("=" * 60)

    if not initial_eda_table.exists():
        raise FileNotFoundError(
            f"Initial EDA table not found: {initial_eda_table}"
        )

    if not encounter_path.exists():
        raise FileNotFoundError(
            f"Encounter file not found: {encounter_path}"
        )

    analysis_df = pd.read_csv(
        initial_eda_table,
        dtype=str,
        low_memory=False,
    )

    encounter = pd.read_csv(
        encounter_path,
        dtype=str,
        low_memory=False,
    )

    if (
        "DiagnosisCode_calc" in encounter.columns
        and "icd9" not in encounter.columns
    ):
        encounter = encounter.rename(
            columns={
                "DiagnosisCode_calc": "icd9"
            }
        )

    # Preserve the original cohort-selection definition.
    cohort_index = filter_cohort(
        analysis_df
    )

    print("\nSelected cohort index:")
    print("Rows:", len(cohort_index))
    print(
        "Patients:",
        cohort_index["Patient_ID"].nunique(),
    )

    # Build diagnosis-name mappings.
    icd9_to_name = (
        build_icd9_name_map_from_encounter(
            encounter
        )
    )

    icd3_to_name = (
        build_icd3_name_map_from_icd9_map(
            icd9_to_name
        )
    )

    # Build encounter-level pre-MSK table.
    pre_long = build_pre_msk_encounter(
        encounter_df=encounter,
        cohort_df=cohort_index,
        icd9_to_name=icd9_to_name,
    )

    pre_long_path = (
        output_dir
        / "analysis_long_pre_msk_only.csv"
    )

    pre_long.to_csv(
        pre_long_path,
        index=False,
    )

    print("\nPre-MSK encounter-level table:")
    print("Rows:", len(pre_long))
    print(
        "Patients:",
        pre_long["Patient_ID"].nunique(),
    )

    # Build patient-level pre-MSK table.
    patient_table = (
        rebuild_pre_msk_patient_table(
            pre_long
        )
    )

    patient_table["icd9 list (named)"] = (
        patient_table["icd9 list"]
        .apply(
            lambda value: format_icd9_list_with_names(
                value,
                icd9_map=icd9_to_name,
                icd3_map=icd3_to_name,
                keep_unknown=False,
            )
        )
    )

    patient_table_path = (
        output_dir
        / "analysis_df2_pre_msk_named.csv"
    )

    patient_table.to_csv(
        patient_table_path,
        index=False,
    )

    print("\nPre-MSK patient-level table:")
    print("Rows:", len(patient_table))
    print(
        "Patients:",
        patient_table["Patient_ID"].nunique(),
    )

    print("\nSaved:")
    print(pre_long_path)
    print(patient_table_path)


if __name__ == "__main__":
    main()