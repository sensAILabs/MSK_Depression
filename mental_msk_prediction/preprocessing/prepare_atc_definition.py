"""
Prepare ATC Level 3 study definitions using medication records from the
final mental-to-MSK cohort.

This script performs deterministic ATC quality control and prepares the
medication-definition tables used before feature engineering.

It:
1. restricts the cleaned medication table to final-cohort patients;
2. uses CodeType_calc == "ATC" and Code_calc as the ATC source;
3. summarizes all observed ATC Level 3 groups;
4. summarizes the prespecified study ATC groups defined in
   clinical_definitions.py;
5. lists the complete seven-character ATC codes and Name_calc values within
   each selected study group;
6. reports complete ATC codes linked to multiple Name_calc values.

Definitions
-----------
`total_drugs` is the number of distinct complete seven-character ATC codes
within an ATC Level 3 study group. `Drug_name` is taken directly from the
cleaned CPCSSN Name_calc field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mental_msk_prediction.config.clinical_definitions import (
    ATC_LEVEL3_DEFINITIONS,
)
from mental_msk_prediction.config.paths import (
    CLEANED_MEDICATION_FILE,
    RESULTS_DATA_DIR,
    TABLES_DIR,
    create_project_directories,
)


# Paths
FINAL_COHORT_FILE = (RESULTS_DATA_DIR/ "cohort" / "final_cohort.csv")
OUTPUT_DIR = (TABLES_DIR/ "medication_atc_qc"/ "final_cohort")

# Validation and normalization
def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    table_name: str,
) -> None:
    """
    Confirm that required columns are present.
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


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize whitespace while preserving missing values.
    """
    values = (
        series.astype("string")
        .str.replace(r"[\r\n\t]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    missing_mask = (
        values.str.upper()
        .isin(
            {
                "",
                "NAN",
                "NONE",
                "NULL",
                "UNKNOWN",
                "UNK",
                "<NA>",
            }
        )
    )

    return values.mask(
        missing_mask,
        pd.NA,
    )


def normalize_atc_code(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize calculated ATC codes to uppercase alphanumeric text.
    """
    values = normalize_text(
        series
    ).str.upper()

    values = values.str.replace(
        r"[^A-Z0-9]",
        "",
        regex=True,
    )

    return values.replace(
        "",
        pd.NA,
    )


def complete_atc_mask(
    series: pd.Series,
) -> pd.Series:
    """
    Identify complete seven-character ATC codes.

    Example
    -------
    N06AB06
    """
    return series.str.fullmatch(
        r"[A-Z]\d{2}[A-Z]{2}\d{2}",
        na=False,
    )


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """
    Save one output table.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_DIR / filename,
        index=False,
    )

# ATC preparation
def prepare_medication_atc(
    medication: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare deterministic ATC fields from the cleaned medication table.
    """
    df = medication.copy()

    df["Patient_ID"] = normalize_text(
        df["Patient_ID"]
    )

    df["Drug_name"] = normalize_text(
        df["Name_calc"]
    )

    df["CodeType_calc_normalized"] = (
        normalize_text(
            df["CodeType_calc"]
        )
        .str.upper()
    )

    df["ATC_code"] = normalize_atc_code(
        df["Code_calc"]
    )

    is_atc = (
        df["CodeType_calc_normalized"]
        .eq("ATC")
    )

    # Only calculated codes explicitly identified as ATC are retained.
    df.loc[
        ~is_atc,
        "ATC_code",
    ] = pd.NA

    df["ATC_Level3"] = (
        df["ATC_code"]
        .str.slice(0, 4)
    )

    valid_level3 = (
        df["ATC_Level3"]
        .str.fullmatch(
            r"[A-Z]\d{2}[A-Z]",
            na=False,
        )
    )

    df.loc[
        ~valid_level3,
        "ATC_Level3",
    ] = pd.NA

    is_complete = complete_atc_mask(
        df["ATC_code"]
    )

    df["ATC_Level7"] = (
        df["ATC_code"]
        .where(is_complete)
    )

    return df

# All-ATC QC
def build_atc_level3_frequency(
    medication: pd.DataFrame,
    total_cohort_patients: int,
) -> pd.DataFrame:
    """
    Summarize every observed ATC Level 3 group.

    Patient percentages use the complete final cohort as the denominator,
    including patients without medication records.
    """
    valid = medication.dropna(
        subset=["ATC_Level3"]
    ).copy()

    row_counts = (
        valid["ATC_Level3"]
        .value_counts()
        .rename_axis("ATC_Level3")
        .reset_index(
            name="n_medication_rows"
        )
    )

    patient_counts = (
        valid[
            [
                "ATC_Level3",
                "Patient_ID",
            ]
        ]
        .dropna(
            subset=["Patient_ID"]
        )
        .drop_duplicates()
        .groupby(
            "ATC_Level3",
            observed=True,
        )["Patient_ID"]
        .nunique()
        .rename(
            "n_unique_patients"
        )
        .reset_index()
    )

    summary = row_counts.merge(
        patient_counts,
        on="ATC_Level3",
        how="left",
        validate="one_to_one",
    )

    summary["n_unique_patients"] = (
        summary["n_unique_patients"]
        .fillna(0)
        .astype(int)
    )

    valid_atc_rows = len(valid)

    summary["percent_of_atc_rows"] = (
        summary["n_medication_rows"]
        / valid_atc_rows
        * 100
        if valid_atc_rows
        else 0.0
    )

    summary["percent_of_final_cohort"] = (
        summary["n_unique_patients"]
        / total_cohort_patients
        * 100
        if total_cohort_patients
        else 0.0
    )

    summary[
        "total_final_cohort_patients"
    ] = total_cohort_patients

    return (
        summary.sort_values(
            [
                "n_unique_patients",
                "n_medication_rows",
                "ATC_Level3",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


# Study ATC definition outputs
def build_study_atc_drug_list(
    medication: pd.DataFrame,
) -> pd.DataFrame:
    """
    List complete ATC codes and Name_calc values for selected study groups.

    One row represents one distinct:
        ATC Level 3 + ATC Level 7 + Name_calc combination.
    """
    definition_table = pd.DataFrame(
        [
            {
                "ATC_Level3": code,
                "official_name": definition[
                    "official_name"
                ],
                "study_group": definition[
                    "study_group"
                ],
            }
            for code, definition
            in ATC_LEVEL3_DEFINITIONS.items()
        ]
    )

    selected = medication[
        medication["ATC_Level3"].isin(
            definition_table["ATC_Level3"]
        )
    ].copy()

    drug_list = (
        selected[
            [
                "ATC_Level3",
                "ATC_Level7",
                "Drug_name",
            ]
        ]
        .drop_duplicates()
        .merge(
            definition_table,
            on="ATC_Level3",
            how="left",
            validate="many_to_one",
        )
    )

    columns = [
        "ATC_Level3",
        "official_name",
        "study_group",
        "ATC_Level7",
        "Drug_name",
    ]

    return (
        drug_list[columns]
        .sort_values(
            [
                "study_group",
                "ATC_Level3",
                "ATC_Level7",
                "Drug_name",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_study_atc_summary(
    medication: pd.DataFrame,
    atc_frequency: pd.DataFrame,
    study_drug_list: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the final selected ATC Level 3 medication groups.

    `total_drugs` counts distinct complete seven-character ATC codes.
    """
    definitions = pd.DataFrame(
        [
            {
                "ATC_Level3": code,
                "official_name": definition[
                    "official_name"
                ],
                "study_group": definition[
                    "study_group"
                ],
            }
            for code, definition
            in ATC_LEVEL3_DEFINITIONS.items()
        ]
    )

    drug_counts = (
        study_drug_list
        .dropna(
            subset=["ATC_Level7"]
        )
        .groupby(
            "ATC_Level3",
            observed=True,
        )["ATC_Level7"]
        .nunique()
        .rename("total_drugs")
        .reset_index()
    )

    name_counts = (
        study_drug_list
        .dropna(
            subset=["Drug_name"]
        )
        .groupby(
            "ATC_Level3",
            observed=True,
        )["Drug_name"]
        .nunique()
        .rename(
            "total_name_calc_values"
        )
        .reset_index()
    )

    frequency_columns = [
        "ATC_Level3",
        "n_medication_rows",
        "n_unique_patients",
        "percent_of_final_cohort",
        "total_final_cohort_patients",
    ]

    summary = (
        definitions
        .merge(
            atc_frequency[
                frequency_columns
            ],
            on="ATC_Level3",
            how="left",
            validate="one_to_one",
        )
        .merge(
            drug_counts,
            on="ATC_Level3",
            how="left",
            validate="one_to_one",
        )
        .merge(
            name_counts,
            on="ATC_Level3",
            how="left",
            validate="one_to_one",
        )
    )

    numeric_columns = [
        "n_medication_rows",
        "n_unique_patients",
        "percent_of_final_cohort",
        "total_final_cohort_patients",
        "total_drugs",
        "total_name_calc_values",
    ]

    for column in numeric_columns:
        summary[column] = (
            summary[column]
            .fillna(0)
        )

    integer_columns = [
        "n_medication_rows",
        "n_unique_patients",
        "total_final_cohort_patients",
        "total_drugs",
        "total_name_calc_values",
    ]

    for column in integer_columns:
        summary[column] = (
            summary[column]
            .astype(int)
        )

    return (
        summary.sort_values(
            [
                "study_group",
                "ATC_Level3",
            ]
        )
        .reset_index(drop=True)
    )



def build_atc_level7_name_qc(
    study_drug_list: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify complete ATC Level 7 codes associated with multiple Name_calc
    values.

    This is a naming-quality report only. The formal drug count remains the
    number of unique complete ATC Level 7 codes.
    """
    valid = study_drug_list.dropna(
        subset=["ATC_Level7"]
    ).copy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "ATC_Level3",
                "official_name",
                "study_group",
                "ATC_Level7",
                "n_name_calc_values",
                "Drug_names",
            ]
        )

    name_counts = (
        valid.dropna(
            subset=["Drug_name"]
        )
        .groupby(
            [
                "ATC_Level3",
                "official_name",
                "study_group",
                "ATC_Level7",
            ],
            observed=True,
        )["Drug_name"]
        .nunique()
        .rename("n_name_calc_values")
        .reset_index()
    )

    multiple_names = name_counts[
        name_counts["n_name_calc_values"] > 1
    ].copy()

    if multiple_names.empty:
        return pd.DataFrame(
            columns=[
                "ATC_Level3",
                "official_name",
                "study_group",
                "ATC_Level7",
                "n_name_calc_values",
                "Drug_names",
            ]
        )

    name_lists = (
        valid.merge(
            multiple_names[
                [
                    "ATC_Level3",
                    "ATC_Level7",
                ]
            ],
            on=[
                "ATC_Level3",
                "ATC_Level7",
            ],
            how="inner",
        )
        .dropna(
            subset=["Drug_name"]
        )
        .groupby(
            [
                "ATC_Level3",
                "ATC_Level7",
            ],
            observed=True,
        )["Drug_name"]
        .agg(
            lambda values: "; ".join(
                sorted(
                    {
                        str(value).strip()
                        for value in values
                        if str(value).strip()
                    }
                )
            )
        )
        .rename("Drug_names")
        .reset_index()
    )

    return (
        multiple_names.merge(
            name_lists,
            on=[
                "ATC_Level3",
                "ATC_Level7",
            ],
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            [
                "n_name_calc_values",
                "study_group",
                "ATC_Level3",
                "ATC_Level7",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


# Main
def main() -> None:
    """
    Run final-cohort ATC definition preparation.
    """
    create_project_directories()
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files = [
        FINAL_COHORT_FILE,
        CLEANED_MEDICATION_FILE,
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
            "Required files were not found:\n"
            f"{missing_text}"
        )

    if not ATC_LEVEL3_DEFINITIONS:
        raise ValueError(
            "ATC_LEVEL3_DEFINITIONS is empty in "
            "clinical_definitions.py."
        )

    print("\n" + "=" * 80)
    print("STEP 1 — LOAD FINAL COHORT")
    print("=" * 80)

    final_cohort = pd.read_csv(
        FINAL_COHORT_FILE,
        usecols=["Patient_ID"],
        dtype=str,
        low_memory=False,
        encoding="utf-8-sig",
    )

    final_cohort["Patient_ID"] = (
        normalize_text(
            final_cohort["Patient_ID"]
        )
    )

    final_cohort = (
        final_cohort
        .dropna(
            subset=["Patient_ID"]
        )
        .drop_duplicates(
            "Patient_ID"
        )
        .reset_index(drop=True)
    )

    total_cohort_patients = len(
        final_cohort
    )

    print(
        "Final cohort patients:",
        f"{total_cohort_patients:,}",
    )

    print("\n" + "=" * 80)
    print("STEP 2 — LOAD CLEANED MEDICATION DATA")
    print("=" * 80)

    medication = pd.read_csv(
        CLEANED_MEDICATION_FILE,
        dtype=str,
        low_memory=False,
        encoding="utf-8-sig",
    )

    require_columns(
        medication,
        [
            "Patient_ID",
            "Name_calc",
            "CodeType_calc",
            "Code_calc",
        ],
        "Cleaned medication table",
    )

    medication["Patient_ID"] = (
        normalize_text(
            medication["Patient_ID"]
        )
    )

    medication = medication.merge(
        final_cohort,
        on="Patient_ID",
        how="inner",
        validate="many_to_one",
    )

    patients_with_medication = (
        medication["Patient_ID"]
        .nunique(dropna=True)
    )

    print(
        "Medication rows for final cohort:",
        f"{len(medication):,}",
    )
    print(
        "Final-cohort patients with medication:",
        f"{patients_with_medication:,}",
    )

    print("\n" + "=" * 80)
    print("STEP 3 — PREPARE ATC FIELDS")
    print("=" * 80)

    medication = prepare_medication_atc(
        medication
    )

    atc_present = (
        medication["ATC_code"]
        .notna()
    )

    complete_atc = (
        medication["ATC_Level7"]
        .notna()
    )

    drug_name_present = (
        medication["Drug_name"]
        .notna()
    )

    print("\n" + "=" * 80)
    print("STEP 4 — SAVE QUALITY-CONTROL TABLES")
    print("=" * 80)

    summary = pd.DataFrame(
        [
            {
                "metric": "Total final-cohort patients",
                "value": total_cohort_patients,
            },
            {
                "metric": (
                    "Final-cohort patients with at least "
                    "one medication record"
                ),
                "value": patients_with_medication,
            },
            {
                "metric": (
                    "Medication rows among final-cohort patients"
                ),
                "value": len(medication),
            },
            {
                "metric": (
                    "Medication rows with calculated ATC code"
                ),
                "value": int(
                    atc_present.sum()
                ),
            },
            {
                "metric": (
                    "Medication rows with complete "
                    "seven-character ATC code"
                ),
                "value": int(
                    complete_atc.sum()
                ),
            },
            {
                "metric": (
                    "Medication rows with Name_calc present"
                ),
                "value": int(
                    drug_name_present.sum()
                ),
            },
            {
                "metric": (
                    "Unique observed ATC Level 3 groups"
                ),
                "value": int(
                    medication[
                        "ATC_Level3"
                    ].nunique(
                        dropna=True
                    )
                ),
            },
        ]
    )

    save_table(
        summary,
        "final_cohort_atc_summary.csv",
    )

    missingness = pd.DataFrame(
        {
            "column": [
                "Patient_ID",
                "Drug_name",
                "CodeType_calc_normalized",
                "ATC_code",
                "ATC_Level3",
                "ATC_Level7",
            ],
            "n_missing": [
                int(
                    medication[column]
                    .isna()
                    .sum()
                )
                for column in [
                    "Patient_ID",
                    "Drug_name",
                    "CodeType_calc_normalized",
                    "ATC_code",
                    "ATC_Level3",
                    "ATC_Level7",
                ]
            ],
        }
    )

    missingness[
        "total_medication_rows"
    ] = len(medication)

    missingness["percent_missing"] = (
        missingness["n_missing"]
        / len(medication)
        * 100
        if len(medication)
        else 0.0
    )

    save_table(
        missingness,
        "final_cohort_medication_missingness.csv",
    )

    exception_columns = [
        column
        for column in [
            "Patient_ID",
            "Name_calc",
            "CodeType_calc",
            "Code_calc",
            "ATC_code",
            "ATC_Level3",
            "ATC_Level7",
        ]
        if column in medication.columns
    ]

    save_table(
        medication.loc[
            drug_name_present
            & ~atc_present,
            exception_columns,
        ].head(10_000),
        "final_cohort_medication_without_atc_examples.csv",
    )

    print("\n" + "=" * 80)
    print("STEP 5 — SAVE ALL ATC LEVEL 3 FREQUENCIES")
    print("=" * 80)

    atc_frequency = (
        build_atc_level3_frequency(
            medication=medication,
            total_cohort_patients=(
                total_cohort_patients
            ),
        )
    )

    save_table(
        atc_frequency,
        "final_cohort_atc_level3_frequency.csv",
    )

    print(
        "Saved full ATC Level 3 frequency table "
        f"({len(atc_frequency):,} groups)."
    )

    print("\n" + "=" * 80)
    print("STEP 6 — BUILD STUDY ATC DEFINITION OUTPUTS")
    print("=" * 80)

    study_drug_list = (
        build_study_atc_drug_list(
            medication
        )
    )

    save_table(
        study_drug_list,
        "study_atc_drug_list.csv",
    )

    atc_level7_name_qc = (
        build_atc_level7_name_qc(
            study_drug_list
        )
    )
    
    study_summary = (
        build_study_atc_summary(
            medication=medication,
            atc_frequency=atc_frequency,
            study_drug_list=study_drug_list,
        )
    )

    save_table(
        study_summary,
        "study_atc_summary.csv",
    )

    display_columns = [
        "ATC_Level3",
        "official_name",
        "study_group",
        "n_unique_patients",
        "percent_of_final_cohort",
        "total_drugs",
    ]

    print("\nStudy ATC groups")
    print(
        study_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nATC Level 7 codes with multiple Name_calc values:",
        f"{len(atc_level7_name_qc):,}",
    )

    print("\n" + "=" * 80)
    print("ATC DEFINITION PREPARATION COMPLETED")
    print("=" * 80)
    print(
        f"Outputs saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
