"""
This script:
- reads and retains all columns from each raw table;
- processes one complete table at a time;
- uses pandas' faster C parser;
- standardizes headers, whitespace, missing values, and selected categories;
- converts date and configured numeric columns;
- removes exact duplicate rows;
- reports duplicated primary keys and data-quality issues;
- saves cleaned patient, diagnosis, and medication tables.
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from mental_msk_prediction.config.paths import (
    CLEANED_DIAGNOSIS_FILE,
    CLEANED_MEDICATION_FILE,
    CLEANED_PATIENT_FILE,
    RAW_DIAGNOSIS_FILE,
    RAW_MEDICATION_FILE,
    RAW_PATIENT_FILE,
    TABLES_DIR,
    create_project_directories,
    validate_required_raw_files,
)

# General settings
SEPARATOR = "|"
REPORT_DIR = TABLES_DIR / "data_cleaning"

# Unknown/UNK are treated as missing rather than valid categories.
NA_VALUES = [
    "",
    " ",
    "NA",
    "N/A",
    "NAN",
    "NaN",
    "NONE",
    "None",
    "NULL",
    "null",
    "UNK",
    "UNKNOWN",
    "Unknown",
    "<NA>",
    ".",
]

SEX_MAP = {
    "F": "Female",
    "FE": "Female",
    "FEM": "Female",
    "FEMALE": "Female",
    "WOMAN": "Female",
    "M": "Male",
    "MA": "Male",
    "MAL": "Male",
    "MALE": "Male",
    "MAN": "Male",
}

CODE_TYPE_MAP = {
    "ICD9": "ICD9",
    "ICD-9": "ICD9",
    "ICD 9": "ICD9",
    "ICD9CM": "ICD9",
    "ICD-9-CM": "ICD9",
    "ATC": "ATC",
    "DIN": "DIN",
    "MULTUM": "MULTUM",
}


TABLE_CONFIGS: dict[str, dict[str, Any]] = {
    "patient": {
        "raw_path": RAW_PATIENT_FILE,
        "cleaned_path": CLEANED_PATIENT_FILE,
        "primary_key": "Patient_ID",
        "category_maps": {
            "Sex": SEX_MAP,
        },
        "uppercase_columns": [
            "Patient_ID",
        ],
        "numeric_columns": [
            "BirthYear",
            "YearOfBirth",
        ],
        "nonnegative_columns": [
            "BirthYear",
            "YearOfBirth",
        ],
    },
    "diagnosis": {
        "raw_path": RAW_DIAGNOSIS_FILE,
        "cleaned_path": CLEANED_DIAGNOSIS_FILE,
        "primary_key": "EncounterDiagnosis_ID",
        "category_maps": {
            "DiagnosisCodeType_orig": CODE_TYPE_MAP,
            "DiagnosisCodeType_calc": CODE_TYPE_MAP,
        },
        "uppercase_columns": [
            "EncounterDiagnosis_ID",
            "Patient_ID",
            "Encounter_ID",
            "DiagnosisCodeType_orig",
            "DiagnosisCodeType_calc",
            "DiagnosisCode_orig",
            "DiagnosisCode_calc",
        ],
        "numeric_columns": [],
        "nonnegative_columns": [],
    },
    "medication": {
        "raw_path": RAW_MEDICATION_FILE,
        "cleaned_path": CLEANED_MEDICATION_FILE,
        "primary_key": "Medication_ID",
        "category_maps": {
            "CodeType_orig": CODE_TYPE_MAP,
            "CodeType_calc": CODE_TYPE_MAP,
        },
        "uppercase_columns": [
            "Medication_ID",
            "Patient_ID",
            "Encounter_ID",
            "DIN",
            "CodeType_orig",
            "CodeType_calc",
            "Code_orig",
            "Code_calc",
            "UnitOfMeasure",
            "DurationUnit",
        ],
        "numeric_columns": [
            "Strength",
            "Dose",
            "DurationCount",
            "DispensedCount",
        ],
        "nonnegative_columns": [
            "Strength",
            "Dose",
            "DurationCount",
            "DispensedCount",
        ],
    },
}


# Reading
def read_raw_table(
    file_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Read all columns from one pipe-delimited CPCSSN table.

    The C parser is used for speed. Malformed rows are skipped with a parser
    warning. The script does not guess or reconstruct corrupted clinical rows.
    """
    with warnings.catch_warnings(record=True) as parser_warnings:
        warnings.simplefilter("always")

        dataframe = pd.read_csv(
            file_path,
            sep=SEPARATOR,
            dtype=str,
            engine="c",
            keep_default_na=True,
            na_values=NA_VALUES,
            encoding="utf-8-sig",
            on_bad_lines="warn",
            low_memory=True,
            memory_map=True,
        )

    dataframe.columns = [
        str(column).strip().strip('"').strip()
        for column in dataframe.columns
    ]

    warning_messages = [
        str(item.message)
        for item in parser_warnings
    ]

    return dataframe, warning_messages


# Cleaning helpers
def normalize_text_column(series: pd.Series) -> pd.Series:
    """
    Normalize whitespace without converting missing values to literal text.
    """
    nonmissing = series.notna()

    if nonmissing.any():
        series.loc[nonmissing] = (
            series.loc[nonmissing]
            .astype(str)
            .str.replace(r"[\r\n\t]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        series.loc[nonmissing & series.eq("")] = pd.NA

    return series


def normalize_text_columns_in_place(
    dataframe: pd.DataFrame,
) -> None:
    """
    Normalize object/string columns one at a time to reduce peak memory.
    """
    for column in dataframe.columns:
        if (
            pd.api.types.is_object_dtype(dataframe[column])
            or pd.api.types.is_string_dtype(dataframe[column])
        ):
            dataframe[column] = normalize_text_column(
                dataframe[column]
            )


def uppercase_columns_in_place(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Uppercase selected identifier and coding fields.
    """
    for column in columns:
        if column not in dataframe.columns:
            continue

        nonmissing = dataframe[column].notna()

        if nonmissing.any():
            dataframe.loc[nonmissing, column] = (
                dataframe.loc[nonmissing, column]
                .astype(str)
                .str.upper()
            )


def standardize_categories_in_place(
    dataframe: pd.DataFrame,
    category_maps: dict[str, dict[str, str]],
    table_name: str,
) -> pd.DataFrame:
    """
    Standardize selected categorical columns and report unmapped values.

    Missing values remain missing. Non-missing values not included in a map
    are retained in normalized uppercase form and written to a review report.
    """
    review_rows: list[dict[str, Any]] = []

    for column, mapping in category_maps.items():
        if column not in dataframe.columns:
            continue

        normalized = dataframe[column].copy()
        nonmissing = normalized.notna()

        normalized.loc[nonmissing] = (
            normalized.loc[nonmissing]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        mapped = normalized.map(mapping)
        unmapped_mask = normalized.notna() & mapped.isna()

        if unmapped_mask.any():
            counts = normalized.loc[
                unmapped_mask
            ].value_counts()

            for value, count in counts.items():
                review_rows.append(
                    {
                        "table": table_name,
                        "column": column,
                        "unmapped_value": value,
                        "n_rows": int(count),
                    }
                )

        dataframe[column] = mapped.fillna(normalized)

        del normalized, mapped, unmapped_mask
        gc.collect()

    return pd.DataFrame(review_rows)


def convert_dates_in_place(
    dataframe: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    """
    Convert columns containing 'date' and report invalid values.
    """
    current_year = pd.Timestamp.today().year
    issue_frames: list[pd.DataFrame] = []

    date_columns = [
        column
        for column in dataframe.columns
        if "date" in column.lower()
    ]

    for column in date_columns:
        original = dataframe[column].copy()
        converted = pd.to_datetime(
            original,
            errors="coerce",
        )

        parse_failure = original.notna() & converted.isna()
        impossible_year = (
            converted.notna()
            & (
                converted.dt.year.lt(1900)
                | converted.dt.year.gt(current_year)
            )
        )

        issue_mask = parse_failure | impossible_year

        if issue_mask.any():
            issue_type = pd.Series(
                "invalid_calendar_year",
                index=dataframe.index[issue_mask],
                dtype="object",
            )

            issue_type.loc[
                parse_failure.loc[issue_mask]
            ] = "date_parse_failure"

            issue_frames.append(
                pd.DataFrame(
                    {
                        "table": table_name,
                        "column": column,
                        "row_index": dataframe.index[issue_mask],
                        "original_value": original.loc[issue_mask],
                        "issue": issue_type,
                    }
                )
            )

        converted.loc[impossible_year] = pd.NaT
        dataframe[column] = converted

        del original, converted
        gc.collect()

    if issue_frames:
        return pd.concat(
            issue_frames,
            ignore_index=True,
        )

    return pd.DataFrame(
        columns=[
            "table",
            "column",
            "row_index",
            "original_value",
            "issue",
        ]
    )


def convert_numeric_in_place(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    nonnegative_columns: list[str],
    table_name: str,
) -> pd.DataFrame:
    """
    Convert configured numeric fields and report invalid values.
    """
    issue_frames: list[pd.DataFrame] = []
    nonnegative_set = set(nonnegative_columns)

    for column in numeric_columns:
        if column not in dataframe.columns:
            continue

        original = dataframe[column].copy()
        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        parse_failure = original.notna() & converted.isna()

        if column in nonnegative_set:
            negative_value = converted.lt(0)
        else:
            negative_value = pd.Series(
                False,
                index=dataframe.index,
            )

        issue_mask = parse_failure | negative_value

        if issue_mask.any():
            issue_type = pd.Series(
                "negative_value",
                index=dataframe.index[issue_mask],
                dtype="object",
            )

            issue_type.loc[
                parse_failure.loc[issue_mask]
            ] = "numeric_parse_failure"

            issue_frames.append(
                pd.DataFrame(
                    {
                        "table": table_name,
                        "column": column,
                        "row_index": dataframe.index[issue_mask],
                        "original_value": original.loc[issue_mask],
                        "issue": issue_type,
                    }
                )
            )

        converted.loc[negative_value] = pd.NA
        dataframe[column] = converted

        del original, converted
        gc.collect()

    if issue_frames:
        return pd.concat(
            issue_frames,
            ignore_index=True,
        )

    return pd.DataFrame(
        columns=[
            "table",
            "column",
            "row_index",
            "original_value",
            "issue",
        ]
    )


def remove_exact_duplicates_in_place(
    dataframe: pd.DataFrame,
    table_name: str,
) -> tuple[pd.DataFrame, int]:
    """
    Remove completely identical rows.
    """
    duplicate_mask = dataframe.duplicated(
        keep="first"
    )
    duplicate_count = int(duplicate_mask.sum())

    examples = dataframe.loc[
        duplicate_mask
    ].head(10_000).copy()

    if not examples.empty:
        examples.insert(
            0,
            "table",
            table_name,
        )

    if duplicate_count:
        dataframe.drop(
            index=dataframe.index[duplicate_mask],
            inplace=True,
        )
        dataframe.reset_index(
            drop=True,
            inplace=True,
        )

    del duplicate_mask
    gc.collect()

    return examples, duplicate_count


def duplicated_primary_key_report(
    dataframe: pd.DataFrame,
    primary_key: str,
    table_name: str,
) -> tuple[pd.DataFrame, int]:
    """
    Report, but do not automatically remove, duplicated primary-key records.
    """
    if primary_key not in dataframe.columns:
        return pd.DataFrame(), 0

    duplicate_mask = (
        dataframe[primary_key].notna()
        & dataframe[primary_key].duplicated(
            keep=False
        )
    )
    duplicate_count = int(duplicate_mask.sum())

    examples = dataframe.loc[
        duplicate_mask
    ].head(10_000).copy()

    if not examples.empty:
        examples.insert(
            0,
            "table",
            table_name,
        )

    del duplicate_mask
    gc.collect()

    return examples, duplicate_count


def missingness_summary(
    dataframe: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    """
    Summarize missing count and percentage for every retained column.
    """
    total_rows = len(dataframe)

    report = pd.DataFrame(
        {
            "table": table_name,
            "column": dataframe.columns,
            "n_missing": [
                int(dataframe[column].isna().sum())
                for column in dataframe.columns
            ],
        }
    )

    report["total_rows"] = total_rows
    report["percent_missing"] = (
        report["n_missing"] / total_rows * 100
        if total_rows
        else 0.0
    )

    return report.sort_values(
        ["percent_missing", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)


def save_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a cleaned table as a comma-delimited CSV.
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


def save_report(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """
    Save one quality-control report.
    """
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        REPORT_DIR / filename,
        index=False,
    )


# One-table cleaning pipeline
def clean_one_table(
    table_name: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Read, clean, save, and release one complete table.
    """
    raw_path: Path = config["raw_path"]
    cleaned_path: Path = config["cleaned_path"]
    primary_key: str = config["primary_key"]

    print(f"\n{'=' * 80}")
    print(f"Cleaning {table_name}")
    print(f"{'=' * 80}")
    print(f"Input: {raw_path}")

    dataframe, parser_warnings = read_raw_table(
        raw_path,
    )

    raw_rows_read = len(dataframe)

    print(f"Rows read: {raw_rows_read:,}")
    print(f"Columns read: {len(dataframe.columns):,}")

    normalize_text_columns_in_place(
        dataframe,
    )

    unmapped_categories = standardize_categories_in_place(
        dataframe,
        config["category_maps"],
        table_name,
    )

    uppercase_columns_in_place(
        dataframe,
        config["uppercase_columns"],
    )

    date_issues = convert_dates_in_place(
        dataframe,
        table_name,
    )

    numeric_issues = convert_numeric_in_place(
        dataframe,
        config["numeric_columns"],
        config["nonnegative_columns"],
        table_name,
    )

    (
        exact_duplicate_examples,
        exact_duplicate_count,
    ) = remove_exact_duplicates_in_place(
        dataframe,
        table_name,
    )

    (
        duplicated_key_examples,
        duplicated_key_count,
    ) = duplicated_primary_key_report(
        dataframe,
        primary_key,
        table_name,
    )

    missing_report = missingness_summary(
        dataframe,
        table_name,
    )

    summary = pd.DataFrame(
        {
            "table": [table_name],
            "raw_rows_read": [raw_rows_read],
            "cleaned_rows": [len(dataframe)],
            "columns_retained": [
                len(dataframe.columns)
            ],
            "exact_duplicate_rows_removed": [
                exact_duplicate_count
            ],
            "duplicated_primary_key_rows_retained": [
                duplicated_key_count
            ],
            "missing_primary_key": [
                int(dataframe[primary_key].isna().sum())
                if primary_key in dataframe.columns
                else pd.NA
            ],
            "unique_patients": [
                dataframe["Patient_ID"].nunique(
                    dropna=True
                )
                if "Patient_ID" in dataframe.columns
                else pd.NA
            ],
            "parser_warning_count": [
                len(parser_warnings)
            ],
        }
    )

    # Save the cleaned table before moving to the next dataset.
    save_dataframe(
        dataframe,
        cleaned_path,
    )

    # Save reports immediately.
    save_report(
        summary,
        f"{table_name}_summary.csv",
    )
    save_report(
        missing_report,
        f"{table_name}_missingness.csv",
    )
    save_report(
        unmapped_categories,
        f"{table_name}_unmapped_categories.csv",
    )
    save_report(
        date_issues,
        f"{table_name}_date_issues.csv",
    )
    save_report(
        numeric_issues,
        f"{table_name}_numeric_issues.csv",
    )
    save_report(
        exact_duplicate_examples,
        f"{table_name}_exact_duplicate_examples.csv",
    )
    save_report(
        duplicated_key_examples,
        f"{table_name}_duplicated_primary_key_examples.csv",
    )
    save_report(
        pd.DataFrame(
            {
                "table": table_name,
                "warning": parser_warnings,
            }
        ),
        f"{table_name}_parser_warnings.csv",
    )

    print(summary.to_string(index=False))
    print(f"Saved cleaned table: {cleaned_path}")

    # Explicitly release the large table before loading the next one.
    del dataframe
    gc.collect()

    return summary


# Main entry point
def main() -> None:
    """
    Clean patient, diagnosis, and medication tables in sequence.
    """
    create_project_directories()
    validate_required_raw_files()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[pd.DataFrame] = []

    for table_name, config in TABLE_CONFIGS.items():
        summary = clean_one_table(
            table_name,
            config,
        )
        summaries.append(summary)
        gc.collect()

    overall_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    save_report(
        overall_summary,
        "summary.csv",
    )

    print(f"\n{'=' * 80}")
    print("Data cleaning completed")
    print(f"{'=' * 80}")
    print(overall_summary.to_string(index=False))
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
