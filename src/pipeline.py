# pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

# --- importfunctions ---
from INPUT_ORIGINAL_DATA_OUTPUT_GOODLINE_BADLINE import read_csv_with_badlines

from MERGE_GOOD_BAD import (
    clean_encounter_dx_bad_text,
    clean_medication_bad_text,
    filter_bad_cleaned_keep_only_readable,
    fix_encounter_dx_still_bad_lines,
    fix_medication_still_bad_lines,
    merge_good_cleaned_and_still_bad,
)

from INPUT_DATA_CACHE import load_data

from DATA_CLEANING import (
    strip_quotes_from_header,
    check_duplicate_by_first_column,
    standardize_missing_values,
    standardize_categorical_columns,
    convert_date_columns,
    report_invalid_dates,
    convert_numeric_cols_and_remove_negative,
    add_msk_tag,
    output_df,
)

from frequency_maps import map_frequency_to_numeric

from medication_classification import (
    add_medication_classification,
    refine_med_class_by_duration,
    impute_startdate_for_acute,
)


# Paths 
def get_output_dir(base_dir: str = "../data/output") -> Path:
    out = Path(base_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# Stage 1: split raw -> good/bad
def stage1_split_good_bad(
    *,
    patient_path: str = "../data/C4MPatient.csv",
    medication_path: str = "../data/C4MMedication.csv",
    encounter_dx_path: str = "../data/C4MEncounterdiagnosis.csv",
    sep: str = "|",
    chunksize: int = 100_000,
    output_dir: str = "../data/output",
) -> None:
    """
    Read raw files -> output good.csv and bad.csv for each dataset.
    """
    out = get_output_dir(output_dir)

    # Patient
    patient, patient_bad = read_csv_with_badlines(patient_path, sep, chunksize)
    print(f"[stage1] Patient good rows: {len(patient)} | bad rows: {len(patient_bad)}")
    patient.to_csv(out / "patient_good.csv", sep=sep, index=False)
    patient_bad.to_csv(out / "patient_bad.csv", index=False)

    # Medication
    medication, medication_bad = read_csv_with_badlines(medication_path, sep, chunksize)
    print(f"[stage1] Medication good rows: {len(medication)} | bad rows: {len(medication_bad)}")
    medication.to_csv(out / "medication_good.csv", sep=sep, index=False)
    medication_bad.to_csv(out / "medication_bad.csv", index=False)

    # Encounter diagnosis
    encounter_dx, encounter_dx_bad = read_csv_with_badlines(encounter_dx_path, sep, chunksize)
    print(f"[stage1] Encounter_dx good rows: {len(encounter_dx)} | bad rows: {len(encounter_dx_bad)}")
    encounter_dx.to_csv(out / "encounter_dx_good.csv", sep=sep, index=False)
    encounter_dx_bad.to_csv(out / "encounter_dx_bad.csv", index=False)

    print(f"[stage1] Outputs written to: {out.resolve()}")


# Stage 2: clean bad, filter unreadable, merge -> final
def stage2_clean_filter_merge(
    *,
    sep: str = "|",
    output_dir: str = "../data/output",
) -> None:
    """
    Clean bad-lines (text streaming), filter still-unreadable,
    then merge good + cleaned_bad_filtered -> final csv for medication & encounter_dx.
    """
    out = get_output_dir(output_dir)

    # (A) clean bad text -> cleaned_bad
    clean_medication_bad_text(str(out / "medication_bad.csv"), str(out / "medication_bad_cleaned.csv"))
    clean_encounter_dx_bad_text(str(out / "encounter_dx_bad.csv"), str(out / "encounter_dx_bad_cleaned.csv"))

    # (B) filter cleaned_bad to keep only readable rows
    filter_bad_cleaned_keep_only_readable(
        good_path=str(out / "medication_good.csv"),
        bad_cleaned_path=str(out / "medication_bad_cleaned.csv"),
        out_cleaned_path=str(out / "medication_bad_cleaned_filtered.csv"),
        out_still_bad_path=str(out / "medication_bad_still_unreadable.csv"),
        sep=sep,
        skip_first_line=False,
        label="medication",
    )

    filter_bad_cleaned_keep_only_readable(
        good_path=str(out / "encounter_dx_good.csv"),
        bad_cleaned_path=str(out / "encounter_dx_bad_cleaned.csv"),
        out_cleaned_path=str(out / "encounter_dx_bad_cleaned_filtered.csv"),
        out_still_bad_path=str(out / "encounter_dx_bad_still_unreadable.csv"),
        sep=sep,
        skip_first_line=False,
        label="encounter_dx",
    )

    out_path, n = fix_encounter_dx_still_bad_lines(
        input_path="../data/output/encounter_dx_bad_still_unreadable.csv",
        output_path="../data/output/encounter_dx_bad_still_unreadable_fixed.csv",
        expected_n_cols=13,
    )

    print(f"Fixed still-bad encounter_dx lines: {n}")
    print(f"Output written to: {out_path}")  

    out_path, n = fix_medication_still_bad_lines(
        input_path="../data/output/medication_bad_still_unreadable.csv",
        output_path="../data/output/medication_bad_still_unreadable_fixed.csv",
    )

    print(f"Fixed medication still-bad lines written: {n}")
    print(f"Output file: {out_path}")  

    # (C) merge good + cleaned_bad_filtered -> final
    merge_good_cleaned_and_still_bad(
        good_path="../data/output/encounter_dx_good.csv",
        bad_cleaned_path="../data/output/encounter_dx_bad_cleaned_filtered.csv",
        still_bad_fixed_path="../data/output/encounter_dx_bad_still_unreadable_fixed.csv",
        out_path="../data/output/encounter_dx_final.csv",
        sep="|",
        label="Encounter_dx"
    )

    merge_good_cleaned_and_still_bad(
        good_path="../data/output/medication_good.csv",
        bad_cleaned_path="../data/output/medication_bad_cleaned_filtered.csv",
        still_bad_fixed_path="../data/output/medication_bad_still_unreadable_fixed.csv",
        out_path="../data/output/medication_final.csv",
        sep="|",
        label="Medication"
    )

    print("[stage2] merge complete -> medication_final.csv & encounter_dx_final.csv")


# Stage 3: load final datasets -> core cleaning
def stage3_load_and_basic_clean() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load patient/medication/encounter_dx (from cache loaders),
    then do basic cleaning: header quotes, duplicates check, missing standardize, categorical standardize, date convert.
    Returns:
      patient_clean, medication_clean, encounter_dx_clean
    """
    # load
    patient = load_data(
        csv_path="../data/output/patient_good.csv",
        parquet_path="../data/cache/patient.parquet",
    )

    medication = load_data(
        csv_path="../data/output/medication_final.csv",
        parquet_path="../data/cache/medication.parquet",
    )

    encounter_dx = load_data(
        csv_path="../data/output/encounter_dx_final.csv",
        parquet_path="../data/cache/encounter_dx.parquet",
    )

    # strip quotes from header
    patient = strip_quotes_from_header(patient)
    medication = strip_quotes_from_header(medication)
    encounter_dx = strip_quotes_from_header(encounter_dx)

    # duplicates check
    check_duplicate_by_first_column(patient, "Patient_ID", "patient")
    check_duplicate_by_first_column(encounter_dx, "EncounterDiagnosis_ID", "encounter_dx")
    check_duplicate_by_first_column(medication, "Medication_ID", "medication")

    # standardize missing
    patient = standardize_missing_values(patient)
    encounter_dx = standardize_missing_values(encounter_dx)
    medication = standardize_missing_values(medication)

    # standardize categorical
    patient = standardize_categorical_columns(patient, ["Sex"])
    encounter_dx = standardize_categorical_columns(encounter_dx, ["DiagnosisText_calc", "DiagnosisCodeType_calc"])
    medication = standardize_categorical_columns(
        medication,
        ["Reason", "Name_calc", "CodeType_calc", "Code_calc", "UnitOfMeasure", "Frequency", "DurationUnit"],
    )

    # convert dates
    patient = convert_date_columns(patient)
    encounter_dx = convert_date_columns(encounter_dx)
    medication = convert_date_columns(medication)

    report_invalid_dates(patient, "patient")
    report_invalid_dates(encounter_dx, "encounter_dx")
    report_invalid_dates(medication, "medication")

    return patient, medication, encounter_dx


# Stage 4: dataset-specific enrichments
def stage4_enrich_encounter_dx(encounter_dx: pd.DataFrame) -> pd.DataFrame:
    """
    Add MSK tag.
    """
    return add_msk_tag(encounter_dx)


def stage4_enrich_medication(medication: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric conversion, frequency mapping, medication classification, refine class, impute acute startdate.
    """
    medication = convert_numeric_cols_and_remove_negative(
        medication,
        ["Strength", "Dose", "DurationCount", "DispensedCount"],
    )

    medication = map_frequency_to_numeric(medication)

    medication = add_medication_classification(medication, out_col="med_class")
    medication = refine_med_class_by_duration(medication)
    medication = impute_startdate_for_acute(medication)

    return medication


# Stage 5: output

def stage5_output_cleaned(
    patient: pd.DataFrame,
    encounter_dx: pd.DataFrame,
    medication: pd.DataFrame,
    *,
    out_patient: str = "../data/output/C4MPatient_clean_1_31.csv",
    out_encounter_dx: str = "../data/output/C4MEncounterDiagnosis_clean_1_31.csv",
    out_medication: str = "../data/output/C4MMedication_clean_1_31.csv",
) -> None:
    output_df(patient, out_patient)
    output_df(encounter_dx, out_encounter_dx)
    output_df(medication, out_medication)
    print("[stage5] cleaned outputs written.")



