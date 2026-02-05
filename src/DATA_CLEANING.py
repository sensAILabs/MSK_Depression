from INPUT_DATA_CACHE import load_data
import numpy as np
import pandas as pd
import statsmodels.api as sm
import csv
from medication_classification import add_medication_classification, summarize_classification, refine_med_class_by_duration, impute_startdate_for_acute, extract_startdate_missing_unknown
from frequency_maps import map_frequency_to_numeric
import time
start = time.time()


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
def strip_quotes_from_header(df):
    df = df.copy()
    df.columns = (
        df.columns     
          .str.strip('"')   
    )
    return df

patient = strip_quotes_from_header(patient)
medication = strip_quotes_from_header(medication)
encounter_dx = strip_quotes_from_header(encounter_dx)


#### clean data (duplication check function)
def check_duplicate_by_first_column(df,primary_col,df_name):
    pk_col = primary_col
    print(f"\nChecking primary key: {pk_col}")
    print(f"Total rows: {len(df)}")
    dup = df[df[pk_col].duplicated(keep=False)]

    if dup.empty:
        print(f"No duplicate primary keys found.({df_name})")
    else:
        print(f"Found {dup.shape[0]} duplicated rows({df_name}):")
        print(dup.sort_values(pk_col))

check_duplicate_by_first_column(patient,"Patient_ID","patient")
check_duplicate_by_first_column(encounter_dx,"EncounterDiagnosis_ID","encounter_dx")
check_duplicate_by_first_column(medication,"Medication_ID","medication")

# all good, no duplicates found

# replace na and pseudo na to pd.NA
def standardize_missing_values(df):
    pseudo_missing = ["","Unknown", "UNK", "N/A", "NA","NAN", "None"]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .str.strip()
                .replace(pseudo_missing, pd.NA)
            )
    return df

patient_check_duplication_NA = standardize_missing_values(patient)
encounter_dx_check_duplication_NA = standardize_missing_values(encounter_dx)
medication_check_duplication_NA = standardize_missing_values(medication)

# Categorical columns standardization
def standardize_categorical_columns(df, categorical_cols):

    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.upper()
            )

    return df

patient_check_duplication_NA_strip_upper = standardize_categorical_columns(patient_check_duplication_NA, ["Sex"])
encounter_dx_check_duplication_NA_strip_upper = standardize_categorical_columns(encounter_dx_check_duplication_NA, 
                                                                                ["DiagnosisText_calc", "DiagnosisCodeType_calc"])
medication_check_duplication_NA_strip_upper = standardize_categorical_columns(medication_check_duplication_NA, 
                                                                              ["Reason","Name_calc","CodeType_calc",
                                                                               "Code_calc","UnitOfMeasure", "Frequency", "DurationUnit"])
    

# Date variable standardization
def convert_date_columns(df):
    """
    Find columns whose name contains 'date' and convert them to datetime.
    """
    df_clean = df.copy()

    for col in df_clean.columns:
        if "date" in col.lower():
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

    return df_clean

patient_check_duplication_NA_strip_upper_date= convert_date_columns(patient_check_duplication_NA_strip_upper)
encounter_dx_check_duplication_NA_strip_upper_date = convert_date_columns(encounter_dx_check_duplication_NA_strip_upper)
medication_check_duplication_NA_strip_upper_date = convert_date_columns(medication_check_duplication_NA_strip_upper)

# Check invalid dates
def report_invalid_dates(df,df_name):
    """
    year < 1900
    year > current year
    """

    current_year = pd.Timestamp.today().year

    for col in df.columns:
        if "date" in col.lower():
            dt = pd.to_datetime(df[col], errors="coerce")

            invalid = dt[
                (dt.dt.year < 1900) |
                (dt.dt.year > current_year)
            ]

            if not invalid.empty:
                print(f"Invalid dates found in column({df_name}): {col}")
                print(invalid.value_counts())

report_invalid_dates(patient_check_duplication_NA_strip_upper_date, "patient")
report_invalid_dates(encounter_dx_check_duplication_NA_strip_upper_date, "encounter_dx")
report_invalid_dates(medication_check_duplication_NA_strip_upper_date, "medication")

# function to delete invalid dates



# convert_numeric_and_remove_negative
def convert_numeric_cols_and_remove_negative(df, cols):
    df_clean = df.copy()
    for col in cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean.loc[df_clean[col] < 0, col] = pd.NA

    return df_clean


medication_check_duplication_NA_strip_upper_date_numric = convert_numeric_cols_and_remove_negative(
    medication_check_duplication_NA_strip_upper_date, ["Strength", "Dose", "DurationCount", "DispensedCount"])

# musculoskeletal disorders 
def add_msk_tag(df):

    df["musculoskeletal_disorders"] = False

    is_icd9 = df["DiagnosisCodeType_calc"].astype("string").eq("ICD9")

    if not is_icd9.any():
        return df  

    codes = pd.to_numeric(df.loc[is_icd9, "DiagnosisCode_orig"], errors="coerce")

    is_msd_row = (
        (codes == 715) |
        ((codes >= 710) & (codes <= 719.9)) |
        ((codes >= 720) & (codes <= 724.9)) |
        ((codes >= 725) & (codes <= 729.9)) |
        ((codes >= 730) & (codes <= 739.9))
    )

    msd_patients = df.loc[is_icd9, "Patient_ID"].loc[is_msd_row].dropna().unique()

    df.loc[df["Patient_ID"].isin(msd_patients), "musculoskeletal_disorders"] = True

    return df

encounter_dx_check_duplication_NA_strip_upper_date_msk = add_msk_tag(encounter_dx_check_duplication_NA_strip_upper_date)


# medicaiton. Convert frequency to num
medication_check_duplication_NA_strip_upper_date_numric_freqnum = map_frequency_to_numeric(medication_check_duplication_NA_strip_upper_date_numric)


# Impute medication classification & missing start date for acute
medication_check_duplication_NA_strip_upper_date_numric_freqnum_medclassification = add_medication_classification(medication_check_duplication_NA_strip_upper_date_numric_freqnum, out_col="med_class")
medication_check_duplication_NA_strip_upper_date_numric_freqnum_medclassification = refine_med_class_by_duration(medication_check_duplication_NA_strip_upper_date_numric_freqnum_medclassification)
medication_check_duplication_NA_strip_upper_date_numric_freqnum_medclassification_startdateimput = impute_startdate_for_acute(medication_check_duplication_NA_strip_upper_date_numric_freqnum_medclassification)








# from pathlib import Path
# CACHE_DIR = Path("../data/cache")
# medication_check_duplication_NA_strip_upper_date_numric_freqnum[["Frequency", "Event_interval_days"]].to_parquet(CACHE_DIR / "medication_frequency.parquet")
# print("frequency parquet saved.")
# medication_check_duplication_NA_strip_upper_date_numric_freqnum[medication_check_duplication_NA_strip_upper_date_numric_freqnum["Event_interval_days"].isna()]["Frequency"].value_counts().head(40)

# Output cleaned dataset
def output_df(df, output_path):
    #bad line
    df.to_csv(output_path, index=False)

output_df(patient_check_duplication_NA_strip_upper_date, "../data/output/C4MPatient_clean_1_31.csv")
output_df(encounter_dx_check_duplication_NA_strip_upper_date_msk, "../data/output/C4MEncounterDiagnosis_clean_1_31.csv")
output_df(medication_check_duplication_NA_strip_upper_date_numric_freqnum_medclassification_startdateimput, "../data/output/C4MMedication_clean_1_31.csv")


print("Completed cleaning and outputting dataset.")

end = time.time()
print(f"Runtime: {end - start:.2f} seconds")
