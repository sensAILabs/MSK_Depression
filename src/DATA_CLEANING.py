from INPUT_DATA_CACHE import load_data
import numpy as np
import pandas as pd
from medication_classification import add_medication_classification, refine_med_class_by_duration, impute_startdate_for_acute, extract_startdate_missing_unknown
from frequency_maps import map_frequency_to_numeric


# strip quotes from header
def strip_quotes_from_header(df):
    df = df.copy()
    df.columns = (
        df.columns     
          .str.strip('"')   
    )
    return df

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

# function to delete invalid dates

# convert_numeric_and_remove_negative
def convert_numeric_cols_and_remove_negative(df, cols):
    df_clean = df.copy()
    for col in cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean.loc[df_clean[col] < 0, col] = pd.NA

    return df_clean

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

# Output cleaned dataset
def output_df(df, output_path):
    #bad line
    df.to_csv(output_path, index=False)
