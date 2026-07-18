import pandas as pd
from eda.clinical_definitions import ICD9_RANGES


# Step 1 — filter cohort
def filter_cohort(df):
    """
    Keep only the 'other_pre_existing_then_msk' cohort.

    This cohort is defined based on the observed diagnosis-record order
    in the upstream EDA table.
    """
    out = df.copy()

    out["Patient_ID"] = out["Patient_ID"].astype(str).str.strip()

    out = out[
        out["msk_type"] == "other_pre_existing_then_msk"
    ].copy()

    return out


# Step 2 — build pre-MSK encounter-level table
def build_pre_msk_encounter(encounter_df, cohort_df, icd9_to_name=None):
    """
    Keep only non-MSK ICD9 encounters that occurred before the first MSK date
    for patients in the selected cohort.

    Output is encounter-level.
    """
    enc = encounter_df.copy()
    cohort = cohort_df.copy()

    # clean patient id
    enc["Patient_ID"] = enc["Patient_ID"].astype(str).str.strip()
    cohort["Patient_ID"] = cohort["Patient_ID"].astype(str).str.strip()

    # keep only cohort patients
    cohort_ids = set(cohort["Patient_ID"].unique())
    enc = enc[enc["Patient_ID"].isin(cohort_ids)].copy()

    # keep ICD9 only if the code type column exists
    if "DiagnosisCodeType_calc" in enc.columns:
        enc = enc[enc["DiagnosisCodeType_calc"] == "ICD9"].copy()

    # standardize date
    enc["DateCreated"] = pd.to_datetime(enc["DateCreated"], errors="coerce")

    # detect source ICD column
    if "DiagnosisCode_calc" in enc.columns:
        raw_code_col = "DiagnosisCode_calc"
    elif "icd9" in enc.columns:
        raw_code_col = "icd9"
    else:
        raise ValueError("No ICD9 code column found. Need 'DiagnosisCode_calc' or 'icd9'.")

    # clean ICD9 code
    enc["icd9_split"] = (
        enc[raw_code_col]
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
    )

    # numeric ICD9
    enc["icd9_num"] = pd.to_numeric(
        enc["icd9_split"].str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce"
    )

    # map system type
    enc["_type"] = pd.NA
    for k, (lo, hi) in ICD9_RANGES.items():
        enc.loc[enc["icd9_num"].between(lo, hi), "_type"] = k

    enc = enc.dropna(
        subset=["Patient_ID", "DateCreated", "icd9_split", "_type"]
    ).copy()

    # optional ICD9 name mapping
    if icd9_to_name is not None:
        enc["icd9_name"] = enc["icd9_split"].map(icd9_to_name)
    else:
        enc["icd9_name"] = pd.NA

    # first diagnosis date for each (patient, icd9)
    first_dates = (
        enc.groupby(["Patient_ID", "icd9_split"])["DateCreated"]
        .min()
        .reset_index()
        .rename(columns={"DateCreated": "icd9_first_date"})
    )

    enc = enc.merge(first_dates, on=["Patient_ID", "icd9_split"], how="left")

    # first MSK date per patient
    first_msk = (
        enc.loc[enc["_type"] == "msk"]
        .groupby("Patient_ID")["DateCreated"]
        .min()
        .rename("first_msk_date")
        .reset_index()
    )

    enc = enc.merge(first_msk, on="Patient_ID", how="inner")

    # keep only diagnoses before first MSK
    enc = enc[enc["DateCreated"] < enc["first_msk_date"]].copy()

    # remove MSK itself
    enc = enc[enc["_type"] != "msk"].copy()

    return enc


# Step 3 — rebuild patient-level table from pre-MSK encounter
def rebuild_pre_msk_patient_table(pre_long):
    """
    Rebuild a patient-level table from the pre-MSK encounter-level table.

    Output variables:
    - icd9 list
    - type list
    - first_msk_date
    - comorbidity_count
    """
    out = pre_long.copy()

    pt = (
        out.groupby("Patient_ID")
        .agg(
            **{
                "icd9 list": (
                    "icd9_split",
                    lambda s: ",".join(sorted(set(s.dropna())))
                ),
                "type list": (
                    "_type",
                    lambda s: ",".join(sorted(set(s.dropna())))
                ),
                "first_msk_date": (
                    "first_msk_date",
                    "first"
                ),
            }
        )
        .reset_index()
    )

    # number of distinct non-MSK systems before first MSK
    pt["comorbidity_count"] = pt["type list"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )

    return pt