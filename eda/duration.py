
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from clinical_definitions import ICD9_RANGES, CONDITION_DRUG_KEYWORDS

#Compute condition duration
def compute_condition_duration(encounter, medication, ICD9_RANGES,condition,drug_keywords=None,   
                               patient_col="Patient_ID", enc_date_col="DateCreated", icd9_col="icd9",
                               med_name_col="Name_calc", med_stop_col="StopDate",
                               med_datecreated_col="DateCreated",out_prefix=None):
    if out_prefix is None:
        out_prefix = condition  
    lo, hi = ICD9_RANGES[condition]
    enc = encounter[[patient_col, enc_date_col, icd9_col]].copy()
    enc[patient_col] = enc[patient_col].astype("string").str.strip()
    enc[enc_date_col] = pd.to_datetime(enc[enc_date_col], errors="coerce")
    num = pd.to_numeric(enc[icd9_col].astype("string").str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    enc_cond = enc[num.between(lo, hi, inclusive="both")].dropna(subset=[patient_col, enc_date_col])
    dates = (enc_cond.groupby(patient_col)[enc_date_col]
             .agg(first_date="min", last_date="max")
             .reset_index()
             .rename(columns={ "first_date": f"first_{out_prefix}_date",
                               "last_date": f"last_{out_prefix}_date" }))
    # default end_date = last diagnosis date
    dates = dates.dropna(subset=[f"first_{out_prefix}_date", f"last_{out_prefix}_date"])
    #  extend end_date by meds (StopDate else DateCreated)
    if drug_keywords:
        pattern = "|".join([str(x).upper() for x in drug_keywords])
        med = medication[[patient_col, med_name_col, med_stop_col, med_datecreated_col]].copy()
        med[patient_col] = med[patient_col].astype("string").str.strip()
        med[med_name_col] = med[med_name_col].astype("string").str.upper()
        med[med_stop_col] = pd.to_datetime(med[med_stop_col], errors="coerce")
        med[med_datecreated_col] = pd.to_datetime(med[med_datecreated_col], errors="coerce")
        med_cond = med[med[med_name_col].str.contains(pattern, regex=True, na=False)]
        med_end = (med_cond.groupby(patient_col)
                   .agg(last_stop=(med_stop_col, "max"),
                        last_created=(med_datecreated_col, "max"))
                   .reset_index())
        med_end["last_drug_end"] = med_end["last_stop"].fillna(med_end["last_created"])
        dates = dates.merge(med_end[[patient_col, "last_drug_end"]], on=patient_col, how="left", validate="1:1")
        # if last_drug_end > last diagnosis date，use medication end date
        dates["end_date"] = dates[f"last_{out_prefix}_date"].where(
            ~dates["last_drug_end"].gt(dates[f"last_{out_prefix}_date"]),
            dates["last_drug_end"] )
    # duration
    dates[f"{out_prefix}_duration_days"] = ((dates["end_date"] - dates[f"first_{out_prefix}_date"]).dt.days).clip(lower=1)
    dates[f"{out_prefix}_duration_years"] = dates[f"{out_prefix}_duration_days"] / 365
    return dates[[patient_col, f"first_{out_prefix}_date",f"last_{out_prefix}_date",
                 "end_date",f"{out_prefix}_duration_years"]]


