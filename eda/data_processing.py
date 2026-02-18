import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from clinical_definitions import ICD9_RANGES

#step 1 build patient info
def build_patient_info(encounter, patient,
                       patient_col="Patient_ID", encounter_date_col="DateCreated",
                       sex_col="Sex", birthyear_col="BirthYear"):
    first = (encounter[[patient_col, encounter_date_col]]
             .assign(**{patient_col: lambda d: d[patient_col].astype("string").str.strip(),
                        encounter_date_col: lambda d: pd.to_datetime(d[encounter_date_col], errors="coerce")})
             .dropna(subset=[patient_col, encounter_date_col])
             .groupby(patient_col, as_index=False)[encounter_date_col].min())
    demo = (patient[[patient_col, sex_col, birthyear_col]]
            .assign(**{patient_col: lambda d: d[patient_col].astype("string").str.strip(),
                       birthyear_col: lambda d: pd.to_numeric(d[birthyear_col], errors="coerce")})
            .drop_duplicates(subset=[patient_col]))
    return (first.merge(demo, on=patient_col, how="left", validate="1:1")
                 .assign(Age=lambda d: d[encounter_date_col].dt.year - d[birthyear_col])
                 .drop(columns=[encounter_date_col]))

#step 2 rename"DiagnosisCode_calc" to "icd9"

#Add patient icd9 list and type list
def add_patient_icd9_lists(df, patient_col="Patient_ID", icd9_col="icd9", ranges=None):
    ranges = ICD9_RANGES if ranges is None else ranges
    out = df.loc[:, ~df.columns.duplicated()].copy()
    out = out.drop(columns=["icd9 list", "type list"], errors="ignore")
    out[patient_col] = out[patient_col].astype("string").str.strip()
    out[icd9_col] = out[icd9_col].astype("string").str.strip()
    num = pd.to_numeric(out[icd9_col].str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    out["_type"] = pd.NA
    for k, (lo, hi) in ranges.items():
        out.loc[num.between(lo, hi, inclusive="both"), "_type"] = k
    out = out[out["_type"].notna()].copy()
    # patient-level lists
    codes = out.groupby(patient_col)[icd9_col].agg(lambda s: ",".join(pd.unique(s.dropna())))
    types = out.groupby(patient_col)["_type"].agg(lambda s: ",".join(pd.unique(s.dropna())))
    m = pd.concat([codes.rename("icd9 list"), types.rename("type list")], axis=1).reset_index()
    out = out.merge(m, on=patient_col, how="left", validate="m:1")
    out[["icd9 list", "type list"]] = out[["icd9 list", "type list"]].fillna("")
    out = out.drop(columns=["_type"])
    return out


#Build msk type table
def build_msk_type_table(enc_icd9,ICD9_RANGES,patient_col="Patient_ID",
                         date_col="DateCreated",icd9_col="icd9",out_col="msk_type"):
    df = enc_icd9.copy()
    df[patient_col] = df[patient_col].astype("string").str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[icd9_col] = df[icd9_col].astype("string").str.strip()
    # icd9 -> numeric
    num = pd.to_numeric(df[icd9_col].str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    df["_type"] = pd.NA
    for k, (lo, hi) in ICD9_RANGES.items():
        df.loc[num.between(lo, hi, inclusive="both"), "_type"] = k
    # keep valid rows only
    df = df.dropna(subset=[patient_col, date_col, "_type"]).copy()
    # first overall encounter date
    first_date = df.groupby(patient_col)[date_col].min()
    df = df.merge(first_date.rename("first_date"),
                  on=patient_col, how="left", validate="m:1")
    # flags relative to first encounter
    df["_is_msk"] = df["_type"].eq("msk")
    df["_is_non_msk"] = ~df["_is_msk"]
    df["_on_first"] = df[date_col].eq(df["first_date"])
    df["_after_first"] = df[date_col].gt(df["first_date"])
    # patient level summaries
    g = df.groupby(patient_col)
    has_msk_on_first = (df["_on_first"] & df["_is_msk"]).groupby(df[patient_col]).any()
    has_non_on_first = (df["_on_first"] & df["_is_non_msk"]).groupby(df[patient_col]).any()
    has_msk_after = (df["_after_first"] & df["_is_msk"]).groupby(df[patient_col]).any()
    has_non_after = (df["_after_first"] & df["_is_non_msk"]).groupby(df[patient_col]).any()
    has_non_ever = df["_is_non_msk"].groupby(df[patient_col]).any()
    m = pd.DataFrame({
        patient_col: has_msk_on_first.index.astype("string"),
        "has_msk_on_first": has_msk_on_first.values,
        "has_non_on_first": has_non_on_first.reindex(has_msk_on_first.index, fill_value=False).values,
        "has_msk_after_first": has_msk_after.reindex(has_msk_on_first.index, fill_value=False).values,
        "has_non_after_first": has_non_after.reindex(has_msk_on_first.index, fill_value=False).values,
        "has_non_ever": has_non_ever.reindex(has_msk_on_first.index, fill_value=False).values, })
    # mutually exclusive msk_type (based on first overall encounter)
    m[out_col] = "unknown"
    m.loc[m["has_msk_on_first"] & ~m["has_non_ever"], out_col] = "msk_only"
    m.loc[m["has_msk_on_first"] & m["has_non_on_first"], out_col] = "co_onset"
    m.loc[~m["has_msk_on_first"] & m["has_msk_after_first"], out_col] = "other_pre_existing_then_msk"
    m.loc[m["has_msk_on_first"] & ~m["has_non_on_first"] & m["has_non_after_first"], out_col] = "msk_comorbidity"
    # only keep msk patient
    m = m[m[out_col] != "unknown"].reset_index(drop=True)
    return m[[patient_col, out_col]]

#analysis df
def build_analysis_df_msk_only(patient_info, enc_icd9_msk, msk_type_table, msk_duration, mental_duration,
                               patient_col="Patient_ID"):
    nid = lambda s: s.astype("string").str.strip()
    base = (enc_icd9_msk.assign(**{patient_col: nid(enc_icd9_msk[patient_col])})
                      .groupby(patient_col, as_index=False)
                      .agg(**{"icd9 list": ("icd9 list", "first"),
                              "type list": ("type list", "first"),
                              "comorbidity_count": ("comorbidity_count", "first"),}))
    out = (base
        .merge(patient_info.assign(**{patient_col: nid(patient_info[patient_col])})[[patient_col, "Sex", "Age"]],
               on=patient_col, how="left")
        .merge(msk_type_table.assign(**{patient_col: nid(msk_type_table[patient_col])}),
               on=patient_col, how="left")
        .merge(msk_duration.assign(**{patient_col: nid(msk_duration[patient_col])})[[patient_col, "msk_duration_years"]],
               on=patient_col, how="left")
        .merge(mental_duration.assign(**{patient_col: nid(mental_duration[patient_col])})[[patient_col, "mental_duration_years"]],
               on=patient_col, how="left"))
    return out[[patient_col, "Sex", "Age", "msk_type", "icd9 list", "type list",
                "comorbidity_count", "msk_duration_years", "mental_duration_years"]]
