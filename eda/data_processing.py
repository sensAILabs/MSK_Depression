import pandas as pd
from eda.clinical_definitions import ICD9_RANGES


# Helper functions
def clean_patient_id(series):
    return series.astype("string").str.strip()

def parse_date(series):
    return pd.to_datetime(series, errors="coerce")


def extract_icd9_numeric(series):
    return pd.to_numeric(
        series.astype("string").str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce")

def map_icd9_system(icd9_num, ranges=None):
    ranges = ICD9_RANGES if ranges is None else ranges

    if pd.isna(icd9_num):
        return pd.NA

    for system, (lo, hi) in ranges.items():
        if lo <= icd9_num <= hi:
            return system

    return pd.NA



# Step 0: prepare encounter data
def prepare_encounter_data(
    encounter,
    patient_col="Patient_ID",
    date_col="DateCreated",
    icd9_col="icd9",
):
    df = encounter.copy()

    df[patient_col] = clean_patient_id(df[patient_col])
    df[date_col] = parse_date(df[date_col])
    df[icd9_col] = df[icd9_col].astype("string").str.strip()
    df["icd9_num"] = extract_icd9_numeric(df[icd9_col])

    return df

# Step 1: build patient info
def build_patient_info(
    encounter,
    patient,
    patient_col="Patient_ID",
    encounter_date_col="DateCreated",
    sex_col="Sex",
    birthyear_col="BirthYear",
):
    enc = encounter.copy()
    pat = patient.copy()

    enc[patient_col] = clean_patient_id(enc[patient_col])
    enc[encounter_date_col] = parse_date(enc[encounter_date_col])

    pat[patient_col] = clean_patient_id(pat[patient_col])
    pat[birthyear_col] = pd.to_numeric(pat[birthyear_col], errors="coerce")

    first = (
        enc[[patient_col, encounter_date_col]]
        .dropna(subset=[patient_col, encounter_date_col])
        .groupby(patient_col, as_index=False)[encounter_date_col]
        .min()
    )

    demo = (
        pat[[patient_col, sex_col, birthyear_col]]
        .drop_duplicates(subset=[patient_col])
        .copy()
    )

    out = first.merge(demo, on=patient_col, how="left", validate="1:1")
    out["Age"] = out[encounter_date_col].dt.year - out[birthyear_col]

    return out.drop(columns=[encounter_date_col])


# Step 2: add patient icd9 list and type list
def add_patient_icd9_lists(
    encounter,
    patient_col="Patient_ID",
    icd9_col="icd9",
    ranges=None,
):
    ranges = ICD9_RANGES if ranges is None else ranges

    df = encounter.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.drop(columns=["icd9 list", "type list"], errors="ignore")

    df[patient_col] = clean_patient_id(df[patient_col])

    if icd9_col not in df.columns:
        raise KeyError(f"Column '{icd9_col}' not found in encounter dataframe.")

    df[icd9_col] = df[icd9_col].astype("string").str.strip()

    if "icd9_num" not in df.columns:
        df["icd9_num"] = extract_icd9_numeric(df[icd9_col])

    df["_type"] = df["icd9_num"].apply(lambda x: map_icd9_system(x, ranges))
    df = df[df["_type"].notna()].copy()

    codes = (
        df.groupby(patient_col)[icd9_col]
        .agg(lambda s: ",".join(pd.unique(s.dropna())))
        .rename("icd9 list"))

    types = (
        df.groupby(patient_col)["_type"]
        .agg(lambda s: ",".join(pd.unique(s.dropna())))
        .rename("type list"))

    patient_lists = pd.concat([codes, types], axis=1).reset_index()

    out = df.merge(patient_lists, on=patient_col, how="left", validate="m:1")
    out[["icd9 list", "type list"]] = out[["icd9 list", "type list"]].fillna("")
    out = out.drop(columns=["_type"])

    return out


# Step 3: build MSK type table
# Based on observed first overall encounter timing

def build_msk_type_table(
    enc_icd9,
    ranges=None,
    patient_col="Patient_ID",
    date_col="DateCreated",
    icd9_col="icd9",
    out_col="msk_type",
):
    ranges = ICD9_RANGES if ranges is None else ranges

    df = enc_icd9.copy()

    df[patient_col] = clean_patient_id(df[patient_col])
    df[date_col] = parse_date(df[date_col])

    if icd9_col not in df.columns:
        raise KeyError(f"Column '{icd9_col}' not found in encounter dataframe.")

    df[icd9_col] = df[icd9_col].astype("string").str.strip()

    if "icd9_num" not in df.columns:
        df["icd9_num"] = extract_icd9_numeric(df[icd9_col])

    df["_type"] = df["icd9_num"].apply(lambda x: map_icd9_system(x, ranges))
    df = df.dropna(subset=[patient_col, date_col, "_type"]).copy()

    first_date = df.groupby(patient_col)[date_col].min().rename("first_date")
    df = df.merge(first_date, on=patient_col, how="left", validate="m:1")

    df["_is_msk"] = df["_type"].eq("msk")
    df["_is_non_msk"] = ~df["_is_msk"]
    df["_on_first"] = df[date_col].eq(df["first_date"])
    df["_after_first"] = df[date_col].gt(df["first_date"])

    has_msk_on_first = (df["_on_first"] & df["_is_msk"]).groupby(df[patient_col]).any()
    has_non_on_first = (df["_on_first"] & df["_is_non_msk"]).groupby(df[patient_col]).any()
    has_msk_after = (df["_after_first"] & df["_is_msk"]).groupby(df[patient_col]).any()
    has_non_after = (df["_after_first"] & df["_is_non_msk"]).groupby(df[patient_col]).any()
    has_non_ever = df["_is_non_msk"].groupby(df[patient_col]).any()

    summary = pd.DataFrame({
        patient_col: has_msk_on_first.index.astype("string"),
        "has_msk_on_first": has_msk_on_first.values,
        "has_non_on_first": has_non_on_first.reindex(has_msk_on_first.index, fill_value=False).values,
        "has_msk_after_first": has_msk_after.reindex(has_msk_on_first.index, fill_value=False).values,
        "has_non_after_first": has_non_after.reindex(has_msk_on_first.index, fill_value=False).values,
        "has_non_ever": has_non_ever.reindex(has_msk_on_first.index, fill_value=False).values,
    })

    summary[out_col] = "unknown"

    summary.loc[
        summary["has_msk_on_first"] & ~summary["has_non_ever"],
        out_col
    ] = "msk_only"

    summary.loc[
        summary["has_msk_on_first"] & summary["has_non_on_first"],
        out_col
    ] = "co_occurrence"

    summary.loc[
        ~summary["has_msk_on_first"] & summary["has_msk_after_first"],
        out_col
    ] = "other_pre_existing_then_msk"

    summary.loc[
        summary["has_msk_on_first"]
        & ~summary["has_non_on_first"]
        & summary["has_non_after_first"],
        out_col
    ] = "msk_comorbidity"

    summary = summary[summary[out_col] != "unknown"].reset_index(drop=True)

    return summary[[patient_col, out_col]]


# Step 4: build analysis dataframe
def build_analysis_df_msk_only(
    patient_info,
    enc_icd9_msk,
    msk_type_table,
    msk_duration,
    mental_duration,
    patient_col="Patient_ID",
):
    normalize_id = lambda s: s.astype("string").str.strip()

    base = (
        enc_icd9_msk.assign(**{patient_col: normalize_id(enc_icd9_msk[patient_col])})
        .groupby(patient_col, as_index=False)
        .agg(
            **{
                "icd9 list": ("icd9 list", "first"),
                "type list": ("type list", "first"),
                "comorbidity_count": ("comorbidity_count", "first"),
            }
        )
    )

    out = (
        base
        .merge(
            patient_info.assign(**{patient_col: normalize_id(patient_info[patient_col])})[
                [patient_col, "Sex", "Age"]
            ],
            on=patient_col,
            how="left",
        )
        .merge(
            msk_type_table.assign(**{patient_col: normalize_id(msk_type_table[patient_col])}),
            on=patient_col,
            how="left",
        )
        .merge(
            msk_duration.assign(**{patient_col: normalize_id(msk_duration[patient_col])})[
                [patient_col, "msk_duration_years"]
            ],
            on=patient_col,
            how="left",
        )
        .merge(
            mental_duration.assign(**{patient_col: normalize_id(mental_duration[patient_col])})[
                [patient_col, "mental_duration_years"]
            ],
            on=patient_col,
            how="left",
        )
    )

    return out[
        [
            patient_col,
            "Sex",
            "Age",
            "msk_type",
            "icd9 list",
            "type list",
            "comorbidity_count",
            "msk_duration_years",
            "mental_duration_years",
        ]
    ]