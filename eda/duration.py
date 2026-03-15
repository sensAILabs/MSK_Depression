import re
import pandas as pd
from eda.clinical_definitions import ICD9_RANGES


def compute_condition_duration(
    encounter,
    medication,
    ranges,
    condition,
    drug_keywords=None,
    patient_col="Patient_ID",
    enc_date_col="DateCreated",
    icd9_col="icd9",
    med_name_col="Name_calc",
    med_stop_col="StopDate",
    med_datecreated_col="DateCreated",
    out_prefix=None,
):
    if out_prefix is None:
        out_prefix = condition

    if condition not in ranges:
        raise KeyError(f"Condition '{condition}' not found in ICD9 ranges.")

    lo, hi = ranges[condition]

    enc = encounter[[patient_col, enc_date_col, icd9_col]].copy()
    enc[patient_col] = enc[patient_col].astype("string").str.strip()
    enc[enc_date_col] = pd.to_datetime(enc[enc_date_col], errors="coerce")

    enc[icd9_col] = enc[icd9_col].astype("string").str.strip()

    if "icd9_num" in encounter.columns:
        enc["icd9_num"] = pd.to_numeric(encounter["icd9_num"], errors="coerce")
    else:
        enc["icd9_num"] = pd.to_numeric(
            enc[icd9_col].str.extract(r"(\d+\.?\d*)")[0],
            errors="coerce"
        )

    enc_cond = enc[
        enc["icd9_num"].between(lo, hi, inclusive="both")
    ].dropna(subset=[patient_col, enc_date_col])

    dates = (
        enc_cond.groupby(patient_col)[enc_date_col]
        .agg(first_date="min", last_date="max")
        .reset_index()
        .rename(
            columns={
                "first_date": f"first_{out_prefix}_date",
                "last_date": f"last_{out_prefix}_date",
            }
        )
    )

    dates = dates.dropna(
        subset=[f"first_{out_prefix}_date", f"last_{out_prefix}_date"]
    ).copy()

    dates["end_date"] = dates[f"last_{out_prefix}_date"]

    if drug_keywords:
        pattern = "|".join(re.escape(str(x).upper()) for x in drug_keywords)

        med = medication[
            [patient_col, med_name_col, med_stop_col, med_datecreated_col]
        ].copy()

        med[patient_col] = med[patient_col].astype("string").str.strip()
        med[med_name_col] = med[med_name_col].astype("string").str.upper()
        med[med_stop_col] = pd.to_datetime(med[med_stop_col], errors="coerce")
        med[med_datecreated_col] = pd.to_datetime(med[med_datecreated_col], errors="coerce")

        med_cond = med[
            med[med_name_col].str.contains(pattern, regex=True, na=False)
        ].copy()

        if not med_cond.empty:
            med_end = (
                med_cond.groupby(patient_col)
                .agg(
                    last_stop=(med_stop_col, "max"),
                    last_created=(med_datecreated_col, "max"),
                )
                .reset_index()
            )

            med_end["last_drug_end"] = med_end["last_stop"].fillna(med_end["last_created"])

            dates = dates.merge(
                med_end[[patient_col, "last_stop", "last_created", "last_drug_end"]],
                on=patient_col,
                how="left",
                validate="1:1",
            )

            dates["end_date"] = dates[f"last_{out_prefix}_date"].where(
                ~dates["last_drug_end"].gt(dates[f"last_{out_prefix}_date"]),
                dates["last_drug_end"],
            )

    dates[f"{out_prefix}_duration_days"] = (
        (dates["end_date"] - dates[f"first_{out_prefix}_date"]).dt.days
    ).clip(lower=1)

    dates[f"{out_prefix}_duration_years"] = dates[f"{out_prefix}_duration_days"] / 365.0

    for c in ["last_stop", "last_created", "last_drug_end"]:
        if c not in dates.columns:
            dates[c] = pd.NaT

    return dates[
        [
            patient_col,
            f"first_{out_prefix}_date",
            f"last_{out_prefix}_date",
            "last_stop",
            "last_created",
            "last_drug_end",
            "end_date",
            f"{out_prefix}_duration_years",
        ]
    ]