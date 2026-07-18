from pathlib import Path
import os
import sys
import warnings
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from eda.clinical_definitions import (
    classify_system_broad,
    classify_subgroup,
    classify_acute_chronic,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(os.environ.get("BIOSTAT_ROOT", str(PROJECT_ROOT)))

COHORT_CSV = ROOT / "final_cohort" / "results" / "tables" / "analysis_cohort_final.csv"
ENC_CSV = ROOT / "data" / "output" / "C4MEncounterDiagnosis_clean_1_31.csv"
MEDICATION_CSV = ROOT / "data" / "output" / "C4MMedication_clean_1_31.csv"
MED_LIBRARY_CSV = (ROOT/ "final_cohort"/ "results"/ "tables"/ "med"/ "medication_llm_library_filled.csv")
OUTDIR = ROOT / "final_cohort" / "results" / "tables"


def check(condition: bool, message: str, fatal: bool = True):
    if condition:
        log.info(f"  CHECK OK  : {message}")
    else:
        msg = f"  CHECK FAIL: {message}"
        if fatal:
            log.error(msg)
            raise AssertionError(msg)
        else:
            log.warning(msg)


def validate_paths():
    log.info("=" * 60)
    log.info("STEP 0: Validating input file paths")
    for p in [COHORT_CSV, ENC_CSV, MEDICATION_CSV]:
        check(p.exists(), f"File exists: {p}", fatal=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    check(OUTDIR.is_dir(), f"Output directory ready: {OUTDIR}")


def load_raw_data():
    log.info("=" * 60)
    log.info("STEP 1: Loading raw CSVs")

    cohort    = pd.read_csv(COHORT_CSV,    dtype=str)
    encounter = pd.read_csv(ENC_CSV,       dtype=str, low_memory=False)
    medication = pd.read_csv(MEDICATION_CSV, dtype=str, low_memory=False)

    log.info(f"  cohort     rows={cohort.shape[0]:,}  cols={cohort.shape[1]}")
    log.info(f"  encounter  rows={encounter.shape[0]:,}  cols={encounter.shape[1]}")
    log.info(f"  medication rows={medication.shape[0]:,}  cols={medication.shape[1]}")

    check("Patient_ID" in cohort.columns,    "cohort has Patient_ID")
    check("Patient_ID" in encounter.columns, "encounter has Patient_ID")
    check("Patient_ID" in medication.columns,"medication has Patient_ID")
    check("DateCreated" in encounter.columns, "encounter has DateCreated")
    check("StartDate" in medication.columns,  "medication has StartDate")
    check(cohort.shape[0] > 0, "cohort is not empty")

    return cohort, encounter, medication


def clean_raw_data(cohort, encounter, medication):
    log.info("=" * 60)
    log.info("STEP 2: Basic cleaning")

    # Strip Patient_ID whitespace
    for df in [cohort, encounter, medication]:
        df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()

    # Parse dates
    encounter["DateCreated"] = pd.to_datetime(encounter["DateCreated"], errors="coerce")

    for col in ["StartDate", "StopDate", "DateCreated"]:
        if col in medication.columns:
            medication[col] = pd.to_datetime(medication[col], errors="coerce")

    for col in ["first_resp", "first_mental", "first_msk",
                "mental_before_msk", "resp_before_msk", "exposure_start"]:
        if col in cohort.columns:
            cohort[col] = pd.to_datetime(cohort[col], errors="coerce")

    for col in ["BirthYear", "Age_at_mental", "time_to_msk_days"]:
        if col in cohort.columns:
            cohort[col] = pd.to_numeric(cohort[col], errors="coerce")

    # Checks
    missing_enc_date  = encounter["DateCreated"].isna().sum()
    missing_med_start = medication["StartDate"].isna().sum()
    log.info(f"  encounter DateCreated missing : {missing_enc_date:,}")
    log.info(f"  medication StartDate missing  : {missing_med_start:,}")
    check(missing_enc_date < encounter.shape[0] * 0.5,
          "Less than 50% of encounter dates missing", fatal=False)
    check(cohort["first_mental"].notna().sum() > 0,
          "cohort has at least some first_mental dates")

    return cohort, encounter, medication


def remap_patient_ids(cohort, encounter, medication):
    log.info("=" * 60)
    log.info("STEP 2B: Remapping Patient_ID to sequential integers")

    cohort = cohort.copy()
    encounter = encounter.copy()
    medication = medication.copy()

    id_map = (
        cohort[["Patient_ID"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Patient_ID")
        .reset_index(drop=True)
        .rename(columns={"Patient_ID": "Patient_ID_original"})
    )
    id_map["Patient_ID"] = np.arange(1, len(id_map) + 1)

    log.info(f"  Number of unique patients in mapping: {len(id_map):,}")
    check(id_map["Patient_ID_original"].duplicated().sum() == 0,
          "Original Patient_ID mapping is unique")
    check(id_map["Patient_ID"].min() == 1,
          "New Patient_ID starts from 1")
    check(id_map["Patient_ID"].max() == len(id_map),
          "New Patient_ID is consecutive")

    def apply_id_map(df, df_name, require_all=True):
        n_before = df["Patient_ID"].nunique()

        df = df.merge(
            id_map,
            left_on="Patient_ID",
            right_on="Patient_ID_original",
            how="left"
        )

        missing_new_id = df["Patient_ID_y"].isna().sum()

        if require_all:
            check(missing_new_id == 0,
                  f"{df_name}: all Patient_ID values mapped successfully")
        else:
            if missing_new_id > 0:
                log.warning(
                    f"  {df_name:<10}: dropping {missing_new_id:,} rows with Patient_ID not found in cohort mapping"
                )
            df = df[df["Patient_ID_y"].notna()].copy()

        df = df.drop(columns=["Patient_ID_x", "Patient_ID_original"])
        df = df.rename(columns={"Patient_ID_y": "Patient_ID"})
        df["Patient_ID"] = df["Patient_ID"].astype(int)

        n_after = df["Patient_ID"].nunique()
        check(n_after <= len(id_map),
              f"{df_name}: mapped patient count is valid")

        log.info(f"  {df_name:<10}: unique patients before={n_before:,}, after={n_after:,}")
        return df

    cohort = apply_id_map(cohort, "cohort", require_all=True)
    encounter = apply_id_map(encounter, "encounter", require_all=False)
    medication = apply_id_map(medication, "medication", require_all=False)

    return cohort, encounter, medication, id_map

def filter_and_classify_encounter(cohort, encounter):
    log.info("=" * 60)
    log.info("STEP 3: Filter encounter to cohort patients & classify ICD9")

    cohort_ids = set(cohort["Patient_ID"].dropna().unique())
    log.info(f"  Total cohort patients: {len(cohort_ids):,}")

    encounter_sub = encounter[encounter["Patient_ID"].isin(cohort_ids)].copy()
    log.info(f"  encounter_sub shape  : {encounter_sub.shape}")
    check(encounter_sub["Patient_ID"].nunique() <= len(cohort_ids),
          "encounter_sub patients ≤ cohort size")
    check(encounter_sub.shape[0] > 0, "encounter_sub is not empty")

    # Keep ICD9 only
    encounter_sub = encounter_sub[
        encounter_sub["DiagnosisCodeType_calc"] == "ICD9"
    ].copy()
    log.info(f"  After ICD9 filter    : {encounter_sub.shape}")
    check(encounter_sub.shape[0] > 0, "Non-zero ICD9 rows remain")

    # Clean diagnosis code
    encounter_sub["DiagnosisCode_calc"] = (
        encounter_sub["DiagnosisCode_calc"].astype(str).str.strip()
    )
    encounter_sub["icd9_split"] = (
        encounter_sub["DiagnosisCode_calc"].str.split(".").str[0].str.strip()
    )

    # Classify system
    encounter_sub["system"] = encounter_sub["icd9_split"].apply(classify_system_broad)

    # Drop rows missing key fields
    before = encounter_sub.shape[0]
    encounter_sub = encounter_sub.dropna(
        subset=["Patient_ID", "DateCreated", "icd9_split", "system"]
    ).copy()
    log.info(f"  Dropped {before - encounter_sub.shape[0]:,} rows missing key fields")
    log.info(f"  Final encounter_sub  : {encounter_sub.shape}")

    log.info("  System distribution:")
    log.info(encounter_sub["system"].value_counts(dropna=False).to_string())

    check("mental" in encounter_sub["system"].values, "mental system present")
    check(encounter_sub["Patient_ID"].nunique() > 0, "encounter_sub has patients after classify")

    return encounter_sub, cohort_ids


def build_baseline_table(cohort, encounter_sub):
    log.info("=" * 60)
    log.info("STEP 4: Building baseline table")

    # Identify first mental visit
    mental_long = encounter_sub[encounter_sub["system"] == "mental"].copy()
    first_mental = (
        mental_long.groupby("Patient_ID", as_index=False)["DateCreated"]
        .min()
        .rename(columns={"DateCreated": "first_mental"})
    )
    log.info(f"  Patients with first_mental : {first_mental.shape[0]:,}")
    check(first_mental.shape[0] > 0, "first_mental is non-empty")

    # Merge first_mental back to get baseline (records ≤ first_mental)
    enc_with_fm = encounter_sub.merge(first_mental, on="Patient_ID", how="inner")
    baseline_dx = enc_with_fm[
        enc_with_fm["DateCreated"] <= enc_with_fm["first_mental"]
    ].copy()
    log.info(f"  Baseline dx rows           : {baseline_dx.shape[0]:,}")

    # Pre-existing non-mental conditions
    baseline_nonmental = baseline_dx[baseline_dx["system"] != "mental"].copy()
    pre_exist = (
        baseline_nonmental.groupby("Patient_ID")["system"]
        .agg(lambda s: len(set(x for x in s.dropna())))
        .reset_index()
        .rename(columns={"system": "pre_exist_count"})
    )
    pre_exist["pre_exist_binary"] = (pre_exist["pre_exist_count"] > 0).astype(int)

    #Add pre-exsting resp/cardio/
    PRE_EXIST_SYSTEMS = [
        "cancer",
        "endocrine",
        "cardiovascular",
        "respiratory",
        "genitourinary",]

    pre_exist_wide = (
        baseline_nonmental
        .loc[
            baseline_nonmental["system"].isin(PRE_EXIST_SYSTEMS),
            ["Patient_ID", "system"]
        ]
        .drop_duplicates()
        .assign(value=1)
        .pivot_table(
            index="Patient_ID",
            columns="system",
            values="value",
            aggfunc="max",
            fill_value=0
        )
        .reset_index()
)

    pre_exist_wide = pre_exist_wide.rename(
        columns={s: f"Pre_{s}" for s in PRE_EXIST_SYSTEMS}
)

    for s in PRE_EXIST_SYSTEMS:
        col = f"Pre_{s}"
        if col not in pre_exist_wide.columns:
            pre_exist_wide[col] = 0

    # Build baseline table from cohort demographics
    required_demo_cols = ["Patient_ID", "Sex", "BirthYear"]
    check(all(c in cohort.columns for c in required_demo_cols),
          "cohort has Sex and BirthYear columns")

    baseline_table = cohort[required_demo_cols].copy()
    baseline_table = baseline_table.merge(first_mental, on="Patient_ID", how="left")
    baseline_table = baseline_table.merge(pre_exist, on="Patient_ID", how="left")
    baseline_table = baseline_table.merge(pre_exist_wide, on="Patient_ID", how="left")

    for s in PRE_EXIST_SYSTEMS:
        col = f"Pre_{s}"
        baseline_table[col] = baseline_table[col].fillna(0).astype(int)

    baseline_table["pre_exist_count"]  = baseline_table["pre_exist_count"].fillna(0).astype(int)
    baseline_table["pre_exist_binary"] = baseline_table["pre_exist_binary"].fillna(0).astype(int)
    baseline_table["Age_at_first_mental"] = (
        baseline_table["first_mental"].dt.year
        - pd.to_numeric(baseline_table["BirthYear"], errors="coerce")
    )

    log.info(f"  baseline_table before Sex filter: {baseline_table.shape}")
    log.info("  Missing per column:")
    log.info(baseline_table.isna().sum().to_string())

    # Drop patients missing Sex (cannot use in analysis)
    n_before = baseline_table.shape[0]
    baseline_table = baseline_table[baseline_table["Sex"].notna()].copy()
    n_dropped = n_before - baseline_table.shape[0]
    log.info(f"  Dropped {n_dropped} patients missing Sex")
    log.info(f"  baseline_table final shape: {baseline_table.shape}")

    check(baseline_table.shape[0] > 0, "baseline_table is non-empty after Sex filter")
    check(baseline_table["Sex"].isna().sum() == 0, "No missing Sex in baseline_table")
    check(baseline_table["Patient_ID"].duplicated().sum() == 0,
          "baseline_table has unique Patient_IDs")
    check(baseline_table["first_mental"].notna().sum() > 0,
          "baseline_table has some first_mental dates")

    log.info(f"  Sex distribution:\n{baseline_table['Sex'].value_counts().to_string()}")

    return baseline_table

def build_enc_long(cohort, encounter_sub, baseline_table):
    log.info("=" * 60)
    log.info("STEP 5: Building longitudinal encounter table (enc_long)")

    # Re-filter encounter to final cohort patients (after Sex drop)
    cohort_ids = set(baseline_table["Patient_ID"])
    encounter_sub = encounter_sub[encounter_sub["Patient_ID"].isin(cohort_ids)].copy()
    log.info(f"  encounter_sub after cohort re-filter: {encounter_sub.shape}")

    # Rebuild first_mental from encounter
    mental_long_full = encounter_sub[encounter_sub["system"] == "mental"].copy()
    first_mental = (
        mental_long_full.groupby("Patient_ID", as_index=False)["DateCreated"]
        .min()
        .rename(columns={"DateCreated": "first_mental"})
    )

    enc_with_fm = encounter_sub.merge(first_mental, on="Patient_ID", how="inner")

    # Merge first_msk from cohort
    cohort_msk = cohort[["Patient_ID", "first_msk"]].copy()
    cohort_msk["first_msk"] = pd.to_datetime(cohort_msk["first_msk"], errors="coerce")

    enc_long = enc_with_fm.merge(cohort_msk, on="Patient_ID", how="left")

    # Keep records: first_mental ≤ DateCreated ≤ first_msk
    enc_long = enc_long[
        enc_long["first_msk"].notna() &
        (enc_long["DateCreated"] >= enc_long["first_mental"]) &
        (enc_long["DateCreated"] <= enc_long["first_msk"])
    ].copy()

    # Time variables
    enc_long["time_since_mental_days"] = (
        enc_long["DateCreated"] - enc_long["first_mental"]
    ).dt.days
    enc_long["time_to_msk_days"] = (
        enc_long["first_msk"] - enc_long["first_mental"]
    ).dt.days

    log.info(f"  enc_long shape   : {enc_long.shape}")
    log.info(f"  enc_long patients: {enc_long['Patient_ID'].nunique():,}")

    neg_since = (enc_long["time_since_mental_days"] < 0).sum()
    neg_to    = (enc_long["time_to_msk_days"] < 0).sum()
    log.info(f"  Negative time_since_mental_days: {neg_since}")
    log.info(f"  Negative time_to_msk_days      : {neg_to}")
    check(neg_since == 0, "No negative time_since_mental_days")
    check(neg_to   == 0, "No negative time_to_msk_days")
    check(enc_long["Patient_ID"].nunique() > 0, "enc_long has patients")

    return enc_long



def build_outcome_table(cohort, baseline_table):
    log.info("=" * 60)
    log.info("STEP 6: Building outcome table")

    needed_cols = ["Patient_ID", "first_mental", "first_msk",
                   "first_msk_icd9_3", "first_msk_subgroup"]
    check(all(c in cohort.columns for c in needed_cols),
          "cohort has all outcome columns")

    outcome_table = cohort[needed_cols].copy()

    cohort_ids = set(baseline_table["Patient_ID"])
    outcome_table = outcome_table[outcome_table["Patient_ID"].isin(cohort_ids)].copy()

    outcome_table["first_mental"] = pd.to_datetime(outcome_table["first_mental"], errors="coerce")
    outcome_table["first_msk"]    = pd.to_datetime(outcome_table["first_msk"],    errors="coerce")

    outcome_table["time_to_msk_days"] = (
        outcome_table["first_msk"] - outcome_table["first_mental"]
    ).dt.days

    log.info(f"  outcome_table shape      : {outcome_table.shape}")
    log.info(f"  Patients with MSK event  : {outcome_table['first_msk'].notna().sum():,}")
    log.info(f"  Patients without MSK     : {outcome_table['first_msk'].isna().sum():,}")
    log.info(f"  MSK subgroup distribution (top 10):")
    log.info(outcome_table["first_msk_subgroup"].value_counts(dropna=False).head(10).to_string())

    check(outcome_table.shape[0] == len(cohort_ids),
          "outcome_table has one row per cohort patient")
    check(outcome_table["Patient_ID"].duplicated().sum() == 0,
          "outcome_table has unique Patient_IDs")

    return outcome_table


def build_mental_long(enc_long):
    log.info("=" * 60)
    log.info("STEP 7: Building mental_long table")

    enc_long = enc_long.copy()
    enc_long["mental_subgroup"] = enc_long["icd9_split"].apply(classify_subgroup)
    enc_long["mental_course"]   = enc_long["icd9_split"].apply(classify_acute_chronic)

    mental_long = enc_long[enc_long["system"] == "mental"].copy()
    mental_long = mental_long[
        [
            "Patient_ID",
            "DateCreated",
            "DiagnosisCode_calc",
            "icd9_split",
            "first_mental",
            "time_since_mental_days",
            "mental_subgroup",
            "mental_course",
        ]
    ].copy()

    mental_long = mental_long.sort_values(
        ["Patient_ID", "DateCreated", "DiagnosisCode_calc"]
    ).reset_index(drop=True)

    log.info(f"  mental_long shape   : {mental_long.shape}")
    log.info(f"  mental_long patients: {mental_long['Patient_ID'].nunique():,}")
    log.info(f"  mental_subgroup distribution:\n{mental_long['mental_subgroup'].value_counts(dropna=False).head(10).to_string()}")
    log.info(f"  mental_course distribution:\n{mental_long['mental_course'].value_counts(dropna=False).to_string()}")
    log.info(f"  Missing values:\n{mental_long.isna().sum().to_string()}")

    check(mental_long.shape[0] > 0, "mental_long is non-empty")
    check(mental_long["Patient_ID"].nunique() > 0, "mental_long has patients")
    check((mental_long["time_since_mental_days"] < 0).sum() == 0,
          "No negative time_since_mental_days in mental_long")

    return mental_long, enc_long

# Build comorbidity_long table
def build_comorbidity_long(enc_long):
    log.info("=" * 60)
    log.info("STEP 8: Building comorbidity_long table")

    comorbidity_long = enc_long[
        enc_long["system"].notna() &
        (enc_long["system"] != "mental") &
        (enc_long["system"] != "msk")
    ].copy()

    comorbidity_long = comorbidity_long[
        [
            "Patient_ID",
            "DateCreated",
            "DiagnosisCode_calc",
            "icd9_split",
            "system",
            "first_mental",
            "time_since_mental_days",
        ]
    ].copy()

    comorbidity_long = comorbidity_long.sort_values(
        ["Patient_ID", "DateCreated", "DiagnosisCode_calc"]
    ).reset_index(drop=True)

    log.info(f"  comorbidity_long shape   : {comorbidity_long.shape}")
    log.info(f"  comorbidity_long patients: {comorbidity_long['Patient_ID'].nunique():,}")
    log.info(f"  System distribution:\n{comorbidity_long['system'].value_counts(dropna=False).to_string()}")

    check(comorbidity_long.shape[0] > 0, "comorbidity_long is non-empty")
    check("mental" not in comorbidity_long["system"].values,
          "No mental entries in comorbidity_long")
    check("msk" not in comorbidity_long["system"].values,
          "No msk entries in comorbidity_long")

    return comorbidity_long


def build_medication_long(medication, baseline_table, outcome_table):
    log.info("=" * 60)
    log.info("STEP 9: Building medication_long table with LLM medication library")

    check(MED_LIBRARY_CSV.exists(), f"LLM medication library exists: {MED_LIBRARY_CSV}")

    med_library = pd.read_csv(MED_LIBRARY_CSV)

    med_library["med_name_calc"] = (
        med_library["med_name_calc"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    keep_library_cols = [
        "med_name_calc",
        "standard_name",
        "major_group",
        "pharmacologic_subclass",
        "clinical_subclass",
        "mental_related",
        "pain_related",
        "chronic_disease_related",
        "multi_category",
    ]

    missing_cols = [c for c in keep_library_cols if c not in med_library.columns]
    check(len(missing_cols) == 0, f"LLM library has required columns: {missing_cols}")

    for col in ["mental_related", "pain_related", "chronic_disease_related", "multi_category"]:
        med_library[col] = (
            pd.to_numeric(med_library[col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    cohort_ids = set(baseline_table["Patient_ID"])
    medication_sub = medication[medication["Patient_ID"].isin(cohort_ids)].copy()

    log.info(f"  medication_sub after cohort filter: {medication_sub.shape}")
    check(medication_sub["Patient_ID"].nunique() > 0, "medication_sub has patients")

    medication_sub["Name_calc"] = (
        medication_sub["Name_calc"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    medication_sub = medication_sub.merge(
        med_library[keep_library_cols],
        left_on="Name_calc",
        right_on="med_name_calc",
        how="left",
        validate="many_to_one",
    )

    unmatched = medication_sub["standard_name"].isna().sum()
    log.info(f"  Medication rows unmatched to LLM library: {unmatched:,}")

    medication_sub = medication_sub.merge(
        outcome_table[["Patient_ID", "first_mental", "first_msk"]],
        on="Patient_ID",
        how="left",
    )

    medication_sub["first_mental"] = pd.to_datetime(medication_sub["first_mental"], errors="coerce")
    medication_sub["first_msk"] = pd.to_datetime(medication_sub["first_msk"], errors="coerce")

    medication_sub["med_start_date"] = pd.to_datetime(medication_sub["StartDate"], errors="coerce")
    medication_sub["med_stop_date"] = pd.to_datetime(medication_sub["StopDate"], errors="coerce")
    medication_sub["med_created_date"] = pd.to_datetime(medication_sub["DateCreated"], errors="coerce")

    medication_sub["med_date"] = medication_sub["med_start_date"].fillna(
        medication_sub["med_created_date"]
    )

    medication_sub["days_from_first_mental"] = (
        medication_sub["med_date"] - medication_sub["first_mental"]
    ).dt.days

    medication_long = medication_sub[
        medication_sub["med_date"].notna()
        & medication_sub["first_mental"].notna()
        & medication_sub["first_msk"].notna()
        & (medication_sub["days_from_first_mental"] >= 0)
        & (medication_sub["med_date"] <= medication_sub["first_msk"])
    ].copy()

    medication_long = medication_long[
        [
            "Patient_ID",
            "med_date",
            "med_start_date",
            "med_stop_date",
            "med_created_date",
            "Name_calc",
            "standard_name",
            "major_group",
            "pharmacologic_subclass",
            "clinical_subclass",
            "mental_related",
            "pain_related",
            "chronic_disease_related",
            "multi_category",
            "first_mental",
            "first_msk",
            "days_from_first_mental",
        ]
    ].copy()

    medication_long = medication_long.sort_values(
        ["Patient_ID", "med_date", "Name_calc"]
    ).reset_index(drop=True)

    total_pts = baseline_table["Patient_ID"].nunique()
    mental_pts = medication_long.loc[medication_long["mental_related"] == 1, "Patient_ID"].nunique()
    pain_pts = medication_long.loc[medication_long["pain_related"] == 1, "Patient_ID"].nunique()

    log.info(f"  medication_long shape   : {medication_long.shape}")
    log.info(f"  medication_long patients: {medication_long['Patient_ID'].nunique():,}")
    log.info(f"  Mental med patients: {mental_pts:,} / {total_pts:,} ({mental_pts / total_pts:.1%})")
    log.info(f"  Pain med patients  : {pain_pts:,} / {total_pts:,} ({pain_pts / total_pts:.1%})")

    log.info("  Clinical subclass distribution:")
    log.info(medication_long["clinical_subclass"].value_counts(dropna=False).head(50).to_string())

    check((medication_long["days_from_first_mental"] < 0).sum() == 0,
          "No negative days_from_first_mental in medication_long")
    check(medication_long.shape[0] > 0, "medication_long is non-empty")

    return medication_long

def save_all_tables(baseline_table, outcome_table, mental_long, comorbidity_long, medication_long, id_map):
    log.info("=" * 60)
    log.info("STEP 10: Saving tables to disk")

    tables = {
        "baseline_table.csv"   : baseline_table,
        "outcome_table.csv"    : outcome_table,
        "mental_long.csv"      : mental_long,
        "comorbidity_long.csv" : comorbidity_long,
        "medication_long.csv"  : medication_long,
        "patient_id_map.csv"   : id_map,
    }

    for fname, df in tables.items():
        path = OUTDIR / fname
        df.to_csv(path, index=False)
        log.info(f"  Saved {fname:<25} rows={df.shape[0]:,}  path={path}")

    log.info("All tables saved successfully.")


# Cross-table patient count consistency check
def final_consistency_check(baseline_table, outcome_table, mental_long, comorbidity_long, medication_long):
    log.info("=" * 60)
    log.info("STEP 11: Cross-table consistency checks")

    cohort_ids = set(baseline_table["Patient_ID"])
    n = len(cohort_ids)

    log.info(f"  Final cohort size: {n:,}")

    # All table patients must be a subset of final cohort
    for name, df in [
        ("outcome_table",    outcome_table),
        ("mental_long",      mental_long),
        ("comorbidity_long", comorbidity_long),
        ("medication_long",  medication_long),
    ]:
        pts = set(df["Patient_ID"].unique())
        extra = pts - cohort_ids
        check(len(extra) == 0,
              f"{name}: all patients in cohort (extra={len(extra)})")
        log.info(f"  {name:<22}: {df['Patient_ID'].nunique():,} / {n:,} patients have records")

    # outcome_table should be 1-to-1 with baseline_table
    check(outcome_table["Patient_ID"].nunique() == n,
          "outcome_table covers all cohort patients", fatal=False)


def main():
    log.info("=" * 60)
    log.info("BUILD ANALYSIS DF PIPELINE START")
    log.info("=" * 60)

    validate_paths()
    cohort, encounter, medication = load_raw_data()
    cohort, encounter, medication = clean_raw_data(cohort, encounter, medication)
    cohort, encounter, medication, id_map = remap_patient_ids(cohort, encounter, medication)
    encounter_sub, _ = filter_and_classify_encounter(cohort, encounter)
    baseline_table   = build_baseline_table(cohort, encounter_sub)
    enc_long         = build_enc_long(cohort, encounter_sub, baseline_table)
    outcome_table    = build_outcome_table(cohort, baseline_table)
    mental_long, enc_long = build_mental_long(enc_long)
    comorbidity_long = build_comorbidity_long(enc_long)
    medication_long  = build_medication_long(medication, baseline_table, outcome_table)

    save_all_tables(baseline_table, outcome_table, mental_long, comorbidity_long, medication_long, id_map)
    final_consistency_check(baseline_table, outcome_table, mental_long, comorbidity_long, medication_long)

    log.info("=" * 60)
    log.info("BUILD ANALYSIS DF PIPELINE COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
