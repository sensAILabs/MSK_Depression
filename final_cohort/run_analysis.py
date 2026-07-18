from pathlib import Path
import os
import sys
import logging
import warnings
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from eda.clinical_definitions import (
    MENTAL_CLASSES as _MENTAL_CLASSES,
    PAIN_CLASSES as _PAIN_CLASSES,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(os.environ.get("BIOSTAT_ROOT", str(PROJECT_ROOT)))
OUTDIR = ROOT / "final_cohort" / "results" / "tables"

# Keep stable class order for output columns
MENTAL_CLASSES = sorted(_MENTAL_CLASSES)
PAIN_CLASSES = sorted(_PAIN_CLASSES)

# Time windows: name → max days from first_mental
WINDOWS = {
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "9m": 270,
    "1yr": 365,
    "2yr": 730,
    "3yr": 1095,
    "4yr": 1460,
    "5yr": 1825,
    "6yr": 2190,
    "7yr": 2555,
    "8yr": 2920,
    "9yr": 3285,
    "10yr+": None,
}
WINDOW_ORDER = list(WINDOWS.keys())

# Cumulative future MSK outcome labels for ML.
# These are patient-level labels copied to every row in final_analysis_long.
# Example: if time_to_msk_days <= 180, then msk_6m and all later labels are 1.
MSK_LABEL_WINDOWS = {
    "msk_3m": 90,
    "msk_6m": 180,
    "msk_9m": 270,
    "msk_1yr": 365,
    "msk_2yr": 730,
    "msk_3yr": 1095,
    "msk_4yr": 1460,
    "msk_5yr": 1825,
    "msk_6yr": 2190,
    "msk_7yr": 2555,
    "msk_8yr": 2920,
    "msk_9yr": 3285,
    "msk_10yr_plus": 4500,
}


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
    log.info("STEP 0: Validating input/output paths")
    required = [
        "baseline_table.csv",
        "outcome_table.csv",
        "mental_long.csv",
        "comorbidity_long.csv",
        "medication_long.csv",
    ]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for fname in required:
        p = OUTDIR / fname
        if p.exists():
            log.info(f"  CHECK OK  : Input file exists: {fname}")
        else:
            log.warning(f"  CHECK FAIL: Missing input file: {fname}")
            missing.append(fname)
    return missing

def maybe_run_build_pipeline(force_rebuild: bool = False):
    missing = validate_paths()
    if not force_rebuild and not missing:
        log.info("All intermediate tables already exist. Skipping build step.")
        return

    log.info("=" * 60)
    if force_rebuild:
        log.info("STEP 0B: Force rebuild requested; running build_analysis_df.py first")
    else:
        log.info("STEP 0B: Missing intermediate tables detected; running build_analysis_df.py first")

    build_script = Path(__file__).with_name("build_analysis_df.py")
    check(build_script.exists(), f"Build script exists next to run_analysis.py: {build_script}")

    cmd = [sys.executable, str(build_script)]
    env = os.environ.copy()
    env["BIOSTAT_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    log.info(f"  Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))

    missing_after = validate_paths()
    check(len(missing_after) == 0, "All required intermediate files exist after build step")

def load_tables():
    log.info("=" * 60)
    log.info("STEP 1: Loading intermediate tables")

    baseline = pd.read_csv(OUTDIR / "baseline_table.csv")
    outcome = pd.read_csv(OUTDIR / "outcome_table.csv")
    mental = pd.read_csv(OUTDIR / "mental_long.csv")
    comorbidity = pd.read_csv(OUTDIR / "comorbidity_long.csv")
    medication = pd.read_csv(OUTDIR / "medication_long.csv")

    # Standardise Patient_ID
    for df in [baseline, outcome, mental, comorbidity, medication]:
        df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()

    for df, cols in [
        (baseline,    ["first_mental", "first_msk"]),
        (outcome,     ["first_mental", "first_msk"]),
        (mental,      ["first_mental", "DateCreated"]),
        (comorbidity, ["first_mental", "DateCreated"]),
        (medication,  ["first_mental", "first_msk", "med_date",
                       "med_start_date", "med_stop_date", "med_created_date"]),
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # Unify time column name in medication
    if "days_from_first_mental" in medication.columns:
        medication = medication.rename(
            columns={"days_from_first_mental": "time_since_mental_days"}
        )

    for col in ["Age_at_first_mental", "pre_exist_count", "pre_exist_binary"]:
        if col in baseline.columns:
            baseline[col] = pd.to_numeric(baseline[col], errors="coerce")

    if "time_to_msk_days" in outcome.columns:
        outcome["time_to_msk_days"] = pd.to_numeric(outcome["time_to_msk_days"], errors="coerce")

    for df in [mental, comorbidity, medication]:
        if "time_since_mental_days" in df.columns:
            df["time_since_mental_days"] = pd.to_numeric(df["time_since_mental_days"], errors="coerce")

    # Summary
    for name, df in [
        ("baseline",    baseline),
        ("outcome",     outcome),
        ("mental",      mental),
        ("comorbidity", comorbidity),
        ("medication",  medication),
    ]:
        log.info(f"  {name:<14}: rows={df.shape[0]:,}  patients={df['Patient_ID'].nunique():,}")

    n_cohort = baseline["Patient_ID"].nunique()
    check(n_cohort > 0, "baseline has patients")
    check(outcome["Patient_ID"].nunique() == n_cohort,
          "outcome has same patient count as baseline", fatal=False)
    check(mental["Patient_ID"].nunique() <= n_cohort,
          "mental patients ⊆ baseline patients")
    check(comorbidity["Patient_ID"].nunique() <= n_cohort,
          "comorbidity patients ⊆ baseline patients")
    # Make the last window cover the entire cohort
    max_followup = int(outcome["time_to_msk_days"].max())

    WINDOWS["10yr+"] = max_followup + 1
    MSK_LABEL_WINDOWS["msk_10yr_plus"] = max_followup + 1

    log.info(f"10yr+ window automatically set to {max_followup + 1} days")
    
    return baseline, outcome, mental, comorbidity, medication


def build_final_analysis(baseline, outcome, mental, comorbidity, medication):
    log.info("=" * 60)
    log.info("STEP 2: Building final_analysis (wide, one row per patient)")

    analysis = baseline.copy()

    analysis = analysis.merge(
        outcome[["Patient_ID", "first_msk", "first_msk_icd9_3",
                 "first_msk_subgroup", "time_to_msk_days"]],
        on="Patient_ID", how="left"
    )
    comorbidity_wide = (
        comorbidity
        .assign(flag=1)
        .pivot_table(index="Patient_ID", columns="system", values="flag", aggfunc="max")
        .fillna(0)
        .reset_index()
    )
    comorbidity_wide.columns = [
        "Patient_ID" if c == "Patient_ID" else f"has_{c}"
        for c in comorbidity_wide.columns
    ]
    analysis = analysis.merge(comorbidity_wide, on="Patient_ID", how="left")
    for col in analysis.columns:
        if col.startswith("has_"):
            analysis[col] = analysis[col].fillna(0).astype(int)

    mental_summary = (
        mental.groupby("Patient_ID")
        .agg(
            mental_visit_count=("DateCreated", "count"),
            mental_acute=("mental_course",   lambda x: int("acute"   in set(x))),
            mental_chronic=("mental_course",  lambda x: int("chronic" in set(x))),
            mental_symptom=("mental_course",  lambda x: int("symptom" in set(x))),
        )
        .reset_index()
    )
    analysis = analysis.merge(mental_summary, on="Patient_ID", how="left")
    for col in ["mental_visit_count", "mental_acute", "mental_chronic", "mental_symptom"]:
        analysis[col] = analysis[col].fillna(0)

    med_summary = (
        medication.groupby("Patient_ID")
        .agg(
            mental_med_count=("mental_related", "sum"),
            pain_med_count=("pain_related", "sum"),
            mental_med_binary=("mental_related", "max"),
            pain_med_binary=("pain_related", "max"),
            n_unique_drugs_overall=("standard_name", "nunique"),
            n_unique_clinical_subclasses_overall=("clinical_subclass", "nunique"),
        )
        .reset_index()
    )
    
    analysis = analysis.merge(med_summary, on="Patient_ID", how="left")
    for col in ["mental_med_count", "pain_med_count", "mental_med_binary", "pain_med_binary"]:
        analysis[col] = analysis[col].fillna(0)

    # Fix duplicate first_mental columns if present
    if "first_mental_x" in analysis.columns:
        analysis = analysis.rename(columns={"first_mental_x": "first_mental"})
        analysis = analysis.drop(columns=["first_mental_y"], errors="ignore")

    analysis["msk_event"] = analysis["first_msk"].notna().astype(int)

    log.info(f"  final_analysis shape  : {analysis.shape}")
    log.info(f"  patients              : {analysis['Patient_ID'].nunique():,}")
    log.info(f"  msk_event rate        : {analysis['msk_event'].mean():.2%}")
    log.info(f"  Missing per column (non-zero only):")
    miss = analysis.isna().sum()
    log.info(miss[miss > 0].to_string())

    check(analysis["Patient_ID"].nunique() == baseline["Patient_ID"].nunique(),
      "final_analysis covers all cohort patients")
    check(analysis["Patient_ID"].duplicated().sum() == 0,
      "final_analysis has unique Patient_IDs")

    # Re-map Patient_ID to sequential integers starting from 1
    analysis = analysis.sort_values("Patient_ID").reset_index(drop=True)
    analysis["Patient_ID_old"] = analysis["Patient_ID"]
    analysis["Patient_ID"] = np.arange(1, len(analysis) + 1)

    check(analysis["Patient_ID"].min() == 1, "final_analysis Patient_ID starts at 1")
    check(analysis["Patient_ID"].max() == len(analysis), "final_analysis Patient_ID is consecutive")

    return analysis 

def prepare_medication_for_windows(medication, baseline):
    log.info("=" * 60)
    log.info("STEP 3: Preparing LLM-classified medication for time-window analysis")

    medication = medication.copy()

    medication["Name_calc"] = medication["Name_calc"].astype(str).str.strip().str.upper()
    medication["standard_name"] = medication["standard_name"].astype(str).str.strip().str.lower()
    medication["major_group"] = medication["major_group"].astype(str).str.strip()
    medication["clinical_subclass"] = medication["clinical_subclass"].astype(str).str.strip().str.lower()
    medication["pharmacologic_subclass"] = medication["pharmacologic_subclass"].astype(str).str.strip().str.lower()

    for col in ["mental_related", "pain_related", "chronic_disease_related", "multi_category"]:
        medication[col] = (
            pd.to_numeric(medication[col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    log.info(f"  medication rows: {medication.shape[0]:,}")
    log.info("  clinical_subclass distribution:")
    log.info(medication["clinical_subclass"].value_counts(dropna=False).head(50).to_string())

    # Prepare active medication timeline using start/stop dates.
    # Missing stop date is treated as still active.
    medication = prepare_active_medication_timeline(medication)

    n_patients_total = baseline["Patient_ID"].nunique()

    mental_med = medication[medication["mental_related"] == 1].copy()
    pain_med = medication[medication["pain_related"] == 1].copy()

    def drug_pct_table(df):
        if df.empty:
            return pd.DataFrame(columns=["n_patients", "percentage"])
        return (
            df[["Patient_ID", "standard_name"]]
            .drop_duplicates()
            .groupby("standard_name")["Patient_ID"]
            .nunique()
            .sort_values(ascending=False)
            .rename("n_patients")
            .to_frame()
            .assign(percentage=lambda d: d["n_patients"] / n_patients_total)
        )

    mental_drug_pct = drug_pct_table(mental_med)
    pain_drug_pct = drug_pct_table(pain_med)
   
    ppt_summary = (
        medication[
            ["clinical_subclass", "standard_name"]
        ]
        .dropna()
        .drop_duplicates()
        .sort_values(["clinical_subclass", "standard_name"])
        .groupby("clinical_subclass")["standard_name"]
        .apply(lambda x: "; ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={
            "clinical_subclass": "Medication class",
            "standard_name": "Included medications",
        })
    )

    ppt_summary["N_medications"] = ppt_summary["Included medications"].apply(
        lambda x: len([i for i in str(x).split(";") if i.strip()])
    )

    ppt_summary.to_csv(
        OUTDIR / "medication_classification_summary_for_ppt.csv",
        index=False,
    )

    log.info(
        f"Saved medication classification summary for PPT: "
        f"{OUTDIR / 'medication_classification_summary_for_ppt.csv'}"
    )


    return medication, mental_drug_pct, pain_drug_pct


# Helper summarise functions for time-window loop
def parse_class_list(x):
    if pd.isna(x) or str(x).strip() in ("", "nan"):
        return set()
    return set(i.strip() for i in str(x).replace(",", ";").split(";") if i.strip())


def add_class_binary_columns(df):
    """Add one binary column per drug class from the class_list columns."""
    df = df.copy()
    mental_sets = df.get("Mental_related_medication_class_list",
                         pd.Series([""] * len(df))).apply(parse_class_list)
    pain_sets   = df.get("Pain_related_medication_class_list",
                         pd.Series([""] * len(df))).apply(parse_class_list)
    for cls in MENTAL_CLASSES:
        df[f"mental_{cls}"] = mental_sets.apply(lambda s: int(cls in s))
    for cls in PAIN_CLASSES:
        df[f"pain_{cls}"] = pain_sets.apply(lambda s: int(cls in s))
    return df


def summarize_mental(df_window: pd.DataFrame) -> pd.DataFrame:
    if df_window.empty:
        return pd.DataFrame(columns=[
            "Patient_ID", "Mental_Diagnosis_count", "Mental_visit_count",
            "Mental_Diagnosis_list", "Mental_Diagnosis_ICD9_list",
            "Mental_Group_acute", "Mental_Group_chronic",
            "Mental_Group_symptom", "Overall_Mental_group","Last_mental_visit_day",
        ])
    df_window = df_window.sort_values(["Patient_ID", "DateCreated"])
    out = (
        df_window.groupby("Patient_ID")
        .agg(
            Mental_visit_count=("DateCreated", "count"),
            Last_mental_visit_day=("time_since_mental_days", "max"),
            Mental_Diagnosis_count=(
                "mental_subgroup",
                lambda x: len(list(dict.fromkeys(x.dropna().astype(str))))
            ),
            Mental_Diagnosis_list=(
                "mental_subgroup",
                lambda x: "; ".join(list(dict.fromkeys(x.dropna().astype(str))))
            ),
            Mental_Diagnosis_ICD9_list=(
                "icd9_split",
                lambda x: "; ".join(list(dict.fromkeys(x.dropna().astype(str))))
            ),
            Mental_Group_acute=(  "mental_course", lambda x: int((x == "acute").any())),
            Mental_Group_chronic=("mental_course", lambda x: int((x == "chronic").any())),
            Mental_Group_symptom=("mental_course", lambda x: int((x == "symptom").any())),
        )
        .reset_index()
    )

    def classify_overall(row):
        a, c, s = row["Mental_Group_acute"] == 1, row["Mental_Group_chronic"] == 1, row["Mental_Group_symptom"] == 1
        if a and c:          return "both"
        if c and not a:      return "chronic"
        if a and not c:      return "acute"
        if s and not a and not c: return "symptom"
        if a and c and s:    return "both"
        return "none"

    out["Overall_Mental_group"] = out.apply(classify_overall, axis=1)
    return out


COMORBIDITY_SYSTEMS = [
    "cancer",
    "endocrine",
    "cardiovascular",
    "respiratory",
    "genitourinary",
]

def summarize_comorbidity(df_window: pd.DataFrame) -> pd.DataFrame:
    multi_cols = [f"Multi_{s}" for s in COMORBIDITY_SYSTEMS]

    if df_window.empty:
        return pd.DataFrame(columns=["Patient_ID", "Multimorbidity_count", "Comorbidity_visit_count"] + multi_cols)

    df_window = df_window[
        df_window["system"].isin(COMORBIDITY_SYSTEMS)
    ].copy()

    out = (
        df_window.groupby("Patient_ID")
        .agg(
            systems=("system", lambda x: sorted(set(v for v in x.dropna()))),
            Comorbidity_visit_count=("system", "count"),
        )
        .reset_index()
    )

    out["Multimorbidity_count"] = out["systems"].apply(len)

    for s in COMORBIDITY_SYSTEMS:
        out[f"Multi_{s}"] = out["systems"].apply(lambda xs: int(s in xs))

    return out[["Patient_ID", "Multimorbidity_count", "Comorbidity_visit_count"] + multi_cols]


def prepare_active_medication_timeline(medication: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare medication start/stop dates for active-medication timeline.

    Active at a window if:
        active_start_days <= window_days
        AND (active_stop_days is missing OR active_stop_days >= window_days)

    Rules:
    - Missing stop date is treated as still active.
    - Missing start date falls back to med_date, then med_created_date.
    - If stop_date < start_date, stop date is treated as missing/still active.
    """
    medication = medication.copy()

    date_cols = ["first_mental", "med_date", "med_start_date", "med_stop_date", "med_created_date"]
    for col in date_cols:
        if col in medication.columns:
            medication[col] = pd.to_datetime(medication[col], errors="coerce")

    n_rows = len(medication)
    if n_rows == 0:
        log.warning("  Medication table is empty; active timeline will be empty")
        return medication

    log.info("  Medication date missingness check:")
    for col in ["med_date", "med_start_date", "med_stop_date", "med_created_date"]:
        if col in medication.columns:
            n_missing = medication[col].isna().sum()
            log.info(f"    {col:<16}: missing={n_missing:,} / {n_rows:,} ({n_missing / n_rows:.1%})")
        else:
            log.warning(f"    {col:<16}: column not found")

    # Check missing stop date by clinical subclass because model uses subclass-level medication variables.
    if "med_stop_date" in medication.columns and "clinical_subclass" in medication.columns:
        log.info("  STOP DATE MISSINGNESS BY CLINICAL SUBCLASS")
        subclass_missing = (
            medication
            .groupby("clinical_subclass", dropna=False)
            .agg(
                n_records=("med_stop_date", "size"),
                n_missing=("med_stop_date", lambda x: x.isna().sum()),
            )
            .assign(missing_pct=lambda d: 100 * d["n_missing"] / d["n_records"])
            .sort_values("missing_pct", ascending=False)
        )
        log.info(subclass_missing.head(30).to_string())
        subclass_missing.to_csv(OUTDIR / "stop_date_missing_by_subclass.csv")

    # Use start date as the primary activation date.
    # If start date is missing, fall back to med_date, then med_created_date.
    medication["active_start_date"] = medication.get("med_start_date")
    if "med_date" in medication.columns:
        medication["active_start_date"] = medication["active_start_date"].combine_first(medication["med_date"])
    if "med_created_date" in medication.columns:
        medication["active_start_date"] = medication["active_start_date"].combine_first(medication["med_created_date"])

    medication["active_stop_date"] = medication.get("med_stop_date")

    medication["active_start_days"] = (
        medication["active_start_date"] - medication["first_mental"]
    ).dt.days
    medication["active_stop_days"] = (
        medication["active_stop_date"] - medication["first_mental"]
    ).dt.days

    missing_active_start = medication["active_start_days"].isna().sum()
    log.info(
        "  active_start_days missing after fallback: "
        f"{missing_active_start:,} / {n_rows:,} ({missing_active_start / n_rows:.1%})"
    )

    missing_active_stop = medication["active_stop_days"].isna().sum()
    log.info(
        "  active_stop_days missing; treated as still active: "
        f"{missing_active_stop:,} / {n_rows:,} ({missing_active_stop / n_rows:.1%})"
    )

    invalid_stop = (
        medication["active_start_days"].notna()
        & medication["active_stop_days"].notna()
        & (medication["active_stop_days"] < medication["active_start_days"])
    )
    n_invalid = int(invalid_stop.sum())
    log.info(f"  invalid stop before start rows: {n_invalid:,}")

    if n_invalid > 0:
        invalid_cols = [
            "Patient_ID", "standard_name", "clinical_subclass",
            "med_start_date", "med_stop_date", "med_date", "med_created_date", "first_mental",
            "active_start_days", "active_stop_days",
        ]
        invalid_cols = [c for c in invalid_cols if c in medication.columns]
        medication.loc[invalid_stop, invalid_cols].to_csv(
            OUTDIR / "invalid_medication_stop_before_start.csv",
            index=False,
        )
        
    invalid_stop = (
        medication["active_start_days"].notna()
        & medication["active_stop_days"].notna()
        & (medication["active_stop_days"] < medication["active_start_days"])
    )

    n_invalid = int(invalid_stop.sum())
    log.info(f"  invalid stop before start rows: {n_invalid:,}")

    if n_invalid > 0:

        invalid_cols = ["Patient_ID","standard_name","clinical_subclass",
            "med_start_date","med_stop_date","med_date","med_created_date",
            "first_mental","active_start_days","active_stop_days",]

        invalid_cols = [c for c in invalid_cols if c in medication.columns]

        medication.loc[invalid_stop, invalid_cols].to_csv(
            OUTDIR / "invalid_medication_stop_before_start.csv",
            index=False,
        )

        log.info(f"  Removing {n_invalid:,} invalid medication rows")

        medication = medication.loc[~invalid_stop].copy()

    return medication

def active_medication_window(medication: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Select medications active at a given time window.

    Missing stop date means the medication is assumed to remain active.
    """
    if medication.empty:
        return medication.copy()

    if "active_start_days" not in medication.columns or "active_stop_days" not in medication.columns:
        medication = prepare_active_medication_timeline(medication)

    active_mask = (
        medication["active_start_days"].notna()
        & (medication["active_start_days"] <= days)
        & (
            medication["active_stop_days"].isna()
            | (medication["active_stop_days"] >= days)
        )
    )
    return medication.loc[active_mask].copy()

def summarize_medication(df_window: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize active medications within one time window.

    Important:
    - Overall burden variables use all active medications.
    - Mental/pain burden variables use mental_related / pain_related flags.
    - medclass_* columns are created ONLY for allowed mental + pain classes.
      This prevents diabetes/GI/respiratory/etc. medication classes from entering final_analysis_long.
    """
    base_columns = [
        "Patient_ID",
        "n_unique_drugs",
        "n_mental_drugs",
        "n_pain_drugs",
        "Any_Mental_related_medication",
        "Any_Pain_related_medication",
        "Mental_related_medication_class_count",
        "Mental_related_medication_drug_count",
        "Pain_related_medication_class_count",
        "Pain_related_medication_drug_count",
        "Mental_related_medication_class_list",
        "Mental_related_medication_drug_list",
        "Pain_related_medication_class_list",
        "Pain_related_medication_drug_list",
    ]

    if df_window.empty:
        return pd.DataFrame(columns=base_columns)

    df_window = df_window.copy()

    # Standardize fields
    df_window["standard_name"] = df_window["standard_name"].astype(str).str.strip().str.lower()
    df_window["major_group"] = df_window["major_group"].astype(str).str.strip()
    df_window["clinical_subclass"] = df_window["clinical_subclass"].astype(str).str.strip().str.lower()
    df_window["pharmacologic_subclass"] = df_window["pharmacologic_subclass"].astype(str).str.strip().str.lower()

    for col in ["mental_related", "pain_related", "chronic_disease_related", "multi_category"]:
        if col in df_window.columns:
            df_window[col] = (
                pd.to_numeric(df_window[col], errors="coerce")
                .fillna(0)
                .astype(int)
            )

    # Overall active medication burden
    out = (
        df_window.groupby("Patient_ID")
        .agg(
            n_unique_drugs=("standard_name", lambda x: x.dropna().nunique()),
            n_mental_drugs=(
                "standard_name",
                lambda x: df_window.loc[x.index]
                .loc[df_window.loc[x.index, "mental_related"] == 1, "standard_name"]
                .nunique(),
            ),
            n_pain_drugs=(
                "standard_name",
                lambda x: df_window.loc[x.index]
                .loc[df_window.loc[x.index, "pain_related"] == 1, "standard_name"]
                .nunique(),
            ),
        )
        .reset_index()
    )

    mental_sub = df_window[df_window["mental_related"] == 1].copy()
    pain_sub = df_window[df_window["pain_related"] == 1].copy()

    def _summarise_group(sub: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if sub.empty:
            return pd.DataFrame(columns=["Patient_ID"])

        return (
            sub.groupby("Patient_ID")
            .agg(
                **{
                    f"{prefix}_medication_class_count": (
                        "clinical_subclass",
                        lambda x: x.dropna().nunique(),
                    )
                },
                **{
                    f"{prefix}_medication_drug_count": (
                        "standard_name",
                        lambda x: x.dropna().nunique(),
                    )
                },
                **{
                    f"{prefix}_medication_class_list": (
                        "clinical_subclass",
                        lambda x: "; ".join(sorted(set(x.dropna()))),
                    )
                },
                **{
                    f"{prefix}_medication_drug_list": (
                        "standard_name",
                        lambda x: "; ".join(sorted(set(x.dropna()))),
                    )
                },
            )
            .reset_index()
            .assign(**{f"Any_{prefix}_medication": 1})
        )

    mental_sum = _summarise_group(mental_sub, "Mental_related")
    pain_sum = _summarise_group(pain_sub, "Pain_related")

    out = out.merge(mental_sum, on="Patient_ID", how="left")
    out = out.merge(pain_sum, on="Patient_ID", how="left")

    # Create class-specific columns ONLY for mental/pain medication classes.
    allowed_med_classes = set(MENTAL_CLASSES) | set(PAIN_CLASSES)
    class_df = df_window[df_window["clinical_subclass"].isin(allowed_med_classes)].copy()

    if not class_df.empty:
        class_count_wide = (
            class_df.groupby(["Patient_ID", "clinical_subclass"])
            .size()
            .reset_index(name="count")
            .pivot_table(
                index="Patient_ID",
                columns="clinical_subclass",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )

        class_count_wide = class_count_wide.rename(
            columns={
                c: f"medclass_{c}_count"
                for c in class_count_wide.columns
                if c != "Patient_ID"
            }
        )
        out = out.merge(class_count_wide, on="Patient_ID", how="left")

        class_binary_wide = (
            class_df[["Patient_ID", "clinical_subclass"]]
            .drop_duplicates()
            .assign(value=1)
            .pivot_table(
                index="Patient_ID",
                columns="clinical_subclass",
                values="value",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
        )

        class_binary_wide = class_binary_wide.rename(
            columns={
                c: f"medclass_{c}"
                for c in class_binary_wide.columns
                if c != "Patient_ID"
            }
        )
        out = out.merge(class_binary_wide, on="Patient_ID", how="left")

    # Fill expected summary columns
    for col in [
        "Any_Mental_related_medication",
        "Any_Pain_related_medication",
        "Mental_related_medication_class_count",
        "Mental_related_medication_drug_count",
        "Pain_related_medication_class_count",
        "Pain_related_medication_drug_count",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
        else:
            out[col] = 0

    for col in [
        "Mental_related_medication_class_list",
        "Mental_related_medication_drug_list",
        "Pain_related_medication_class_list",
        "Pain_related_medication_drug_list",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna("")
        else:
            out[col] = ""

    # Fill medclass columns created above
    medclass_cols = [c for c in out.columns if c.startswith("medclass_")]
    for col in medclass_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    return out
def add_medication_change_features(final: pd.DataFrame, medication: pd.DataFrame | None = None) -> pd.DataFrame:
    final = final.copy()

    final["window"] = pd.Categorical(
        final["window"],
        categories=WINDOW_ORDER,
        ordered=True,
    )
    final = final.sort_values(["Patient_ID", "window"]).reset_index(drop=True)

    # 1) Existing count-change features
    all_medclass_cols = [c for c in final.columns if c.startswith("medclass_")]
    for col in all_medclass_cols:
        final[col] = pd.to_numeric(final[col], errors="coerce").fillna(0)

    med_count_cols = [
        c for c in final.columns
        if c.startswith("medclass_")
        and c.endswith("_count")
        and not c.endswith("_change")
    ]

    change_cols = [c for c in med_count_cols if c in final.columns]

    def first_nonzero_value(x):
        x = pd.to_numeric(x, errors="coerce").fillna(0)
        nonzero = x[x > 0]
        return 0 if nonzero.empty else nonzero.iloc[0]

    for col in change_cols:
        final[col] = pd.to_numeric(final[col], errors="coerce").fillna(0)

        first_observed = final.groupby("Patient_ID")[col].transform(first_nonzero_value)

        # final count - first non-zero count
        final[f"{col}_change"] = final[col] - first_observed


        log.info(
            f"  Added medication class change features "
            f"for {len(change_cols)} medication classes"
        )
    # 2) Parse active medication class patterns
    def _parse_classes(x):
        if pd.isna(x):
            return set()
        out = set()
        for item in str(x).replace(",", ";").split(";"):
            item = item.strip().lower()
            if item and item not in {"nan", "none", "<na>"}:
                out.add(item)
        return out

    mental_sets = final.get(
        "Mental_related_medication_class_list",
        pd.Series([""] * len(final), index=final.index),
    ).apply(_parse_classes)

    pain_sets = final.get(
        "Pain_related_medication_class_list",
        pd.Series([""] * len(final), index=final.index),
    ).apply(_parse_classes)

    all_sets = [m.union(p) for m, p in zip(mental_sets, pain_sets)]

    final["_med_pattern"] = [tuple(sorted(x)) for x in all_sets]
    final["active_mental_class_count"] = [len(x) for x in mental_sets]
    final["active_pain_class_count"] = [len(x) for x in pain_sets]

    first_patient_row = final["Patient_ID"].ne(final["Patient_ID"].shift(1))
    previous_pattern = final.groupby("Patient_ID")["_med_pattern"].shift(1)

    # 3) Addition / discontinuation counts
    def _transition_type(current, previous):
        if not isinstance(current, tuple) or not isinstance(previous, tuple):
            return "no_change"

        current_set = set(current)
        previous_set = set(previous)

        added = current_set - previous_set
        removed = previous_set - current_set

        if len(added) == 0 and len(removed) == 0:
            return "no_change"
        elif len(added) > 0 and len(removed) == 0:
            return "addition"
        elif len(added) == 0 and len(removed) > 0:
            return "discontinuation"
        else:
            return "switch"

    current_change_type = pd.Series(
        [_transition_type(cur, prev) for cur, prev in zip(final["_med_pattern"], previous_pattern)],
        index=final.index,
    )
    current_change_type.loc[first_patient_row] = "no_change"

    final["med_addition_count"] = (
        (current_change_type == "addition").astype(int)
        .groupby(final["Patient_ID"])
        .cumsum()
    )

    final["med_discontinuation_count"] = (
        (current_change_type == "discontinuation").astype(int)
        .groupby(final["Patient_ID"])
        .cumsum()
    )

    # Mental + pain overlap ratio
    combo_event = (
        (final["active_mental_class_count"] > 0)
        & (final["active_pain_class_count"] > 0)
    ).astype(int)

    final["mental_pain_overlap_days"] = 0.0

    def _merge_intervals(intervals):
        intervals = sorted(intervals)
        merged = []
        for start, stop in intervals:
            if pd.isna(start) or pd.isna(stop):
                continue
            start = float(start)
            stop = float(stop)
            if stop <= start:
                continue
            if not merged or start > merged[-1][1]:
                merged.append([start, stop])
            else:
                merged[-1][1] = max(merged[-1][1], stop)
        return merged

    def _interval_length(intervals):
        return float(sum(stop - start for start, stop in intervals if stop > start))

    def _overlap_length(intervals_a, intervals_b):
        i = j = 0
        total = 0.0
        while i < len(intervals_a) and j < len(intervals_b):
            start = max(intervals_a[i][0], intervals_b[j][0])
            stop = min(intervals_a[i][1], intervals_b[j][1])
            if stop > start:
                total += stop - start
            if intervals_a[i][1] <= intervals_b[j][1]:
                i += 1
            else:
                j += 1
        return float(total)

    if medication is not None and not medication.empty:
        med = medication.copy()

        for col in ["mental_related", "pain_related", "active_start_days", "active_stop_days"]:
            if col in med.columns:
                med[col] = pd.to_numeric(med[col], errors="coerce")

        med["Patient_ID"] = med["Patient_ID"].astype(str).str.strip()
        final_patient_id_as_str = final["Patient_ID"].astype(str).str.strip()
        med_groups = {pid: g for pid, g in med.groupby("Patient_ID", sort=False)}

        mental_med_days = []
        pain_med_days = []
        active_medication_days = []
        overlap_days = []

        for pid, window_day in zip(final_patient_id_as_str, final["window_days"]):
            current_day = pd.to_numeric(pd.Series([window_day]), errors="coerce").iloc[0]

            if pd.isna(current_day) or current_day <= 0 or pid not in med_groups:
                mental_med_days.append(0.0)
                pain_med_days.append(0.0)
                active_medication_days.append(0.0)
                overlap_days.append(0.0)
                continue

            current_day = float(current_day)
            g = med_groups[pid]

            def _intervals_for(flag_col):
                sub = g[g[flag_col] == 1]
                intervals = []

                for _, row in sub.iterrows():
                    start = row.get("active_start_days", np.nan)
                    stop = row.get("active_stop_days", np.nan)

                    if pd.isna(start):
                        continue

                    start = max(float(start), 0.0)
                    stop = current_day if pd.isna(stop) else min(float(stop), current_day)
                    start = min(start, current_day)

                    if stop > start:
                        intervals.append((start, stop))

                return _merge_intervals(intervals)

            mental_intervals = _intervals_for("mental_related")
            pain_intervals = _intervals_for("pain_related")

            mental_days = _interval_length(mental_intervals)
            pain_days = _interval_length(pain_intervals)
            overlap = _overlap_length(mental_intervals, pain_intervals)
            any_days = mental_days + pain_days - overlap

            mental_med_days.append(mental_days)
            pain_med_days.append(pain_days)
            active_medication_days.append(any_days)
            overlap_days.append(overlap)

        final["mental_med_days"] = mental_med_days
        final["pain_med_days"] = pain_med_days
        final["active_medication_days"] = active_medication_days
        final["mental_pain_overlap_days"] = overlap_days

    else:
        # Fallback if the medication table is unavailable:
        # estimate medication days from active class status within each retained window interval.
        previous_window_days = final.groupby("Patient_ID")["window_days"].shift(1).fillna(0)
        interval_days = (final["window_days"] - previous_window_days).clip(lower=0)

        mental_event = (final["active_mental_class_count"] > 0).astype(int)
        pain_event = (final["active_pain_class_count"] > 0).astype(int)
        any_event = ((mental_event == 1) | (pain_event == 1)).astype(int)

        final["mental_med_days"] = (
            mental_event * interval_days
        ).groupby(final["Patient_ID"]).cumsum()

        final["pain_med_days"] = (
            pain_event * interval_days
        ).groupby(final["Patient_ID"]).cumsum()

        final["mental_pain_overlap_days"] = (
            combo_event * interval_days
        ).groupby(final["Patient_ID"]).cumsum()

        final["active_medication_days"] = (
            any_event * interval_days
        ).groupby(final["Patient_ID"]).cumsum()

    # Ratio definition:
    # overlap ratio = days with BOTH mental and pain medications active
    #                 / days with ANY mental or pain medication active.
    # Therefore, days without medication are NOT included in the denominator.
    final["mental_pain_overlap_ratio"] = (
        final["mental_pain_overlap_days"]
        / final["active_medication_days"].replace(0, np.nan)
    ).fillna(0).clip(lower=0, upper=1)

    drop_cols = [
        "_med_pattern",
        "active_mental_class_count",
        "active_pain_class_count",
    ]

    final = final.drop(columns=drop_cols, errors="ignore")

    log.info(
        "  Added final compact medication trajectory features: "
        "addition/discontinuation counts and mental-pain overlap ratio "
        "(overlap days / any medication days)"
    )

    return final

def build_final_analysis_long(baseline, outcome, mental, comorbidity, medication, final_analysis):
    log.info("=" * 60)
    log.info("STEP 4: Building final_analysis_long (long format, all time windows)")

    # Base per-patient frame
    base = baseline.merge(
        outcome[["Patient_ID", "first_msk", "first_msk_icd9_3",
                 "first_msk_subgroup", "time_to_msk_days"]],
        on="Patient_ID", how="left"
    )
    if "first_mental_x" in base.columns:
        base = base.rename(columns={"first_mental_x": "first_mental"})
        base = base.drop(columns=["first_mental_y"], errors="ignore")

    rename_map = {
        "Age_at_first_mental":   "Age",
        "pre_exist_binary":      "Any_Pre_existing_condition",
        "pre_exist_count":       "Pre_existing_condition_count",
    }
    for old, new in rename_map.items():
        if old in base.columns:
            base = base.rename(columns={old: new})

    n_total = base["Patient_ID"].nunique()
    log.info(f"  Base table: {base.shape[0]:,} patients")

    rows = []

    for window_name, days in WINDOWS.items():
        log.info(f"  Processing window {window_name:>5} ({days:>5} days)...")

        df = base.copy()
        df["window"]      = window_name
        df["window_days"] = days

        # Outcome flag for this window
        df["MSK_Diagnosis"] = (df["time_to_msk_days"].notna() & (df["time_to_msk_days"] <= days)).astype(int)

        # Hide MSK detail for patients who haven't had MSK yet in this window
        df.loc[df["MSK_Diagnosis"] == 0, "first_msk_subgroup"] = pd.NA

        # Mental summary up to window
        mental_w  = mental[mental["time_since_mental_days"].notna() &
                           (mental["time_since_mental_days"] <= days)].copy()
        mental_s  = summarize_mental(mental_w)
        df = df.merge(mental_s, on="Patient_ID", how="left")

        df["Last_mental_visit_day"] = pd.to_numeric(
            df["Last_mental_visit_day"],
            errors="coerce"
        )

        df["Mental_visit_recency_days"] = (
            df["window_days"] - df["Last_mental_visit_day"]
        ).clip(lower=0)

        df["Recent_30d_mental_visit"] = (
            df["Mental_visit_recency_days"] <= 30
        ).astype(int)

        df.loc[df["Last_mental_visit_day"].isna(), "Mental_visit_recency_days"] = np.nan
        df.loc[df["Last_mental_visit_day"].isna(), "Recent_30d_mental_visit"] = 0

        df = df.drop(columns=["Last_mental_visit_day"], errors="ignore")

        # Comorbidity summary up to window
        comorb_w  = comorbidity[comorbidity["time_since_mental_days"].notna() &
                                (comorbidity["time_since_mental_days"] <= days)].copy()
        comorb_s  = summarize_comorbidity(comorb_w)
        df = df.merge(comorb_s, on="Patient_ID", how="left")

        # Medication summary active at this window.
        # Active means start_date <= window day and (stop_date >= window day or stop_date is missing).
        # Missing stop date is treated as still active.
        med_w = active_medication_window(medication, days)
        med_s = summarize_medication(med_w)
        df = df.merge(med_s, on="Patient_ID", how="left")
    
        # Auto-detect drug dummy columns
        mental_drug_cols = sorted(c for c in df.columns if c.startswith("MENTAL_DRUG_"))
        pain_drug_cols   = sorted(c for c in df.columns if c.startswith("PAIN_DRUG_"))
        drug_dummy_cols  = mental_drug_cols + pain_drug_cols

        # Fill zeros for numeric active-window features
        multi_cols = [f"Multi_{s}" for s in COMORBIDITY_SYSTEMS]

        zero_fill_cols = [
            "Mental_Diagnosis_count", "Mental_visit_count", "Mental_Group_acute", "Mental_Group_chronic",
            "Mental_Group_symptom", "Multimorbidity_count", "Comorbidity_visit_count",
            *multi_cols,
            "Any_Mental_related_medication", "Mental_related_medication_class_count",
            "Mental_related_medication_drug_count",
            "Any_Pain_related_medication", "Pain_related_medication_class_count",
            "Pain_related_medication_drug_count",
            "n_unique_drugs",
            "n_mental_drugs",
            "n_pain_drugs", "Mental_visit_recency_days","Recent_30d_mental_visit",
        ]
        for col in zero_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        for col in drug_dummy_cols:
            df[col] = df[col].fillna(0).astype(int)

        class_binary_cols = [f"mental_{c}" for c in MENTAL_CLASSES] + \
                            [f"pain_{c}" for c in PAIN_CLASSES]
        
        class_count_cols = [
            f"mental_{c}_count"
            for c in MENTAL_CLASSES
        ] + [
            f"pain_{c}_count"
            for c in PAIN_CLASSES
        ]
        
        for col in class_binary_cols + class_count_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        list_fill_cols = [
            "Mental_related_medication_class_list", "Mental_related_medication_drug_list",
            "Pain_related_medication_class_list",   "Pain_related_medication_drug_list",
        ]
        for col in list_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna("")

        if "Overall_Mental_group" in df.columns:
            df["Overall_Mental_group"] = df["Overall_Mental_group"].fillna("none")

        # Per-window check
        n_pts = df["Patient_ID"].nunique()
        n_msk = df["MSK_Diagnosis"].sum()
        check(n_pts == n_total,
              f"Window {window_name}: all {n_total} patients present (got {n_pts})")
        log.info(f"    MSK_Diagnosis=1: {n_msk:,} / {n_total:,} ({n_msk/n_total:.1%})")

        # Column ordering
        base_cols = [
            "Patient_ID", "window", "window_days",
            "Sex", "Age",
            "Any_Pre_existing_condition", "Pre_existing_condition_count",
            "first_mental", "first_msk",
            "Mental_Diagnosis_count", "Mental_visit_count", "Mental_Diagnosis_list", "Mental_Diagnosis_ICD9_list",
            "Mental_Group_acute", "Mental_Group_chronic", "Mental_Group_symptom",
            "Overall_Mental_group",
            "Multimorbidity_count", "Comorbidity_visit_count", *multi_cols,
            "Any_Mental_related_medication", "Mental_related_medication_class_count",
            "Mental_related_medication_drug_count",
            "Any_Pain_related_medication", "Pain_related_medication_class_count",
            "Pain_related_medication_drug_count", "Pre_cancer","Pre_endocrine","Pre_cardiovascular",
            "Pre_respiratory","Pre_genitourinary",
            "Mental_related_medication_class_list", "Mental_related_medication_drug_list",
            "Pain_related_medication_class_list",   "Pain_related_medication_drug_list",
            "MSK_Diagnosis", "first_msk_subgroup", "first_msk_icd9_3", "time_to_msk_days",
            "n_unique_drugs","n_mental_drugs","n_pain_drugs","Mental_visit_recency_days","Recent_30d_mental_visit",
        ]
        # Keep only mental/pain medication class variables.
        # This prevents unrelated medication classes such as diabetes/GI/respiratory
        # from entering final_analysis_long.
        allowed_med_classes = set(MENTAL_CLASSES) | set(PAIN_CLASSES)
        allowed_medclass_cols = set()
        for cls in allowed_med_classes:
            allowed_medclass_cols.add(f"medclass_{cls}")
            allowed_medclass_cols.add(f"medclass_{cls}_count")

        medclass_cols = sorted(
            c for c in df.columns
            if c.startswith("medclass_") and c in allowed_medclass_cols
        )

        all_cols = (
            [c for c in base_cols if c in df.columns]
            + medclass_cols
        )
        df = df[[c for c in all_cols if c in df.columns]].copy()
        rows.append(df)

    # Combine windows
    final = pd.concat(rows, ignore_index=True)

    # Rename for clarity
    final = final.rename(columns={
        "first_mental":      "first_mental_date",
        "first_msk":         "first_msk_date",
        "first_msk_subgroup":"MSK_subgroup",
        "first_msk_icd9_3":  "MSK_subgroup_ICD9",
    })

    # Show MSK detail only for rows where MSK has occurred
    for col in ["MSK_subgroup", "MSK_subgroup_ICD9", "time_to_msk_days"]:
        if col in final.columns:
            final[col] = final[col].where(final["MSK_Diagnosis"] == 1, pd.NA)

    if "MSK_subgroup" in final.columns:
        final["MSK_subgroup"] = final["MSK_subgroup"].astype("string")
    if "MSK_subgroup_ICD9" in final.columns:
        final["MSK_subgroup_ICD9"] = final["MSK_subgroup_ICD9"].astype("string")

    # Ordered categorical window
    final["window"] = pd.Categorical(final["window"], categories=WINDOW_ORDER, ordered=True)
    final = final.sort_values(["Patient_ID", "window"]).reset_index(drop=True)

    # Add cumulative future MSK outcome labels.
    # These are patient-level labels copied to every remaining row for that patient.
    # IMPORTANT: use the true first_msk_date - first_mental_date, because
    # time_to_msk_days was hidden/NA for rows before the event above.
    event_days_for_labels = (
        pd.to_datetime(final["first_msk_date"], errors="coerce")
        - pd.to_datetime(final["first_mental_date"], errors="coerce")
    ).dt.days

    for col, days in MSK_LABEL_WINDOWS.items():
        final[col] = (
            event_days_for_labels.notna()
            & (event_days_for_labels <= days)
        ).astype(int)

    # Truncate after first MSK event window.
    # Example: if first MSK is at 6m, keep 1m, 3m, 6m and remove 9m, 1yr, ...
    # Patients without MSK keep all available window rows.
    def trim_after_first_msk(group):
        hits = group.index[group["MSK_Diagnosis"] == 1].tolist()
        if not hits:
            return group
        return group.loc[:hits[0]]

    final = (
        final
        .groupby("Patient_ID", group_keys=False)
        .apply(trim_after_first_msk)
        .reset_index(drop=True)
    )

    # Add medication change / trajectory features within retained timeline
    final = add_medication_change_features(final, medication)

    # Remove detailed drug-level dummy columns, but keep class-level medication columns.
    detail_drug_cols = [
        c for c in final.columns
        if c.startswith("MENTAL_DRUG_") or c.startswith("PAIN_DRUG_")
    ]
    final = final.drop(columns=detail_drug_cols, errors="ignore")

    # Re-map to sequential Patient_ID used in final_analysis
    id_map = dict(zip(final_analysis["Patient_ID_old"], final_analysis["Patient_ID"]))
    final["Patient_ID"] = final["Patient_ID"].map(id_map)

    check(final["Patient_ID"].notna().all(), "final_analysis_long Patient_ID mapping successful")
    final["Patient_ID"] = final["Patient_ID"].astype(int)

    log.info(f"  final_analysis_long shape   : {final.shape}")
    log.info(f"  patients                    : {final['Patient_ID'].nunique():,}")
    log.info(f"  Overall MSK_Diagnosis rate  : {final['MSK_Diagnosis'].mean():.2%}")
    log.info(f"  Rows per patient (describe):")
    log.info(final.groupby("Patient_ID").size().describe().to_string())

    check(final["Patient_ID"].nunique() == n_total,
          "final_analysis_long covers all cohort patients")
    check(final["MSK_Diagnosis"].isin([0, 1]).all(),
          "MSK_Diagnosis is binary")

    return final

def drug_frequency_checks(final):
    log.info("=" * 60)
    log.info("STEP 5: Drug frequency checks on final_analysis_long")

    n_patients = final["Patient_ID"].nunique()

    for list_col, label in [
        ("Mental_related_medication_drug_list", "Mental"),
        ("Pain_related_medication_drug_list",   "Pain"),
    ]:
        if list_col not in final.columns:
            log.warning(f"  Column {list_col} not found – skipping")
            continue

        df = final.copy()
        df["_drug_list"] = df[list_col].fillna("").apply(
            lambda x: [i.strip() for i in x.split(";") if i.strip()]
        )
        exploded = df.explode("_drug_list")
        counts = (
            exploded[exploded["_drug_list"] != ""]
            .groupby("_drug_list")["Patient_ID"]
            .nunique()
            .sort_values(ascending=False)
        )
        freq = (counts / n_patients).sort_values(ascending=False)

        log.info(f"\n  {label} drug frequency (patient %):")
        log.info(freq.apply(lambda x: f"{x:.1%}").to_string())

def save_outputs(final_analysis, final_analysis_long, mental_drug_pct, pain_drug_pct):
    log.info("=" * 60)
    log.info("STEP 6: Saving output files")

    outputs = {
        "final_analysis.csv":          final_analysis,
        "final_analysis_long.csv":     final_analysis_long,
        "mental_drug_percentage.csv":  mental_drug_pct,
        "pain_drug_percentage.csv":    pain_drug_pct,
    }
    for fname, df in outputs.items():
        path = OUTDIR / fname
        df.to_csv(path, index=(fname.endswith("percentage.csv")))
        log.info(f"  Saved {fname:<35} rows={df.shape[0]:,}  path={path}")

    log.info("All outputs saved.")


def main():
    log.info("=" * 60)
    log.info("RUN ANALYSIS PIPELINE START")
    log.info("=" * 60)

    force_rebuild = os.environ.get("FORCE_REBUILD", "0") == "1"
    maybe_run_build_pipeline(force_rebuild=force_rebuild)
    baseline, outcome, mental, comorbidity, medication = load_tables()

    final_analysis = build_final_analysis(
        baseline, outcome, mental, comorbidity, medication
    )

    medication_prep, mental_drug_pct, pain_drug_pct = prepare_medication_for_windows(
        medication, baseline
    )

    final_analysis_long = build_final_analysis_long(
        baseline, outcome, mental, comorbidity, medication_prep, final_analysis
    )

    drug_frequency_checks(final_analysis_long)

    final_analysis = final_analysis.drop(columns=["Patient_ID_old"], errors="ignore")

    save_outputs(final_analysis, final_analysis_long, mental_drug_pct, pain_drug_pct)
    log.info("=" * 60)
    log.info("RUN ANALYSIS PIPELINE COMPLETE")
    log.info(f"Outputs written to: {OUTDIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()


