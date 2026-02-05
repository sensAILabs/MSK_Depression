from __future__ import annotations
import pandas as pd

# medication_classification.py


import re
import pandas as pd

# ------------------------------------------------------------
# 1) ATC prefix rules (core)

CHRONIC_ATC_PREFIX = {
    "A10",   # diabetes drugs
    "B01",   # antithrombotic agents
    "C01",   # cardiac therapy
    "C02",   # antihypertensives
    "C03",   # diuretics
    "C07",   # beta blocking agents
    "C08",   # calcium channel blockers
    "C09",   # ACE inhibitors / ARBs
    "C10",   # lipid modifying agents (statins)
    "H03",   # thyroid therapy (levothyroxine)
    "N03",   # antiepileptics
    "N05",   # psycholeptics (many chronic; be conservative)
    "N06",   # psychoanaleptics (SSRIs etc; often chronic)
    "R03",   # drugs for obstructive airway diseases (asthma/COPD maintenance)
    "L01",   # antineoplastic
}

ACUTE_ATC_PREFIX = {
    "J01",   # systemic antibacterials (antibiotics) - usually acute
    "J02",   # systemic antifungals - often acute
    "J04",   # antimycobacterials - could be long, but often specific regimens; optional
    "J05",   # antivirals - mixed, but often course-based; optional
    "M01A",  # NSAIDs - often acute/pain/inflammation; mixed but ok for "acute-biased"
    "R05",   # cough and cold preparations
    "S01",   # ophthalmologicals (often short courses)
    "D06",   # topical antibiotics
}


ACUTE_PREFIX_EXCEPTIONS_TO_UNKNOWN = {
    # "M01A", 
    # "J05",
    # "J04",
}

# 2) text fallback（when Code_calc missing）
ACUTE_KEYWORDS = [
    r"\bAMOXICILLIN\b",
    r"\bAZITHROMYCIN\b",
    r"\bCEPHALEXIN\b",
    r"\bCIPROFLOXACIN\b",
    r"\bDOXYCYCLINE\b",
    r"\bCLINDAMYCIN\b",
    r"\bPENICILLIN\b",
    r"\bMETRONIDAZOLE\b",
    r"\bNITROFURANTOIN\b",
    r"\bTMP\b|\bTRIMETHOPRIM\b|\bSULFAMETHOXAZOLE\b",
    r"\bIBUPROFEN\b|\bNAPROXEN\b|\bDICLOFENAC\b",
    r"\bCOUGH\b|\bCOLD\b|\bINFLUENZA\b",
]

CHRONIC_KEYWORDS = [
    r"\bLEVOTHYROXINE\b",
    r"\bMETFORMIN\b",
    r"\bINSULIN\b",
    r"\bATORVASTATIN\b|\bROSUVASTATIN\b|\bSIMVASTATIN\b",
    r"\bAMLODIPINE\b|\bLISINOPRIL\b|\bLOSARTAN\b|\bRAMIPRIL\b",
    r"\bWARFARIN\b|\bAPIXABAN\b|\bRIVAROXABAN\b",
    r"\bSERTRALINE\b|\bCITALOPRAM\b|\bFLUOXETINE\b|\bESCITALOPRAM\b",
]

ACUTE_RE = re.compile("|".join(ACUTE_KEYWORDS), flags=re.IGNORECASE) if ACUTE_KEYWORDS else None
CHRONIC_RE = re.compile("|".join(CHRONIC_KEYWORDS), flags=re.IGNORECASE) if CHRONIC_KEYWORDS else None



def _match_prefix(atc: str, prefix_set: set[str]) -> bool:
    """
    True if ATC begins with any prefix in prefix_set.
    """
    for p in prefix_set:
        if atc.startswith(p):
            return True
    return False


def classify_medication_record(
    code_calc: str | None,
    name_calc: str | None = None,
    reason: str | None = None,
) -> str:
    """
    Return: "acute" / "chronic" / "unknown"
    Priority:
      1) ATC prefix (Code_calc)
      2) fallback keyword search on Name_calc / Reason
    """
    atc = code_calc

    if atc:
        if _match_prefix(atc, ACUTE_PREFIX_EXCEPTIONS_TO_UNKNOWN):
            return "unknown"

        if _match_prefix(atc, CHRONIC_ATC_PREFIX):
            return "chronic"
        if _match_prefix(atc, ACUTE_ATC_PREFIX):
            return "acute"

        return "unknown"

    text = " ".join([str(x) for x in [name_calc, reason] if x not in (None, "", "NA", "nan", "<NA>")])
    if not text:
        return "unknown"

    if CHRONIC_RE and CHRONIC_RE.search(text):
        return "chronic"
    if ACUTE_RE and ACUTE_RE.search(text):
        return "acute"

    return "unknown"


def add_medication_classification(
    df: pd.DataFrame,
    *,
    code_col: str = "Code_calc",
    name_col: str = "Name_calc",
    reason_col: str = "Reason",
    out_col: str = "med_class",
) -> pd.DataFrame:
    """
    Add:
      - df[out_col] = acute/chronic/unknown

    Rule:
      1) ATC prefix (Code_calc) decides first (fast)
      2) Only if Code_calc is missing/empty -> fallback keyword search on Name_calc/Reason
      3) chronic > acute priority (same as your row-wise function)
      4) "acute exception -> unknown" supported
    """

    for c in [code_col, name_col, reason_col]:
        if c not in df.columns:
            df[c] = pd.NA

    atc = (
    df[code_col]
    .astype("string")
    .fillna("")
    .str.strip()
    .str.upper()
)

    # start all as unknown
    out = pd.Series("unknown", index=df.index, dtype="string")

    # rows that actually have an ATC/code
    has_atc = atc.ne("")

    # --- ATC exception -> keep unknown (optional) ---
    exc_mask = pd.Series(False, index=df.index)
    if "ACUTE_PREFIX_EXCEPTIONS_TO_UNKNOWN" in globals() and ACUTE_PREFIX_EXCEPTIONS_TO_UNKNOWN:
        exc_mask = has_atc & atc.str.startswith(tuple(ACUTE_PREFIX_EXCEPTIONS_TO_UNKNOWN))

    # --- chronic / acute by prefix (only for rows with ATC and not exception) ---
    base_mask = has_atc & ~exc_mask

    if "CHRONIC_ATC_PREFIX" in globals() and CHRONIC_ATC_PREFIX:
        chronic_mask = base_mask & atc.str.startswith(tuple(CHRONIC_ATC_PREFIX))
        out.loc[chronic_mask] = "chronic"
    else:
        chronic_mask = pd.Series(False, index=df.index)

    if "ACUTE_ATC_PREFIX" in globals() and ACUTE_ATC_PREFIX:
        acute_mask = base_mask & atc.str.startswith(tuple(ACUTE_ATC_PREFIX))
        out.loc[acute_mask & out.eq("unknown")] = "acute"

    # --- fallback: ONLY when code is missing/empty ---
    need_fallback = ~has_atc

    if need_fallback.any():
        # combine name/reason into one text blob
        name_s = df[name_col].astype("string").fillna("").replace("<NA>", "").replace("NA", "").replace("nan", "")
        reason_s = df[reason_col].astype("string").fillna("").replace("<NA>", "").replace("NA", "").replace("nan", "")
        text = (name_s.str.strip() + " " + reason_s.str.strip()).str.strip()

        # chronic keywords first
        if "CHRONIC_RE" in globals() and CHRONIC_RE is not None:
            chronic_kw = need_fallback & text.str.contains(CHRONIC_RE, na=False)
            out.loc[chronic_kw] = "chronic"

        # acute keywords second (only still unknown)
        if "ACUTE_RE" in globals() and ACUTE_RE is not None:
            acute_kw = need_fallback & out.eq("unknown") & text.str.contains(ACUTE_RE, na=False)
            out.loc[acute_kw] = "acute"

    df[out_col] = out
    return df

import pandas as pd

def refine_med_class_by_duration(
    df: pd.DataFrame,
    *,
    class_col: str = "med_class",
    count_col: str = "DurationCount",
    unit_col: str = "DurationUnit",
    threshold_days: float = 14,
) -> pd.DataFrame:
    """
    Refine med_class in-place logic (returns a copy):

    If:
      - med_class == "unknown"
      - DurationUnit == "DAYS"
      - DurationCount < threshold_days

    Then:
      - set med_class = "acute"

    Otherwise:
      - leave med_class unchanged
    """
    df = df.copy()

    for c in [class_col, count_col, unit_col]:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in DataFrame")

    unit = (
        df[unit_col]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
        .replace({
            "DAY": "DAYS",
            "D": "DAYS",
            "DYS": "DAYS",
        })
    )

    count = pd.to_numeric(df[count_col], errors="coerce")

    mask = (
        df[class_col].eq("unknown") &
        unit.eq("DAYS") &
        count.lt(threshold_days)
    )

    df.loc[mask, class_col] = "acute"

    return df

def impute_startdate_for_acute(
    df: pd.DataFrame,
    *,
    med_class_col: str = "med_class",
    start_col: str = "StartDate",
    created_col: str = "DateCreated",
) -> pd.DataFrame:
    """
    Impute StartDate for acute medications only:
      - if med_class == 'acute'
      - and StartDate is missing
      - then StartDate := DateCreated

    Returns a copy of df.
    """
    df = df.copy()

    for c in [med_class_col, start_col, created_col]:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in DataFrame")

    mask = (
        df[med_class_col].eq("acute")
        & df[start_col].isna()
        & df[created_col].notna()
    )

    df.loc[mask, start_col] = df.loc[mask, created_col]

    return df

def extract_startdate_missing_unknown(
    df: pd.DataFrame,
    *,
    start_col: str = "StartDate",
    class_col: str = "med_class",
) -> pd.DataFrame:
    """
    Return a DataFrame containing rows where:
      - StartDate is missing
      - med_class == 'unknown'
    """
    # --- sanity checks ---
    for c in [start_col, class_col]:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in DataFrame")

    mask = df[start_col].isna() & df[class_col].eq("unknown")

    return df.loc[mask].copy()



def summarize_classification(df: pd.DataFrame, class_col: str = "med_class", top_n: int = 20) -> None:
    """
    Print distribution and top codes in each class (quick sanity check).
    """
    print(df[class_col].value_counts(dropna=False))

    if "Code_calc" in df.columns:
        for cls in ["acute", "chronic", "unknown"]:
            sub = df[df[class_col] == cls]
            print(f"\nTop ATC for {cls}:")
            print(sub["Code_calc"].astype("string").value_counts().head(top_n))





# acute_mask = medication["impute_startdate_ok"] & medication["StartDate"].isna() & medication["DateCreated"].notna()
# medication.loc[acute_mask, "StartDate"] = medication.loc[acute_mask, "DateCreated"]