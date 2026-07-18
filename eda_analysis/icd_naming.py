
import pandas as pd
import re


def build_icd9_name_map_from_encounter(
    encounter: pd.DataFrame,
    icd9_col: str = "icd9",
    name_calc_col: str = "DiagnosisText_calc",
    name_orig_col: str = "DiagnosisText_orig",
    allow_VE: bool = True,
) -> dict:
    df = encounter.copy()

    df[icd9_col] = (
        df[icd9_col]
        .astype("string")
        .str.strip()
        .replace({"nan": pd.NA, "NaN": pd.NA, "": pd.NA, "None": pd.NA})
    )

    # prefer calc, fallback to orig
    name = pd.Series(pd.NA, index=df.index, dtype="string")
    if name_calc_col in df.columns:
        name = df[name_calc_col].astype("string").str.strip()
    if name_orig_col in df.columns:
        name = name.fillna(df[name_orig_col].astype("string").str.strip())

    df = pd.DataFrame({icd9_col: df[icd9_col], "name": name}).dropna()

    # ICD-9-like pattern (optionally includes V/E codes)
    pattern = (
        r"^(?:\d{3}(?:\.\d{1,2})?|V\d{2}(?:\.\d{1,2})?|E\d{3}(?:\.\d)?)$"
        if allow_VE
        else r"^\d{3}(?:\.\d{1,2})?$"
    )

    df = df[df[icd9_col].str.match(pattern, na=False)]

    # most common name per full code
    return (
        df.groupby(icd9_col)["name"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )


def build_icd3_name_map_from_icd9_map(icd9_map: dict) -> dict:
    icd3_map = {}
    for code, name in icd9_map.items():
        m = re.match(r"^(\d{3})", str(code))
        if m and name:
            icd3_map.setdefault(m.group(1), name)
    return icd3_map


def format_icd9_list_with_names(
    icd9_list_str: str,
    icd9_map: dict,
    icd3_map: dict | None = None,
    keep_unknown: bool = False,
) -> str:
    if pd.isna(icd9_list_str) or str(icd9_list_str).strip() == "":
        return ""

    codes = [c.strip() for c in str(icd9_list_str).split(",") if c.strip()]
    out = []

    for code in codes:
        name = icd9_map.get(code)

        if name is None and icd3_map is not None:
            m = re.match(r"^(\d{3})", code)
            if m:
                name = icd3_map.get(m.group(1))

        if name:
            out.append(f"{code} ({name})")
        elif keep_unknown:
            out.append(f"{code} (Unknown)")

    return ", ".join(out)