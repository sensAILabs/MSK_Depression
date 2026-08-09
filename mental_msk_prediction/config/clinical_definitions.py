"""
Clinical definitions used by the Mental-to-MSK cohort pipeline.

This module contains:
- broad ICD-9 disease-system ranges;
- official ICD-9 category names for Mental and MSK diagnoses;
- comorbidity-system lists used by EDA and modelling;
- ICD-9 classification and naming helpers;
- study medication definitions based on ATC Level 3;
- ATC classification and naming helpers.

Medication definitions are restricted to the prespecified mental-related
and pain-related ATC Level 3 groups used by this study.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


# Broad ICD-9 disease-system ranges
ICD9_RANGES: dict[str, tuple[float, float]] = {
    "cancer": (140, 239.9),
    "endocrine": (240, 279.9),
    "mental": (290, 319.9),
    "cardiovascular": (390, 459.9),
    "respiratory": (460, 519.9),
    "genitourinary": (580, 629.9),
    "msk": (710, 739.9),
}


# Comorbidity systems
# Used for descriptive EDA where mental disorders may be displayed
COMORBIDITY_SYSTEMS_EDA = [
    "cancer",
    "endocrine",
    "mental",
    "cardiovascular",
    "respiratory",
    "genitourinary",
]

# Used for prediction features. Mental and MSK diagnoses are excluded
# from these non-mental, non-MSK comorbidity categories.
COMORBIDITY_SYSTEMS_MODEL = [
    "cancer",
    "endocrine",
    "cardiovascular",
    "respiratory",
    "genitourinary",
]

SYS_ABBR = {
    "cancer": "CA",
    "endocrine": "E",
    "mental": "M",
    "cardiovascular": "C",
    "respiratory": "R",
    "genitourinary": "G",
    "msk": "MSK",
}


# Medication definitions: ATC Level 3
ATC_LEVEL3_DEFINITIONS: dict[str, dict[str, str]] = {
    # Mental-related medications
    "N05A": {
        "official_name": "Antipsychotics",
        "study_group": "mental",
    },
    "N05B": {
        "official_name": "Anxiolytics",
        "study_group": "mental",
    },
    "N05C": {
        "official_name": "Hypnotics and sedatives",
        "study_group": "mental",
    },
    "N06A": {
        "official_name": "Antidepressants",
        "study_group": "mental",
    },
    "N06B": {
        "official_name": "Psychostimulants and ADHD medications",
        "study_group": "mental",
    },
    "N06C": {
        "official_name": "Psycholeptic and psychoanaleptic combinations",
        "study_group": "mental",
    },

    # Pain-related medications
    "N02A": {
        "official_name": "Opioids",
        "study_group": "pain",
    },
    "N02B": {
        "official_name": "Other analgesics and antipyretics",
        "study_group": "pain",
    },
    "N02C": {
        "official_name": "Antimigraine preparations",
        "study_group": "pain",
    },
    "M01A": {
        "official_name": "Anti-inflammatory and antirheumatic products",
        "study_group": "pain",
    },
    "M02A": {
        "official_name": "Topical products for joint and muscular pain",
        "study_group": "pain",
    },
    "M03B": {
        "official_name": "Muscle relaxants",
        "study_group": "pain",
    },
}


MENTAL_ATC_LEVEL3: frozenset[str] = frozenset(
    code
    for code, definition in ATC_LEVEL3_DEFINITIONS.items()
    if definition["study_group"] == "mental"
)

PAIN_ATC_LEVEL3: frozenset[str] = frozenset(
    code
    for code, definition in ATC_LEVEL3_DEFINITIONS.items()
    if definition["study_group"] == "pain"
)

STUDY_ATC_LEVEL3: frozenset[str] = frozenset(
    ATC_LEVEL3_DEFINITIONS
)


def normalize_atc_level3(code: Any) -> str | None:
    """
    Normalize and validate an ATC Level 3 or complete ATC code.

    Examples
    --------
    " n06a " -> "N06A"
    "N06AB04" -> "N06A"
    None -> None
    """
    if code is None or pd.isna(code):
        return None

    normalized = re.sub(
        r"[^A-Z0-9]",
        "",
        str(code).strip().upper(),
    )

    if normalized in {"", "NAN", "NONE", "NULL", "<NA>"}:
        return None

    match = re.match(
        r"^([A-Z]\d{2}[A-Z])",
        normalized,
    )

    return match.group(1) if match else None


def get_atc_level3_definition(
    atc_code: Any,
) -> dict[str, str] | None:
    """
    Return the study definition for an ATC Level 3 or complete ATC code.
    """
    level3 = normalize_atc_level3(atc_code)

    if level3 is None:
        return None

    return ATC_LEVEL3_DEFINITIONS.get(level3)


def get_atc_level3_name(
    atc_code: Any,
) -> str | None:
    """
    Return the official ATC class name used in this study.
    """
    definition = get_atc_level3_definition(atc_code)

    if definition is None:
        return None

    return definition["official_name"]


def get_atc_study_group(
    atc_code: Any,
) -> str | None:
    """
    Return 'mental', 'pain', or None.
    """
    definition = get_atc_level3_definition(atc_code)

    if definition is None:
        return None

    return definition["study_group"]


def is_study_atc(
    atc_code: Any,
) -> bool:
    """
    Return True when an ATC code belongs to a selected study group.
    """
    return normalize_atc_level3(atc_code) in STUDY_ATC_LEVEL3


# Official Mental ICD-9 category names
MENTAL_ICD9_NAMES: dict[int, str] = {
    290: "Dementias",
    291: "Alcohol-induced mental disorders",
    292: "Drug-induced mental disorders",
    293: "Transient mental disorders due to conditions classified elsewhere",
    294: "Persistent mental disorders due to conditions classified elsewhere",
    295: "Schizophrenic disorders",
    296: "Episodic mood disorders",
    297: "Delusional disorders",
    298: "Other nonorganic psychoses",
    299: "Pervasive developmental disorders",
    300: "Anxiety, dissociative and somatoform disorders",
    301: "Personality disorders",
    302: "Sexual and gender identity disorders",
    303: "Alcohol dependence syndrome",
    304: "Drug dependence",
    305: "Nondependent abuse of drugs",
    306: "Physiological malfunction arising from mental factors",
    307: "Special symptoms or syndromes, not elsewhere classified",
    308: "Acute reaction to stress",
    309: "Adjustment reaction",
    310: "Specific nonpsychotic mental disorders due to brain damage",
    311: "Depressive disorder, not elsewhere classified",
    312: "Disturbance of conduct, not elsewhere classified",
    313: "Disturbance of emotions specific to childhood and adolescence",
    314: "Hyperkinetic syndrome of childhood",
    315: "Specific delays in development",
    316: "Psychic factors associated with diseases classified elsewhere",
    317: "Mild mental retardation",
    318: "Other specified mental retardation",
    319: "Unspecified mental retardation",
}


# Official MSK ICD-9 category names
MSK_ICD9_NAMES: dict[int, str] = {
    710: "Diffuse diseases of connective tissue",
    711: "Arthropathy associated with infections",
    712: "Crystal arthropathies",
    713: "Arthropathy associated with other disorders classified elsewhere",
    714: "Rheumatoid arthritis and other inflammatory polyarthropathies",
    715: "Osteoarthrosis and allied disorders",
    716: "Other and unspecified arthropathies",
    717: "Internal derangement of knee",
    718: "Other derangement of joint",
    719: "Other and unspecified disorders of joint",
    720: "Ankylosing spondylitis and other inflammatory spondylopathies",
    721: "Spondylosis and allied disorders",
    722: "Intervertebral disc disorders",
    723: "Other disorders of cervical region",
    724: "Other and unspecified disorders of back",
    725: "Polymyalgia rheumatica",
    726: "Peripheral enthesopathies and allied syndromes",
    727: "Other disorders of synovium, tendon, and bursa",
    728: "Disorders of muscle, ligament, and fascia",
    729: "Other disorders of soft tissues",
    730: "Osteomyelitis, periostitis, and other infections involving bone",
    731: (
        "Osteitis deformans and osteopathies associated with other "
        "disorders classified elsewhere"
    ),
    732: "Osteochondropathies",
    733: "Other disorders of bone and cartilage",
    734: "Flat foot",
    735: "Acquired deformities of toe",
    736: "Other acquired deformities of limbs",
    737: "Curvature of spine",
    738: "Other acquired deformity",
    739: "Nonallopathic lesions, not elsewhere classified",
}


# Combined category-name lookup for Mental and MSK diagnoses.
ICD9_CATEGORY_NAMES: dict[int, str] = {
    **MENTAL_ICD9_NAMES,
    **MSK_ICD9_NAMES,
}



# The official names above remain the underlying definitions.
# These labels are only for more readable figures/tables.
ICD9_DISPLAY_LABELS: dict[int, str] = {
    # Mental
    290: "Dementias",
    291: "Alcohol-induced mental disorders",
    292: "Drug-induced mental disorders",
    293: "Transient mental disorders",
    294: "Persistent mental disorders",
    295: "Schizophrenic disorders",
    296: "Mood disorders",
    297: "Delusional disorders",
    298: "Other psychoses",
    299: "Pervasive developmental disorders",
    300: "Anxiety, dissociative and somatoform disorders",
    301: "Personality disorders",
    302: "Sexual and gender identity disorders",
    303: "Alcohol dependence",
    304: "Drug dependence",
    305: "Nondependent drug abuse",
    306: "Physiological malfunction from mental factors",
    307: "Special symptoms or syndromes",
    308: "Acute stress reaction",
    309: "Adjustment reaction",
    310: "Nonpsychotic disorders due to brain damage",
    311: "Depressive disorder",
    312: "Conduct disturbance",
    313: "Childhood/adolescent emotional disturbance",
    314: "Hyperkinetic syndrome of childhood",
    315: "Developmental delays",
    316: "Psychic factors associated with other diseases",
    317: "Mild intellectual disability",
    318: "Other specified intellectual disability",
    319: "Unspecified intellectual disability",

    # MSK
    710: "Connective tissue diseases",
    711: "Infection-associated arthropathy",
    712: "Crystal arthropathies",
    713: "Secondary arthropathy",
    714: "Rheumatoid and inflammatory polyarthropathies",
    715: "Osteoarthrosis",
    716: "Other arthropathies",
    717: "Internal derangement of knee",
    718: "Other joint derangement",
    719: "Other joint disorders",
    720: "Inflammatory spondylopathies",
    721: "Spondylosis",
    722: "Intervertebral disc disorders",
    723: "Cervical disorders",
    724: "Back disorders",
    725: "Polymyalgia rheumatica",
    726: "Peripheral enthesopathies",
    727: "Synovium, tendon, and bursa disorders",
    728: "Muscle, ligament, and fascia disorders",
    729: "Other soft tissue disorders",
    730: "Bone infections",
    731: "Osteitis deformans and secondary osteopathies",
    732: "Osteochondropathies",
    733: "Other bone and cartilage disorders",
    734: "Flat foot",
    735: "Acquired toe deformities",
    736: "Other acquired limb deformities",
    737: "Spinal curvature",
    738: "Other acquired deformities",
    739: "Nonallopathic lesions",
}


# ICD-9 parsing and classification helpers
def normalize_icd9_code(code: Any) -> str | None:
    """
    Normalize an ICD-9 code for matching.

    Examples
    --------
    " 296.2 " -> "296.2"
    296.2     -> "296.2"
    None      -> None
    """
    if code is None or pd.isna(code):
        return None

    normalized = str(code).strip().upper()

    if normalized in {"", "NAN", "NONE", "<NA>"}:
        return None

    return normalized


def extract_icd9_3(code: Any) -> int | None:
    """
    Extract the leading three-digit numeric ICD-9 category.

    Examples
    --------
    "296.2" -> 296
    "724"   -> 724
    "V10"   -> None
    """
    normalized = normalize_icd9_code(code)
    if normalized is None:
        return None

    match = re.match(r"^(\d{3})", normalized)
    return int(match.group(1)) if match else None


# Backward-compatible private alias used by older pipeline code.
def _icd9_3_int(code: Any) -> int | None:
    return extract_icd9_3(code)


def classify_system_broad(icd9_code: Any) -> str | None:
    """
    Classify a numeric ICD-9 code into a broad disease system.
    """
    icd3 = extract_icd9_3(icd9_code)

    if icd3 is None:
        return None

    for system, (low, high) in ICD9_RANGES.items():
        if low <= icd3 <= high:
            return system

    return None


def classify_system(icd9_code: Any) -> str | None:
    """
    Alias for broad ICD-9 system classification.
    """
    return classify_system_broad(icd9_code)


def classify_system_from_icd9_3(icd9_code: Any) -> str | None:
    """
    Backward-compatible alias for broad ICD-9 system classification.
    """
    return classify_system_broad(icd9_code)


def classify_subgroup(icd9_code: Any) -> str | None:
    """
    Return the official Mental or MSK ICD-9 category name.

    This function retains the previous name `classify_subgroup` for
    compatibility with existing code, although the returned value is now
    an official ICD-9 diagnostic category rather than a custom subgroup.
    """
    icd3 = extract_icd9_3(icd9_code)

    if icd3 is None:
        return None

    return ICD9_CATEGORY_NAMES.get(icd3)


def classify_category(icd9_code: Any) -> str | None:
    """
    Return the official Mental or MSK ICD-9 category name.
    """
    return classify_subgroup(icd9_code)


def get_icd9_category_name(icd9_code: Any) -> str | None:
    """
    Return the official Mental or MSK ICD-9 category name.
    """
    return classify_subgroup(icd9_code)


def get_icd9_display_label(icd9_code: Any) -> str | None:
    """
    Return the shorter display label for a Mental or MSK category.
    """
    icd3 = extract_icd9_3(icd9_code)

    if icd3 is None:
        return None

    return ICD9_DISPLAY_LABELS.get(
        icd3,
        ICD9_CATEGORY_NAMES.get(icd3),
    )


def build_icd9_name_map_from_encounter(
    encounter: pd.DataFrame,
    icd9_col: str = "icd9",
    name_calc_col: str = "DiagnosisText_calc",
    name_orig_col: str = "DiagnosisText_orig",
    allow_ve: bool = True,
) -> dict[str, str]:
    """
    Build a full-code-to-name map from the encounter diagnosis table.

    The calculated diagnosis name is preferred. The original diagnosis
    name is used as a fallback when the calculated name is unavailable.

    This helper preserves full-code names from the source data. For broad
    Mental and MSK patient-profile categories, use ICD9_CATEGORY_NAMES.
    """
    if icd9_col not in encounter.columns:
        raise KeyError(
            f"Required ICD-9 column not found: {icd9_col}"
        )

    df = encounter.copy()
    df[icd9_col] = (
        df[icd9_col]
        .astype("string")
        .str.strip()
        .replace(
            {
                "nan": pd.NA,
                "NaN": pd.NA,
                "": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )

    name = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    if name_calc_col in df.columns:
        name = (
            df[name_calc_col]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "nan": pd.NA,
                    "NaN": pd.NA,
                    "": pd.NA,
                    "None": pd.NA,
                    "<NA>": pd.NA,
                }
            )
        )

    if name_orig_col in df.columns:
        original_name = (
            df[name_orig_col]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "nan": pd.NA,
                    "NaN": pd.NA,
                    "": pd.NA,
                    "None": pd.NA,
                    "<NA>": pd.NA,
                }
            )
        )
        name = name.fillna(original_name)

    df = pd.DataFrame(
        {
            icd9_col: df[icd9_col],
            "name": name,
        }
    ).dropna()

    pattern = (
        r"^(?:\d{3}(?:\.\d{1,2})?|V\d{2}(?:\.\d{1,2})?|E\d{3}(?:\.\d)?)$"
        if allow_ve
        else r"^\d{3}(?:\.\d{1,2})?$"
    )

    df = df[
        df[icd9_col].str.match(
            pattern,
            na=False,
        )
    ]

    if df.empty:
        return {}

    return (
        df.groupby(icd9_col)["name"]
        .agg(
            lambda values: values.value_counts().index[0]
        )
        .to_dict()
    )


def build_icd3_name_map_from_icd9_map(
    icd9_map: dict[str, str],
) -> dict[str, str]:
    """
    Collapse a full ICD-9 name map to leading three-digit categories.
    """
    icd3_map: dict[str, str] = {}

    for code, name in icd9_map.items():
        match = re.match(r"^(\d{3})", str(code))

        if match and name:
            icd3_map.setdefault(
                match.group(1),
                name,
            )

    return icd3_map


def format_icd9_list_with_names(
    icd9_list_str: str,
    icd9_map: dict[str, str],
    icd3_map: dict[str, str] | None = None,
    keep_unknown: bool = False,
) -> str:
    """
    Format a comma-separated ICD-9 list as 'code (diagnosis name)'.

    Full-code names are used first. A three-digit fallback map may be
    supplied when full-code names are unavailable.
    """
    if pd.isna(icd9_list_str) or str(icd9_list_str).strip() == "":
        return ""

    codes = [
        code.strip()
        for code in str(icd9_list_str).split(",")
        if code.strip()
    ]

    output: list[str] = []

    for code in codes:
        name = icd9_map.get(code)

        if name is None and icd3_map is not None:
            match = re.match(r"^(\d{3})", code)

            if match:
                name = icd3_map.get(
                    match.group(1)
                )

        if name:
            output.append(f"{code} ({name})")
        elif keep_unknown:
            output.append(f"{code} (Unknown)")

    return ", ".join(output)
