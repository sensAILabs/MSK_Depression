"""
Clinical Definitions Module

This module defines:
- ICD-9 system classification ranges
- ICD-9 acute/chronic classification rules for respiratory and mental conditions
- ICD-9 subgroup classification rules for respiratory and mental conditions
- Medication keyword lists used for condition duration extension
- Helper functions for ICD-9 based classification

These definitions are used by the EDA and resp/mental analysis pipelines.
The module should contain definitions and lightweight helper functions only.
"""

# ICD-9 broad system classification

ICD9_RANGES = {
    "cancer": (140, 239.9),
    "endocrine": (240, 279.9),
    "mental": (290, 319.9),
    "cardiovascular": (390, 459.9),
    "respiratory": (460, 519.9),
    "genitourinary": (580, 629.9),
    "msk": (710, 739.9),
}

SYS_ABBR = {
    "cardiovascular": "C",
    "endocrine": "E",
    "genitourinary": "G",
    "mental": "M",
    "respiratory": "R",
    "cancer": "CA",
    "msk": "MSK",
}



# Medication keywords for duration extension

CONDITION_DRUG_KEYWORDS = {
    "msk": [
        "IBUPROFEN",
        "NAPROXEN",
        "DICLOFENAC",
        "CELECOXIB",
        "ACETAMINOPHEN",
        "PARACETAMOL",
        "CYCLOBENZAPRINE",
        "METHOCARBAMOL",
        "CARISOPRODOL",
        "TIZANIDINE",
        "METAXALONE",
    ],
    "mental": [
        "SERTRALINE",
        "FLUOXETINE",
        "CITALOPRAM",
        "ESCITALOPRAM",
        "PAROXETINE",
        "VENLAFAXINE",
        "DULOXETINE",
        "BUPROPION",
        "QUETIAPINE",
        "OLANZAPINE",
        "RISPERIDONE",
        "LORAZEPAM",
        "CLONAZEPAM",
        "DIAZEPAM",
    ],
}


# Unified ICD-9 classification table
# Each 3-digit ICD-9 appears only once
ICD9_INFO = {
    # ---------------- Respiratory ----------------
    460: {"system": "respiratory", "subgroup": "acute_respiratory_infection", "course": "acute"},
    461: {"system": "respiratory", "subgroup": "acute_respiratory_infection", "course": "acute"},
    462: {"system": "respiratory", "subgroup": "acute_respiratory_infection", "course": "acute"},
    463: {"system": "respiratory", "subgroup": "acute_respiratory_infection", "course": "acute"},
    464: {"system": "respiratory", "subgroup": "acute_respiratory_infection", "course": "acute"},
    465: {"system": "respiratory", "subgroup": "acute_respiratory_infection", "course": "acute"},
    466: {"system": "respiratory", "subgroup": "acute_respiratory_infection", "course": "acute"},

    470: {"system": "respiratory", "subgroup": "upper_airway_disorder", "course": "chronic"},
    471: {"system": "respiratory", "subgroup": "upper_airway_disorder", "course": "chronic"},
    472: {"system": "respiratory", "subgroup": "upper_airway_disorder", "course": "chronic"},
    473: {"system": "respiratory", "subgroup": "upper_airway_disorder", "course": "chronic"},
    474: {"system": "respiratory", "subgroup": "upper_airway_disorder", "course": "chronic"},
    475: {"system": "respiratory", "subgroup": "upper_airway_disorder", "course": "acute"},
    476: {"system": "respiratory", "subgroup": "upper_airway_disorder", "course": "chronic"},
    477: {"system": "respiratory", "subgroup": "upper_airway_disorder", "course": "chronic"},
    478: {"system": "respiratory", "subgroup": "upper_airway_disorder", "course": "chronic"},

    480: {"system": "respiratory", "subgroup": "pneumonia", "course": "acute"},
    481: {"system": "respiratory", "subgroup": "pneumonia", "course": "acute"},
    482: {"system": "respiratory", "subgroup": "pneumonia", "course": "acute"},
    483: {"system": "respiratory", "subgroup": "pneumonia", "course": "acute"},
    484: {"system": "respiratory", "subgroup": "pneumonia", "course": "acute"},
    485: {"system": "respiratory", "subgroup": "pneumonia", "course": "acute"},
    486: {"system": "respiratory", "subgroup": "pneumonia", "course": "acute"},

    487: {"system": "respiratory", "subgroup": "influenza", "course": "acute"},
    488: {"system": "respiratory", "subgroup": "influenza", "course": "acute"},

    490: {"system": "respiratory", "subgroup": "bronchitis_unspecified", "course": "unknown"},

    491: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # chronic bronchitis
    492: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # emphysema
    493: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # asthma
    494: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # bronchiectasis
    495: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # hypersensitivity pneumonitis
    496: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # COPD

    500: {"system": "respiratory", "subgroup": "occupational_lung_disease", "course": "chronic"},
    501: {"system": "respiratory", "subgroup": "occupational_lung_disease", "course": "chronic"},
    502: {"system": "respiratory", "subgroup": "occupational_lung_disease", "course": "chronic"},
    503: {"system": "respiratory", "subgroup": "occupational_lung_disease", "course": "chronic"},
    504: {"system": "respiratory", "subgroup": "occupational_lung_disease", "course": "chronic"},
    505: {"system": "respiratory", "subgroup": "occupational_lung_disease", "course": "chronic"},

    506: {"system": "respiratory", "subgroup": "chemical_lung_injury", "course": "acute"},
    507: {"system": "respiratory", "subgroup": "chemical_lung_injury", "course": "acute"},
    508: {"system": "respiratory", "subgroup": "chemical_lung_injury", "course": "acute"},

    510: {"system": "respiratory", "subgroup": "pleural_disease", "course": "acute"},
    511: {"system": "respiratory", "subgroup": "pleural_disease", "course": "acute"},
    512: {"system": "respiratory", "subgroup": "pleural_disease", "course": "acute"},
    513: {"system": "respiratory", "subgroup": "pleural_disease", "course": "acute"},

    514: {"system": "respiratory", "subgroup": "interstitial_lung_disease", "course": "chronic"},
    515: {"system": "respiratory", "subgroup": "interstitial_lung_disease", "course": "chronic"},
    516: {"system": "respiratory", "subgroup": "interstitial_lung_disease", "course": "chronic"},
    517: {"system": "respiratory", "subgroup": "interstitial_lung_disease", "course": "chronic"},

    518: {"system": "respiratory", "subgroup": "respiratory_failure", "course": "acute"},
    519: {"system": "respiratory", "subgroup": "respiratory_other", "course": "unknown"},

    # ---------------- Mental ----------------
    290: {"system": "mental", "subgroup": "cognitive_disorder", "course": "acute"},
    291: {"system": "mental", "subgroup": "cognitive_disorder", "course": "acute"},
    292: {"system": "mental", "subgroup": "cognitive_disorder", "course": "acute"},
    293: {"system": "mental", "subgroup": "cognitive_disorder", "course": "acute"},
    294: {"system": "mental", "subgroup": "cognitive_disorder", "course": "acute"},

    295: {"system": "mental", "subgroup": "psychotic_disorder", "course": "chronic"},
    296: {"system": "mental", "subgroup": "depression", "course": "chronic"},
    297: {"system": "mental", "subgroup": "psychotic_disorder", "course": "chronic"},
    298: {"system": "mental", "subgroup": "psychotic_disorder", "course": "chronic"},

    299: {"system": "mental", "subgroup": "developmental_behavioral", "course": "chronic"},

    300: {"system": "mental", "subgroup": "anxiety", "course": "chronic"},
    301: {"system": "mental", "subgroup": "personality_disorder", "course": "chronic"},
    302: {"system": "mental", "subgroup": "sexual_disorder", "course": "unknown"},

    303: {"system": "mental", "subgroup": "substance_use", "course": "chronic"},
    304: {"system": "mental", "subgroup": "substance_use", "course": "chronic"},
    305: {"system": "mental", "subgroup": "substance_use", "course": "chronic"},

    306: {"system": "mental", "subgroup": "stress_related", "course": "chronic"},
    307: {"system": "mental", "subgroup": "stress_related", "course": "chronic"},
    308: {"system": "mental", "subgroup": "stress_related", "course": "acute"},
    309: {"system": "mental", "subgroup": "stress_related", "course": "acute"},

    310: {"system": "mental", "subgroup": "brain_related_disorder", "course": "chronic"},
    311: {"system": "mental", "subgroup": "depression", "course": "chronic"},

    312: {"system": "mental", "subgroup": "developmental_behavioral", "course": "unknown"},
    313: {"system": "mental", "subgroup": "developmental_behavioral", "course": "unknown"},
    314: {"system": "mental", "subgroup": "developmental_behavioral", "course": "unknown"},
    315: {"system": "mental", "subgroup": "developmental_behavioral", "course": "unknown"},
    316: {"system": "mental", "subgroup": "developmental_behavioral", "course": "unknown"},

    317: {"system": "mental", "subgroup": "intellectual_disability", "course": "chronic"},
    318: {"system": "mental", "subgroup": "intellectual_disability", "course": "chronic"},
    319: {"system": "mental", "subgroup": "intellectual_disability", "course": "chronic"},

     # ---------------- MSK ----------------
    710: {"system": "msk", "subgroup": "connective_tissue_disease", "course": "chronic"},

    711: {"system": "msk", "subgroup": "infectious_arthritis", "course": "acute"},

    712: {"system": "msk", "subgroup": "inflammatory_arthritis", "course": "chronic"},
    713: {"system": "msk", "subgroup": "inflammatory_arthritis", "course": "chronic"},

    714: {"system": "msk", "subgroup": "rheumatoid_arthritis", "course": "chronic"},

    715: {"system": "msk", "subgroup": "osteoarthritis", "course": "chronic"},

    716: {"system": "msk", "subgroup": "other_arthropathy", "course": "unknown"},

    717: {"system": "msk", "subgroup": "joint_derangement", "course": "chronic"},
    718: {"system": "msk", "subgroup": "joint_derangement", "course": "unknown"},

    719: {"system": "msk", "subgroup": "joint_pain", "course": "unknown"},

    # spine
    720: {"system": "msk", "subgroup": "inflammatory_spine_disorder", "course": "chronic"},

    721: {"system": "msk", "subgroup": "spondylosis", "course": "chronic"},

    722: {"system": "msk", "subgroup": "disc_disorder", "course": "chronic"},

    723: {"system": "msk", "subgroup": "neck_disorder", "course": "chronic"},

    724: {"system": "msk", "subgroup": "back_disorder", "course": "chronic"},

    739: {"system": "msk", "subgroup": "somatic_spine_dysfunction", "course": "unknown"},

    # soft tissue
    725: {"system": "msk", "subgroup": "polymyalgia", "course": "chronic"},

    726: {"system": "msk", "subgroup": "bursitis_enthesopathy", "course": "chronic"},

    727: {"system": "msk", "subgroup": "tendon_synovium_disorder", "course": "chronic"},

    728: {"system": "msk", "subgroup": "muscle_ligament_disorder", "course": "chronic"},

    729: {"system": "msk", "subgroup": "other_soft_tissue_disorder", "course": "unknown"},

    # bone
    730: {"system": "msk", "subgroup": "bone_infection", "course": "acute"},

    731: {"system": "msk", "subgroup": "bone_disorder", "course": "chronic"},
    732: {"system": "msk", "subgroup": "bone_disorder", "course": "chronic"},

    733: {"system": "msk", "subgroup": "osteoporosis", "course": "chronic"},

    # deformity
    734: {"system": "msk", "subgroup": "deformity", "course": "chronic"},
    735: {"system": "msk", "subgroup": "deformity", "course": "chronic"},
    736: {"system": "msk", "subgroup": "deformity", "course": "chronic"},
    737: {"system": "msk", "subgroup": "deformity", "course": "chronic"},
    738: {"system": "msk", "subgroup": "deformity", "course": "unknown"},
}


def _build_subgroup_dict(system_name: str) -> dict[str, set[int]]:
    out = {}
    for icd3, info in ICD9_INFO.items():
        if info.get("system") == system_name and info.get("subgroup") is not None:
            out.setdefault(info["subgroup"], set()).add(icd3)
    return out


RESP_SUBGROUPS = _build_subgroup_dict("respiratory")
MENTAL_SUBGROUPS = _build_subgroup_dict("mental")
MSK_SUBGROUPS = _build_subgroup_dict("msk")

# Display labels for subgroup figures/tables
SUBGROUP_LABELS = {
    # respiratory
    "acute_respiratory_infection": "Acute respiratory infection",
    "upper_airway_disorder": "Upper airway disorder",
    "pneumonia": "Pneumonia",
    "influenza": "Influenza",
    "bronchitis_unspecified": "Bronchitis (unspecified)",
    "copd_spectrum": "COPD spectrum",
    "occupational_lung_disease": "Occupational lung disease",
    "chemical_lung_injury": "Chemical / aspiration lung injury",
    "pleural_disease": "Pleural disease",
    "interstitial_lung_disease": "Interstitial lung disease",
    "respiratory_failure": "Respiratory failure",
    "respiratory_other": "Other respiratory disease",

    # mental
    "cognitive_disorder": "Cognitive / organic disorder",
    "psychotic_disorder": "Psychotic disorder",
    "depression": "Depression",
    "anxiety": "Anxiety",
    "personality_disorder": "Personality disorder",
    "sexual_disorder": "Sexual disorder",
    "substance_use": "Substance use",
    "stress_related": "Stress-related disorder",
    "brain_related_disorder": "Brain-related disorder",
    "developmental_behavioral": "Developmental / behavioral disorder",
    "intellectual_disability": "Intellectual disability",

    # msk
    "connective_tissue_disease": "710 connective tissue disease",
    "infectious_arthritis": "711 infectious arthritis",
    "inflammatory_arthritis": "712/713 inflammatory arthritis",
    "rheumatoid_arthritis": "714 rheumatoid arthritis",
    "osteoarthritis": "715 osteoarthritis",
    "other_arthropathy": "716 other arthropathy",

    "joint_derangement": "717/718 joint derangement",
    "joint_pain": "719 joint pain",

    "inflammatory_spine_disorder": "720 inflammatory spine disorder",
    "spondylosis": "721 spondylosis",
    "disc_disorder": "722 disc disorder",
    "neck_disorder": "723 neck disorder",
    "back_disorder": "724 back disorder",
    "somatic_spine_dysfunction": "739 somatic spine dysfunction",

    "polymyalgia": "725 polymyalgia",
    "bursitis_enthesopathy": "726 bursitis / enthesopathy",
    "tendon_synovium_disorder": "727 tendon / synovium disorder",
    "muscle_ligament_disorder": "728 muscle / ligament disorder",
    "other_soft_tissue_disorder": "729 other soft tissue disorder",

    "bone_infection": "730 bone infection",
    "bone_disorder": "731/732 bone disorder",
    "osteoporosis": "733 osteoporosis",

    "deformity": "734/735/736/737/738 deformity",   
}


# Helper functions

def _icd9_3_int(code) -> int | None:
    try:
        return int(str(code).strip().split(".")[0][:3])
    except Exception:
        return None


def classify_icd9(icd9_code: str) -> dict:
    icd9_3 = _icd9_3_int(icd9_code)
    if icd9_3 is None:
        return {"system": None, "subgroup": None, "course": "unknown"}
    return ICD9_INFO.get(icd9_3, {"system": None, "subgroup": None, "course": "unknown"})


def classify_system(icd9_code: str) -> str | None:
    return classify_icd9(icd9_code)["system"]


def classify_system_from_icd9_3(icd9_code: str) -> str | None:
    return classify_system(icd9_code)


def classify_subgroup(icd9_code: str) -> str | None:
    return classify_icd9(icd9_code)["subgroup"]


def classify_acute_chronic(icd9_code: str) -> str:
    return classify_icd9(icd9_code)["course"]