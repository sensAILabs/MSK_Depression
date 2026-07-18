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
    # Acute respiratory infection
    460: {"system": "respiratory", "subgroup": "acute_upper_respiratory_infection", "course": "acute"},
    461: {"system": "respiratory", "subgroup": "acute_upper_respiratory_infection", "course": "acute"},
    462: {"system": "respiratory", "subgroup": "acute_upper_respiratory_infection", "course": "acute"},
    463: {"system": "respiratory", "subgroup": "acute_upper_respiratory_infection", "course": "acute"},
    464: {"system": "respiratory", "subgroup": "acute_upper_respiratory_infection", "course": "acute"},
    465: {"system": "respiratory", "subgroup": "acute_upper_respiratory_infection", "course": "acute"},
    466: {"system": "respiratory", "subgroup": "acute_lower_respiratory_infection", "course": "acute"},
    # Upper airway
    470: {"system": "respiratory", "subgroup": "upper_airway_structural_disorder", "course": "symptom"},
    471: {"system": "respiratory", "subgroup": "upper_airway_chronic_inflammation", "course": "chronic"},
    472: {"system": "respiratory", "subgroup": "upper_airway_chronic_inflammation", "course": "chronic"},
    473: {"system": "respiratory", "subgroup": "upper_airway_chronic_inflammation", "course": "chronic"},
    474: {"system": "respiratory", "subgroup": "upper_airway_hypertrophy", "course": "chronic"},
    475: {"system": "respiratory", "subgroup": "upper_airway_abscess", "course": "acute"},
    476: {"system": "respiratory", "subgroup": "upper_airway_chronic_inflammation", "course": "chronic"},
    477: {"system": "respiratory", "subgroup": "allergic_rhinitis", "course": "chronic"},
    478: {"system": "respiratory", "subgroup": "other_upper_airway_disorder", "course": "chronic"},
    # Pneumonia & influenza
    480: {"system": "respiratory", "subgroup": "pneumonia_infectious", "course": "acute"},
    481: {"system": "respiratory", "subgroup": "pneumonia_infectious", "course": "acute"},
    482: {"system": "respiratory", "subgroup": "pneumonia_infectious", "course": "acute"},
    483: {"system": "respiratory", "subgroup": "pneumonia_infectious", "course": "acute"},
    484: {"system": "respiratory", "subgroup": "pneumonia_infectious", "course": "acute"},
    485: {"system": "respiratory", "subgroup": "pneumonia_infectious", "course": "acute"},
    486: {"system": "respiratory", "subgroup": "pneumonia_infectious", "course": "acute"},
    487: {"system": "respiratory", "subgroup": "influenza", "course": "acute"},
    488: {"system": "respiratory", "subgroup": "influenza", "course": "acute"},

    490: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "symptom"},
    491: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # chronic bronchitis
    492: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # emphysema
    493: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # asthma
    494: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # bronchiectasis
    495: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # hypersensitivity pneumonitis
    496: {"system": "respiratory", "subgroup": "copd_spectrum", "course": "chronic"},  # COPD
    # Occupational
    500: {"system": "respiratory", "subgroup": "pneumoconiosis", "course": "chronic"},
    501: {"system": "respiratory", "subgroup": "pneumoconiosis", "course": "chronic"},
    502: {"system": "respiratory", "subgroup": "pneumoconiosis", "course": "chronic"},
    503: {"system": "respiratory", "subgroup": "pneumoconiosis", "course": "chronic"},
    504: {"system": "respiratory", "subgroup": "pneumoconiosis", "course": "chronic"},
    505: {"system": "respiratory", "subgroup": "pneumoconiosis_unspecified", "course": "chronic"},

    506: {"system": "respiratory", "subgroup": "chemical_lung_injury", "course": "acute"},
    507: {"system": "respiratory", "subgroup": "chemical_lung_injury", "course": "acute"},
    508: {"system": "respiratory", "subgroup": "chemical_lung_injury", "course": "symptom"}, # Respiratory conditions due to other and unspecified external agents) includes both acute and chronic conditions, depending on the sub-code used.

    # Pleura & others
    510: {"system": "respiratory", "subgroup": "pleural_infection", "course": "acute"},
    511: {"system": "respiratory", "subgroup": "pleural_effusion", "course": "acute"},
    512: {"system": "respiratory", "subgroup": "pneumothorax", "course": "acute"},
    513: {"system": "respiratory", "subgroup": "lung_abscess", "course": "acute"},

    514: {"system": "respiratory", "subgroup": "pulmonary_congestion", "course": "chronic"},
    515: {"system": "respiratory", "subgroup": "interstitial_lung_disease", "course": "chronic"},
    516: {"system": "respiratory", "subgroup": "interstitial_lung_disease", "course": "chronic"},
    517: {"system": "respiratory", "subgroup": "lung_involvement_systemic_disease", "course": "symptom"},
    518: {"system": "respiratory", "subgroup": "respiratory_failure", "course": "symptom"},
    519: {"system": "respiratory", "subgroup": "other_respiratory_disorder", "course": "symptom"},

    # ---------------- Mental ----------------
    290: {"system": "mental", "subgroup": "chronic_cognitive_disorder", "course": "chronic"},
    291: {"system": "mental", "subgroup": "acute_organic_mental_disorder", "course": "acute"},
    292: {"system": "mental", "subgroup": "acute_organic_mental_disorder", "course": "acute"},
    293: {"system": "mental", "subgroup": "acute_organic_mental_disorder", "course": "acute"},
    294: {"system": "mental", "subgroup": "chronic_cognitive_disorder", "course": "chronic"},
   # Psychotic & Mood 
    295: {"system": "mental", "subgroup": "schizophrenic_disorder", "course": "chronic"},# Schizophrenic disorders, a category that includes both acute and chronic conditions, along with subchronic, exacerbated, and remission states
    296: {"system": "mental", "subgroup": "mood_disorder", "course": "chronic"}, #such as bipolar disorder or major depression
    297: {"system": "mental", "subgroup": "delusional_disorder", "course": "chronic"},
    298: {"system": "mental", "subgroup": "acute_psychotic_disorder", "course": "acute"},
    299: {"system": "mental", "subgroup": "neurodevelopmental_disorder", "course": "chronic"},
    # Neurotic / personality
    300: {"system": "mental", "subgroup": "anxiety_neurotic_disorder", "course": "chronic"},
    301: {"system": "mental", "subgroup": "personality_disorder", "course": "chronic"},
    302: {"system": "mental", "subgroup": "sexual_gender_identity_disorder", "course": "chronic"},
    # Substance
    303: {"system": "mental", "subgroup": "substance_dependence", "course": "chronic"},
    304: {"system": "mental", "subgroup": "substance_dependence", "course": "chronic"},
    305: {"system": "mental", "subgroup": "substance_abuse_nondependent", "course": "chronic"},
    # Somatic / stress
    306: {"system": "mental", "subgroup": "psychophysiologic_disorder", "course": "chronic"},
    307: {"system": "mental", "subgroup": "special_symptom_syndrome", "course": "symptom"},
    308: {"system": "mental", "subgroup": "acute_stress_reaction", "course": "acute"},
    309: {"system": "mental", "subgroup": "adjustment_disorder", "course": "acute"},
    # Brain / depression 
    310: {"system": "mental", "subgroup": "brain_related_disorder", "course": "chronic"},
    311: {"system": "mental", "subgroup": "depressive_disorder", "course": "chronic"},
    # Childhood 
    312: {"system": "mental", "subgroup": "conduct_disorder", "course": "chronic"},# Disturbance of conduct, not elsewhere classified.
    313: {"system": "mental", "subgroup": "child_emotional_disorder", "course": "chronic"},
    314: {"system": "mental", "subgroup": "adhd_hyperkinetic", "course": "chronic"},
    315: {"system": "mental", "subgroup": "developmental_delay", "course": "chronic"},
    316: {"system": "mental", "subgroup": "psychological_factor_physical_condition", "course": "chronic"},
    # Intellectual disability 
    317: {"system": "mental", "subgroup": "intellectual_disability", "course": "chronic"},
    318: {"system": "mental", "subgroup": "intellectual_disability", "course": "chronic"},
    319: {"system": "mental", "subgroup": "intellectual_disability", "course": "chronic"},

    # ---------------- MSK ----------------
    #Arthritis 
    710: {"system": "msk", "subgroup": "connective_tissue_disease", "course": "chronic"},
    711: {"system": "msk", "subgroup": "infectious_arthritis", "course": "symptom"},# Arthropathy associated with infections, depending on the specific subcode used
    712: {"system": "msk", "subgroup": "crystal_inflammatory_arthritis", "course": "chronic"},
    713: {"system": "msk", "subgroup": "secondary_inflammatory_arthritis", "course": "chronic"},
    714: {"system": "msk", "subgroup": "rheumatoid_arthritis", "course": "chronic"},
    715: {"system": "msk", "subgroup": "osteoarthritis", "course": "chronic"},
    716: {"system": "msk", "subgroup": "other_arthropathy", "course": "chronic"},# Other and unspecified arthropathies
    # Joint disorders
    717: {"system": "msk", "subgroup": "internal_joint_derangement", "course": "chronic"},
    718: {"system": "msk", "subgroup": "joint_instability", "course": "chronic"},
    719: {"system": "msk", "subgroup": "joint_pain_symptom", "course": "symptom"},# Other and unspecified disorders of joint, general category for joint issues
    # Spine 
    720: {"system": "msk", "subgroup": "inflammatory_spine_disorder", "course": "chronic"},
    721: {"system": "msk", "subgroup": "spondylosis_degeneration", "course": "chronic"},
    722: {"system": "msk", "subgroup": "intervertebral_disc_disorder", "course": "chronic"},# Intervertebral disc disorders
    723: {"system": "msk", "subgroup": "neck_disorder", "course": "symptom"}, #Other disorders of cervical region
    724: {"system": "msk", "subgroup": "back_pain_symptom", "course": "symptom"},
    739: {"system": "msk", "subgroup": "somatic_spine_dysfunction", "course": "symptom"},# a functional condition that can be acute or chronic
    # Soft tissue 
    725: {"system": "msk", "subgroup": "polymyalgia", "course": "chronic"},
    726: {"system": "msk", "subgroup": "bursitis_enthesopathy", "course": "symptom"},
    727: {"system": "msk", "subgroup": "tendon_synovium_disorder", "course": "symptom"},# Other disorders of synovium, tendon, and bursa, is a general category that includes both acute and chronic conditions
    728: {"system": "msk", "subgroup": "muscle_ligament_disorder", "course": "symptom"},# Disorders of muscle, ligament, and fascia, including both conditions
    729: {"system": "msk", "subgroup": "other_soft_tissue_disorder", "course": "symptom"},
    # Bone 
    730: {"system": "msk", "subgroup": "bone_infection", "course": "chronic"},# Osteomyelitis, periostitis, and other infections involving bon, including both conditions
    731: {"system": "msk", "subgroup": "bone_remodeling_disorder", "course": "chronic"},
    732: {"system": "msk", "subgroup": "osteochondropathy", "course": "chronic"},
    733: {"system": "msk", "subgroup": "osteoporosis", "course": "chronic"},
    # Deformity 
    734: {"system": "msk", "subgroup": "foot_deformity", "course": "chronic"},
    735: {"system": "msk", "subgroup": "toe_deformity", "course": "chronic"},
    736: {"system": "msk", "subgroup": "other_limb_deformity", "course": "chronic"},
    737: {"system": "msk", "subgroup": "spinal_deformity", "course": "chronic"},
    738: {"system": "msk", "subgroup": "other_acquired_deformity", "course": "chronic"},
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
    # ---------------- Respiratory ----------------
    "acute_upper_respiratory_infection": "Acute upper respiratory infection",
    "acute_lower_respiratory_infection": "Acute lower respiratory infection",
    "upper_airway_structural_disorder": "Upper airway structural disorder",
    "upper_airway_chronic_inflammation": "Upper airway chronic inflammation",
    "upper_airway_hypertrophy": "Upper airway hypertrophy",
    "upper_airway_abscess": "Upper airway abscess",
    "allergic_rhinitis": "Allergic rhinitis",
    "other_upper_airway_disorder": "Other upper airway disorder",
    "pneumonia_infectious": "Infectious pneumonia",
    "influenza": "Influenza",
    "copd_spectrum": "COPD spectrum",
    "pneumoconiosis": "Pneumoconiosis",
    "pneumoconiosis_unspecified": "Pneumoconiosis (unspecified)",
    "chemical_lung_injury": "Chemical / external-agent lung injury",
    "pleural_infection": "Pleural infection",
    "pleural_effusion": "Pleural effusion",
    "pneumothorax": "Pneumothorax",
    "lung_abscess": "Lung abscess",
    "pulmonary_congestion": "Pulmonary congestion",
    "interstitial_lung_disease": "Interstitial lung disease",
    "lung_involvement_systemic_disease": "Lung involvement in systemic disease",
    "respiratory_failure": "Respiratory failure",
    "other_respiratory_disorder": "Other respiratory disorder",

    # ---------------- Mental ----------------
    "chronic_cognitive_disorder": "Chronic cognitive disorder",
    "acute_organic_mental_disorder": "Acute organic mental disorder",
    "schizophrenic_disorder": "Schizophrenic disorder",
    "mood_disorder": "Mood disorder",
    "delusional_disorder": "Delusional disorder",
    "acute_psychotic_disorder": "Acute psychotic disorder",
    "neurodevelopmental_disorder": "Neurodevelopmental disorder",
    "anxiety_neurotic_disorder": "Anxiety / neurotic disorder",
    "personality_disorder": "Personality disorder",
    "sexual_gender_identity_disorder": "Sexual / gender identity disorder",
    "substance_dependence": "Substance dependence",
    "substance_abuse_nondependent": "Nondependent substance abuse",
    "psychophysiologic_disorder": "Psychophysiologic disorder",
    "special_symptom_syndrome": "Special symptom syndrome",
    "acute_stress_reaction": "Acute stress reaction",
    "adjustment_disorder": "Adjustment disorder",
    "brain_related_disorder": "Brain-related disorder",
    "depressive_disorder": "Depressive disorder",
    "conduct_disorder": "Conduct disorder",
    "child_emotional_disorder": "Child emotional disorder",
    "adhd_hyperkinetic": "ADHD / hyperkinetic disorder",
    "developmental_delay": "Developmental delay",
    "psychological_factor_physical_condition": "Psychological factor affecting physical condition",
    "intellectual_disability": "Intellectual disability",

    # ---------------- MSK ----------------
    "connective_tissue_disease": "Connective tissue disease",
    "infectious_arthritis": "Infectious arthritis",
    "crystal_inflammatory_arthritis": "Crystal inflammatory arthritis",
    "secondary_inflammatory_arthritis": "Secondary inflammatory arthritis",
    "rheumatoid_arthritis": "Rheumatoid arthritis",
    "osteoarthritis": "Osteoarthritis",
    "other_arthropathy": "Other arthropathy",
    "internal_joint_derangement": "Internal joint derangement",
    "joint_instability": "Joint instability",
    "joint_pain_symptom": "Joint pain / joint symptom",
    "inflammatory_spine_disorder": "Inflammatory spine disorder",
    "spondylosis_degeneration": "Spondylosis / spinal degeneration",
    "intervertebral_disc_disorder": "Intervertebral disc disorder",
    "neck_disorder": "Neck disorder",
    "back_pain_symptom": "Back pain / back symptom",
    "somatic_spine_dysfunction": "Somatic spine dysfunction",
    "polymyalgia": "Polymyalgia",
    "bursitis_enthesopathy": "Bursitis / enthesopathy",
    "tendon_synovium_disorder": "Tendon / synovium disorder",
    "muscle_ligament_disorder": "Muscle / ligament disorder",
    "other_soft_tissue_disorder": "Other soft tissue disorder",
    "bone_infection": "Bone infection",
    "bone_remodeling_disorder": "Bone remodeling disorder",
    "osteochondropathy": "Osteochondropathy",
    "osteoporosis": "Osteoporosis",
    "foot_deformity": "Foot deformity",
    "toe_deformity": "Toe deformity",
    "other_limb_deformity": "Other limb deformity",
    "spinal_deformity": "Spinal deformity",
    "other_acquired_deformity": "Other acquired deformity",
}



DRUG_CLASS_KEYWORDS = {
    "antidepressant": [
        "IMIPRAMINE", "DESIPRAMINE", "AMITRIPTYLINE", "NORTRIPTYLINE",
        "CLOMIPRAMINE", "TRAZODONE", "NEFAZODONE",
        "FLUOXETINE", "BUPROPION", "SERTRALINE", "PAROXETINE",
        "VENLAFAXINE", "DESVENLAFAXINE", "FLUVOXAMINE",
        "MIRTAZAPINE", "CITALOPRAM", "ESCITALOPRAM",
        "DULOXETINE", "VILAZODONE", "VORTIOXETINE",
        "LEVOMILNACIPRAN"
    ],

    "antipsychotic": [
        "CHLORPROMAZINE", "CLOZAPINE", "QUETIAPINE",
        "PERPHENAZINE", "LOXAPINE", "FLUPHENAZINE",
        "HALOPERIDOL", "PIMOZIDE", "RISPERIDONE",
        "PALIPERIDONE", "OLANZAPINE", "ZIPRASIDONE",
        "ILOPERIDONE", "ASENAPINE", "LURASIDONE",
        "ARIPIPRAZOLE", "BREXPIPRAZOLE", "CARIPRAZINE"
    ],

    "anti_anxiety": [
        # benzodiazepines
        "DIAZEPAM", "CHLORDIAZEPOXIDE", "CLONAZEPAM",
        "LORAZEPAM", "ALPRAZOLAM",

        # other anti-anxiety agents
        "BUSPIRONE", "GABAPENTIN", "HYDROXYZINE",
        "PROPRANOLOL", "ATENOLOL", "GUANFACINE",
        "CLONIDINE", "PREGABALIN", "PRAZOSIN",

        "TEMAZEPAM", "TRIAZOLAM"
    ],

    "hypnotic": [
        #no benzo
        "ZOLPIDEM", "ZALEPLON", "ESZOPICLONE",
        "RAMELTEON", "DIPHENHYDRAMINE",
        "DOXEPIN", "SUVOREXANT"
    ],

    "nsaid": [
        "IBUPROFEN", "NAPROXEN", "DICLOFENAC",
        "INDOMETHACIN", "KETOROLAC",
        "PIROXICAM", "MELOXICAM",
        "CELECOXIB",       
        "ASPIRIN"           
    ],

    "acetaminophen": [
        "ACETAMINOPHEN", "PARACETAMOL","APAP"
    ],

    "opioid": [
        "MORPHINE", "CODEINE", "TRAMADOL",
        "OXYCODONE", "HYDROMORPHONE",
        "FENTANYL", "METHADONE",
        "HYDROCODONE", "BUPRENORPHINE",
        "TAPENTADOL"
    ],

    "muscle_relaxant": [
        "CYCLOBENZAPRINE", "METHOCARBAMOL",
        "CARISOPRODOL", "TIZANIDINE",
        "METAXALONE", "BACLOFEN",
        "ORPHENADRINE"
    ],
}


MENTAL_CLASSES = {"antidepressant", "antipsychotic", "anti_anxiety", "hypnotic"}
PAIN_CLASSES = {"nsaid", "acetaminophen", "muscle_relaxant", "opioid"}




# Helper functions

def _icd9_3_int(code) -> int | None:
    try:
        return int(str(code).strip().split(".")[0][:3])
    except Exception:
        return None


def classify_icd9(icd9_code: str) -> dict:
    icd9_3 = _icd9_3_int(icd9_code)
    if icd9_3 is None:
        return {"system": None, "subgroup": None, "course": "symptom"}
    return ICD9_INFO.get(icd9_3, {"system": None, "subgroup": None, "course": "symptom"})


def classify_system(icd9_code: str) -> str | None:
    return classify_icd9(icd9_code)["system"]


def classify_system_from_icd9_3(icd9_code: str) -> str | None:
    return classify_system(icd9_code)


def classify_subgroup(icd9_code: str) -> str | None:
    return classify_icd9(icd9_code)["subgroup"]


def classify_acute_chronic(icd9_code: str) -> str:
    return classify_icd9(icd9_code)["course"]



def icd9_to_num(code: str):
    try:
        return float(str(code).strip().split(".")[0])
    except:
        return None


def classify_system_broad(icd9_code: str) -> str | None:
    x = icd9_to_num(icd9_code)
    if x is None:
        return None

    for system, (lo, hi) in ICD9_RANGES.items():
        if lo <= x <= hi:
            return system
    return None