"""
Clinical Definitions Module

This module defines:
- ICD-9 system classification ranges
- Medication keyword lists used for condition duration extension

These definitions are used by the EDA pipeline and should not contain
any execution logic.
"""

# ---------------------------------------------------------------------
# ICD-9 Chronic Disease System Classification
# ---------------------------------------------------------------------

ICD9_RANGES = {
    "cancer": (140, 239.9),
    "endocrine": (240, 279.9),
    "mental": (290, 319),
    "circulatory": (390, 459.9),
    "respiratory": (460, 519.9),
    "genitourinary": (580, 629.9),
    "msk": (710, 739.9),
}


# ---------------------------------------------------------------------
# Medication Keywords for Duration Extension
# ---------------------------------------------------------------------

CONDITION_DRUG_KEYWORDS = {

    "msk": [
        # NSAIDs
        "IBUPROFEN",
        "NAPROXEN",
        "DICLOFENAC",
        "CELECOXIB",

        # Analgesics
        "ACETAMINOPHEN",
        "PARACETAMOL",

        # Muscle Relaxants
        "CYCLOBENZAPRINE",
        "METHOCARBAMOL",
        "CARISOPRODOL",
        "TIZANIDINE",
        "METAXALONE",
    ],

    "mental": [
        # SSRIs
        "SERTRALINE",
        "FLUOXETINE",
        "CITALOPRAM",
        "ESCITALOPRAM",
        "PAROXETINE",

        # SNRIs
        "VENLAFAXINE",
        "DULOXETINE",

        # Others
        "BUPROPION",

        # Antipsychotics
        "QUETIAPINE",
        "OLANZAPINE",
        "RISPERIDONE",

        # Benzodiazepines
        "LORAZEPAM",
        "CLONAZEPAM",
        "DIAZEPAM",
    ],
}
