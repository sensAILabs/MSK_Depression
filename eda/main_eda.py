# eda/main_eda.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from eda.clinical_definitions import ICD9_RANGES, CONDITION_DRUG_KEYWORDS
from eda.data_processing import (
    build_patient_info,
    add_patient_icd9_lists,
    build_msk_type_table,
    build_analysis_df_msk_only,
)
from eda.duration import compute_condition_duration
from eda.plotting import *

def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():

    ROOT = Path(__file__).resolve().parents[1]
    DATA_OUT = ROOT / "data" / "output"

    FIG_DIR = Path(__file__).parent / "results" / "figures"
    TAB_DIR = Path(__file__).parent / "results" / "tables"

    patient = pd.read_csv(DATA_OUT / "C4MPatient_clean_1_31.csv", dtype=str)
    encounter = pd.read_csv(DATA_OUT / "C4MEncounterDiagnosis_clean_1_31.csv", dtype=str)
    medication = pd.read_csv(DATA_OUT / "C4MMedication_clean_1_31.csv", dtype=str)

    encounter = encounter.rename(columns={"DiagnosisCode_calc": "icd9"})

    patient_info = build_patient_info(encounter, patient)
    enc_icd9 = add_patient_icd9_lists(encounter)
    enc_icd9_msk = enc_icd9[
    enc_icd9["type list"].str.contains(r"(?:^|,)msk(?:,|$)", na=False, regex=True)
    ].copy()
    enc_icd9_msk["comorbidity_count"] = enc_icd9_msk["type list"].apply(
        lambda s: sum(t not in ("", "msk") for t in str(s).split(",")))



    msk_type_table = build_msk_type_table(enc_icd9, ICD9_RANGES)

    mental_duration = compute_condition_duration(
        encounter, medication, ICD9_RANGES,
        "mental", CONDITION_DRUG_KEYWORDS["mental"]
    )

    msk_duration = compute_condition_duration(
        encounter, medication, ICD9_RANGES,
        "msk", CONDITION_DRUG_KEYWORDS["msk"]
    )

    analysis_df = build_analysis_df_msk_only(
        patient_info, enc_icd9_msk, msk_type_table,
        msk_duration, mental_duration
    )

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(TAB_DIR / "analysis_df.csv", index=False)

    # save plots
    save_fig(step5_plot_msk_comorbidities(analysis_df), FIG_DIR / "comorbidities.png")
    save_fig(step5_plot_msk_multimorbidity(analysis_df), FIG_DIR / "multimorbidity.png")
    save_fig(step5_plot_msk_age_groups(analysis_df), FIG_DIR / "age.png")
    save_fig(step5_plot_msk_sex(analysis_df), FIG_DIR / "sex.png")
    save_fig(step5_plot_msk_age_sex_patient(analysis_df), FIG_DIR / "age_sex.png")
    save_fig(boxplot_msk_vs_mental_by_sex(analysis_df), FIG_DIR / "duration.png")
    save_fig(barplot_msk_type_percentage(analysis_df), FIG_DIR / "msk_type.png")
    save_fig(plot_msk_type_by_sex_percent(analysis_df), FIG_DIR / "msk_type_sex.png")

if __name__ == "__main__":
    main()
