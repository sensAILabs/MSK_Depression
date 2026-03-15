from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from eda.clinical_definitions import ICD9_RANGES, CONDITION_DRUG_KEYWORDS
from eda.data_processing import (
    prepare_encounter_data,
    build_patient_info,
    add_patient_icd9_lists,
    build_msk_type_table,
    build_analysis_df_msk_only,
)
from eda.duration import compute_condition_duration
from eda.plotting import (
    step5_plot_msk_comorbidities,
    step5_plot_msk_multimorbidity,
    step5_plot_msk_age_groups,
    step5_plot_msk_sex,
    step5_plot_msk_age_sex_patient,
    boxplot_msk_vs_mental_by_sex,
    barplot_msk_type_percentage,
    plot_msk_type_by_sex_percent,
)


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    data_out = root / "data" / "output"

    fig_dir = Path(__file__).parent / "results" / "figures"
    tab_dir = Path(__file__).parent / "results" / "tables"

    patient = pd.read_csv(data_out / "C4MPatient_clean_1_31.csv", dtype=str)
    encounter = pd.read_csv(data_out / "C4MEncounterDiagnosis_clean_1_31.csv", dtype=str)
    medication = pd.read_csv(data_out / "C4MMedication_clean_1_31.csv", dtype=str)

    encounter = encounter.rename(columns={"DiagnosisCode_calc": "icd9"})
    encounter = prepare_encounter_data(encounter)

    patient_info = build_patient_info(encounter, patient)

    enc_icd9 = add_patient_icd9_lists(encounter, ranges=ICD9_RANGES)

    enc_icd9_msk = enc_icd9[
        enc_icd9["type list"].str.contains(r"(?:^|,)msk(?:,|$)", na=False, regex=True)
    ].copy()

    enc_icd9_msk["comorbidity_count"] = enc_icd9_msk["type list"].apply(
        lambda s: sum(t not in ("", "msk") for t in str(s).split(","))
    )

    msk_type_table = build_msk_type_table(enc_icd9, ranges=ICD9_RANGES)

    mental_duration = compute_condition_duration(
        encounter=encounter,
        medication=medication,
        ranges=ICD9_RANGES,
        condition="mental",
        drug_keywords=CONDITION_DRUG_KEYWORDS["mental"],
    )

    msk_duration = compute_condition_duration(
        encounter=encounter,
        medication=medication,
        ranges=ICD9_RANGES,
        condition="msk",
        drug_keywords=CONDITION_DRUG_KEYWORDS["msk"],
    )

    analysis_df = build_analysis_df_msk_only(
        patient_info=patient_info,
        enc_icd9_msk=enc_icd9_msk,
        msk_type_table=msk_type_table,
        msk_duration=msk_duration,
        mental_duration=mental_duration,
    )

    tab_dir.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(tab_dir / "analysis_df.csv", index=False)

    save_fig(step5_plot_msk_comorbidities(analysis_df), fig_dir / "comorbidities.png")
    save_fig(step5_plot_msk_multimorbidity(analysis_df), fig_dir / "multimorbidity.png")
    save_fig(step5_plot_msk_age_groups(analysis_df), fig_dir / "age.png")
    save_fig(step5_plot_msk_sex(analysis_df), fig_dir / "sex.png")
    save_fig(step5_plot_msk_age_sex_patient(analysis_df), fig_dir / "age_sex.png")
    save_fig(boxplot_msk_vs_mental_by_sex(analysis_df), fig_dir / "duration.png")
    save_fig(barplot_msk_type_percentage(analysis_df), fig_dir / "msk_type.png")
    save_fig(plot_msk_type_by_sex_percent(analysis_df), fig_dir / "msk_type_sex.png")


if __name__ == "__main__":
    main()