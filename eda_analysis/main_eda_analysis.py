from pathlib import Path
import pandas as pd

from eda.duration import compute_condition_duration
from eda_analysis.icd_naming import (
    build_icd9_name_map_from_encounter,
    build_icd3_name_map_from_icd9_map,
    format_icd9_list_with_names,
)
from eda_analysis.eda_steps import (
    filter_cohort,
    build_pre_msk_encounter,
    rebuild_pre_msk_patient_table,
)
from eda_analysis.plots import (
    step5_plot_msk_multimorbidity,
    plot_topk_nonmsk_combos_all_counts,
    prep,
    plot_n,
)
from eda.clinical_definitions import ICD9_RANGES, CONDITION_DRUG_KEYWORDS
from eda_analysis.duration_quality import stopdate_missing_rate_among_med_users


def main():
    ROOT = Path(__file__).resolve().parents[1]
    DATA_OUT = ROOT / "data" / "output"
    EDA_TABLES = ROOT / "eda" / "results" / "tables"

    RESULTS = Path(__file__).parent / "results"
    FIG_DIR = RESULTS / "figures"
    TAB_DIR = RESULTS / "tables"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    analysis_df = pd.read_csv(EDA_TABLES / "analysis_df.csv", dtype=str)
    encounter = pd.read_csv(DATA_OUT / "C4MEncounterDiagnosis_clean_1_31.csv", dtype=str)
    medication = pd.read_csv(DATA_OUT / "C4MMedication_clean_1_31.csv", dtype=str)

    if "DiagnosisCode_calc" in encounter.columns and "icd9" not in encounter.columns:
        encounter = encounter.rename(columns={"DiagnosisCode_calc": "icd9"})

    # Duration analysis
    print("\nRunning MSK duration...")
    msk_duration = compute_condition_duration(
        encounter,
        medication,
        ICD9_RANGES,
        condition="msk",
        drug_keywords=CONDITION_DRUG_KEYWORDS["msk"],
    )
    stopdate_missing_rate_among_med_users(msk_duration, "MSK")

    print("\nRunning Mental duration...")
    mental_duration = compute_condition_duration(
        encounter,
        medication,
        ICD9_RANGES,
        condition="mental",
        drug_keywords=CONDITION_DRUG_KEYWORDS["mental"],
    )
    stopdate_missing_rate_among_med_users(mental_duration, "Mental")

    # Step 1 — filter cohort
    # analysis_df2 is only a cohort index table
    analysis_df2 = filter_cohort(analysis_df)

    print("\nFiltered cohort:")
    print("Rows:", len(analysis_df2))
    print("Patients:", analysis_df2["Patient_ID"].nunique())

    # Step 2 — ICD naming maps
    icd9_to_name = build_icd9_name_map_from_encounter(encounter)
    icd3_to_name = build_icd3_name_map_from_icd9_map(icd9_to_name)

    # Step 3 — encounter level pre-MSK table
    pre_long = build_pre_msk_encounter(
        encounter_df=encounter,
        cohort_df=analysis_df2,
        icd9_to_name=icd9_to_name,
    )

    print("\nPre-MSK encounter-level table:")
    print("Rows:", len(pre_long))
    print("Patients:", pre_long["Patient_ID"].nunique())

    pre_long.to_csv(TAB_DIR / "analysis_long_pre_msk_only.csv", index=False)

    # Step 4 — rebuild patient-level pre-MSK table
    # analysis_df2_pre_msk is the final analysis table
    analysis_df2_pre_msk = rebuild_pre_msk_patient_table(pre_long)

    print("\nPre-MSK patient-level table:")
    print("Rows:", len(analysis_df2_pre_msk))
    print("Patients:", analysis_df2_pre_msk["Patient_ID"].nunique())

    # unified naming only here
    analysis_df2_pre_msk["icd9 list (named)"] = analysis_df2_pre_msk["icd9 list"].apply(
        lambda x: format_icd9_list_with_names(
            x,
            icd9_map=icd9_to_name,
            icd3_map=icd3_to_name,
            keep_unknown=False,
        )
    )

    analysis_df2_pre_msk.to_csv(TAB_DIR / "analysis_df2_pre_msk_named.csv", index=False)

    print("\nSaved tables:")
    print(TAB_DIR / "analysis_long_pre_msk_only.csv")
    print(TAB_DIR / "analysis_df2_pre_msk_named.csv")

    # Step 5 — plots based on final pre-MSK patient-level table
    print("\nGenerating plots from pre-MSK patient-level table...")

    step5_plot_msk_multimorbidity(
        analysis_df2_pre_msk,
        save_path=FIG_DIR / "pre_msk_multimorbidity.png"
    )

    plot_topk_nonmsk_combos_all_counts(
        analysis_df2_pre_msk,
        save_dir=FIG_DIR / "top_nonmsk_combos_pre_msk"
    )

    df_prepped = prep(analysis_df2_pre_msk)

    for n in [1, 2, 3, 4]:
        plot_n(
            df_prepped=df_prepped,
            name_map=icd3_to_name,
            n=n,
            save_path=FIG_DIR / f"pre_msk_combo_n{n}.png"
        )

    print("\nSaved figures:")
    print(FIG_DIR / "pre_msk_multimorbidity.png")
    print(FIG_DIR / "top_nonmsk_combos_pre_msk")
    print(FIG_DIR / "pre_msk_combo_n1.png")
    print(FIG_DIR / "pre_msk_combo_n2.png")
    print(FIG_DIR / "pre_msk_combo_n3.png")
    print(FIG_DIR / "pre_msk_combo_n4.png")


if __name__ == "__main__":
    main()