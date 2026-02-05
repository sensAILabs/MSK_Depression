import time
# from pipeline import run_full_pipeline
from pipeline import (
    stage1_split_good_bad,
    stage2_clean_filter_merge,
    stage3_load_and_basic_clean,
    stage4_enrich_encounter_dx,
    stage4_enrich_medication,
    stage5_output_cleaned,
)


def run_full_pipeline(
    *,
    do_stage1_split: bool = False,
    do_stage2_merge: bool = False,
) -> None:
    """
    If you already have *_final.csv and cache loaders read them, you can skip stage1/stage2.
    """
    if do_stage1_split:
        stage1_split_good_bad()

    if do_stage2_merge:
        stage2_clean_filter_merge()

    # real cleaning
    patient, medication, encounter_dx = stage3_load_and_basic_clean()
    encounter_dx = stage4_enrich_encounter_dx(encounter_dx)
    medication = stage4_enrich_medication(medication)
    stage5_output_cleaned(patient, encounter_dx, medication)

    print("[pipeline] Done.")


def main():
    start = time.time()

    # run_full_pipeline(do_stage1_split=False, do_stage2_merge=False)

    # Rerun the entire process from the original CSV:
    run_full_pipeline(do_stage1_split=True, do_stage2_merge=True)

    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()