from pathlib import Path

import pandas as pd


def compare_tables(
    old_path: Path,
    new_path: Path,
    table_name: str,
) -> None:
    old = pd.read_csv(
        old_path,
        dtype=str,
        low_memory=False,
    )
    new = pd.read_csv(
        new_path,
        dtype=str,
        low_memory=False,
    )

    old["Patient_ID"] = (
        old["Patient_ID"]
        .astype(str)
        .str.strip()
    )
    new["Patient_ID"] = (
        new["Patient_ID"]
        .astype(str)
        .str.strip()
    )

    old_ids = set(old["Patient_ID"])
    new_ids = set(new["Patient_ID"])

    print("\n" + "=" * 60)
    print(table_name)
    print("=" * 60)

    print("Old rows:", len(old))
    print("New rows:", len(new))

    print(
        "Old patients:",
        old["Patient_ID"].nunique(),
    )
    print(
        "New patients:",
        new["Patient_ID"].nunique(),
    )

    print(
        "Only in old:",
        len(old_ids - new_ids),
    )
    print(
        "Only in new:",
        len(new_ids - old_ids),
    )

    print(
        "Same Patient_ID set:",
        old_ids == new_ids,
    )

    common_columns = sorted(
        set(old.columns)
        & set(new.columns)
    )

    old_compare = (
        old[common_columns]
        .fillna("")
        .sort_values(common_columns)
        .reset_index(drop=True)
    )

    new_compare = (
        new[common_columns]
        .fillna("")
        .sort_values(common_columns)
        .reset_index(drop=True)
    )

    print(
        "Same values in common columns:",
        old_compare.equals(new_compare),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    old_dir = (
        root
        / "resp_mental_analysis"
        / "results"
        / "tables"
    )

    new_dir = (
        root
        / "final_cohort"
        / "results"
        / "tables"
    )

    compare_tables(
        old_dir
        / "analysis_cohort_final.csv",
        new_dir
        / "analysis_cohort_final.csv",
        "Encounter-level pre-MSK table",
    )



if __name__ == "__main__":
    main()