"""Cohort-level EDA for the mental-to-MSK project.

Produces two output groups:
1. 02_pre_msk
2. 03_final_cohort

Run from repository root:
    python -m mental_msk_prediction.visualization.cohort_eda
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mental_msk_prediction.config.clinical_definitions import (
    COMORBIDITY_SYSTEMS_EDA,
    COMORBIDITY_SYSTEMS_MODEL,
    classify_system_broad,
    extract_icd9_3,
)
from mental_msk_prediction.config.paths import (
    CLEANED_DIAGNOSIS_FILE,
    FIGURES_DIR,
    RESULTS_DATA_DIR,
    TABLES_DIR,
    create_project_directories,
)

COHORT_DATA_DIR = RESULTS_DATA_DIR / "cohort"
FINAL_COHORT_FILE = COHORT_DATA_DIR / "final_cohort.csv"
FIRST_MENTAL_FILE = COHORT_DATA_DIR / "first_mental_date_diagnoses.csv"
FIRST_MSK_FILE = COHORT_DATA_DIR / "first_msk_date_diagnoses.csv"

PRE_TABLE_DIR = TABLES_DIR / "eda" / "02_pre_msk"
PRE_FIG_DIR = FIGURES_DIR / "eda" / "02_pre_msk"
FINAL_TABLE_DIR = TABLES_DIR / "eda" / "03_final_cohort"
FINAL_FIG_DIR = FIGURES_DIR / "eda" / "03_final_cohort"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8-sig")


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _add_total(ax: plt.Axes, n: int, label: str = "Total N") -> None:
    ax.text(
        0.99,
        0.98,
        f"{label} = {n:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )


def _plot_vertical(
    table: pd.DataFrame,
    category: str,
    title: str,
    xlabel: str,
    ylabel: str,
    total_n: int,
    path: Path,
    rotate: int = 0,
    figsize: tuple[float, float] = (9, 5),
    total_label: str = "Total N",
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(table[category].astype(str), table["Percent"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    offset = max(float(table["Percent"].max()) * 0.015, 0.2) if not table.empty else 0.2
    for bar, count, pct in zip(bars, table["N_patients"], table["Percent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{pct:.1f}%\n(n={int(count):,})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _add_total(ax, total_n, total_label)
    if not table.empty:
        ax.set_ylim(0, max(float(table["Percent"].max()) * 1.23, 5))
    fig.tight_layout()
    _save_fig(fig, path)


def _plot_horizontal(
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    total_n: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, max(8, 0.48 * len(table) + 2)))
    bars = ax.barh(table["Display_label"].astype(str), table["Percent"])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    max_value = float(table["Percent"].max()) if not table.empty else 0
    offset = max(max_value * 0.015, 0.15)
    for bar, count, pct in zip(bars, table["N_patients"], table["Percent"]):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% (n={int(count):,})",
            ha="left",
            va="center",
            fontsize=8,
        )
    _add_total(ax, total_n)
    ax.set_xlim(0, max(max_value * 1.35, 5))
    fig.tight_layout()
    _save_fig(fig, path)


def validate_inputs() -> None:
    required = [
        CLEANED_DIAGNOSIS_FILE,
        FINAL_COHORT_FILE,
        FIRST_MENTAL_FILE,
        FIRST_MSK_FILE,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing EDA inputs:\n" + "\n".join(f"  - {p}" for p in missing)
        )


def prepare_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """Match the cohort definition exactly: calculated ICD-9 codes only."""
    required = [
        "Patient_ID",
        "DateCreated",
        "DiagnosisCodeType_calc",
        "DiagnosisCode_calc",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("Diagnosis table is missing: " + ", ".join(missing))

    out = df.copy()
    out["Patient_ID"] = out["Patient_ID"].astype("string").str.strip()
    out["DiagnosisCodeType_calc"] = (
        out["DiagnosisCodeType_calc"].astype("string").str.strip().str.upper()
    )
    out = out[out["DiagnosisCodeType_calc"].eq("ICD9")].copy()
    out["diagnosis_date"] = pd.to_datetime(out["DateCreated"], errors="coerce")
    out["icd9"] = out["DiagnosisCode_calc"].astype("string").str.strip()
    out["icd9_3"] = out["icd9"].apply(extract_icd9_3).astype("Int64")
    out["system"] = out["icd9"].apply(classify_system_broad).astype("string")
    return out.dropna(
        subset=["Patient_ID", "diagnosis_date", "icd9", "icd9_3", "system"]
    ).copy()


def _comorbidity_summary(
    patient_ids: pd.Series,
    records: pd.DataFrame,
    systems: list[str],
) -> pd.DataFrame:
    ids = patient_ids.astype("string").str.strip().dropna().drop_duplicates()
    total_n = ids.nunique()
    rows = []
    for system in systems:
        count = records.loc[records["system"].eq(system), "Patient_ID"].nunique()
        rows.append(
            {
                "system": system,
                "N_patients": int(count),
                "Percent": count / total_n * 100 if total_n else 0,
                "Total": total_n,
            }
        )
    return pd.DataFrame(rows).sort_values("Percent", ascending=False).reset_index(drop=True)


def _multimorbidity_summary(
    patient_ids: pd.Series,
    records: pd.DataFrame,
    systems: list[str],
) -> pd.DataFrame:
    ids = patient_ids.astype("string").str.strip().dropna().drop_duplicates()
    total_n = ids.nunique()
    counts = (
        records.loc[records["system"].isin(systems), ["Patient_ID", "system"]]
        .drop_duplicates()
        .groupby("Patient_ID", observed=True)["system"]
        .nunique()
        .reindex(ids, fill_value=0)
    )
    summary = (
        counts.value_counts()
        .sort_index()
        .rename_axis("n_systems")
        .reset_index(name="N_patients")
    )
    summary["Percent"] = summary["N_patients"] / total_n * 100 if total_n else 0
    summary["Total"] = total_n
    return summary


def _save_comorbidity_pair(
    patient_ids: pd.Series,
    records: pd.DataFrame,
    systems: list[str],
    table_dir: Path,
    fig_dir: Path,
    prefix: str,
    comorbidity_title: str,
    multimorbidity_title: str,
    total_label: str,
) -> None:
    total_n = patient_ids.astype("string").str.strip().dropna().nunique()
    comorbidity = _comorbidity_summary(patient_ids, records, systems)
    multimorbidity = _multimorbidity_summary(patient_ids, records, systems)
    _save_table(comorbidity, table_dir / f"{prefix}_comorbidity.csv")
    _save_table(multimorbidity, table_dir / f"{prefix}_multimorbidity.csv")
    _plot_vertical(
        comorbidity,
        "system",
        comorbidity_title,
        "Disease system",
        "Patients (%)",
        total_n,
        fig_dir / f"{prefix}_comorbidity.png",
        rotate=30,
        figsize=(10, 6),
        total_label=total_label,
    )
    _plot_vertical(
        multimorbidity,
        "n_systems",
        multimorbidity_title,
        "Number of pre-MSK disease systems",
        "Patients (%)",
        total_n,
        fig_dir / f"{prefix}_multimorbidity.png",
        figsize=(9, 5),
        total_label=total_label,
    )


def run_pre_msk_eda(diagnosis: pd.DataFrame) -> None:
    first_msk = (
        diagnosis.loc[diagnosis["system"].eq("msk")]
        .groupby("Patient_ID", observed=True)["diagnosis_date"]
        .min()
        .rename("first_msk_date")
    )
    first_non_msk = (
        diagnosis.loc[~diagnosis["system"].eq("msk")]
        .groupby("Patient_ID", observed=True)["diagnosis_date"]
        .min()
        .rename("first_non_msk_date")
    )
    patients = first_msk.to_frame().join(first_non_msk, how="left").reset_index()
    patients["msk_timing_type"] = np.select(
        [
            patients["first_non_msk_date"] < patients["first_msk_date"],
            patients["first_non_msk_date"] == patients["first_msk_date"],
            patients["first_non_msk_date"] > patients["first_msk_date"],
            patients["first_non_msk_date"].isna(),
        ],
        [
            "Pre-existing then MSK",
            "Co-occurrence",
            "MSK with later comorbidity",
            "MSK only",
        ],
        default="Unknown",
    )
    order = [
        "Pre-existing then MSK",
        "Co-occurrence",
        "MSK with later comorbidity",
        "MSK only",
    ]
    total_n = patients["Patient_ID"].nunique()
    timing = (
        patients["msk_timing_type"]
        .value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("msk_timing_type")
        .reset_index(name="N_patients")
    )
    timing["Percent"] = timing["N_patients"] / total_n * 100
    timing["Total"] = total_n
    _save_table(timing, PRE_TABLE_DIR / "msk_timing_type_distribution.csv")
    _plot_vertical(
        timing,
        "msk_timing_type",
        "MSK diagnosis timing relative to other recorded disease systems",
        "MSK record pattern",
        "Patients with MSK diagnosis (%)",
        total_n,
        PRE_FIG_DIR / "msk_timing_type_distribution.png",
        rotate=20,
        figsize=(11, 6),
    )

    group = patients[patients["msk_timing_type"].eq("Pre-existing then MSK")].copy()
    records = diagnosis.merge(
        group[["Patient_ID", "first_msk_date"]],
        on="Patient_ID",
        how="inner",
        validate="many_to_one",
    )
    records = records[
        ~records["system"].eq("msk")
        & (records["diagnosis_date"] < records["first_msk_date"])
    ].copy()
    _save_comorbidity_pair(
        group["Patient_ID"],
        records,
        COMORBIDITY_SYSTEMS_EDA,
        PRE_TABLE_DIR,
        PRE_FIG_DIR,
        "pre_existing_then_msk",
        "Pre-MSK comorbidities in patients with pre-existing conditions before MSK",
        "Pre-MSK multimorbidity in patients with pre-existing conditions before MSK",
        "Total pre-existing then MSK N",
    )


def _age_summary(age: pd.Series, total_n: int) -> pd.DataFrame:
    clean = pd.to_numeric(age, errors="coerce")
    max_age = int(np.floor(clean.max()))
    starts = list(range(18, max_age + 1, 5)) or [18]
    final_start = starts[-1]
    regular = starts[:-1]
    bins = regular + [final_start, np.inf]
    labels = [f"{x}-{x + 4}" for x in regular] + [f"{final_start}+"]
    groups = pd.cut(clean, bins=bins, labels=labels, right=False, include_lowest=True)
    summary = (
        groups.value_counts()
        .sort_index()
        .rename_axis("Age_group")
        .reset_index(name="N_patients")
    )
    summary["Percent"] = summary["N_patients"] / total_n * 100
    summary["Total"] = total_n
    return summary


def _plot_age(age: pd.Series, total_n: int) -> None:
    clean = pd.to_numeric(age, errors="coerce").dropna()
    bins = np.arange(np.floor(clean.min()), np.ceil(clean.max()) + 3, 2)
    counts, edges = np.histogram(clean, bins=bins)
    pct = counts / total_n * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(edges[:-1], pct, width=np.diff(edges), align="edge")
    ax.set_title("Age at first mental health diagnosis")
    ax.set_xlabel("Age at first mental health diagnosis (years)")
    ax.set_ylabel("Patients (%)")
    _add_total(ax, total_n)
    ax.text(
        0.99,
        0.89,
        f"Mean = {clean.mean():.1f}; Median = {clean.median():.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    _save_fig(fig, FINAL_FIG_DIR / "age_at_first_mental.png")


def _first_date_summary(
    records: pd.DataFrame,
    condition: str,
    final_ids: set[str],
    total_n: int,
    top_n: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = f"first_{condition}"
    label_col = f"{prefix}_display_label"
    code_col = f"{prefix}_icd9_3"
    required = ["Patient_ID", label_col, code_col]
    missing = [c for c in required if c not in records.columns]
    if missing:
        raise KeyError(f"First-{condition} table is missing: " + ", ".join(missing))
    df = records.copy()
    df["Patient_ID"] = df["Patient_ID"].astype("string").str.strip()
    df = df[df["Patient_ID"].isin(final_ids)].copy()
    df[label_col] = df[label_col].astype("string").str.strip().fillna("Unknown")
    df[code_col] = df[code_col].astype("string").str.strip().fillna("Unknown")
    df["Display_label"] = np.where(
        df[code_col].eq("Unknown"),
        df[label_col],
        df[code_col] + " " + df[label_col],
    )
    patient_groups = df[["Patient_ID", "Display_label"]].drop_duplicates()
    full = (
        patient_groups["Display_label"]
        .value_counts()
        .rename_axis("Display_label")
        .reset_index(name="N_patients")
    )
    full["Percent"] = full["N_patients"] / total_n * 100
    full = full.sort_values(["N_patients", "Display_label"], ascending=[False, True]).reset_index(drop=True)
    plot = full.head(top_n).copy()
    omitted = full.iloc[top_n:]
    if not omitted.empty:
        omitted_labels = set(omitted["Display_label"])
        other_n = patient_groups.loc[
            patient_groups["Display_label"].isin(omitted_labels), "Patient_ID"
        ].nunique()
        plot = pd.concat(
            [
                plot,
                pd.DataFrame(
                    {
                        "Display_label": ["Other"],
                        "N_patients": [other_n],
                        "Percent": [other_n / total_n * 100],
                    }
                ),
            ],
            ignore_index=True,
        )
    return full, plot


def run_final_cohort_eda(
    diagnosis: pd.DataFrame,
    final: pd.DataFrame,
    first_mental: pd.DataFrame,
    first_msk: pd.DataFrame,
) -> None:
    final = final.copy()
    final["Patient_ID"] = final["Patient_ID"].astype("string").str.strip()
    final["first_msk_date"] = pd.to_datetime(final["first_msk_date"], errors="coerce")
    total_n = final["Patient_ID"].nunique()
    final_ids = set(final["Patient_ID"])

    age = pd.to_numeric(final["Age_at_first_mental"], errors="coerce")
    age.describe().rename("value").to_csv(FINAL_TABLE_DIR / "age_at_first_mental_summary.csv")
    _plot_age(age, total_n)
    age_groups = _age_summary(age, total_n)
    _save_table(age_groups, FINAL_TABLE_DIR / "age_at_first_mental_5year_groups.csv")
    _plot_vertical(
        age_groups,
        "Age_group",
        "Age at first mental health diagnosis by five-year group",
        "Age group (years)",
        "Patients (%)",
        total_n,
        FINAL_FIG_DIR / "age_at_first_mental_5year_groups.png",
        rotate=45,
        figsize=(12, 6),
    )

    sex = final["Sex"].fillna("Unknown").value_counts().rename_axis("Sex").reset_index(name="N_patients")
    sex["Percent"] = sex["N_patients"] / total_n * 100
    sex["Total"] = total_n
    _save_table(sex, FINAL_TABLE_DIR / "sex_distribution.csv")
    _plot_vertical(
        sex,
        "Sex",
        "Sex distribution in the final cohort",
        "Sex",
        "Patients (%)",
        total_n,
        FINAL_FIG_DIR / "sex_distribution.png",
        figsize=(7, 5),
    )

    records = diagnosis.merge(
        final[["Patient_ID", "first_msk_date"]],
        on="Patient_ID",
        how="inner",
        validate="many_to_one",
    )
    records = records[
        ~records["system"].eq("msk")
        & (records["diagnosis_date"] < records["first_msk_date"])
    ].copy()
    _save_comorbidity_pair(
        final["Patient_ID"],
        records,
        COMORBIDITY_SYSTEMS_MODEL,
        FINAL_TABLE_DIR,
        FINAL_FIG_DIR,
        "final_cohort_pre_msk",
        "Pre-MSK comorbidities in the final mental-to-MSK cohort (excluding mental and MSK)",
        "Pre-MSK multimorbidity in the final mental-to-MSK cohort (excluding mental and MSK)",
        "Final cohort N",
    )

    mental_full, mental_plot = _first_date_summary(
        first_mental, "mental", final_ids, total_n
    )
    msk_full, msk_plot = _first_date_summary(first_msk, "msk", final_ids, total_n)
    _save_table(mental_full, FINAL_TABLE_DIR / "first_mental_subgroup_full.csv")
    _save_table(mental_plot, FINAL_TABLE_DIR / "first_mental_subgroup_plot.csv")
    _save_table(msk_full, FINAL_TABLE_DIR / "first_msk_subgroup_full.csv")
    _save_table(msk_plot, FINAL_TABLE_DIR / "first_msk_subgroup_plot.csv")
    _plot_horizontal(
        mental_plot,
        "Mental diagnoses recorded on the first mental health diagnosis date",
        "Final cohort patients (%)",
        total_n,
        FINAL_FIG_DIR / "first_mental_subgroup.png",
    )
    _plot_horizontal(
        msk_plot,
        "MSK diagnoses recorded on the first MSK diagnosis date",
        "Final cohort patients (%)",
        total_n,
        FINAL_FIG_DIR / "first_msk_subgroup.png",
    )

    days = pd.to_numeric(final["time_to_msk_days"], errors="coerce")
    edges = [0] + [365 * i for i in range(1, 11)] + [np.inf]
    labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10", "10+"]
    groups = pd.cut(days, bins=edges, labels=labels, right=True, include_lowest=False)
    years = groups.value_counts().sort_index().rename_axis("years_to_first_msk").reset_index(name="N_patients")
    years["Percent"] = years["N_patients"] / total_n * 100
    years["Total"] = total_n
    _save_table(years, FINAL_TABLE_DIR / "time_from_first_mental_to_first_msk.csv")
    _plot_vertical(
        years,
        "years_to_first_msk",
        "Time from first mental health diagnosis to first MSK diagnosis",
        "Years to first MSK diagnosis",
        "Patients (%)",
        total_n,
        FINAL_FIG_DIR / "time_from_first_mental_to_first_msk.png",
        figsize=(12, 6),
    )

    within = days[(days > 0) & (days <= 365)].copy()
    months = np.ceil(within / (365 / 12)).astype(int).clip(1, 12)
    within_n = len(within)
    monthly = months.value_counts().reindex(range(1, 13), fill_value=0).rename_axis("month").reset_index(name="N_patients")
    monthly["Percent"] = monthly["N_patients"] / within_n * 100 if within_n else 0
    monthly["Total_within_one_year"] = within_n
    _save_table(monthly, FINAL_TABLE_DIR / "time_to_msk_within_one_year_by_month.csv")
    _plot_vertical(
        monthly,
        "month",
        "Time to first MSK diagnosis within one year of first mental health diagnosis",
        "Month after first mental health diagnosis",
        "Patients within one year (%)",
        within_n,
        FINAL_FIG_DIR / "time_to_msk_within_one_year_by_month.png",
        figsize=(11, 6),
        total_label="Within-one-year N",
    )


def main() -> None:
    create_project_directories()
    validate_inputs()
    PRE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PRE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STEP 1 — LOAD EDA INPUTS")
    print("=" * 80)
    diagnosis_raw = _read(CLEANED_DIAGNOSIS_FILE)
    final = _read(FINAL_COHORT_FILE)
    first_mental = _read(FIRST_MENTAL_FILE)
    first_msk = _read(FIRST_MSK_FILE)
    print(f"Diagnosis rows loaded: {len(diagnosis_raw):,}")
    print(f"Final cohort patients: {final['Patient_ID'].nunique():,}")

    print("\n" + "=" * 80)
    print("STEP 2 — PREPARE DIAGNOSIS RECORDS")
    print("=" * 80)
    diagnosis = prepare_diagnosis(diagnosis_raw)
    print(f"Usable classified diagnosis rows: {len(diagnosis):,}")

    print("\n" + "=" * 80)
    print("STEP 3 — RUN 02_PRE_MSK EDA")
    print("=" * 80)
    run_pre_msk_eda(diagnosis)

    print("\n" + "=" * 80)
    print("STEP 4 — RUN 03_FINAL_COHORT EDA")
    print("=" * 80)
    run_final_cohort_eda(diagnosis, final, first_mental, first_msk)

    print("\n" + "=" * 80)
    print("COHORT EDA COMPLETED")
    print("=" * 80)
    print(f"Pre-MSK figures: {PRE_FIG_DIR}")
    print(f"Final-cohort figures: {FINAL_FIG_DIR}")
    print(f"Pre-MSK tables: {PRE_TABLE_DIR}")
    print(f"Final-cohort tables: {FINAL_TABLE_DIR}")


if __name__ == "__main__":
    main()
