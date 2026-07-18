from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from eda.clinical_definitions import classify_acute_chronic, SUBGROUP_LABELS


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "resp_mental_analysis" / "results" / "tables"
EDA_DIR = ROOT / "eda_analysis" / "results" / "tables"
FIG_DIR = ROOT / "resp_mental_analysis" / "results" / "figures" / "step2"

FIG_DIR.mkdir(parents=True, exist_ok=True)

COHORT_CSV = DATA_DIR / "analysis_cohort_final.csv"
PRE_LONG_CSV = EDA_DIR / "analysis_long_pre_msk_only.csv"


def save_and_show(fig, save_path=None, show=False):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# data loading and preprocessing
def load_data(cohort_csv=COHORT_CSV, pre_long_csv=PRE_LONG_CSV):
    df = pd.read_csv(cohort_csv)
    rm_long = pd.read_csv(pre_long_csv, low_memory=False)

    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()
    rm_long["Patient_ID"] = rm_long["Patient_ID"].astype(str).str.strip()

    for col in ["Age_at_mental", "time_to_msk_days", "BirthYear"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Age_at_mental"].between(18, 102, inclusive="both")].copy()

    return df, rm_long


def add_acute_chronic_label(rm_long):
    if "icd9_split" in rm_long.columns:
        rm_long["acute_chronic"] = rm_long["icd9_split"].apply(classify_acute_chronic)
    elif "icd9" in rm_long.columns:
        rm_long["acute_chronic"] = rm_long["icd9"].apply(classify_acute_chronic)
    else:
        raise ValueError("analysis_long_pre_msk_only.csv must contain 'icd9_split' or 'icd9'.")

    type_col = "_type" if "_type" in rm_long.columns else "system"
    rm_long = rm_long[rm_long[type_col].isin(["respiratory", "mental"])].copy()

    return rm_long, type_col


def patient_course(x):
    vals = pd.Series(x).dropna().astype(str)

    if "chronic" in vals.values:
        return "chronic"
    if "acute" in vals.values:
        return "acute"
    return None


def patient_course(x):
    vals = pd.Series(x).dropna().astype(str)

    if "chronic" in vals.values:
        return "chronic"
    if "acute" in vals.values:
        return "acute"
    return None


def build_patient_level_courses(df, rm_long, type_col):
   
    df = df.copy()
    rm_long = rm_long.copy()
    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()
    rm_long["Patient_ID"] = rm_long["Patient_ID"].astype(str).str.strip()
    rm_long["DateCreated"] = pd.to_datetime(rm_long["DateCreated"], errors="coerce")

    for col in ["first_msk", "first_mental"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_datetime(df[col], errors="coerce")
    rm_long = rm_long.merge(
        df[["Patient_ID", "first_mental", "first_msk"]],
        on="Patient_ID",
        how="left"
    )

    # keep only rows with usable event dates
    rm_long = rm_long.dropna(subset=["DateCreated", "first_msk"]).copy()

    # mental course: mental before first MSK
    mental_long = rm_long[
        (rm_long[type_col] == "mental") &
        (rm_long["DateCreated"] < rm_long["first_msk"])
    ].copy()

    mental_course = (
        mental_long.groupby("Patient_ID")["acute_chronic"]
        .apply(patient_course)
        .reset_index()
        .rename(columns={"acute_chronic": "mental_course"})
    )

    # respiratory comorbidity: first_mental <= resp < first_msk
    resp_long = rm_long[
        (rm_long[type_col] == "respiratory") &
        (rm_long["first_mental"].notna()) &
        (rm_long["DateCreated"] >= rm_long["first_mental"]) &
        (rm_long["DateCreated"] < rm_long["first_msk"])
    ].copy()

    resp_course = (
        resp_long.groupby("Patient_ID")["acute_chronic"]
        .apply(patient_course)
        .reset_index()
        .rename(columns={"acute_chronic": "resp_course"})
    )

   
    df = df.merge(mental_course, on="Patient_ID", how="left")
    df = df.merge(resp_course, on="Patient_ID", how="left")

    # respiratory comorbidity indicators
    df["resp_any"] = df["resp_course"].notna().astype(int)
    df["acute_resp"] = (df["resp_course"] == "acute").astype(int)
    df["chronic_resp"] = (df["resp_course"] == "chronic").astype(int)

    return df


def build_cohorts(df):
    cohorts = {
        "acute_resp": df[df["acute_resp"] == 1].copy(),
        "chronic_resp": df[df["chronic_resp"] == 1].copy(),
        "acute_mental": df[df["mental_course"] == "acute"].copy(),
        "chronic_mental": df[df["mental_course"] == "chronic"].copy(),
        "resp_all": df[df["resp_any"] == 1].copy(),
        "mental_all": df[df["mental_course"].notna()].copy(),
    }
    return cohorts


def print_cohort_counts(cohorts):
    print("Respiratory patients:", cohorts["resp_all"]["Patient_ID"].nunique())
    print("Mental patients:", cohorts["mental_all"]["Patient_ID"].nunique())

    print("acute_resp n =", cohorts["acute_resp"]["Patient_ID"].nunique())
    print("chronic_resp n =", cohorts["chronic_resp"]["Patient_ID"].nunique())
    print("acute_mental n =", cohorts["acute_mental"]["Patient_ID"].nunique())
    print("chronic_mental n =", cohorts["chronic_mental"]["Patient_ID"].nunique())


def make_age_groups(series):
    bins = [18, 25] + list(range(30, 95, 5)) + [np.inf]
    labels = ["18-24"] + [f"{i}-{i+4}" for i in range(25, 90, 5)] + ["90+"]
    return pd.cut(series, bins=bins, labels=labels, right=False), labels


def plot_age_distribution_5yr(data, title, age_col="Age_at_mental", save_path=None):
    d = data.copy()
    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d = d.dropna(subset=[age_col]).copy()

    d["age_group"], labels = make_age_groups(d[age_col])

    counts = d["age_group"].value_counts().sort_index()
    pct = counts / counts.sum() * 100 if counts.sum() > 0 else counts * 0
    total_n = int(counts.sum())

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(counts.index.astype(str), pct.values)

    ax.set_ylabel("Percentage of patients (%)")
    ax.set_xlabel("Age at first MSK (years)")
    ax.set_title(f"{title} (n={total_n:,})")
    plt.xticks(rotation=45, ha="right")

    ymax = pct.max() if len(pct) else 0
    ax.set_ylim(0, max(8, ymax + 3))

    for bar, p, n in zip(bars, pct.values, counts.values):
        if n == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{p:.1f}%\n(n={n:,})",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    save_and_show(fig, save_path)


def plot_sex_distribution(data, title="Sex distribution", sex_col="Sex", save_path=None):
    sex_counts = data[sex_col].value_counts()
    sex_percent = sex_counts / sex_counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(sex_percent.index.astype(str), sex_percent.values)

    ax.set_ylabel("Percentage of patients (%)")
    ax.set_xlabel("Sex")
    ax.set_title(title)
    ymax = sex_percent.max()
    ax.set_ylim(0, ymax + 10)
    for bar, pct, n in zip(bars, sex_percent.values, sex_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%\n(n={n})",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    save_and_show(fig, save_path)


def plot_sex_by_msk_subtype(
    df,
    subgroup_col="first_msk_subgroup",
    sex_col="Sex",
    top_n=10,
    save_path=None
):
    df = df.copy()

    df["msk_label"] = df[subgroup_col].map(SUBGROUP_LABELS).fillna(df[subgroup_col])
    subgroup_col = "msk_label"

    df = df.dropna(subset=[sex_col, subgroup_col]).copy()

    sex_msk_counts = pd.crosstab(df[subgroup_col], df[sex_col])

    top_idx = sex_msk_counts.sum(axis=1).sort_values(ascending=False).head(top_n).index
    sex_msk_counts = sex_msk_counts.loc[top_idx]

    sex_msk_pct = sex_msk_counts.div(sex_msk_counts.sum(axis=1), axis=0) * 100

    ax = sex_msk_pct.plot(
        kind="bar",
        stacked=True,
        figsize=(14, 6)
    )
    fig = ax.get_figure()

    plt.ylabel("Percentage of patients (%)")
    plt.xlabel("First MSK subtype")
    plt.title(f"Sex distribution by first MSK subtype (Top {top_n})")
    plt.xticks(rotation=60, ha="right")

    for i, subtype in enumerate(sex_msk_pct.index):
        cumulative = 0
        for sex in sex_msk_pct.columns:
            pct = sex_msk_pct.loc[subtype, sex]
            if pct >= 8:
                ax.text(
                    i,
                    cumulative + pct / 2,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white"
                )
            cumulative += pct

    plt.legend(title="Sex")
    plt.tight_layout()
    save_and_show(fig, save_path)


def plot_resp_vs_mental_age(resp_df, mental_df, save_path=None):
    resp = resp_df.copy()
    mental = mental_df.copy()

    resp["age_group"], labels = make_age_groups(resp["Age_at_mental"])
    mental["age_group"], _ = make_age_groups(mental["Age_at_mental"])

    resp_counts = resp["age_group"].value_counts().sort_index()
    mental_counts = mental["age_group"].value_counts().sort_index()

    resp_pct = resp_counts / resp_counts.sum() * 100
    mental_pct = mental_counts / mental_counts.sum() * 100

    x = np.arange(len(labels))
    width = 0.42

    fig, ax = plt.subplots(figsize=(13, 5))

    bars1 = ax.bar(
        x - width / 2,
        resp_pct,
        width,
        label=f"Respiratory (n={resp_counts.sum():,})"
    )

    bars2 = ax.bar(
        x + width / 2,
        mental_pct,
        width,
        label=f"Mental (n={mental_counts.sum():,})"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Percentage of patients (%)")
    ax.set_xlabel("Age at first MSK (years)")
    ax.set_title("Age distribution: respiratory vs mental conditions")
    ax.legend()

    for bar, pct, n in zip(bars1, resp_pct, resp_counts):
        if n > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{pct:.1f}%\n n={n:,}",
                ha="center",
                va="bottom",
                fontsize=6
            )

    for bar, pct, n in zip(bars2, mental_pct, mental_counts):
        if n > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{pct:.1f}%\n n={n:,}",
                ha="center",
                va="bottom",
                fontsize=6
            )

    plt.tight_layout()
    save_and_show(fig, save_path)



def main():
    df, rm_long = load_data()
    rm_long, type_col = add_acute_chronic_label(rm_long)
    df = build_patient_level_courses(df, rm_long, type_col)
    cohorts = build_cohorts(df)

    print_cohort_counts(cohorts)

    # age plots
    plot_age_distribution_5yr(
        cohorts["acute_resp"],
        "Age distribution among patients with acute respiratory conditions",
        save_path=FIG_DIR / "age_acute_resp.png"
    )
    plot_age_distribution_5yr(
        cohorts["chronic_resp"],
        "Age distribution among patients with chronic respiratory conditions",
        save_path=FIG_DIR / "age_chronic_resp.png"
    )
    plot_age_distribution_5yr(
        cohorts["acute_mental"],
        "Age distribution among patients with acute mental conditions",
        save_path=FIG_DIR / "age_acute_mental.png"
    )
    plot_age_distribution_5yr(
        cohorts["chronic_mental"],
        "Age distribution among patients with chronic mental conditions",
        save_path=FIG_DIR / "age_chronic_mental.png"
    )
    plot_age_distribution_5yr(
        cohorts["resp_all"],
        "Age distribution among patients with respiratory conditions",
        save_path=FIG_DIR / "age_resp_all.png"
    )
    plot_age_distribution_5yr(
        cohorts["mental_all"],
        "Age distribution among patients with mental conditions",
        save_path=FIG_DIR / "age_mental_all.png"
    )
    plot_age_distribution_5yr(
        df,
        "Age distribution at first MSK diagnosis",
        save_path=FIG_DIR / "age_msk.png"
    )

    # sex plots
    plot_sex_distribution(
        df,
        title="Sex distribution at first MSK diagnosis",
        save_path=FIG_DIR / "sex_distribution.png"
    )
    plot_sex_by_msk_subtype(
        df,
        top_n=10,
        save_path=FIG_DIR / "sex_by_msk.png"
    )

    # comparison plots
    plot_resp_vs_mental_age(
        cohorts["resp_all"],
        cohorts["mental_all"],
        save_path=FIG_DIR / "age_compare.png"
    )

if __name__ == "__main__":
    main()