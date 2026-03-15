import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def _add_bar_labels(ax, bars, percentages, counts=None, y_offset=0.5, fontsize=6):
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if not np.isfinite(h):
            continue

        if counts is None:
            label = f"{percentages[i]:.1f}%"
        else:
            label = f"{percentages[i]:.1f}% (n={int(counts[i])})"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

def step5_plot_msk_comorbidities(df, patient_col="Patient_ID"):
    types = ["cancer", "endocrine", "mental", "cardiovascular", "respiratory", "genitourinary"]
    total_n = len(df)
    counts = []
    percentages = []
    for t in types:
        c = df["type list"].fillna("").str.split(",").apply(lambda x: t in x).sum()
        p = c / total_n * 100 if total_n > 0 else 0
        counts.append(c)
        percentages.append(p)
    s_pct = pd.Series(percentages, index=types).sort_values(ascending=False)
    s_cnt = pd.Series(counts, index=types).reindex(s_pct.index)
    fig, ax = plt.subplots()
    bars = ax.bar(s_pct.index, s_pct.values)
    ax.set_ylabel("Percentage of MSK patients (%)")
    ax.set_xlabel("Comorbid chronic system")
    ax.set_title(f"Comorbid chronic diseases in MSK patients (N={total_n})")
    plt.xticks(rotation=30, ha="right")
    _add_bar_labels(ax, bars, s_pct.values, s_cnt.values, y_offset=0.5)

    plt.tight_layout()
    return fig


def step5_plot_msk_multimorbidity(df, count_col="comorbidity_count", patient_col="Patient_ID"):
    total_n = len(df)
    counts = df[count_col].value_counts().sort_index()
    percentages = counts / total_n * 100 if total_n > 0 else counts * 0
    fig, ax = plt.subplots()
    bars = ax.bar(counts.index.astype(str), percentages.values)

    ax.set_xlabel("Number of non-MSK chronic systems")
    ax.set_ylabel("Percentage of MSK patients (%)")
    ax.set_title(f"Multimorbidity among MSK patients (N={total_n})")
    _add_bar_labels(ax, bars, percentages.values, counts.values, y_offset=0.5)

    plt.tight_layout()
    return fig



def step5_plot_msk_age_groups(df):
    bins = list(range(0, 100, 5)) + [np.inf]
    labels = [f"{i}-{i+4}" for i in range(0, 95, 5)] + ["95+"]

    age_grp = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    pct = age_grp.value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    pct.plot(kind="bar", ax=ax)

    ax.set_ylim(0, 12)
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel("Percentage of MSK patients (%)")
    ax.set_title("Age distribution of MSK patients")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=8)

    plt.tight_layout()
    return fig


def step5_plot_msk_sex(df, sex_col="Sex", patient_col="Patient_ID"):
    df = df[df[sex_col].isin(["FEMALE", "MALE"])]
    total_n = len(df)
    counts = df[sex_col].value_counts().reindex(["FEMALE", "MALE"], fill_value=0)
    percentages = counts / total_n * 100 if total_n > 0 else counts * 0
    fig, ax = plt.subplots()
    bars = ax.bar(counts.index, percentages.values)
    ax.set_ylabel("Percentage of MSK patients (%)")
    ax.set_xlabel("Sex")
    ax.set_title("Sex distribution among MSK patients")
    for bar, pct in zip(bars, percentages.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{pct:.1f}%",
            ha="center",fontsize=8)
    plt.tight_layout()
    return fig


def step5_plot_msk_age_sex_patient(df, patient_col="Patient_ID", age_col="Age", sex_col="Sex"):
    bins = list(range(0, 100, 5)) + [np.inf]
    labels = [f"{i}-{i+4}" for i in range(0, 95, 5)] + ["95+"]

    d = (
        df[[patient_col, age_col, sex_col]]
        .drop_duplicates(patient_col)
        .dropna()
        .copy()
    )

    d["AgeGroup"] = pd.cut(d[age_col], bins=bins, labels=labels, right=False)
    pct = pd.crosstab(d["AgeGroup"], d[sex_col], normalize="columns") * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    pct.plot(kind="bar", ax=ax, width=0.8)

    ax.set_ylim(0, 13)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Percentage of MSK patients (%)")
    ax.set_title("Age × Sex distribution of MSK patients")
    plt.xticks(rotation=45, ha="right")

    for j, container in enumerate(ax.containers):
        x_offset = -0.08 if j % 2 == 0 else 0.08
        y_offset = 0.10 if j % 2 == 0 else 0.22

        for bar in container:
            h = bar.get_height()
            if not np.isfinite(h) or h <= 0:
                continue
            x = bar.get_x() + bar.get_width() / 2.0
            ax.text(
                x + x_offset,
                h + y_offset,
                f"{h:.2f}%",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    plt.tight_layout()
    return fig


def boxplot_msk_vs_mental_by_sex(
    df,
    msk_col="msk_duration_years",
    mental_col="mental_duration_years",
    sex_col="Sex",
    bins=(0, 1, 3, 5, 10, 15, 1e9),
    labels=("<1", "1-2", "3-4", "5-9", "10-14", "15+"),
    showfliers=False,
    figsize=(9, 5),
):
    d = df[[msk_col, mental_col, sex_col]].dropna().copy()
    d[sex_col] = d[sex_col].astype(str).str.upper()
    d["mental_bin"] = pd.cut(d[mental_col], bins=bins, labels=labels, right=False)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_yscale("linear")

    x = np.arange(len(labels))
    width = 0.35

    for i, lab in enumerate(labels):
        f = d.loc[(d["mental_bin"] == lab) & (d[sex_col] == "FEMALE"), msk_col]
        m = d.loc[(d["mental_bin"] == lab) & (d[sex_col] == "MALE"), msk_col]

        ax.boxplot(
            f,
            positions=[i - width / 2],
            widths=width,
            patch_artist=True,
            showfliers=showfliers,
            boxprops=dict(facecolor="#4C72B0"),
            medianprops=dict(color="black"),
        )

        ax.boxplot(
            m,
            positions=[i + width / 2],
            widths=width,
            patch_artist=True,
            showfliers=showfliers,
            boxprops=dict(facecolor="#DD8452"),
            medianprops=dict(color="black"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Mental duration (years)")
    ax.set_ylabel("MSK duration (years)")
    ax.set_title("MSK duration vs Mental duration (by sex)")
    ax.legend(
        handles=[
            Patch(facecolor="#4C72B0", label="Female"),
            Patch(facecolor="#DD8452", label="Male"),
        ],
        title="Sex",
        loc="best",
    )

    plt.tight_layout()
    return fig


def barplot_msk_type_percentage(df, type_col="msk_type", patient_col="Patient_ID"):
    df = df[df[type_col].notna()]
    total_n = len(df)
    counts = df[type_col].value_counts().sort_index()
    percentages = counts / total_n * 100 if total_n > 0 else counts * 0
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, percentages.values)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("MSK type")
    ax.set_title(f"Distribution of MSK types (N={total_n})")
    ax.set_ylim(0, max(percentages.max() + 8, 10) if len(percentages) > 0 else 10)
    plt.xticks(rotation=30, ha="right")
    _add_bar_labels(ax, bars, percentages.values, counts.values, y_offset=0.8)
    plt.tight_layout()
    return fig

def plot_msk_type_by_sex_percent(
    df,
    type_col="msk_type",
    sex_col="Sex",
    order=("co_occurrence", "msk_comorbidity", "msk_only", "other_pre_existing_then_msk"),
    figsize=(9, 5),
    rotate=30,
):
    d = df[[type_col, sex_col]].dropna().copy()
    d[sex_col] = d[sex_col].astype(str).str.upper()

    tab = d.groupby([type_col, sex_col]).size().unstack(fill_value=0)

    for c in ["FEMALE", "MALE"]:
        if c not in tab.columns:
            tab[c] = 0

    tab = tab[["FEMALE", "MALE"]]

    order = [x for x in order if x in tab.index]
    tab = tab.loc[order]

    pct = tab.div(tab.sum(axis=0), axis=1) * 100

    x = np.arange(len(pct))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    b1 = ax.bar(x - width / 2, pct["FEMALE"], width, label="FEMALE")
    b2 = ax.bar(x + width / 2, pct["MALE"], width, label="MALE")

    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 1,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pct.index, rotation=rotate, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage within sex (%)")
    ax.set_xlabel("MSK type")
    ax.set_title("Distribution of MSK types by Sex (%)")
    ax.legend(title="Sex")

    plt.tight_layout()
    return fig