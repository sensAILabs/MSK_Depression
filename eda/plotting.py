import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


from eda.clinical_definitions import ICD9_RANGES

#EDA plots
def step5_plot_msk_comorbidities(df, patient_col="Patient_ID"):
    types = ["cancer", "endocrine", "mental", "cardiovascular", "respiratory", "genitourinary"]
    pt = df.drop_duplicates(patient_col).copy()
    n = len(pt)
    prevalence = {t: pt["type list"].str.split(",").apply(lambda x: t in x).sum() / n * 100
                 for t in types}
    s = pd.Series(prevalence).sort_values(ascending=False)
    ax = s.plot(kind="bar")
    ax.set_ylabel("Percentage of MSK patients (%)")
    ax.set_title("Comorbid chronic diseases in MSK patients")
    for i, v in enumerate(s):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    return plt.gcf()



def step5_plot_msk_multimorbidity(df, count_col="comorbidity_count"):
    counts = df[count_col].value_counts().sort_index()
    perc = counts / counts.sum() * 100
    ax = perc.plot(kind="bar")
    plt.xlabel("Number of non-MSK chronic systems")
    plt.ylabel("Percentage of MSK patients (%)")
    plt.title("Multimorbidity among MSK patients")
    for i, v in enumerate(perc.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    return plt.gcf()


def step5_plot_msk_age_groups(df):
    bins = list(range(0, 100, 5)) + [np.inf]
    labels = [f"{i}-{i+4}" for i in range(0, 95, 5)] + ["95+"]
    age_grp = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    pct = age_grp.value_counts(normalize=True).sort_index() * 100
    ax = pct.plot(kind="bar", figsize=(12,6))
    ax.set_ylim(0, 12)
    plt.xlabel("Age group (years)")
    plt.ylabel("Percentage of MSK patients (%)")
    plt.title("Age distribution of MSK patients")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=8)
    plt.tight_layout()
    return plt.gcf()


def step5_plot_msk_sex(df):
    counts = df["Sex"].value_counts()
    percentages = counts / len(df) * 100
    ax = counts.plot(kind="bar")
    plt.ylabel("Percentage of MSK patients (%)")
    plt.title("Sex distribution of MSK patients")
    # Add percentage labels
    for i, (n, p) in enumerate(zip(counts, percentages)):
        ax.text(i, n + n*0.02, f"{p:.1f}%", ha="center")
    plt.tight_layout()
    return plt.gcf()


def step5_plot_msk_age_sex_patient(df, patient_col="Patient_ID", age_col="Age", sex_col="Sex"):
    bins = list(range(0, 100, 5)) + [np.inf]
    labels = [f"{i}-{i+4}" for i in range(0, 95, 5)] + ["95+"]
    # patient level: one row per patient
    d = ( df[[patient_col, age_col, sex_col]]
        .drop_duplicates(patient_col)
        .dropna()
        .copy())
    d["AgeGroup"] = pd.cut(d[age_col], bins=bins, labels=labels, right=False)
    pct = pd.crosstab(d["AgeGroup"], d[sex_col], normalize="columns") * 100
    ax = pct.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_ylim(0, 13)
    plt.xlabel("Age group")
    plt.ylabel("Percentage of MSK patients (%)")
    plt.title("Age × Sex distribution of MSK patients")
    plt.xticks(rotation=45, ha="right")
    for j, container in enumerate(ax.containers):
        x_offset = -0.08 if j % 2 == 0 else 0.08     
        y_offset = 0.10 if j % 2 == 0 else 0.22      
        for bar in container:
            h = bar.get_height()
            if not np.isfinite(h) or h <= 0:
                continue
            x = bar.get_x() + bar.get_width() / 2.0
            ax.text(x + x_offset,h + y_offset,f"{h:.2f}%",ha="center",va="bottom",fontsize=6)
    plt.tight_layout()
    return plt.gcf()


def boxplot_msk_vs_mental_by_sex(df, msk_col="msk_duration_years", mental_col="mental_duration_years",
                                 sex_col="Sex",bins=(0, 1, 3, 5, 10, 15, 1e9),
                                 labels=("<1", "1-2", "3-4", "5-9", "10-14", "15+"),showfliers=False,figsize=(9, 5)):
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
        ax.boxplot(f, positions=[i - width/2], widths=width, patch_artist=True,
                   showfliers=showfliers,
                   boxprops=dict(facecolor="#4C72B0"), medianprops=dict(color="black"))
        ax.boxplot(m, positions=[i + width/2], widths=width, patch_artist=True,
                   showfliers=showfliers,
                   boxprops=dict(facecolor="#DD8452"), medianprops=dict(color="black"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Mental duration (years)")
    ax.set_ylabel("MSK duration (years)")
    ax.set_title("MSK duration vs Mental duration (by sex)")
    ax.legend(handles=[Patch(facecolor="#4C72B0", label="Female"),Patch(facecolor="#DD8452", label="Male")],
        title="Sex", loc="best")
    plt.tight_layout()
    return plt.gcf()


def barplot_msk_type_percentage(df, type_col="msk_type"):
    d = df[type_col].dropna()
    pct = d.value_counts(normalize=True).sort_index() * 100
    plt.figure(figsize=(5, 4))
    bars = plt.bar(pct.index, pct.values)
    plt.ylabel("Percentage (%)")
    plt.xlabel("MSK type")
    plt.title("Distribution of MSK types (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(pct.values):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    return plt.gcf()


def plot_msk_type_by_sex_percent(df,type_col="msk_type",sex_col="Sex",
                                 order=("co_occurrence", "msk_comorbidity", "msk_only", "other_pre_existing_then_msk"),
                                 figsize=(9, 5),rotate=30):
    d = df[[type_col, sex_col]].dropna().copy()
    d[sex_col] = d[sex_col].astype(str).str.upper()
    # count
    tab = d.groupby([type_col, sex_col]).size().unstack(fill_value=0)
    # ensure columns
    for c in ["FEMALE", "MALE"]:
        if c not in tab.columns:
            tab[c] = 0
    tab = tab[["FEMALE", "MALE"]]
    # order MSK types
    order = [x for x in order if x in tab.index]
    tab = tab.loc[order]
    #  percent within sex (column-wise)
    pct = tab.div(tab.sum(axis=0), axis=1) * 100
    x = np.arange(len(pct))
    width = 0.35
    plt.figure(figsize=figsize)
    b1 = plt.bar(x - width/2, pct["FEMALE"], width, label="FEMALE")
    b2 = plt.bar(x + width/2, pct["MALE"],   width, label="MALE")
    # add percentage labels
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width()/2,h + 1,f"{h:.1f}%",
                ha="center",va="bottom",fontsize=10)
    plt.xticks(x, pct.index, rotation=rotate, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Percentage within sex (%)")
    plt.xlabel("MSK type")
    plt.title("Distribution of MSK types by Sex (%)")
    plt.legend(title="Sex")
    plt.tight_layout()
    return plt.gcf()