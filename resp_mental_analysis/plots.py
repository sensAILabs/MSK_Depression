from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from eda.clinical_definitions import RESP_SUBGROUPS, MENTAL_SUBGROUPS

# Step 1 plots
def fig_groups_before_msk(
    group_summary: pd.DataFrame,
    fig_path: Path,
    cohort_label: str = "All ages"
) -> None:
    groups = group_summary["group"].tolist()
    perc = group_summary["percent"].to_numpy()
    xs = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(xs, perc)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("Percent (%)")
    ax.set_title(
        f"Distribution of Respiratory/Mental groups before first MSK ({cohort_label})"
    )

    ymax = float(np.nanmax(perc)) if len(perc) else 1.0
    for x, y in zip(xs, perc):
        ax.text(x, y + 0.02 * ymax, f"{y:.1f}%", ha="center", va="bottom")

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ordered_group_labels(df_msk: pd.DataFrame) -> list[str]:
    preferred = ["Resp only", "Mental only", "Both"]
    present = [g for g in preferred if g in df_msk["group"].dropna().astype(str).unique().tolist()]
    return present


def fig_sex_percent_by_group(
    df_msk: pd.DataFrame,
    fig_path: Path,
    cohort_label: str = "All ages"
) -> None:
    grp_order = _ordered_group_labels(df_msk)
    perc = []

    for g in grp_order:
        d = df_msk[df_msk["group"] == g]
        pct = (d["Sex"] == "F").mean() * 100
        perc.append(pct)

    xs = np.arange(len(grp_order))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(xs, perc)
    ax.set_xticks(xs)
    ax.set_xticklabels(grp_order, rotation=20, ha="right")
    ax.set_ylabel("Female (%)")
    ax.set_title(
        f"Female proportion by Respiratory/Mental group ({cohort_label})"
    )

    ymax = float(np.nanmax(perc)) if len(perc) else 1.0
    for x, y in zip(xs, perc):
        ax.text(x, y + 0.02 * ymax, f"{y:.1f}%", ha="center", va="bottom")

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_order_among_both(
    order_summary: pd.DataFrame,
    fig_path: Path,
    cohort_label: str = "All ages"
) -> None:
    order = order_summary["order"].tolist()
    perc = order_summary["percent"].to_numpy()
    xs = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(xs, perc)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Percent (%) among BOTH")
    ax.set_title(
        f"Temporal order among BOTH before first MSK ({cohort_label})"
    )

    ymax = float(np.nanmax(perc)) if len(perc) else 1.0
    for x, y in zip(xs, perc):
        ax.text(x, y + 0.02 * ymax, f"{y:.1f}%", ha="center", va="bottom")

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def fig_first_msk_subgroup_topk(
    counts: pd.Series,
    fig_path: Path,
    title: str,
    top_k: int = 15,
) -> None:
    """
    Plot first MSK diagnosis subgroup distribution.
    Keep top_k groups and combine the rest into 'Other'.
    """
    s = counts.copy()
    s = s[s.notna()]
    s = s[s > 0].sort_values(ascending=False)

    if len(s) == 0:
        return

    if len(s) > top_k:
        top = s.iloc[:top_k].copy()
        other_n = int(s.iloc[top_k:].sum())
        if other_n > 0:
            top.loc["Other"] = other_n
        s = top

    total = int(s.sum())
    pct = s / total * 100

    labels = s.index.tolist()
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, max(6, 0.45 * len(labels) + 1.5)))
    ax.barh(y, s.values)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title)

    xmax = s.max() if len(s) else 1
    ax.set_xlim(0, xmax * 1.22)

    for i, (n, p) in enumerate(zip(s.values, pct.values)):
        pct_text = f"{p:.2f}%" if p < 1 else f"{p:.2f}%"
        ax.text(
            n + xmax * 0.02,
            i,
            f"n={int(n):,} ({pct_text})",
            va="center",
            fontsize=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    

def fig_overlap_four_groups(
    group_summary: pd.DataFrame,
    fig_path: Path,
    cohort_label: str = "All ages"
) -> None:
    """
    Plot overlap of respiratory / mental groups within the pre-existing then MSK cohort.
    Expected groups: Respiratory only, Mental only, Both, None
    """
    label_map = {
        "Resp only": "Respiratory only",
        "Mental only": "Mental only",
        "Both": "Both",
        "None": "Neither",
    }

    tmp = group_summary.copy()
    tmp["plot_label"] = tmp["group"].map(label_map).fillna(tmp["group"])

    groups = tmp["plot_label"].tolist()
    perc = pd.to_numeric(tmp["percent"], errors="coerce").fillna(0).to_numpy()
    ns = pd.to_numeric(tmp["N"], errors="coerce").fillna(0).astype(int).to_numpy()
    total_n = int(ns.sum())

    xs = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(xs, perc)

    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=90)
    ax.set_ylabel("Percentage of patients (%)")
    ax.set_title("Respiratory and Mental diagnoses overlap")

    ymax = float(np.nanmax(perc)) if len(perc) else 1.0
    for x, y, n in zip(xs, perc, ns):
        ax.text(
            x, y + 0.5,
            f"{y:.1f}%\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=11
        )

    fig.suptitle(f"Pre-existing then MSK cohort (n = {total_n})", fontsize=18, y=0.98)

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _range_label(name: str, code_set: set[int]) -> str:
    codes = sorted(code_set)
    if len(codes) == 1:
        code_text = f"{codes[0]}"
    elif codes == list(range(codes[0], codes[-1] + 1)):
        code_text = f"{codes[0]}–{codes[-1]}"
    else:
        code_text = "/".join(map(str, codes))
    return f"{code_text}\n{name.replace('_', ' ')}"

def fig_acute_chronic_composition(count_table, fig_path):

    pct = count_table.div(count_table.sum(axis=1), axis=0) * 100

    labels = count_table.index.tolist()

    acute = pct.get("acute", pd.Series([0]*len(labels)))
    chronic = pct.get("chronic", pd.Series([0]*len(labels)))
    unknown = pct.get("unknown", pd.Series([0]*len(labels)))

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8,6))

    ax.bar(x, acute, label="acute")
    ax.bar(x, chronic, bottom=acute, label="chronic")
    ax.bar(x, unknown, bottom=acute+chronic, label="unknown")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Acute vs Chronic Composition in Respiratory and Mental Conditions")

    ax.legend()

    for i, lab in enumerate(labels):

        total = count_table.loc[lab].sum()

        for cat, bottom in zip(
            ["acute", "chronic", "unknown"],
            [0, acute.iloc[i], acute.iloc[i] + chronic.iloc[i]]
        ):
            if cat in count_table.columns:

                val = count_table.loc[lab, cat]
                pct_val = pct.loc[lab, cat]

                ax.text(
                    i,
                    bottom + pct_val/2,
                    f"{pct_val:.1f}%\n(n={val})",
                    ha="center",
                    va="center",
                    fontsize=10
                )

        ax.text(
            i,
            102,
            f"Total n={total}",
            ha="center",
            fontsize=11
        )

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close()
   
def fig_subgroup_bar(
    counts: pd.Series,
    fig_path: Path,
    title: str,
    code_map: dict[str, set[int]] | None = None,
    ylabel: str = "Count",
    rotate: int = 45,
    top_k: int | None = None,
) -> None:
    """
    Generic subgroup bar plot with percent labels.
    """
    s = counts.copy()
    s = s[s.notna()]
    s = s[s.index.notna()]
    s = s.sort_values(ascending=False)

    if top_k is not None:
        s = s.head(top_k).copy()

    if code_map is not None:
        new_index = []
        for x in s.index:
            if x in code_map:
                new_index.append(_range_label(x, code_map[x]))
            else:
                new_index.append(str(x))
        s.index = new_index

    total = int(s.sum()) if len(s) else 0

    fig, ax = plt.subplots(figsize=(16, 7))
    s.plot(kind="bar", ax=ax)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=55, ha="right")

    for p, n in zip(ax.patches, s.values):
        pct = (n / total * 100) if total > 0 else 0
        pct_text = f"{pct:.2f}%" if pct < 1 else f"{pct:.1f}%"
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height(),
            f"n={int(n)}\n({pct_text})",
            ha="center",
            va="bottom",
            fontsize=9
        )

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def fig_course_pair_subgroup_composition(
    acute_counts: pd.Series,
    chronic_counts: pd.Series,
    fig_path: Path,
    title: str,
    acute_title: str,
    chronic_title: str,
    code_map: dict[str, set[int]] | None = None,
    rotate: int = 50,
) -> None:
    """
    Two side-by-side bar charts:
    left = acute subgroup composition
    right = chronic subgroup composition
    with the same y-axis scale.
    """
    acute_counts = acute_counts.copy()
    chronic_counts = chronic_counts.copy()

    def relabel_index(s: pd.Series) -> pd.Series:
        if code_map is None:
            return s
        new_index = []
        for x in s.index:
            if x in code_map:
                new_index.append(_range_label(x, code_map[x]))
            else:
                new_index.append(str(x))
        s.index = new_index
        return s

    acute_counts = relabel_index(acute_counts)
    chronic_counts = relabel_index(chronic_counts)

    acute_counts = acute_counts.sort_values(ascending=False)
    chronic_counts = chronic_counts.sort_values(ascending=False)

    ymax = max(
        acute_counts.max() if len(acute_counts) else 0,
        chronic_counts.max() if len(chronic_counts) else 0,
        1
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(title, fontsize=20, y=1.02)

    # acute
    ax = axes[0]
    bars = ax.bar(range(len(acute_counts)), acute_counts.values)
    ax.set_title(acute_title, fontsize=11)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(acute_counts)))
    ax.set_xticklabels(acute_counts.index, rotation=rotate, ha="right", fontsize=8)
    ax.set_ylim(0, ymax * 1.15)

    acute_total = acute_counts.sum()
    for i, (label, value) in enumerate(acute_counts.items()):
        pct = value / acute_total * 100 if acute_total > 0 else 0
        ax.text(
            i,
            value + ymax * 0.01,
            f"n={value}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=7
        )

    # chronic
    ax = axes[1]
    bars = ax.bar(range(len(chronic_counts)), chronic_counts.values)
    ax.set_title(chronic_title, fontsize=11)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(chronic_counts)))
    ax.set_xticklabels(chronic_counts.index, rotation=rotate, ha="right", fontsize=8)
    ax.set_ylim(0, ymax * 1.15)

    chronic_total = chronic_counts.sum()
    for i, (label, value) in enumerate(chronic_counts.items()):
        pct = value / chronic_total * 100 if chronic_total > 0 else 0
        ax.text(
            i,
            value + ymax * 0.01,
            f"n={value}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=7
        )

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def fig_subgroup_by_course_stacked(
    rm_long: pd.DataFrame,
    system_name: str,
    fig_path: str | Path,
    title: str,
    code_map: dict[str, set[int]] | None = None,
    course_order: list[str] | None = None,
    top_k: int | None = None,
    normalize: bool = False,
) -> None:
    d = rm_long[rm_long["system"] == system_name].copy()
    d = d.dropna(subset=["subgroup", "acute_chronic"]).copy()

    if d.empty:
        print(f"[WARN] No data found for system={system_name}. Skip figure: {fig_path}")
        return

    if course_order is None:
        course_order = ["acute", "chronic", "unknown"]

    # subgroup x course count table
    tab = pd.crosstab(d["subgroup"], d["acute_chronic"])

    # ensure all course columns exist
    for c in course_order:
        if c not in tab.columns:
            tab[c] = 0

    tab = tab[course_order].copy()
    tab["total"] = tab.sum(axis=1)
    tab = tab.sort_values("total", ascending=False)

    if top_k is not None:
        tab = tab.head(top_k).copy()

    plot_df = tab.drop(columns="total").copy()

    if normalize:
        plot_df = plot_df.div(plot_df.sum(axis=1).replace(0, np.nan), axis=0) * 100
        ylabel = "Percentage within subgroup"
    else:
        ylabel = "Count"

    if code_map is not None:
        x_labels = []
        for idx in plot_df.index:
            if idx in code_map:
                x_labels.append(_range_label(idx, code_map[idx]))
            else:
                x_labels.append(str(idx).replace("_", " "))
    else:
        x_labels = [idx.replace("_", " ") for idx in plot_df.index]

    fig, ax = plt.subplots(figsize=(16, 7))

    bottom = np.zeros(len(plot_df))
    color_map = {
        "acute": "#d95f02",
        "chronic": "#1b9e77",
        "unknown": "#7570b3",
    }

    for course in course_order:
        vals = plot_df[course].fillna(0).values
        ax.bar(
            range(len(plot_df)),
            vals,
            bottom=bottom,
            label=course.capitalize(),
            color=color_map.get(course, None),
        )
        bottom += vals

    # total labels on top
    totals = tab["total"].values
    grand_total = totals.sum()

    if not normalize:
        for i, total in enumerate(totals):
            pct = total / grand_total * 100 if grand_total > 0 else 0
            ax.text(
                i,
                total,
                f"n={int(total)}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(x_labels, rotation=55, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Course")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def fig_first_msk_subgroup_by_course_stacked(
    df: pd.DataFrame,
    fig_path: Path,
    title: str,
    code_map: dict[str, set[int]] | None = None,
    course_order: list[str] | None = None,
    top_k: int | None = None,
    normalize: bool = False,
) -> None:
    d = df.copy()
    d = d.dropna(subset=["first_msk_subgroup", "first_msk_course"]).copy()

    if d.empty:
        print(f"[WARN] No first MSK subgroup/course data found. Skip figure: {fig_path}")
        return

    if course_order is None:
        course_order = ["acute", "chronic", "unknown"]

    tab = pd.crosstab(d["first_msk_subgroup"], d["first_msk_course"])

    for c in course_order:
        if c not in tab.columns:
            tab[c] = 0

    tab = tab[course_order].copy()
    tab["total"] = tab.sum(axis=1)
    tab = tab.sort_values("total", ascending=False)

    if top_k is not None:
        tab = tab.head(top_k).copy()

    plot_df = tab.drop(columns="total").copy()

    if normalize:
        plot_df = plot_df.div(plot_df.sum(axis=1).replace(0, np.nan), axis=0) * 100
        ylabel = "Percentage within subgroup"
    else:
        ylabel = "Count"

    if code_map is not None:
        x_labels = []
        for idx in plot_df.index:
            if idx in code_map:
                x_labels.append(_range_label(idx, code_map[idx]))
            else:
                x_labels.append(str(idx).replace("_", " "))
    else:
        x_labels = [idx.replace("_", " ") for idx in plot_df.index]

    fig, ax = plt.subplots(figsize=(16, 7))

    bottom = np.zeros(len(plot_df))
    color_map = {
        "acute": "#d95f02",
        "chronic": "#1b9e77",
        "unknown": "#7570b3",
    }

    for course in course_order:
        vals = plot_df[course].fillna(0).values
        ax.bar(
            range(len(plot_df)),
            vals,
            bottom=bottom,
            label=course.capitalize(),
            color=color_map.get(course, None),
        )
        bottom += vals

    totals = tab["total"].values
    grand_total = totals.sum()

    if not normalize:
        for i, total in enumerate(totals):
            pct = total / grand_total * 100 if grand_total > 0 else 0
            ax.text(
                i,
                total,
                f"n={int(total)}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(x_labels, rotation=55, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Course")

    plt.tight_layout()
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()