
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import textwrap
from matplotlib import gridspec
from eda.clinical_definitions import ICD9_RANGES, CONDITION_DRUG_KEYWORDS, SYS_ABBR


def step5_plot_msk_multimorbidity(
    df: pd.DataFrame,
    count_col: str = "comorbidity_count",
    save_path=None
):
    counts = df[count_col].value_counts().sort_index()
    perc = counts / counts.sum() * 100
    n_total = counts.sum()

    fig, ax = plt.subplots()
    perc.plot(kind="bar", ax=ax)

    ax.set_xlabel("Number of non-MSK chronic systems")
    ax.set_ylabel("Percentage of MSK patients (%)")
    ax.set_title("Multimorbidity among MSK patients")

    fig.suptitle(
        f"Total MSK patients (n = {n_total})",
        fontsize=12,
        y=0.98
    )

    for i, (v, n) in enumerate(zip(perc.values, counts.values)):
        ax.text(i, v + 0.5, f"{v:.1f}% (n={n:,})", ha="center", fontsize=6)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return perc.reset_index().rename(columns={"index": count_col, count_col: "percent"})


def plot_top_msk_icd9(
    df: pd.DataFrame,
    icd_col: str = "icd9 list (named)",
    pid_col: str = "Patient_ID",
    top_k: int = 15,
    save_path=None
):
    total = df[pid_col].nunique()

    # build 3-digit name_map from "(code) (name)" pairs in the named column
    name_map = {}
    for cell in df[icd_col].dropna():
        text = " ".join(cell) if isinstance(cell, (list, tuple, set)) else str(cell)
        matches = re.findall(r"(\d{3})\s*\((.*?)\)", text)
        for code, name in matches:
            try:
                if 710 <= int(code) <= 739:
                    name_map[code] = name
            except ValueError:
                continue

    def extract_codes(cell):
        if cell is None or (isinstance(cell, float) and pd.isna(cell)):
            return set()
        text = " ".join(cell) if isinstance(cell, (list, tuple, set)) else str(cell)
        codes = set(re.findall(r"\b(\d{3})\b", text))
        out = set()
        for c in codes:
            try:
                if 710 <= int(c) <= 739:
                    out.add(c)
            except ValueError:
                continue
        return out

    tmp = df[[pid_col, icd_col]].copy()
    tmp["msk_codes"] = tmp[icd_col].apply(extract_codes)

    msk = tmp.explode("msk_codes").dropna(subset=["msk_codes"])

    tab = (
        msk.groupby("msk_codes")[pid_col]
        .nunique()
        .sort_values(ascending=False)
        .head(top_k)
        .reset_index(name="n")
    )

    tab["pct"] = tab["n"] / total * 100
    tab = tab.sort_values("pct")
    tab["label"] = tab["msk_codes"].apply(lambda c: f"{c}-{name_map.get(c,'')}")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(tab["label"], tab["pct"])
    ax.set_xlabel("Percent of MSK patients (%)")
    ax.set_title(
        f"Top MSK ICD-9 (3-digit) among MSK patients\nTotal MSK patients (N = {total:,})"
    )

    xpad = tab["pct"].max() * 0.01 if len(tab) else 0.01
    for y, (p, n) in enumerate(zip(tab["pct"], tab["n"])):
        ax.text(p + xpad, y, f"{p:.2f}% (n={n:,})", va="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return tab

    
def plot_topk_nonmsk_combos_all_counts(
    df,
    type_list_col="type list",
    msk_token="msk",
    top_k=6,
    counts_to_plot=(1,2,3,4,5),
    save_dir=None
):
    # build non-MSK combo + count
    def to_other_combo(s):
        if pd.isna(s) or str(s).strip() == "":
            return ""
        parts = [x.strip() for x in str(s).split(",") if x.strip() != ""]
        other = [x for x in parts if x.lower() != msk_token]
        other = sorted(set(other))
        return ",".join(other)

    def other_count(combo):
        if combo == "":
            return 0
        return len([x for x in combo.split(",") if x.strip() != ""])

    work = df.copy()
    work["other_combo"] = work[type_list_col].apply(to_other_combo)
    work["other_comorbidity_count"] = work["other_combo"].apply(other_count)

    counts = (
        work.groupby(["other_comorbidity_count", "other_combo"])
            .size()
            .reset_index(name="n")
    )
    totals = (
        work.groupby("other_comorbidity_count")
            .size()
            .reset_index(name="N")
    )
    counts = counts.merge(totals, on="other_comorbidity_count", how="left")
    counts["pct"] = counts["n"] / counts["N"] * 100

    topk = (
        counts.sort_values(["other_comorbidity_count", "n"], ascending=[True, False])
              .groupby("other_comorbidity_count")
              .head(top_k)
              .copy()
    )

    def plot_one(sub, c):
        sub = sub.sort_values("pct", ascending=True).reset_index(drop=True)
        labels = []
        for combo in sub["other_combo"]:
            combo = "None" if combo == "" else combo
            labels.append("\n".join(textwrap.wrap(combo, width=34)))

        y = np.arange(len(sub))

        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax.barh(y, sub["pct"].to_numpy())
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Percentage within count group (%)", fontsize=11)
        ax.set_title(f"Top {top_k} non-MSK comorbidity combinations (count={c})", fontsize=12)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

        xmax = sub["pct"].max() if len(sub) else 1
        ax.set_xlim(0, xmax * 1.18)

        for i, row in sub.iterrows():
            ax.text(
                row["pct"] + xmax * 0.02,
                i,
                f"{row['pct']:.2f}% (n={int(row['n'])})",
                va="center",
                fontsize=10
            )

        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"top{top_k}_nonMSK_count{c}.png"),
                        dpi=300, bbox_inches="tight")
        plt.show()

    for c in counts_to_plot:
        sub = topk[topk["other_comorbidity_count"] == c].copy()
        if sub.empty:
            continue
        plot_one(sub, c)

def split_csv(x):
    if pd.isna(x) or x is None:
        return []
    return [t.strip() for t in str(x).split(",") if t.strip()]

def icd3_int(x):
    try:
        return int(str(x).strip()[:3])
    except:
        return None

def in_ranges(code, ranges):
    v = icd3_int(code)
    if v is None:
        return False
    if isinstance(ranges, tuple):
        lo, hi = ranges
        return lo <= v <= hi
    return any(lo <= v <= hi for lo, hi in ranges)


def pick_main_icd3(icd_list, system):
    codes = [c for c in icd_list if in_ranges(c, ICD9_RANGES[system])]
    if not codes:
        return None
    icd3s = []
    for c in codes:
        v = icd3_int(c)
        if v is not None:
            icd3s.append(f"{v:03d}")
    if not icd3s:
        return None
    vc = pd.Series(icd3s).value_counts()
    top = vc[vc == vc.max()].index.tolist()
    return sorted(top)[0]

def build_icd3_name_map(df, named_col="icd9 list (named)"):
    pat = re.compile(r"^\s*(\d{3})(?:\.\d+)?\s*\(([^)]+)\)")
    m = {}
    for row in df[named_col].dropna():
        for part in str(row).split(","):
            mm = pat.match(part.strip())
            if mm:
                m[mm.group(1)] = mm.group(2).strip()
    return m

def prep(df, type_col="type list", icd_col="icd9 list", id_col="Patient_ID", msk_token="msk"):
    out = df.copy()
    out["type_list"] = out[type_col].apply(split_csv)
    out["icd3_list"] = out[icd_col].apply(split_csv)
    out["non_msk_types"] = out["type_list"].apply(lambda L: sorted([t for t in L if t != msk_token]))
    out["non_msk_count"] = out["non_msk_types"].str.len()
    out["combo_system"] = out["non_msk_types"].apply(",".join)
    return out
def top_combos(df_prepped, n, top_k=3):
    sub = df_prepped[df_prepped["non_msk_count"] == n]
    vc = sub["combo_system"].value_counts().head(top_k)
    out = vc.rename_axis("combo_system").reset_index(name="n_people")
    out["percent_within_n"] = out["n_people"] / len(sub) * 100 if len(sub) else 0
    out["n_total"] = len(sub)
    return out
def top_tuples(df_prepped, n, combo_system, top_k=5, id_col="Patient_ID"):
    systems = combo_system.split(",") if combo_system else []
    sub = df_prepped[(df_prepped["non_msk_count"] == n) & (df_prepped["combo_system"] == combo_system)].copy()
    if sub.empty or not systems:
        return pd.DataFrame()

    for s in systems:
        sub[f"{s}_icd3"] = sub["icd3_list"].apply(lambda L, ss=s: pick_main_icd3(L, ss))

    cols = [f"{s}_icd3" for s in systems]
    sub = sub.dropna(subset=cols)[[id_col] + cols].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["icd_tuple"] = sub[cols].apply(lambda r: tuple(r.tolist()), axis=1)

    out = (sub.groupby("icd_tuple")[id_col]
             .nunique()
             .sort_values(ascending=False)
             .head(top_k)
             .reset_index(name="n_people"))
    out["percent_within_combo"] = out["n_people"] / sub[id_col].nunique() * 100
    out["combo_total_people"] = sub[id_col].nunique()
    out["systems"] = ",".join(systems)
    return out

def label_code_only(icd_tuple, systems):
    systems = systems.split(",") if isinstance(systems, str) else list(systems)

    def c3(x):
        m = re.match(r"^(\d{3})", str(x).strip())
        return m.group(1) if m else str(x).strip()[:3].zfill(3)

    return " | ".join([f"{SYS_ABBR.get(s, s[:1].upper())}{c3(c)}" for s, c in zip(systems, icd_tuple)])

def label_code_name(icd_tuple, systems, name_map, max_len=28):
    systems = systems.split(",") if isinstance(systems, str) else list(systems)
    parts = []
    for s, c in zip(systems, icd_tuple):
        code = re.match(r"^(\d{3})", str(c).strip()).group(1)
        nm = name_map.get(code, "")
        if nm and len(nm) > max_len:
            nm = nm[:max_len-1] + "…"
        ab = SYS_ABBR.get(s, s[:1].upper())
        parts.append(f"{ab}{code} ({nm})" if nm else f"{ab}{code}")
    return " | ".join(parts)

def make_name_table_for_codes(codes, name_map):
    df = pd.DataFrame({"icd3": sorted(set(codes))})
    df["name"] = df["icd3"].map(name_map)
    return df

def plot_n(df_prepped,name_map,n,top_combo_k=3,top_tuple_k=5,
    id_col="Patient_ID",save_path=None, return_name_table=True   
):
    combos = top_combos(df_prepped, n, top_k=top_combo_k)
    if combos.empty:
        print(f"No data for n={n}")
        return None
    nrows = len(combos)
    fig = plt.figure(figsize=(18 if n >= 4 else 14, max(3.8, 3.2 * nrows)))

    # left: plots, right: table
    if n >= 4:
        gs = gridspec.GridSpec(
            nrows, 2, figure=fig,
            width_ratios=[3.6, 1.4], wspace=0.15, hspace=0.35
        )
    else:
        gs = gridspec.GridSpec(nrows, 1, figure=fig, hspace=0.35)

    used_codes = []
    for i, row in enumerate(combos.itertuples(index=False)):
        ax = fig.add_subplot(gs[i, 0] if n >= 4 else gs[i, 0])

        bd = top_tuples(df_prepped, n, row.combo_system, top_k=top_tuple_k, id_col=id_col)
        if bd.empty:
            ax.axis("off")
            continue

        systems = bd["systems"].iloc[0]

        if n <= 3:
            labels = [label_code_name(t, systems, name_map, max_len=28) for t in bd["icd_tuple"]]
        else:
            labels = [label_code_only(t, systems) for t in bd["icd_tuple"]]
            for t in bd["icd_tuple"]:
                for c in t:
                    m = re.match(r"^(\d{3})", str(c).strip())
                    if m:
                        used_codes.append(m.group(1))

        vals = bd["percent_within_combo"].values
        y = range(len(labels))

        ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("% within this combo subgroup")

        ax.set_title(
            f"n={n} | combo={row.combo_system} (within-n: {row.n_people} ppl, {row.percent_within_n:.2f}%)",
            fontsize=11
        )

        vmax = max(vals) if len(vals) else 1
        for yi, (v, nppl) in enumerate(zip(vals, bd["n_people"].values)):
            ax.text(v + vmax * 0.02, yi, f"n={int(nppl)}", va="center", fontsize=9)

    name_df = None

    # right table (only when n>=4)
    if n >= 4 and used_codes:
        used_codes = sorted(set(used_codes))
        name_df = pd.DataFrame({
            "ICD9": used_codes,
            "Name": [name_map.get(c, "") for c in used_codes]
        })

        wrap_w = 26
        name_df["Name"] = (
            name_df["Name"].fillna("").astype(str)
                  .apply(lambda s: "\n".join(textwrap.wrap(s, width=wrap_w)))
        )

        table_ax = fig.add_subplot(gs[:, 1])
        table_ax.axis("off")

        table = table_ax.table(
            cellText=name_df.values,
            colLabels=name_df.columns,
            cellLoc="left",
            colLoc="center",
            loc="upper center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(6)

        n_table_rows = len(name_df) + 1
        for r in range(n_table_rows):
            table[(r, 0)].set_width(0.28)
            table[(r, 1)].set_width(0.72)

        for c in range(2):
            table[(0, c)].set_text_props(weight="bold")

        max_lines = max(1, name_df["Name"].apply(lambda s: s.count("\n") + 1).max())
        table.scale(1.0, 1.15 + 0.15 * (max_lines - 1))

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

