from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2_contingency

from resp_mental_analysis.plots import (
    fig_groups_before_msk,
    fig_order_among_both,
    fig_subgroup_bar,
    fig_first_msk_subgroup_topk,
    fig_acute_chronic_composition,
    fig_subgroup_by_course_stacked,
    fig_first_msk_subgroup_by_course_stacked,
)

from eda.clinical_definitions import (
    _icd9_3_int,
    classify_system,
    classify_acute_chronic,
    classify_subgroup,
    RESP_SUBGROUPS,
    MENTAL_SUBGROUPS,
    MSK_SUBGROUPS,
)


def _normalize_sex(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.upper()
    x = x.replace({"FEMALE": "F", "MALE": "M"})
    x = x.where(x.isin(["F", "M"]))
    return x


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    pre_long_csv: Path
    pre_pt_csv: Path
    patient_csv: Path
    encounter_csv: Path
    out_fig_dir: Path
    out_tab_dir: Path


def default_paths() -> PipelinePaths:
    root = Path(__file__).resolve().parents[1]
    eda_tables = root / "eda_analysis" / "results" / "tables"
    data_out = root / "data" / "output"
    results_dir = root / "resp_mental_analysis" / "results"

    return PipelinePaths(
        root=root,
        pre_long_csv=eda_tables / "analysis_long_pre_msk_only.csv",
        pre_pt_csv=eda_tables / "analysis_df2_pre_msk_named.csv",
        patient_csv=root / "data" / "C4MPatient.csv",
        encounter_csv=data_out / "C4MEncounterDiagnosis_clean_1_31.csv",
        out_fig_dir=results_dir / "figures",
        out_tab_dir=results_dir / "tables",
    )


def load_inputs(
    paths: PipelinePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre_long = pd.read_csv(paths.pre_long_csv, dtype=str)
    pre_pt = pd.read_csv(paths.pre_pt_csv, dtype=str)

    patient = pd.read_csv(paths.patient_csv, dtype=str, sep="|")
    patient["Patient_ID"] = patient["Patient_ID"].astype(str).str.strip()

    encounter = pd.read_csv(paths.encounter_csv, dtype=str)
    encounter["Patient_ID"] = encounter["Patient_ID"].astype(str).str.strip()

    pre_long["Patient_ID"] = pre_long["Patient_ID"].astype(str).str.strip()
    pre_pt["Patient_ID"] = pre_pt["Patient_ID"].astype(str).str.strip()

    keep_ids = set(pre_pt["Patient_ID"].unique())
    patient = patient[patient["Patient_ID"].isin(keep_ids)].copy()
    encounter = encounter[encounter["Patient_ID"].isin(keep_ids)].copy()

    return pre_long, pre_pt, patient, encounter


# Build first dates from pre-MSK encounter table
def build_first_system_dates_from_pre_long(
    pre_long: pd.DataFrame,
    patient_col: str = "Patient_ID",
    date_col: str = "DateCreated",
    type_col: str = "_type",
) -> pd.DataFrame:
    df = pre_long[[patient_col, date_col, type_col]].copy()
    df[patient_col] = df[patient_col].astype(str).str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[type_col] = df[type_col].astype(str).str.strip()
    df = df.dropna(subset=[patient_col, date_col, type_col])

    out = (
        df.groupby([patient_col, type_col], as_index=False)[date_col]
        .min()
        .rename(columns={date_col: "first_date", type_col: "system"})
    )
    return out


def pivot_first_dates(
    first_sys: pd.DataFrame,
    patient_col: str = "Patient_ID",
) -> pd.DataFrame:
    wide = (
        first_sys.pivot_table(
            index=patient_col,
            columns="system",
            values="first_date",
            aggfunc="min",
        )
        .reset_index()
    )

    rename = {}
    if "respiratory" in wide.columns:
        rename["respiratory"] = "first_resp"
    if "mental" in wide.columns:
        rename["mental"] = "first_mental"
    if "msk" in wide.columns:
        rename["msk"] = "first_msk"

    return wide.rename(columns=rename)


def add_first_msk_from_pre_pt(
    wide_dates: pd.DataFrame,
    pre_pt: pd.DataFrame,
) -> pd.DataFrame:
    df = wide_dates.copy()
    pt = pre_pt.copy()

    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()
    pt["Patient_ID"] = pt["Patient_ID"].astype(str).str.strip()

    keep = ["Patient_ID", "first_msk_date"]
    pt = pt[keep].copy().rename(columns={"first_msk_date": "first_msk"})
    pt["first_msk"] = pd.to_datetime(pt["first_msk"], errors="coerce")

    df = df.merge(pt, on="Patient_ID", how="left")
    return df


# =========================================================
# First MSK subgroup
# =========================================================
def build_first_msk_subgroup_from_encounter(
    encounter: pd.DataFrame,
    pre_pt: pd.DataFrame,
) -> pd.DataFrame:
    enc = encounter.copy()
    pt = pre_pt.copy()

    enc["Patient_ID"] = enc["Patient_ID"].astype(str).str.strip()
    pt["Patient_ID"] = pt["Patient_ID"].astype(str).str.strip()

    if "DiagnosisCodeType_calc" in enc.columns:
        enc = enc[enc["DiagnosisCodeType_calc"] == "ICD9"].copy()

    if "DiagnosisCode_calc" in enc.columns and "icd9" not in enc.columns:
        enc = enc.rename(columns={"DiagnosisCode_calc": "icd9"})

    if "icd9" not in enc.columns:
        raise ValueError("Encounter table must contain 'icd9' or 'DiagnosisCode_calc'.")

    enc["DateCreated"] = pd.to_datetime(enc["DateCreated"], errors="coerce")
    enc["icd9"] = enc["icd9"].astype(str).str.strip()
    enc["icd9_3"] = enc["icd9"].apply(_icd9_3_int)
    enc["system"] = enc["icd9"].apply(classify_system)
    enc["subgroup"] = enc["icd9"].apply(classify_subgroup)
    enc["acute_chronic"] = enc["icd9"].apply(classify_acute_chronic)

    pt = pt[["Patient_ID", "first_msk_date"]].copy()
    pt["first_msk_date"] = pd.to_datetime(pt["first_msk_date"], errors="coerce")

    df = enc.merge(pt, on="Patient_ID", how="inner")
    df = df[(df["system"] == "msk") & (df["DateCreated"] == df["first_msk_date"])].copy()
    df = df.dropna(subset=["Patient_ID", "icd9_3", "subgroup"]).copy()
    df = df.sort_values(["Patient_ID", "DateCreated", "icd9_3", "icd9"])

    first_msk = (
        df.drop_duplicates(subset=["Patient_ID"], keep="first")
        .loc[:, ["Patient_ID", "icd9", "icd9_3", "subgroup", "acute_chronic"]]
        .rename(
            columns={
                "icd9": "first_msk_icd9",
                "icd9_3": "first_msk_icd9_3",
                "subgroup": "first_msk_subgroup",
                "acute_chronic": "first_msk_course",
            }
        )
        .copy()
    )
    return first_msk


def add_msk_exposure_features(
    wide_dates: pd.DataFrame,
    patient: pd.DataFrame,
) -> pd.DataFrame:
    df = wide_dates.copy()
    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()

    patient = patient.copy()
    patient["Patient_ID"] = patient["Patient_ID"].astype(str).str.strip()

    keep_cols = [c for c in ["Patient_ID", "Sex", "BirthYear"] if c in patient.columns]
    if keep_cols:
        df = df.merge(patient[keep_cols], on="Patient_ID", how="left")

    if "first_msk" not in df.columns:
        raise ValueError("Missing required column 'first_msk'.")

    df["first_msk"] = pd.to_datetime(df["first_msk"], errors="coerce")
    df = df.dropna(subset=["first_msk"]).copy()

    if "first_resp" not in df.columns:
        df["first_resp"] = pd.NaT
    if "first_mental" not in df.columns:
        df["first_mental"] = pd.NaT

    df["first_resp"] = pd.to_datetime(df["first_resp"], errors="coerce")
    df["first_mental"] = pd.to_datetime(df["first_mental"], errors="coerce")

    if "Sex" in df.columns:
        df["Sex"] = _normalize_sex(df["Sex"])

    if "BirthYear" in df.columns:
        df["BirthYear"] = pd.to_numeric(df["BirthYear"], errors="coerce")
        df["Age_at_MSK"] = pd.to_datetime(df["first_msk"]).dt.year - df["BirthYear"]
        df["Age_at_MSK"] = df["Age_at_MSK"].clip(lower=0, upper=110)
        df["Age_at_mental"] = pd.to_datetime(df["first_mental"]).dt.year - df["BirthYear"]
        df["Age_at_mental"] = df["Age_at_mental"].clip(lower=0, upper=110)

        df["resp_before_msk"] = df["first_resp"].where(
        (df["first_resp"] >= df["first_mental"]) &
        (df["first_resp"] < df["first_msk"])
    )

    df["mental_before_msk"] = df["first_mental"].where(
        df["first_mental"] < df["first_msk"]
    )

    has_r = df["resp_before_msk"].notna()
    has_m = df["mental_before_msk"].notna()

    grp_order = ["None", "Resp only", "Mental only", "Both"]
    df["group"] = np.select(
        [
            (~has_r) & (~has_m),
            has_r & (~has_m),
            (~has_r) & has_m,
            has_r & has_m,
        ],
        grp_order,
        default="None",
    )
    df["group"] = pd.Categorical(df["group"], categories=grp_order, ordered=True)

    df["has_resp"] = df["group"].isin(["Resp only", "Both"])
    df["has_mental"] = df["group"].isin(["Mental only", "Both"])

    df["exposure_start"] = pd.NaT

    mask_resp_only = df["group"] == "Resp only"
    mask_mental_only = df["group"] == "Mental only"
    mask_both = df["group"] == "Both"

    df.loc[mask_resp_only, "exposure_start"] = df.loc[
        mask_resp_only, "resp_before_msk"
    ]
    df.loc[mask_mental_only, "exposure_start"] = df.loc[
        mask_mental_only, "mental_before_msk"
    ]
    df.loc[mask_both, "exposure_start"] = df.loc[
        mask_both, ["resp_before_msk", "mental_before_msk"]
    ].min(axis=1)

    df["time_to_msk_days"] = (df["first_msk"] - df["exposure_start"]).dt.days
    df.loc[df["time_to_msk_days"] < 0, "time_to_msk_days"] = np.nan

    return df


def build_analysis_cohorts(
    df_msk_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1) df_analysis_all   : Mental only / Both, all ages
    2) df_analysis_18plus: Mental only / Both, age >=18, used for IQR
    3) df_analysis_adult : Mental only / Both, age 18-102, final cohort
    """
    df_analysis_all = df_msk_all[
        df_msk_all["group"].isin(["Mental only", "Both"])
    ].copy()

    df_analysis_18plus = df_analysis_all[
        pd.to_numeric(df_analysis_all["Age_at_mental"], errors="coerce") >= 18
    ].copy()

    df_analysis_adult = df_analysis_all[
        pd.to_numeric(df_analysis_all["Age_at_mental"], errors="coerce").between(18, 102, inclusive="both")
    ].copy()

    return df_analysis_all, df_analysis_18plus, df_analysis_adult

# =========================================================
# Respiratory / mental long table
# =========================================================
def build_rm_long_from_pre_long(pre_long: pd.DataFrame) -> pd.DataFrame:
    df = pre_long.copy()
    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()

    if "icd9_split" in df.columns:
        df["icd9"] = df["icd9_split"].astype(str).str.strip()
    elif "icd9" in df.columns:
        df["icd9"] = df["icd9"].astype(str).str.strip()
    else:
        raise ValueError("pre_long must contain 'icd9_split' or 'icd9'.")

    df["icd9_3"] = df["icd9"].apply(_icd9_3_int)
    df["system"] = df["icd9"].apply(classify_system)
    df["acute_chronic"] = df["icd9"].apply(classify_acute_chronic)
    df["subgroup"] = df["icd9"].apply(classify_subgroup)

    rm_long = df[df["system"].isin(["mental", "respiratory"])].copy()
    return rm_long

def summarize_rm_code_patterns(
    rm_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize acute/chronic composition within respiratory and mental systems.
    Returns:
    1) count table
    2) row-wise percent table
    """
    count_table = pd.crosstab(rm_long["system"], rm_long["acute_chronic"])

    pct_table = (
        pd.crosstab(
            rm_long["system"],
            rm_long["acute_chronic"],
            normalize="index",
        ) * 100
    ).round(1)

    return count_table, pct_table



def subgroup_counts_by_system(rm_long: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    resp_subgroup = (
        rm_long[rm_long["system"] == "respiratory"]["subgroup"]
        .dropna()
        .value_counts()
    )
    mental_subgroup = (
        rm_long[rm_long["system"] == "mental"]["subgroup"]
        .dropna()
        .value_counts()
    )
    return resp_subgroup, mental_subgroup


# =========================================================
# Table summaries
# =========================================================
def summarize_groups(df_msk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grp_order = list(df_msk["group"].cat.categories)

    grp_counts = df_msk["group"].value_counts().reindex(grp_order).fillna(0).astype(int)
    grp_perc = (grp_counts / max(1, grp_counts.sum()) * 100).round(2)
    group_summary = pd.DataFrame(
        {"group": grp_order, "N": grp_counts.values, "percent": grp_perc.values}
    )

    if "Sex" in df_msk.columns:
        female_pct = (
            df_msk.groupby("group", observed=False)["Sex"]
            .apply(lambda s: (s == "F").mean() * 100)
            .reindex(grp_order)
        )
    else:
        female_pct = pd.Series(index=grp_order, data=np.nan)

    demographics = (
        df_msk.groupby("group", as_index=False, observed=False)
        .agg(
            N=("Patient_ID", "count"),
            Age_mean=("Age_at_mental", "mean"),
            Age_sd=("Age_at_mental", "std"),
            Median_time_to_MSK=("time_to_msk_days", "median"),
        )
        .sort_values("group")
    )
    demographics["Female_pct"] = demographics["group"].map(female_pct).astype(float)

    return group_summary, demographics


def _format_p(x) -> str:
    if pd.isna(x):
        return ""
    x = float(x)
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def compute_profile_pvalues(df_msk: pd.DataFrame) -> pd.DataFrame:
    rows = []

    age_groups = []
    for _, g in df_msk.groupby("group", observed=False):
        v = pd.to_numeric(g["Age_at_mental"], errors="coerce").dropna().values
        age_groups.append(v)

    if all(len(v) > 0 for v in age_groups) and len(age_groups) >= 2:
        H, p_age = kruskal(*age_groups)
        rows.append(
            {
                "variable": "Age_at_mental",
                "test": "Kruskal-Wallis",
                "statistic": float(H),
                "p_value": float(p_age),
            }
        )
    else:
        rows.append(
            {
                "variable": "Age_at_mental",
                "test": "Kruskal-Wallis",
                "statistic": np.nan,
                "p_value": np.nan,
            }
        )

    time_groups = []
    for _, g in df_msk.groupby("group", observed=False):
        v = pd.to_numeric(g["time_to_msk_days"], errors="coerce").dropna().values
        time_groups.append(v)

    if all(len(v) > 0 for v in time_groups) and len(time_groups) >= 2:
        H, p_time = kruskal(*time_groups)
        rows.append(
            {
                "variable": "time_to_msk_days",
                "test": "Kruskal-Wallis",
                "statistic": float(H),
                "p_value": float(p_time),
            }
        )
    else:
        rows.append(
            {
                "variable": "time_to_msk_days",
                "test": "Kruskal-Wallis",
                "statistic": np.nan,
                "p_value": np.nan,
            }
        )

    if "Sex" in df_msk.columns:
        d2 = df_msk.dropna(subset=["Sex"]).copy()
        if len(d2) > 0:
            ct = pd.crosstab(d2["group"], d2["Sex"])
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2, p, _, _ = chi2_contingency(ct)
                rows.append(
                    {
                        "variable": "Sex",
                        "test": "Chi-square",
                        "statistic": float(chi2),
                        "p_value": float(p),
                    }
                )
            else:
                rows.append(
                    {
                        "variable": "Sex",
                        "test": "Chi-square",
                        "statistic": np.nan,
                        "p_value": np.nan,
                    }
                )
        else:
            rows.append(
                {
                    "variable": "Sex",
                    "test": "Chi-square",
                    "statistic": np.nan,
                    "p_value": np.nan,
                }
            )
    else:
        rows.append(
            {
                "variable": "Sex",
                "test": "Chi-square",
                "statistic": np.nan,
                "p_value": np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out["p_value"] = out["p_value"].apply(_format_p)
    return out


def make_table1_full(demo: pd.DataFrame, pvals: pd.DataFrame) -> pd.DataFrame:
    out = demo.copy()
    out["Age_mean_sd"] = out.apply(
        lambda r: f"{r['Age_mean']:.2f} ({r['Age_sd']:.2f})" if pd.notna(r["Age_mean"]) else "",
        axis=1,
    )
    out["Female_%"] = out["Female_pct"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    out["Median_time_to_MSK"] = out["Median_time_to_MSK"].map(
        lambda x: f"{x:.1f}" if pd.notna(x) else ""
    )
    out = out[["group", "N", "Age_mean_sd", "Female_%", "Median_time_to_MSK"]].copy()

    p_age = pvals.loc[pvals["variable"] == "Age_at_mental", "p_value"]
    p_sex = pvals.loc[pvals["variable"] == "Sex", "p_value"]
    p_time = pvals.loc[pvals["variable"] == "time_to_msk_days", "p_value"]

    p_age = p_age.iloc[0] if len(p_age) else ""
    p_sex = p_sex.iloc[0] if len(p_sex) else ""
    p_time = p_time.iloc[0] if len(p_time) else ""

    footer = pd.DataFrame(
        [
            {"group": "P-value (Age_at_mental)", "N": "", "Age_mean_sd": p_age, "Female_%": "", "Median_time_to_MSK": ""},
            {"group": "P-value (Sex)", "N": "", "Age_mean_sd": "", "Female_%": p_sex, "Median_time_to_MSK": ""},
            {"group": "P-value (time_to_msk_days)", "N": "", "Age_mean_sd": "", "Female_%": "", "Median_time_to_MSK": p_time},
        ]
    )
    return pd.concat([out, footer], ignore_index=True)



# =========================================================
# IQR age screening
# =========================================================
def get_age_iqr_outliers(
    df: pd.DataFrame,
    age_col: str = "Age_at_mental",
    min_age: int | None = None,
    keep_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d = d.dropna(subset=[age_col]).copy()

    if min_age is not None:
        d = d[d[age_col] >= min_age].copy()

    q1 = d[age_col].quantile(0.25)
    q3 = d[age_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    stats = pd.Series({
        "N": int(len(d)),
        "Q1": round(q1, 2),
        "Q3": round(q3, 2),
        "IQR": round(iqr, 2),
        "Lower_bound": round(lower, 2),
        "Upper_bound": round(upper, 2),
        "Min_age": round(d[age_col].min(), 2),
        "Max_age": round(d[age_col].max(), 2),
    })

    outliers = d[(d[age_col] < lower) | (d[age_col] > upper)].copy()

    if keep_cols is not None:
        keep_cols = [c for c in keep_cols if c in outliers.columns]
        outliers = outliers[keep_cols].copy()

    outliers = outliers.sort_values(age_col)
    return outliers, stats


# =========================================================
# Other summaries
# =========================================================
def relabel_msk_subgroup_with_code(
    counts: pd.Series,
    code_map: dict[str, set[int]],
) -> pd.Series:
    def _label(subgroup: str) -> str:
        codes = sorted(code_map.get(subgroup, []))
        if len(codes) == 1:
            code_txt = str(codes[0])
        elif len(codes) > 1:
            code_txt = "/".join(map(str, codes))
        else:
            code_txt = ""
        name_txt = subgroup.replace("_", " ")
        return f"{code_txt} {name_txt}".strip()

    out = counts.copy()
    out.index = [_label(x) for x in out.index]
    return out


def summarize_both_temporal(df_msk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    both = df_msk[df_msk["group"] == "Both"].copy()

    if len(both) == 0:
        order_summary = pd.DataFrame(
            {
                "order": ["Resp → Mental", "Mental → Resp", "Same day"],
                "N": [0, 0, 0],
                "percent": [0.0, 0.0, 0.0],
            }
        )
        return order_summary, both

    both["order"] = np.where(
        both["resp_before_msk"] < both["mental_before_msk"],
        "Resp → Mental",
        np.where(
            both["mental_before_msk"] < both["resp_before_msk"],
            "Mental → Resp",
            "Same day",
        ),
    )

    order_order = ["Resp → Mental", "Mental → Resp", "Same day"]
    order_counts = both["order"].value_counts().reindex(order_order).fillna(0).astype(int)
    order_perc = (order_counts / max(1, len(both)) * 100).round(2)

    order_summary = pd.DataFrame(
        {"order": order_order, "N": order_counts.values, "percent": order_perc.values}
    )

    both["gap_days_mental_minus_resp"] = (both["mental_before_msk"] - both["resp_before_msk"]).dt.days
    both["first_nonmsk"] = both[["resp_before_msk", "mental_before_msk"]].min(axis=1)
    both["lag_to_msk_days"] = (both["first_msk"] - both["first_nonmsk"]).dt.days

    return order_summary, both


# =========================================================
# Core export
# =========================================================
def build_cohort_flow_table(
    pre_pt: pd.DataFrame,
    wide: pd.DataFrame,
    df_msk_all: pd.DataFrame,
    df_analysis_all: pd.DataFrame,
    df_analysis_adult: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "step": "Pre-MSK patient table",
            "cohort": "all_ages",
            "N_patients": pre_pt["Patient_ID"].nunique(),
        },
        {
            "step": "Wide first-date table",
            "cohort": "all_ages",
            "N_patients": wide["Patient_ID"].nunique(),
        },
        {
            "step": "Patients with first MSK",
            "cohort": "all_ages",
            "N_patients": df_msk_all["Patient_ID"].nunique(),
        },
        {
            "step": "Analysis cohort (Mental / Both)",
            "cohort": "all_ages",
            "N_patients": df_analysis_all["Patient_ID"].nunique(),
        },
        {
            "step": "Final cohort ( Mental / Both, age 18-102)",
            "cohort": "age_18_102",
            "N_patients": df_analysis_adult["Patient_ID"].nunique(),
        },
    ]
    return pd.DataFrame(rows)


def build_group_count_table(
    df: pd.DataFrame,
    group_col: str = "group",
) -> pd.DataFrame:
    counts = df[group_col].value_counts(dropna=False)
    pct = df[group_col].value_counts(normalize=True, dropna=False) * 100

    out = pd.DataFrame({
        group_col: counts.index,
        "N_patients": counts.values,
        "Percent": pct.round(2).values,
    })
    return out


# =========================================================
# Main pipeline
# =========================================================
def run_step1_pubfigs(
    paths: PipelinePaths,
    do_both_temporal: bool = True,
    export_group_details: bool = False,
) -> None:
    paths.out_fig_dir.mkdir(parents=True, exist_ok=True)
    paths.out_tab_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # Load inputs
    # ------------------------------------------------
    pre_long, pre_pt, patient, encounter = load_inputs(paths)

    # ------------------------------------------------
    # Build patient-level first-event table
    # ------------------------------------------------
    first_sys = build_first_system_dates_from_pre_long(pre_long)
    wide = pivot_first_dates(first_sys)
    wide = add_first_msk_from_pre_pt(wide, pre_pt)

    first_msk_sub = build_first_msk_subgroup_from_encounter(encounter, pre_pt)
    wide = wide.merge(first_msk_sub, on="Patient_ID", how="left")

    df_msk_all = add_msk_exposure_features(wide, patient)

    # ------------------------------------------------
    # Build cohorts
    # ------------------------------------------------
    df_analysis_all, df_analysis_18plus, df_analysis_adult = build_analysis_cohorts(
        df_msk_all
    )

    print("\n=== Cohort counts ===")
    print("Pre-MSK patient table:", pre_pt["Patient_ID"].nunique())
    print("Wide first-date table:", wide["Patient_ID"].nunique())
    print("Patients with first MSK (all ages):", df_msk_all["Patient_ID"].nunique())
    print("Analysis cohort (all ages):", df_analysis_all["Patient_ID"].nunique())
    print("Final cohort (18–102):", df_analysis_adult["Patient_ID"].nunique())

    # ------------------------------------------------
    # Age IQR on analysis cohort (Age >=18, before 18-102 restriction)
    # ------------------------------------------------
    age_outliers_18plus, age_iqr_stats_18plus = get_age_iqr_outliers(
        df_analysis_18plus,
        age_col="Age_at_mental",
        min_age=None,
        keep_cols=["Patient_ID", "Age_at_mental", "Sex", "group"],
    )

    age_iqr_summary_18plus = age_iqr_stats_18plus.to_frame().T
    age_iqr_summary_18plus["Outlier_n"] = len(age_outliers_18plus)
    age_iqr_summary_18plus["Outlier_pct"] = round(
        len(age_outliers_18plus) / age_iqr_summary_18plus.loc[0, "N"] * 100, 4
    ) if age_iqr_summary_18plus.loc[0, "N"] > 0 else 0.0

    print("\n=== Age IQR check on analysis cohort (Age >=18) ===")
    print(age_iqr_summary_18plus.to_string(index=False))

    print("\n=== Age outliers from analysis cohort (Age >=18) ===")
    if len(age_outliers_18plus) == 0:
        print("No age outliers found.")
    else:
        print(age_outliers_18plus[["Patient_ID", "Age_at_mental"]].to_string(index=False))

    # ------------------------------------------------
    # Cohort flow + group counts
    # ------------------------------------------------
    cohort_flow = build_cohort_flow_table(
        pre_pt=pre_pt,
        wide=wide,
        df_msk_all=df_msk_all,
        df_analysis_all=df_analysis_all,
        df_analysis_adult=df_analysis_adult,
    )

    group_count_adult = build_group_count_table(df_analysis_adult, group_col="group")

    # ------------------------------------------------
    # Table 1 (final cohort)
    # ------------------------------------------------
    group_summary_adult, demographics_adult = summarize_groups(df_analysis_adult)
    pvals_adult = compute_profile_pvalues(df_analysis_adult)
    table1_full_adult = make_table1_full(demographics_adult, pvals_adult)

    # ------------------------------------------------
    # Long-table summaries on final cohort
    # ------------------------------------------------
    rm_long = build_rm_long_from_pre_long(pre_long)

    keep_ids_adult = set(df_analysis_adult["Patient_ID"].astype(str).str.strip().unique())
    rm_long_adult = rm_long[rm_long["Patient_ID"].isin(keep_ids_adult)].copy()

    resp_subgroup_adult, mental_subgroup_adult = subgroup_counts_by_system(rm_long_adult)
    acute_chronic_count_adult, acute_chronic_pct_adult = summarize_rm_code_patterns(rm_long_adult)

    fig_acute_chronic_composition(
        acute_chronic_count_adult,
        paths.out_fig_dir / "fig_acute_chronic_composition_age18_102.png"
    )
    # ------------------------------------------------
    # First MSK subgroup (final cohort)
    # ------------------------------------------------
    first_msk_counts_adult = df_analysis_adult["first_msk_subgroup"].dropna().value_counts()

    first_msk_counts_adult_plot = relabel_msk_subgroup_with_code(
        first_msk_counts_adult,
        MSK_SUBGROUPS,
    )

    # ------------------------------------------------
    # Figures
    # ------------------------------------------------
    fig_first_msk_subgroup_topk(
        first_msk_counts_adult_plot,
        paths.out_fig_dir / "fig_first_msk_subgroup_age18_102.png",
        title="First MSK diagnosis subgroup (Age 18–102 years)",
        top_k=15,
    )

    fig_subgroup_bar(
        resp_subgroup_adult,
        paths.out_fig_dir / "fig_resp_subgroup_age18_102.png",
        title="Respiratory subgroups before first MSK (Age 18-102 years)",
        code_map=RESP_SUBGROUPS,
        top_k=15,
    )

    fig_subgroup_bar(
        mental_subgroup_adult,
        paths.out_fig_dir / "fig_mental_subgroup_age18_102.png",
        title="Mental subgroups before first MSK (Age 18–102 years)",
        code_map=MENTAL_SUBGROUPS,
        top_k=15,
    )

    fig_groups_before_msk(
        group_summary_adult[group_summary_adult["group"].isin(["Mental only", "Both"])].copy(),
        paths.out_fig_dir / "fig_groups_before_msk_age18_102.png",
        cohort_label="Age 18-102 years",
    )


    # Subgroup composition by course (one figure only)
    fig_subgroup_by_course_stacked(
        rm_long=rm_long_adult,
        system_name="mental",
        fig_path=paths.out_fig_dir / "fig_mental_subgroup_by_course_age18_102.png",
        title="Mental subgroups by course before first MSK (Age 18-102 years)",
        code_map=MENTAL_SUBGROUPS,
        course_order=["acute", "chronic", "unknown"],
        top_k=15,
        normalize=False,
    )

    fig_subgroup_by_course_stacked(
        rm_long=rm_long_adult,
        system_name="respiratory",
        fig_path=paths.out_fig_dir / "fig_resp_subgroup_by_course_age18_102.png",
        title="Respiratory subgroups by course before first MSK (Age 18–102 years)",
        code_map=RESP_SUBGROUPS,
        course_order=["acute", "chronic", "unknown"],
        top_k=15,
        normalize=False,
    )
    fig_first_msk_subgroup_by_course_stacked(
        df=df_analysis_adult,
        fig_path=paths.out_fig_dir / "fig_first_msk_subgroup_by_course_age18_102.png",
        title="First MSK diagnosis subgroups by course (Age 18–102 years)",
        code_map=MSK_SUBGROUPS,
        course_order=["acute", "chronic", "unknown"],
        top_k=15,
        normalize=False,
    )

    # Both-temporal summary
    if do_both_temporal:
        order_summary_adult, both_detail_adult = summarize_both_temporal(df_analysis_adult)

        fig_order_among_both(
            order_summary_adult,
            paths.out_fig_dir / "fig_order_among_both_age18_102.png",
            cohort_label="Age 18-102 years",
        )


    # Export final tables only
    cohort_flow.to_csv(
        paths.out_tab_dir / "cohort_flow.csv",
        index=False,
    )

    group_count_adult.to_csv(
        paths.out_tab_dir / "group_count_final_cohort.csv",
        index=False,
    )

    table1_full_adult.to_csv(
        paths.out_tab_dir / "table1_final_cohort.csv",
        index=False,
    )

    age_iqr_summary_18plus.to_csv(
        paths.out_tab_dir / "age_iqr_summary.csv",
        index=False,
    )

    first_msk_counts_adult.to_csv(
        paths.out_tab_dir / "first_msk_subgroup_final_cohort.csv",
        header=["N"],
    )

    resp_subgroup_adult.to_csv(
        paths.out_tab_dir / "resp_subgroup_final_cohort.csv",
        header=["N"],
    )

    mental_subgroup_adult.to_csv(
        paths.out_tab_dir / "mental_subgroup_final_cohort.csv",
        header=["N"],
    )

    acute_chronic_pct_adult.to_csv(
        paths.out_tab_dir / "acute_chronic_percent_final_cohort.csv"
    )

    analysis_cols = [
        "Patient_ID",
        "Sex",
        "BirthYear",
        "Age_at_mental",
        "first_resp",
        "first_mental",
        "first_msk",
        "first_msk_icd9",
        "first_msk_icd9_3",
        "first_msk_subgroup",
        "resp_before_msk",
        "mental_before_msk",
        "exposure_start",
        "time_to_msk_days",
        "group",
        "has_resp",
        "has_mental",
    ]
    analysis_cols = [c for c in analysis_cols if c in df_analysis_adult.columns]

    df_analysis_adult[analysis_cols].to_csv(
        paths.out_tab_dir / "analysis_cohort_final.csv",
        index=False,
    )

    print("\n[Step1] Saved figures to:", paths.out_fig_dir)
    print("[Step1] Saved tables to :", paths.out_tab_dir)


def main():
    paths = default_paths()
    run_step1_pubfigs(paths, do_both_temporal=True, export_group_details=False)


if __name__ == "__main__":
    main()