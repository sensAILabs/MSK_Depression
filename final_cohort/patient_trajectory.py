from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path("/Users/xinyichen/Desktop/BiostatisticsPracticum_Depression")
SENSITIVITY_DIR = (ROOT / "final_cohort" / "results" / "tables"/ "sensitivity_analysis")
HIGH_PATH = SENSITIVITY_DIR / "03_high_applicability_Q1_patients.csv"
INTERMEDIATE_PATH = (SENSITIVITY_DIR/ "04_intermediate_applicability_Q2_Q3_patients.csv")
LOW_PATH = SENSITIVITY_DIR / "05_low_applicability_Q4_patients.csv"
LONGITUDINAL_PATH = (ROOT / "final_cohort"/ "results"/ "tables"/ "final_analysis_long.csv")
OUTPUT_DIR = (ROOT/ "final_cohort"/ "results"/ "tables"/ "patient_trajectory")
FIGURE_DIR = (ROOT/ "final_cohort"/ "results" / "figures" / "patient_trajectory")

N_TRAJECTORY_SETS = 3
PATIENTS_PER_TYPE_TO_REVIEW = 20

ANALYSIS_MAX_DAYS = 365
DISPLAY_MAX_DAYS = 400

GOOD_MIN_WINDOWS = 4
GOOD_MIN_CHANGED_DOMAINS = 2
GOOD_MIN_TOTAL_CHANGE_EVENTS = 2

NORMAL_MIN_WINDOWS = 4
NORMAL_MIN_DOMAINS = 2

FN_MIN_PRE_MSK_WINDOWS = 2

FP_MIN_WINDOWS = 5
FP_MIN_DOMAINS = 2

TRAJECTORY_TYPES = [
    "Good example",
    "Normal example",
    "Bad example 1",
    "Bad example 2",
]

LINESTYLES = {
    "Good example": "-",
    "Normal example": "--",
    "Bad example 1": "-.",
    "Bad example 2": ":",
}

MARKER_X_OFFSETS = {
    "Good example": 0.0,
    "Normal example": 2.5,
    "Bad example 1": 5.0,
    "Bad example 2": 7.5,
}

PATIENT_COLORS = {
    "Good example": "#2ca02c",     
    "Normal example": "#1f77b4",    
    "Bad example 1": "#ff7f0e",    
    "Bad example 2": "#d62728",     
}

FINAL_SELECTED_PATIENTS = {
    "Good example": "6550",
    "Normal example": "7346",
    "Bad example 1": "2526",
    "Bad example 2": "20982",
}

FINAL_SELECTION_RATIONALE = {
    "Good example": (
        "High-applicability true positive with persistent mental-health "
        "diagnosis records and medication change before MSK."
    ),
    "Normal example": (
        "Intermediate-applicability patient with complete mental-health and "
        "medication information but a relatively flat pre-MSK trajectory."
    ),
    "Bad example 1": (
        "Low-applicability false negative with no or minimal medication "
        "information before MSK."
    ),
    "Bad example 2": (
        "Low-applicability false negative with sparse mental-health "
        "diagnosis information before MSK."
    ),
}



def first_existing_column(
    df: pd.DataFrame,
    possible_columns: list[str],
) -> str | None:
    for column in possible_columns:
        if column in df.columns:
            return column
    return None


def safe_numeric_column(
    df: pd.DataFrame,
    possible_columns: list[str],
    default: float = 0.0,
) -> pd.Series:
    column = first_existing_column(df, possible_columns)

    if column is None:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


def robust_quantile(
    series: pd.Series,
    q: float,
    default: float = 0.0,
) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return default

    return float(values.quantile(q))


def count_change_events(series: pd.Series) -> int:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0).to_numpy()

    if len(values) <= 1:
        return 0

    return int(np.sum(~np.isclose(values[1:], values[:-1])))


def count_increase_events(series: pd.Series) -> int:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0).to_numpy()

    if len(values) <= 1:
        return 0

    return int(np.sum(values[1:] > values[:-1]))


def load_patient_groups() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    high = pd.read_csv(HIGH_PATH)
    intermediate = pd.read_csv(INTERMEDIATE_PATH)
    low = pd.read_csv(LOW_PATH)

    for frame, label in [
        (high, "High applicability"),
        (intermediate, "Intermediate applicability"),
        (low, "Low applicability"),
    ]:
        frame["Patient_ID"] = frame["Patient_ID"].astype(str)
        frame["source_group"] = label

    return high, intermediate, low


def load_longitudinal_data() -> pd.DataFrame:
    df = pd.read_csv(
        LONGITUDINAL_PATH,
        low_memory=False,
    )
    df["Patient_ID"] = df["Patient_ID"].astype(str)

    if "window_days" not in df.columns:
        raise KeyError(
            "final_analysis_long.csv must contain 'window_days'."
        )

    df["window_days"] = pd.to_numeric(
        df["window_days"],
        errors="coerce",
    )

    return df.dropna(
        subset=["window_days"]
    ).copy()

def add_trajectory_measures(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["mental_frequency"] = safe_numeric_column(
        result,
        ["Mental_Diagnosis_count"],
    )

    mental_medications = safe_numeric_column(
        result,
        [
            "n_mental_drugs",
            "Mental_related_medication_drug_count",
            "Mental_related_medication_class_count",
        ],
    )

    pain_medications = safe_numeric_column(
        result,
        [
            "n_pain_drugs",
            "Pain_related_medication_drug_count",
            "Pain_related_medication_class_count",
        ],
    )

    result["medication_frequency"] = (
        mental_medications + pain_medications
    )

    result["comorbidity_frequency"] = safe_numeric_column(
        result,
        ["Multimorbidity_count"],
    )

    return result


def aggregate_patient_windows(
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    measure_columns = [
        "mental_frequency",
        "medication_frequency",
        "comorbidity_frequency",
    ]

    return (
        patient_df
        .groupby(
            "window_days",
            as_index=False,
        )[measure_columns]
        .max()
        .sort_values("window_days")
        .reset_index(drop=True)
    )


# First MSK day
def derive_first_msk_day(
    patient_longitudinal: pd.DataFrame,
    patient_summary_row: pd.Series | None = None,
) -> float:
    if patient_summary_row is None:
        patient_summary_row = pd.Series(dtype=object)

    direct_day_columns = [
        "first_msk_day",
        "days_to_first_msk",
        "days_from_first_mental_to_msk",
        "time_to_msk_days",
    ]

    for column in direct_day_columns:
        if column in patient_longitudinal.columns:
            values = pd.to_numeric(
                patient_longitudinal[column],
                errors="coerce",
            ).dropna()

            if not values.empty:
                return float(values.iloc[0])

        if column in patient_summary_row.index:
            value = pd.to_numeric(
                pd.Series([patient_summary_row[column]]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(value):
                return float(value)

    date_pairs = [
        ("first_mental_date", "first_msk_date"),
        ("First_Mental_Date", "First_MSK_Date"),
    ]

    for mental_column, msk_column in date_pairs:
        if (
            mental_column in patient_longitudinal.columns
            and msk_column in patient_longitudinal.columns
        ):
            mental_dates = pd.to_datetime(
                patient_longitudinal[mental_column],
                errors="coerce",
            ).dropna()

            msk_dates = pd.to_datetime(
                patient_longitudinal[msk_column],
                errors="coerce",
            ).dropna()

            if not mental_dates.empty and not msk_dates.empty:
                return float(
                    (
                        msk_dates.iloc[0]
                        - mental_dates.iloc[0]
                    ).days
                )

    return np.nan


# Longitudinal profile summary
def summarize_trajectory_block(
    patient_df: pd.DataFrame,
    prefix: str = "",
) -> dict[str, float | int]:
    if patient_df.empty:
        return {
            f"{prefix}n_windows": 0,
            f"{prefix}mental_change_events": 0,
            f"{prefix}medication_change_events": 0,
            f"{prefix}comorbidity_change_events": 0,
            f"{prefix}trajectory_change_events": 0,
            f"{prefix}trajectory_increase_events": 0,
            f"{prefix}changed_domain_count": 0,
            f"{prefix}trajectory_change_density": 0.0,
            f"{prefix}mental_max": 0.0,
            f"{prefix}medication_max": 0.0,
            f"{prefix}comorbidity_max": 0.0,
            f"{prefix}trajectory_total_max": 0.0,
            f"{prefix}profile_domain_count": 0,
            f"{prefix}all_three_domains_present": 0,
            f"{prefix}flat_trajectory": 1,
        }

    patient_df = aggregate_patient_windows(
        patient_df.sort_values("window_days")
    )

    mental = patient_df["mental_frequency"]
    medication = patient_df["medication_frequency"]
    comorbidity = patient_df["comorbidity_frequency"]

    mental_changes = count_change_events(mental)
    medication_changes = count_change_events(medication)
    comorbidity_changes = count_change_events(comorbidity)

    total_changes = (
        mental_changes
        + medication_changes
        + comorbidity_changes
    )

    total_increases = (
        count_increase_events(mental)
        + count_increase_events(medication)
        + count_increase_events(comorbidity)
    )

    changed_domain_count = int(mental_changes > 0)
    changed_domain_count += int(medication_changes > 0)
    changed_domain_count += int(comorbidity_changes > 0)

    profile_domain_count = int(mental.max() > 0)
    profile_domain_count += int(medication.max() > 0)
    profile_domain_count += int(comorbidity.max() > 0)

    possible_transitions = max(len(patient_df) - 1, 1) * 3

    return {
        f"{prefix}n_windows": int(
            patient_df["window_days"].nunique()
        ),
        f"{prefix}mental_change_events": mental_changes,
        f"{prefix}medication_change_events": medication_changes,
        f"{prefix}comorbidity_change_events": comorbidity_changes,
        f"{prefix}trajectory_change_events": total_changes,
        f"{prefix}trajectory_increase_events": total_increases,
        f"{prefix}changed_domain_count": changed_domain_count,
        f"{prefix}trajectory_change_density": float(
            total_changes / possible_transitions
        ),
        f"{prefix}mental_max": float(mental.max()),
        f"{prefix}medication_max": float(medication.max()),
        f"{prefix}comorbidity_max": float(comorbidity.max()),
        f"{prefix}trajectory_total_max": float(
            mental.max() + medication.max() + comorbidity.max()
        ),
        f"{prefix}profile_domain_count": profile_domain_count,
        f"{prefix}all_three_domains_present": int(
            profile_domain_count == 3
        ),
        f"{prefix}flat_trajectory": int(total_changes == 0),
    }


def create_longitudinal_patient_summary(
    longitudinal_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    analysis_df = longitudinal_df[
        longitudinal_df["window_days"].between(
            0,
            ANALYSIS_MAX_DAYS,
            inclusive="both",
        )
    ].copy()

    for patient_id, patient_raw in analysis_df.groupby("Patient_ID"):
        patient_raw = patient_raw.sort_values("window_days")

        first_msk_day = derive_first_msk_day(patient_raw)
        full_summary = summarize_trajectory_block(patient_raw)

        if (
            pd.notna(first_msk_day)
            and 0 <= first_msk_day <= ANALYSIS_MAX_DAYS
        ):
            pre_msk_df = patient_raw[
                patient_raw["window_days"] < float(first_msk_day)
            ].copy()
            msk_within_1yr = 1
        else:
            pre_msk_df = patient_raw.copy()
            msk_within_1yr = 0

        pre_summary = summarize_trajectory_block(
            pre_msk_df,
            prefix="pre_msk_",
        )

        aggregated = aggregate_patient_windows(patient_raw)

        if pre_msk_df.empty:
            pre_msk_mental_always_present = 0
            pre_msk_last_mental_positive = 0
        else:
            pre_msk_aggregated = aggregate_patient_windows(
                pre_msk_df
            )
            pre_msk_mental_always_present = int(
                (
                    pre_msk_aggregated["mental_frequency"]
                    > 0
                ).all()
            )
            pre_msk_last_mental_positive = int(
                float(
                    pre_msk_aggregated[
                        "mental_frequency"
                    ].iloc[-1]
                )
                > 0
            )

        rows.append({
            "Patient_ID": str(patient_id),
            "max_followup_days": float(
                aggregated["window_days"].max()
            ),
            "first_msk_day": first_msk_day,
            "msk_within_1yr": msk_within_1yr,
            "pre_msk_mental_always_present": (
                pre_msk_mental_always_present
            ),
            "pre_msk_last_mental_positive": (
                pre_msk_last_mental_positive
            ),
            **full_summary,
            **pre_summary,
        })

    return pd.DataFrame(rows)


def add_selection_variables(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["Recent_30d_mental_visit"] = safe_numeric_column(
        result,
        ["Recent_30d_mental_visit"],
    )
    result["Mental_Diagnosis_count"] = safe_numeric_column(
        result,
        ["Mental_Diagnosis_count"],
    )
    result["Multimorbidity_count"] = safe_numeric_column(
        result,
        ["Multimorbidity_count"],
    )
    result["n_mental_drugs"] = safe_numeric_column(
        result,
        [
            "n_mental_drugs",
            "Mental_related_medication_drug_count",
        ],
    )
    result["n_pain_drugs"] = safe_numeric_column(
        result,
        [
            "n_pain_drugs",
            "Pain_related_medication_drug_count",
        ],
    )
    result["med_addition_count"] = safe_numeric_column(
        result,
        ["med_addition_count"],
    )
    result["med_discontinuation_count"] = safe_numeric_column(
        result,
        ["med_discontinuation_count"],
    )
    result["raw_missing_feature_count"] = safe_numeric_column(
        result,
        ["raw_missing_feature_count"],
    )
    result["model_nonzero_feature_count"] = safe_numeric_column(
        result,
        ["model_nonzero_feature_count"],
    )

    result["medication_richness"] = (
        result["n_mental_drugs"]
        + result["n_pain_drugs"]
        + result["med_addition_count"]
        + result["med_discontinuation_count"]
    )

    result["clinical_richness"] = (
        result["Mental_Diagnosis_count"]
        + result["medication_richness"]
        + result["Multimorbidity_count"]
    )

    return result


def prepare_candidate_table(
    high: pd.DataFrame,
    intermediate: pd.DataFrame,
    low: pd.DataFrame,
    longitudinal_summary: pd.DataFrame,
) -> pd.DataFrame:
    all_groups = pd.concat(
        [high, intermediate, low],
        ignore_index=True,
    )

    all_groups = all_groups.merge(
        longitudinal_summary,
        on="Patient_ID",
        how="left",
        validate="one_to_one",
    )

    for column in longitudinal_summary.columns:
        if column == "Patient_ID":
            continue

        all_groups[column] = pd.to_numeric(
            all_groups[column],
            errors="coerce",
        ).fillna(0)

    return add_selection_variables(all_groups)


def assign_rule_based_candidates(
    all_groups: pd.DataFrame,
) -> pd.DataFrame:
   
    candidate_frames: list[pd.DataFrame] = []

    good = all_groups[
        (all_groups["source_group"] == "High applicability")
        & (all_groups["prediction_type"] == "TP")
        & (all_groups["msk_within_1yr"] == 1)
        & (all_groups["pre_msk_mental_max"] >= 1)
        & (all_groups["pre_msk_medication_max"] >= 1)
        & (all_groups["pre_msk_n_windows"] >= GOOD_MIN_WINDOWS)
        & (
            (all_groups["pre_msk_mental_always_present"] == 1)
            | (all_groups["pre_msk_last_mental_positive"] == 1)
        )
        & (
            all_groups["pre_msk_medication_change_events"]
            >= 1
        )
    ].copy()

    good["mental_medication_change_events"] = (
        good["pre_msk_mental_change_events"]
        + good["pre_msk_medication_change_events"]
    )
    good["mental_medication_information"] = (
        good["pre_msk_mental_max"]
        + good["pre_msk_medication_max"]
    )

    strict_good = good[
        good["mental_medication_change_events"] >= 2
    ].copy()

    if len(strict_good) >= N_TRAJECTORY_SETS:
        good = strict_good

    good["trajectory_example"] = "Good example"
    good["selection_rule"] = (
        "High-applicability true positive with persistent or recent "
        "pre-MSK mental-health records and medication change before MSK"
    )

    good = good.sort_values(
        [
            "pre_msk_mental_always_present",
            "pre_msk_last_mental_positive",
            "pre_msk_medication_change_events",
            "pre_msk_mental_change_events",
            "mental_medication_information",
            "model_nonzero_feature_count",
            "pre_msk_n_windows",
            "raw_missing_feature_count",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
    )
    candidate_frames.append(good)

  
    normal_base = all_groups[
        (all_groups["source_group"] == "Intermediate applicability")
        & (all_groups["msk_within_1yr"] == 1)
        & (all_groups["pre_msk_mental_max"] >= 1)
        & (all_groups["pre_msk_medication_max"] >= 1)
        & (all_groups["pre_msk_n_windows"] >= NORMAL_MIN_WINDOWS)
    ].copy()

    normal_base["mental_medication_change_events"] = (
        normal_base["pre_msk_mental_change_events"]
        + normal_base["pre_msk_medication_change_events"]
    )
    normal_base["mental_medication_information"] = (
        normal_base["pre_msk_mental_max"]
        + normal_base["pre_msk_medication_max"]
    )

    normal = normal_base[
        normal_base["mental_medication_change_events"] == 1
    ].copy()

    if len(normal) < N_TRAJECTORY_SETS:
        normal = normal_base[
            normal_base["mental_medication_change_events"] <= 1
        ].copy()

    normal["trajectory_example"] = "Normal example"
    normal["selection_rule"] = (
        "Intermediate-applicability patient with complete mental-health and "
        "medication information but a nearly flat pre-MSK trajectory"
    )

    normal = normal.sort_values(
        [
            "mental_medication_change_events",
            "pre_msk_trajectory_change_events",
            "mental_medication_information",
            "pre_msk_n_windows",
            "model_nonzero_feature_count",
            "raw_missing_feature_count",
        ],
        ascending=[
            True,
            True,
            False,
            False,
            False,
            True,
        ],
    )
    candidate_frames.append(normal)

    bad1_base = all_groups[
        (all_groups["source_group"] == "Low applicability")
        & (all_groups["prediction_type"] == "FN")
        & (all_groups["msk_within_1yr"] == 1)
        & (
            all_groups["pre_msk_n_windows"]
            >= FN_MIN_PRE_MSK_WINDOWS
        )
    ].copy()

    bad1 = bad1_base[
        bad1_base["pre_msk_medication_max"] == 0
    ].copy()

    if len(bad1) < N_TRAJECTORY_SETS:
        bad1 = bad1_base[
            bad1_base["pre_msk_medication_max"] <= 1
        ].copy()

    bad1["trajectory_example"] = "Bad example 1"
    bad1["selection_rule"] = (
        "Low-applicability false negative with no or minimal observable "
        "medication information before MSK"
    )

    bad1 = bad1.sort_values(
        [
            "pre_msk_medication_max",
            "pre_msk_medication_change_events",
            "pre_msk_profile_domain_count",
            "pre_msk_trajectory_total_max",
            "model_nonzero_feature_count",
            "raw_missing_feature_count",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            False,
        ],
    )
    candidate_frames.append(bad1)


    bad2_base = all_groups[
        (all_groups["source_group"] == "Low applicability")
        & (all_groups["prediction_type"] == "FN")
        & (all_groups["msk_within_1yr"] == 1)
        & (
            all_groups["pre_msk_n_windows"]
            >= FN_MIN_PRE_MSK_WINDOWS
        )
        & (all_groups["pre_msk_mental_max"] <= 1)
        & (all_groups["pre_msk_mental_change_events"] == 0)
    ].copy()

    bad2 = bad2_base[
        bad2_base["pre_msk_medication_max"] > 0
    ].copy()

    if len(bad2) < N_TRAJECTORY_SETS:
        bad2 = bad2_base.copy()

    bad2["trajectory_example"] = "Bad example 2"
    bad2["selection_rule"] = (
        "Low-applicability false negative with very sparse observable "
        "mental-health information before MSK"
    )

    bad2 = bad2.sort_values(
        [
            "pre_msk_mental_max",
            "pre_msk_mental_change_events",
            "pre_msk_profile_domain_count",
            "pre_msk_trajectory_total_max",
            "pre_msk_medication_max",
            "model_nonzero_feature_count",
            "raw_missing_feature_count",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
            True,
            False,
        ],
    )
    candidate_frames.append(bad2)

    non_empty_frames = [
        frame
        for frame in candidate_frames
        if not frame.empty
    ]

    if not non_empty_frames:
        raise RuntimeError(
            "No trajectory candidates were identified."
        )

    candidate_table = pd.concat(
        non_empty_frames,
        ignore_index=True,
    )

    candidate_table["candidate_rank"] = (
        candidate_table
        .groupby("trajectory_example")
        .cumcount()
        + 1
    )

    return candidate_table



def demographic_match_penalty(
    candidate: pd.Series,
    already_selected: list[pd.Series],
) -> float:
    if not already_selected:
        return 0.0

    selected_ages = [
        float(row["Age"])
        for row in already_selected
        if pd.notna(row.get("Age"))
    ]
    selected_sexes = [
        str(row["Sex"])
        for row in already_selected
        if pd.notna(row.get("Sex"))
    ]

    age_penalty = 0.0

    if selected_ages and pd.notna(candidate.get("Age")):
        age_penalty = (
            abs(
                float(candidate["Age"])
                - float(np.mean(selected_ages))
            )
            / 20.0
        )

    sex_penalty = 0.0

    if selected_sexes and pd.notna(candidate.get("Sex")):
        modal_sex = pd.Series(selected_sexes).mode().iloc[0]

        if str(candidate["Sex"]) != modal_sex:
            sex_penalty = 0.5

    return age_penalty + sex_penalty


def select_multiple_trajectory_sets(
    candidate_table: pd.DataFrame,
    n_sets: int = N_TRAJECTORY_SETS,
) -> pd.DataFrame:
    used_patient_ids: set[str] = set()
    selected_rows: list[pd.Series] = []

    for set_number in range(1, n_sets + 1):
        current_set: list[pd.Series] = []

        for trajectory_type in TRAJECTORY_TYPES:
            available = candidate_table[
                (
                    candidate_table["trajectory_example"]
                    == trajectory_type
                )
                & (
                    ~candidate_table["Patient_ID"].isin(
                        used_patient_ids
                    )
                )
            ].copy()

            if available.empty:
                print(
                    "Warning: no unused candidate for "
                    f"{trajectory_type}, set {set_number}."
                )
                continue

            available["demographic_match_penalty"] = available.apply(
                lambda row: demographic_match_penalty(
                    row,
                    current_set,
                ),
                axis=1,
            )

            available["final_rank_score"] = (
                available["candidate_rank"]
                + available["demographic_match_penalty"]
            )

            chosen = available.sort_values(
                [
                    "final_rank_score",
                    "candidate_rank",
                ],
                ascending=[
                    True,
                    True,
                ],
            ).iloc[0].copy()

            chosen["trajectory_set"] = set_number

            current_set.append(chosen)
            selected_rows.append(chosen)
            used_patient_ids.add(
                str(chosen["Patient_ID"])
            )

    if not selected_rows:
        return pd.DataFrame()

    return pd.DataFrame(
        selected_rows
    ).reset_index(drop=True)



def create_final_selected_patient_set(
    all_groups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the final manually selected representative patient set.

    Patients were chosen after reviewing the automatically generated candidate
    tables and trajectories. This preserves the transparent workflow:
    automatic candidate generation first, followed by manual representative
    selection.
    """
    rows: list[pd.Series] = []

    for trajectory_example in TRAJECTORY_TYPES:
        patient_id = FINAL_SELECTED_PATIENTS[
            trajectory_example
        ]

        matched = all_groups[
            all_groups["Patient_ID"].astype(str)
            == str(patient_id)
        ].copy()

        if matched.empty:
            raise ValueError(
                f"Final selected patient {patient_id} "
                f"({trajectory_example}) was not found in the "
                "three sensitivity-analysis CSV files."
            )

        selected = matched.iloc[0].copy()
        selected["trajectory_example"] = trajectory_example
        selected["trajectory_set"] = 1
        selected["manual_selection"] = 1
        selected["manual_selection_rationale"] = (
            FINAL_SELECTION_RATIONALE[
                trajectory_example
            ]
        )
        rows.append(selected)

    final_selected = pd.DataFrame(
        rows
    ).reset_index(drop=True)

    return final_selected


def create_final_selected_patient_review(
    final_selected: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a compact CSV documenting the final four manually selected patients.
    """
    columns = [
        "trajectory_set",
        "trajectory_example",
        "Patient_ID",
        "source_group",
        "prediction_type",
        "true_msk_1yr",
        "pred_prob",
        "Age",
        "Sex",
        "manual_selection",
        "manual_selection_rationale",
        "first_msk_day",
        "Recent_30d_mental_visit",
        "clinical_richness",
        "raw_missing_feature_count",
        "model_nonzero_feature_count",
        "pre_msk_n_windows",
        "pre_msk_mental_always_present",
        "pre_msk_last_mental_positive",
        "pre_msk_profile_domain_count",
        "pre_msk_mental_max",
        "pre_msk_medication_max",
        "pre_msk_comorbidity_max",
        "pre_msk_mental_change_events",
        "pre_msk_medication_change_events",
        "pre_msk_comorbidity_change_events",
        "pre_msk_trajectory_change_events",
        "pre_msk_trajectory_change_density",
    ]

    columns = [
        column
        for column in columns
        if column in final_selected.columns
    ]

    return final_selected[
        columns
    ].copy()


def create_candidate_review_table(
    candidate_table: pd.DataFrame,
    top_n_per_type: int = PATIENTS_PER_TYPE_TO_REVIEW,
) -> pd.DataFrame:
    review = (
        candidate_table
        .groupby(
            "trajectory_example",
            group_keys=False,
        )
        .head(top_n_per_type)
        .copy()
    )

    columns = [
        "trajectory_example",
        "candidate_rank",
        "selection_rule",
        "Patient_ID",
        "source_group",
        "Age",
        "Sex",
        "true_msk_1yr",
        "pred_prob",
        "pred_label",
        "prediction_type",
        "prediction_error",
        "Recent_30d_mental_visit",
        "clinical_richness",
        "raw_missing_feature_count",
        "model_nonzero_feature_count",
        "n_windows",
        "max_followup_days",
        "first_msk_day",
        "profile_domain_count",
        "all_three_domains_present",
        "changed_domain_count",
        "trajectory_change_events",
        "trajectory_increase_events",
        "trajectory_change_density",
        "pre_msk_n_windows",
        "pre_msk_mental_always_present",
        "pre_msk_last_mental_positive",
        "pre_msk_profile_domain_count",
        "pre_msk_mental_max",
        "pre_msk_medication_max",
        "pre_msk_mental_change_events",
        "pre_msk_medication_change_events",
        "pre_msk_changed_domain_count",
        "pre_msk_trajectory_change_events",
        "pre_msk_trajectory_total_max",
        "flat_trajectory",
    ]

    columns = [
        column
        for column in columns
        if column in review.columns
    ]

    return review[columns].reset_index(drop=True)


def create_selected_set_review(
    selected_sets: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "trajectory_set",
        "trajectory_example",
        "candidate_rank",
        "selection_rule",
        "Patient_ID",
        "source_group",
        "Age",
        "Sex",
        "prediction_type",
        "true_msk_1yr",
        "pred_prob",
        "Recent_30d_mental_visit",
        "clinical_richness",
        "raw_missing_feature_count",
        "model_nonzero_feature_count",
        "n_windows",
        "first_msk_day",
        "profile_domain_count",
        "all_three_domains_present",
        "changed_domain_count",
        "trajectory_change_events",
        "trajectory_increase_events",
        "trajectory_change_density",
        "pre_msk_n_windows",
        "pre_msk_mental_always_present",
        "pre_msk_last_mental_positive",
        "pre_msk_profile_domain_count",
        "pre_msk_mental_max",
        "pre_msk_medication_max",
        "pre_msk_mental_change_events",
        "pre_msk_medication_change_events",
        "pre_msk_changed_domain_count",
        "pre_msk_trajectory_change_events",
        "pre_msk_trajectory_total_max",
        "flat_trajectory",
    ]

    columns = [
        column
        for column in columns
        if column in selected_sets.columns
    ]

    return selected_sets[columns].sort_values(
        [
            "trajectory_set",
            "trajectory_example",
        ]
    )



def add_first_msk_comorbidity_event(
    patient_data: pd.DataFrame,
    msk_day: float,
) -> pd.DataFrame:
    """
    Add First MSK to the comorbidity trajectory as a change event.

    The new MSK point is connected to the patient's previous comorbidity
    record. Its value is the last observed comorbidity count plus one.
    Mental-health and medication values are left missing at the MSK date
    unless an actual snapshot already exists there.
    """
    result = patient_data.copy().sort_values(
        "window_days"
    ).reset_index(drop=True)

    previous_rows = result[
        result["window_days"] < float(msk_day)
    ].copy()

    previous_values = pd.to_numeric(
        previous_rows["comorbidity_frequency"],
        errors="coerce",
    ).dropna()

    previous_comorbidity = (
        float(previous_values.iloc[-1])
        if not previous_values.empty
        else 0.0
    )

    msk_comorbidity = previous_comorbidity + 1.0

    exact_mask = np.isclose(
        result["window_days"].to_numpy(dtype=float),
        float(msk_day),
    )

    if exact_mask.any():
        result.loc[
            exact_mask,
            "comorbidity_frequency",
        ] = msk_comorbidity
    else:
        msk_row = pd.DataFrame({
            "window_days": [float(msk_day)],
            "mental_frequency": [np.nan],
            "medication_frequency": [np.nan],
            "comorbidity_frequency": [msk_comorbidity],
        })

        result = pd.concat(
            [result, msk_row],
            ignore_index=True,
        )

    return (
        result
        .sort_values("window_days")
        .reset_index(drop=True)
    )



def change_mask(
    values: pd.Series,
) -> np.ndarray:
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0).to_numpy()

    mask = np.zeros(
        len(numeric_values),
        dtype=bool,
    )

    if len(numeric_values) > 1:
        mask[1:] = ~np.isclose(
            numeric_values[1:],
            numeric_values[:-1],
        )

    return mask


def plot_patient_measure(
    axis: plt.Axes,
    patient_data: pd.DataFrame,
    measure: str,
    color: object,
    label: str,
    linestyle: str,
    marker_x_offset: float,
) -> None:
    measure_data = patient_data[
        ["window_days", measure]
    ].dropna().copy()

    if measure_data.empty:
        return

    x = measure_data["window_days"]
    y = measure_data[measure]

    # Lines remain at the true dates. Only markers receive a small
    # horizontal display offset so overlapping patients remain visible.
    marker_x = np.clip(
        x.to_numpy(dtype=float) + marker_x_offset,
        0,
        DISPLAY_MAX_DAYS,
    )

    axis.plot(
        x,
        y,
        linewidth=2,
        linestyle=linestyle,
        color=color,
        label=label,
        zorder=2,
    )

    changed = change_mask(y)
    unchanged = ~changed

    # Ordinary records: small upward triangles.
    axis.scatter(
        marker_x[unchanged],
        y[unchanged],
        marker="^",
        s=45,
        color=color,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    # Change events: large upward triangles.
    axis.scatter(
        marker_x[changed],
        y[changed],
        marker="^",
        s=150,
        color=color,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )


def calculate_shared_y_limit(
    plotted_patient_data: list[pd.DataFrame],
) -> int:
    """
    Calculate one shared integer Y-axis limit across all three panels.

    NaN values are expected at day 0 for medication and comorbidity because
    only the index mental diagnosis is explicitly added there. Therefore,
    missing values must be ignored when calculating the maximum.
    """
    measure_columns = [
        "mental_frequency",
        "medication_frequency",
        "comorbidity_frequency",
    ]

    observed_values: list[np.ndarray] = []

    for patient_df in plotted_patient_data:
        if patient_df.empty:
            continue

        values = (
            patient_df[measure_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(
                dtype=float
            )
            .ravel()
        )

        finite_values = values[
            np.isfinite(values)
        ]

        if finite_values.size > 0:
            observed_values.append(
                finite_values
            )

    if not observed_values:
        return 1

    maximum = float(
        np.max(
            np.concatenate(
                observed_values
            )
        )
    )

    return max(
        1,
        int(
            np.ceil(
                maximum
            )
        )
        + 1,
    )


def plot_trajectory_set(
    set_df: pd.DataFrame,
    longitudinal_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 6.5),
        sharex=True,
        sharey=True,
    )

    panel_definitions = [
        ("mental_frequency", "Mental health diagnosis count"),
        ("medication_frequency", "Medication"),
        ("comorbidity_frequency", "Comorbidity"),
    ]

    sorted_set = (
        set_df
        .set_index(
            "trajectory_example",
            drop=False,
        )
        .reindex(TRAJECTORY_TYPES)
        .dropna(subset=["Patient_ID"])
    )

    plot_records: list[
        tuple[
            pd.Series,
            pd.DataFrame,
            pd.DataFrame,
            float,
        ]
    ] = []

    for _, selected_patient in sorted_set.iterrows():
        patient_id = str(selected_patient["Patient_ID"])

        patient_raw = longitudinal_df[
            longitudinal_df["Patient_ID"] == patient_id
        ].copy()

        if patient_raw.empty:
            print(
                f"Warning: no longitudinal records for {patient_id}."
            )
            continue

        patient_raw = patient_raw[
            patient_raw["window_days"].between(
                0,
                ANALYSIS_MAX_DAYS,
                inclusive="both",
            )
        ].copy()

        patient_data = aggregate_patient_windows(patient_raw)

        msk_day = derive_first_msk_day(
            patient_raw,
            selected_patient,
        )

        # Keep only the pre-outcome trajectory, including the first MSK day.
        if (
            pd.notna(msk_day)
            and 0 <= float(msk_day) <= ANALYSIS_MAX_DAYS
        ):
            patient_data = patient_data[
                patient_data["window_days"] <= float(msk_day)
            ].copy()

        # Add only the index mental diagnosis at day 0.
        # Medication and comorbidity remain missing at day 0 unless they were
        # genuinely recorded there.
        if patient_data.empty:
            continue

        if not np.isclose(
            patient_data["window_days"].iloc[0],
            0,
        ):
            index_row = pd.DataFrame({
                "window_days": [0.0],
                "mental_frequency": [1.0],
                "medication_frequency": [np.nan],
                "comorbidity_frequency": [np.nan],
            })

            patient_data = pd.concat(
                [index_row, patient_data],
                ignore_index=True,
            ).sort_values(
                "window_days"
            ).reset_index(drop=True)
        else:
            patient_data.loc[
                patient_data["window_days"] == 0,
                "mental_frequency",
            ] = patient_data.loc[
                patient_data["window_days"] == 0,
                "mental_frequency",
            ].clip(lower=1)

        # Represent First MSK as a connected comorbidity change event.
        if (
            pd.notna(msk_day)
            and 0 <= float(msk_day) <= ANALYSIS_MAX_DAYS
        ):
            patient_data = add_first_msk_comorbidity_event(
                patient_data=patient_data,
                msk_day=float(msk_day),
            )

        plot_records.append(
            (
                selected_patient,
                patient_raw,
                patient_data,
                msk_day,
            )
        )

    shared_y_max = calculate_shared_y_limit(
        [record[2] for record in plot_records]
    )

    patient_colors: dict[str, object] = {}

    for axis, (_, title) in zip(
        axes,
        panel_definitions,
    ):
        axis.set_title(
            title,
            fontweight="bold",
        )
        axis.set_xlabel(
            "Days since first mental diagnosis"
        )
        axis.set_xlim(
            0,
            DISPLAY_MAX_DAYS,
        )
        axis.set_xticks(
            np.arange(
                0,
                DISPLAY_MAX_DAYS + 1,
                50,
            )
        )
        axis.set_ylim(
            0,
            shared_y_max,
        )
        axis.yaxis.set_major_locator(
            MaxNLocator(integer=True)
        )
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Frequency")

    for (
        selected_patient,
        patient_raw,
        patient_data,
        msk_day,
    ) in plot_records:
        patient_id = str(selected_patient["Patient_ID"])

        display_label = selected_patient["trajectory_example"]

        trajectory_type = selected_patient["trajectory_example"]
        color = PATIENT_COLORS[trajectory_type]
        linestyle = LINESTYLES[trajectory_type]
        marker_x_offset = MARKER_X_OFFSETS[trajectory_type]
        linestyle = LINESTYLES[
            selected_patient["trajectory_example"]
        ]
        marker_x_offset = MARKER_X_OFFSETS[
            selected_patient["trajectory_example"]
        ]

        has_msk_within_plot = (
            int(selected_patient.get("true_msk_1yr", 0)) == 1
            and pd.notna(msk_day)
            and 0 <= float(msk_day) <= ANALYSIS_MAX_DAYS
        )

        for axis, (measure, _) in zip(
            axes,
            panel_definitions,
        ):
            plot_patient_measure(
                axis=axis,
                patient_data=patient_data,
                measure=measure,
                color=color,
                label=display_label,
                linestyle=linestyle,
                marker_x_offset=marker_x_offset,
            )

            if has_msk_within_plot:
                axis.axvline(
                    float(msk_day),
                    linestyle=linestyle,
                    linewidth=1.6,
                    color=color,
                    alpha=0.75,
                    zorder=1,
                )

    set_number = int(
        set_df["trajectory_set"].iloc[0]
    )

    fig.suptitle(
        (
            "Representative 1-Year "
            f"Patient Trajectories – Set {set_number}"
        ),
        fontsize=16,
        fontweight="bold",
    )

    handles, labels = axes[0].get_legend_handles_labels()

    unique_legend: dict[str, object] = {}

    for handle, label in zip(handles, labels):
        if label not in unique_legend:
            unique_legend[label] = handle

    if unique_legend:
        fig.legend(
            unique_legend.values(),
            unique_legend.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=2,
            frameon=False,
        )

    fig.text(
        0.5,
        0.012,
        (
            "Small triangle = repeated record; "
            "large triangle = change event, including first MSK; "
            "patient-coloured vertical line = first MSK date. "
            "Markers are slightly offset horizontally for visibility."
        ),
        ha="center",
        fontsize=9.5,
    )

    plt.tight_layout(
        rect=[0, 0.17, 1, 0.92]
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_all_selected_sets(
    selected_sets: pd.DataFrame,
    longitudinal_df: pd.DataFrame,
) -> None:
    for set_number, set_df in selected_sets.groupby(
        "trajectory_set"
    ):
        plot_trajectory_set(
            set_df=set_df,
            longitudinal_df=longitudinal_df,
            output_path=(
                FIGURE_DIR
                / (
                    f"trajectory_set_{int(set_number):02d}"
                    "_all_patients.png"
                )
            ),
        )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    high, intermediate, low = load_patient_groups()

    longitudinal_df = load_longitudinal_data()
    longitudinal_df = add_trajectory_measures(longitudinal_df)

    longitudinal_summary = create_longitudinal_patient_summary(
        longitudinal_df
    )

    longitudinal_summary.to_csv(
        OUTPUT_DIR / "01_longitudinal_patient_summary.csv",
        index=False,
    )

    all_groups = prepare_candidate_table(
        high=high,
        intermediate=intermediate,
        low=low,
        longitudinal_summary=longitudinal_summary,
    )

    candidate_table = assign_rule_based_candidates(
        all_groups
    )

    candidate_table.to_csv(
        OUTPUT_DIR
        / "02_all_rule_based_trajectory_candidates.csv",
        index=False,
    )

    candidate_review = create_candidate_review_table(
        candidate_table
    )

    candidate_review.to_csv(
        OUTPUT_DIR
        / "03_trajectory_candidate_review.csv",
        index=False,
    )

    # Automatically selected candidate sets are preserved for transparency.
    selected_sets = select_multiple_trajectory_sets(
        candidate_table,
        n_sets=N_TRAJECTORY_SETS,
    )

    selected_sets.to_csv(
        OUTPUT_DIR
        / "04_selected_trajectory_sets_full.csv",
        index=False,
    )

    selected_set_review = create_selected_set_review(
        selected_sets
    )

    selected_set_review.to_csv(
        OUTPUT_DIR
        / "05_selected_trajectory_sets_review.csv",
        index=False,
    )

    automatic_selected_ids = (
        selected_sets["Patient_ID"]
        .astype(str)
        .unique()
    )

    automatic_selected_longitudinal = longitudinal_df[
        longitudinal_df["Patient_ID"].isin(
            automatic_selected_ids
        )
    ].copy()

    automatic_selected_longitudinal = (
        automatic_selected_longitudinal.merge(
            selected_sets[
                [
                    "Patient_ID",
                    "trajectory_set",
                    "trajectory_example",
                ]
            ],
            on="Patient_ID",
            how="left",
            validate="many_to_one",
        )
    )

    automatic_selected_longitudinal.to_csv(
        OUTPUT_DIR
        / "06_selected_patient_longitudinal_data.csv",
        index=False,
    )

    # Final four patients were manually selected after candidate review.
    final_selected = create_final_selected_patient_set(
        all_groups
    )

    final_selected.to_csv(
        OUTPUT_DIR
        / "07_final_manually_selected_patients_full.csv",
        index=False,
    )

    final_selected_review = (
        create_final_selected_patient_review(
            final_selected
        )
    )

    final_selected_review.to_csv(
        OUTPUT_DIR
        / "08_final_manually_selected_patients_review.csv",
        index=False,
    )

    final_selected_ids = (
        final_selected["Patient_ID"]
        .astype(str)
        .unique()
    )

    final_selected_longitudinal = longitudinal_df[
        longitudinal_df["Patient_ID"].isin(
            final_selected_ids
        )
    ].copy()

    final_selected_longitudinal = (
        final_selected_longitudinal.merge(
            final_selected[
                [
                    "Patient_ID",
                    "trajectory_set",
                    "trajectory_example",
                    "manual_selection_rationale",
                ]
            ],
            on="Patient_ID",
            how="left",
            validate="many_to_one",
        )
    )

    final_selected_longitudinal.to_csv(
        OUTPUT_DIR
        / "09_final_manually_selected_patient_longitudinal_data.csv",
        index=False,
    )

    # Plot only the final manually selected four patients.
    plot_all_selected_sets(
        selected_sets=final_selected,
        longitudinal_df=longitudinal_df,
    )

    print("\nAutomatically selected trajectory sets:")
    print(
        selected_set_review.to_string(
            index=False
        )
    )

    print("\nFinal manually selected patients:")
    print(
        final_selected_review.to_string(
            index=False
        )
    )

    print("\nSaved tables to:", OUTPUT_DIR)
    print("Saved figures to:", FIGURE_DIR)


if __name__ == "__main__":
    main()
