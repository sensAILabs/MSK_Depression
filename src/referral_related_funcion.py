import pandas as pd
import matplotlib.pyplot as plt


# extract related referral record
def get_referrals_after_msk(df, first_date_column, referral_date, period):
    df = df.copy()
    df[first_date_column] = pd.to_datetime(df[first_date_column], errors="coerce")
    df[referral_date] = pd.to_datetime(df[referral_date], errors="coerce")
    period = pd.to_timedelta(period)
    df["time_diff"] = df[referral_date] - df[first_date_column]
    result = df[(df["time_diff"] >= pd.Timedelta(0)) & (df["time_diff"] <= period)]

    return result.drop(columns="time_diff")


# msk subtype
msk_subtype_dict = {
    "Spine": [(720, 724), (737, 738)],
    
    "Degenerative_joint": [(715, 715), (717, 719)],
    
    "Inflammatory_rheumatic": [(710, 710), (712, 712), (714, 714)],
    
    "Soft_tissue": [(726, 729)],
    
    "Bone_osteoporosis_deformity": [(730, 736)]
}

# classify MSK subtype based on DiagnosisCode_calc (ICD9)
def classify_msk_subtype(code):

    if pd.isna(code):
        return None

    try:
        icd3 = str(code).split(".")[0]
        icd3_num = int(icd3)
    except:
        return None

    for subtype, ranges in msk_subtype_dict.items():
        for start, end in ranges:
            if start <= icd3_num <= end:
                return subtype

    return None


#plot MSK subtype distribution
def plot_msk_subtype_distribution(
    df,
    subtype_col="msk_subtype",
    title="MSK Subtype Distribution",
    figsize=(8,5)
):
    subtype_dist = (
        df[subtype_col]
        .dropna()
        .value_counts(normalize=True)
        .mul(100)
    )

    plt.figure(figsize=figsize)
    ax = subtype_dist.plot(kind="bar")

    plt.ylabel("Percentage (%)")
    plt.xlabel("MSK Subtype")
    plt.title(title)
    plt.xticks(rotation=30)

    # annotate percentages
    for i, v in enumerate(subtype_dist):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")

    plt.tight_layout()
    plt.show()

    return subtype_dist
