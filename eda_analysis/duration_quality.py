from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def stopdate_missing_rate_among_med_users(df, label):
    med_users = df["last_drug_end"].notna()
    total_med_users = med_users.sum()
    if total_med_users == 0:
        print(f"[{label}] No medication users found.")
        return
    missing_rate = (
        df.loc[med_users, "last_stop"].isna().mean() * 100
    )
    print(f"[{label}] StopDate missing rate among medication users: "
          f"{missing_rate:.2f}% (n={total_med_users})")
