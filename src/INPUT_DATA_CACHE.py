import pandas as pd
from pathlib import Path

# CACHE_DIR = Path("../data/cache")
USE_CACHE = True   # True when devoloping

# def load_patient():
#     if USE_CACHE and (CACHE_DIR / "patient.parquet").exists():
#         return pd.read_parquet(CACHE_DIR / "patient.parquet")

#     df = pd.read_csv(
#         "../data/output/patient_good.csv",
#         sep="|",
#         engine="python",
#         dtype=str,
#         keep_default_na=False
#     )
#     df.to_parquet(CACHE_DIR / "patient.parquet")
#     return df


# def load_medication():
#     if USE_CACHE and (CACHE_DIR / "medication.parquet").exists():
#         return pd.read_parquet(CACHE_DIR / "medication.parquet")

#     df = pd.read_csv(
#         "../data/output/medication_final.csv",
#         sep="|",
#         engine="python",
#         dtype=str,
#         keep_default_na=False
#     )
#     df.to_parquet(CACHE_DIR / "medication.parquet")
#     return df

# def load_encounter_dx():
#     if USE_CACHE and (CACHE_DIR / "encounter_dx.parquet").exists():
#         return pd.read_parquet(CACHE_DIR / "encounter_dx.parquet")

#     df = pd.read_csv(
#         "../data/output/encounter_dx_final.csv",
#         sep="|",
#         engine="python",
#         dtype=str,
#         keep_default_na=False
#     )
#     df.to_parquet(CACHE_DIR / "encounter_dx.parquet")
#     return df
def load_data(
    *,
    csv_path: str,
    parquet_path: str,
    sep: str = "|",
) -> pd.DataFrame:
    """
    Generic data loader with parquet caching.

    Logic:
      - If USE_CACHE=True and parquet exists -> read parquet
      - Else read csv, save to parquet, return df

    Parameters
    ----------
    csv_path : str
        Path to source CSV file.
    parquet_path : str
        Path to cached parquet file.
    sep : str
        CSV delimiter (default "|").
    """
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if USE_CACHE and parquet_path.exists():
        return pd.read_parquet(parquet_path)

    df = pd.read_csv(
        csv_path,
        sep=sep,
        engine="python",
        dtype=str,
        keep_default_na=False,
    )

    df.to_parquet(parquet_path)
    return df


# patient = load_data(
#     csv_path="../data/output/patient_good.csv",
#     parquet_path="../data/cache/patient.parquet",
# )

# medication = load_data(
#     csv_path="../data/output/medication_final.csv",
#     parquet_path="../data/cache/medication.parquet",
# )

# encounter_dx = load_data(
#     csv_path="../data/output/encounter_dx_final.csv",
#     parquet_path="../data/cache/encounter_dx.parquet",
# )
