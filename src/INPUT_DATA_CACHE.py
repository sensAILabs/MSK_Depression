import pandas as pd
from pathlib import Path

# CACHE_DIR = Path("../data/cache")
USE_CACHE = True   # True when devoloping

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


