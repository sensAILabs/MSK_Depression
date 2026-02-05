# Environment: Python 3.12.2
import pandas as pd
from typing import Tuple, List
# from pathlib import Path
from pandas.errors import ParserError
import pandas as pd
from typing import Tuple, List
import io
pd.set_option("display.max_columns", None)

# First, determine how many lines of data are in the original file.
## Run code in terminal:
# wc -l original_data_file.txt
# 7706630 c4mmedication.csv
# 12471786 c4mencounterdiagnosis.csv
# 352162 c4mpatient.csv


# helper for input data reading with bad lines
def _try_parse(lines, sep, col_names):
    return pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=sep,
        header=None,
        names=col_names,
        engine="python"
    )

def _find_first_bad_line(buffer, sep, col_names):
    lo, hi = 0, len(buffer)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        try:
            _try_parse(buffer[:mid], sep, col_names)
            lo = mid
        except ParserError:
            hi = mid
    idx = hi - 1
    return idx, buffer[idx]

def read_csv_with_badlines(filepath: str, sep: str = "|", chunksize: int = 100_000) -> Tuple[pd.DataFrame, pd.DataFrame]:

    good_chunks: List[pd.DataFrame] = []
    bad_lines: List[str] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n")
        col_names = header.split(sep)
        n_cols = len(col_names)
        buffer = []
        
        for line in f:
            line = line.rstrip("\r\n")  

            if line.count(sep) == n_cols - 1 and (line.count('"') % 2 == 0):
                buffer.append(line)
            else:
                bad_lines.append(line)

            if len(buffer) >= chunksize:
                while buffer:
                    try:
                        chunk_df = _try_parse(buffer, sep, col_names)
                        good_chunks.append(chunk_df)
                        buffer.clear()
                    except ParserError:
                        bad_idx, bad_line = _find_first_bad_line(buffer, sep, col_names)
                        bad_lines.append(bad_line)
                        buffer.pop(bad_idx)  # remove the bad line and retry

        if buffer:
            while buffer:
                try:
                    chunk_df = _try_parse(buffer, sep, col_names)
                    good_chunks.append(chunk_df)
                    buffer.clear()
                except ParserError:
                    bad_idx, bad_line = _find_first_bad_line(buffer, sep, col_names)
                    bad_lines.append(bad_line)
                    buffer.pop(bad_idx)

    df_good = pd.concat(good_chunks, ignore_index=True) if good_chunks else pd.DataFrame(columns=col_names)
    df_bad = pd.DataFrame({"raw_line": bad_lines})

    return df_good, df_bad



