Biostatistics Practicum – Data Cleaning Pipeline

This repository contains a practical data-cleaning pipeline for the Care4Minds/CPCSSN-style extracts.

The raw source files are pipe-delimited (|) and may contain malformed rows (“bad lines”) caused by embedded separators, unbalanced quotes, or extra trailing fields. The pipeline separates good/bad rows, repairs common issues using streaming text processing (no pandas for corrupted files), and merges everything back into final *_final.csv.

(2026.02.05)**Current final version data**: 
    └── output/
        ├── C4MPatient_clean_1_31.csv
        ├── C4MEncounterDiagnosis_clean_1_31.csv
        └── C4MMedication_clean_1_31.csv


Project layout:
BiostatisticsPracticum_Depression/
├── src/
│   ├── main.py
│   ├── pipeline.py
│   ├── INPUT_ORIGINAL_DATA_OUTPUT_GOODLINE_BADLINE.py
│   ├── MERGE_GOOD_BAD.py
│   ├── INPUT_DATA_CACHE.py
│   ├── DATA_CLEANING.py
│   ├── medication_classification.py
│   └── frequency_maps.py
└── data/
    ├── (raw input files not committed)
    └── output/
        ├── patient_good.csv
        ├── patient_bad.csv
        ├── medication_good.csv
        ├── medication_bad.csv
        ├── medication_bad_cleaned.csv
        ├── medication_bad_cleaned_filtered.csv
        ├── medication_bad_still_unreadable.csv
        ├── medication_bad_still_unreadable_fixed.csv
        ├── medication_final.csv
        ├── encounter_dx_good.csv
        ├── encounter_dx_bad.csv
        ├── encounter_dx_bad_cleaned.csv
        ├── encounter_dx_bad_cleaned_filtered.csv
        ├── encounter_dx_bad_still_unreadable.csv
        ├── encounter_dx_bad_still_unreadable_fixed.csv
        └── encounter_dx_final.csv

1) INPUT_ORIGINAL_DATA_OUTPUT_GOODLINE_BADLINE.py

Goal: Split a raw pipe-delimited file into:
	•	*_good.csv (rows that parse correctly)
	•	*_bad.csv (raw lines that fail basic checks)

Key function:
	•	read_csv_with_badlines(filepath, sep="|", chunksize=100_000) -> (df_good, df_bad)
	•	Reads as text line-by-line
	•	Uses basic heuristics + “robust parsing loop” to isolate truly bad rows
	•	Returns:
	•	df_good: a valid DataFrame
	•	df_bad: DataFrame with one column: raw_line

Outputs (typical):
	•	patient_good.csv, patient_bad.csv
	•	medication_good.csv, medication_bad.csv
	•	encounter_dx_good.csv, encounter_dx_bad.csv

⸻

2) MERGE_GOOD_BAD.py

Goal: Repair bad lines, filter lines still unreadable, fix “still-bad” special cases, then merge into final CSV.

This file contains multiple stages:

(A) Clean *_bad.csv into a parsable cleaned text file
	•	encounter_dx: remove quotes/backslashes, enforce line-ending rules, etc.
	•	medication: remove quotes/backslashes; fix known embedded separator case:
	•	|anxiety associated with depression → (anxiety associated with depression)

Outputs:
	•	*_bad_cleaned.csv

(B) Validate cleaned bad lines
Some “cleaned” lines may still have the wrong number of columns.
The pipeline typically:
	•	Extract still unreadable lines into *_bad_still_unreadable.csv
	•	Keep the remaining valid ones as *_bad_cleaned_filtered.csv

Outputs:
	•	*_bad_cleaned_filtered.csv
	•	*_bad_still_unreadable.csv

(C) Fix “still bad” lines (special-case patch)
You already wrote:
	•	fix_encounter_dx_still_bad_lines(...)
	•	If a line has more than expected columns (13), truncate extras.
	•	fix_medication_still_bad_lines(...)
	•	Drop known garbage “Confirmed with SDM pharmacy tech …”
	•	Remove trailing | if present

Outputs:
	•	*_bad_still_unreadable_fixed.csv

(D) Merge good + cleaned_bad_filtered + still_bad_fixed → final
You currently have merge_good_and_cleaned_bad(...) (2-way merge).
You plan to upgrade it to a 3-way merge:

Inputs:
	•	good with header: *_good.csv
	•	cleaned filtered without header: *_bad_cleaned_filtered.csv
	•	still-bad fixed without header: *_bad_still_unreadable_fixed.csv

Outputs:
	•	*_final.csv

⸻

3) INPUT_DATA_CACHE.py

Goal: Speed up development by caching big CSVs in parquet.

Contains:
	•	load_data(csv_path, parquet_path, *, sep="|", use_cache=True, ...)

Pattern:
	•	If use_cache and parquet exists → read_parquet
	•	Else → read_csv then to_parquet

This avoids re-reading multi-million-row CSVs every test run.

⸻

4) medication_classification.py

Goal: Classify each medication record into:
	•	acute
	•	chronic
	•	unknown

Core ideas:
	•	Primary rule: ATC prefix (Code_calc)
	•	Fallback (optional): keyword search over Name_calc / Reason
	•	Includes a vectorized function add_medication_classification(...) for speed
	•	Includes a quick sanity-check utility summarize_classification(...)

Output column:
	•	med_class (string)

⸻

5) frequency_maps.py

Goal: Store frequency mappings (dictionaries) used to standardize medication frequency fields, including:
	•	mapping Frequency → daily times, or
	•	mapping Frequency → Event_interval_days (interval in days)

This is kept separate so the main pipeline code stays clean.

⸻

6) DATA_CLEANING.py

Goal: Downstream data cleaning / feature creation after final CSVs exist:
	•	standardize missing values
	•	strip quotes from header
	•	deduplicate checks
	•	date standardization
	•	frequency to numeric / interval
	•	attach derived flags/tags (e.g., musculoskeletal disorders)

This is “stage 2” after *_final.csv is produced.

⸻

7) pipeline.py

Goal: Provide wrapper functions so main.py is short and readable.

Why this exists:
	•	Keeps main.py from calling 20+ small functions in a row
	•	Makes it easy for other people to understand the workflow:
	•	“one function = one stage”

Typical wrappers:
	•	“build final medication”
	•	“build final encounter_dx”
	•	“load + cache + clean + enrich”

⸻

8) main.py

Goal: The entry point that should stay minimal.

Best practice:
	•	Only call wrapper functions from pipeline.py
	•	Avoid dumping all low-level steps directly in main.py