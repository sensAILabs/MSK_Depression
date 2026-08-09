# Mental Health to Musculoskeletal Disorder Prediction

This repository contains the analysis pipeline for predicting musculoskeletal (MSK) disorders following a recorded mental health diagnosis using longitudinal primary care electronic medical record (EMR) data.

The study uses data from the Canadian Primary Care Sentinel Surveillance Network (CPCSSN) and integrates demographic, clinical, comorbidity, and longitudinal medication information for machine-learning-based risk prediction.

The primary analysis code is contained in the `mental_msk_prediction/` package.

---

## Project Structure

```text
mental_msk_prediction/
│
├── cohort/
│   ├── __init__.py
│   └── build_cohort.py
│
├── config/
│   ├── __init__.py
│   ├── clinical_definitions.py
│   └── paths.py
│
├── features/
│   ├── __init__.py
│   └── build_analysis_df.py
│
├── modeling/
│   ├── __init__.py
│   ├── ml_modeling.py
│   ├── sensitivity_analysis.py
│   └── cluster_analysis.py
│
├── preprocessing/
│   ├── __init__.py
│   ├── clean_data.py
│   └── prepare_atc_definition.py
│
├── visualization/
│   ├── __init__.py
│   ├── cohort_eda.py
│   └── patient_trajectory.py
│
├── __init__.py
└── run_analysis.py
```

---

## Analysis Workflow

The analysis is organized as a sequential pipeline from raw data preprocessing to model development and additional analyses.

### 1. Data Preprocessing

**File:** `mental_msk_prediction/preprocessing/clean_data.py`

Cleans and standardizes the patient, encounter diagnosis, and medication datasets required for the downstream analysis.

The cleaned datasets are used as the common input for cohort construction and subsequent analyses.

---

### 2. Cohort Construction

**File:** `mental_msk_prediction/cohort/build_cohort.py`

Constructs the final study cohort based on the predefined clinical and temporal eligibility criteria.

The study population consists of adults with a recorded mental health diagnosis. The first eligible mental health diagnosis serves as the index event for the longitudinal analysis.

Patients with a prior musculoskeletal diagnosis before the index mental health diagnosis are excluded.

---

### 3. ATC Definition Preparation

**File:** `mental_msk_prediction/preprocessing/prepare_atc_definition.py`

Performs deterministic Anatomical Therapeutic Chemical (ATC) code quality control using medication records from patients in the final study cohort.

The script:

- restricts medication records to final-cohort patients;
- identifies calculated medication codes explicitly recorded as ATC codes;
- derives ATC Level 3 groups;
- identifies complete seven-character ATC codes;
- summarizes observed ATC Level 3 medication groups;
- summarizes prespecified study medication groups; and
- generates medication-definition and quality-control tables used before feature engineering.

Study ATC definitions are specified centrally in `clinical_definitions.py`.

---

### 4. Feature Engineering

**File:** `mental_msk_prediction/features/build_analysis_df.py`

Constructs the patient-level analysis dataset used for prediction.

Predictor domains include:

- demographic characteristics;
- mental health diagnosis information;
- pre-existing clinical conditions;
- multimorbidity information;
- mental-health-related medication use;
- pain-related medication use; and
- longitudinal medication features.

Clinical events and medication information are restricted according to the study timeline to avoid incorporating information occurring after the prediction endpoint.

---

### 5. Cohort Descriptive Analysis

**File:** `mental_msk_prediction/visualization/cohort_eda.py`

Generates descriptive analyses and visualizations of the final study cohort.

These analyses are used to characterize the study population and examine the distributions of relevant demographic, clinical, and medication variables.

---

### 6. Machine-Learning Modeling

**File:** `mental_msk_prediction/modeling/ml_modeling.py`

Develops and evaluates supervised machine-learning models for prediction of MSK outcomes.

The modeling workflow evaluates:

- Logistic Regression;
- Random Forest; and
- Extreme Gradient Boosting (XGBoost).

Model performance is evaluated using multiple classification metrics, including discrimination and threshold-based performance measures.

The modeling workflow also includes model interpretation and comparison of predictor sets.

---

### 7. Sensitivity Analysis

**File:** `mental_msk_prediction/modeling/sensitivity_analysis.py`

Performs additional analyses to evaluate the robustness of model performance and investigate differences across patient subgroups and prediction settings.

---

### 8. Cluster Analysis

**File:** `mental_msk_prediction/modeling/cluster_analysis.py`

Uses unsupervised clustering to investigate heterogeneity among patients who are more difficult for the prediction model to classify.

---

### 9. Patient Trajectory Visualization

**File:** `mental_msk_prediction/visualization/patient_trajectory.py`

Generates longitudinal patient trajectory visualizations illustrating clinical and medication events during the study timeline.

These visualizations are used to demonstrate how longitudinal information is represented between the mental health index event and subsequent follow-up.

---

## Complete Pipeline

The full analysis is executed in the following order:

```text
clean_data
    ↓
build_cohort
    ↓
prepare_atc_definition
    ↓
build_analysis_df
    ↓
cohort_eda
    ↓
ml_modeling
    ↓
sensitivity_analysis
    ↓
cluster_analysis
    ↓
patient_trajectory
```

The pipeline is controlled by:

```text
mental_msk_prediction/run_analysis.py
```

---

## Running the Analysis

Run commands from the repository root.

### Run the Complete Pipeline

```bash
python -m mental_msk_prediction.run_analysis
```

This executes all analysis stages in their predefined order.

### Start From a Specific Stage

For example:

```bash
python -m mental_msk_prediction.run_analysis \
    --start-at prepare_atc_definition
```

This skips earlier stages and executes `prepare_atc_definition` and all subsequent stages.

### Run Selected Stages Only

For example:

```bash
python -m mental_msk_prediction.run_analysis \
    --only ml_modeling sensitivity_analysis cluster_analysis
```

The selected stages are executed in their original pipeline order.

### Skip Selected Stages

For example:

```bash
python -m mental_msk_prediction.run_analysis \
    --skip cohort_eda patient_trajectory
```

All other stages are executed normally.

---

## Configuration

Shared project definitions and paths are centralized under:

```text
mental_msk_prediction/config/
```

### Clinical Definitions

Clinical definitions are stored in:

```text
mental_msk_prediction/config/clinical_definitions.py
```

This file contains study definitions used across the analysis pipeline, including clinical and medication-related definitions.

Centralizing these definitions ensures that the same criteria are used consistently across cohort construction, feature engineering, and downstream analyses.

### Project Paths

Project paths are defined in:

```text
mental_msk_prediction/config/paths.py
```

Paths are constructed relative to the repository rather than using machine-specific absolute paths.

This allows the code to be transferred between computing environments without modifying hard-coded local directories.

---

## Data Directory

The analysis expects the study data to be stored locally outside the tracked analysis package.

The expected general structure is:

```text
BiostatisticsPracticum_Depression/
│
├── data/
│   ├── raw/
│   └── cleaned/
│
└── mental_msk_prediction/
```

Raw source data are stored under:

```text
data/raw/
```

Cleaned intermediate datasets are stored under:

```text
data/cleaned/
```

The `data/` directory is excluded from version control.

---

## Data Availability

Patient-level CPCSSN data used for this study are **not included in this repository**.

CPCSSN data contain restricted, de-identified primary care electronic medical record information and cannot be distributed through this GitHub repository.

Users with appropriate access to the required source data should place the input files in the expected local data directories before running the pipeline.

---

## Analysis Outputs

Generated analysis outputs are stored under:

```text
mental_msk_prediction/results/
```

These may include:

- processed analysis datasets;
- cohort summaries;
- ATC quality-control tables;
- model performance results;
- sensitivity analysis results;
- clustering results; and
- figures and visualizations.

Generated outputs are excluded from version control and can be reproduced by running the corresponding analysis modules.

---

## Reproducibility

The project is organized as a modular Python package so that each major stage of the analysis can be run independently or as part of the complete pipeline.

The master runner:

```text
mental_msk_prediction/run_analysis.py
```

provides a single entry point for reproducing the analysis while preserving the intended order of data preprocessing, cohort construction, feature engineering, modeling, and downstream analyses.

---

## Study Data

This project uses data from the **Canadian Primary Care Sentinel Surveillance Network (CPCSSN)**.

CPCSSN provides longitudinal primary care electronic medical record data used for health services, epidemiological, and clinical research in Canada.

No patient-level CPCSSN data are distributed with this repository.

---

## Disclaimer

This repository is intended for research and reproducibility purposes.

The code assumes access to the required CPCSSN-derived input data and the data structure used for this study. Because the underlying patient-level data cannot be publicly distributed, the complete analysis cannot be reproduced from this repository alone without authorized access to the required source data.
