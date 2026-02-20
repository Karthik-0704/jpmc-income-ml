# Income ML Pipeline: Classification and Population Segmentation

This repository contains two production-ready machine learning pipelines designed to:

- Classify high-income earners (>$50K) using advanced predictive models  
- Segment the U.S. population into interpretable socio-economic personas using weighted clustering  

Both pipelines support command-line execution, automated evaluation, and headless operation for server and production environments.

---
# 0. Environment Setup and Dependencies

## Python Version
Requires **Python 3.8 or higher**  
Recommended: **Python 3.8**

## Install Dependencies

A `requirements.txt` file is provided.

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Core packages used:
```bash
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
xgboost>=1.7.0
matplotlib>=3.5.0
seaborn>=0.12.0
ipython>=8.0.0
Jinja2
```
Required Input Files

Ensure the following files are present in the project root directory before running either pipeline:
```
census-bureau.data
census-bureau.columns
```

These files are required for: Dataset loading, Column mapping and Feature preprocessing

The code expects these files in the current working directory:

```
"./census-bureau.data"
"./census-bureau.columns"
```


## 1. Income Classification Pipeline (`classification_model.py`)

This script trains, cross-validates, and evaluates machine learning models to predict whether an individual belongs to the high-income (>$50k) bracket.  
It supports Baseline vs. Hybrid feature testing and multiple algorithm architectures.

---

### Execution

```bash
# Run all algorithms on the default Hybrid(Engineered) model with Cross-Validaton
python classification_model.py --pipeline 1 --algo all --cv --verbose

# Run all algorithms on the Baseline feature set with Cross-Validation
python classification_model.py --pipeline 2 --algo all --cv --verbose

# Run a fast 10% data test in headless mode using XGBoost
python classification_model.py --fast --algo xgb --save_plots
```

---

### CLI Arguments

`--pipeline {1, 2}`: Selects the feature engineering path.

- **1 (Default):** Uses the heavily optimized Hybrid feature set (32 features), which consolidates financial data and removes collinearity. This is the self-prepared model.

- **2:** Uses the raw Baseline data (40 features) for control testing. This is a comparison model for comparing with the default optimized model.

---

`--algo {xgb, rf, logreg, all}`: Selects the algorithm to train and run a standard testing.

Options include:

- XGBoost (`xgb`)
- Random Forest (`rf`)
- Logistic Regression (`logreg`)
- Sequentially running all three (`all`)

---

`--cv`: (Flag) In addition to the standard testing, runs a rigorous 5-Fold Stratified Cross-Validation returning the mean ROC-AUC score across all folds.

---

`--fast`: (Flag) Down-samples the dataset to a random 10% slice. Highly recommended for quickly testing code functionality or debugging before a full 190k+ row training run.

---

`--verbose`: (Flag) Prints metadata regarding the feature split, imputation mapping, and cardinality thresholds.

---

`--save_plots`: (Flag) Enables "Silent/Headless Mode." Suppresses console metric printing and interactive plot pop-ups, executing cleanly in the background.

---

### Generated Outputs and Visualizations

All metrics and visuals are saved automatically to the `results_classification/` directory. The pipeline exports a permanent receipt of all model performances (Accuracy, Precision, Recall, F1, ROC-AUC) to:

```
classification_metrics.txt
```

Visually, the script generates a Business Performance Dashboard for tree-based models (XGBoost and Random Forest). This dashboard includes isolated Precision-Recall (PR) Curves for both the >$50k and <$50k classes, alongside a Cumulative Gains Chart.

**Note:** Logistic Regression is utilized strictly as a linear baseline metric and does not generate visual dashboards.

---

## 2. Population Segmentation Pipeline (`segmentation_model.py`)

This script utilizes a weighted K-Means clustering algorithm to identify distinct socio-economic personas based on demographic, occupational, and financial data.

---

### Execution

Run the script using standard Python execution. By default, it runs interactively, printing reports to the console and displaying UI plot pop-ups.

```bash
# Standard execution
python segmentation_model.py --verbose

# Execution with verbose logging and headless/silent mode
python segmentation_model.py --verbose --save_plots
```

---

### CLI Arguments

`--verbose`: (Flag) Enables detailed terminal logging. It prints data shapes, feature cardinality checks, missing value imputation steps, and near-zero variance flags during the preprocessing phase.

---

`--save_plots`: (Flag) Enables "Silent/Headless Mode." By default, the script prints text reports to the console and halts execution to show interactive plot pop-ups. Passing this flag suppresses all console output and UI pop-ups, running the script silently in the background.

---

### Generated Outputs and Visualizations

Regardless of the execution mode, all artifacts are permanently saved to the `results_segmentation/` directory.

The pipeline generates:

- Elbow & Silhouette Evaluation Plot to justify the optimal number of clusters (K)

For each generated persona, the script outputs:

- Comprehensive CSV data slice  
- Detailed `.txt` socio-economic report  
- Visual Business Dashboard (`.png`)  

The dashboard contains four charts:

- Weighted Income Probability  
- Age Distribution (KDE)  
- Top 5 Occupations  
- Top 5 Education Levels  

## Extended Production Deployment and API Integration

A production-ready deployment of the income classification model is available as a containerized API, enabling real-time prediction. The deployment leverages Docker for containerization and AWS infrastructure for streamlined, model serving with MLFlow tracking enabled.

For implementation details, deployment architecture, and API usage, please refer to:

https://github.com/Karthik-0704/income_ml


