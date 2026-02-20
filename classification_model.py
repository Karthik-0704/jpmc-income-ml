# -*- coding: utf-8 -*-
"""
Classification_Model.py
Production-Ready CLI Pipeline for Census Segmentation & Classification
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from IPython.display import display, HTML

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
# ==========================================================
# 1. ARGUMENT PARSER SETUP
# ==========================================================
parser = argparse.ArgumentParser(description="Census Classification Pipeline")
parser.add_argument("--pipeline", type=int, choices=[1, 2], default=1, 
                    help="1: Engineered Hybrid Model (32 feats) | 2: Baseline Model (40 feats)")
parser.add_argument("--algo", type=str, choices=["xgb", "rf", "logreg", "all"], default="all", 
                    help="Algorithm to train (xgb, rf, logreg, or all)")
parser.add_argument("--cv", action="store_true", 
                    help="Trigger Cross-Validation mode (skips standard test eval)")
parser.add_argument("--verbose", action="store_true", 
                    help="Print all preprocessing metadata and shapes")
parser.add_argument("--fast", action="store_true", 
                    help="Sample 10% of data for fast pipeline testing")
parser.add_argument("--save_plots", action="store_true", 
                    help="Silent mode: Save plots/metrics without displaying or printing them")

args, unknown = parser.parse_known_args()

VERBOSE = args.verbose
FAST_MODE = args.fast
SAVE_PLOTS = args.save_plots

# ==========================================================
# HEADLESS PLOTTING FIX & DIRECTORY SETUP
# ==========================================================
RESULTS_DIR = "results_classification"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

import matplotlib
if SAVE_PLOTS:
    matplotlib.use('Agg') # Forces non-interactive backend to prevent Colab/Server crashes
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 2. CONSOLE HEADERS
# ==========================================================
print("\n" + "="*71)
if args.pipeline == 1:
    print("[ RUNNING CUSTOM ENGINEERED HYBRID MODEL ]")
    print("="*71)
    print("Dataset:   Optimized Hybrid Pipeline")
    print("Features:  32 (Engineered)")
    print("Objective: Training the fully optimized custom classification model.")
    print("           This pipeline aggressively reduces the original 40 features")
    print("           down to 32 by eliminating near-zero variance noise, resolving")
    print("           collinearity overlaps, and safely transforming skewed data.\n")
else:
    print("[ RUNNING BASELINE CLASSIFICATION MODEL ]")
    print("="*71)
    print("Dataset:   Original Raw Data")
    print("Features:  40 (Unoptimized)")
    print("Objective: Establishing a foundational performance baseline using the")
    print("           raw, unengineered census dataset as a control metric.\n")
print("-"*71 + "\n")

# ==========================================================
# 3. DATA LOADING & CLEANING
# ==========================================================
if VERBOSE: print("[INFO] Loading dataset...")

PROJECT_DIR = "./"
if os.path.exists(PROJECT_DIR):
    os.chdir(PROJECT_DIR)

COLUMNS_FILE = "census-bureau.columns"
DATA_FILE = "census-bureau.data"

with open(COLUMNS_FILE, "r", encoding="utf-8") as f:
    cols = [line.strip() for line in f if line.strip()]

df = pd.read_csv(
    DATA_FILE,
    header=None,
    names=cols,
    sep=",",
    na_values=["?", "NA", "NAN", "NaN", "na", "nan", ""],
    skipinitialspace=True,
    engine="python"
)

if VERBOSE:
    print(f"[INFO] Loaded raw dataset. Shape: {df.shape}")

# ==========================================================
# PREPROCESSING: FEATURE CLASSIFICATION
# ==========================================================
TARGET_COL = "label"
WEIGHT_COL = "weight"

feature_cols = [c for c in df.columns if c not in [TARGET_COL, WEIGHT_COL]]

numeric_cols = []
categorical_cols = []
numeric_coverage = {}

for c in feature_cols:
    coerced = pd.to_numeric(df[c], errors="coerce")
    coverage = coerced.notna().mean()
    numeric_coverage[c] = coverage
    if coverage >= 0.95:
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

n_rows = len(df)

num_cardinality = pd.DataFrame({
    "n_unique": df[numeric_cols].nunique(dropna=True),
    "unique_ratio": df[numeric_cols].nunique(dropna=True) / n_rows,
    "min": df[numeric_cols].min(),
    "max": df[numeric_cols].max(),
    "mean": df[numeric_cols].mean(),
    "std": df[numeric_cols].std()
}).sort_values("n_unique")

for col in numeric_cols:
    series = df[col]
    coerced = pd.to_numeric(series, errors="coerce")
    int_like = False
    if coerced.notna().any():
        int_like = np.all(np.isclose(coerced.dropna() % 1, 0))

cat_cardinality = pd.DataFrame({
    "n_unique": df[categorical_cols].nunique(dropna=True),
    "unique_ratio": df[categorical_cols].nunique(dropna=True) / n_rows,
}).sort_values("n_unique", ascending=False)

NUMERIC_FEATURES = [
    "age", "wage per hour", "capital gains", "capital losses",
    "dividends from stocks", "num persons worked for employer",
    "weeks worked in year", "year"
]

CATEGORICAL_FEATURES = [
    "class of worker", "detailed industry recode", "detailed occupation recode",
    "education", "enroll in edu inst last wk", "marital stat",
    "major industry code", "major occupation code", "race",
    "hispanic origin", "sex", "member of a labor union",
    "reason for unemployment", "full or part time employment stat",
    "tax filer stat", "region of previous residence",
    "state of previous residence", "detailed household and family stat",
    "detailed household summary in household", "migration code-change in msa",
    "migration code-change in reg", "migration code-move within reg",
    "live in this house 1 year ago", "migration prev res in sunbelt",
    "family members under 18", "country of birth father",
    "country of birth mother", "country of birth self", "citizenship",
    "own business or self employed", "fill inc questionnaire for veteran's admin",
    "veterans benefits"
]

if VERBOSE:
    print(f"[INFO] Feature classification complete: {len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical.")

# ==========================================================
# PREPROCESSING: MISSING VALUE IMPUTATION
# ==========================================================
missing_summary = df.isna().mean().sort_values(ascending=False)

categorical_missing_cols = df[CATEGORICAL_FEATURES].columns[
    df[CATEGORICAL_FEATURES].isna().any()
].tolist()

exclude_cols = ["hispanic origin", "country of birth self"]
cols_to_impute_unknown = [col for col in categorical_missing_cols if col not in exclude_cols]

df[cols_to_impute_unknown] = df[cols_to_impute_unknown].fillna("NA")

mask_missing = (df["hispanic origin"].isna() | df["country of birth self"].isna())
ct = pd.crosstab(df["country of birth self"], df["hispanic origin"], dropna=True)

deterministic_map = {}
for birthplace, row in ct.iterrows():
    nonzero = row[row > 0]
    if len(nonzero) == 1:
        hispanic_value = nonzero.index[0]
        deterministic_map[birthplace] = hispanic_value

mask = (df["hispanic origin"].isna() & df["country of birth self"].isin(deterministic_map.keys()))
df.loc[mask, "hispanic origin"] = df.loc[mask, "country of birth self"].map(deterministic_map)

df["hispanic origin"] = df["hispanic origin"].fillna("Unknown")
df["country of birth self"] = df["country of birth self"].fillna("Unknown")

if VERBOSE:
    print(f"[INFO] Missing values imputed. Filled {len(cols_to_impute_unknown)} generic features with 'NA'.")
    print(f"[INFO] Applied deterministic mapping for 'hispanic origin' based on 'country of birth'.")

# ==========================================================
# PREPROCESSING: DATA CLEANSING & TARGET MAPPING
# ==========================================================
df["label"] = df["label"].str.strip()
df["income_binary"] = df["label"].map({"- 50000.": 0, "50000+.": 1})

initial_rows = len(df)
df = df.drop_duplicates()

if VERBOSE:
    print(f"[INFO] Target label converted to binary classification.")
    print(f"[INFO] Dropped {initial_rows - len(df)} duplicate rows. New shape: {df.shape}")

for col in CATEGORICAL_FEATURES:
    if df[col].dtype.name in ["object", "category"]:
        df[col] = df[col].str.strip()

NUMERIC_CODED_CATEGORICAL = [
    "detailed industry recode", "detailed occupation recode",
    "own business or self employed", "veterans benefits"
]

for col in CATEGORICAL_FEATURES:
    df[col] = df[col].astype("category")

for col in NUMERIC_CODED_CATEGORICAL:
    df[col] = df[col].astype("category")

if VERBOSE:
    print("[INFO] Standardized whitespace and applied 'category' data types.")

# Apply Fast Mode
if FAST_MODE:
    print("[WARNING] FAST MODE ENABLED: Running on a 10% random sample.\n")
    df = df.sample(frac=0.1, random_state=42)

# ==========================================================
# 4. FEATURE DEFINITIONS (ROUTING BASED ON --pipeline)
# ==========================================================
if args.pipeline == 1:
    # ----------------------------------------------------
    # PIPELINE 1: ENGINEERED 32 FEATURES
    # ----------------------------------------------------
    ZERO_INFLATED = ["wage per hour", "capital gains", "capital losses", "dividends from stocks"]
    for col in ZERO_INFLATED:
        df[f"{col}_nonzero"] = (df[col] > 0).astype(int)
        df[f"{col}_log1p"] = np.log1p(df[col])
    
    df["work_intensity"] = df["weeks worked in year"] / 52.0
    df["investment_income"] = df["capital gains"] + df["dividends from stocks"]
    df["investment_income_log1p"] = np.log1p(df["investment_income"])

    raw_gains = np.expm1(df["capital gains_log1p"])
    raw_dividends = np.expm1(df["dividends from stocks_log1p"])
    raw_losses = df["capital losses"]
    df["net_investment_income"] = raw_gains + raw_dividends - raw_losses
    df["net_investment_income_log1p"] = np.sign(df["net_investment_income"]) * np.log1p(np.abs(df["net_investment_income"]))

    df["marital_tax_combo"] = (df["marital stat"].astype(str) + "__" + df["tax filer stat"].astype(str)).astype("category")
    df["veteran_affiliation"] = (df["veterans benefits"].astype(str) + "__" + df["fill inc questionnaire for veteran's admin"].astype(str)).astype("category")
    business_map = {0: "Not_in_universe", 1: "Yes", 2: "No"}
    mapped_business = df["own business or self employed"].map(business_map).astype(str)
    df["class_business_combo"] = (df["class of worker"].astype(str) + "__" + mapped_business).astype("category")

    num_cols = ['age', 'wage per hour', 'num persons worked for employer', 'weeks worked in year',  "capital gains", "capital losses", "dividends from stocks"]
    cat_cols = ['detailed occupation recode', 'education', 'marital_tax_combo', 'major industry code',
                'race', 'hispanic origin', 'sex', 'member of a labor union', 'reason for unemployment',
                'full or part time employment stat', 'region of previous residence', 'state of previous residence',
                'detailed household summary in household', 'migration code-change in msa', 'migration code-change in reg',
                'migration code-move within reg', 'live in this house 1 year ago', 'migration prev res in sunbelt',
                'family members under 18', 'country of birth father', 'country of birth mother', 'country of birth self',
                'citizenship', 'class_business_combo', 'veteran_affiliation']
    CARDINALITY_THRESHOLD = 45

else:
    # ----------------------------------------------------
    # PIPELINE 2: BASELINE 40 FEATURES
    # ----------------------------------------------------
    num_cols = ["age", "wage per hour", "capital gains", "capital losses", "dividends from stocks", 
                "num persons worked for employer", "weeks worked in year", "year"]
    cat_cols = ["class of worker", "detailed industry recode", "detailed occupation recode", "education", 
                "enroll in edu inst last wk", "marital stat", "major industry code", "major occupation code", 
                "race", "hispanic origin", "sex", "member of a labor union", "reason for unemployment", 
                "full or part time employment stat", "tax filer stat", "region of previous residence", 
                "state of previous residence", "detailed household and family stat", 
                "detailed household summary in household", "migration code-change in msa", 
                "migration code-change in reg", "migration code-move within reg", "live in this house 1 year ago",
                "migration prev res in sunbelt", "family members under 18", "country of birth father",
                "country of birth mother", "country of birth self", "citizenship", "own business or self employed", 
                "fill inc questionnaire for veteran's admin", "veterans benefits"]
    CARDINALITY_THRESHOLD = 40

# ==========================================================
# 5. PREPROCESSING & SPLITTING
# ==========================================================
for col in cat_cols:
    df[col] = df[col].astype("category")

X = df[num_cols + cat_cols].copy()
y = df["income_binary"].astype(int)
w = df["weight"].astype(float)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, stratify=y, random_state=42
)

if VERBOSE:
    print(f"[INFO] Train split: {X_train.shape[0]:,} rows | Test split: {X_test.shape[0]:,} rows")

# ==========================================================
# 6. ALGORITHM DEFINITIONS
# ==========================================================
weight_negative = np.sum(w_train[y_train == 0])
weight_positive = np.sum(w_train[y_train == 1])
xgb_scale_weight = weight_negative / weight_positive

models_to_run = {}

if args.algo in ["logreg", "all"]:
    models_to_run["Logistic Regression"] = {
        "Unbalanced": LogisticRegression(C=0.1, class_weight=None, max_iter=3000, random_state=42),
        "Balanced": LogisticRegression(C=10.0, class_weight="balanced", max_iter=3000, random_state=42)
    }
if args.algo in ["rf", "all"]:
    models_to_run["Random Forest"] = {
        "Unbalanced": RandomForestClassifier(max_depth=None, min_samples_leaf=2, n_estimators=500, class_weight=None, n_jobs=-1, random_state=42),
        "Balanced": RandomForestClassifier(max_depth=None, min_samples_leaf=2, n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42)
    }
if args.algo in ["xgb", "all"]:
    models_to_run["XGBoost"] = {
        "Unbalanced": XGBClassifier(colsample_bytree=0.8, learning_rate=0.05, max_depth=6, n_estimators=500, subsample=0.8, scale_pos_weight=1, tree_method="hist", eval_metric="auc", n_jobs=-1, random_state=42),
        "Balanced": XGBClassifier(colsample_bytree=1.0, learning_rate=0.05, max_depth=4, n_estimators=500, subsample=0.8, scale_pos_weight=xgb_scale_weight, tree_method="hist", eval_metric="auc", n_jobs=-1, random_state=42)
    }

# ==========================================================
# 7. EXECUTION ENGINE (CV vs STANDARD)
# ==========================================================
all_results = []
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for algo_name, modes in models_to_run.items():
    print(f"\n[{algo_name.upper()}]")
    print("="*50)

    if algo_name == "XGBoost":
      HIGH_CARD_COLS = [c for c in cat_cols if X_train[c].nunique() > 45]
      LOW_CARD_COLS = [c for c in cat_cols if c not in HIGH_CARD_COLS]

      preprocessor = ColumnTransformer([
          ("num", StandardScaler(), num_cols),
          ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW_CARD_COLS),
          ("cat_high", TargetEncoder(target_type='binary', smooth="auto"), HIGH_CARD_COLS)
      ])

    elif algo_name == "Random Forest" or algo_name == "Logistic Regression":
        HIGH_CARD_COLS = [c for c in cat_cols if X_train[c].nunique() > 40]
        LOW_CARD_COLS = [c for c in cat_cols if c not in HIGH_CARD_COLS]

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW_CARD_COLS),
            ("cat_high", TargetEncoder(target_type='binary', smooth="auto"), HIGH_CARD_COLS)
        ])

    else:
      pass
    
    for mode, model in modes.items():
        print(f" -> Training {mode}...")

        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        
        # Default value
        cv_result_str = "N/A"
        
        # --- PATH A: CROSS-VALIDATION MODE ---            
        if args.cv:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore") # Silences all deprecation/future warnings
                    try:
                        # Try modern scikit-learn syntax (>= 1.4)
                        cv_scores = cross_val_score(
                            pipe, X_train, y_train, cv=cv_strategy, scoring='roc_auc', 
                            params={'model__sample_weight': w_train}, n_jobs=1
                        )
                    except TypeError:
                        # Fallback for older scikit-learn environments (< 1.4)
                        cv_scores = cross_val_score(
                            pipe, X_train, y_train, cv=cv_strategy, scoring='roc_auc', 
                            fit_params={'model__sample_weight': w_train}, n_jobs=1
                        )
                
                cv_result_str = f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
                if not SAVE_PLOTS: 
                    print(f"    CV ROC-AUC: {cv_result_str}")
        
        # --- PATH B: STANDARD EVALUATION ---
        pipe.fit(X_train, y_train, model__sample_weight=w_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred, sample_weight=w_test)
        prec = precision_score(y_test, y_pred, sample_weight=w_test)
        rec = recall_score(y_test, y_pred, sample_weight=w_test)
        f1 = f1_score(y_test, y_pred, sample_weight=w_test)
        auc_score = roc_auc_score(y_test, y_prob, sample_weight=w_test)

        if not SAVE_PLOTS:
            print(f"  Mode: {mode} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} |  Test ROC-AUC: {auc_score:.4f} | F1: {f1:.4f}")
        
        if cv_result_str != "N/A":
            all_results.append({
                "Algorithm": algo_name, "Mode": mode, "CV ROC-AUC": cv_result_str, "Accuracy": acc, 
                "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc_score
                
            })

        else:
            all_results.append({
                "Algorithm": algo_name, "Mode": mode, "Accuracy": acc, 
                "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc_score
            })

        # --- PLOT BUSINESS DASHBOARD (For Balanced Tree Models Only) ---
        if mode == "Balanced" and algo_name in ["Random Forest", "XGBoost"]:
            print("    Generating Business Dashboards...")
            
            y_prob_0 = 1.0 - y_prob
            y_test_0 = (y_test == 0).astype(int)
            
            plt.figure(figsize=(16, 12))
            
            # PR Curve Class 1
            plt.subplot(2, 2, 1)
            precision_1, recall_1, _ = precision_recall_curve(y_test, y_prob, sample_weight=w_test)
            ap_1 = average_precision_score(y_test, y_prob, sample_weight=w_test)
            base_1 = sum(w_test[y_test == 1]) / sum(w_test)
            plt.plot(recall_1, precision_1, color='purple', lw=2.5, label=f'>$50k (AP={ap_1:.4f})')
            plt.plot([0, 1], [base_1, base_1], color='grey', lw=2, linestyle='--')
            plt.title(algo_name +  'PR Curve (>$50k Class)', fontweight='bold')
            plt.legend()

            # PR Curve Class 0
            plt.subplot(2, 2, 2)
            precision_0, recall_0, _ = precision_recall_curve(y_test_0, y_prob_0, sample_weight=w_test)
            ap_0 = average_precision_score(y_test_0, y_prob_0, sample_weight=w_test)
            base_0 = sum(w_test[y_test == 0]) / sum(w_test)
            plt.plot(recall_0, precision_0, color='crimson', lw=2.5, label=f'<$50k (AP={ap_0:.4f})')
            plt.plot([0, 1], [base_0, base_0], color='grey', lw=2, linestyle='--')
            plt.title(algo_name + ' PR Curve (<$50k Class)', fontweight='bold')
            plt.legend()

            # Gains Chart
            plt.subplot(2, 1, 2)
            gains_df = pd.DataFrame({'actual': y_test, 'prob': y_prob, 'weight': w_test}).sort_values(by='prob', ascending=False)
            cum_data = (gains_df['weight'].cumsum() / gains_df['weight'].sum()) * 100
            cum_pos = ((gains_df['actual'] * gains_df['weight']).cumsum() / (gains_df['actual'] * gains_df['weight']).sum()) * 100
            plt.plot(cum_data, cum_pos, color='green', lw=2.5, label='Model Gains')
            plt.plot([0, 100], [0, 100], color='grey', lw=2, linestyle='--')
            plt.title(f'{algo_name} Cumulative Gains Chart', fontweight='bold')
            plt.legend()
            
            plt.tight_layout()
            
            # ALWAYS save the plot
            plt.savefig(f"{RESULTS_DIR}/{algo_name.replace(' ', '_')}_Dashboard.png")
            
            # Only display if SAVE_PLOTS is False
            if not SAVE_PLOTS:
                plt.show()

# ==========================================================
# 8. FINAL METRICS EXPORT
# ==========================================================
if all_results:
    results_df = pd.DataFrame(all_results).round(4)
    
    # Save a permanent receipt regardless of save_plots arg
    results_df.to_csv(f"{RESULTS_DIR}/classification_metrics.txt", index=False, sep="\t")
    
    # Only print to console if not in silent mode
    if not SAVE_PLOTS:
        print("\n==========================================================")
        print("FINAL PERFORMANCE METRICS")
        print("==========================================================")
        print(results_df.to_string(index=False))
    
    if VERBOSE:
        print(f"\n[INFO] Metrics successfully saved to '{RESULTS_DIR}/classification_metrics.txt'")