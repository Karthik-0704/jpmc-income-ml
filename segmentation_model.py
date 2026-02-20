# -*- coding: utf-8 -*-
"""
Segmentation_Model.py
Optimized Clustering Pipeline with Verbose Preprocessing Controls
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

# ==========================================================
# ARGUMENT PARSER SETUP
# ==========================================================
parser = argparse.ArgumentParser(description="Census Data Segmentation Pipeline")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output for preprocessing metadata.")
parser.add_argument("--save_plots", action="store_true", help="Suppress console display and popups; run in silent/headless mode")

args, unknown = parser.parse_known_args()

VERBOSE = args.verbose
SAVE_PLOTS = args.save_plots

# ==========================================================
# HEADLESS PLOTTING FIX & DIRECTORY SETUP
# ==========================================================
RESULTS_DIR = "results_segmentation"

import matplotlib
if SAVE_PLOTS:
    matplotlib.use('Agg') # Forces non-interactive backend to prevent Colab/Server crashes
import matplotlib.pyplot as plt
import seaborn as sns

# ALWAYS create the results directory
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if VERBOSE:
    print("==========================================================")
    print("[INFO] VERBOSE MODE ENABLED: Displaying preprocessing steps.")
    print("==========================================================\n")

# ==========================================================
# DATA LOADING
# ==========================================================

PROJECT_DIR = "./"
os.chdir(PROJECT_DIR)

COLUMNS_FILE = "census-bureau.columns"

with open(COLUMNS_FILE, "r", encoding="utf-8") as f:
    cols = [line.strip() for line in f if line.strip()]

DATA_FILE = "census-bureau.data"

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

for col in categorical_cols:
    series = df[col]


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
ct_prob = ct.div(ct.sum(axis=1), axis=0)

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

# ==========================================================
# MASTER FEATURE ENGINEERING & TRAIN-TEST SPLIT
# ==========================================================
ZERO_INFLATED_NUMERIC = ["wage per hour", "capital gains", "capital losses", "dividends from stocks"]
engineered_numeric = []

for col in ZERO_INFLATED_NUMERIC:
    nonzero_col = f"{col}_nonzero"
    log_col = f"{col}_log1p"
    if nonzero_col not in df.columns:
        df[nonzero_col] = (df[col] > 0).astype(int)
    if log_col not in df.columns:
        df[log_col] = np.log1p(df[col])
    engineered_numeric.extend([nonzero_col, log_col])

if "work_intensity" not in df.columns:
    df["work_intensity"] = df["weeks worked in year"] / 52.0
if "work_intensity" not in engineered_numeric:
    engineered_numeric.append("work_intensity")

if "investment_income" not in df.columns:
    df["investment_income"] = df["capital gains"] + df["dividends from stocks"]
if "investment_income_log1p" not in df.columns:
    df["investment_income_log1p"] = np.log1p(df["investment_income"])

for c in ["investment_income", "investment_income_log1p"]:
    if c not in engineered_numeric:
        engineered_numeric.append(c)

raw_gains = np.expm1(df["capital gains_log1p"]) if "capital gains_log1p" in df.columns else df["capital gains"]
raw_dividends = np.expm1(df["dividends from stocks_log1p"]) if "dividends from stocks_log1p" in df.columns else df["dividends from stocks"]
raw_losses = df["capital losses"]
df["net_investment_income"] = raw_gains + raw_dividends - raw_losses
df["net_investment_income_log1p"] = np.sign(df["net_investment_income"]) * np.log1p(np.abs(df["net_investment_income"]))

df["marital_tax_combo"] = (df["marital stat"].astype(str) + "__" + df["tax filer stat"].astype(str)).astype("category")
df["veteran_affiliation"] = (df["veterans benefits"].astype(str) + "__" + df["fill inc questionnaire for veteran's admin"].astype(str)).astype("category")

business_map = {0: "Not_in_universe", 1: "Yes", 2: "No"}
mapped_business = df["own business or self employed"].map(business_map).astype(str)
df["class_business_combo"] = (df["class of worker"].astype(str) + "__" + mapped_business).astype("category")

num_cols = [
    'age', 'wage per hour', 'num persons worked for employer',
    'weeks worked in year', "capital gains", "capital losses",
    "dividends from stocks","year", 'net_investment_income'
]

cat_cols = [
    'detailed occupation recode', 'education', 'marital_tax_combo', 'major industry code',
    'major occupation code', 'race', 'hispanic origin', 'sex', 'member of a labor union',
    'reason for unemployment', 'full or part time employment stat', 'region of previous residence',
    'state of previous residence', 'detailed household summary in household', 'migration code-change in msa',
    'migration code-change in reg', 'migration code-move within reg', 'live in this house 1 year ago',
    'migration prev res in sunbelt', 'family members under 18', 'country of birth father',
    'country of birth mother', 'country of birth self', 'citizenship', 'class_business_combo',
    'veteran_affiliation', "enroll in edu inst last wk"
]

MODEL_TARGET = "income_binary"
WEIGHT_COL = "weight"

X = df[num_cols + cat_cols].copy()
y = df[MODEL_TARGET].astype(int)
w = df[WEIGHT_COL].astype(float)

X_train_final, X_test_final, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, stratify=y, random_state=42
)

HIGH_CARD_COLS = [c for c in cat_cols if X_train_final[c].nunique() > 30]
LOW_CARD_COLS = [c for c in cat_cols if c not in HIGH_CARD_COLS]

if VERBOSE:
    print("[INFO] Feature Engineering complete. Added zero-inflated flags, work intensity, and composite categories.")
    print(f"[INFO] Split complete. Training Set: {X_train_final.shape[0]:,} rows | Testing Set: {X_test_final.shape[0]:,} rows.")
    print(f"[INFO] Cardinality Check: {len(HIGH_CARD_COLS)} High (>30), {len(LOW_CARD_COLS)} Low (<=30).")

# ==========================================================
# PRE-CLUSTERING ANALYSIS (VARIANCE & CORRELATION)
# ==========================================================
features_to_test = [
    'age', 'wage per hour', 'num persons worked for employer', 'net_investment_income',
    'weeks worked in year', "capital gains", "capital losses", "dividends from stocks",
    "year", 'major occupation code', 'education', 'marital_tax_combo', 'major industry code',
    'race', 'hispanic origin', 'sex', 'member of a labor union', 'reason for unemployment',
    'full or part time employment stat', 'region of previous residence', 'state of previous residence',
    'detailed household summary in household', 'migration code-change in msa', 'migration code-change in reg',
    'migration code-move within reg', 'live in this house 1 year ago', 'migration prev res in sunbelt',
    'family members under 18', 'country of birth father', 'country of birth mother', 'country of birth self',
    'citizenship', 'class_business_combo', 'veteran_affiliation', "enroll in edu inst last wk"
]

union_features = list(set(features_to_test))
df_cluster_test = X_train_final[union_features].copy()

if VERBOSE:
    print("\n[INFO] Running near-zero variance and collinearity checks...")
    low_variance_cols = []
    for col in df_cluster_test.columns:
        top_value_pct = df_cluster_test[col].value_counts(normalize=True).iloc[0] * 100
        if top_value_pct > 90.0:
            top_value_name = df_cluster_test[col].value_counts().index[0]
            print(f"       -> FLAG: '{col}' ({top_value_pct:.1f}% is '{top_value_name}')")
            low_variance_cols.append(col)

    num_cols_only = df_cluster_test.select_dtypes(include=[np.number])
    if not num_cols_only.empty:
        corr_matrix = num_cols_only.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper_tri.columns:
            for row in upper_tri.index:
                if upper_tri.loc[row, col] > 0.75:
                    print(f"       -> FLAG: '{row}' and '{col}' are highly correlated ({upper_tri.loc[row, col]:.2f})")

FINAL_CLUSTER_FEATURES = [
    'age', 'num persons worked for employer', 'net_investment_income', 'weeks worked in year',
    'major occupation code', 'education', 'marital_tax_combo', 'major industry code',
    'race', 'hispanic origin', 'sex', 'full or part time employment stat',
    'detailed household summary in household', 'family members under 18', 'citizenship',
    'class_business_combo', 'veteran_affiliation'
]

# ==========================================================
# CLUSTERING STEP 1: PREPROCESSING & EVALUATION
# ==========================================================
print("\n==========================================================")
print("CLUSTERING STEP 1: EVALUATION (K=2 to 12)")
print("==========================================================\n")

NUMERIC_CLUSTER_FEATS = [
    'age', 'num persons worked for employer', 'net_investment_income', 'weeks worked in year'
]

CATEGORICAL_CLUSTER_FEATS = [
    'major occupation code', 'education', 'marital_tax_combo', 'major industry code',
    'race', 'hispanic origin', 'sex', 'full or part time employment stat',
    'detailed household summary in household', 'citizenship', 'family members under 18',
    'class_business_combo', 'veteran_affiliation'
]

X_cluster_data = X_train_final[NUMERIC_CLUSTER_FEATS + CATEGORICAL_CLUSTER_FEATS].copy()
X_cluster_data['net_investment_income'] = np.sign(X_cluster_data['net_investment_income']) * \
                                          np.log1p(np.abs(X_cluster_data['net_investment_income']))

cluster_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_CLUSTER_FEATS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_CLUSTER_FEATS)
    ]
)

if VERBOSE:
    print("[INFO] Applied Signed Log Transformation to 'net_investment_income'.")
    print("[INFO] Transforming the data (StandardScaler for numbers, OneHotEncoder for text).")

X_cluster_scaled = cluster_preprocessor.fit_transform(X_cluster_data)

if VERBOSE:
    print(f"[INFO] Data transformed! Final matrix shape for clustering: {X_cluster_scaled.shape}")

print("Running K-Means configurations to calculate Inertia and Silhouette scores...")

X_sample = resample(X_cluster_scaled, n_samples=50000, random_state=42)

inertia_values = []
silhouette_values = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled, sample_weight=w_train)
    inertia_values.append(kmeans.inertia_)

    sample_preds = kmeans.predict(X_sample)
    sil_score = silhouette_score(X_sample, sample_preds)
    silhouette_values.append(sil_score)

    print(f"  -> Finished k={k} | Inertia: {kmeans.inertia_:,.0f} | Silhouette: {sil_score:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(k_values, inertia_values, marker='o', color='teal', lw=2.5, markersize=8)
axes[0].set_title('The Elbow Method (Inertia)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (Lower is better)', fontsize=12)
axes[0].set_xticks(k_values)
axes[0].grid(alpha=0.3)

axes[1].plot(k_values, silhouette_values, marker='o', color='coral', lw=2.5, markersize=8)
axes[1].set_title('Silhouette Score (Cluster Separation)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score (Higher is better)', fontsize=12)
axes[1].set_xticks(k_values)
axes[1].grid(alpha=0.3)

plt.tight_layout()

# ALWAYS save the plot
plt.savefig(f"{RESULTS_DIR}/Clustering_Elbow_Silhouette.png", bbox_inches='tight')

# Only show plot if NOT in save_plots (silent) mode
if not SAVE_PLOTS:
    plt.show()
else:
    print(f"\n[INFO] Saved KMeans Evaluation plot to {RESULTS_DIR}/Clustering_Elbow_Silhouette.png")

# ==========================================================
# CLUSTERING STEP 2: VISUAL BUSINESS REPORT
# ==========================================================
print("\n==========================================================")
print("CLUSTERING STEP 2: VISUAL BUSINESS REPORT (K=4)")
print("==========================================================\n")

OPTIMAL_K = 4
final_kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
final_kmeans.fit(X_cluster_scaled, sample_weight=w_train)

df_personas = X_train_final[NUMERIC_CLUSTER_FEATS + CATEGORICAL_CLUSTER_FEATS].copy()
df_personas['Cluster_ID'] = final_kmeans.labels_
df_personas['Weight'] = w_train
df_personas['Makes_Over_50k'] = y_train.values

cluster_sizes = df_personas.groupby('Cluster_ID')['Weight'].sum()
total_weight = cluster_sizes.sum()

def get_weighted_top_5_str(df, column, weight_col):
    # Added observed=False to silence the Pandas FutureWarning
    weighted_counts = df.groupby(column, observed=False)[weight_col].sum()
    total_weight = weighted_counts.sum()
    dist = (weighted_counts / total_weight * 100).sort_values(ascending=False)
    # Replaced HTML tags with plain text formatting
    lines = [f"{i+1}. {val} ({pct:.1f}%)" for i, (val, pct) in enumerate(dist.head(5).items())]
    return " | ".join(lines)

def display_persona_report(cluster_id):
    group = df_personas[df_personas['Cluster_ID'] == cluster_id].copy()

    pop_weight = cluster_sizes[cluster_id]
    pop_pct = (pop_weight / total_weight) * 100
    sample_count = len(group)

    weighted_income_sum = (group['Makes_Over_50k'] * group['Weight']).sum()
    income_prob = (weighted_income_sum / pop_weight) * 100

    def weighted_mean(series, weights):
        return (series * weights).sum() / weights.sum()

    def weighted_median(df, val_col, weight_col):
        df_sorted = df.sort_values(val_col)
        cumsum = df_sorted[weight_col].cumsum()
        cutoff = df_sorted[weight_col].sum() / 2.0
        return df_sorted[cumsum >= cutoff][val_col].iloc[0]

    sections = [
        ("1. Population Structure", [
            ("Population Weight", f"{pop_weight:,.0f} people (Weighted)"),
            ("Population Percent", f"{pop_pct:.1f}% of total population"),
            ("Raw Sample Count", f"{sample_count:,} observations")
        ]),
        ("2. Economic Indicators", [
            (">$50k Probability", f"{income_prob:.1f}%"),
            ("Net Investment Income", f"W.Mean: ${weighted_mean(group['net_investment_income'], group['Weight']):,.2f} | W.Median: ${weighted_median(group, 'net_investment_income', 'Weight'):,.2f}"),
            ("Weeks Worked/Year", f"W.Mean: {weighted_mean(group['weeks worked in year'], group['Weight']):.1f} | W.Median: {weighted_median(group, 'weeks worked in year', 'Weight'):.0f}"),
            ("Employment Intensity", get_weighted_top_5_str(group, 'full or part time employment stat', 'Weight'))
        ]),
        ("3. Demographics", [
            ("Age Profile", f"W.Mean: {weighted_mean(group['age'], group['Weight']):.1f} yrs | W.Median: {weighted_median(group, 'age', 'Weight'):.0f} yrs"),
            ("Sex Distribution", get_weighted_top_5_str(group, 'sex', 'Weight')),
            ("Race Distribution", get_weighted_top_5_str(group, 'race', 'Weight')),
            ("Citizenship Status", get_weighted_top_5_str(group, 'citizenship', 'Weight')),
            ("Veteran Status", get_weighted_top_5_str(group, 'veteran_affiliation', 'Weight'))
        ]),
        ("4. Employment Structure", [
            ("Career Class", get_weighted_top_5_str(group, 'class_business_combo', 'Weight')),
            ("Major Occupation", get_weighted_top_5_str(group, 'major occupation code', 'Weight')),
            ("Major Industry", get_weighted_top_5_str(group, 'major industry code', 'Weight')),
            ("Avg Company Size", f"{weighted_mean(group['num persons worked for employer'], group['Weight']):.1f} employees"),
            ("Education Profile", get_weighted_top_5_str(group, 'education', 'Weight'))
        ]),
        ("5. Household Structure", [
            ("Marital / Tax Status", get_weighted_top_5_str(group, 'marital_tax_combo', 'Weight')),
            ("Household Summary", get_weighted_top_5_str(group, 'detailed household summary in household', 'Weight')),
            ("Family Members <18", get_weighted_top_5_str(group, 'family members under 18', 'Weight'))
        ])
    ]

    # --- Build the text report ---
    report_lines = []
    report_lines.append(f"\n{'='*75}")
    report_lines.append(f"MARKETING PERSONA {cluster_id}: EXECUTIVE WEIGHTED PROFILE")
    report_lines.append(f"{'='*75}")

    for title, rows in sections:
        report_lines.append(f"\n{title.upper()}")
        report_lines.append("-" * 75)
        for metric, stat in rows:
            report_lines.append(f"{metric:<25} : {stat}")

    full_report_text = "\n".join(report_lines)

    # --- Save Reports to File (Always) ---
    text_filepath = f"{RESULTS_DIR}/Persona_{cluster_id}_Report.txt"
    csv_filepath = f"{RESULTS_DIR}/Persona_{cluster_id}_Data.csv"
    
    with open(text_filepath, "w", encoding="utf-8") as f:
        f.write(full_report_text)
        
    group.to_csv(csv_filepath, index=False)

    # --- Console Display Logic ---
    if not SAVE_PLOTS:
        print(full_report_text)
        print(f"\n[INFO] Generating Visual Data Dashboard for Persona {cluster_id}...")
    else:
        print(f"[INFO] Persona {cluster_id} saved: {text_filepath} (Text) | {csv_filepath} (Data)")

    # --- Plotting Section ---
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    sns.barplot(
        x=["Makes <$50k", "Makes >$50k"], 
        y=[100 - income_prob, income_prob], 
        hue=["Makes <$50k", "Makes >$50k"], 
        palette="RdYlGn", 
        legend=False, 
        ax=axes[0,0]
    )
    axes[0,0].set_title("Weighted Income Probability (%)", fontsize=14, fontweight='bold')
    axes[0,0].set_ylim(0, 100)

    sns.histplot(data=group, x='age', weights='Weight', bins=20, kde=True, color="#3498db", ax=axes[0,1])
    axes[0,1].set_title("Weighted Age Distribution", fontsize=14, fontweight='bold')

    weighted_occ_counts = group.groupby('major occupation code', observed=False)['Weight'].sum().sort_values(ascending=False).head(5)
    weighted_occ_pct = (weighted_occ_counts / weighted_occ_counts.sum() * 100)
    sns.barplot(
        y=weighted_occ_counts.index, 
        x=weighted_occ_pct.values, 
        hue=weighted_occ_counts.index, 
        palette="Blues_d", 
        legend=False, 
        ax=axes[1,0]
    )
    axes[1,0].set_title("Top 5 Occupations (%)", fontsize=14, fontweight='bold')

    weighted_edu_counts = group.groupby('education', observed=False)['Weight'].sum().sort_values(ascending=False).head(5)
    weighted_edu_pct = (weighted_edu_counts / weighted_edu_counts.sum() * 100)
    sns.barplot(
        y=weighted_edu_counts.index, 
        x=weighted_edu_pct.values, 
        hue=weighted_edu_counts.index, 
        palette="Purples_d", 
        legend=False, 
        ax=axes[1,1]
    )
    axes[1,1].set_title("Top 5 Education Levels (%)", fontsize=14, fontweight='bold')

    plt.tight_layout(pad=4.0)

    # ALWAYS save the plot
    plt.savefig(f"{RESULTS_DIR}/Persona_{cluster_id}_Dashboard.png", bbox_inches='tight')
    
    # Only show plot if NOT in save_plots (silent) mode
    if not SAVE_PLOTS:
        plt.show()
    else:
        print(f"[INFO] Saved Persona {cluster_id} Dashboard to {RESULTS_DIR}/Persona_{cluster_id}_Dashboard.png")

# Run the display functions
for i in range(OPTIMAL_K):
    display_persona_report(i)