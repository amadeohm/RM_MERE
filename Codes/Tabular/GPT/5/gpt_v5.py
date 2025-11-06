#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated EDA, Preprocessing, and Feature Relationship Analysis Script
Requirements: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
Usage:
    python analysis.py [optional_path_to_csv]
Notes:
    - The script attempts to infer the target column. If a column named one of
      ["target", "label", "class", "outcome", "y"] exists, it will be chosen.
      Otherwise, the last column is used as target.
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# ==============================
# 0) TIMER START
# ==============================
t0 = time.time()

# ==============================
# 1) LOAD DATA
# ==============================
print("\n" + "="*80)
print("SECTION 1: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

# Determine input dataset path
default_candidates = [
    "/Users/amadeo/Documents/Code/diabetes_dataset.csv",  # uploaded example (if present)
    "data.csv"                          # generic default
]
csv_path = None
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    for cand in default_candidates:
        try:
            pd.read_csv(cand, nrows=1)
            csv_path = cand
            break
        except Exception:
            continue

if csv_path is None:
    print("ERROR: No dataset found. Please pass a CSV path as an argument or place 'data.csv' in the working directory.")
    sys.exit(1)

print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# ==============================
# 1A) BASIC INSPECTION
# ==============================
print("\n--- Basic Information ---")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values (count and %):")
missing_count = df.isna().sum()
missing_pct = (missing_count / len(df)) * 100
missing_df = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct.round(2)})
print(missing_df.sort_values("missing_count", ascending=False))

dup_count = df.duplicated().sum()
print(f"\nDuplicate Rows: {dup_count}")

print("\n--- Descriptive Statistics (Numerical) ---")
if df.select_dtypes(include=[np.number]).shape[1] > 0:
    print(df.select_dtypes(include=[np.number]).describe().T)
else:
    print("No numerical columns detected.")

print("\n--- Descriptive Statistics (Categorical) ---")
cat_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()
if len(cat_cols_all) > 0:
    desc_cats = df[cat_cols_all].describe().T
    print(desc_cats)
else:
    print("No categorical columns detected.")

# ==============================
# 1B) TARGET VARIABLE DETECTION & DISTRIBUTION
# ==============================
print("\n--- Target Variable Detection ---")
candidate_targets = ["target", "label", "class", "outcome", "y"]
target_col = None
for c in candidate_targets:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    target_col = df.columns[-1]  # fallback to last column

print(f"Selected target column: '{target_col}'")

# Analyze target distribution
print("\n--- Target Variable Distribution ---")
target = df[target_col]
if pd.api.types.is_numeric_dtype(target):
    print(target.describe())
    plt.figure()
    sns.histplot(target, kde=True)
    plt.title(f"Target Distribution: {target_col}")
    plt.xlabel(target_col); plt.ylabel("Count")
    plt.tight_layout(); plt.show()
else:
    print(target.value_counts())
    plt.figure()
    sns.countplot(x=target, order=target.value_counts().index)
    plt.title(f"Target Distribution: {target_col}")
    plt.xlabel(target_col); plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout(); plt.show()

# ==============================
# 1C) COLUMN TYPE SPLIT
# ==============================
# Treat low cardinality numerics as numeric (not categorical) but we will also capture "categorical-like" as object/category
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop([target_col], errors="ignore").tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.drop([target_col], errors="ignore").tolist()

# Additionally, some int-coded categories may be numeric with low cardinality. We'll keep them numeric for modeling,
# but they can be shown in countplots for EDA if cardinality is small.
low_card_numeric_as_cat = []
for col in numeric_cols:
    if df[col].nunique(dropna=True) <= 15:
        low_card_numeric_as_cat.append(col)

# ==============================
# 1D) EDA PLOTS
# ==============================
# Histograms/KDE for numerical features
if len(numeric_cols) > 0:
    print("\nGenerating histograms/KDE plots for numerical features...")
    for col in numeric_cols[:30]:  # limit to 30 to avoid plot overload
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram with KDE: {col}")
        plt.xlabel(col); plt.ylabel("Count")
        plt.tight_layout(); plt.show()
else:
    print("No numerical features found for histogram/KDE plots.")

# Boxplots for outlier detection (numerical)
if len(numeric_cols) > 0:
    print("\nGenerating boxplots for outlier detection (numerical features)...")
    for col in numeric_cols[:30]:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col}")
        plt.xlabel(col)
        plt.tight_layout(); plt.show()
else:
    print("No numerical features found for boxplots.")

# Correlation heatmap (numerical)
if len(numeric_cols) > 0:
    print("\nGenerating correlation heatmap for numerical features...")
    corr = df[numeric_cols].corr(method="pearson")
    plt.figure(figsize=(min(12, 0.6*len(numeric_cols)+4), min(10, 0.6*len(numeric_cols)+3)))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout(); plt.show()
else:
    print("No numerical features to compute correlation heatmap.")

# Countplots for categorical variables (including low-card numeric shown as categorical)
countplot_cols = categorical_cols + low_card_numeric_as_cat
countplot_cols = [c for c in countplot_cols if c != target_col]
if len(countplot_cols) > 0:
    print("\nGenerating countplots for categorical/low-cardinality features...")
    for col in countplot_cols[:30]:
        if df[col].nunique(dropna=True) > 50:
            continue  # skip too many categories
        plt.figure()
        order = df[col].value_counts().index
        sns.countplot(x=df[col], order=order)
        plt.title(f"Countplot: {col}")
        plt.xlabel(col); plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout(); plt.show()
else:
    print("No categorical/low-cardinality features for countplots.")

# ==============================
# 2) DATA PREPROCESSING & FEATURE ENGINEERING
# ==============================
print("\n" + "="*80)
print("SECTION 2: DATA PREPROCESSING AND FEATURE ENGINEERING")
print("="*80)

# Separate features and target
X = df.drop(columns=[target_col], errors="ignore").copy()
y_raw = df[target_col].copy()

# Encode target if categorical or non-numeric
le = None
y = y_raw.copy()
if not pd.api.types.is_numeric_dtype(y):
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    print("\nTarget encoded with LabelEncoder.")
else:
    # If numeric but looks categorical (few unique values), keep as-is but note it
    if y.nunique(dropna=True) <= 10:
        print("\nTarget detected as numeric with low cardinality (likely categorical/binary).")

# Missing value handling (impute before encoding/scaling for correlation & class balance, but ColumnTransformer will also handle)
print("\nImputing missing values for quick checks...")
X_impute_for_checks = X.copy()
num_imputer_quick = SimpleImputer(strategy="median")
cat_imputer_quick = SimpleImputer(strategy="most_frequent")

if len(numeric_cols) > 0:
    X_impute_for_checks[numeric_cols] = num_imputer_quick.fit_transform(X_impute_for_checks[numeric_cols])
if len(categorical_cols) > 0:
    X_impute_for_checks[categorical_cols] = cat_imputer_quick.fit_transform(X_impute_for_checks[categorical_cols])

# Report class imbalance (only if target is discrete)
print("\n--- Class Balance Check (Before Resampling) ---")
if pd.Series(y).nunique() <= 20:
    class_counts_before = pd.Series(y).value_counts().sort_index()
    print(class_counts_before)
    # Simple imbalance heuristic: minority < 40% of majority
    majority = class_counts_before.max()
    minority = class_counts_before.min()
    imbalance_ratio = minority / majority if majority > 0 else 1.0
    is_imbalanced = (imbalance_ratio < 0.6) and (class_counts_before.shape[0] == 2)  # focus on binary for resampling
else:
    print("Target seems continuous or multi-class with high cardinality; no imbalance check applied.")
    is_imbalanced = False

# If imbalanced and binary, apply undersampling of majority class
X_resampled = X.copy()
y_resampled = y.copy()
if is_imbalanced and pd.Series(y).nunique() == 2:
    print("\nApplying undersampling to address class imbalance (binary target)...")
    data_temp = X_impute_for_checks.copy()
    data_temp["__y__"] = y
    # Identify classes
    cls_vals = data_temp["__y__"].unique()
    cls0, cls1 = cls_vals[0], cls_vals[1]
    n0 = (data_temp["__y__"] == cls0).sum()
    n1 = (data_temp["__y__"] == cls1).sum()
    minority_cls = cls0 if n0 < n1 else cls1
    majority_cls = cls1 if minority_cls == cls0 else cls0
    n_min = min(n0, n1)

    df_min = data_temp[data_temp["__y__"] == minority_cls]
    df_maj = data_temp[data_temp["__y__"] == majority_cls]
    df_maj_down = resample(df_maj, replace=False, n_samples=n_min, random_state=42)

    balanced = pd.concat([df_min, df_maj_down], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    y_resampled = balanced["__y__"].values
    X_resampled = balanced.drop(columns="__y__")
    print("\n--- Class Balance (After Undersampling) ---")
    print(pd.Series(y_resampled).value_counts().sort_index())
else:
    if pd.Series(y).nunique() == 2:
        print("\nClass imbalance not severe; no resampling applied.")
    else:
        print("\nTarget is not binary; skipping resampling.")

# Encoding & Scaling
print("\nBuilding preprocessing pipeline (impute -> encode -> scale)...")
num_features = X_resampled.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_resampled.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())  # or MinMaxScaler()
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ],
    remainder="drop"
)

X_preprocessed = preprocessor.fit_transform(X_resampled)
print("Preprocessing complete.")
print(f"Transformed feature matrix shape: {X_preprocessed.shape}")

# ==============================
# 3) FEATURE RELATIONSHIP ANALYSIS
# ==============================
print("\n" + "="*80)
print("SECTION 3: FEATURE RELATIONSHIP ANALYSIS")
print("="*80)

# For correlation against target, we focus on original numerical features (pre-imputation with clean fills)
# Prepare a clean numerical dataset for correlation analysis
num_for_corr = numeric_cols.copy()
if target_col in num_for_corr:
    num_for_corr.remove(target_col)

corr_results = []

def safe_stat(func, a, b):
    """Compute statistic and p-value; handle constant arrays."""
    if len(np.unique(a[~np.isnan(a)])) <= 1 or len(np.unique(b[~np.isnan(b)])) <= 1:
        return np.nan, np.nan
    try:
        stat = func(a, b)
        if isinstance(stat, tuple) and len(stat) == 2:
            return stat[0], stat[1]
        else:
            return stat, np.nan
    except Exception:
        return np.nan, np.nan

if len(num_for_corr) > 0:
    # Prepare filled numeric data
    X_num_filled = df[num_for_corr].copy()
    X_num_filled = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_num_filled), columns=num_for_corr)

    # Encode/prepare target for stats
    y_for_stats = y.copy()
    is_binary_target = (pd.Series(y_for_stats).nunique() == 2)
    is_numeric_target = pd.api.types.is_numeric_dtype(y_raw)

    if is_binary_target:
        print("\nTarget detected as BINARY -> Using Point-Biserial correlation with numerical features.")
        for col in num_for_corr:
            r, p = safe_stat(stats.pointbiserialr, X_num_filled[col].values, y_for_stats.astype(float))
            corr_results.append({"feature": col, "method": "pointbiserial", "stat": r, "p_value": p})
    else:
        print("\nTarget detected as CONTINUOUS/MULTICLASS -> Reporting Pearson and Spearman with numerical features.")
        for col in num_for_corr:
            r_p, p_p = safe_stat(stats.pearsonr, X_num_filled[col].values, y_for_stats.astype(float))
            r_s, p_s = safe_stat(stats.spearmanr, X_num_filled[col].values, y_for_stats.astype(float))
            corr_results.append({
                "feature": col,
                "method": "pearson",
                "stat": r_p, "p_value": p_p
            })
            corr_results.append({
                "feature": col,
                "method": "spearman",
                "stat": r_s, "p_value": p_s
            })

    corr_table = pd.DataFrame(corr_results)
    # Sort by absolute statistic (descending), but keep NaNs at bottom
    corr_table["abs_stat"] = corr_table["stat"].abs()
    corr_table_sorted = corr_table.sort_values(["abs_stat", "p_value"], ascending=[False, True])
    print("\n--- Correlation with Target (Top 20 by |stat|) ---")
    print(corr_table_sorted.drop(columns=["abs_stat"]).head(20).to_string(index=False))

    # Visual correlation plots
    if is_binary_target:
        # Boxplots of top numerical features vs binary target
        top_feats = corr_table_sorted.dropna(subset=["stat"]).head(6)["feature"].unique().tolist()
        print("\nGenerating boxplots for top correlated numerical features vs binary target...")
        temp = pd.concat([X_num_filled[top_feats], pd.Series(y_for_stats, name=target_col)], axis=1)
        # Map target to labels if LabelEncoder existed
        if le is not None:
            inv_map = dict(zip(range(len(le.classes_)), le.classes_))
            temp[target_col] = temp[target_col].map(inv_map)
        for col in top_feats:
            plt.figure()
            sns.boxplot(x=target_col, y=col, data=temp)
            plt.title(f"{col} vs {target_col}")
            plt.tight_layout(); plt.show()
    else:
        # Pairplot among top numerical features + target (if numeric target)
        if is_numeric_target:
            top_feats = corr_table_sorted.dropna(subset=["stat"]).head(5)["feature"].unique().tolist()
            if len(top_feats) > 0:
                print("\nGenerating pairplot for top correlated numerical features with target...")
                temp = pd.concat([X_num_filled[top_feats], y_raw.rename(target_col)], axis=1)
                sns.pairplot(temp[top_feats + [target_col]].dropna())
                plt.suptitle("Pairplot: Top Correlated Features with Target", y=1.02)
                plt.show()
        # Scatterplots of top correlated
        top_feats_scatter = corr_table_sorted.dropna(subset=["stat"]).head(6)["feature"].unique().tolist()
        if is_numeric_target and len(top_feats_scatter) > 0:
            print("\nGenerating scatterplots for top correlated features vs target...")
            for col in top_feats_scatter:
                plt.figure()
                sns.scatterplot(x=df[col], y=y_raw)
                plt.xlabel(col); plt.ylabel(target_col)
                plt.title(f"{col} vs {target_col}")
                plt.tight_layout(); plt.show()
else:
    print("No numerical features available for target correlation analysis.")

# ==============================
# 4) TIME MEASUREMENT
# ==============================
t1 = time.time()
elapsed = int(round(t1 - t0))
minutes = elapsed // 60
seconds = elapsed % 60

print("\n" + "="*80)
print("SECTION 4: TIME MEASUREMENT")
print("="*80)
print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds")

# ==============================
# 5) CONCLUSION ON DATA SUITABILITY
# ==============================
print("\n" + "="*80)
print("SECTION 5: CONCLUSION ON DATA SUITABILITY FOR CLASSIFICATION")
print("="*80)

# Heuristic summary based on EDA
n_rows, n_cols = df.shape
missing_ratio = (df.isna().sum().sum()) / (n_rows * n_cols) if n_rows*n_cols > 0 else 0.0

# Multicollinearity heuristic: average absolute off-diagonal correlation among numerics
multicoll_score = np.nan
if len(numeric_cols) > 1:
    c = df[numeric_cols].corr().abs().values
    # off-diagonal mean
    multicoll_score = (c.sum() - np.trace(c)) / (c.size - len(numeric_cols))

# Class balance heuristic (binary only)
class_balance_str = "N/A (non-discrete target)"
if pd.Series(y).nunique() <= 10:
    vc = pd.Series(y).value_counts(normalize=True)
    class_balance_str = f"max class share = {vc.max():.2f}"

noise_flags = []
if missing_ratio > 0.2:
    noise_flags.append("high missingness")
if isinstance(multicoll_score, float) and not np.isnan(multicoll_score) and multicoll_score > 0.6:
    noise_flags.append("potential multicollinearity among numerics")
if pd.Series(y).nunique() > 20:
    noise_flags.append("target appears continuous/high-cardinality")

suitability_msgs = []
if pd.Series(y).nunique() <= 10:
    suitability_msgs.append("Target has discrete/limited classes; classification is plausible.")
else:
    suitability_msgs.append("Target may be continuous; consider regression or discretization for classification.")

if missing_ratio <= 0.2:
    suitability_msgs.append("Missingness is manageable.")
else:
    suitability_msgs.append("Substantial missingness may hinder classification.")

if isinstance(multicoll_score, float) and not np.isnan(multicoll_score):
    if multicoll_score <= 0.6:
        suitability_msgs.append("No strong evidence of multicollinearity.")
    else:
        suitability_msgs.append("High correlation among features suggests feature selection/regularization.")

if pd.Series(y).nunique() == 2:
    # check balance again
    if vc.max() > 0.7:
        suitability_msgs.append("Binary target is imbalanced; resampling or class-weighting recommended.")
    else:
        suitability_msgs.append("Binary target appears reasonably balanced.")

print(f"- Rows: {n_rows}, Columns: {n_cols}")
print(f"- Missingness (overall): {missing_ratio*100:.2f}%")
if isinstance(multicoll_score, float) and not np.isnan(multicoll_score):
    print(f"- Avg absolute inter-feature correlation (numerics): {multicoll_score:.2f}")
print(f"- Class balance (if applicable): {class_balance_str}")
if noise_flags:
    print(f"- Potential issues: {', '.join(noise_flags)}")
else:
    print("- Potential issues: none detected")

print("\nSummary Assessment:")
for m in suitability_msgs:
    print(f"• {m}")

print("\nDone.")
