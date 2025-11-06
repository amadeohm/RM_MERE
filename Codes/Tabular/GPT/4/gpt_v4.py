#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis.py
Senior Data Scientist Health Analytics - Automated EDA, Preprocessing, and Feature Relationship Analysis
Requirements satisfied:
- Uses pandas, numpy, matplotlib, seaborn, scipy.stats, sklearn
- Clean sections with print statements
- Plots shown inline with plt.show()
- Handles missing values, encoding, scaling
- Detects class imbalance and applies undersampling (no external deps like imblearn)
- Correlation analysis with appropriate statistics (Pearson/Spearman/Point-Biserial)
- Measures and prints total runtime
- Prints concise conclusion about data suitability for classification
"""

# ==============================
# Imports
# ==============================
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import pearsonr, spearmanr, pointbiserialr

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

# ==============================
# Helper Functions
# ==============================

def infer_target_column(df: pd.DataFrame):
    """Heuristic to infer target column."""
    candidates = ['target', 'label', 'class', 'outcome', 'y', 'diagnosis', 'readmitted']
    for c in candidates:
        for col in df.columns:
            if col.lower() == c:
                return col

    # Prefer a low-cardinality column (<=10 unique) that is not obviously an ID
    low_card_cols = [c for c in df.columns
                     if df[c].nunique(dropna=True) <= 10 and not is_id_like(df[c], c)]
    if len(low_card_cols) > 0:
        # choose the rightmost/last such column (often target is at the end)
        return low_card_cols[-1]

    # Fallback: last column
    return df.columns[-1]

def is_id_like(series: pd.Series, name: str):
    """Detect id-like columns by name or uniqueness ratio."""
    name_lower = str(name).lower()
    if 'id' in name_lower or 'uuid' in name_lower:
        return True
    nunique = series.nunique(dropna=True)
    if nunique > 0.9 * len(series):
        return True
    return False

def summarize_missingness(df: pd.DataFrame):
    miss = df.isna().sum()
    miss_pct = (miss / len(df)) * 100
    return pd.DataFrame({'missing_count': miss, 'missing_pct': miss_pct}).sort_values('missing_pct', ascending=False)

def detect_imbalance(y: pd.Series, threshold: float = 0.6):
    """
    For discrete targets: report imbalance if the largest class proportion >= threshold.
    For continuous targets: not applicable.
    """
    if is_continuous(y):
        return False, None
    counts = y.value_counts(dropna=False)
    proportions = counts / counts.sum()
    max_prop = proportions.max()
    imbalanced = max_prop >= threshold
    return imbalanced, counts

def is_binary(y: pd.Series):
    return y.dropna().nunique() == 2

def is_continuous(s: pd.Series):
    return pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 10

def undersample(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Random undersampling to the minority class count for multi-class as well.
    """
    rng = check_random_state(random_state)
    df_xy = X.copy()
    df_xy['_y_'] = y.values
    classes = df_xy['_y_'].dropna().unique()
    class_counts = df_xy['_y_'].value_counts()
    min_count = class_counts.min()
    frames = []
    for c in classes:
        idx = df_xy[df_xy['_y_'] == c].index.values
        if len(idx) > min_count:
            sel = rng.choice(idx, size=min_count, replace=False)
            frames.append(df_xy.loc[sel])
        else:
            frames.append(df_xy.loc[idx])
    balanced = pd.concat(frames, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    y_bal = balanced.pop('_y_')
    return balanced, y_bal

def compute_correlations_num_target(X_num: pd.DataFrame, y: pd.Series):
    """
    If target is continuous -> Pearson & Spearman for each numerical feature.
    If target is binary -> Point-biserial for each numerical feature.
    If target is multiclass categorical -> Spearman between feature and label-encoded target (as a fallback).
    Returns: DataFrame with columns [feature, method, corr, p_value]
    """
    results = []
    if X_num.shape[1] == 0:
        return pd.DataFrame(columns=['feature', 'method', 'corr', 'p_value'])

    if is_continuous(y):
        # Continuous target
        for col in X_num.columns:
            x = X_num[col]
            # Pearson
            try:
                r, p = pearsonr(x.dropna(), y.loc[x.dropna().index])
                results.append([col, 'pearson', r, p])
            except Exception:
                results.append([col, 'pearson', np.nan, np.nan])
            # Spearman
            try:
                r, p = spearmanr(x, y, nan_policy='omit')
                results.append([col, 'spearman', r, p])
            except Exception:
                results.append([col, 'spearman', np.nan, np.nan])

    elif is_binary(y):
        # Binary target (point-biserial)
        y_bin = y.copy()
        # Convert to 0/1 if not numeric
        if not pd.api.types.is_numeric_dtype(y_bin):
            classes = sorted(y_bin.dropna().unique())
            mapping = {cls: i for i, cls in enumerate(classes)}
            y_bin = y_bin.map(mapping)
        for col in X_num.columns:
            x = X_num[col]
            try:
                r, p = pointbiserialr(y_bin, x)
                results.append([col, 'point-biserial', r, p])
            except Exception:
                results.append([col, 'point-biserial', np.nan, np.nan])
    else:
        # Multiclass categorical target: use Spearman with label-encoded target (ordinal proxy)
        y_enc = y.astype('category').cat.codes
        for col in X_num.columns:
            x = X_num[col]
            try:
                r, p = spearmanr(x, y_enc, nan_policy='omit')
                results.append([col, 'spearman (with encoded y)', r, p])
            except Exception:
                results.append([col, 'spearman (with encoded y)', np.nan, np.nan])

    res_df = pd.DataFrame(results, columns=['feature', 'method', 'corr', 'p_value'])
    return res_df.sort_values(['method', 'corr'], ascending=[True, False]).reset_index(drop=True)

def compute_vif_like_flag(X_num: pd.DataFrame, corr_threshold: float = 0.9):
    """
    Simple multicollinearity flag based on high pairwise correlations.
    """
    flag = False
    pairs = []
    if X_num.shape[1] <= 1:
        return flag, pairs
    corr = X_num.corr(numeric_only=True)
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if j <= i:
                continue
            val = corr.loc[c1, c2]
            if abs(val) >= corr_threshold:
                flag = True
                pairs.append((c1, c2, float(val)))
    return flag, pairs

def outlier_ratio_iqr(X_num: pd.DataFrame):
    """
    Compute proportion of outliers (beyond 1.5 IQR) per feature and return average ratio.
    """
    ratios = []
    for col in X_num.columns:
        x = X_num[col].dropna()
        if x.empty:
            continue
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((x < lower) | (x > upper)).mean()
        ratios.append(outliers)
    return np.mean(ratios) if len(ratios) else 0.0

def print_title(title):
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")

def safe_show():
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.show()

# ==============================
# Main
# ==============================
def main():
    start_time = time.time()

    # ------------------------------------------
    # Load dataset
    # ------------------------------------------
    default_paths = [
        len(sys.argv) > 1 and sys.argv[1] or None,
        "/Users/amadeo/Documents/Code/diabetes_dataset.csv",  # if running in environments where this exists
        "data.csv"
    ]
    dataset_path = next((p for p in default_paths if p), "data.csv")

    print_title("1) Exploratory Data Analysis (EDA)")
    print(f"[INFO] Attempting to load dataset from: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    # Basic overview
    print("[BASIC INFO] Dataset Shape:", df.shape)
    print("\n[D_TYPES]\n", df.dtypes)
    print("\n[MISSING VALUES]\n", summarize_missingness(df))
    dup_count = df.duplicated().sum()
    print("\n[DUPLICATES] Number of duplicate rows:", dup_count)

    # Descriptive stats
    with pd.option_context('display.max_columns', None):
        print("\n[DESCRIPTIVE STATISTICS - NUMERICAL]\n", df.describe(include=[np.number]).T)
        print("\n[DESCRIPTIVE STATISTICS - CATEGORICAL]\n", df.describe(include=['object', 'category', 'bool']).T)

    # Infer target
    target_col = infer_target_column(df)
    print(f"\n[TARGET] Inferred target column: '{target_col}'")
    if target_col not in df.columns:
        print("[ERROR] Target column inference failed.")
        sys.exit(1)

    # Target distribution analysis
    y = df[target_col]
    print("\n[TARGET DISTRIBUTION]")
    if is_continuous(y):
        print(y.describe())
        plt.figure()
        sns.kdeplot(y.dropna(), fill=True)
        plt.title(f"Target Distribution (KDE) - {target_col}")
        plt.xlabel(target_col)
        safe_show()
    else:
        counts = y.value_counts(dropna=False)
        print(counts)
        plt.figure()
        sns.countplot(x=target_col, data=df)
        plt.title(f"Target Distribution (Countplot) - {target_col}")
        plt.xticks(rotation=45)
        safe_show()

    # Identify feature types
    feature_cols = [c for c in df.columns if c != target_col and not is_id_like(df[c], c)]
    X = df[feature_cols].copy()
    X_num = X.select_dtypes(include=[np.number])
    X_cat = X.select_dtypes(include=['object', 'category', 'bool'])

    # Histograms / KDE for numerical features
    if X_num.shape[1] > 0:
        print("\n[PLOTS] Histograms for numerical features")
        for c in X_num.columns:
            plt.figure()
            sns.histplot(X_num[c].dropna(), kde=True)
            plt.title(f"Histogram & KDE - {c}")
            plt.xlabel(c)
            safe_show()

        # Boxplots for outlier detection
        print("\n[PLOTS] Boxplots for numerical features (outlier detection)")
        for c in X_num.columns:
            plt.figure()
            sns.boxplot(x=X_num[c].dropna())
            plt.title(f"Boxplot - {c}")
            safe_show()

        # Correlation heatmap among numeric features
        print("\n[PLOTS] Correlation heatmap for numerical features")
        corr = X_num.corr(numeric_only=True)
        plt.figure(figsize=(min(12, 1 + X_num.shape[1]), min(10, 1 + X_num.shape[1])))
        sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f", square=True)
        plt.title("Correlation Heatmap (Numeric Features)")
        safe_show()
    else:
        print("[INFO] No numerical features found for histogram/boxplot/heatmap.")

    # Countplots for categorical features
    if X_cat.shape[1] > 0:
        print("\n[PLOTS] Countplots for categorical features")
        for c in X_cat.columns:
            plt.figure()
            sns.countplot(x=c, data=X)
            plt.title(f"Countplot - {c}")
            plt.xticks(rotation=45)
            safe_show()
    else:
        print("[INFO] No categorical features found for countplots.")

    # ------------------------------------------
    # 2) Data Preprocessing and Feature Engineering
    # ------------------------------------------
    print_title("2) Data Preprocessing and Feature Engineering")

    numeric_features = list(X_num.columns)
    categorical_features = list(X_cat.columns)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Fit/transform features
    print("[PREPROCESS] Fitting transformers and transforming features...")
    X_processed = preprocessor.fit_transform(X)

    # Build feature names after OneHot
    feature_names = []
    if numeric_features:
        feature_names += numeric_features
    if categorical_features:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        ohe_names = list(ohe.get_feature_names_out(categorical_features))
        feature_names += ohe_names

    print(f"[PREPROCESS] Processed feature matrix shape: {X_processed.shape}")
    print(f"[PREPROCESS] Number of numeric features: {len(numeric_features)} | Number of categorical features: {len(categorical_features)}")
    print(f"[PREPROCESS] Total engineered features: {len(feature_names)}")

    # Class imbalance detection and handling (for categorical/binary target)
    imbalanced, class_counts = detect_imbalance(y)
    if class_counts is not None:
        print("\n[CLASS DISTRIBUTION - BEFORE]")
        print(class_counts)

    if imbalanced and not is_continuous(y):
        print("\n[IMBALANCE] Detected class imbalance. Applying random undersampling to minority count.")
        # Perform undersampling on original (preprocessed later or now?)
        # We'll undersample on the raw data indices and then re-transform to keep pipeline consistent.
        X_bal, y_bal = undersample(X, y, random_state=42)
        # Refit-transform on balanced set (to avoid data leakage in real pipelines we'd fit on train only; here for analysis end-to-end)
        X_processed = preprocessor.fit_transform(X_bal)
        # Update feature names after refit
        feature_names = []
        if numeric_features:
            feature_names += numeric_features
        if categorical_features:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            ohe_names = list(ohe.get_feature_names_out(categorical_features))
            feature_names += ohe_names

        print("[CLASS DISTRIBUTION - AFTER (undersampled)]")
        print(y_bal.value_counts())
        y_for_corr = y_bal.copy()
        X_num_for_corr = X_bal.select_dtypes(include=[np.number]).copy()
    else:
        if imbalanced:
            print("[IMBALANCE] Target appears continuous; skipping resampling.")
        else:
            print("\n[IMBALANCE] No significant class imbalance detected (or below threshold).")
        y_for_corr = y.copy()
        X_num_for_corr = X_num.copy()

    # ------------------------------------------
    # 3) Feature Relationship Analysis
    # ------------------------------------------
    print_title("3) Feature Relationship Analysis")

    # Correlation between numerical features and target
    print("[CORRELATION] Computing correlations between numerical features and target...")
    corr_table = compute_correlations_num_target(X_num_for_corr, y_for_corr)
    if corr_table.empty:
        print("[CORRELATION] No numerical features available for correlation with target.")
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("\n[CORRELATION TABLE: feature / method / corr / p_value]\n", corr_table)

        # Visual correlation plots: pairplot for top 4 strongest absolute correlations (unique by feature)
        unique_corr = corr_table.sort_values('corr', key=lambda s: s.abs(), ascending=False).drop_duplicates('feature')
        top_features = list(unique_corr['feature'].head(4).values)

        if len(top_features) > 0:
            print(f"\n[PLOTS] Pairplot for top features vs target (numerical target or encoded categories if possible).")
            # Prepare a small DataFrame for pairplot
            plot_df = X_num_for_corr[top_features].copy()
            # Add a visualization-friendly target
            if is_continuous(y_for_corr):
                plot_df[target_col] = y_for_corr.values
                sns.pairplot(plot_df, diag_kind='kde')
                plt.suptitle("Pairplot: Top Correlated Features & Target", y=1.02)
                safe_show()
            else:
                # For categorical targets, scatter versus each top feature
                for f in top_features:
                    plt.figure()
                    # Jitter y for visibility if categorical
                    y_codes = y_for_corr.astype('category').cat.codes
                    plt.scatter(X_num_for_corr[f], y_codes, alpha=0.6)
                    plt.xlabel(f)
                    plt.ylabel(f"{target_col} (encoded)")
                    plt.title(f"Scatter: {f} vs {target_col} (encoded)")
                    safe_show()

        # Heatmap of correlations among numerical features and target (if target numeric add; else add encoded y)
        print("\n[PLOTS] Heatmap including target (encoded if categorical).")
        corr_df = X_num_for_corr.copy()
        if is_continuous(y_for_corr):
            corr_df[target_col] = y_for_corr
        else:
            corr_df[target_col + "_enc"] = y_for_corr.astype('category').cat.codes
        plt.figure(figsize=(min(12, 1 + corr_df.shape[1]), min(10, 1 + corr_df.shape[1])))
        sns.heatmap(corr_df.corr(numeric_only=True), cmap="coolwarm", annot=False, square=True)
        plt.title("Correlation Heatmap (with Target)")
        safe_show()

    # ------------------------------------------
    # 4) Time Measurement
    # ------------------------------------------
    print_title("4) Time Measurement")
    total_seconds = time.time() - start_time
    minutes = int(total_seconds // 60)
    seconds = int(round(total_seconds % 60))
    print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds")

    # ------------------------------------------
    # 5) Conclusion on Data Suitability for Classification
    # ------------------------------------------
    print_title("5) Conclusion on Data Suitability for Classification")

    # Metrics for suitability:
    # - Missingness
    miss_df = summarize_missingness(df)
    avg_missing_pct = miss_df['missing_pct'].mean()

    # - Class balance (if categorical)
    imbalance_note = "N/A (continuous target)"
    if not is_continuous(y):
        _, counts_now = detect_imbalance(y_for_corr)
        if counts_now is not None and counts_now.sum() > 0:
            props = (counts_now / counts_now.sum()).sort_values(ascending=False)
            top_prop = float(props.iloc[0])
            imbalance_note = f"Top class proportion = {top_prop:.2f}"
        else:
            imbalance_note = "Unable to compute (no counts)."

    # - Multicollinearity among numeric features
    mc_flag, mc_pairs = compute_vif_like_flag(X_num)
    mc_note = "High pairwise correlations detected" if mc_flag else "No strong multicollinearity flags"
    if mc_flag:
        # Show up to 5 example pairs
        example_pairs = ", ".join([f"{a}~{b} (r={r:.2f})" for a,b,r in mc_pairs[:5]])
        mc_note += f" (e.g., {example_pairs})"

    # - Noise via outliers
    out_ratio = outlier_ratio_iqr(X_num)

    # Suitability summary
    suitability_msgs = []
    if is_continuous(y):
        suitability_msgs.append("Target appears continuous; dataset may be more suitable for regression.")
    else:
        suitability_msgs.append("Target appears categorical; dataset is suitable for classification tasks.")

    if avg_missing_pct > 20:
        suitability_msgs.append("High average missingness; imputation strategies needed and may affect model reliability.")
    elif avg_missing_pct > 5:
        suitability_msgs.append("Moderate missingness; imputation should be manageable.")
    else:
        suitability_msgs.append("Low missingness overall.")

    if not is_continuous(y) and isinstance(imbalance_note, str) and "Top class proportion" in imbalance_note:
        prop = float(imbalance_note.split('=')[-1])
        if prop >= 0.8:
            suitability_msgs.append("Severe class imbalance observed; consider resampling, class weights, or proper metrics (AUC, F1).")
        elif prop >= 0.6:
            suitability_msgs.append("Moderate class imbalance; monitor metrics beyond accuracy and apply resampling if needed.")
        else:
            suitability_msgs.append("Class distribution is reasonably balanced.")

    if mc_flag:
        suitability_msgs.append("Potential multicollinearity among numerical features; consider feature selection/regularization.")
    else:
        suitability_msgs.append("No strong multicollinearity concerns detected.")

    if out_ratio >= 0.15:
        suitability_msgs.append("Notable proportion of outliers; consider robust scalers, transformations, or outlier handling.")
    elif out_ratio >= 0.05:
        suitability_msgs.append("Some outliers present; monitor influence on models.")
    else:
        suitability_msgs.append("Few outliers detected.")

    print("[SUMMARY]")
    print(f"- Target column: {target_col}")
    print(f"- Average missingness across columns: {avg_missing_pct:.2f}%")
    print(f"- Class balance: {imbalance_note}")
    print(f"- Multicollinearity: {mc_note}")
    print(f"- Approx. outlier ratio (avg across numeric features): {out_ratio:.2f}")

    print("\n[ASSESSMENT]")
    for m in suitability_msgs:
        print(f"* {m}")

    print("\n[NOTE] This automated report is exploratory. For modeling, create a proper train/validation split and fit pipelines on training data only to avoid leakage.")

# ==============================
# Entrypoint
# ==============================
if __name__ == "__main__":
    main()
