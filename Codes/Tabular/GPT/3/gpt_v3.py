#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated EDA + Preprocessing + Feature Relationship Analysis
Requirements: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
Usage:
    python analysis.py --file /path/to/data.csv --target TARGET_COLUMN_NAME(optional)
If --target is omitted, the script will try to infer it (common names or last column).
"""

import argparse
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, List, Optional

from scipy import stats
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =========================
# Utility functions
# =========================
def infer_target_column(df: pd.DataFrame, provided_target: Optional[str] = None) -> str:
    if provided_target is not None and provided_target in df.columns:
        return provided_target

    candidates = ["target", "label", "class", "y", "outcome"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: last column
    return df.columns[-1]


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def is_categorical_series(s: pd.Series, max_unique_as_categorical: int = 20) -> bool:
    # Treat object or category as categorical; also consider low-cardinality numerics as categorical candidates
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s):
        return True
    if is_numeric_series(s) and s.nunique(dropna=True) <= max_unique_as_categorical:
        return True
    return False


def is_binary_series(s: pd.Series) -> bool:
    return s.dropna().nunique() == 2


def normalize_binary_labels(y: pd.Series) -> Tuple[pd.Series, dict]:
    # Ensure binary labels are 0/1 if not already
    mapping = {}
    unique_vals = sorted(y.dropna().unique(), key=lambda x: str(x))
    if len(unique_vals) == 2:
        if set(unique_vals) == {0, 1}:
            return y, {0: 0, 1: 1}
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        y_mapped = y.map(mapping)
        return y_mapped, mapping
    return y, mapping


def class_balance_report(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts(dropna=False).rename("count")
    perc = (counts / counts.sum() * 100).rename("percent")
    return pd.concat([counts, perc], axis=1)


def needs_undersampling(y: pd.Series, imbalance_threshold: float = 0.6) -> bool:
    # If majority class proportion > threshold -> imbalance
    vc = y.value_counts(normalize=True)
    if len(vc) < 2:
        return False
    return vc.max() > imbalance_threshold


def random_undersample(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    # Simple random undersampling: downsample majority classes to the size of the smallest class
    rng = np.random.default_rng(random_state)
    df = X.copy()
    df["_y_"] = y.values
    min_count = df["_y_"].value_counts().min()
    frames = []
    for cls, group in df.groupby("_y_"):
        if len(group) > min_count:
            idx = rng.choice(group.index.values, size=min_count, replace=False)
            frames.append(group.loc[idx])
        else:
            frames.append(group)
    res = pd.concat(frames).sample(frac=1.0, random_state=random_state)  # shuffle
    y_res = res["_y_"].copy()
    X_res = res.drop(columns=["_y_"])
    return X_res, y_res


def select_scaler(kind: str = "standard"):
    if kind.lower() == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def correlation_with_target_table(df_num: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute correlation between numerical features and target using:
    - If y is numeric & continuous: Pearson and Spearman
    - If y is binary: Point-biserial (equivalent to Pearson with binary) and Spearman (rank on y)
    - If y is ordinal-like numeric with few unique values (>2): Spearman
    Returns a table with columns: feature, method, correlation, p_value
    """
    results = []
    y_nonnull = y.dropna()
    # Align
    df_num = df_num.loc[y_nonnull.index]
    y_aligned = y_nonnull

    y_is_numeric = is_numeric_series(y_aligned)
    y_unique = y_aligned.nunique()

    # For binary y, ensure 0/1 for point-biserial
    y_pb = y_aligned.copy()
    if y_is_numeric and is_binary_series(y_aligned) is False and y_unique <= 10 and np.allclose(y_aligned % 1, 0, equal_nan=True):
        # Multi-class integer target treated as ordinal for Spearman
        pass
    elif is_binary_series(y_aligned):
        y_pb, _ = normalize_binary_labels(y_aligned)

    for col in df_num.columns:
        x = df_num[col].dropna()
        idx = x.index.intersection(y_aligned.index)
        x = x.loc[idx]
        y_i = y_aligned.loc[idx]

        if len(x) < 3:
            continue

        if is_binary_series(y_i):  # Binary target
            # Point-biserial
            try:
                y_bin, _ = normalize_binary_labels(y_i)
                r, p = stats.pointbiserialr(x, y_bin)
                results.append([col, "point-biserial", r, p])
            except Exception as e:
                results.append([col, "point-biserial", np.nan, np.nan])
            # Spearman as nonparametric check
            try:
                rho, p2 = stats.spearmanr(x, y_i)
                results.append([col, "spearman", rho, p2])
            except Exception as e:
                results.append([col, "spearman", np.nan, np.nan])

        elif y_is_numeric and y_unique > 10:  # Continuous target
            # Pearson
            try:
                r, p = stats.pearsonr(x, y_i)
                results.append([col, "pearson", r, p])
            except Exception as e:
                results.append([col, "pearson", np.nan, np.nan])
            # Spearman
            try:
                rho, p2 = stats.spearmanr(x, y_i)
                results.append([col, "spearman", rho, p2])
            except Exception as e:
                results.append([col, "spearman", np.nan, np.nan])

        else:
            # Ordinal / few unique numeric target -> Spearman
            try:
                rho, p2 = stats.spearmanr(x, y_i)
                results.append([col, "spearman", rho, p2])
            except Exception as e:
                results.append([col, "spearman", np.nan, np.nan])

    out = pd.DataFrame(results, columns=["feature", "method", "correlation", "p_value"])
    return out.sort_values(by=["method", "correlation"], ascending=[True, False]).reset_index(drop=True)


def brief_suitability_assessment(
    y: pd.Series,
    missing_ratio: float,
    multicollinearity_flag: bool,
    outlier_flag: bool
) -> str:
    # Balance
    balance_df = class_balance_report(y)
    majority_pct = balance_df["percent"].max() if len(balance_df) > 0 else 100.0
    balance_txt = "balanced" if majority_pct <= 60 else "imbalanced"

    # Missingness
    miss_txt = f"{missing_ratio:.1f}% missing overall"
    miss_eval = "low" if missing_ratio < 5 else ("moderate" if missing_ratio < 20 else "high")

    # Multicollinearity
    multi_txt = "potential multicollinearity detected" if multicollinearity_flag else "no strong multicollinearity detected"

    # Outliers
    outlier_txt = "notable outliers present" if outlier_flag else "no severe outliers observed"

    recommendation = []
    if balance_txt == "imbalanced":
        recommendation.append("consider resampling strategies (undersampling or class-weighted models)")
    if miss_eval != "low":
        recommendation.append("impute or collect more data to reduce missingness")
    if multicollinearity_flag:
        recommendation.append("apply feature selection or regularization to mitigate multicollinearity")
    if outlier_flag:
        recommendation.append("apply robust scaling or outlier mitigation")
    if not recommendation:
        recommendation.append("dataset appears suitable for classification with standard preprocessing")

    return (
        f"Dataset appears {balance_txt}; {miss_txt} ({miss_eval}). "
        f"{multi_txt}; {outlier_txt}. Recommendation: {', '.join(recommendation)}."
    )


# =========================
# Main routine
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name (optional)")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"], help="Scaler type")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 80)
    print("AUTOMATED DATA ANALYSIS PIPELINE")
    print("=" * 80)
    print("\n[INFO] Loading dataset...")

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file at {args.file}: {e}")
        sys.exit(1)

    print(f"[INFO] Dataset loaded from: {args.file}")
    print()

    # --------------------------------------------------------------------------------
    # 1. Exploratory Data Analysis (EDA)
    # --------------------------------------------------------------------------------
    print("#" * 80)
    print("1) EXPLORATORY DATA ANALYSIS (EDA)")
    print("#" * 80)

    # Basic info
    print("\n[EDA] Dataset Shape:", df.shape)
    print("\n[EDA] Column Types:")
    print(df.dtypes)

    print("\n[EDA] Missing Values per Column:")
    print(df.isna().sum())

    print("\n[EDA] Duplicate Rows:", df.duplicated().sum())

    # Descriptive statistics
    print("\n[EDA] Descriptive Statistics (Numeric):")
    print(df.describe(include=[np.number]).T)

    print("\n[EDA] Descriptive Statistics (Categorical/Object):")
    try:
        print(df.describe(include=['object', 'category', 'bool']).T)
    except Exception:
        print("[EDA] No categorical/object columns or describe failed.")

    # Target identification
    target_col = infer_target_column(df, args.target)
    if target_col not in df.columns:
        print(f"[ERROR] Target column '{target_col}' not found after inference.")
        sys.exit(1)

    print(f"\n[EDA] Target variable detected: '{target_col}'")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Target distribution
    print("\n[EDA] Target Distribution:")
    if is_numeric_series(y) and y.nunique() > 15:
        print(y.describe())
        plt.figure()
        sns.histplot(y, kde=True)
        plt.title(f"Target Distribution: {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
    else:
        print(y.value_counts(dropna=False))
        plt.figure()
        sns.countplot(x=y.astype(str))
        plt.title(f"Target Distribution: {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Split features by type
    num_cols = [c for c in X.columns if is_numeric_series(X[c])]
    cat_cols = [c for c in X.columns if not is_numeric_series(X[c]) or is_categorical_series(X[c])]

    # Histograms/KDE for numerical features
    if len(num_cols) > 0:
        print("\n[EDA] Plotting Histograms/KDE for Numerical Features...")
        for col in num_cols:
            plt.figure()
            sns.histplot(X[col], kde=True)
            plt.title(f"Histogram/KDE - {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

        # Boxplots for outlier detection
        print("\n[EDA] Plotting Boxplots for Outlier Detection (Numerical Features)...")
        for col in num_cols:
            plt.figure()
            sns.boxplot(x=X[col])
            plt.title(f"Boxplot - {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()
    else:
        print("[EDA] No numerical features found.")

    # Correlation heatmap (numerical)
    if len(num_cols) > 1:
        print("\n[EDA] Correlation Heatmap (Numerical Features)...")
        corr = X[num_cols].corr(numeric_only=True)
        plt.figure(figsize=(min(12, 0.75 * len(num_cols) + 4), min(10, 0.75 * len(num_cols) + 4)))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (Features)")
        plt.tight_layout()
        plt.show()
    else:
        print("[EDA] Not enough numerical features for correlation heatmap.")

    # Countplots for categorical variables
    if len(cat_cols) > 0:
        print("\n[EDA] Countplots for Categorical Features...")
        for col in cat_cols:
            if X[col].nunique(dropna=False) > 50:
                print(f"[EDA] Skipping countplot for '{col}' (high cardinality: {X[col].nunique()})")
                continue
            plt.figure()
            sns.countplot(x=X[col].astype(str))
            plt.title(f"Countplot - {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    else:
        print("[EDA] No categorical features found.")

    # --------------------------------------------------------------------------------
    # 2. Data Preprocessing and Feature Engineering
    # --------------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("2) DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("#" * 80)

    # Missing value handling + Encoding + Scaling
    print("\n[PREP] Setting up imputers, encoders, and scalers...")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", select_scaler(args.scaler))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )

    # Class imbalance detection
    print("\n[PREP] Class Imbalance Check (only applicable if classification-like target)...")
    y_is_classification_like = (not is_numeric_series(y)) or (y.nunique() <= 20)
    if y_is_classification_like:
        # If target is categorical-like but dtype numeric, do not scale/encode y; just ensure labels are simple
        y_encoded = y.copy()
        if not is_numeric_series(y_encoded) or not np.issubdtype(y_encoded.dropna().dtype, np.number):
            # Use LabelEncoder for y if needed
            try:
                le_y = LabelEncoder()
                y_encoded = pd.Series(le_y.fit_transform(y_encoded.astype(str)), index=y.index, name=target_col)
                print("[PREP] Target label-encoded for internal analysis.")
            except Exception:
                print("[PREP] Label encoding target failed; proceeding with raw target for balance report.")
                y_encoded = y.copy()

        print("[PREP] Class distribution BEFORE resampling:")
        print(class_balance_report(y_encoded))

        X_bal, y_bal = X.copy(), y_encoded.copy()
        if needs_undersampling(y_encoded):
            print("[PREP] Imbalance detected -> Applying Random Undersampling...")
            X_bal, y_bal = random_undersample(X, y_encoded)
            print("[PREP] Class distribution AFTER undersampling:")
            print(class_balance_report(y_bal))
        else:
            print("[PREP] No severe imbalance detected; skipping resampling.")
    else:
        print("[PREP] Target appears continuous; skipping class imbalance step.")
        X_bal, y_bal = X.copy(), y.copy()

    print("\n[PREP] Fitting preprocessing transformers and transforming features...")
    X_processed = preprocessor.fit_transform(X_bal)
    processed_feature_names: List[str] = []

    # Retrieve feature names after OneHotEncoder
    try:
        num_names = num_cols
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_imputer = preprocessor.named_transformers_["cat"].named_steps["imputer"]
        # After imputation, categories belong to original columns:
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        processed_feature_names = list(num_names) + cat_feature_names
    except Exception:
        processed_feature_names = [f"f{i}" for i in range(X_processed.shape[1])]

    print(f"[PREP] Processed feature matrix shape: {X_processed.shape}")
    print(f"[PREP] Number of processed features: {len(processed_feature_names)}")

    # --------------------------------------------------------------------------------
    # 3. Feature Relationship Analysis
    # --------------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("3) FEATURE RELATIONSHIP ANALYSIS")
    print("#" * 80)

    # Build a numeric-only DataFrame of processed features for correlation with target.
    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names, index=X_bal.index)

    # Ensure y_bal numeric if binary for point-biserial; otherwise keep as is
    y_corr = y_bal.copy()
    if is_binary_series(y_corr):
        y_corr, _ = normalize_binary_labels(y_corr)

    print("\n[REL] Computing correlations between NUMERICAL features and the target...")
    corr_table = correlation_with_target_table(X_processed_df, y_corr)
    if len(corr_table) == 0:
        print("[REL] No correlations computed (insufficient numeric features or data).")
    else:
        print("\n[REL] Correlation Table (top 20 rows):")
        print(corr_table.head(20))

    # Visual correlation plots
    # Pairplot for up to 5 strongest correlated features (absolute correlation)
    try:
        if len(corr_table) > 0:
            # Select top 5 by absolute correlation (take first method per feature to avoid duplicates)
            top_features = (
                corr_table
                .drop_duplicates(subset=["feature"], keep="first")
                .assign(abs_corr=lambda d: d["correlation"].abs())
                .sort_values("abs_corr", ascending=False)
                .head(5)["feature"]
                .tolist()
            )
            if len(top_features) >= 2:
                print("\n[REL] Pairplot of top correlated features (up to 5) and target (if categorical-like)...")
                plot_df = X_processed_df[top_features].copy()
                plot_df[target_col] = y_bal.values
                # For hue, ensure categorical-like target not overly high cardinality
                hue_arg = target_col if (not is_numeric_series(y_bal) or y_bal.nunique() <= 10) else None
                sns.pairplot(plot_df, hue=hue_arg, diag_kind="hist", corner=True)
                plt.suptitle("Pairplot - Top Correlated Features", y=1.02)
                plt.tight_layout()
                plt.show()
    except Exception as e:
        print(f"[REL] Pairplot generation failed: {e}")

    # Heatmap of correlations among selected features
    try:
        if len(corr_table) > 0:
            top_features = (
                corr_table
                .drop_duplicates(subset=["feature"], keep="first")
                .assign(abs_corr=lambda d: d["correlation"].abs())
                .sort_values("abs_corr", ascending=False)
                .head(10)["feature"]
                .tolist()
            )
            if len(top_features) >= 2:
                print("\n[REL] Heatmap of correlations among top features...")
                corr_top = X_processed_df[top_features].corr()
                plt.figure(figsize=(10, 7))
                sns.heatmap(corr_top, annot=False, cmap="coolwarm", center=0)
                plt.title("Correlation Heatmap - Top Features")
                plt.tight_layout()
                plt.show()
    except Exception as e:
        print(f"[REL] Heatmap generation failed: {e}")

    # Scatterplots vs target (if target continuous or binary encoded)
    try:
        if is_numeric_series(y_corr):
            # pick up to 4 top features for scatter
            top_features_sc = (
                corr_table
                .drop_duplicates(subset=["feature"], keep="first")
                .assign(abs_corr=lambda d: d["correlation"].abs())
                .sort_values("abs_corr", ascending=False)
                .head(4)["feature"]
                .tolist()
            )
            for f in top_features_sc:
                plt.figure()
                plt.scatter(X_processed_df[f], y_corr)
                plt.xlabel(f)
                plt.ylabel(target_col)
                plt.title(f"Scatter: {f} vs {target_col}")
                plt.tight_layout()
                plt.show()
    except Exception as e:
        print(f"[REL] Scatterplot generation failed: {e}")

    # --------------------------------------------------------------------------------
    # 4. Time Measurement
    # --------------------------------------------------------------------------------
    end_time = time.time()
    runtime_seconds = int(round(end_time - start_time))
    minutes = runtime_seconds // 60
    seconds = runtime_seconds % 60
    print("\n" + "#" * 80)
    print("4) TIME MEASUREMENT")
    print("#" * 80)
    print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds")

    # --------------------------------------------------------------------------------
    # 5. Conclusion on Data Suitability
    # --------------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("5) CONCLUSION ON DATA SUITABILITY FOR CLASSIFICATION")
    print("#" * 80)

    # Compute simple indicators
    overall_missing_ratio = float(df.isna().sum().sum()) / (df.shape[0] * df.shape[1]) * 100 if df.size > 0 else 0.0

    # Multicollinearity heuristic: max |corr| among numeric features > 0.9
    multi_flag = False
    if len(num_cols) > 1:
        corr_abs = X[num_cols].corr().abs()
        np.fill_diagonal(corr_abs.values, 0.0)
        multi_flag = (corr_abs.values.max() > 0.9)

    # Outliers heuristic: any numeric feature with >1% values outside Q1-1.5IQR or Q3+1.5IQR
    outlier_flag = False
    for col in num_cols:
        s = X[col].dropna()
        if len(s) > 0:
            q1, q3 = np.percentile(s, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            prop = ((s < lower) | (s > upper)).mean() if iqr > 0 else 0.0
            if prop > 0.01:
                outlier_flag = True
                break

    summary = brief_suitability_assessment(y, overall_missing_ratio, multi_flag, outlier_flag)
    print(summary)

    print("\n[DONE] Automated analysis complete.")


if __name__ == "__main__":
    main()
