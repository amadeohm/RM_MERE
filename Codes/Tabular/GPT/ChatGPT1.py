#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated Health Analytics & Predictive Modelling EDA Pipeline
- Loads a dataset and performs: EDA, preprocessing/feature engineering, feature-relationship analysis,
  runtime measurement, and an auto-generated conclusion about classification suitability.
- Uses only: pandas, numpy, matplotlib, seaborn, scipy.stats, sklearn
- Ready to run as: python analysis.py --path /path/to/data.csv --target target_column_name
"""

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# Helper functions
# -----------------------------

def infer_target_column(df: pd.DataFrame, user_target: str = None) -> str:
    if user_target is not None and user_target in df.columns:
        return user_target

    candidates = [c for c in df.columns if c.lower() in ("target", "label", "class", "outcome", "y")]
    if candidates:
        return candidates[0]

    # Fallback: last column
    return df.columns[-1]

def is_binary_series(s: pd.Series) -> bool:
    return s.dropna().nunique() == 2

def is_categorical_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        # treat low-cardinality numeric as categorical (e.g., 0/1, 0/1/2)
        return s.dropna().nunique() <= 10
    return True

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    miss_pct = (miss / len(df)) * 100
    return pd.DataFrame({"missing_count": miss, "missing_percent": miss_pct}).sort_values("missing_percent", ascending=False)

def impute_values(df: pd.DataFrame, num_cols, cat_cols):
    df_imputed = df.copy()
    for col in num_cols:
        median_val = df_imputed[col].median()
        df_imputed[col] = df_imputed[col].fillna(median_val)
    for col in cat_cols:
        mode_val = df_imputed[col].mode(dropna=True)
        mode_val = mode_val.iloc[0] if len(mode_val) else np.nan
        df_imputed[col] = df_imputed[col].fillna(mode_val)
    return df_imputed

def manual_random_undersample(X, y, random_state=42):
    """
    Simple undersampling to the size of the minority class for multi-class/binary.
    """
    rng = np.random.RandomState(random_state)
    df_xy = X.copy()
    df_xy["_temp_target_"] = y.values if isinstance(y, pd.Series) else y
    counts = df_xy["_temp_target_"].value_counts()
    min_size = counts.min()
    idx_list = []
    for cls, cnt in counts.items():
        idx_cls = df_xy[df_xy["_temp_target_"] == cls].index
        if cnt > min_size:
            idx_sel = rng.choice(idx_cls, size=min_size, replace=False)
            idx_list.append(pd.Index(idx_sel))
        else:
            idx_list.append(idx_cls)
    idx_final = idx_list[0]
    for i in range(1, len(idx_list)):
        idx_final = idx_final.union(idx_list[i])
    df_balanced = df_xy.loc[idx_final].copy()
    y_bal = df_balanced["_temp_target_"].copy()
    X_bal = df_balanced.drop(columns=["_temp_target_"])
    return X_bal, y_bal

def compute_correlations(numeric_df: pd.DataFrame, target: pd.Series):
    """
    Returns a DataFrame with correlation coefficients and p-values for each numeric feature vs target
    using appropriate tests: point-biserial for binary target; Pearson & Spearman otherwise.
    """
    results = []
    clean_numeric_df = numeric_df.copy()
    # Ensure aligned indices
    clean_numeric_df, target = clean_numeric_df.align(target, join="inner", axis=0)

    target_is_binary = is_binary_series(target)

    for col in clean_numeric_df.columns:
        x = clean_numeric_df[col]
        # Drop NaNs pairwise
        df_pair = pd.DataFrame({"x": x, "y": target}).dropna()
        if len(df_pair) < 3:
            continue

        if target_is_binary and pd.api.types.is_numeric_dtype(df_pair["y"]):
            # point-biserial requires numeric binary target
            # If target not numeric, label-encode
            y_vals = df_pair["y"]
            if not pd.api.types.is_numeric_dtype(y_vals):
                y_vals = LabelEncoder().fit_transform(y_vals)
            r_pb, p_pb = stats.pointbiserialr(df_pair["x"], y_vals)
            results.append({
                "feature": col,
                "method": "point-biserial",
                "correlation": r_pb,
                "p_value": p_pb
            })
        else:
            # Continuous or multiclass target: compute Pearson and Spearman
            # For non-numeric target, label-encode to numeric for rank-based measure
            y_vals = df_pair["y"]
            if not pd.api.types.is_numeric_dtype(y_vals):
                y_vals = LabelEncoder().fit_transform(y_vals)
            pearson_r, pearson_p = stats.pearsonr(df_pair["x"], y_vals)
            spearman_rho, spearman_p = stats.spearmanr(df_pair["x"], y_vals)
            results.append({
                "feature": col,
                "method": "pearson",
                "correlation": pearson_r,
                "p_value": pearson_p
            })
            results.append({
                "feature": col,
                "method": "spearman",
                "correlation": spearman_rho,
                "p_value": spearman_p
            })

    return pd.DataFrame(results).sort_values(by=["method", "correlation"], ascending=[True, False])

def print_section(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")

def plot_hist_kde(df, num_cols):
    if not num_cols:
        print("No numerical features for histogram/KDE.")
        return
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram & KDE: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

def plot_boxplots(df, num_cols):
    if not num_cols:
        print("No numerical features for boxplots.")
        return
    for col in num_cols:
        plt.figure()
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot (Outlier Detection): {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

def plot_corr_heatmap(df, num_cols):
    if len(num_cols) < 2:
        print("Not enough numerical features for correlation heatmap.")
        return
    corr = df[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(min(12, 1 + 0.6*len(num_cols)), min(10, 1 + 0.6*len(num_cols))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=False)
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout()
    plt.show()

def plot_countplots(df, cat_cols):
    if not cat_cols:
        print("No categorical features for countplots.")
        return
    for col in cat_cols:
        plt.figure()
        sns.countplot(y=col, data=df)
        plt.title(f"Countplot: {col}")
        plt.ylabel(col)
        plt.xlabel("Count")
        plt.tight_layout()
        plt.show()

def plot_pairplot(df, num_cols, target_col):
    subset_cols = [c for c in num_cols if c != target_col][:6]  # keep it readable
    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        subset = subset_cols + [target_col]
    else:
        subset = subset_cols
    if len(subset) >= 2:
        sns.pairplot(df[subset].dropna(), corner=True)
        plt.suptitle("Pairplot (subset of numerical features)", y=1.02)
        plt.show()
    else:
        print("Not enough numerical features for pairplot.")

def plot_scatter_or_box_by_target(df, num_cols, target_col):
    if target_col not in df.columns:
        return
    # If target is categorical -> boxplots of feature vs target
    target_series = df[target_col]
    if is_categorical_series(target_series):
        for col in num_cols[:6]:
            if col == target_col:
                continue
            plt.figure()
            sns.boxplot(x=target_col, y=col, data=df)
            plt.title(f"{col} vs {target_col} (Boxplot)")
            plt.tight_layout()
            plt.show()
    else:
        # target continuous -> scatterplots
        for col in num_cols[:6]:
            if col == target_col:
                continue
            plt.figure()
            sns.scatterplot(x=col, y=target_col, data=df)
            plt.title(f"{col} vs {target_col} (Scatter)")
            plt.tight_layout()
            plt.show()

def assess_suitability_for_classification(df, target_col, num_cols, cat_cols, corr_df):
    summary_parts = []
    # Balance
    if target_col in df.columns:
        target_counts = df[target_col].value_counts(dropna=False)
        max_prop = (target_counts.max() / target_counts.sum()) if target_counts.sum() else 1.0
        if max_prop > 0.7:
            summary_parts.append("Target appears imbalanced (>70% in one class).")
        else:
            summary_parts.append("Target balance looks acceptable.")
    else:
        summary_parts.append("Target not found; cannot assess balance.")

    # Missingness
    miss_pct = df.isna().mean().mean() * 100
    if miss_pct > 10:
        summary_parts.append(f"Non-trivial missingness (~{miss_pct:.1f}%).")
    else:
        summary_parts.append(f"Low overall missingness (~{miss_pct:.1f}%).")

    # Multicollinearity: check if any |corr| > 0.9 among numeric
    high_corr_flag = False
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True).abs()
        np.fill_diagonal(corr.values, 0.0)
        if (corr.values > 0.9).any():
            high_corr_flag = True
    summary_parts.append("Potential multicollinearity detected." if high_corr_flag else "No strong multicollinearity detected.")

    # Noise/outliers proxy: IQR-based count
    outlier_ratio = 0.0
    checked_cols = 0
    for col in num_cols[:10]:  # sample up to 10
        x = df[col].dropna()
        if len(x) < 5:
            continue
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
        outlier_ratio += ((x < lb) | (x > ub)).mean()
        checked_cols += 1
    if checked_cols > 0:
        outlier_ratio /= checked_cols
        if outlier_ratio > 0.1:
            summary_parts.append(f"Notable outlier presence (~{outlier_ratio*100:.1f}% of values).")
        else:
            summary_parts.append(f"Limited outlier presence (~{outlier_ratio*100:.1f}% of values).")

    # Signals from correlations
    if corr_df is not None and not corr_df.empty:
        top = corr_df.reindex(corr_df["correlation"].abs().sort_values(ascending=False).index).head(3)
        if (top["correlation"].abs() >= 0.2).any():
            summary_parts.append("Some features show meaningful correlation with target.")
        else:
            summary_parts.append("Weak correlations between numeric features and target.")

    verdict = "Overall: dataset is reasonably suitable for classification." \
              if ("Target balance looks acceptable." in summary_parts and
                  "No strong multicollinearity detected." in summary_parts) else \
              "Overall: dataset may require rebalancing/feature selection/cleaning before classification."

    print_section("5) CONCLUSION ON DATA SUITABILITY FOR CLASSIFICATION")
    print(" | ".join(summary_parts))
    print(verdict)
    print()  # newline


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Automated EDA & Preprocessing Pipeline")
    parser.add_argument("--path", type=str, default="data.csv", help="Path to CSV dataset (default: data.csv)")
    parser.add_argument("--target", type=str, default=None, help="Target column name (optional)")
    args = parser.parse_args()

    start_time = time.time()

    print_section("0) SETUP")
    print(f"Dataset path: {args.path}")
    if args.target:
        print(f"Requested target column: {args.target}")

    # 1. Load dataset
    print_section("1) EXPLORATORY DATA ANALYSIS (EDA)")
    print("Loading dataset using pandas...")
    df = pd.read_csv(args.path)
    print("Dataset loaded.\n")

    # Determine target column
    target_col = infer_target_column(df, args.target)
    print(f"Detected target column: {target_col}")

    # Basic info
    print("\n--- Shape ---")
    print(df.shape)

    print("\n--- Column Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    miss_table = summarize_missing(df)
    print(miss_table)

    print("\n--- Duplicates ---")
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count}")

    print("\n--- Descriptive Statistics (Numeric) ---")
    print(df.describe(include=[np.number]).T)

    print("\n--- Descriptive Statistics (Categorical) ---")
    try:
        print(df.describe(include=['object', 'category']).T)
    except Exception:
        print("No categorical columns or describe failed.")

    # Analyze target distribution
    if target_col in df.columns:
        print("\n--- Target Variable Distribution ---")
        print(df[target_col].value_counts(dropna=False))
        plt.figure()
        if is_categorical_series(df[target_col]):
            sns.countplot(x=target_col, data=df)
            plt.title(f"Target Distribution: {target_col}")
            plt.xticks(rotation=45)
        else:
            sns.histplot(df[target_col].dropna(), kde=True)
            plt.title(f"Target Distribution (continuous): {target_col}")
        plt.tight_layout()
        plt.show()
    else:
        print("\nTarget column not found in dataset; skipping target distribution plot.")

    # Identify numerical and categorical features
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    print("\n--- Feature Types ---")
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    # Plots
    print("\nGenerating plots: Histograms/KDE for numeric features...")
    plot_hist_kde(df, num_cols)

    print("Generating plots: Boxplots for outlier detection...")
    plot_boxplots(df, num_cols)

    print("Generating plots: Correlation heatmap...")
    plot_corr_heatmap(df, num_cols + ([target_col] if target_col in num_cols else []))

    print("Generating plots: Countplots for categorical variables...")
    plot_countplots(df, cat_cols)

    # 2. Data Preprocessing and Feature Engineering
    print_section("2) DATA PREPROCESSING & FEATURE ENGINEERING")

    # Handle missing values
    print("Imputing missing values (median for numeric, mode for categorical)...")
    df_imputed = impute_values(df, num_cols, cat_cols)
    print("Imputation complete.\n")

    # Encoding categorical features and scaling numeric features
    print("Encoding categorical features and scaling numeric features...")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, num_cols),
            ("cat", ohe, cat_cols)
        ],
        remainder="drop"
    )

    X = df_imputed[feature_cols]
    y = df_imputed[target_col] if target_col in df_imputed.columns else None

    # Encode target if object/category for downstream balance check consistency
    y_encoded = None
    if y is not None:
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y_encoded = pd.Series(le.fit_transform(y), index=y.index, name=target_col)
        else:
            y_encoded = y.copy()
    else:
        print("No target available, skipping resampling and correlation steps dependent on target.")

    # Fit-transform X
    X_processed = preprocessor.fit_transform(X)
    # Capture processed feature names
    processed_feature_names = []
    if num_cols:
        processed_feature_names.extend(num_cols)
    if cat_cols:
        try:
            processed_feature_names.extend(list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)))
        except Exception:
            processed_feature_names.extend([f"cat_{i}" for i in range(preprocessor.named_transformers_["cat"].transform(X[cat_cols]).shape[1])])

    print("Encoding & scaling complete.\n")

    # Detect and report class imbalance; apply undersampling if present
    X_balanced = None
    y_balanced = None
    if y_encoded is not None:
        print("--- Class Balance Check (Before) ---")
        before_counts = y_encoded.value_counts().sort_index()
        print(before_counts)

        imbalance = before_counts.max() / before_counts.sum() > 0.7  # >70% threshold
        if imbalance and is_categorical_series(y):
            print("\nImbalance detected. Applying random undersampling to minority class size...")
            # For undersampling we need original (not-transformed) X for interpretability; we will re-transform after
            X_under, y_under = manual_random_undersample(X, y_encoded)
            # Refit transform to maintain proper scaling on undersampled dataset (optional; we keep fit on full data for robustness)
            X_balanced_processed = preprocessor.transform(X_under)
            X_balanced = pd.DataFrame(X_balanced_processed, index=X_under.index, columns=processed_feature_names)
            y_balanced = y_under
            print("\n--- Class Balance Check (After) ---")
            print(y_balanced.value_counts().sort_index())
        else:
            print("\nNo severe imbalance detected or target not categorical. Skipping undersampling.")
            X_balanced = pd.DataFrame(X_processed, index=X.index, columns=processed_feature_names)
            y_balanced = y_encoded.copy()
    else:
        X_balanced = pd.DataFrame(X_processed, index=X.index, columns=processed_feature_names)

    # 3. Feature Relationship Analysis
    print_section("3) FEATURE RELATIONSHIP ANALYSIS")

    # Correlations between numerical features and the target
    corr_df = pd.DataFrame()
    if y is not None:
        num_with_target = [c for c in num_cols if c in df_imputed.columns]
        if num_with_target:
            print("Computing correlations between numerical features and the target variable...")
            corr_df = compute_correlations(df_imputed[num_with_target], df_imputed[target_col])
            if corr_df.empty:
                print("Correlation table is empty (insufficient data or non-numeric target).")
            else:
                print("\n--- Correlations & p-values ---")
                print(corr_df.to_string(index=False))
        else:
            print("No numerical features to correlate with target.")
    else:
        print("Target not available; skipping correlation computations.")

    # Visual correlation plots
    print("\nGenerating pairplot/scatter/box plots to visualize feature-target relationships...")
    plot_pairplot(df_imputed, num_cols, target_col)
    plot_scatter_or_box_by_target(df_imputed, num_cols, target_col)

    # 4. Time Measurement
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(round(elapsed % 60))

    print_section("4) TIME MEASUREMENT")
    print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds\n")

    # 5. Conclusion on Data Suitability
    assess_suitability_for_classification(df_imputed, target_col, num_cols, cat_cols, corr_df)

    print_section("DONE")
    print("Analysis completed successfully.")

if __name__ == "__main__":
    main()
