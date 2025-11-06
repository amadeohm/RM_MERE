#!/usr/bin/env python3
# analysis.py
# Complete Automated Data Analysis Script
# Requirements: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
# Usage:
#   python analysis.py --data /path/to/data.csv --target TargetColumnName
# If --target is omitted, the script will try to infer it.

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (9, 6)
sns.set(style="whitegrid")


def infer_target_column(df: pd.DataFrame):
    # Heuristics: explicit names, last column, binary-looking columns
    candidates = [c for c in df.columns if c.lower() in ("target", "label", "class", "y", "outcome")]
    if candidates:
        return candidates[0]
    # Binary-looking integer columns
    for c in df.columns[::-1]:
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            unique_vals = df[c].dropna().unique()
            if 2 <= unique_vals.size <= 10:
                return c
    # Fallback: last column
    return df.columns[-1]


def is_classification_target(s: pd.Series):
    # Decide task type based on dtype and cardinality
    if s.dtype.name in ["object", "category", "bool"]:
        return True
    unique = s.dropna().unique()
    # Integer with small number of unique values -> classification
    if pd.api.types.is_integer_dtype(s) and unique.size <= 20:
        return True
    return False


def summarize_missingness(df: pd.DataFrame):
    miss = df.isna().mean()
    overall = miss.mean()
    return miss.sort_values(ascending=False), overall


def basic_eda(df: pd.DataFrame, target: str):
    print("\n" + "=" * 80)
    print("1) EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    # Shape and dtypes
    print("\nDataset Shape:", df.shape)
    print("\nColumn Types:")
    print(df.dtypes)

    # Missing values
    miss_by_col, overall_miss = summarize_missingness(df)
    print("\nMissing Values per Column (fraction):")
    print(miss_by_col)
    print(f"\nOverall Missingness: {overall_miss:.3%}")

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {dup_count}")

    # Descriptive statistics
    print("\nBasic Descriptive Statistics (Numerical):")
    print(df.select_dtypes(include=[np.number]).describe().T)

    # Target distribution
    if target in df.columns:
        print(f"\nTarget Column: {target}")
        if is_classification_target(df[target]):
            print("Target appears to be CATEGORICAL/CLASSIFICATION.")
            print("\nTarget Value Counts:")
            print(df[target].value_counts(dropna=False))
            plt.figure()
            sns.countplot(x=df[target].astype(str))
            plt.title("Target Distribution (Countplot)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
        else:
            print("Target appears to be NUMERICAL/REGRESSION.")
            print("\nTarget Summary:")
            print(df[target].describe())
            plt.figure()
            sns.histplot(df[target].dropna(), kde=True)
            plt.title("Target Distribution (Histogram + KDE)")
            plt.tight_layout()
            plt.show()

    # Numerical features histograms / KDE
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    if num_cols:
        print("\nPlotting Histograms/KDE for Numerical Features...")
        n_show = min(len(num_cols), 12)
        for c in num_cols[:n_show]:
            plt.figure()
            sns.histplot(df[c].dropna(), kde=True)
            plt.title(f"{c} - Histogram + KDE")
            plt.tight_layout()
            plt.show()

        # Boxplots for outlier detection
        print("\nPlotting Boxplots for Outlier Detection (Numerical Features)...")
        for c in num_cols[:n_show]:
            plt.figure()
            sns.boxplot(x=df[c])
            plt.title(f"{c} - Boxplot (Outliers)")
            plt.tight_layout()
            plt.show()

        # Correlation heatmap among numerical features
        print("\nCorrelation Heatmap (Numerical Features)...")
        corr = df[num_cols].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (Numerical Features)")
        plt.tight_layout()
        plt.show()

    # Categorical countplots
    cat_cols = [c for c in df.select_dtypes(include=["object", "category", "bool"]).columns if c != target]
    if cat_cols:
        print("\nPlotting Countplots for Categorical Variables...")
        n_show = min(len(cat_cols), 12)
        for c in cat_cols[:n_show]:
            plt.figure()
            sns.countplot(x=df[c].astype(str))
            plt.title(f"{c} - Countplot")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()


def preprocess_and_engineer(df: pd.DataFrame, target: str, scale_type: str = "standard"):
    print("\n" + "=" * 80)
    print("2) DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("=" * 80)

    # Separate target
    y = df[target]
    X = df.drop(columns=[target])

    # Parse datetime columns; engineer basic features
    datetime_cols = []
    for c in X.columns:
        if np.issubdtype(X[c].dtype, np.datetime64):
            datetime_cols.append(c)
        elif X[c].dtype == object:
            # attempt to parse datetimes
            try:
                parsed = pd.to_datetime(X[c], errors="raise", infer_datetime_format=True)
                # if many non-na timestamps, treat as datetime
                if parsed.notna().mean() > 0.8:
                    X[c] = parsed
                    datetime_cols.append(c)
            except Exception:
                pass
    if datetime_cols:
        print("\nDetected datetime columns and engineered features (year, month, day, dow):", datetime_cols)
        for c in datetime_cols:
            X[f"{c}__year"] = X[c].dt.year
            X[f"{c}__month"] = X[c].dt.month
            X[f"{c}__day"] = X[c].dt.day
            X[f"{c}__dow"] = X[c].dt.dayofweek
        X = X.drop(columns=datetime_cols)

    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("\nNumerical Features:", num_features if num_features else "None")
    print("Categorical Features:", cat_features if cat_features else "None")

    # Imputers
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # Encoders
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Scalers
    if scale_type == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        scaler = MinMaxScaler()

    transformers = []
    if num_features:
        transformers.append(("num", Pipeline(steps=[("imputer", num_imputer), ("scaler", scaler)]), num_features))
    if cat_features:
        transformers.append(("cat", Pipeline(steps=[("imputer", cat_imputer), ("encoder", ohe)]), cat_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    print("\nFitting preprocessor and transforming features...")
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after encoding
    feature_names = []
    if num_features:
        feature_names.extend(num_features)
    if cat_features:
        try:
            ohe_feature_names = list(preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_features))
        except Exception:
            ohe_feature_names = []
        feature_names.extend(ohe_feature_names)

    X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    # Encode target for classification if needed
    y_processed = y.copy()
    le = None
    is_classif = is_classification_target(y)
    if is_classif:
        if y_processed.dtype.name not in ["int64", "int32", "int16", "int8", "uint8", "uint16", "uint32", "category", "bool"]:
            le = LabelEncoder()
            y_processed = le.fit_transform(y_processed.astype(str))
            print("\nApplied Label Encoding to target.")
        else:
            # ensure contiguous integers starting at 0 for clarity
            le = LabelEncoder()
            y_processed = le.fit_transform(y_processed)
        class_counts = pd.Series(y_processed).value_counts().sort_index()
        print("\nClass Distribution (before resampling):")
        print(class_counts)

        # Detect class imbalance (majority / minority > 1.5)
        imbalance_ratio = class_counts.max() / max(1, class_counts.min())
        if imbalance_ratio > 1.5:
            print(f"\nDetected class imbalance (max/min ratio = {imbalance_ratio:.2f}). Applying RANDOM UNDERSAMPLING.")
            # Random undersampling to match minority class count
            min_count = class_counts.min()
            idx_balanced = []
            for cls in class_counts.index:
                cls_idx = np.where(y_processed == cls)[0]
                chosen = np.random.choice(cls_idx, size=min_count, replace=False)
                idx_balanced.append(chosen)
            idx_balanced = np.concatenate(idx_balanced)
            X_bal = X_processed.iloc[idx_balanced].reset_index(drop=True)
            y_bal = pd.Series(y_processed[idx_balanced]).reset_index(drop=True)
            print("\nClass Distribution (after undersampling):")
            print(y_bal.value_counts().sort_index())
            return X_processed, y_processed, X_bal, y_bal, preprocessor, le
        else:
            print("\nNo significant class imbalance detected (undersampling not applied).")
    else:
        print("\nTarget treated as NUMERICAL (regression). Skipping class imbalance step.")

    return X_processed, y_processed, None, None, preprocessor, le


def feature_relationship_analysis(df: pd.DataFrame, target: str):
    print("\n" + "=" * 80)
    print("3) FEATURE RELATIONSHIP ANALYSIS")
    print("=" * 80)

    results = []
    y = df[target]
    num_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    is_classif = is_classification_target(y)

    if not num_feats:
        print("\nNo numerical features available for correlation with the target.")
        return pd.DataFrame(columns=["feature", "method", "correlation", "p_value"])

    print("\nComputing correlations between NUMERICAL features and the TARGET...")
    for c in num_feats:
        x = df[c]
        # Drop rows with NaNs in either
        mask = x.notna() & y.notna()
        x_ = x[mask]
        y_ = y[mask]

        try:
            if is_classif:
                # If binary classes: point-biserial. If multiclass, use Spearman as a fallback.
                y_unique = np.unique(y_)
                if y_unique.size == 2:
                    # Ensure binary encoding 0/1 for point-biserial
                    y_bin = pd.Categorical(y_).codes
                    r, p = stats.pointbiserialr(x_, y_bin)
                    method = "Point-Biserial"
                else:
                    r, p = stats.spearmanr(x_, pd.Categorical(y_).codes)
                    method = "Spearman (multiclass fallback)"
            else:
                # Regression target: Pearson and Spearman (report Pearson)
                r, p = stats.pearsonr(x_, y_)
                method = "Pearson"
        except Exception as e:
            r, p, method = np.nan, np.nan, f"Error: {e}"

        results.append({"feature": c, "method": method, "correlation": r, "p_value": p})

    corr_df = pd.DataFrame(results).sort_values(by="correlation", key=lambda s: s.abs(), ascending=False)
    print("\nCorrelation Results (Numerical Features vs Target):")
    print(corr_df.to_string(index=False))

    # Visual correlation plots
    # Pairplot (sample up to 5 strongest correlated features)
    top_feats = corr_df["feature"].head(min(5, len(corr_df))).tolist()
    if top_feats:
        print("\nGenerating Pairplot for Top Correlated Features with Target (if target numeric, included)...")
        pairplot_cols = top_feats + ([target] if not is_classif else [])
        try:
            sns.pairplot(df[pairplot_cols].dropna())
            plt.suptitle("Pairplot - Top Correlated Features", y=1.02)
            plt.show()
        except Exception:
            pass

        # Scatterplots against target (only if target numeric)
        if not is_classif:
            for c in top_feats:
                plt.figure()
                sns.scatterplot(x=df[c], y=df[target])
                plt.title(f"Scatterplot: {c} vs {target}")
                plt.tight_layout()
                plt.show()

    # Heatmap of correlations between numerical features and target (numeric target only)
    if not is_classif:
        print("\nHeatmap: Correlation of Numerical Features with Target (Pearson)")
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
        corr_all = df[num_cols].corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_all[[target]].sort_values(by=target, ascending=False), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation with Target")
        plt.tight_layout()
        plt.show()

    return corr_df


def conclusion_on_data_suitability(df: pd.DataFrame, target: str, X_processed: pd.DataFrame, y_processed, X_balanced=None, y_balanced=None):
    print("\n" + "=" * 80)
    print("5) CONCLUSION ON DATA SUITABILITY FOR CLASSIFICATION")
    print("=" * 80)

    # Missingness
    miss_by_col, overall_miss = summarize_missingness(df)
    high_missing_cols = miss_by_col[miss_by_col > 0.3].index.tolist()

    # Class balance (only if classification)
    class_note = "N/A (regression target)"
    if is_classification_target(df[target]):
        y_enc = pd.Series(y_processed)
        class_counts = y_enc.value_counts()
        imb_ratio = class_counts.max() / max(1, class_counts.min())
        class_note = f"Max/Min class ratio: {imb_ratio:.2f}. "
        if X_balanced is not None and y_balanced is not None:
            class_note += "Undersampling applied."
        else:
            class_note += "No resampling applied."

    # Multicollinearity (simple heuristic using correlation > 0.9 among numerical)
    num_df = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    high_corr_pairs = []
    if not num_df.empty:
        corr = num_df.corr()
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if j <= i:
                    continue
                if abs(corr.loc[c1, c2]) > 0.9:
                    high_corr_pairs.append((c1, c2, corr.loc[c1, c2]))

    noise_hint = "Potential noise/outliers detected" if (df.select_dtypes(include=[np.number]).apply(lambda s: (np.abs(stats.zscore(s.dropna())) > 3).mean() if s.dropna().size > 0 else 0).mean() > 0.02) else "No strong outlier signal"

    print("\nSUMMARY:")
    print(f"- Target: {target} | Type: {'Classification' if is_classification_target(df[target]) else 'Regression'}")
    print(f"- Overall Missingness: {overall_miss:.2%} | High-missing columns (>30%): {high_missing_cols if high_missing_cols else 'None'}")
    print(f"- Class Balance: {class_note}")
    print(f"- Multicollinearity: {'Yes' if high_corr_pairs else 'No'}")
    if high_corr_pairs:
        print(f"  Highly correlated pairs (|r|>0.9): {[(a,b,round(r,3)) for a,b,r in high_corr_pairs[:10]]} {'...' if len(high_corr_pairs)>10 else ''}")
    print(f"- Outliers/Noise: {noise_hint}")

    suitability = "LIKELY SUITABLE for classification" if is_classification_target(df[target]) and overall_miss < 0.3 else "POTENTIALLY UNSUITABLE without further cleaning/feature engineering"
    print(f"\nAUTOMATED ASSESSMENT: {suitability}")


def time_to_str(seconds: float):
    minutes = int(seconds // 60)
    secs = int(round(seconds - minutes * 60))
    return f"{minutes} minutes and {secs} seconds"


def main():
    parser = argparse.ArgumentParser(description="Automated EDA, Preprocessing, and Feature Relationship Analysis")
    parser.add_argument("--data", type=str, default="data.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name (optional)")
    parser.add_argument("--scaler", type=str, choices=["standard", "minmax"], default="standard", help="Scaling method")
    args = parser.parse_args()

    start_time = time.time()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[Warning] File {data_path} not found. Trying default '/mnt/data/diabetes_dataset.csv' if available...")
        fallback = Path("/mnt/data/diabetes_dataset.csv")
        if fallback.exists():
            data_path = fallback
        else:
            print("ERROR: Dataset file not found. Please provide a valid --data path.")
            sys.exit(1)

    print("=" * 80)
    print("AUTOMATED DATA ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"\nLoading dataset from: {data_path.resolve()}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Failed to read CSV with default options due to: {e}")
        try:
            df = pd.read_csv(data_path, sep=";")
            print("Loaded with semicolon delimiter ';'.")
        except Exception as e2:
            print(f"Failed with ';' as well: {e2}")
            sys.exit(1)

    # Infer target if not provided
    target = args.target if args.target in df.columns else infer_target_column(df)
    if args.target and args.target not in df.columns:
        print(f"[Warning] Provided target '{args.target}' not found. Using inferred target: '{target}'")
    print(f"\nUsing target column: {target}")

    # SECTION 1: EDA
    basic_eda(df, target)

    # SECTION 2: Preprocessing / Feature Engineering
    X_processed, y_processed, X_bal, y_bal, preprocessor, label_encoder = preprocess_and_engineer(df, target, scale_type=args.scaler)

    # Show before/after counts if classification and resampled
    if is_classification_target(df[target]):
        before_counts = pd.Series(y_processed).value_counts().sort_index()
        print("\nClass Counts (Before):")
        print(before_counts)
        if y_bal is not None:
            after_counts = y_bal.value_counts().sort_index()
            print("\nClass Counts (After Undersampling):")
            print(after_counts)

    # SECTION 3: Feature Relationship Analysis
    # For correlation analysis, operate on the ORIGINAL df to respect types
    corr_df = feature_relationship_analysis(df.copy(), target)

    # SECTION 4: TIME MEASUREMENT
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("4) TIME MEASUREMENT")
    print("=" * 80)
    print(f"\nTotal Simulated Runtime: {time_to_str(elapsed)}")

    # SECTION 5: Conclusion
    conclusion_on_data_suitability(df, target, X_processed, y_processed, X_bal, y_bal)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
