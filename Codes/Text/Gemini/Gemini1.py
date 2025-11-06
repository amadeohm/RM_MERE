import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import pointbiserialr, pearsonr
import warnings
import sys

# --- Script Parameters ---

# The dataset file to analyze.
# This script assumes the file is in the same directory.
DATASET_PATH = 'diabetes_dataset_nan.csv'

# The name of the target variable column for classification.
TARGET_VARIABLE = 'diagnosed_diabetes'

# Set a threshold for high correlation warnings (e.g., 0.8)
HIGH_CORR_THRESHOLD = 0.8

# Set imbalance threshold (e.g., 0.5 means one class is < 50% of the other)
IMBALANCE_THRESHOLD = 0.5

# --- End Parameters ---


def run_analysis(dataset_path, target_variable):
    """
    Runs the full data analysis pipeline.
    """
    
    print("==============================================")
    print("= Starting Automated Data Analysis Script =")
    print("==============================================")
    print(f"Loading dataset: {dataset_path}")
    print(f"Target variable: {target_variable}\n")

    # Store metrics for final conclusion
    analysis_metrics = {}

    # --- 1. Exploratory Data Analysis (EDA) ---
    print("\n--- 1. Exploratory Data Analysis (EDA) ---")
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {dataset_path}")
        print("Script execution halted.")
        sys.exit()

    print("\n[EDA] Dataset Shape:")
    print(df.shape)
    analysis_metrics['shape'] = df.shape

    print("\n[EDA] Dataset Info (Column Types):")
    df.info()

    print("\n[EDA] Missing Values per Column:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    total_missing = missing_values.sum()
    total_cells = np.product(df.shape)
    analysis_metrics['missing_percentage'] = (total_missing / total_cells) * 100

    print(f"\nTotal Missing Values: {total_missing} ({analysis_metrics['missing_percentage']:.2f}% of all data)")

    print("\n[EDA] Duplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"Found {duplicates} duplicate rows.")
    analysis_metrics['duplicates'] = duplicates

    print("\n[EDA] Descriptive Statistics (Numerical Features):")
    print(df.describe().to_string())

    if target_variable not in df.columns:
        print(f"ERROR: Target variable '{target_variable}' not found in dataset columns.")
        print("Script execution halted.")
        sys.exit()

    print(f"\n[EDA] Target Variable Distribution ('{target_variable}'):")
    print(df[target_variable].value_counts())
    print(df[target_variable].value_counts(normalize=True))
    analysis_metrics['target_distribution'] = df[target_variable].value_counts()
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_variable, data=df)
    plt.title(f'Distribution of Target Variable ({target_variable})')
    plt.tight_layout()
    plt.savefig('eda_target_distribution.png')
    plt.show()
    print("Saved plot to eda_target_distribution.png")

    # --- Identify Feature Types ---
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include='object').columns.tolist()

    # Remove target from feature lists
    if target_variable in numerical_features:
        numerical_features.remove(target_variable)
    if target_variable in categorical_features:
        categorical_features.remove(target_variable)

    print(f"\nIdentified {len(numerical_features)} numerical features.")
    print(f"Identified {len(categorical_features)} categorical features.")

    # --- EDA Plots ---
    if numerical_features:
        print("\n[EDA] Generating Histograms for Numerical Features...")
        df[numerical_features].hist(bins=20, figsize=(15, max(10, len(numerical_features) * 0.5)), layout=(-1, 4))
        plt.tight_layout()
        plt.savefig('eda_histograms.png')
        plt.show()
        print("Saved plot to eda_histograms.png")

        print("\n[EDA] Generating Boxplots for Numerical Features (Outlier Detection)...")
        df[numerical_features].plot(kind='box', subplots=True, layout=(-1, 4), figsize=(15, max(10, len(numerical_features) * 0.5)), sharex=False, sharey=False)
        plt.tight_layout()
        plt.savefig('eda_boxplots.png')
        plt.show()
        print("Saved plot to eda_boxplots.png")

        print("\n[EDA] Generating Correlation Heatmap (Numerical Features)...")
        plt.figure(figsize=(12, 10))
        # Include target if it's numeric for the heatmap
        heatmap_cols = numerical_features + ([target_variable] if df[target_variable].dtype != 'object' else [])
        corr_matrix = df[heatmap_cols].corr()
        analysis_metrics['corr_matrix'] = corr_matrix
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('eda_correlation_heatmap.png')
        plt.show()
        print("Saved plot to eda_correlation_heatmap.png")
    else:
        print("\n[EDA] No numerical features found to plot.")

    if categorical_features:
        print("\n[EDA] Generating Countplots for Categorical Features...")
        for i, col in enumerate(categorical_features):
            plt.figure(figsize=(10, 5))
            top_n = df[col].value_counts().index[:20] # Show top 20 categories
            sns.countplot(y=col, data=df, order=top_n)
            plt.title(f'Countplot for {col} (Top 20)')
            plt.tight_layout()
            filename = f'eda_countplot_{i}_{col.replace(" ", "_")[:20]}.png'
            plt.savefig(filename)
            plt.show()
            print(f"Saved plot to {filename}")
    else:
        print("\n[EDA] No categorical features found to plot.")

    
    # --- 2. Data Preprocessing and Feature Engineering ---
    print("\n--- 2. Data Preprocessing and Feature Engineering ---")

    # Separate features (X) and target (y)
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    # Encode target variable y if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"\nTarget variable '{target_variable}' was categorical and has been LabelEncoded.")
        print("Class mapping:", {cl: i for i, cl in enumerate(le.classes_)})
    
    # --- Define Preprocessing Pipelines ---
    
    # Pipeline for numerical features: impute (median) + scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features: impute (mode) + one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mode')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create the full preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any columns not listed
    )

    print("\n[Preprocessing] Applying preprocessing pipelines (Imputation, Scaling, OneHotEncoding)...")
    # Apply the preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after transformation
    try:
        ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        processed_feature_names = numerical_features + list(ohe_feature_names)
    except Exception:
        processed_feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
        print("Could not get feature names, using generic names.")

    # Convert processed data back to DataFrame for inspection
    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

    print("\n[Preprocessing] Data preprocessing complete.")
    print(f"Processed feature shape: {X_processed_df.shape}")

    # --- Class Imbalance Handling ---
    print("\n[Preprocessing] Checking for Class Imbalance...")
    
    # Use bincount for efficiency if y is numeric
    counts = np.bincount(y)
    if len(counts) == 0:
        print("Warning: Target variable seems to have no data.")
        imbalance_ratio_before = 0
    elif len(counts) == 1:
        print("Warning: Target variable has only one class.")
        imbalance_ratio_before = 0
    else:
        imbalance_ratio_before = np.min(counts) / np.max(counts)

    print(f"\nClass Distribution Before Resampling:\n{pd.Series(y).value_counts()}")
    print(f"Imbalance Ratio (min/max): {imbalance_ratio_before:.3f}")
    
    analysis_metrics['imbalance_ratio'] = imbalance_ratio_before
    
    if imbalance_ratio_before < IMBALANCE_THRESHOLD and len(counts) > 1:
        print(f"Class imbalance detected (ratio < {IMBALANCE_THRESHOLD}). Applying SMOTE...")
        smote = SMOTE(random_state=42)
        try:
            X_resampled, y_resampled = smote.fit_resample(X_processed_df, y)
            print("\nClass Distribution After SMOTE:")
            print(pd.Series(y_resampled).value_counts())
            analysis_metrics['resampling_applied'] = True
            X_final, y_final = X_resampled, y_resampled
        except Exception as e:
            print(f"Error during SMOTE: {e}. Skipping resampling.")
            analysis_metrics['resampling_applied'] = False
            X_final, y_final = X_processed_df, y
    else:
        print("Classes are relatively balanced or SMOTE is not applicable. No resampling applied.")
        analysis_metrics['resampling_applied'] = False
        X_final, y_final = X_processed_df, y
    
    print(f"\nFinal training data shape: X={X_final.shape}, y={y_final.shape}")


    # --- 3. Feature Relationship Analysis ---
    print("\n--- 3. Feature Relationship Analysis (Numerical vs. Target) ---")
    print("Note: This analysis uses original (unscaled) numerical data vs. original target.")

    correlation_results = []
    
    # Use the original, numeric-encoded target `y`
    for col in numerical_features:
        # Drop NaNs for this specific correlation pair
        valid_data = df[[col, target_variable]].dropna()
        valid_y = y[valid_data.index] # Get corresponding original y
        valid_col = valid_data[col]
        
        if valid_data.shape[0] < 2:
            corr, pval = np.nan, np.nan
        else:
            try:
                # Use point-biserial if target is binary, else Pearson
                if len(np.unique(valid_y)) == 2:
                    corr, pval = pointbiserialr(valid_col, valid_y)
                else:
                    corr, pval = pearsonr(valid_col, valid_y)
            except ValueError:
                corr, pval = np.nan, np.nan
        
        correlation_results.append({'Feature': col, 'Correlation': corr, 'P-Value': pval})

    if correlation_results:
        corr_table = pd.DataFrame(correlation_results).sort_values(by='Correlation', key=abs, ascending=False)
        print("\nCorrelation of Numerical Features with Target:")
        print(corr_table.to_string())
        
        # Plot top 5
        top_features = corr_table.head(5)['Feature'].tolist()
        print(f"\n[Feature Analysis] Plotting for top 5 correlated features: {top_features}")
        for i, col in enumerate(top_features):
            plt.figure(figsize=(8, 5))
            # Boxplot is good for binary target, scatter for continuous
            if len(np.unique(y)) == 2:
                sns.boxplot(x=target_variable, y=col, data=df)
            else:
                sns.scatterplot(x=col, y=target_variable, data=df)
            plt.title(f'{col} vs. {target_variable}')
            plt.tight_layout()
            filename = f'feature_analysis_plot_{i}_{col}.png'
            plt.savefig(filename)
            plt.show()
            print(f"Saved plot to {filename}")
    else:
        print("\nNo numerical features to analyze correlation with target.")


    # --- 5. Conclusion on Data Suitability ---
    # (Section 4 is Time, computed at the end)
    print("\n--- 5. Conclusion on Data Suitability ---")
    print("Automated Data Suitability Assessment:")

    # Missingness
    print(f"\n- Missingness: {analysis_metrics['missing_percentage']:.2f}% of all cells were missing.")
    if analysis_metrics['missing_percentage'] > 20:
        print("  - WARNING: High missingness. Model performance may be affected. Advanced imputation or feature removal should be considered.")
    elif analysis_metrics['missing_percentage'] > 5:
        print("  - INFO: Moderate missing data. Was handled with median/mode imputation.")
    else:
        print("  - INFO: Low missing data. Unlikely to be a major issue.")
        
    # Balance
    print(f"\n- Class Balance: Initial imbalance ratio (min/max) was {analysis_metrics['imbalance_ratio']:.3f}.")
    if analysis_metrics['imbalance_ratio'] < IMBALANCE_THRESHOLD and analysis_metrics.get('resampling_applied', False):
        print("  - WARNING: Significant class imbalance was detected and corrected using SMOTE.")
    elif analysis_metrics['imbalance_ratio'] < IMBALANCE_THRESHOLD:
        print("  - WARNING: Significant class imbalance was detected but NOT corrected (e.g., error in SMOTE).")
    else:
        print("  - INFO: The dataset is reasonably balanced.")
        
    # Multicollinearity
    if 'corr_matrix' in analysis_metrics:
        corr_mat = analysis_metrics['corr_matrix'].abs()
        upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        high_corr_pairs = [column for column in upper_tri.columns if any(upper_tri[column] > HIGH_CORR_THRESHOLD)]
        analysis_metrics['high_corr_features'] = high_corr_pairs
        print(f"\n- Multicollinearity: Found {len(high_corr_pairs)} numerical features with correlation > {HIGH_CORR_THRESHOLD} with another feature.")
        if len(high_corr_pairs) > 0:
            print(f"  - WARNING: Potential multicollinearity detected. Features to review: {high_corr_pairs}. Consider VIF analysis or feature selection.")
        else:
            print("  - INFO: No strong multicollinearity detected between numerical features.")
    
    # Noise/Outliers
    print("\n- Noise/Outliers: Boxplots were generated for visual inspection. Review 'eda_boxplots.png' for significant outliers that may impact model performance.")
    
    # Overall
    print("\n- Overall Suitability:")
    if analysis_metrics['missing_percentage'] > 20 or analysis_metrics['imbalance_ratio'] < IMBALANCE_THRESHOLD or len(analysis_metrics.get('high_corr_features', [])) > 0:
        print("  - CAUTION: The dataset presents one or more challenges (high missingness, imbalance, or multicollinearity).")
    else:
        print("  - GOOD: The dataset appears relatively clean, balanced, and without strong multicollinearity. Good suitability for modeling.")
    
    print("\n  - The data has been preprocessed, scaled, and (if necessary) resampled. It is now ready for model training.")
    
    print("\n==============================================")
    print("= Automated Data Analysis Complete =")
    print("==============================================")


# --- Main execution ---
if __name__ == "__main__":
    
    # Suppress warnings for a cleaner report
    warnings.filterwarnings('ignore')
    
    # --- 4. Time Measurement (Start) ---
    start_time = time.time()
    
    # Run the entire analysis
    run_analysis(DATASET_PATH, TARGET_VARIABLE)
    
    # --- 4. Time Measurement (End) ---
    end_time = time.time()
    total_time = end_time - start_time
    
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n--- 4. Time Measurement ---")
    print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds")