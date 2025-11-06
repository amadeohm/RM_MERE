import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.stats import pointbiserialr, spearmanr

# --- Configuration ---
DATASET_FILE = "/Users/amadeo/Documents/Code/diabetes_dataset_nan.csv"
TARGET_VARIABLE = "diagnosed_diabetes"
# --- End Configuration ---

def main_analysis(dataset_path, target_column):
    """
    Main function to run the complete automated data analysis.
    """
    warnings.filterwarnings('ignore')
    sns.set_style('darkgrid')
    
    # Storage for summary
    summary_metrics = {}
    
    print(f"Starting analysis for '{dataset_path}' with target '{target_column}'")
    start_time = time.time()

    try:
        # Load data
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_path}'")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if target_column not in df.columns:
        print(f"Error: Target variable '{target_column}' not found in dataset columns.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    ############################################################################
    # 1. Exploratory Data Analysis (EDA)
    ############################################################################
    print("\n" + "="*80)
    print("--- 1. Exploratory Data Analysis (EDA) ---")
    print("="*80)

    print(f"\nDataset Shape: {df.shape}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    print("\nDataset Info (Column Types & Non-Null Counts):")
    df.info()

    print("\nMissing Values (Sum):")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Store missingness for summary
    total_cells = np.product(df.shape)
    total_missing = missing_values.sum()
    summary_metrics['missing_percent'] = total_missing / total_cells

    print("\nDescriptive Statistics (Numerical):")
    print(df.describe())

    print("\nDescriptive Statistics (Categorical):")
    print(df.describe(include=['object', 'category']))

    # Separate feature types
    numerical_features = df.select_dtypes(include=np.number).columns.drop(target_column, errors='ignore').tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\nIdentified {len(numerical_features)} numerical features.")
    print(f"Identified {len(categorical_features)} categorical features.")

    # --- Target Variable Analysis ---
    print(f"\nTarget Variable ({target_column}) Distribution:")
    print(df[target_column].value_counts())
    print("\nTarget Variable Proportions:")
    print(df[target_column].value_counts(normalize=True))
    
    # Store imbalance for summary
    summary_metrics['balance_ratio'] = df[target_column].value_counts(normalize=True).min()

    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_column, data=df)
    plt.title(f'Distribution of Target Variable ({target_column})')
    plt.tight_layout()
    plt.show()

    # --- Histograms for Numerical Features ---
    print("\nGenerating Histograms for Numerical Features...")
    for col in numerical_features:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Histogram of {col}')
        plt.tight_layout()
        plt.show()

    # --- Boxplots for Numerical Features (Outlier Detection) ---
    print("\nGenerating Boxplots for Numerical Features...")
    for col in numerical_features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

    # --- Countplots for Categorical Variables ---
    print("\nGenerating Countplots for Categorical Features...")
    for col in categorical_features:
        plt.figure(figsize=(10, 6))
        # Use y-axis for better readability if many categories
        if df[col].nunique() > 10:
             sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        else:
             sns.countplot(x=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Countplot of {col}')
        plt.tight_layout()
        plt.show()

    # --- Correlation Heatmap ---
    print("\nGenerating Correlation Heatmap (Numerical Features)...")
    if numerical_features:
        corr_matrix = df[numerical_features].corr()
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.show()
        
        # Store high correlation for summary
        corr_matrix_abs = corr_matrix.abs()
        upper_tri = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))
        summary_metrics['high_corr_features'] = [col for col in upper_tri.columns if any(upper_tri[col] > 0.85)]
    else:
        print("Skipping numerical correlation heatmap (no numerical features).")
        summary_metrics['high_corr_features'] = []


    ############################################################################
    # 2. Data Preprocessing and Feature Engineering
    ############################################################################
    print("\n" + "="*80)
    print("--- 2. Data Preprocessing and Feature Engineering ---")
    print("="*80)
    
    # Create a copy for preprocessing
    df_processed = df.copy()

    # --- Handle Missing Values ---
    print("\nHandling Missing Values...")
    # Numerical Imputation (Median)
    if numerical_features:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numerical_features] = num_imputer.fit_transform(df_processed[numerical_features])
        print("Numerical features imputed with median.")
    
    # Categorical Imputation (Mode)
    if categorical_features:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_features] = cat_imputer.fit_transform(df_processed[categorical_features])
        print("Categorical features imputed with mode.")

    # --- Encode Categorical Features ---
    print("\nEncoding Categorical Features...")
    
    # Separate target variable before encoding
    if target_column in df_processed.columns:
        y = df_processed[target_column]
        X = df_processed.drop(target_column, axis=1)
        
        # Re-identify features in X
        X_numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        X_categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    else:
        print("Target variable not in processed dataframe, something is wrong.")
        X = df_processed
        y = None
        X_numerical_features = numerical_features
        X_categorical_features = categorical_features

    # Encode categorical features in X using One-Hot Encoding
    if X_categorical_features:
        X = pd.get_dummies(X, columns=X_categorical_features, drop_first=True)
        print(f"Categorical features one-hot encoded. New shape of X: {X.shape}")
    else:
        print("No categorical features to encode.")

    # --- Scale Numerical Features ---
    print("\nScaling Numerical Features...")
    
    # Find numerical features in the new X (original + dummies)
    final_numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    
    if final_numerical_features:
        scaler = StandardScaler()
        X[final_numerical_features] = scaler.fit_transform(X[final_numerical_features])
        print("All numerical features scaled using StandardScaler.")
    else:
        print("No numerical features to scale.")

    print("\nFinal processed features (X) head:")
    print(X.head())

    # --- Handle Class Imbalance ---
    print("\nChecking for Class Imbalance...")
    if y is not None:
        print(f"Original class distribution: {Counter(y)}")
        
        # Use balance_ratio from EDA
        if summary_metrics['balance_ratio'] < 0.3: # Threshold for imbalance
            print(f"Class imbalance detected (min class: {summary_metrics['balance_ratio']:.2%}). Applying SMOTE...")
            try:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                print(f"Resampled class distribution: {Counter(y_resampled)}")
            except Exception as e:
                print(f"Error during SMOTE: {e}. Proceeding with imbalanced data.")
                X_resampled, y_resampled = X, y
        else:
            print("Class distribution is relatively balanced. No resampling applied.")
            X_resampled, y_resampled = X, y
    else:
        print("Cannot check balance, target variable (y) is missing.")


    ############################################################################
    # 3. Feature Relationship Analysis (with Target)
    ############################################################################
    print("\n" + "="*80)
    print("--- 3. Feature Relationship Analysis (with Target) ---")
    print("="*80)
    
    # Use the *original* dataframe (df) for meaningful correlation
    
    correlation_results = []
    
    # Ensure target is numeric for correlation (if not already)
    # This is crucial for point-biserial
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        target_numeric = le.fit_transform(df[target_column])
        print(f"Target '{target_column}' was encoded for correlation analysis.")
    else:
        target_numeric = df[target_column]

    is_binary_target = len(np.unique(target_numeric[~np.isnan(target_numeric)])) == 2

    print("\nCalculating Correlation between Numerical Features and Target...")
    for col in numerical_features:
        try:
            # Drop NaNs for this specific correlation pair
            valid_data = df[[col, target_column]].dropna()
            feature_col = valid_data[col]
            target_col = target_numeric.loc[valid_data.index] # Align target

            if feature_col.empty or target_col.empty or len(feature_col) < 2:
                continue

            if is_binary_target:
                # Point-Biserial for numerical <-> binary
                corr, pval = pointbiserialr(feature_col, target_col)
                method = "Point-Biserial"
            else:
                # Spearman for numerical <-> ordinal/continuous (more robust)
                corr, pval = spearmanr(feature_col, target_col)
                method = "Spearman"
                
            correlation_results.append({
                'Feature': col, 
                'Correlation': corr, 
                'P-Value': pval, 
                'Method': method
            })
        except Exception as e:
            print(f"Could not calculate correlation for '{col}': {e}")

    if correlation_results:
        corr_df = pd.DataFrame(correlation_results)
        corr_df = corr_df.sort_values(by='Correlation', key=abs, ascending=False)
        print("\nCorrelation with Target Variable (Top 10):")
        print(corr_df.head(10).to_string())
        
        # --- Visual Correlation Plots ---
        print("\nGenerating relationship plots for top 5 correlated features...")
        top_5_features = corr_df['Feature'].head(5).tolist()
        
        for col in top_5_features:
            plt.figure(figsize=(8, 5))
            if is_binary_target:
                # Boxplot is great for binary target
                sns.boxplot(data=df, x=target_column, y=col)
                plt.title(f'Feature: {col} by {target_column}')
            else:
                # Regplot for continuous target
                sns.regplot(data=df, x=col, y=target_column, lowess=True, line_kws={'color': 'red'})
                plt.title(f'Scatterplot: {col} vs {target_column}')
            plt.tight_layout()
            plt.show()
    else:
        print("No numerical feature correlations to display.")

    ############################################################################
    # 4. Time Measurement
    ############################################################################
    # This section is just a placeholder; the final time is calculated at the end.
    print("\n" + "="*80)
    print("--- 4. Time Measurement ---")
    print("="*80)
    print("\nTotal runtime will be calculated at the end.")


    ############################################################################
    # 5. Conclusion on Data Suitability
    ############################################################################
    print("\n" + "="*80)
    print("--- 5. Conclusion on Data Suitability ---")
    print("="*80)
    
    summary = ["\nAuto-Generated Data Suitability Assessment:"]
    
    # 1. Missingness
    miss_pct = summary_metrics['missing_percent']
    if miss_pct > 0.2:
        summary.append(f"- WARNING: High missingness ({miss_pct:.1%}). Imputation may introduce significant bias.")
    elif miss_pct > 0.01:
        summary.append(f"- INFO: Moderate missingness ({miss_pct:.1%}) detected and imputed.")
    else:
        summary.append("- SUCCESS: Low missingness. Data is relatively clean.")

    # 2. Class Balance
    balance = summary_metrics['balance_ratio']
    if balance < 0.1:
        summary.append(f"- CRITICAL: Severe class imbalance ({balance:.1%}). Model will be heavily biased without resampling (like SMOTE).")
    elif balance < 0.3:
        summary.append(f"- WARNING: Significant class imbalance ({balance:.1%}). Resampling (e.g., SMOTE) is recommended.")
    else:
        summary.append("- SUCCESS: Target variable is reasonably balanced.")

    # 3. Multicollinearity
    high_corr = summary_metrics['high_corr_features']
    if high_corr:
        summary.append(f"- WARNING: High multicollinearity detected (r > 0.85) in features: {high_corr}. Consider VIF or feature removal.")
    else:
        summary.append("- INFO: No severe multicollinearity (r > 0.85) detected among numerical features.")
    
    # 4. Final Verdict
    if (balance < 0.1) or (miss_pct > 0.2) or high_corr:
        summary.append("- OVERALL: The dataset appears SUITABLE for classification, but requires attention to the CRITICAL/WARNING issues noted above (especially imbalance, missingness, or high collinearity) before modeling.")
    else:
        summary.append("- OVERALL: The dataset appears highly suitable for classification modeling. Preprocessing was successful and no major issues were detected.")

    for line in summary:
        print(line)
        
    ############################################################################
    # End of Analysis
    ############################################################################
    
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*80)
    print("--- Analysis Complete ---")
    print(f"\nTotal Simulated Runtime: {minutes} minutes and {seconds} seconds")
    print("="*80)


if __name__ == "__main__":
    main_analysis(DATASET_FILE, TARGET_VARIABLE)