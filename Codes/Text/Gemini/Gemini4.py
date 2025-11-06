import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample  # Import resample instead of SMOTE
import warnings
import os

# --- Script Configuration ---
FILE_PATH = "diabetes_dataset_nan.csv"
TARGET_COLUMN = 'diagnosed_diabetes'
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Helper Functions ---
def print_separator(title=""):
    """Prints a formatted section separator."""
    print("\n" + "="*80)
    if title:
        print(f"| {title.upper()} |".center(80))
        print("="*80 + "\n")
    else:
        print("\n")

def start_timer():
    """Starts the global timer."""
    return time.time()

def end_timer(start_time):
    """Ends the timer and prints the formatted output."""
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print_separator("TIME MEASUREMENT")
    print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds")
    return total_time

def plot_and_save(plot_func, *args, filename="plot.png", **kwargs):
    """Helper to create, save, and close plots."""
    try:
        plt.figure(figsize=(12, 7))
        plot_func(*args, **kwargs)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved to '{filename}'")
    except Exception as e:
        print(f"Warning: Could not save plot {filename}. Error: {e}")

# --- Main Analysis Script ---
def run_analysis(file_path, target_column):
    """
    Runs the full automated data analysis pipeline.
    """
    
    # 1. ====================
    #    EXPLORATORY DATA ANALYSIS (EDA)
    #    ====================
    print_separator("1. EXPLORATORY DATA ANALYSIS (EDA)")
    
    try:
        df = pd.read_csv(file_path)
        # Drop the first column if it's an unnamed index
        if df.columns[0].lower() in ['unnamed: 0', 'id']:
            df = df.drop(df.columns[0], axis=1)
        print(f"Successfully loaded data from '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Display basic info
    print("\n--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- Dataset Info (Column Types) ---")
    # Use a buffer to capture the .info() output
    from io import StringIO
    buffer = StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())

    print("\n--- Missing Values (Sum) ---")
    missing_values = df.isnull().sum()
    missing_values_report = missing_values[missing_values > 0]
    if missing_values_report.empty:
        print("No missing values found.")
    else:
        print(missing_values_report)
    total_missing = missing_values.sum()
    print(f"\nTotal missing values: {total_missing}")

    print("\n--- Duplicate Rows ---")
    duplicate_count = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicate_count}")

    print("\n--- Descriptive Statistics (Numerical) ---")
    print(df.describe())

    print("\n--- Target Variable Distribution ---")
    if target_column in df.columns:
        print(df[target_column].value_counts())
        print("\n--- Target Variable Distribution (Percentage) ---")
        print(df[target_column].value_counts(normalize=True) * 100)
    else:
        print(f"Error: Target column '{target_column}' not found.")
        return

    # Identify feature types
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include='object').columns.tolist()
    
    # Ensure target is not in the feature lists
    if target_column in numerical_features:
        numerical_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    print(f"\nIdentified Numerical Features: {numerical_features}")
    print(f"Identified Categorical Features: {categorical_features}")

    # --- EDA Plots ---
    print("\n--- Generating EDA Plots ---")
    
    # Histograms for numerical features
    if numerical_features:
        num_plots = len(numerical_features)
        cols = 3
        rows = int(np.ceil(num_plots / cols))
        
        try:
            plt.figure(figsize=(cols * 5, rows * 4))
            for i, col in enumerate(numerical_features):
                plt.subplot(rows, cols, i + 1)
                sns.histplot(df[col].dropna(), kde=True, bins=30)
                plt.title(f'Histogram of {col}')
            plt.tight_layout()
            plt.savefig("eda_histograms.png")
            plt.close()
            print("Plot saved to 'eda_histograms.png'")
        except Exception as e:
            print(f"Warning: Could not save histograms plot. Error: {e}")

    # Boxplots for numerical features
    if numerical_features:
        try:
            plt.figure(figsize=(15, max(8, len(numerical_features) * 0.5)))
            sns.boxplot(data=df[numerical_features], orient='h')
            plt.title('Boxplots for Numerical Features')
            plt.tight_layout()
            plt.savefig("eda_boxplots.png")
            plt.close()
            print("Plot saved to 'eda_boxplots.png'")
        except Exception as e:
            print(f"Warning: Could not save boxplots. Error: {e}")

    # Countplots for categorical variables
    if categorical_features:
        num_plots = len(categorical_features)
        cols = 2
        rows = int(np.ceil(num_plots / cols))
        
        try:
            plt.figure(figsize=(cols * 7, rows * 5))
            for i, col in enumerate(categorical_features):
                plt.subplot(rows, cols, i + 1)
                # Get top N categories to avoid clutter, e.g., top 15
                top_categories = df[col].value_counts().index[:15]
                sns.countplot(y=df[col], order=top_categories)
                plt.title(f'Countplot of {col} (Top 15)')
            plt.tight_layout()
            plt.savefig("eda_countplots.png")
            plt.close()
            print("Plot saved to 'eda_countplots.png'")
        except Exception as e:
            print(f"Warning: Could not save countplots. Error: {e}")


    # Correlation heatmap
    if numerical_features:
        try:
            plt.figure(figsize=(16, 10))
            corr_matrix = df[numerical_features].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
            plt.title('Correlation Heatmap of Numerical Features')
            plt.tight_layout()
            plt.savefig("eda_correlation_heatmap.png")
            plt.close()
            print("Plot saved to 'eda_correlation_heatmap.png'")
        except Exception as e:
            print(f"Warning: Could not save heatmap. Error: {e}")
            corr_matrix = None # Ensure corr_matrix exists for summary
    else:
        corr_matrix = None


    # 2. ====================
    #    DATA PREPROCESSING & FEATURE ENGINEERING
    #    ====================
    print_separator("2. DATA PREPROCESSING & FEATURE ENGINEERING")
    
    df_processed = df.copy()

    # --- Handle Missing Values ---
    print("\n--- Handling Missing Values ---")
    
    # Impute numerical features with median
    if numerical_features:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numerical_features] = num_imputer.fit_transform(df_processed[numerical_features])
        print("Numerical features imputed with median.")

    # Impute categorical features with mode
    if categorical_features:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_features] = cat_imputer.fit_transform(df_processed[categorical_features])
        print("Categorical features imputed with mode.")

    # --- Encode Categorical Features ---
    print("\n--- Encoding Categorical Features ---")
    le = LabelEncoder()
    df_for_corr = df_processed.copy() # Save a copy for correlation analysis before scaling
    
    for col in categorical_features:
        df_processed[col] = le.fit_transform(df_processed[col])
        df_for_corr[col] = le.fit_transform(df_for_corr[col]) # Also encode for corr
    print("Categorical features encoded using LabelEncoder.")

    # --- Scale Numerical Features ---
    print("\n--- Scaling Numerical Features ---")
    if numerical_features:
        scaler = StandardScaler()
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
        print("Numerical features scaled using StandardScaler.")

    # --- Handle Class Imbalance (Using Random Oversampling) ---
    print("\n--- Handling Class Imbalance ---")
    
    print("Class distribution before resampling:")
    print(df_processed[target_column].value_counts())

    val_counts = df_processed[target_column].value_counts()
    is_imbalanced = (val_counts.min() / val_counts.max()) < 0.4 

    if is_imbalanced and len(val_counts) == 2:
        print("\nClass imbalance detected. Applying Random Oversampling...")
        try:
            # Separate majority and minority classes
            majority_class_label = val_counts.idxmax()
            minority_class_label = val_counts.idxmin()
            
            df_majority = df_processed[df_processed[target_column] == majority_class_label]
            df_minority = df_processed[df_processed[target_column] == minority_class_label]
            
            # Oversample minority class
            df_minority_resampled = resample(df_minority, 
                                             replace=True,     # Sample with replacement
                                             n_samples=len(df_majority), # To match majority class
                                             random_state=42) 
            
            # Combine majority class with resampled minority class
            df_resampled = pd.concat([df_majority, df_minority_resampled])
            
            print("\nClass distribution after Random Oversampling:")
            print(df_resampled[target_column].value_counts())
            
            X_resampled = df_resampled.drop(target_column, axis=1)
            y_resampled = df_resampled[target_column]
        
        except Exception as e:
            print(f"Error during resampling: {e}. Proceeding without resampling.")
            X_resampled = df_processed.drop(target_column, axis=1)
            y_resampled = df_processed[target_column]
    else:
        if len(val_counts) != 2:
            print("\nResampling skipped: Target variable is not binary.")
        else:
            print("\nClass balance is acceptable. No resampling applied.")
        X_resampled = df_processed.drop(target_column, axis=1)
        y_resampled = df_processed[target_column]

        
    # 3. ====================
    #    FEATURE RELATIONSHIP ANALYSIS
    #    ====================
    print_separator("3. FEATURE RELATIONSHIP ANALYSIS")
    
    # Use the imputed & label-encoded (but not scaled) dataframe
    df_analysis = df_for_corr
    
    # --- Numerical vs. Target ---
    print("\n--- Correlation with Target (Numerical Features) ---")
    corr_results = []
    for col in numerical_features:
        # Ensure target is binary for pointbiserialr
        if df_analysis[target_column].nunique() == 2:
            try:
                # dropna() is important in case imputation failed or wasn't run
                col_data = df_analysis[col].dropna()
                target_data = df_analysis[target_column][col_data.index]
                corr, p_value = pointbiserialr(col_data, target_data)
                corr_results.append({'Feature': col, 'Correlation': corr, 'P-Value': p_value})
            except Exception as e:
                print(f"Could not calculate point-biserial correlation for {col}: {e}")
        else:
            print("Target is not binary. Skipping point-biserial correlation.")
            break
            
    if corr_results:
        corr_df = pd.DataFrame(corr_results).sort_values(by='Correlation', key=abs, ascending=False)
        print(corr_df.to_string())
        
        # Plotting top correlated features
        top_features = corr_df.head(min(5, len(corr_df)))['Feature'].tolist()
        if top_features:
            try:
                plt.figure(figsize=(15, 6))
                for i, col in enumerate(top_features):
                    plt.subplot(1, len(top_features), i + 1)
                    sns.scatterplot(x=df_analysis[col], y=df_analysis[target_column])
                    plt.title(f'{col} (Corr: {corr_df.iloc[i]["Correlation"]:.2f})')
                plt.tight_layout()
                plt.savefig("feature_relationship_scatter.png")
                plt.close()
                print("\nPlot saved to 'feature_relationship_scatter.png'")
            except Exception as e:
                print(f"Warning: Could not save scatter plots. Error: {e}")


    # --- Categorical vs. Target (Chi-Square) ---
    print("\n--- Correlation with Target (Categorical Features) ---")
    chi2_results = []
    for col in categorical_features:
        try:
            contingency_table = pd.crosstab(df_analysis[col], df_analysis[target_column])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            chi2_results.append({'Feature': col, 'Chi2 Statistic': chi2, 'P-Value': p_value})
        except Exception as e:
            print(f"Could not calculate Chi-square for {col}: {e}")

    if chi2_results:
        chi2_df = pd.DataFrame(chi2_results).sort_values(by='P-Value', ascending=True)
        print(chi2_df.to_string())


    # 4. ====================
    #    CONCLUSION ON DATA SUITABILITY
    #    ====================
    print_separator("4. CONCLUSION ON DATA SUITABILITY")
    
    # Gather metrics for summary
    target_dist = df[target_column].value_counts(normalize=True)
    balance_ratio = target_dist.min() / target_dist.max()
    
    multicollinearity = False
    if corr_matrix is not None:
        # Check for high multicollinearity (excluding self-correlation)
        corr_matrix_no_diag = corr_matrix.copy()
        np.fill_diagonal(corr_matrix_no_diag.values, 0)
        if (corr_matrix_no_diag.abs() > 0.8).any().any():
            multicollinearity = True

    # Generate summary
    print("--- Automated Data Suitability Assessment ---")
    
    print("\n1. Missingness:")
    if total_missing == 0:
        print("  - Excellent: No missing values found.")
    elif (total_missing / (df.shape[0] * df.shape[1])) < 0.05:
        print(f"  - Good: Low level of missingness ({total_missing} total). Imputation was successful.")
    else:
        print(f"  - Warning: Significant missingness ({total_missing} total). Imputation may affect model accuracy.")

    print("\n2. Class Balance (Target Variable):")
    if balance_ratio > 0.4:
        print(f"  - Good: The dataset is well-balanced (Ratio: {balance_ratio:.2f}).")
    elif balance_ratio > 0.2:
        print(f"  - Warning: Moderate imbalance detected (Ratio: {balance_ratio:.2f}). Resampling was applied.")
    else:
        print(f"  - Critical: Severe imbalance detected (Ratio: {balance_ratio:.2f}). Resampling was applied, but model may be biased.")

    print("\n3. Multicollinearity (Numerical Features):")
    if corr_matrix is None:
        print("  - Unknown: Could not generate correlation matrix.")
    elif multicollinearity:
        print("  - Warning: High multicollinearity (correlation > 0.8) detected between one or more feature pairs.")
        print("             Consider feature selection (e.g., VIF) before modeling.")
    else:
        print("  - Good: No strong multicollinearity detected between numerical features.")
    
    print("\n4. Noise/Duplicates:")
    if duplicate_count > 0:
        print(f"  - Warning: {duplicate_count} duplicate rows found. These should be investigated and likely removed.")
    else:
        print("  - Good: No duplicate rows found.")

    print("\n--- Overall Conclusion ---")
    if total_missing == 0 and balance_ratio > 0.4 and not multicollinearity and duplicate_count == 0:
        print("Overall Suitability: HIGH. The data is clean, balanced, and appears suitable for classification modeling.")
    else:
        print("Overall Suitability: MODERATE. The data is usable, but preprocessing steps (imputation, resampling,")
        print("feature selection) are critical. Pay close attention to the warnings above before modeling.")
    

# --- Main Execution ---
if __name__ == "__main__":
    start_time = start_timer()
    
    run_analysis(FILE_PATH, TARGET_COLUMN)
    
    # 5. ====================
    #    TIME MEASUREMENT
    #    ====================
    # Note: The timer end function is called here, 
    # but it also prints its own section header.
    end_timer(start_time)