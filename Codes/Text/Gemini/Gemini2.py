import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from scipy.stats import pearsonr, spearmanr, pointbiserialr, chi2_contingency
import warnings
from io import StringIO

# --- Configuration ---
FILE_PATH = 'diabetes_dataset_nan.csv'  # Path to the dataset
TARGET_VARIABLE = 'diagnosed_diabetes'      # The target variable for classification

# p-value threshold for significance in correlation analysis
P_VALUE_THRESHOLD = 0.05

# Suppress warnings for a cleaner automated report
warnings.filterwarnings('ignore')

# Set plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.autolayout'] = True

def generate_conclusion(metrics):
    """
    Generates an automated summary of data suitability based on EDA metrics.
    
    Args:
        metrics (dict): A dictionary containing 'missing_pct', 'imbalance_ratio',
                        and 'max_inter_corr'.
    """
    print("\n" + "="*80)
    print("SECTION 5: AUTOMATED CONCLUSION ON DATA SUITABILITY")
    print("="*80)
    
    suitability = "Good"
    comments = []
    
    # Assess Missingness
    if metrics['missing_pct'] > 20:
        suitability = "Poor"
        comments.append(f"- High Missingness: Over {metrics['missing_pct']:.1f}% of data is missing. Requires careful imputation or data collection.")
    elif metrics['missing_pct'] > 5:
        suitability = "Fair"
        comments.append(f"- Moderate Missingness: {metrics['missing_pct']:.1f}% missing. Imputation is necessary.")
    else:
        comments.append(f"- Low Missingness: ({metrics['missing_pct']:.1f}%). Good.")

    # Assess Imbalance
    if metrics['imbalance_ratio'] > 5:
        suitability = "Poor" if suitability != "Poor" else "Poor"
        comments.append(f"- Severe Class Imbalance: Ratio of 1:{metrics['imbalance_ratio']:.1f}. Requires advanced sampling (SMOTE, etc.).")
    elif metrics['imbalance_ratio'] > 2:
        suitability = "Fair" if suitability == "Good" else suitability
        comments.append(f"- Moderate Class Imbalance: Ratio of 1:{metrics['imbalance_ratio']:.1f}. Sampling techniques are recommended.")
    else:
        comments.append("- Good Class Balance: Ratio of 1:{metrics['imbalance_ratio']:.1f}.")

    # Assess Multicollinearity
    if metrics['max_inter_corr'] > 0.9:
        suitability = "Poor" if suitability != "Poor" else "Poor"
        comments.append(f"- Severe Multicollinearity Detected: Max correlation of {metrics['max_inter_corr']:.2f}. May require feature removal.")
    elif metrics['max_inter_corr'] > 0.7:
        suitability = "Fair" if suitability == "Good" else suitability
        comments.append(f"- High Multicollinearity Detected: Max correlation of {metrics['max_inter_corr']:.2f}. Review correlated features.")
    else:
        comments.append("- Low-to-Moderate Multicollinearity: Max correlation of {metrics['max_inter_corr']:.2f}. Acceptable.")

    # Final Summary
    print(f"\n--- Automated Suitability Assessment for Classification ---")
    print(f"\nOverall Suitability: {suitability}")
    print("\nKey Observations:")
    for comment in comments:
        print(comment)
        
    if suitability == "Good":
        print("\nConclusion: The dataset appears well-suited for classification modeling with standard preprocessing.")
    elif suitability == "Fair":
        print("\nConclusion: The dataset is conditionally suitable. Address moderate missingness, imbalance, or multicollinearity before modeling.")
    else:
        print("\nConclusion: The dataset has significant issues (high missingness, severe imbalance, or multicollinearity). Proceed with caution and advanced preprocessing.")
    print("\n" + "-"*80)


def analyze_data(file_path, target_variable):
    """
    Main function to run the complete automated analysis.
    """
    
    # 0. Start Timer
    start_time = time.time()
    
    # Metrics for final conclusion
    conclusion_metrics = {
        'missing_pct': 0,
        'imbalance_ratio': 1,
        'max_inter_corr': 0
    }

    print("="*80)
    print("STARTING AUTOMATED DATA ANALYSIS")
    print(f"Dataset: {file_path}")
    print(f"Target Variable: {target_variable}")
    print("="*80)

    # 1. Exploratory Data Analysis (EDA)
    print("\n" + "="*80)
    print("SECTION 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)

    try:
        print(f"\n[1.1] Loading dataset from '{file_path}'...")
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'. Please check the path and try again.")
        return
    except Exception as e:
        print(f"ERROR: An error occurred while loading the data: {e}")
        return

    # Check if target variable exists
    if target_variable not in df.columns:
        print(f"ERROR: Target variable '{target_variable}' not found in the dataset columns.")
        print(f"Available columns: {list(df.columns)}")
        return

    # [1.2] Basic Information
    print("\n[1.2] Basic Dataset Information")
    print(f"Shape (Rows, Columns): {df.shape}")
    
    print("\nColumn Data Types:")
    # Capture df.info() output
    buffer = StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())

    # [1.3] Missing Values
    print("\n[1.3] Missing Values Analysis")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({'Count': missing_values, 'Percent (%)': missing_percent})
    missing_df = missing_df[missing_df['Count'] > 0].sort_values(by='Percent (%)', ascending=False)
    
    total_missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    conclusion_metrics['missing_pct'] = total_missing_pct
    
    if missing_df.empty:
        print("No missing values found.")
    else:
        print(f"Total missing value percentage (all cells): {total_missing_pct:.2f}%")
        print("Columns with missing values:")
        print(missing_df)

    # [1.4] Duplicate Rows
    print("\n[1.4] Duplicate Row Analysis")
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        print("Recommendation: Consider dropping duplicates.")
        # df = df.drop_duplicates()
        # print("Duplicates dropped for further analysis.")

    # [1.5] Descriptive Statistics (Numerical)
    print("\n[1.5] Descriptive Statistics (Numerical Features)")
    print(df.describe().to_string())

    # [1.6] Descriptive Statistics (Categorical)
    print("\n[1.6] Descriptive Statistics (Categorical Features)")
    print(df.describe(include=['object', 'category']).to_string())
    
    # Identify feature types
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    # Remove target from numerical features if it's there
    if target_variable in numerical_features:
        numerical_features.remove(target_variable)
        
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Remove target from categorical features if it's there
    if target_variable in categorical_features:
        categorical_features.remove(target_variable)

    print(f"\nIdentified {len(numerical_features)} numerical features.")
    print(f"Identified {len(categorical_features)} categorical features.")

    # [1.7] Target Variable Distribution
    print(f"\n[1.7] Target Variable Distribution ('{target_variable}')")
    target_counts = df[target_variable].value_counts()
    target_percent = df[target_variable].value_counts(normalize=True) * 100
    print("Value Counts:")
    print(target_counts)
    print("\nPercentage:")
    print(target_percent)
    
    if len(target_counts) > 0:
        min_class_count = target_counts.min()
        max_class_count = target_counts.max()
        if min_class_count > 0:
            conclusion_metrics['imbalance_ratio'] = max_class_count / min_class_count
        else:
            conclusion_metrics['imbalance_ratio'] = np.inf
    
    try:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=target_variable, data=df, palette='pastel')
        plt.title(f'Distribution of Target Variable: {target_variable}')
        plt.xlabel(target_variable)
        plt.ylabel('Count')
        print(f"\nDisplaying plot: Target Variable Distribution...")
        plt.show()
    except Exception as e:
        print(f"Could not plot target variable distribution: {e}")

    # [1.8] Visualizations: Numerical Features
    print("\n[1.8] Generating plots for Numerical Features (Histograms and Boxplots)...")
    for col in numerical_features:
        try:
            # Histogram
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Histogram of {col}')
            
            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            
            print(f"Displaying plot: Distribution and Outliers for '{col}'...")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot '{col}': {e}")

    # [1.9] Visualizations: Categorical Features
    print("\n[1.9] Generating plots for Categorical Features (Countplots)...")
    for col in categorical_features:
        try:
            # Limit to top 20 most frequent categories for readability
            top_categories = df[col].value_counts().nlargest(20).index
            df_top = df[df[col].isin(top_categories)]
            
            plt.figure(figsize=(10, 5))
            sns.countplot(y=col, data=df_top, order=top_categories, palette='viridis')
            plt.title(f'Countplot of {col} (Top 20)')
            plt.xlabel('Count')
            plt.ylabel(col)
            print(f"Displaying plot: Distribution of '{col}'...")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot '{col}': {e}")

    # [1.10] Correlation Heatmap
    print("\n[1.10] Generating Correlation Heatmap (Numerical Features)...")
    try:
        corr_matrix = df[numerical_features].corr()
        
        # Store max inter-feature correlation (excluding 1.0 on diagonal)
        corr_matrix_no_diag = corr_matrix.mask(np.eye(len(corr_matrix), dtype=bool))
        conclusion_metrics['max_inter_corr'] = corr_matrix_no_diag.abs().max().max()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
        plt.title('Correlation Heatmap of Numerical Features')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        print("Displaying plot: Correlation Heatmap...")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not generate correlation heatmap: {e}")

    # 2. Data Preprocessing and Feature Engineering
    print("\n" + "="*80)
    print("SECTION 2: DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("="*80)
    
    # Create a copy for preprocessing
    df_prep = df.copy()

    # [2.1] Handle Missing Values
    print("\n[2.1] Handling Missing Values...")
    
    # Numerical Imputation (Median)
    if not df_prep[numerical_features].isnull().sum().sum() == 0:
        print("Imputing numerical features with 'median'...")
        num_imputer = SimpleImputer(strategy='median')
        df_prep[numerical_features] = num_imputer.fit_transform(df_prep[numerical_features])
    else:
        print("No missing values in numerical features to impute.")
        
    # Categorical Imputation (Mode)
    if not df_prep[categorical_features].isnull().sum().sum() == 0:
        print("Imputing categorical features with 'most_frequent' (mode)...")
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_prep[categorical_features] = cat_imputer.fit_transform(df_prep[categorical_features])
    else:
        print("No missing values in categorical features to impute.")

    print("Missing value imputation complete.")
    # Keep a copy of the imputed data for correlation analysis
    df_imputed = df_prep.copy()

    # [2.2] Encode Categorical Features
    print("\n[2.2] Encoding Categorical Features (One-Hot Encoding)...")
    # Using get_dummies for simplicity in an automated script
    original_cols = set(df_prep.columns)
    df_prep = pd.get_dummies(df_prep, columns=categorical_features, drop_first=True)
    new_cols = set(df_prep.columns)
    encoded_cols = list(new_cols - original_cols)
    print(f"Encoded {len(categorical_features)} features into {len(encoded_cols)} new columns.")
    print("Data head after encoding:")
    print(df_prep.head().to_string())

    # [2.3] Scale Numerical Features
    print("\n[2.3] Scaling Numerical Features (StandardScaler)...")
    scaler = StandardScaler()
    df_prep[numerical_features] = scaler.fit_transform(df_prep[numerical_features])
    print("Numerical features scaled.")
    print("Data head after scaling:")
    print(df_prep.head().to_string())

    # [2.4] Handle Class Imbalance (SMOTE)
    print("\n[2.4] Handling Class Imbalance...")
    
    # Use the imputed data (before scaling/OHE) for SMOTE
    # Ensure target is encoded for SMOTE
    y = df_imputed[target_variable]
    X = df_imputed.drop(columns=[target_variable])
    
    # SMOTE requires all features to be numerical.
    # For this demonstration, we'll apply it only to the *original* numerical features.
    # A full pipeline would require encoding categorical features first.
    print("Note: Applying SMOTE to numerical features only for demonstration.")
    X_smote = X[numerical_features]
    
    print("\nClass distribution before SMOTE:")
    print(y.value_counts())
    
    try:
        # Check if SMOTE is needed
        if conclusion_metrics['imbalance_ratio'] > 1.5:
            print("Applying SMOTE (Synthetic Minority Over-sampling Technique)...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_smote, y)
            
            print("\nClass distribution after SMOTE:")
            print(y_resampled.value_counts())
        else:
            print("Class imbalance is not significant. SMOTE not applied.")
            
    except Exception as e:
        print(f"Could not apply SMOTE. This may be due to incompatible feature types (e.g., categorical). Error: {e}")
        print("Skipping SMOTE.")


    # 3. Feature Relationship Analysis (with Target)
    print("\n" + "="*80)
    print("SECTION 3: FEATURE RELATIONSHIP ANALYSIS (WITH TARGET)")
    print("="*80)
    
    # We use the imputed dataframe (df_imputed) for this analysis
    # Ensure target variable is numerical (0/1) for correlation
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_imputed[target_variable])
    
    # [3.1] Numerical Features vs. Target
    print(f"\n[3.1] Numerical Features vs. Target ('{target_variable}')")
    print("Using Point-Biserial Correlation (for continuous-binary)...")
    
    num_corr_results = []
    for col in numerical_features:
        try:
            # Ensure no NaNs in the column (should be handled, but good to check)
            valid_mask = ~np.isnan(df_imputed[col]) & ~np.isnan(y_encoded)
            if np.sum(valid_mask) > 2: # Need at least 2 valid pairs
                corr, p_value = pointbiserialr(df_imputed[col][valid_mask], y_encoded[valid_mask])
                num_corr_results.append({
                    'Feature': col,
                    'Correlation': corr,
                    'P-Value': p_value,
                    'Significant': 'Yes' if p_value < P_VALUE_THRESHOLD else 'No'
                })
        except Exception as e:
            print(f"Could not calculate correlation for {col}: {e}")

    if num_corr_results:
        num_corr_df = pd.DataFrame(num_corr_results).sort_values(by='Correlation', ascending=False, key=abs)
        print(num_corr_df.to_string())
        
        # Plotting
        print("\nGenerating Boxplots for significant numerical features vs. target...")
        for res in num_corr_results:
            if res['Significant'] == 'Yes':
                try:
                    plt.figure(figsize=(8, 5))
                    sns.boxplot(x=target_variable, y=res['Feature'], data=df_imputed, palette='pastel')
                    plt.title(f"{res['Feature']} vs. {target_variable} (Corr: {res['Correlation']:.2f}, P: {res['P-Value']:.3g})")
                    print(f"Displaying plot: Boxplot for '{res['Feature']}'...")
                    plt.show()
                except Exception as e:
                    print(f"Could not plot boxplot for {res['Feature']}: {e}")

    # [3.2] Categorical Features vs. Target
    print(f"\n[3.2] Categorical Features vs. Target ('{target_variable}')")
    print("Using Chi-Square Test of Independence...")
    
    cat_corr_results = []
    for col in categorical_features:
        try:
            contingency_table = pd.crosstab(df_imputed[col], df_imputed[target_variable])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            cat_corr_results.append({
                'Feature': col,
                'Chi-Square Stat': chi2,
                'P-Value': p_value,
                'Significant': 'Yes' if p_value < P_VALUE_THRESHOLD else 'No'
            })
        except Exception as e:
            print(f"Could not calculate Chi-Square for {col}: {e}")
            
    if cat_corr_results:
        cat_corr_df = pd.DataFrame(cat_corr_results).sort_values(by='P-Value')
        print(cat_corr_df.to_string())

        # Plotting
        print("\nGenerating Stacked Bar Charts for significant categorical features vs. target...")
        for res in cat_corr_results:
            if res['Significant'] == 'Yes':
                try:
                    # Calculate proportions
                    prop_df = df_imputed.groupby(col)[target_variable].value_counts(normalize=True).unstack().fillna(0)
                    
                    prop_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
                    plt.title(f"Proportion of {target_variable} by {res['Feature']} (P: {res['P-Value']:.3g})")
                    plt.ylabel("Proportion")
                    plt.xlabel(res['Feature'])
                    plt.xticks(rotation=45, ha='right')
                    plt.legend(title=target_variable)
                    print(f"Displaying plot: Stacked Bar Chart for '{res['Feature']}'...")
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Could not plot stacked bar for {res['Feature']}: {e}")

    # 4. Time Measurement
    print("\n" + "="*80)
    print("SECTION 4: TIME MEASUREMENT")
    print("="*80)
    
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print(f"Full analysis completed.")
    print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds")

    # 5. Conclusion on Data Suitability
    # This section is printed by the helper function
    generate_conclusion(conclusion_metrics)


if __name__ == "__main__":
    try:
        analyze_data(FILE_PATH, TARGET_VARIABLE)
    except Exception as e:
        print(f"\nAn unexpected critical error occurred: {e}")
        print("Automated analysis aborted.")