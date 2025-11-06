import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, pointbiserialr

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter

# ==============================================================================
#                      --- MAIN ANALYSIS SCRIPT ---
# ==============================================================================

def run_automated_analysis(file_path, target_column):
    """
    Executes a full, automated data analysis (EDA, Preprocessing, Correlation)
    on a given dataset.
    """
    
    # --------------------------------------------------------------------------
    # 0. Start Time Measurement
    # --------------------------------------------------------------------------
    print("======================================================================")
    print("                    STARTING AUTOMATED DATA ANALYSIS                  ")
    print("======================================================================")
    start_time = time.time()

    
    # --------------------------------------------------------------------------
    # 1. Exploratory Data Analysis (EDA)
    # --------------------------------------------------------------------------
    print("\n\n--- SECTION 1: EXPLORATORY DATA ANALYSIS (EDA) ---")

    # Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- 1.1 Basic Information ---
    print("\n(1.1) Basic Dataset Information:")
    
    # Drop potential high-leakage or irrelevant columns
    # 'Unnamed: 0' is an index, 'diabetes_stage' is a label highly correlated with the binary target
    columns_to_drop = ['Unnamed: 0', 'diabetes_stage']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=col)
            print(f"Dropped irrelevant/leakage column: {col}")

    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        return

    print(f"\nDataset Shape: {df.shape}")
    print("\nColumn Data Types:")
    print(df.info())

    print("\nMissing Values (Top 10):")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0].sort_values(ascending=False).head(10))

    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    print("\nDescriptive Statistics (Numerical):")
    print(df.describe())

    print("\nDescriptive Statistics (Categorical):")
    print(df.describe(include=['object', 'category']))

    # --- 1.2 Target Variable Analysis ---
    print(f"\n(1.2) Target Variable Analysis ('{target_column}'):")
    print(df[target_column].value_counts(normalize=True))
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df)
    plt.title('Target Variable Distribution')
    plt.savefig('plot_1_target_distribution.png')
    plt.show()
    plt.close()

    # --- 1.3 Feature Distributions (Numerical) ---
    print("\n(1.3) Generating Numerical Feature Distributions...")
    numerical_features = df.select_dtypes(include=np.number).columns.drop(target_column)
    if len(numerical_features) > 0:
        num_plots = len(numerical_features)
        num_cols = 4
        num_rows = (num_plots + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3.5))
        axes = axes.flatten()

        for i, col in enumerate(numerical_features):
            sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'Distribution of {col}', fontsize=10)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.suptitle('Histograms of Numerical Features', y=1.02, fontsize=16)
        plt.savefig('plot_2_numerical_histograms.png')
        plt.show()
        plt.close()
    else:
        print("No numerical features found to plot.")

    # --- 1.4 Outlier Detection (Boxplots) ---
    print("\n(1.4) Generating Boxplots for Outlier Detection...")
    if len(numerical_features) > 0:
        plt.figure(figsize=(15, 8))
        # Log transform for better visualization if data is highly skewed (optional, but good for wide ranges)
        # We will plot 10 most variant features to avoid clutter
        top_variant_features = df[numerical_features].std().sort_values(ascending=False).index[:10]
        sns.boxplot(data=df[top_variant_features], orient='h')
        plt.title('Boxplots for Top 10 Variant Numerical Features')
        plt.xscale('symlog') # Use 'symlog' to handle wide ranges, including zeros
        plt.tight_layout()
        plt.savefig('plot_3_boxplots.png')
        plt.show()
        plt.close()
    else:
        print("No numerical features found for boxplots.")


    # --- 1.5 Correlation Heatmap ---
    print("\n(1.5) Generating Correlation Heatmap...")
    if len(numerical_features) > 0:
        corr_matrix = df[numerical_features].corr()
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
        plt.title('Correlation Heatmap of Numerical Features')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig('plot_4_correlation_heatmap.png')
        plt.show()
        plt.close()
    else:
        print("Not enough numerical features for a heatmap.")


    # --- 1.6 Categorical Variable Analysis ---
    print("\n(1.6) Generating Categorical Feature Counts...")
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_features) > 0:
        num_plots = len(categorical_features)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(categorical_features):
            # Show top 10 most frequent categories + 'Other'
            if df[col].nunique() > 10:
                top_10 = df[col].value_counts().index[:10]
                # Create a copy to avoid SettingWithCopyWarning
                plot_data = df[[col]].copy()
                plot_data[col] = plot_data[col].apply(lambda x: x if x in top_10 else 'Other')
                sns.countplot(y=col, data=plot_data, order=plot_data[col].value_counts().index, ax=axes[i])
                axes[i].set_title(f'Countplot of {col} (Top 10 + Other)', fontsize=10)
            else:
                sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=axes[i])
                axes[i].set_title(f'Countplot of {col}', fontsize=10)
            
            axes[i].set_xlabel('Count')
            axes[i].set_ylabel('')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.suptitle('Countplots for Categorical Features', y=1.02, fontsize=16)
        plt.savefig('plot_5_categorical_counts.png')
        plt.show()
        plt.close()
    else:
        print("No categorical features found to plot.")

    
    # --------------------------------------------------------------------------
    # 2. Data Preprocessing and Feature Engineering
    # --------------------------------------------------------------------------
    print("\n\n--- SECTION 2: DATA PREPROCESSING & FEATURE ENGINEERING ---")

    # Separate features and target
    X = df.drop(columns=target_column)
    y = df[target_column]

    # Re-identify feature types from X
    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    print(f"Identified {len(numerical_features)} numerical features.")
    print(f"Identified {len(categorical_features)} categorical features.")

    # --- 2.1 Define Preprocessing Pipelines ---
    print("\n(2.1) Defining preprocessing pipelines...")
    
    # Numerical pipeline: Impute with median (robust to outliers) + Scale
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute with mode + One-Hot Encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create the full preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 2.2 Apply Preprocessing ---
    print("\n(2.2) Applying preprocessing (fit_transform)...")
    try:
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after OHE
        cat_feature_names = preprocessor.named_transformers_['cat'] \
                                        .named_steps['onehot'] \
                                        .get_feature_names_out(categorical_features)
        
        all_feature_names = list(numerical_features) + list(cat_feature_names)
        
        # Convert back to DataFrame for easier analysis (if needed, though not strictly required)
        X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
        
        print("Preprocessing successful.")
        print(f"New processed shape: {X_processed_df.shape}")
        print("Processed data head (first 5 rows):")
        print(X_processed_df.head())

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        # Create a dummy X_processed_df for the script to continue
        X_processed_df = pd.DataFrame() # Empty
        y = pd.Series() # Empty
        
    if X_processed_df.empty:
        print("Skipping subsequent steps due to preprocessing failure.")
        
    else:
        # --- 2.3 Handle Class Imbalance ---
        print("\n(2.3) Handling Class Imbalance...")
        original_counts = Counter(y)
        print(f"Class distribution before SMOTE: {original_counts}")

        is_imbalanced = min(original_counts.values()) / max(original_counts.values()) < 0.4 # Threshold
        
        if is_imbalanced:
            print("Class imbalance detected. Applying SMOTE...")
            try:
                smote = SMOTE(random_state=42, k_neighbors=max(1, min(original_counts.values())-1)) # k_neighbors must be < smallest class
                X_resampled, y_resampled = smote.fit_resample(X_processed_df, y)
                resampled_counts = Counter(y_resampled)
                print(f"Class distribution after SMOTE: {resampled_counts}")
                
                # Plot comparison
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sns.countplot(x=y, ax=axes[0]).set_title('Before SMOTE')
                sns.countplot(x=y_resampled, ax=axes[1]).set_title('After SMOTE')
                plt.suptitle('Class Imbalance Handling')
                plt.savefig('plot_6_smote_comparison.png')
                plt.show()
                plt.close()
            except ValueError as e:
                print(f"Could not apply SMOTE (e.g., a class is too small): {e}")
                X_resampled, y_resampled = X_processed_df, y
        else:
            print("Classes are relatively balanced. No resampling applied.")
            X_resampled, y_resampled = X_processed_df, y


    # --------------------------------------------------------------------------
    # 3. Feature Relationship Analysis (with original, pre-scaled data)
    # --------------------------------------------------------------------------
    print("\n\n--- SECTION 3: FEATURE RELATIONSHIP ANALYSIS ---")
    
    # We use the original df for this, but with missing values imputed
    # for correlation calculations to work.
    
    # --- 3.1 Impute for Correlation ---
    # Create copies to avoid changing original df
    df_corr_analysis = df.copy()
    
    # Impute numerical with median
    for col in numerical_features:
        if df_corr_analysis[col].isnull().any():
            median_val = df_corr_analysis[col].median()
            df_corr_analysis[col] = df_corr_analysis[col].fillna(median_val)
            
    # Impute categorical with mode
    for col in categorical_features:
        if df_corr_analysis[col].isnull().any():
            mode_val = df_corr_analysis[col].mode()[0]
            df_corr_analysis[col] = df_corr_analysis[col].fillna(mode_val)
            
    # --- 3.2 Correlation with Target ---
    print(f"\n(3.1) Correlation with Target Variable ('{target_column}'):")
    correlation_results = []
    
    # Numerical features vs. Target (Point-Biserial)
    print("Calculating Point-Biserial Correlation (Numerical vs. Binary Target)...")
    for col in numerical_features:
        try:
            # Drop rows where target or feature is NaN (though we imputed features)
            clean_data = df_corr_analysis[[col, target_column]].dropna()
            if clean_data.shape[0] > 2 and clean_data[col].nunique() > 1:
                corr, p_value = pointbiserialr(clean_data[col], clean_data[target_column])
                correlation_results.append({
                    'Feature': col,
                    'Type': 'Numerical',
                    'Correlation_Method': 'Point-Biserial',
                    'Correlation_Value': corr,
                    'P_Value': p_value
                })
        except Exception as e:
            print(f"Could not calculate correlation for {col}: {e}")

    # Categorical features vs. Target (Cramér's V or similar)
    # For simplicity, we'll label-encode and use Spearman for a proxy
    print("Calculating Spearman Correlation (Encoded Categorical vs. Binary Target)...")
    for col in categorical_features:
        try:
            le = LabelEncoder()
            encoded_col = le.fit_transform(df_corr_analysis[col])
            
            clean_data = pd.DataFrame({
                'feature': encoded_col, 
                'target': df_corr_analysis[target_column]
            }).dropna()
            
            if clean_data.shape[0] > 2 and clean_data['feature'].nunique() > 1:
                corr, p_value = spearmanr(clean_data['feature'], clean_data['target'])
                correlation_results.append({
                    'Feature': col,
                    'Type': 'Categorical (Encoded)',
                    'Correlation_Method': 'Spearman',
                    'Correlation_Value': corr,
                    'P_Value': p_value
                })
        except Exception as e:
            print(f"Could not calculate correlation for {col}: {e}")

    if correlation_results:
        corr_df = pd.DataFrame(correlation_results)
        corr_df = corr_df.sort_values(by='Correlation_Value', key=abs, ascending=False)
        print("\nCorrelation Table (Top 15 features):")
        print(corr_df.to_string(index=False, float_format="%.4f"))

        # --- 3.3 Visual Correlation Plots ---
        print("\n(3.2) Generating Visual Correlation Plots...")
        
        # Plot top 5 numerical features against target
        top_5_num_features = corr_df[corr_df['Type'] == 'Numerical']['Feature'].head(5).tolist()
        
        if top_5_num_features:
            fig, axes = plt.subplots(1, len(top_5_num_features), figsize=(len(top_5_num_features) * 4, 5))
            if len(top_5_num_features) == 1: axes = [axes] # Make it iterable
                
            for i, col in enumerate(top_5_num_features):
                sns.boxplot(x=target_column, y=col, data=df_corr_analysis, ax=axes[i])
                axes[i].set_title(f'{col} vs. {target_column}')
            
            plt.tight_layout()
            plt.suptitle('Top 5 Numerical Features vs. Target', y=1.03, fontsize=16)
            plt.savefig('plot_7_top_numerical_vs_target.png')
            plt.show()
            plt.close()
        
        # Plot top 5 categorical features against target
        top_5_cat_features = corr_df[corr_df['Type'] == 'Categorical (Encoded)']['Feature'].head(5).tolist()

        if top_5_cat_features:
            fig, axes = plt.subplots(len(top_5_cat_features), 1, figsize=(8, len(top_5_cat_features) * 4))
            if len(top_5_cat_features) == 1: axes = [axes] # Make it iterable

            for i, col in enumerate(top_5_cat_features):
                # Create a normalized crosstab for plotting
                ctab = pd.crosstab(df_corr_analysis[col], df_corr_analysis[target_column], normalize='index')
                ctab.plot(kind='barh', stacked=True, ax=axes[i], legend=True)
                axes[i].set_title(f'Proportional Distribution of {col} vs. {target_column}')
                axes[i].set_xlabel('Proportion')
            
            plt.tight_layout()
            plt.suptitle('Top 5 Categorical Features vs. Target', y=1.02, fontsize=16)
            plt.savefig('plot_8_top_categorical_vs_target.png')
            plt.show()
            plt.close()

    else:
        print("No correlation results to display.")


    # --------------------------------------------------------------------------
    # 4. Time Measurement
    # --------------------------------------------------------------------------
    print("\n\n--- SECTION 4: RUNTIME MEASUREMENT ---")
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print(f"\nTotal Simulated Runtime: {minutes} minutes and {seconds} seconds")


    # --------------------------------------------------------------------------
    # 5. Conclusion on Data Suitability
    # --------------------------------------------------------------------------
    print("\n\n--- SECTION 5: AUTOMATED SUITABILITY ASSESSMENT ---")
    
    # Gather metrics for assessment
    try:
        total_rows = df.shape[0]
        missing_pct = df.isnull().sum().sum() / (total_rows * df.shape[1])
        is_imbalanced = min(original_counts.values()) / max(original_counts.values()) < 0.4
        
        # Check for high multicollinearity (absolute correlation > 0.8)
        if 'corr_matrix' in locals():
            corr_matrix_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [col for col in corr_matrix_upper.columns if any(corr_matrix_upper[col].abs() > 0.8)]
            multicollinearity_issue = len(high_corr_features) > 0
        else:
            multicollinearity_issue = False
            
        # Check for significant correlation with target
        if 'corr_df' in locals():
            strong_target_corr = any(corr_df['Correlation_Value'].abs() > 0.3)
        else:
            strong_target_corr = False

        # Generate summary
        print("Summary of Data Suitability for Classification:")
        print("------------------------------------------------")
        
        if total_rows < 1000:
            print(f"- WARNING: Low sample size ({total_rows} rows). Model may not generalize well.")
        else:
            print(f"- INFO: Sufficient sample size ({total_rows} rows).")

        if missing_pct > 0.2:
            print(f"- WARNING: High proportion of missing data ({missing_pct:.1%}). May require advanced imputation or feature removal.")
        elif missing_pct > 0.01:
            print(f"- INFO: Moderate missing data ({missing_pct:.1%}) detected and imputed.")
        else:
            print(f"- INFO: Low missing data ({missing_pct:.1%}).")

        if is_imbalanced:
            print(f"- WARNING: Significant class imbalance detected (Ratio: {min(original_counts.values()) / max(original_counts.values()):.2f}). SMOTE was applied.")
        else:
            print("- INFO: Target variable is reasonably balanced.")

        if multicollinearity_issue:
            print(f"- WARNING: High multicollinearity detected in features like {high_corr_features}. Consider using regularization (L1) or PCA.")
        else:
            print("- INFO: No significant multicollinearity (r > 0.8) detected between numerical features.")
            
        if not strong_target_corr:
            print("- WARNING: No single feature shows strong (r > 0.3) linear/rank correlation with the target. Model may need to rely on feature interactions.")
        else:
            print("- INFO: At least one feature shows a moderate-to-strong correlation with the target.")

        print("\n**Overall Assessment:** The dataset appears [SUITABLE / MARGINALLY SUITABLE] for classification modeling,")
        print("but attention should be paid to the warnings listed above (e.g., imbalance, multicollinearity, feature relevance) during the modeling phase.")
        print("Preprocessing steps (scaling, encoding, imputation) have been successfully defined.")

    except Exception as e:
        print(f"Could not generate automated assessment due to an error: {e}")

    print("\n======================================================================")
    print("                    AUTOMATED ANALYSIS COMPLETE                     ")
    print("======================================================================")


# ==============================================================================
#                      --- SCRIPT EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    # Define the path to the dataset
    DATASET_FILE_PATH = 'diabetes_dataset_nan.csv'
    
    # Define the name of the target variable (column) for classification
    TARGET_COLUMN_NAME = 'diagnosed_diabetes'
    # ---------------------

    # Run the analysis
    run_automated_analysis(DATASET_FILE_PATH, TARGET_COLUMN_NAME)