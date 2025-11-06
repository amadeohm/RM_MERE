import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pointbiserialr
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Start timer
start_time = time.time()

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 80)
print("COMPREHENSIVE DIABETES DATASET ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "="*50)
print("1. EXPLORATORY DATA ANALYSIS")
print("="*50)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('diabetes_dataset_nan.csv', index_col=0)
print(f"Dataset loaded successfully!")

# Basic dataset info
print(f"\nDataset Shape: {df.shape}")
print(f"Number of features: {df.shape[1]}")
print(f"Number of samples: {df.shape[0]}")

# Column types and missing values
print("\n" + "-"*30)
print("DATASET INFORMATION")
print("-"*30)
print(df.info())

print("\n" + "-"*30)
print("MISSING VALUES ANALYSIS")
print("-"*30)
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0])

# Duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Descriptive statistics
print("\n" + "-"*30)
print("DESCRIPTIVE STATISTICS")
print("-"*30)
print(df.describe())

# Target variable analysis
print("\n" + "-"*30)
print("TARGET VARIABLE DISTRIBUTION")
print("-"*30)
target_col = 'diagnosed_diabetes'
if target_col in df.columns:
    target_dist = df[target_col].value_counts()
    target_percent = df[target_col].value_counts(normalize=True) * 100
    target_summary = pd.DataFrame({
        'Count': target_dist,
        'Percentage': target_percent
    })
    print(target_summary)
    
    # Plot target distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df[target_col].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title('Target Variable Distribution')
    plt.xlabel('Diagnosed Diabetes')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    plt.pie(target_dist.values, labels=target_dist.index, autopct='%1.1f%%', 
            colors=['lightblue', 'lightcoral'])
    plt.title('Target Variable Proportion')
    plt.tight_layout()
    plt.show()

# Separate numerical and categorical features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove target from numerical cols if it's there
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

print(f"\nNumerical features ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# Histograms for numerical features
print("\nGenerating numerical features distribution plots...")
if numerical_cols:
    # Select only a subset for visualization to avoid too many plots
    plot_numerical = numerical_cols[:12]  # First 12 numerical features
    
    n_rows = (len(plot_numerical) + 3) // 4
    plt.figure(figsize=(20, 5*n_rows))
    for i, col in enumerate(plot_numerical, 1):
        plt.subplot(n_rows, 4, i)
        df[col].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Boxplots for outlier detection
print("\nGenerating boxplots for outlier detection...")
if numerical_cols:
    plot_numerical_box = numerical_cols[:12]
    
    n_rows = (len(plot_numerical_box) + 3) // 4
    plt.figure(figsize=(20, 5*n_rows))
    for i, col in enumerate(plot_numerical_box, 1):
        plt.subplot(n_rows, 4, i)
        df.boxplot(column=col)
        plt.title(f'Boxplot of {col}')
        plt.ylabel('Values')
    plt.tight_layout()
    plt.show()

# Correlation heatmap
print("\nGenerating correlation heatmap...")
if len(numerical_cols) > 1:
    plt.figure(figsize=(16, 14))
    correlation_matrix = df[numerical_cols].corr()
    
    # Plot only lower triangle for better readability
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap (Numerical Features)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    # Find high correlations
    high_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                high_corr.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
    
    if high_corr:
        print("\nHighly correlated feature pairs (|r| > 0.8):")
        for feat1, feat2, corr in high_corr:
            print(f"  {feat1} - {feat2}: {corr:.3f}")

# Countplots for categorical variables
print("\nGenerating categorical features distribution plots...")
if categorical_cols:
    n_rows = (len(categorical_cols) + 1) // 2
    plt.figure(figsize=(20, 5*n_rows))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(n_rows, 2, i)
        df[col].value_counts().head(10).plot(kind='bar', color='lightgreen')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. DATA PREPROCESSING AND FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*50)
print("2. DATA PREPROCESSING AND FEATURE ENGINEERING")
print("="*50)

# Create a copy for preprocessing
df_processed = df.copy()

# Handle missing values
print("\n" + "-"*30)
print("HANDLING MISSING VALUES")
print("-"*30)

# Numerical imputation (exclude target)
numerical_imputer = SimpleImputer(strategy='median')
df_processed[numerical_cols] = numerical_imputer.fit_transform(df_processed[numerical_cols])

# Categorical imputation  
if categorical_cols:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df_processed[categorical_cols] = categorical_imputer.fit_transform(df_processed[categorical_cols])

print("Missing values after imputation:")
print(f"Total missing values: {df_processed.isnull().sum().sum()}")

# Encode categorical features
print("\n" + "-"*30)
print("ENCODING CATEGORICAL FEATURES")
print("-"*30)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique categories")

# Prepare features and target
if target_col in df_processed.columns:
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
else:
    X = df_processed
    y = None

print(f"\nFeatures shape: {X.shape}")
if y is not None:
    print(f"Target shape: {y.shape}")

# Scale numerical features
print("\n" + "-"*30)
print("SCALING NUMERICAL FEATURES")
print("-"*30)

scaler = StandardScaler()
X_scaled = X.copy()

# Only scale numerical columns that exist in X_scaled
numerical_to_scale = [col for col in numerical_cols if col in X_scaled.columns]
if numerical_to_scale:
    X_scaled[numerical_to_scale] = scaler.fit_transform(X_scaled[numerical_to_scale])
    print(f"Numerical features scaled using StandardScaler: {len(numerical_to_scale)} features")
else:
    print("No numerical features to scale")

# Handle class imbalance
if y is not None:
    print("\n" + "-"*30)
    print("CLASS IMBALANCE ANALYSIS")
    print("-"*30)
    
    original_counts = y.value_counts()
    print("Original class distribution:")
    for class_val, count in original_counts.items():
        print(f"  Class {class_val}: {count} samples ({count/len(y)*100:.2f}%)")
    
    imbalance_ratio = original_counts.min() / original_counts.max()
    print(f"Imbalance ratio: {imbalance_ratio:.3f}")
    
    if imbalance_ratio < 0.5:  # Apply SMOTE if significant imbalance
        print("\nSignificant class imbalance detected. Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
        
        balanced_counts = y_balanced.value_counts()
        print("After SMOTE balancing:")
        for class_val, count in balanced_counts.items():
            print(f"  Class {class_val}: {count} samples ({count/len(y_balanced)*100:.2f}%)")
    else:
        print("\nNo significant class imbalance detected.")
        X_balanced, y_balanced = X_scaled, y

# =============================================================================
# 3. FEATURE RELATIONSHIP ANALYSIS
# =============================================================================
print("\n" + "="*50)
print("3. FEATURE RELATIONSHIP ANALYSIS")
print("="*50)

if y is not None:
    # Correlation with target
    print("\n" + "-"*30)
    print("FEATURE-TARGET CORRELATIONS")
    print("-"*30)
    
    correlation_results = []
    
    for col in X_scaled.columns:
        try:
            if col in numerical_to_scale:
                # Pearson correlation for numerical-numerical
                corr, p_value = stats.pearsonr(X_scaled[col], y)
                corr_type = 'Pearson'
            else:
                # Point-biserial correlation for categorical-numerical
                corr, p_value = pointbiserialr(X_scaled[col], y)
                corr_type = 'Point-Biserial'
            
            correlation_results.append({
                'Feature': col,
                'Correlation': corr,
                'P-value': p_value,
                'Correlation_Type': corr_type,
                'Abs_Correlation': abs(corr)
            })
        except Exception as e:
            print(f"Could not compute correlation for {col}: {e}")
    
    if correlation_results:
        correlation_df = pd.DataFrame(correlation_results)
        correlation_df = correlation_df.sort_values('Abs_Correlation', ascending=False)
        
        print("\nTop 15 features most correlated with target:")
        print(correlation_df.head(15).round(4))
        
        # Visualize top correlations
        top_features = correlation_df.head(10)['Feature'].tolist()
        
        if top_features:
            plt.figure(figsize=(15, 8))
            
            # Correlation plot
            plt.subplot(1, 2, 1)
            top_corr_data = correlation_df.head(10)[['Feature', 'Correlation']]
            colors = ['red' if x < 0 else 'blue' for x in top_corr_data['Correlation']]
            plt.barh(top_corr_data['Feature'], top_corr_data['Correlation'], color=colors)
            plt.xlabel('Correlation with Target')
            plt.title('Top 10 Features Correlated with Target')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # P-values plot
            plt.subplot(1, 2, 2)
            top_p_data = correlation_df.head(10)[['Feature', 'P-value']]
            plt.barh(top_p_data['Feature'], -np.log10(top_p_data['P-value']))
            plt.xlabel('-log10(P-value)')
            plt.title('Statistical Significance (-log10 P-value)')
            plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        # Feature importance using ANOVA F-value
        print("\n" + "-"*30)
        print("FEATURE IMPORTANCE (ANOVA F-VALUE)")
        print("-"*30)
        
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X_scaled, y)
        
        feature_scores = pd.DataFrame({
            'Feature': X_scaled.columns,
            'F_Score': selector.scores_,
            'P_Value': selector.pvalues_
        })
        feature_scores = feature_scores.sort_values('F_Score', ascending=False)
        
        print("Top 15 most important features (ANOVA F-test):")
        print(feature_scores.head(15).round(4))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_importance = feature_scores.head(15)
        plt.barh(top_importance['Feature'], top_importance['F_Score'])
        plt.xlabel('ANOVA F-value')
        plt.title('Top 15 Feature Importance Scores')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

# =============================================================================
# 4. TIME MEASUREMENT
# =============================================================================
end_time = time.time()
total_seconds = end_time - start_time
minutes = int(total_seconds // 60)
seconds = total_seconds % 60

print("\n" + "="*50)
print("4. TIME MEASUREMENT")
print("="*50)
print(f"Total Simulated Runtime: {minutes} minutes and {seconds:.2f} seconds")

# =============================================================================
# 5. CONCLUSION ON DATA SUITABILITY
# =============================================================================
print("\n" + "="*50)
print("5. DATA SUITABILITY ASSESSMENT")
print("="*50)

# Automated assessment
assessments = []

# Dataset size assessment
if df.shape[0] < 1000:
    assessments.append("⚠️  Dataset size is relatively small for robust modeling")
else:
    assessments.append("✅  Dataset has sufficient samples for modeling")

# Missing data assessment
missing_percent_total = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
if missing_percent_total > 10:
    assessments.append(f"⚠️  Significant missing data ({missing_percent_total:.1f}%)")
else:
    assessments.append(f"✅  Minimal missing data ({missing_percent_total:.1f}%)")

# Class balance assessment
if y is not None:
    balance_ratio = y.value_counts().min() / y.value_counts().max()
    if balance_ratio < 0.3:
        assessments.append("⚠️  Significant class imbalance detected")
    elif balance_ratio < 0.5:
        assessments.append("⚡  Moderate class imbalance")
    else:
        assessments.append("✅  Well-balanced classes")

# Feature quality assessment
if 'correlation_df' in locals() and len(correlation_df) > 0:
    high_corr_count = len([corr for corr in correlation_df['Abs_Correlation'] if corr > 0.3])
    if high_corr_count >= 5:
        assessments.append(f"✅  Good feature-target relationships ({high_corr_count} features with |r|>0.3)")
    else:
        assessments.append(f"⚠️  Limited strong feature-target relationships")
else:
    assessments.append("⚡  Feature correlation analysis incomplete")

# Multicollinearity assessment
if len(numerical_cols) > 1:
    corr_matrix = df[numerical_cols].corr().abs()
    high_multicoll = ((corr_matrix > 0.8) & (corr_matrix < 1.0)).sum().sum() / 2
    if high_multicoll > 10:
        assessments.append(f"⚠️  Potential multicollinearity ({high_multicoll} high correlations)")
    else:
        assessments.append("✅  Acceptable multicollinearity levels")

# Print assessment
print("\nAUTOMATED ASSESSMENT:")
for assessment in assessments:
    print(f"  {assessment}")

# Overall suitability
warning_count = sum(1 for a in assessments if '⚠️' in a)
moderate_count = sum(1 for a in assessments if '⚡' in a)

if warning_count >= 3:
    overall = "REQUIRES SIGNIFICANT ATTENTION"
elif warning_count >= 1 or moderate_count >= 2:
    overall = "MODERATE - REQUIRES SOME ATTENTION"
else:
    overall = "GOOD - READY FOR MODELING"

print(f"\nOVERALL SUITABILITY: {overall}")
print(f"Recommended next steps: {'Address critical issues before modeling' if warning_count >= 2 else 'Proceed with modeling with caution' if warning_count >= 1 else 'Proceed with modeling'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("="*80)