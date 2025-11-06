import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pointbiserialr, spearmanr
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Start timing
start_time = time.time()

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("COMPREHENSIVE DIABETES DATASET ANALYSIS")
print("=" * 60)

# 1. EXPLORATORY DATA ANALYSIS (EDA)
print("\n" + "="*50)
print("1. EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('diabetes_dataset_nan.csv')

# Basic dataset info
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Remove unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
if '' in df.columns:
    df = df.drop('', axis=1)

# Dataset information
print("\nDataset Info:")
print(df.info())

# Missing values analysis
print("\nMissing Values Analysis:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0])

# Duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Target variable analysis
target_column = 'diagnosed_diabetes'
if target_column in df.columns:
    print(f"\nTarget Variable ('{target_column}') Distribution:")
    target_counts = df[target_column].value_counts()
    target_percent = df[target_column].value_counts(normalize=True) * 100
    target_summary = pd.DataFrame({
        'Count': target_counts,
        'Percentage': target_percent
    })
    print(target_summary)
    
    # Plot target distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df[target_column].value_counts().plot(kind='bar')
    plt.title('Target Variable Distribution')
    plt.xlabel('Diagnosed Diabetes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    df[target_column].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Target Variable Percentage')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical Columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")

# Histograms for numerical features
print("\nGenerating numerical features distribution plots...")
if numerical_cols:
    # Select first 12 numerical features for visualization
    cols_to_plot = numerical_cols[:12] if len(numerical_cols) > 12 else numerical_cols
    
    n_cols = 4
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(cols_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Boxplots for outlier detection
print("\nGenerating boxplots for outlier detection...")
if numerical_cols:
    cols_to_plot = numerical_cols[:12] if len(numerical_cols) > 12 else numerical_cols
    
    n_cols = 4
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        if i < len(axes):
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')
    
    for i in range(len(cols_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Correlation heatmap
print("\nGenerating correlation heatmap...")
if len(numerical_cols) > 1:
    plt.figure(figsize=(16, 14))
    correlation_matrix = df[numerical_cols].corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Find highly correlated features
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
        print("\nHighly Correlated Features (|r| > 0.8):")
        for feat1, feat2, corr in high_corr:
            print(f"  {feat1} - {feat2}: {corr:.3f}")

# Countplots for categorical variables
print("\nGenerating countplots for categorical variables...")
if categorical_cols:
    cols_to_plot = categorical_cols[:6] if len(categorical_cols) > 6 else categorical_cols
    
    n_cols = 2
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        if i < len(axes):
            value_counts = df[col].value_counts()
            # Show top 10 categories if too many
            if len(value_counts) > 10:
                top_categories = value_counts.head(10)
                top_categories.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Top 10 Categories in {col}')
            else:
                value_counts.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
    
    for i in range(len(cols_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# 2. DATA PREPROCESSING AND FEATURE ENGINEERING
print("\n" + "="*50)
print("2. DATA PREPROCESSING AND FEATURE ENGINEERING")
print("="*50)

# Create a copy for preprocessing
df_processed = df.copy()

# Handle missing values
print("\nHandling missing values...")

# Separate features and target
if target_column in df_processed.columns:
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]
else:
    X = df_processed
    y = None

# Impute numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns
if len(numerical_features) > 0:
    num_imputer = SimpleImputer(strategy='median')
    X[numerical_features] = num_imputer.fit_transform(X[numerical_features])
    print(f"Imputed {len(numerical_features)} numerical features using median")

# Impute categorical features  
categorical_features = X.select_dtypes(include=['object']).columns
if len(categorical_features) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])
    print(f"Imputed {len(categorical_features)} categorical features using mode")

# Encode categorical features
print("\nEncoding categorical features...")
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col} with {len(le.classes_)} categories")

# Scale numerical features
print("\nScaling numerical features...")
if len(numerical_features) > 0:
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    print("Scaled numerical features using StandardScaler")

# Handle class imbalance
if y is not None:
    print(f"\nClass distribution before handling imbalance:")
    class_counts_before = y.value_counts()
    print(class_counts_before)
    
    # Check for imbalance (if minority class < 40% of majority)
    minority_ratio = class_counts_before.min() / class_counts_before.max()
    if minority_ratio < 0.4:
        print("Significant class imbalance detected. Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"Class distribution after SMOTE:")
        class_counts_after = pd.Series(y_resampled).value_counts()
        print(class_counts_after)
        
        X = X_resampled
        y = y_resampled
    else:
        print("No significant class imbalance detected.")
        y_resampled = y

# 3. FEATURE RELATIONSHIP ANALYSIS
print("\n" + "="*50)
print("3. FEATURE RELATIONSHIP ANALYSIS")
print("="*50)

if y is not None:
    # Prepare data for correlation analysis
    analysis_df = X.copy()
    analysis_df[target_column] = y_resampled if 'y_resampled' in locals() else y
    
    # Calculate correlations with target
    correlation_results = []
    
    for col in X.columns:
        if col in numerical_features:
            # For numerical features vs binary target: point-biserial correlation
            corr, p_value = pointbiserialr(analysis_df[col], analysis_df[target_column])
        else:
            # For categorical vs binary target: Spearman correlation
            corr, p_value = spearmanr(analysis_df[col], analysis_df[target_column])
        
        correlation_results.append({
            'Feature': col,
            'Correlation': corr,
            'P-value': p_value,
            'Abs_Correlation': abs(corr)
        })
    
    # Create correlation results dataframe
    corr_df = pd.DataFrame(correlation_results)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    print("\nTop 20 Features by Absolute Correlation with Target:")
    print(corr_df.head(20).round(4))
    
    # Plot top correlated features
    top_features = corr_df.head(10)['Feature'].tolist()
    
    if top_features:
        print(f"\nVisualizing top {len(top_features)} correlated features...")
        
        # Create subplots
        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                if feature in numerical_features:
                    # For numerical features: boxplot by target
                    analysis_df.boxplot(column=feature, by=target_column, ax=axes[i])
                    axes[i].set_title(f'{feature}\n(Corr: {corr_df[corr_df["Feature"]==feature]["Correlation"].values[0]:.3f})')
                else:
                    # For categorical features: countplot
                    pd.crosstab(analysis_df[feature], analysis_df[target_column]).plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'{feature}\n(Corr: {corr_df[corr_df["Feature"]==feature]["Correlation"].values[0]:.3f})')
                    axes[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Top Features Correlation with Target Variable', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # Feature importance using ANOVA F-value
    print("\nCalculating feature importance using ANOVA F-value...")
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y_resampled if 'y_resampled' in locals() else y)
    
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_
    })
    feature_scores = feature_scores.sort_values('F_Score', ascending=False)
    
    print("\nTop 20 Features by ANOVA F-score:")
    print(feature_scores.head(20).round(4))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_f_features = feature_scores.head(15)
    plt.barh(top_f_features['Feature'], top_f_features['F_Score'])
    plt.xlabel('F-Score')
    plt.title('Top 15 Features by ANOVA F-Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# 4. TIME MEASUREMENT
print("\n" + "="*50)
print("4. TIME MEASUREMENT")
print("="*50)

end_time = time.time()
total_seconds = end_time - start_time
minutes = int(total_seconds // 60)
seconds = total_seconds % 60

print(f"Total Simulated Runtime: {minutes} minutes and {seconds:.2f} seconds")

# 5. CONCLUSION ON DATA SUITABILITY
print("\n" + "="*50)
print("5. DATA SUITABILITY ASSESSMENT")
print("="*50)

# Automated assessment
suitability_summary = []

# Check dataset size
if df.shape[0] < 1000:
    suitability_summary.append("⚠️  Dataset size is relatively small for robust modeling")
else:
    suitability_summary.append("✅  Dataset has sufficient samples for modeling")

# Check feature count
if df.shape[1] < 10:
    suitability_summary.append("⚠️  Limited number of features available")
else:
    suitability_summary.append("✅  Good number of features for analysis")

# Check missing values
total_missing = df.isnull().sum().sum()
missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100
if missing_percentage > 10:
    suitability_summary.append(f"⚠️  High missing data ({missing_percentage:.1f}%) - may require advanced imputation")
else:
    suitability_summary.append(f"✅  Low missing data ({missing_percentage:.1f}%) - easily manageable")

# Check class balance if target exists
if target_column in df.columns:
    target_balance = df[target_column].value_counts(normalize=True)
    imbalance_ratio = target_balance.min() / target_balance.max()
    if imbalance_ratio < 0.3:
        suitability_summary.append(f"⚠️  Significant class imbalance detected (ratio: {imbalance_ratio:.2f})")
    else:
        suitability_summary.append(f"✅  Reasonable class balance (ratio: {imbalance_ratio:.2f})")

# Check for high correlations
if 'high_corr' in locals() and high_corr:
    suitability_summary.append(f"⚠️  {len(high_corr)} highly correlated feature pairs detected")
else:
    suitability_summary.append("✅  No severe multicollinearity issues")

# Check data types diversity
if len(numerical_cols) > 5 and len(categorical_cols) > 2:
    suitability_summary.append("✅  Good mix of numerical and categorical features")
else:
    suitability_summary.append("⚠️  Limited diversity in feature types")

# Overall assessment
print("DATA SUITABILITY SUMMARY:")
print("-" * 30)
for item in suitability_summary:
    print(item)

# Final recommendation
positive_indicators = sum(1 for item in suitability_summary if item.startswith("✅"))
total_indicators = len(suitability_summary)

if positive_indicators / total_indicators >= 0.7:
    print("\n🎯 OVERALL ASSESSMENT: Dataset is WELL SUITED for classification modeling")
elif positive_indicators / total_indicators >= 0.5:
    print("\n🔄 OVERALL ASSESSMENT: Dataset is MODERATELY SUITED for classification modeling (some preprocessing needed)")
else:
    print("\n⛔ OVERALL ASSESSMENT: Dataset may be CHALLENGING for classification modeling")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)