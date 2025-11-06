import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pointbiserialr, spearmanr
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

# =============================================================================
# 1. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "="*50)
print("1. EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('diabetes_dataset_nan.csv')

# Basic dataset info
print(f"Dataset Shape: {df.shape}")
print(f"Number of features: {df.shape[1]}")
print(f"Number of samples: {df.shape[0]}")

# Remove unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
if '' in df.columns:
    df = df.drop('', axis=1)

print("\nColumn Types:")
print(df.dtypes)

print("\nMissing Values:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_info[missing_info['Missing Count'] > 0])

print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# Basic statistics
print("\nBasic Descriptive Statistics:")
print(df.describe())

# Target variable analysis
print("\n" + "-"*30)
print("TARGET VARIABLE ANALYSIS")
print("-"*30)

# Check which column is the target (looking for binary diabetes indicator)
target_candidates = ['diagnosed_diabetes', 'diabetes_stage', 'diabetes_risk_score']
target_col = None
for col in target_candidates:
    if col in df.columns:
        target_col = col
        break

if target_col:
    print(f"Target variable: {target_col}")
    print(f"Target distribution:\n{df[target_col].value_counts()}")
    if df[target_col].dtype in ['object', 'category']:
        print(f"Target classes: {df[target_col].unique()}")
    
    plt.figure(figsize=(10, 6))
    if df[target_col].dtype in ['object', 'category']:
        df[target_col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {target_col}')
    else:
        plt.hist(df[target_col].dropna(), bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No clear target variable found in common diabetes columns")

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Histograms for numerical features
print("\nGenerating numerical feature distributions...")
numerical_to_plot = [col for col in numerical_cols if df[col].nunique() > 2]  # Avoid binary vars

if numerical_to_plot:
    n_cols = 4
    n_rows = (len(numerical_to_plot) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_to_plot):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Boxplots for outlier detection
print("\nGenerating boxplots for outlier detection...")
if numerical_to_plot:
    n_cols = 4
    n_rows = (len(numerical_to_plot) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_to_plot):
        if i < len(axes):
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Correlation heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(16, 12))
correlation_matrix = df[numerical_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()

# Countplots for categorical variables
print("\nGenerating categorical variable distributions...")
if categorical_cols:
    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            value_counts = df[col].value_counts()
            # Limit to top 10 categories if too many
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
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
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

print("Handling missing values...")
# Separate numerical and categorical columns again (in case types changed)
numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()

# Impute missing values
print("Numerical imputation (median):")
for col in numerical_cols:
    if df_processed[col].isnull().sum() > 0:
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        print(f"  {col}: {df_processed[col].isnull().sum()} missing -> filled with {median_val:.2f}")

print("\nCategorical imputation (mode):")
for col in categorical_cols:
    if df_processed[col].isnull().sum() > 0:
        mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
        df_processed[col].fillna(mode_val, inplace=True)
        print(f"  {col}: {df_processed[col].isnull().sum()} missing -> filled with '{mode_val}'")

print(f"\nRemaining missing values: {df_processed.isnull().sum().sum()}")

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}
for col in categorical_cols:
    if col != target_col:  # Don't encode target if it's categorical
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: encoded {len(le.classes_)} categories")

# If target is categorical, encode it separately
if target_col and df[target_col].dtype in ['object', 'category']:
    target_encoder = LabelEncoder()
    df_processed[target_col] = target_encoder.fit_transform(df_processed[target_col])
    print(f"  Target {target_col}: encoded {len(target_encoder.classes_)} classes")

# Scale numerical features (excluding target if it's numerical)
print("\nScaling numerical features...")
numerical_to_scale = [col for col in numerical_cols if col != target_col]
scaler = StandardScaler()
df_processed[numerical_to_scale] = scaler.fit_transform(df_processed[numerical_to_scale])
print(f"Scaled {len(numerical_to_scale)} numerical features")

# Check for class imbalance
if target_col:
    print("\n" + "-"*30)
    print("CLASS IMBALANCE ANALYSIS")
    print("-"*30)
    
    target_counts = df_processed[target_col].value_counts()
    print(f"Target distribution:\n{target_counts}")
    
    imbalance_ratio = target_counts.min() / target_counts.max()
    print(f"Imbalance ratio: {imbalance_ratio:.3f}")
    
    if imbalance_ratio < 0.5:
        print("Significant class imbalance detected!")
        print("Applying SMOTE...")
        
        # Prepare features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Create new balanced dataset
        df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        df_balanced[target_col] = y_resampled
        
        print("Before SMOTE:")
        print(f"  Class distribution: {dict(target_counts)}")
        print("After SMOTE:")
        print(f"  Class distribution: {dict(df_balanced[target_col].value_counts())}")
        
        df_processed = df_balanced
    else:
        print("No significant class imbalance detected")
        df_balanced = df_processed

# =============================================================================
# 3. FEATURE RELATIONSHIP ANALYSIS
# =============================================================================
print("\n" + "="*50)
print("3. FEATURE RELATIONSHIP ANALYSIS")
print("="*50)

if target_col:
    # Prepare data for correlation analysis
    X = df_balanced.drop(columns=[target_col])
    y = df_balanced[target_col]
    
    # Calculate correlations with target
    print("\nFeature correlations with target:")
    correlations = []
    
    for col in X.columns:
        if col in numerical_cols and col != target_col:
            # Use point-biserial correlation for numerical vs binary target
            if len(np.unique(y)) == 2:
                corr, p_value = pointbiserialr(X[col], y)
                corr_type = "Point-biserial"
            else:
                # Use Spearman for numerical vs categorical/multi-class
                corr, p_value = spearmanr(X[col], y, nan_policy='omit')
                corr_type = "Spearman"
        else:
            # For categorical vs categorical, use a different approach
            try:
                corr, p_value = spearmanr(X[col], y, nan_policy='omit')
                corr_type = "Spearman"
            except:
                corr, p_value = np.nan, np.nan
                corr_type = "N/A"
        
        correlations.append({
            'Feature': col,
            'Correlation': corr,
            'P-value': p_value,
            'Type': corr_type,
            'Abs_Correlation': abs(corr) if not np.isnan(corr) else 0
        })
    
    # Create correlation dataframe
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    # Display top correlations
    print("\nTop 15 features by absolute correlation with target:")
    display_cols = ['Feature', 'Correlation', 'P-value', 'Type']
    print(corr_df[display_cols].head(15).to_string(index=False))
    
    # Visualize top correlations
    plt.figure(figsize=(12, 8))
    top_features = corr_df.head(15)
    colors = ['red' if x < 0 else 'blue' for x in top_features['Correlation']]
    
    plt.barh(top_features['Feature'], top_features['Abs_Correlation'], color=colors)
    plt.xlabel('Absolute Correlation with Target')
    plt.title('Top 15 Features Correlated with Target Variable')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    # Scatter plots for top 4 correlated features
    top_4_features = corr_df.head(4)['Feature'].tolist()
    if len(top_4_features) >= 2:
        print(f"\nGenerating scatter plots for top features: {top_4_features}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_4_features[:4]):
            if i < len(axes):
                axes[i].scatter(df_balanced[feature], df_balanced[target_col], alpha=0.6)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel(target_col)
                axes[i].set_title(f'{feature} vs {target_col}\nCorr: {corr_df.iloc[i]["Correlation"]:.3f}')
        
        plt.tight_layout()
        plt.show()
    
    # Pairplot for top correlated features (limit to 5 to avoid overcrowding)
    if len(top_4_features) >= 3:
        print("\nGenerating pairplot for top correlated features...")
        pairplot_cols = top_4_features[:3] + [target_col]
        sns.pairplot(df_balanced[pairplot_cols], hue=target_col, diag_kind='kde', palette='viridis')
        plt.suptitle('Pairplot of Top Correlated Features with Target', y=1.02)
        plt.tight_layout()
        plt.show()

# =============================================================================
# 4. TIME MEASUREMENT
# =============================================================================
end_time = time.time()
total_time = end_time - start_time

minutes = int(total_time // 60)
seconds = total_time % 60

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

# Calculate key metrics for assessment
suitability_metrics = {}

# Missing data assessment
initial_missing = df.isnull().sum().sum()
initial_missing_pct = (initial_missing / (df.shape[0] * df.shape[1])) * 100
suitability_metrics['missing_data_pct'] = initial_missing_pct

# Dataset size assessment
suitability_metrics['sample_size'] = df.shape[0]
suitability_metrics['feature_count'] = df.shape[1]

# Class balance assessment
if target_col:
    target_counts = df[target_col].value_counts()
    imbalance_ratio = target_counts.min() / target_counts.max()
    suitability_metrics['imbalance_ratio'] = imbalance_ratio

# Correlation assessment
if target_col and 'corr_df' in locals():
    max_corr = corr_df['Abs_Correlation'].max()
    strong_correlations = len(corr_df[corr_df['Abs_Correlation'] > 0.3])
    suitability_metrics['max_correlation'] = max_corr
    suitability_metrics['strong_features'] = strong_correlations

# Generate suitability assessment
print("AUTOMATED DATA SUITABILITY ASSESSMENT:")
print("-" * 40)

# Missing data assessment
if initial_missing_pct < 5:
    print("✓ EXCELLENT: Minimal missing data (<5%)")
elif initial_missing_pct < 15:
    print("✓ GOOD: Moderate missing data (5-15%)")
elif initial_missing_pct < 30:
    print("⚠ FAIR: Considerable missing data (15-30%)")
else:
    print("✗ POOR: High missing data (>30%)")

# Sample size assessment
if suitability_metrics['sample_size'] > 1000:
    print("✓ EXCELLENT: Large sample size (>1000 samples)")
elif suitability_metrics['sample_size'] > 500:
    print("✓ GOOD: Adequate sample size (500-1000 samples)")
elif suitability_metrics['sample_size'] > 100:
    print("⚠ FAIR: Moderate sample size (100-500 samples)")
else:
    print("✗ POOR: Small sample size (<100 samples)")

# Feature count assessment
if suitability_metrics['feature_count'] > 20:
    print("✓ RICH: Comprehensive feature set (>20 features)")
elif suitability_metrics['feature_count'] > 10:
    print("✓ ADEQUATE: Good feature set (10-20 features)")
else:
    print("⚠ LIMITED: Small feature set (<10 features)")

# Class balance assessment
if target_col:
    if suitability_metrics['imbalance_ratio'] > 0.7:
        print("✓ EXCELLENT: Well-balanced classes")
    elif suitability_metrics['imbalance_ratio'] > 0.5:
        print("✓ GOOD: Moderately balanced classes")
    elif suitability_metrics['imbalance_ratio'] > 0.3:
        print("⚠ FAIR: Some class imbalance")
    else:
        print("✗ POOR: Significant class imbalance detected")

# Predictive potential assessment
if target_col and 'max_correlation' in suitability_metrics:
    if suitability_metrics['max_correlation'] > 0.5:
        print("✓ HIGH: Strong feature-target correlations detected")
    elif suitability_metrics['max_correlation'] > 0.3:
        print("✓ MODERATE: Moderate feature-target correlations")
    elif suitability_metrics['max_correlation'] > 0.1:
        print("⚠ LOW: Weak feature-target correlations")
    else:
        print("✗ MINIMAL: Very weak feature-target relationships")

print("\nOVERALL ASSESSMENT:")
if (initial_missing_pct < 15 and suitability_metrics['sample_size'] > 500 and 
    suitability_metrics['feature_count'] > 10 and 
    (not target_col or suitability_metrics['imbalance_ratio'] > 0.3)):
    print("✅ HIGHLY SUITABLE for classification modeling")
elif (initial_missing_pct < 30 and suitability_metrics['sample_size'] > 100 and 
      suitability_metrics['feature_count'] > 5):
    print("✅ MODERATELY SUITABLE for classification modeling")
else:
    print("❌ LIMITED SUITABILITY for classification modeling")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)