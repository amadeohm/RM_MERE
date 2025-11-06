import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pointbiserialr, spearmanr
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Start timing
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
print("1. EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('diabetes_dataset_nan.csv')

# Basic dataset info
print(f"\nDataset Shape: {df.shape}")
print(f"\nColumn Types:")
print(df.dtypes)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Missing values analysis
print("\nMissing Values:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0])

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Target variable analysis
print("\nTarget Variable Distribution (diagnosed_diabetes):")
target_dist = df['diagnosed_diabetes'].value_counts()
print(target_dist)
print(f"\nTarget Proportions:")
print(df['diagnosed_diabetes'].value_counts(normalize=True))

# Visualizations
print("\nGenerating visualizations...")

# Set up the plotting layout
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Exploratory Data Analysis - Key Visualizations', fontsize=16)

# 1. Target variable distribution
axes[0,0].pie(target_dist.values, labels=target_dist.index, autopct='%1.1f%%', startangle=90)
axes[0,0].set_title('Target Variable Distribution (diagnosed_diabetes)')

# 2. Histograms for key numerical features
numerical_cols = ['age', 'bmi', 'glucose_fasting', 'hba1c', 'diabetes_risk_score']
df[numerical_cols].hist(ax=axes[0,1], bins=20)
axes[0,1].set_title('Distribution of Key Numerical Features')

# 3. Boxplots for outlier detection
df[numerical_cols].boxplot(ax=axes[1,0])
axes[1,0].set_title('Boxplots for Outlier Detection')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Correlation heatmap (first 15 numerical features for clarity)
numerical_features = df.select_dtypes(include=[np.number]).columns[:15]
corr_matrix = df[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,1])
axes[1,1].set_title('Correlation Heatmap (Top 15 Numerical Features)')

plt.tight_layout()
plt.show()

# Additional plots
# Categorical variables count plots
categorical_cols = ['gender', 'ethnicity', 'education_level', 'smoking_status', 'diabetes_stage']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, col in enumerate(categorical_cols):
    if col in df.columns:
        df[col].value_counts().plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].tick_params(axis='x', rotation=45)

# Hide empty subplots
for i in range(len(categorical_cols), 6):
    axes[i].set_visible(False)

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
print("\nHandling missing values...")

# Separate numerical and categorical columns
numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

# Remove target and identifier columns
if 'diagnosed_diabetes' in numerical_cols:
    numerical_cols.remove('diagnosed_diabetes')
if 'Unnamed: 0' in numerical_cols:
    numerical_cols.remove('Unnamed: 0')
if 'diabetes_stage' in categorical_cols:
    categorical_cols.remove('diabetes_stage')

# Impute numerical missing values with median
num_imputer = SimpleImputer(strategy='median')
df_processed[numerical_cols] = num_imputer.fit_transform(df_processed[numerical_cols])

# Impute categorical missing values with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

print("Missing values after imputation:")
print(df_processed.isnull().sum().sum())

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}
for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

# Scale numerical features
print("Scaling numerical features...")
scaler = StandardScaler()
df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

# Check class imbalance
print("\nClass Balance Analysis:")
target_counts = df_processed['diagnosed_diabetes'].value_counts()
print(f"Class distribution: {dict(target_counts)}")
imbalance_ratio = target_counts[0] / target_counts[1] if len(target_counts) > 1 else 1
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

# Apply SMOTE if significant imbalance exists
# Prepare features for modeling (exclude non-predictive columns)
exclude_cols = ['Unnamed: 0', 'diabetes_stage']  # Remove identifier and target-related columns
feature_cols = [col for col in df_processed.columns if col not in exclude_cols and col != 'diagnosed_diabetes']

X = df_processed[feature_cols]
y = df_processed['diagnosed_diabetes']

if imbalance_ratio > 1.5 or imbalance_ratio < 0.67:
    print("Significant class imbalance detected. Applying SMOTE...")
    print(f"Before SMOTE: {dict(y.value_counts())}")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"After SMOTE: {dict(pd.Series(y_resampled).value_counts())}")
    df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                           pd.Series(y_resampled, name='diagnosed_diabetes')], axis=1)
else:
    print("Class balance is acceptable. No resampling needed.")
    df_balanced = df_processed.copy()

print(f"\nFinal dataset shape: {df_balanced.shape}")

# =============================================================================
# 3. FEATURE RELATIONSHIP ANALYSIS
# =============================================================================
print("\n" + "="*50)
print("3. FEATURE RELATIONSHIP ANALYSIS")
print("="*50)

# Prepare features for correlation analysis
features_for_corr = [col for col in df_balanced.columns if col != 'diagnosed_diabetes']
# Use only numerical features for correlation analysis
numerical_features_for_corr = [col for col in features_for_corr if df_balanced[col].dtype in [np.int64, np.float64]]

print("Computing correlation with target variable...")

# Calculate correlations with target
correlation_results = []
for feature in numerical_features_for_corr:
    if len(df_balanced[feature].unique()) > 1:  # Avoid constant features
        corr, p_value = pointbiserialr(df_balanced[feature], df_balanced['diagnosed_diabetes'])
        correlation_results.append({
            'Feature': feature,
            'Correlation': corr,
            'P-value': p_value,
            'Abs_Correlation': abs(corr)
        })

# Create correlation dataframe
corr_df = pd.DataFrame(correlation_results)
corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)

print("\nTop 15 features by absolute correlation with target:")
print(corr_df.head(15).round(4))

# Visualize top correlations
top_features = corr_df.head(10)['Feature'].tolist()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Top correlations bar plot
axes[0,0].barh(range(len(top_features)), corr_df.head(10)['Correlation'])
axes[0,0].set_yticks(range(len(top_features)))
axes[0,0].set_yticklabels(top_features)
axes[0,0].set_xlabel('Correlation Coefficient')
axes[0,0].set_title('Top 10 Features Correlation with Diabetes Diagnosis')
axes[0,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)

# 2. Correlation heatmap for top features
top_features_with_target = top_features + ['diagnosed_diabetes']
corr_top = df_balanced[top_features_with_target].corr()
sns.heatmap(corr_top, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0,1])
axes[0,1].set_title('Correlation Heatmap - Top Features')

# 3. Scatter plots for top 2 correlated features
if len(top_features) >= 2:
    top1, top2 = top_features[:2]
    sns.scatterplot(data=df_balanced, x=top1, y=top2, hue='diagnosed_diabetes', alpha=0.6, ax=axes[1,0])
    axes[1,0].set_title(f'{top1} vs {top2} by Diabetes Status')
    
    # 4. Distribution of top feature by target
    sns.boxplot(data=df_balanced, x='diagnosed_diabetes', y=top1, ax=axes[1,1])
    axes[1,1].set_title(f'Distribution of {top1} by Diabetes Status')

plt.tight_layout()
plt.show()

# Additional pairplot for top 4 features
if len(top_features) >= 4:
    print("\nGenerating pairplot for top 4 correlated features...")
    pairplot_features = top_features[:4] + ['diagnosed_diabetes']
    sns.pairplot(df_balanced[pairplot_features], hue='diagnosed_diabetes', diag_kind='kde', palette='viridis')
    plt.suptitle('Pairplot of Top 4 Correlated Features with Diabetes Diagnosis', y=1.02)
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

# Calculate key metrics for assessment
missing_before = df.isnull().sum().sum()
missing_after = df_processed.isnull().sum().sum()
original_shape = df.shape
processed_shape = df_balanced.shape
high_corr_features = len(corr_df[corr_df['Abs_Correlation'] > 0.3])
strong_corr_features = len(corr_df[corr_df['Abs_Correlation'] > 0.5])
sig_features = len(corr_df[corr_df['P-value'] < 0.05])

# Generate automated assessment
print("AUTOMATED DATA SUITABILITY ASSESSMENT:")
print(f"✓ Original dataset: {original_shape[0]} samples, {original_shape[1]} features")
print(f"✓ Missing values handled: {missing_before} → {missing_after}")
print(f"✓ Final balanced dataset: {processed_shape[0]} samples, {processed_shape[1]} features")
print(f"✓ Statistically significant features (p<0.05): {sig_features}/{len(corr_df)}")
print(f"✓ Features with moderate correlation (>0.3): {high_corr_features}")
print(f"✓ Features with strong correlation (>0.5): {strong_corr_features}")
print(f"✓ Class imbalance ratio: {imbalance_ratio:.2f}")

# Overall suitability judgment
if (high_corr_features >= 5 and 
    processed_shape[0] > 500 and 
    0.5 <= imbalance_ratio <= 2.0 and
    missing_after == 0 and
    sig_features >= 10):
    suitability = "HIGHLY SUITABLE"
    reasoning = "Excellent sample size, strong feature correlations, balanced classes, clean data with statistical significance"
elif (high_corr_features >= 3 and 
      processed_shape[0] > 300 and 
      missing_after == 0 and
      sig_features >= 5):
    suitability = "MODERATELY SUITABLE" 
    reasoning = "Adequate features and sample size with statistical significance"
else:
    suitability = "LESS SUITABLE"
    reasoning = "Limited predictive features, small sample size, or data quality issues detected"

print(f"\nOVERALL SUITABILITY: {suitability}")
print(f"REASONING: {reasoning}")

# Additional recommendations
print(f"\nRECOMMENDATIONS:")
if high_corr_features < 5:
    print("- Consider feature engineering to create more predictive features")
if processed_shape[0] < 1000:
    print("- Larger dataset would improve model robustness")
if strong_corr_features < 3:
    print("- Focus on the most correlated features for model building")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)