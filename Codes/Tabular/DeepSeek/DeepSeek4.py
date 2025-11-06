import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Start timing
start_time = time.time()

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("COMPREHENSIVE DIABETES DATASET ANALYSIS")
print("=" * 60)

# 1. EXPLORATORY DATA ANALYSIS (EDA)
print("\n" + "="*50)
print("1. EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)

# Load dataset
df = pd.read_csv('diabetes_dataset_nan.csv')
print(f"Dataset shape: {df.shape}")

# Basic info
print("\nDataset Info:")
print(df.info())

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Missing values analysis
print("\nMissing values per column:")
missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0])

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Target variable analysis
target_column = 'diagnosed_diabetes'
print(f"\nTarget variable '{target_column}' distribution:")
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
df[target_column].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Target Variable Distribution')
plt.xlabel('Diagnosed Diabetes')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df[target_column].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.title('Target Variable Percentage')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Histograms for numerical features
print("\nGenerating numerical features histograms...")
numerical_df = df[numerical_cols]
if len(numerical_cols) > 0:
    n_cols = 4
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Boxplots for outlier detection
print("\nGenerating boxplots for outlier detection...")
if len(numerical_cols) > 0:
    n_cols = 4
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')
    
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Correlation heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(16, 12))
correlation_matrix = df[numerical_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap of Numerical Features', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Countplots for categorical variables
print("\nGenerating countplots for categorical variables...")
if len(categorical_cols) > 0:
    n_cols = 3
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            value_counts = df[col].value_counts()
            # Limit to top 10 categories if too many
            if len(value_counts) > 10:
                top_categories = value_counts.head(10)
                data_to_plot = df[df[col].isin(top_categories.index)]
            else:
                data_to_plot = df
            
            sns.countplot(data=data_to_plot, x=col, ax=axes[i], order=value_counts.index)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
    
    for i in range(len(categorical_cols), len(axes)):
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

# Numerical columns: impute with median
numerical_imputer = SimpleImputer(strategy='median')
df_processed[numerical_cols] = numerical_imputer.fit_transform(df_processed[numerical_cols])

# Categorical columns: impute with mode
for col in categorical_cols:
    if df_processed[col].isnull().sum() > 0:
        mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
        df_processed[col].fillna(mode_value, inplace=True)

print("Missing values after imputation:")
print(df_processed.isnull().sum().sum(), "total missing values remaining")

# Encode categorical features
print("\nEncoding categorical features...")
label_encoders = {}
for col in categorical_cols:
    if col != target_column:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col} with {len(le.classes_)} categories")

# Scale numerical features (excluding target)
print("\nScaling numerical features...")
scaler = StandardScaler()
numerical_cols_to_scale = [col for col in numerical_cols if col != target_column]
df_processed[numerical_cols_to_scale] = scaler.fit_transform(df_processed[numerical_cols_to_scale])

print("Data preprocessing completed successfully!")

# Check class imbalance
print(f"\nClass distribution in target variable:")
print(df_processed[target_column].value_counts())
imbalance_ratio = df_processed[target_column].value_counts().min() / df_processed[target_column].value_counts().max()
print(f"Imbalance ratio: {imbalance_ratio:.3f}")

# Apply SMOTE if imbalance is significant
X = df_processed.drop(columns=[target_column])
y = df_processed[target_column]

if imbalance_ratio < 0.5:
    print("\nSignificant class imbalance detected. Applying SMOTE...")
    print(f"Before SMOTE - Class distribution: {dict(y.value_counts())}")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"After SMOTE - Class distribution: {dict(pd.Series(y_resampled).value_counts())}")
    X, y = X_resampled, y_resampled
else:
    print("\nNo significant class imbalance detected. SMOTE not applied.")

# 3. FEATURE RELATIONSHIP ANALYSIS
print("\n" + "="*50)
print("3. FEATURE RELATIONSHIP ANALYSIS")
print("="*50)

# Prepare data for correlation analysis
analysis_df = pd.concat([X, y], axis=1)

# Calculate correlation with target
correlation_with_target = []
p_values = []

for col in X.columns:
    if col in numerical_cols:
        # Pearson correlation for numerical features
        corr, p_val = stats.pearsonr(analysis_df[col], analysis_df[target_column])
    else:
        # Point-biserial correlation for categorical vs binary target
        corr, p_val = stats.pointbiserialr(analysis_df[col], analysis_df[target_column])
    
    correlation_with_target.append(corr)
    p_values.append(p_val)

# Create correlation summary
correlation_summary = pd.DataFrame({
    'Feature': X.columns,
    'Correlation': correlation_with_target,
    'P-value': p_values,
    'Abs_Correlation': np.abs(correlation_with_target)
}).sort_values('Abs_Correlation', ascending=False)

print("\nTop 20 features most correlated with target:")
print(correlation_summary.head(20).to_string(index=False))

# Visualize top correlations
plt.figure(figsize=(12, 8))
top_features = correlation_summary.head(15)
sns.barplot(data=top_features, x='Correlation', y='Feature', palette='viridis')
plt.title('Top 15 Features Correlated with Diabetes Diagnosis', fontsize=16)
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()

# Feature importance using ANOVA F-value
print("\nCalculating feature importance using ANOVA F-value...")
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'F_Score': selector.scores_,
    'P_Value': selector.pvalues_
}).sort_values('F_Score', ascending=False)

print("\nTop 20 most important features (ANOVA F-score):")
print(feature_scores.head(20).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_important = feature_scores.head(15)
sns.barplot(data=top_important, x='F_Score', y='Feature', palette='plasma')
plt.title('Top 15 Most Important Features (ANOVA F-score)', fontsize=16)
plt.xlabel('F-Score')
plt.tight_layout()
plt.show()

# Pairplot for top 5 correlated features
print("\nGenerating pairplot for top 5 correlated features...")
top_corr_features = correlation_summary.head(5)['Feature'].tolist()
if len(top_corr_features) >= 3:  # Need at least 3 features for meaningful pairplot
    pairplot_data = analysis_df[top_corr_features + [target_column]]
    pairplot_data[target_column] = pairplot_data[target_column].astype(str)
    
    g = sns.pairplot(pairplot_data, hue=target_column, palette='Set1', diag_kind='kde')
    g.fig.suptitle('Pairplot of Top 5 Correlated Features with Target', y=1.02)
    plt.show()

# 4. TIME MEASUREMENT
print("\n" + "="*50)
print("4. TIME MEASUREMENT")
print("="*50)

end_time = time.time()
total_time = end_time - start_time

minutes = int(total_time // 60)
seconds = total_time % 60

print(f"Total Simulated Runtime: {minutes} minutes and {seconds:.2f} seconds")

# 5. CONCLUSION ON DATA SUITABILITY
print("\n" + "="*50)
print("5. DATA SUITABILITY ASSESSMENT")
print("="*50)

# Automated assessment
print("\nAUTOMATED DATA SUITABILITY ASSESSMENT:")
print("-" * 40)

# Check data quality metrics
original_missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
final_missing_percent = (df_processed.isnull().sum().sum() / (df_processed.shape[0] * df_processed.shape[1])) * 100
class_balance_original = df[target_column].value_counts(normalize=True).min()
feature_count = X.shape[1]
sample_size = X.shape[0]

# High correlation features count
high_corr_features = correlation_summary[correlation_summary['Abs_Correlation'] > 0.3].shape[0]
strong_corr_features = correlation_summary[correlation_summary['Abs_Correlation'] > 0.5].shape[0]

print(f"✓ Original dataset missingness: {original_missing_percent:.2f}%")
print(f"✓ Final dataset missingness: {final_missing_percent:.2f}%")
print(f"✓ Original class balance: {class_balance_original:.3f}")
print(f"✓ Number of features: {feature_count}")
print(f"✓ Sample size: {sample_size}")
print(f"✓ Features with |correlation| > 0.3: {high_corr_features}")
print(f"✓ Features with |correlation| > 0.5: {strong_corr_features}")

# Suitability assessment
suitability_score = 0

if original_missing_percent < 10:
    suitability_score += 2
    print("✓ GOOD: Low initial missing data")
elif original_missing_percent < 25:
    suitability_score += 1
    print("○ MODERATE: Moderate missing data handled well")
else:
    print("✗ CONCERN: High initial missing data")

if class_balance_original > 0.3:
    suitability_score += 2
    print("✓ GOOD: Balanced classes")
elif class_balance_original > 0.2:
    suitability_score += 1
    print("○ MODERATE: Moderate class imbalance")
else:
    print("✗ CONCERN: Severe class imbalance detected")

if sample_size > 1000:
    suitability_score += 2
    print("✓ GOOD: Large sample size")
elif sample_size > 500:
    suitability_score += 1
    print("○ MODERATE: Moderate sample size")
else:
    print("✗ CONCERN: Small sample size")

if high_corr_features >= 5:
    suitability_score += 2
    print("✓ GOOD: Multiple features show strong correlation with target")
elif high_corr_features >= 2:
    suitability_score += 1
    print("○ MODERATE: Some features correlate with target")
else:
    print("✗ CONCERN: Few features correlate with target")

# Final recommendation
print(f"\nOVERALL SUITABILITY SCORE: {suitability_score}/8")

if suitability_score >= 6:
    print("🎯 RECOMMENDATION: HIGHLY SUITABLE for classification modeling")
    print("   - Good data quality and feature relationships")
    print("   - Expected to yield good predictive performance")
elif suitability_score >= 4:
    print("🎯 RECOMMENDATION: MODERATELY SUITABLE for classification modeling") 
    print("   - Some data quality concerns but workable")
    print("   - May require additional feature engineering")
else:
    print("🎯 RECOMMENDATION: LIMITED SUITABILITY for classification modeling")
    print("   - Significant data quality issues")
    print("   - Consider data collection improvements or alternative approaches")

print("\n" + "="*60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("="*60)