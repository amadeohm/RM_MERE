import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import time
from collections import Counter

# For sentiment analysis
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# For topic modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

# Start timing
start_time = time.time()

print("=" * 70)
print("AUTOMATED TEXT ANALYSIS PIPELINE")
print("=" * 70)

# Section 1: Text Data Loading and Overview
print("\n" + "="*50)
print("1. TEXT DATA LOADING AND OVERVIEW")
print("="*50)

# Load the dataset with error handling for inconsistent fields
df = pd.read_csv('SemEval2017-task4-dev.subtask-A.english.INPUT.csv', encoding='utf-8', on_bad_lines='skip')


print(f"Dataset shape: {df.shape}")

# Display basic information
print(f"\nColumn names: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Check if dataframe is empty
if df.empty:
    print("Error: The dataframe is empty. Please check the CSV file.")
    exit()

# Display first few rows to understand structure
print(f"\nFirst 3 rows of the dataset:")
print(df.head(3))

# Automatically detect text column
text_columns = []
for col in df.columns:
    if df[col].dtype == 'object':
        # Check if this column contains substantial text
        avg_length = df[col].astype(str).str.len().mean()
        if avg_length > 10:  # Reasonable threshold for text
            text_columns.append((col, avg_length))

if text_columns:
    # Sort by average length and pick the longest one
    text_columns.sort(key=lambda x: x[1], reverse=True)
    text_column = text_columns[0][0]
    print(f"\nDetected text column: '{text_column}' (avg length: {text_columns[0][1]:.1f} chars)")
else:
    # If no obvious text column, use the first object column
    text_column = df.select_dtypes(include=['object']).columns[0]
    print(f"\nUsing first object column as text: '{text_column}'")

# Clean the text data
print("\nCleaning text data...")
df['cleaned_text'] = df[text_column].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
df['cleaned_text'] = df['cleaned_text'].str.replace('\s+', ' ', regex=True).str.strip()

# Remove empty texts
df = df[df['cleaned_text'].str.len() > 0]

df['text_length_chars'] = df['cleaned_text'].str.len()
df['text_length_words'] = df['cleaned_text'].str.split().str.len()

print(f"\nNumber of text entries: {len(df)}")
print(f"Average text length (characters): {df['text_length_chars'].mean():.2f}")
print(f"Average text length (words): {df['text_length_words'].mean():.2f}")

# Display sample rows
print(f"\nSample text entries:")
for i, text in enumerate(df['cleaned_text'].head(3).values):
    print(f"{i+1}. {text[:100]}...")

# Text length distribution histogram
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df['text_length_words'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Text Length (Words)')
plt.ylabel('Frequency')
plt.title('Distribution of Text Lengths (Words)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(df['text_length_chars'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.xlabel('Text Length (Characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Text Lengths (Characters)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Section 2: Sentiment Analysis
print("\n" + "="*50)
print("2. SENTIMENT ANALYSIS")
print("="*50)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    """Get sentiment scores using VADER"""
    return sia.polarity_scores(text)

def get_sentiment_label(scores):
    """Convert sentiment scores to labels"""
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

print("Calculating sentiment scores...")
# Calculate sentiment scores
df['sentiment_scores'] = df['cleaned_text'].apply(get_sentiment_scores)
df['sentiment_compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
df['sentiment_label'] = df['sentiment_scores'].apply(get_sentiment_label)

# Display sentiment distribution
sentiment_counts = df['sentiment_label'].value_counts()
print(f"\nSentiment Distribution:")
for label, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{label.capitalize()}: {count} ({percentage:.2f}%)")

print(f"\nAverage Sentiment Score: {df['sentiment_compound'].mean():.3f}")
print(f"Sentiment Score Std Dev: {df['sentiment_compound'].std():.3f}")

# Display sample sentiment results
print(f"\nSample Sentiment Analysis Results:")
sample_df = df[['cleaned_text', 'sentiment_label', 'sentiment_compound']].head(5)
for idx, row in sample_df.iterrows():
    print(f"Text: {row['cleaned_text'][:80]}...")
    print(f"Sentiment: {row['sentiment_label']} (Score: {row['sentiment_compound']:.3f})\n")

# Sentiment visualizations
plt.figure(figsize=(15, 5))

# Pie chart of sentiment labels
plt.subplot(1, 3, 1)
colors = ['#ff9999', '#66b3ff', '#99ff99']
plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Sentiment Label Distribution')

# Bar chart of sentiment labels
plt.subplot(1, 3, 2)
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Label Distribution')

# Histogram of sentiment scores
plt.subplot(1, 3, 3)
plt.hist(df['sentiment_compound'], bins=30, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Scores')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Section 3: Topic Modeling
print("\n" + "="*50)
print("3. TOPIC MODELING")
print("="*50)

# Prepare text data for topic modeling
texts = df['cleaned_text'].tolist()

print(f"Preparing {len(texts)} documents for topic modeling...")

# Filter out very short texts for better topic modeling
filtered_texts = [text for text in texts if len(text.split()) >= 3]
filtered_indices = [i for i, text in enumerate(texts) if len(text.split()) >= 3]
filtered_df = df.iloc[filtered_indices].copy()

print(f"Using {len(filtered_texts)} documents with 3+ words for topic modeling")

if len(filtered_texts) < 10:
    print("Warning: Too few documents for meaningful topic modeling. Skipping this section.")
else:
    # Create stop words list
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback stop words if NLTK download fails
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
                         "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                         'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                         'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
                         'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                         'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                         'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                         'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
                         'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'])
    
    # Add custom stop words
    custom_stop_words = list(stop_words.union(['oh', 'yeah', 'hey', 'like', 'get', 'go', 'know', 'would', 'could', 'said']))
    
    vectorizer = CountVectorizer(
        max_df=0.95, 
        min_df=2, 
        stop_words=custom_stop_words,
        max_features=1000
    )
    
    try:
        doc_term_matrix = vectorizer.fit_transform(filtered_texts)
        
        # Determine optimal number of topics
        n_topics = min(5, len(filtered_texts) // 10)
        n_topics = max(2, n_topics)
        print(f"Using {n_topics} topics for modeling")
        
        # Apply LDA
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda_output = lda_model.fit_transform(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Display top keywords per topic
        print(f"\nTop Keywords per Topic:")
        for topic_idx, topic in enumerate(lda_model.components_):
            top_keywords_idx = topic.argsort()[-10:][::-1]
            top_keywords = [feature_names[i] for i in top_keywords_idx]
            print(f"Topic {topic_idx+1}: {', '.join(top_keywords)}")
        
        # Assign each document to its most likely topic
        filtered_df['dominant_topic'] = lda_output.argmax(axis=1)
        
        # Update original dataframe
        df.loc[filtered_df.index, 'dominant_topic'] = filtered_df['dominant_topic']
        
        # Display topic distribution
        topic_counts = filtered_df['dominant_topic'].value_counts().sort_index()
        print(f"\nTopic Distribution:")
        for topic_idx, count in topic_counts.items():
            percentage = (count / len(filtered_df)) * 100
            print(f"Topic {topic_idx+1}: {count} documents ({percentage:.2f}%)")
        
        # Display sample texts per topic
        print(f"\nSample Texts per Topic:")
        for topic_idx in range(n_topics):
            print(f"\nTopic {topic_idx+1} Sample Texts:")
            topic_texts = filtered_df[filtered_df['dominant_topic'] == topic_idx]['cleaned_text'].head(2)
            for i, text in enumerate(topic_texts):
                print(f"{i+1}. {text[:100]}...")
        
        # Topic visualization
        plt.figure(figsize=(12, 5))
        
        # Bar plot of topic frequencies
        plt.subplot(1, 2, 1)
        topic_labels = [f'Topic {i+1}' for i in range(n_topics)]
        plt.bar(topic_labels, topic_counts.values, color=plt.cm.Set3(np.linspace(0, 1, n_topics)))
        plt.xlabel('Topics')
        plt.ylabel('Number of Documents')
        plt.title('Topic Frequency Distribution')
        plt.xticks(rotation=45)
        
        # Sentiment by topic
        plt.subplot(1, 2, 2)
        sentiment_by_topic = filtered_df.groupby('dominant_topic')['sentiment_compound'].mean()
        plt.bar(topic_labels, sentiment_by_topic.values, 
                 color=plt.cm.coolwarm(np.linspace(0, 1, n_topics)))
        plt.xlabel('Topics')
        plt.ylabel('Average Sentiment Score')
        plt.title('Average Sentiment by Topic')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in topic modeling: {e}")
        print("Skipping topic modeling section.")

# Section 4: Overall Summary
print("\n" + "="*50)
print("4. OVERALL SUMMARY")
print("="*50)

# Calculate overall statistics
dominant_sentiment = sentiment_counts.idxmax()
sentiment_variance = df['sentiment_compound'].var()

print(f"Dataset Overview:")
print(f"- Total documents: {len(df)}")
print(f"- Dominant sentiment: {dominant_sentiment} ({sentiment_counts.max()/len(df)*100:.1f}%)")
print(f"- Sentiment variance: {sentiment_variance:.3f}")

# Check if topic modeling was successful
if 'dominant_topic' in df.columns and not df['dominant_topic'].isna().all():
    topic_counts_summary = df['dominant_topic'].value_counts()
    n_topics_summary = len(topic_counts_summary)
    topic_diversity = len(topic_counts_summary) / n_topics_summary if n_topics_summary > 0 else 0
    
    print(f"- Number of topics identified: {n_topics_summary}")
    print(f"- Topic diversity index: {topic_diversity:.3f}")
    
    # Concluding statement with topics
    if sentiment_variance > 0.1:
        sentiment_diversity = "highly diverse"
    elif sentiment_variance > 0.05:
        sentiment_diversity = "moderately diverse"
    else:
        sentiment_diversity = "relatively uniform"

    if topic_diversity > 0.7:
        topic_statement = "good thematic diversity"
    elif topic_diversity > 0.4:
        topic_statement = "moderate thematic diversity"
    else:
        topic_statement = "limited thematic range"

    print(f"\nCONCLUSION:")
    print(f"The dataset shows {sentiment_diversity} sentiment profiles with {topic_statement}.")
    print(f"Sentiment is predominantly {dominant_sentiment}, while topics cover {n_topics_summary} distinct themes.")
else:
    print(f"\nCONCLUSION:")
    if sentiment_variance > 0.1:
        sentiment_diversity = "highly diverse"
    elif sentiment_variance > 0.05:
        sentiment_diversity = "moderately diverse"
    else:
        sentiment_diversity = "relatively uniform"
    
    print(f"The dataset shows {sentiment_diversity} sentiment profiles.")
    print(f"Sentiment is predominantly {dominant_sentiment}.")

# End timing and print runtime
end_time = time.time()
total_seconds = end_time - start_time
minutes = int(total_seconds // 60)
seconds = int(total_seconds % 60)

print(f"\n" + "="*70)
print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds")
print("="*70)

print("\nAnalysis complete!")