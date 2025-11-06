import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from collections import Counter

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# ML and topic modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Start timer
start_time = time.time()

print("=" * 80)
print("AUTOMATED TEXT ANALYSIS PIPELINE")
print("=" * 80)

# Section 1: Text Data Loading and Overview
print("\n" + "="*50)
print("1. TEXT DATA LOADING AND OVERVIEW")
print("="*50)

# Load dataset
df = pd.read_csv('SemEval2017-task4-dev.subtask-A.english.INPUT.csv')

# Basic dataset information
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data Types:\n{df.dtypes}")
print(f"Missing Values:\n{df.isnull().sum()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")

# Handle missing values
print("\nHandling missing values...")
initial_count = len(df)
df = df.dropna(subset=['text'])  # Remove rows with missing text
print(f"Removed {initial_count - len(df)} rows with missing text data")

# Auto-detect text column
text_columns = [col for col in df.columns if df[col].dtype == 'object' or isinstance(df[col].iloc[0], str)]
if 'text' in df.columns:
    text_col = 'text'
else:
    text_col = text_columns[0] if text_columns else df.columns[0]
    print(f"Auto-detected text column: '{text_col}'")

# Ensure all text entries are strings
df[text_col] = df[text_col].astype(str)

print(f"\nSample Text Entries:")
for i, text in enumerate(df[text_col].head(3).values):
    # Safely handle text slicing
    text_preview = str(text)[:100] + "..." if len(str(text)) > 100 else str(text)
    print(f"{i+1}. {text_preview}")

# Text statistics
text_lengths = df[text_col].apply(lambda x: len(str(x)))
word_counts = df[text_col].apply(lambda x: len(str(x).split()))

print(f"\nText Statistics:")
print(f"Total Text Entries: {len(df)}")
print(f"Average Text Length (characters): {text_lengths.mean():.2f}")
print(f"Average Word Count: {word_counts.mean():.2f}")
print(f"Shortest Text: {text_lengths.min()} characters")
print(f"Longest Text: {text_lengths.max()} characters")

# Text length distribution
plt.figure(figsize=(10, 6))
plt.hist(text_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Text Lengths (Characters)')
plt.xlabel('Text Length (Characters)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Section 2: Text Preprocessing
print("\n" + "="*50)
print("2. TEXT PREPROCESSING")
print("="*50)

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == 'nan' or text == 'None':
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

# Apply preprocessing
print("Preprocessing text data...")
df['cleaned_text'] = df[text_col].apply(preprocess_text)

# Remove empty texts after preprocessing
initial_clean_count = len(df)
df = df[df['cleaned_text'].str.len() > 0]
print(f"Removed {initial_clean_count - len(df)} empty texts after preprocessing")

print(f"Sample Preprocessed Texts:")
for i, text in enumerate(df['cleaned_text'].head(3).values):
    text_preview = str(text)[:100] + "..." if len(str(text)) > 100 else str(text)
    print(f"{i+1}. {text_preview}")

# Section 3: Sentiment Analysis
print("\n" + "="*50)
print("3. SENTIMENT ANALYSIS")
print("="*50)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    """Get sentiment scores using VADER"""
    return sia.polarity_scores(str(text))

def get_sentiment_label(compound_score):
    """Convert compound score to sentiment label"""
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis
print("Performing sentiment analysis...")
df['sentiment_scores'] = df['cleaned_text'].apply(get_sentiment_scores)
df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])
df['sentiment_label'] = df['compound_score'].apply(get_sentiment_label)

# Sentiment distribution
sentiment_counts = df['sentiment_label'].value_counts()
print(f"\nSentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{sentiment.capitalize()}: {count} ({percentage:.2f}%)")

# Display sample results
print(f"\nSample Sentiment Analysis Results:")
sample_results = df[['cleaned_text', 'compound_score', 'sentiment_label']].head(5)
for idx, row in sample_results.iterrows():
    text_preview = str(row['cleaned_text'])[:80] + "..." if len(str(row['cleaned_text'])) > 80 else str(row['cleaned_text'])
    print(f"Text: {text_preview}")
    print(f"Score: {row['compound_score']:.3f}, Label: {row['sentiment_label']}")
    print("-" * 50)

# Sentiment visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Pie chart for sentiment distribution
colors = ['#ff9999', '#66b3ff', '#99ff99']
ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax1.set_title('Sentiment Label Distribution')

# Histogram for sentiment scores
ax2.hist(df['compound_score'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
ax2.set_title('Distribution of Sentiment Scores')
ax2.set_xlabel('Compound Sentiment Score')
ax2.set_ylabel('Frequency')
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Section 4: Topic Modeling
print("\n" + "="*50)
print("4. TOPIC MODELING")
print("="*50)

# Prepare data for topic modeling
print("Preparing data for topic modeling...")
texts_for_tm = df['cleaned_text'][df['cleaned_text'].str.len() > 10].tolist()

if len(texts_for_tm) > 50:  # Only perform topic modeling if we have sufficient data
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    dtm = vectorizer.fit_transform(texts_for_tm)
    
    # Apply LDA
    n_topics = min(5, len(texts_for_tm) // 20)  # Adaptive number of topics
    n_topics = max(2, n_topics)  # At least 2 topics
    
    print(f"Training LDA with {n_topics} topics on {len(texts_for_tm)} documents...")
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(dtm)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Display top keywords per topic
    print(f"\nTop Keywords per Topic:")
    n_top_words = 10
    
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topic_weights = [topic[i] for i in top_features_ind]
        
        print(f"Topic {topic_idx + 1}:")
        print(" | ".join([f"{word} ({weight:.3f})" for word, weight in zip(top_features, topic_weights)]))
    
    # Assign topics to documents
    topic_results = lda.transform(dtm)
    df_tm = df[df['cleaned_text'].str.len() > 10].copy()
    df_tm['dominant_topic'] = topic_results.argmax(axis=1) + 1
    df_tm['topic_confidence'] = topic_results.max(axis=1)
    
    # Topic distribution
    topic_distribution = df_tm['dominant_topic'].value_counts().sort_index()
    print(f"\nTopic Distribution:")
    for topic, count in topic_distribution.items():
        percentage = (count / len(df_tm)) * 100
        print(f"Topic {topic}: {count} documents ({percentage:.2f}%)")
    
    # Display sample texts per topic
    print(f"\nSample Texts per Topic:")
    for topic in range(1, n_topics + 1):
        topic_texts = df_tm[df_tm['dominant_topic'] == topic]['cleaned_text'].head(2)
        print(f"\nTopic {topic} Samples:")
        for i, text in enumerate(topic_texts):
            text_preview = str(text)[:100] + "..." if len(str(text)) > 100 else str(text)
            print(f"  {i+1}. {text_preview}")
    
    # Topic visualization
    plt.figure(figsize=(10, 6))
    topic_distribution.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Topic Distribution')
    plt.xlabel('Topic Number')
    plt.ylabel('Number of Documents')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.show()
    
else:
    print(f"Insufficient data for topic modeling. Need at least 50 documents, have {len(texts_for_tm)}")
    df_tm = pd.DataFrame()  # Empty dataframe for consistency

# Section 5: Overall Summary
print("\n" + "="*50)
print("5. OVERALL SUMMARY")
print("="*50)

# Calculate overall statistics
total_texts = len(df)
avg_sentiment = df['compound_score'].mean()
sentiment_std = df['compound_score'].std()
most_common_sentiment = df['sentiment_label'].value_counts().index[0]

print(f"Dataset Summary:")
print(f"Total texts analyzed: {total_texts}")
print(f"Average sentiment score: {avg_sentiment:.3f}")
print(f"Sentiment score standard deviation: {sentiment_std:.3f}")
print(f"Most common sentiment: {most_common_sentiment}")

# Word frequency analysis (top 20 words)
all_words = ' '.join(df['cleaned_text']).split()
word_freq = Counter(all_words)
top_words = word_freq.most_common(20)

print(f"\nTop 20 Most Frequent Words:")
for i, (word, freq) in enumerate(top_words[:10], 1):
    print(f"{i:2d}. {word:15s} ({freq} occurrences)")
for i, (word, freq) in enumerate(top_words[10:], 11):
    print(f"{i:2d}. {word:15s} ({freq} occurrences)")

# Final assessment
print(f"\nCONCLUSION:")
if avg_sentiment > 0.1:
    sentiment_profile = "overall positive"
elif avg_sentiment < -0.1:
    sentiment_profile = "overall negative"
else:
    sentiment_profile = "relatively neutral"

if not df_tm.empty and 'dominant_topic' in df_tm.columns:
    topic_diversity = len(df_tm['dominant_topic'].unique())
    if topic_diversity >= 4:
        thematic_diversity = "highly diverse"
    elif topic_diversity >= 3:
        thematic_diversity = "moderately diverse"
    else:
        thematic_diversity = "focused"
else:
    thematic_diversity = "insufficient data for assessment"

print(f"The dataset shows {sentiment_profile} sentiment with {thematic_diversity} thematic content.")
print(f"Text analysis completed successfully.")

# Calculate and display runtime
end_time = time.time()
total_seconds = end_time - start_time
minutes = int(total_seconds // 60)
seconds = int(total_seconds % 60)

print("\n" + "="*80)
print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds")
print("="*80)