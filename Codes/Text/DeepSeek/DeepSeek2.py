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

# Sentiment analysis alternatives
from textblob import TextBlob

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Start timing
start_time = time.time()

print("=" * 80)
print("COMPREHENSIVE TEXT ANALYSIS PIPELINE")
print("=" * 80)

# Section 1: Text Data Loading and Overview
print("\n" + "="*50)
print("1. TEXT DATA LOADING AND OVERVIEW")
print("="*50)

# Load dataset
df = pd.read_csv('SemEval2017-task4-dev.subtask-A.english.INPUT.csv')

# Basic dataset info
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data Types:\n{df.dtypes}")
print(f"Missing Values:\n{df.isnull().sum()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")

# Detect text column
text_column = None
for col in df.columns:
    if df[col].dtype == 'object' and col.lower() in ['text', 'tweet', 'message', 'content', 'document']:
        text_column = col
        break
if text_column is None:
    # Use the first string column as text
    for col in df.columns:
        if df[col].dtype == 'object':
            text_column = col
            break

print(f"Using text column: '{text_column}'")

# Check if we have actual text data
non_empty_texts = df[text_column].notna().sum()
print(f"Non-empty text entries: {non_empty_texts}/{len(df)} ({non_empty_texts/len(df)*100:.1f}%)")

# If too many missing values, try to find alternative text column
if non_empty_texts < len(df) * 0.1:  # Less than 10% non-empty
    print("Warning: Primary text column has too many missing values.")
    # Look for other potential text columns
    for col in df.columns:
        if col != text_column and df[col].dtype == 'object':
            non_empty_count = df[col].notna().sum()
            if non_empty_count > non_empty_texts:
                text_column = col
                non_empty_texts = non_empty_count
                print(f"Switching to column '{text_column}' with {non_empty_texts} non-empty entries")
                break

# Display sample rows (only non-empty)
non_empty_samples = df[df[text_column].notna()][text_column].head(10)
print("\nSample Text Entries (non-empty):")
if len(non_empty_samples) > 0:
    for i, text in enumerate(non_empty_samples):
        print(f"{i+1}. {str(text)[:100]}...")
else:
    print("No non-empty text entries found!")

# Text statistics (only for non-empty texts)
non_empty_df = df[df[text_column].notna()].copy()
if len(non_empty_df) > 0:
    non_empty_df['text_length_chars'] = non_empty_df[text_column].astype(str).apply(len)
    non_empty_df['text_length_words'] = non_empty_df[text_column].astype(str).apply(lambda x: len(str(x).split()))
    
    print(f"\nText Statistics (non-empty entries only):")
    print(f"Total non-empty text entries: {len(non_empty_df)}")
    print(f"Average characters per text: {non_empty_df['text_length_chars'].mean():.2f}")
    print(f"Average words per text: {non_empty_df['text_length_words'].mean():.2f}")
    print(f"Shortest text: {non_empty_df['text_length_words'].min()} words")
    print(f"Longest text: {non_empty_df['text_length_words'].max()} words")
    
    # Text length distribution plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(non_empty_df['text_length_words'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Text Length (Words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Lengths (Words)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(non_empty_df['text_length_chars'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Text Length (Characters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Lengths (Characters)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    print("\nNo text data available for analysis!")
    # Create empty columns to avoid errors
    non_empty_df = pd.DataFrame()
    df['cleaned_text'] = ""
    df['sentiment_score'] = 0.0
    df['sentiment_label'] = 'neutral'

# Section 2: Text Preprocessing
print("\n" + "="*50)
print("2. TEXT PREPROCESSING")
print("="*50)

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == 'nan' or text == '':
        return ""
    
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Simple tokenization (avoid NLTK issues)
    tokens = text.split()
    # Remove stopwords and short tokens, lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

# Apply preprocessing only to non-empty texts
if len(non_empty_df) > 0:
    df['cleaned_text'] = df[text_column].apply(preprocess_text)
    
    print("Text preprocessing completed.")
    # Show samples of original vs cleaned text
    non_empty_cleaned = df[df['cleaned_text'] != '']
    if len(non_empty_cleaned) > 0:
        print(f"Sample original text: {df[text_column].iloc[0][:100]}...")
        print(f"Sample cleaned text: {df['cleaned_text'].iloc[0][:100]}...")
    else:
        print("No valid text after preprocessing.")
else:
    print("Skipping preprocessing - no text data available.")

# Section 3: Sentiment Analysis
print("\n" + "="*50)
print("3. SENTIMENT ANALYSIS")
print("="*50)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def get_sentiment_vader(text):
    """Get sentiment using VADER"""
    if not text or text == "":
        return 0.0
    scores = sia.polarity_scores(text)
    return scores['compound']

def get_sentiment_label(score):
    """Convert sentiment score to label"""
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Calculate sentiment scores
df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment_vader)
df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)

# Filter for meaningful sentiment analysis (non-empty cleaned texts)
meaningful_sentiment = df[df['cleaned_text'] != '']

if len(meaningful_sentiment) > 0:
    # Sentiment statistics
    sentiment_counts = meaningful_sentiment['sentiment_label'].value_counts()
    print("\nSentiment Distribution (non-empty texts):")
    print(sentiment_counts)
    print(f"\nOverall Sentiment Statistics:")
    print(f"Average Sentiment Score: {meaningful_sentiment['sentiment_score'].mean():.3f}")
    print(f"Sentiment Score Std: {meaningful_sentiment['sentiment_score'].std():.3f}")
    
    # Display sample sentiment results
    print("\nSample Sentiment Analysis Results:")
    sample_results = meaningful_sentiment[['cleaned_text', 'sentiment_score', 'sentiment_label']].head(10)
    for idx, row in sample_results.iterrows():
        print(f"Text: {row['cleaned_text'][:80]}...")
        print(f"Score: {row['sentiment_score']:.3f}, Label: {row['sentiment_label']}")
        print("-" * 50)
    
    # Sentiment visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sentiment_counts.plot(kind='bar', color=['lightcoral', 'lightblue', 'lightgreen'])
    plt.title('Sentiment Label Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.hist(meaningful_sentiment['sentiment_score'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    sentiment_pie = sentiment_counts.plot(kind='pie', autopct='%1.1f%%', 
                                         colors=['lightcoral', 'lightblue', 'lightgreen'])
    plt.title('Sentiment Distribution (Pie Chart)')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()
else:
    print("No meaningful text data available for sentiment analysis.")

# Section 4: Topic Modeling
print("\n" + "="*50)
print("4. TOPIC MODELING")
print("="*50)

# Prepare text for topic modeling (only meaningful texts)
meaningful_texts = df[df['cleaned_text'] != '']['cleaned_text']

if len(meaningful_texts) > 10:  # Need sufficient data for topic modeling
    texts_for_tm = [text for text in meaningful_texts if len(text.split()) > 3]
    
    if len(texts_for_tm) > 0:
        # Create document-term matrix
        vectorizer = CountVectorizer(max_features=500, stop_words='english', 
                                    min_df=2, max_df=0.8)
        dtm = vectorizer.fit_transform(texts_for_tm)
        feature_names = vectorizer.get_feature_names_out()
        
        # Determine optimal number of topics (simple heuristic)
        n_topics = min(5, max(2, len(texts_for_tm) // 20))
        print(f"Using {n_topics} topics for modeling")
        
        # Apply LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                       max_iter=10, learning_method='online')
        lda.fit(dtm)
        
        # Get topic distributions for documents
        topic_distributions = lda.transform(dtm)
        
        # Assign topics back to original dataframe
        topic_assignments = []
        topic_idx = 0
        for idx, row in df.iterrows():
            if row['cleaned_text'] != '' and len(row['cleaned_text'].split()) > 3:
                if topic_idx < len(topic_distributions):
                    dominant_topic = topic_distributions[topic_idx].argmax()
                    topic_assignments.append(dominant_topic)
                    topic_idx += 1
                else:
                    topic_assignments.append(-1)
            else:
                topic_assignments.append(-1)
        
        df['dominant_topic'] = topic_assignments
        
        # Display top words for each topic
        print("\nTop Keywords per Topic:")
        n_top_words = 8
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"Topic {topic_idx}: {', '.join(top_words)}")
        
        # Topic distribution (excluding -1 for no topic)
        valid_topics = df[df['dominant_topic'] != -1]['dominant_topic']
        if len(valid_topics) > 0:
            topic_counts = valid_topics.value_counts().sort_index()
            print(f"\nTopic Distribution:")
            print(topic_counts)
            
            # Display sample texts for each topic
            print("\nSample Texts per Topic:")
            for topic_id in range(n_topics):
                topic_texts = df[df['dominant_topic'] == topic_id]['cleaned_text'].head(2)
                if len(topic_texts) > 0:
                    print(f"\nTopic {topic_id} Sample Texts:")
                    for i, text in enumerate(topic_texts):
                        print(f"  {i+1}. {text[:80]}...")
            
            # Topic visualization
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            topic_counts.plot(kind='bar', color=plt.cm.Set3(range(n_topics)))
            plt.title('Topic Frequency Distribution')
            plt.xlabel('Topic ID')
            plt.ylabel('Number of Documents')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            # Topic-sentiment relationship
            topic_sentiment_df = df[df['dominant_topic'] != -1]
            if len(topic_sentiment_df) > 0:
                topic_sentiment = topic_sentiment_df.groupby('dominant_topic')['sentiment_score'].mean()
                topic_sentiment.plot(kind='bar', color=plt.cm.coolwarm(range(n_topics)))
                plt.title('Average Sentiment by Topic')
                plt.xlabel('Topic ID')
                plt.ylabel('Average Sentiment Score')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
        else:
            print("No valid topics assigned.")
    
    else:
        print("Insufficient data for topic modeling after filtering short texts")
        df['dominant_topic'] = -1
else:
    print("Insufficient data for topic modeling (need at least 10 meaningful texts)")
    df['dominant_topic'] = -1

# Section 5: Key Phrase Extraction
print("\n" + "="*50)
print("5. KEY PHRASE EXTRACTION")
print("="*50)

# Extract most common phrases from meaningful texts
meaningful_words = ' '.join(df[df['cleaned_text'] != '']['cleaned_text']).split()

if len(meaningful_words) > 0:
    word_freq = Counter(meaningful_words)
    common_words = word_freq.most_common(15)
    
    print("Most Common Words:")
    for word, freq in common_words:
        print(f"  {word}: {freq}")
    
    # Bigram analysis
    def extract_ngrams(text, n=2):
        tokens = text.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [' '.join(ngram) for ngram in ngrams]
    
    # Get bigrams
    all_bigrams = []
    for text in df[df['cleaned_text'] != '']['cleaned_text']:
        all_bigrams.extend(extract_ngrams(text, 2))
    
    if len(all_bigrams) > 0:
        bigram_freq = Counter(all_bigrams)
        common_bigrams = bigram_freq.most_common(10)
        
        print("\nMost Common Bigrams:")
        for bigram, freq in common_bigrams:
            print(f"  {bigram}: {freq}")
    else:
        print("\nNo bigrams found.")
else:
    print("No meaningful words for phrase extraction.")

# Section 6: Comprehensive Summary
print("\n" + "="*50)
print("6. COMPREHENSIVE SUMMARY")
print("="*50)

# Dataset summary
total_docs = len(df)
non_empty_docs = len(df[df['cleaned_text'] != ''])
meaningful_docs = len(df[(df['cleaned_text'] != '') & (df['cleaned_text'].str.len() > 10)])

print(f"Dataset Summary:")
print(f"- Total documents: {total_docs}")
print(f"- Non-empty documents: {non_empty_docs} ({non_empty_docs/total_docs*100:.1f}%)")
print(f"- Meaningful documents (>10 chars): {meaningful_docs} ({meaningful_docs/total_docs*100:.1f}%)")

if meaningful_docs > 0:
    # Sentiment summary
    meaningful_sentiment_df = df[df['cleaned_text'] != '']
    if len(meaningful_sentiment_df) > 0:
        sentiment_summary = meaningful_sentiment_df['sentiment_label'].value_counts()
        if len(sentiment_summary) > 0:
            dominant_sentiment = sentiment_summary.idxmax()
            sentiment_ratio = sentiment_summary.max() / len(meaningful_sentiment_df)
            
            print(f"\nSentiment Profile:")
            print(f"- Dominant sentiment: {dominant_sentiment} ({sentiment_ratio:.1%})")
            print(f"- Overall sentiment score: {meaningful_sentiment_df['sentiment_score'].mean():.3f}")
    
    # Topic modeling summary (if applicable)
    valid_topics_df = df[df['dominant_topic'] != -1]
    if len(valid_topics_df) > 0:
        topic_diversity = len(valid_topics_df['dominant_topic'].unique())
        print(f"\nThematic Analysis:")
        print(f"- Number of identified topics: {topic_diversity}")
        diversity_rating = 'good' if topic_diversity > 2 else 'limited'
        print(f"- Topic distribution shows {diversity_rating} thematic diversity")
    
    # Key insights
    print(f"\nKey Insights:")
    if len(meaningful_words) > 0:
        top_word, top_freq = common_words[0] if 'common_words' in locals() and common_words else ('N/A', 0)
        print(f"- Most frequent word: '{top_word}' (appears {top_freq} times)")
    
    print(f"- Dataset contains {non_empty_docs} analyzable text entries")
    if len(meaningful_sentiment_df) > 0:
        print(f"- Sentiment analysis reveals a {dominant_sentiment} overall tone")
else:
    print("\nDataset contains insufficient meaningful text data for comprehensive analysis.")
    print("Please check the input data format and ensure text columns contain actual content.")

# End timing and display runtime
end_time = time.time()
total_seconds = end_time - start_time
minutes = int(total_seconds // 60)
seconds = int(total_seconds % 60)

print("\n" + "="*80)
print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds")
print("="*80)

print("\nAnalysis complete! All visualizations and summaries generated.")