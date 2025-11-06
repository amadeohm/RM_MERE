#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated NLP Text Analysis Script
----------------------------------
Requirements satisfied:
- pandas, numpy, nltk, sklearn, matplotlib, seaborn, gensim, (vader via nltk)
- Complete end-to-end: load -> overview -> sentiment -> topic modeling -> summaries -> visuals
- Clear section titles and print statements
- Plots shown with plt.show()
- Automatic text column detection if not specified
- Runtime measurement printed at the end
"""

import os
import re
import sys
import time
import math
import glob
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

# ============ TIMER START ============
t0 = time.time()

# ============ IMPORTS ============
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize

try:
    from gensim import corpora
    from gensim.models import LdaModel, Phrases
    from gensim.models.phrases import Phraser
except ImportError as exc:
    # Some SciPy builds omit linalg.triu; fall back to NumPy implementation if needed.
    if "scipy.linalg" in str(exc) and "triu" in str(exc):
        import numpy as _np
        import scipy.linalg as _sla

        if not hasattr(_sla, "triu"):
            _sla.triu = _np.triu  # type: ignore[attr-defined]

        from gensim import corpora  # retry after patching
        from gensim.models import LdaModel, Phrases
        from gensim.models.phrases import Phraser
    else:
        raise

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Try optional visualization (pyLDAvis)
_has_pyldavis = False
try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    _has_pyldavis = True
except Exception:
    _has_pyldavis = False

# ============ NLTK DATA ============
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# ============ CONFIG ============
sns.set(style="whitegrid")
np.random.seed(42)

# ============ HELPER FUNCTIONS ============
def find_dataset_path():
    """
    Determine dataset path. Priority:
    1) CLI arg
    2) Common filenames in cwd
    3) Any CSV in cwd
    4) Known example path (if exists)
    """
    # 1) CLI
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]

    # 2) Common names
    candidates = [
        "data.csv",
        "dataset.csv",
        "text.csv",
        "texts.csv",
        "input.csv",
        "reviews.csv",
        "comments.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # 3) Any CSV in cwd
    csvs = sorted(glob.glob("*.csv"))
    if csvs:
        return csvs[0]

    # 4) Known uploaded path (if present)
    default_uploaded = "/mnt/data/SemEval2017-task4-dev.subtask-A.english.INPUT.csv"
    if os.path.exists(default_uploaded):
        return default_uploaded

    raise FileNotFoundError("No dataset found. Place a CSV in the working directory or pass a file path as the first argument.")


def robust_read_csv(path):
    """
    Try multiple encodings and separators.
    """
    try:
        df = pd.read_csv(path, engine="python")
        return df
    except Exception:
        pass
    # Try latin-1
    try:
        df = pd.read_csv(path, engine="python", encoding="latin-1")
        return df
    except Exception:
        pass
    # Try tab
    try:
        df = pd.read_csv(path, engine="python", sep="\t")
        return df
    except Exception:
        pass
    # Try semicolon
    try:
        df = pd.read_csv(path, engine="python", sep=";")
        return df
    except Exception as e:
        raise e


def detect_text_column(df, preferred="text"):
    """
    Detect text column:
    - If 'preferred' exists, use it.
    - Else choose the object/string column with the highest avg char length.
    - If none, try any column convertible to str with long avg length.
    """
    if preferred in df.columns:
        return preferred

    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        # Fall back to any column
        candidates = df.columns.tolist()
    else:
        candidates = obj_cols

    best_col, best_len = None, -1
    for c in candidates:
        try:
            series = df[c].astype(str)
            avg_len = series.str.len().mean()
            # Heuristic: text columns typically have non-trivial length
            if avg_len is not None and avg_len > best_len:
                best_col, best_len = c, avg_len
        except Exception:
            continue
    return best_col


def clean_text_basic(s):
    s = str(s)
    s = re.sub(r"http\S+|www\.\S+", " ", s)  # URLs
    s = re.sub(r"@[A-Za-z0-9_]+", " ", s)    # mentions
    s = re.sub(r"#", " ", s)                 # hashtags symbol
    s = re.sub(r"\d+", " ", s)               # numbers
    s = re.sub(r"[^\w\s']", " ", s)          # punctuation except apostrophes
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def preprocess_for_topic_modeling(texts, enable_bigrams=True):
    # Stopwords union (nltk + sklearn)
    stop_set = set(stopwords.words("english")) | set(ENGLISH_STOP_WORDS)
    lemmatizer = WordNetLemmatizer()

    tokenized = []
    for t in texts:
        t = clean_text_basic(t)
        toks = [w for w in word_tokenize(t) if len(w) > 2 and w.isalpha()]
        toks = [w for w in toks if w not in stop_set]
        toks = [lemmatizer.lemmatize(w) for w in toks]
        tokenized.append(toks)

    if enable_bigrams:
        phrases = Phrases(tokenized, min_count=5, threshold=10.0)
        bigram = Phraser(phrases)
        tokenized = [bigram[doc] for doc in tokenized]

    return tokenized


def choose_num_topics(n_docs):
    # heuristic: sqrt(n_docs), bounded between 2 and 10
    k = int(round(math.sqrt(max(n_docs, 1))))
    return max(2, min(10, k))


def label_from_compound(compound):
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


# ============ 1) TEXT DATA LOADING AND OVERVIEW ============
print_section("1) TEXT DATA LOADING AND OVERVIEW")

data_path = find_dataset_path()
print(f"Loading dataset from: {data_path}")
df = robust_read_csv(data_path)

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(list(df.columns))

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values per Column ---")
print(df.isna().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

print("\n--- Sample Rows (head) ---")
print(df.head(5))

text_col = detect_text_column(df, preferred="text")
if text_col is None:
    raise ValueError("Could not detect a text column. Ensure your dataset has a 'text' column or a textual field.")

print(f"\nDetected text column: '{text_col}'")

texts = df[text_col].fillna("").astype(str)
df = df.copy()
df["__text__"] = texts  # internal working copy

# Basic length stats
df["__n_chars__"] = df["__text__"].str.len()
df["__n_words__"] = df["__text__"].str.split().apply(len)

print("\n--- Text Entry Counts & Lengths ---")
print(f"Number of text entries: {df['__text__'].shape[0]}")
print(f"Average length (characters): {df['__n_chars__'].mean():.2f}")
print(f"Average length (words): {df['__n_words__'].mean():.2f}")

# Histogram of text lengths (by words)
plt.figure(figsize=(8, 5))
sns.histplot(df["__n_words__"], bins=40, kde=True)
plt.title("Distribution of Text Lengths (Words)")
plt.xlabel("Number of Words")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============ 2) REQUIREMENTS ============
print_section("2) SENTIMENT ANALYSIS, TOPIC MODELING & SUMMARIES")

# ---- Sentiment Analysis ----
print("\n[Sentiment Analysis] Computing VADER sentiment scores for each text entry...")
sia = SentimentIntensityAnalyzer()
sent_scores = df["__text__"].apply(sia.polarity_scores)
df["sent_neg"] = sent_scores.apply(lambda d: d["neg"])
df["sent_neu"] = sent_scores.apply(lambda d: d["neu"])
df["sent_pos"] = sent_scores.apply(lambda d: d["pos"])
df["sent_compound"] = sent_scores.apply(lambda d: d["compound"])
df["sent_label"] = df["sent_compound"].apply(label_from_compound)

print("\n--- Sentiment Distribution (labels) ---")
label_counts = df["sent_label"].value_counts(dropna=False)
print(label_counts)
print("\n--- Sentiment Score Summary (compound) ---")
print(df["sent_compound"].describe())

print("\n--- Sample Sentiment Results ---")
print(df[[text_col, "sent_label", "sent_compound"]].head(10))

# Visualizations: label distribution
plt.figure(figsize=(7, 5))
sns.countplot(x="sent_label", data=df, order=["negative", "neutral", "positive"])
plt.title("Sentiment Label Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Visualizations: histogram of compound scores
plt.figure(figsize=(8, 5))
sns.histplot(df["sent_compound"], bins=40, kde=True)
plt.title("Distribution of Sentiment Compound Scores")
plt.xlabel("Compound Score")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ---- Topic Modeling ----
print("\n[Topic Modeling] Preprocessing texts for LDA...")
preprocessed_docs = preprocess_for_topic_modeling(df["__text__"].tolist(), enable_bigrams=True)

# Filter out any empty docs for modeling, keep mapping
valid_idx = [i for i, doc in enumerate(preprocessed_docs) if len(doc) > 0]
if len(valid_idx) < 2:
    print("\nNot enough valid documents for topic modeling after preprocessing. Skipping LDA.")
    df["topic_id"] = np.nan
    df["topic_confidence"] = np.nan
    topic_freq = {}
else:
    filtered_docs = [preprocessed_docs[i] for i in valid_idx]
    id_map = {new_i: orig_i for new_i, orig_i in enumerate(valid_idx)}

    dictionary = corpora.Dictionary(filtered_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000)
    corpus = [dictionary.doc2bow(doc) for doc in filtered_docs]

    num_topics = choose_num_topics(len(filtered_docs))
    print(f"\nTraining LDA with num_topics={num_topics} on {len(filtered_docs)} documents...")
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        chunksize=2000,
        passes=10,
        alpha="auto",
        eta="auto",
        minimum_probability=0.0,
    )

    # Top keywords per topic
    print("\n--- Top Keywords per Topic ---")
    topic_keywords = {}
    for k in range(num_topics):
        terms = lda.show_topic(k, topn=10)
        topic_keywords[k] = [w for w, p in terms]
        print(f"Topic {k}: {', '.join(topic_keywords[k])}")

    # Assign each text to most likely topic
    topic_assignments = [None] * len(df)
    topic_confidences = [np.nan] * len(df)
    for new_i, bow in enumerate(corpus):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        if dist:
            best_topic, best_prob = max(dist, key=lambda x: x[1])
            orig_i = id_map[new_i]
            topic_assignments[orig_i] = best_topic
            topic_confidences[orig_i] = best_prob
    df["topic_id"] = topic_assignments
    df["topic_confidence"] = topic_confidences

    # Topic distribution
    print("\n--- Topic Distribution (counts) ---")
    topic_counts = pd.Series([t for t in df["topic_id"] if t is not None]).value_counts().sort_index()
    print(topic_counts)

    # Sample texts per topic
    print("\n--- Sample Texts per Topic ---")
    for k in range(num_topics):
        samples = (
            df.loc[df["topic_id"] == k, "__text__"]
            .head(3)
            .astype(str)
            .tolist()
        )
        print(f"\nTopic {k} samples:")
        for s in samples:
            snippet = s.replace("\n", " ").strip()
            snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
            if not snippet:
                snippet = "[empty text]"
            print(f"  - {snippet}")

    # Visualization: topic frequencies
    plt.figure(figsize=(8, 5))
    freq_df = pd.DataFrame({
        "topic_id": list(topic_counts.index),
        "count": list(topic_counts.values)
    }).sort_values("topic_id")
    sns.barplot(x="topic_id", y="count", data=freq_df)
    plt.title("Topic Frequency (LDA)")
    plt.xlabel("Topic ID")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Optional: pyLDAvis visualization
    if _has_pyldavis:
        try:
            print("\n[Optional] Preparing pyLDAvis visualization...")
            vis = gensimvis.prepare(lda, corpus, dictionary)
            # Try to display in notebook; otherwise, save to HTML.
            try:
                get_ipython  # noqa
                pyLDAvis.display(vis)
                print("pyLDAvis displayed inline.")
            except Exception:
                out_html = "lda_vis.html"
                pyLDAvis.save_html(vis, out_html)
                print(f"pyLDAvis saved to: {os.path.abspath(out_html)}")
        except Exception as e:
            print(f"pyLDAvis visualization skipped due to error: {e}")

# ---- Summaries ----
print("\n--- Summary: Overall Sentiment Profile & Thematic Diversity ---")
# Sentiment summary
total = max(int(label_counts.sum()), 1)
neg = int(label_counts.get("negative", 0))
neu = int(label_counts.get("neutral", 0))
pos = int(label_counts.get("positive", 0))
neg_pct = 100.0 * neg / total
neu_pct = 100.0 * neu / total
pos_pct = 100.0 * pos / total

sent_overall = "mixed/neutral"
if pos_pct > max(neg_pct, neu_pct):
    sent_overall = "overall positive"
elif neg_pct > max(pos_pct, neu_pct):
    sent_overall = "overall negative"

print(f"Sentiment profile: {sent_overall} "
      f"(positive={pos_pct:.1f}%, neutral={neu_pct:.1f}%, negative={neg_pct:.1f}%).")

# Thematic diversity summary
if "topic_id" in df.columns and df["topic_id"].notna().any():
    unique_topics = int(pd.Series([t for t in df["topic_id"] if pd.notna(t)]).nunique())
    print(f"Thematic diversity: {unique_topics} topic(s) discovered via LDA.")
else:
    print("Thematic diversity: Topic modeling not available (insufficient valid documents).")

# ============ 3) TIME MEASUREMENT ============
print_section("3) TIME MEASUREMENT")
elapsed = time.time() - t0
mins = int(elapsed // 60)
secs = int(round(elapsed % 60))
print(f"Total Simulated Runtime: {mins} minutes and {secs} seconds")

# ============ 4) CLEAN OUTPUT / FINAL NOTES ============
print_section("Analysis Complete")
print("All steps executed: Loading, Overview, Sentiment, Topic Modeling, Visualizations, Summary, and Runtime.")
