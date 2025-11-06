#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated NLP Text Analysis Pipeline
Requirements covered:
- pandas, numpy, nltk, sklearn, matplotlib, seaborn, gensim, and vaderSentiment (alternative to transformers)
- Complete end-to-end automated analysis on a dataset with a "text" column (or auto-detected)
- Clear sections with print statements and inline plots
- Sentiment analysis, topic modeling, summaries, visualizations
- Total runtime measurement
"""

# =========================
# Imports
# =========================
import os
import re
import sys
import time
import math
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK + VADER (sentiment)
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Sentiment (VADER)
from nltk.sentiment import SentimentIntensityAnalyzer

# Topic Modeling (gensim)
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Scikit-learn utilities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Optional: pyLDAvis
try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    PYLDA_AVAILABLE = True
except Exception:
    PYLDA_AVAILABLE = False

# Optional: transformers pipeline (not required; we use VADER by default)
try:
    import transformers  # noqa: F401
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


# =========================
# Helper Functions
# =========================
def safe_nltk_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)

def detect_text_column(df: pd.DataFrame) -> str:
    """Detect a suitable text column if 'text' not present."""
    # Priority 1: exact match 'text'
    if 'text' in df.columns:
        return 'text'
    # Priority 2: object/string columns with largest average length
    candidate_cols = [c for c in df.columns if df[c].dtype == 'object']
    if not candidate_cols:
        # Try columns with potentially mixed types but many strings
        for c in df.columns:
            if df[c].apply(lambda x: isinstance(x, str)).mean() > 0.5:
                candidate_cols.append(c)
    if candidate_cols:
        avg_lengths = []
        for c in candidate_cols:
            lengths = df[c].dropna().astype(str).apply(lambda s: len(s))
            avg_lengths.append(lengths.mean() if not lengths.empty else 0)
        best_idx = int(np.argmax(avg_lengths)) if avg_lengths else 0
        return candidate_cols[best_idx]
    # Fallback: first column
    return df.columns[0]

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts."""
    tag = tag[0].upper()
    mapping = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return mapping.get(tag, wordnet.NOUN)

def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"[\d]+", " ", text)                     # remove digits
    text = re.sub(r"[^\w\s'-]", " ", text)                 # remove punctuation except ' and -
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_lemmatize(texts, extra_stopwords=None):
    safe_nltk_download('tokenizers/punkt')
    safe_nltk_download('corpora/wordnet')
    safe_nltk_download('taggers/averaged_perceptron_tagger')
    lemmatizer = WordNetLemmatizer()
    sw = set(stopwords.words('english')) | set(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        sw |= set(extra_stopwords)
    processed_tokens = []
    for t in texts:
        t = basic_clean(t)
        tokens = word_tokenize(t)
        # POS tagging
        pos_tags = nltk.pos_tag(tokens)
        lemmas = []
        for tok, pos in pos_tags:
            if tok in {"'", "-", "--"}:
                continue
            if tok in sw:
                continue
            if len(tok) <= 2 and tok not in {"ai", "ml", "nlp"}:
                continue
            lemma = WordNetLemmatizer().lemmatize(tok, get_wordnet_pos(pos))
            lemmas.append(lemma)
        processed_tokens.append(lemmas)
    return processed_tokens

def compute_text_lengths(text_series: pd.Series):
    texts = text_series.fillna("").astype(str)
    char_lengths = texts.apply(len)
    word_lengths = texts.apply(lambda s: len(s.split()))
    return word_lengths, char_lengths

def vader_sentiment(texts):
    safe_nltk_download('sentiment/vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    scores = []
    labels = []
    for t in texts:
        s = sia.polarity_scores(t)
        comp = s['compound']
        if comp >= 0.05:
            lab = 'positive'
        elif comp <= -0.05:
            lab = 'negative'
        else:
            lab = 'neutral'
        scores.append(comp)
        labels.append(lab)
    return np.array(scores), np.array(labels)

def plot_histogram(data, title, xlabel, bins=30):
    plt.figure(figsize=(8, 4.5))
    sns.histplot(data, bins=bins, kde=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()

if __name__ == "__main__":
    start = time.time()

    # 1. Cargar dataset (ajusta la ruta)
    df = pd.read_csv("/Users/amadeo/Documents/Code/SemEval2017-task4-dev.subtask-A.english.INPUT.csv")  # <-- pon aquí tu dataset
    text_col = detect_text_column(df)
    texts = df[text_col].fillna("").astype(str).tolist()

    print(f"[INFO] Usando la columna de texto: {text_col}")
    print(f"[INFO] Número de textos: {len(texts)}")

    # 2. Análisis rápido: longitudes
    word_lens, char_lens = compute_text_lengths(df[text_col])
    print(f"[INFO] Longitud media (palabras): {word_lens.mean():.2f}")
    print(f"[INFO] Longitud media (caracteres): {char_lens.mean():.2f}")

    # 3. Sentimiento con VADER
    scores, labels = vader_sentiment(texts)
    print("[INFO] Sentimiento (primeros 5):")
    for t, s, l in zip(texts[:5], scores[:5], labels[:5]):
        print(f"  - {l} ({s:.3f}): {t[:80]}...")

    # 4. (Opcional) plot
    plot_histogram(scores, "Distribución de sentimiento", "compound score")
    plt.show()

    end = time.time()
    print(f"[INFO] Tiempo total: {end - start:.2f} s")
