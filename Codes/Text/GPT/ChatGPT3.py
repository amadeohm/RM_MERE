# -*- coding: utf-8 -*-
"""
Automated NLP & Text Analytics Pipeline
- Loads a dataset
- Explores text data
- Performs sentiment analysis (VADER)
- Runs topic modeling (Gensim LDA)
- Visualizes distributions
- Summarizes findings
- Measures total runtime

Requirements: pandas, numpy, nltk, sklearn, matplotlib, seaborn, gensim, (optional) pyLDAvis
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text as sklearn_text

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess

# ---------------------------- Utility: NLTK Downloads ---------------------------- #
def ensure_nltk_resources():
    resources = [
        ("vader_lexicon", "sentiment/vader_lexicon.zip"),
        ("stopwords", "corpora/stopwords.zip"),
        ("punkt", "tokenizers/punkt.zip")
    ]
    for res_name, _ in resources:
        try:
            nltk.data.find(_)
        except LookupError:
            nltk.download(res_name, quiet=True)

# ---------------------------- Utility: Detect Text Column ---------------------------- #
def detect_text_column(df: pd.DataFrame, preferred_name: str = "text") -> str:
    # If explicit 'text' column exists
    if preferred_name in df.columns:
        return preferred_name
    # Candidate object columns
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not object_cols:
        # Try string dtype
        string_cols = df.select_dtypes(include=["string"]).columns.tolist()
        object_cols = string_cols
    if not object_cols:
        raise ValueError(
            "No suitable text column found. Please ensure a 'text' column or provide a dataset with a string column."
        )
    # Heuristic: choose column with largest average character length among object columns
    avg_lengths = {}
    for col in object_cols:
        try:
            avg_lengths[col] = df[col].astype(str).str.len().mean()
        except Exception:
            avg_lengths[col] = -1
    best_col = max(avg_lengths.items(), key=lambda x: x[1])[0]
    return best_col

# ---------------------------- Utility: Clean & Tokenize ---------------------------- #
def tokenize_for_gensim(text_series, extra_stopwords=None, min_len=2, max_len=20):
    stop_words = set(stopwords.words("english"))
    if extra_stopwords:
        stop_words |= set([w.lower() for w in extra_stopwords])
    tokenized = []
    for t in text_series.astype(str).fillna(""):
        tokens = simple_preprocess(t, deacc=True, min_len=min_len, max_len=max_len)
        tokens = [w for w in tokens if w not in stop_words]
        tokenized.append(tokens)
    return tokenized

# ---------------------------- Utility: Topic Count Heuristic ---------------------------- #
def choose_num_topics(n_docs: int) -> int:
    # Heuristic: between 3 and 12 topics depending on corpus size
    if n_docs <= 100:
        return 5
    elif n_docs <= 500:
        return 8
    elif n_docs <= 2000:
        return 10
    else:
        return 12

# ---------------------------- Main Pipeline ---------------------------- #
def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Automated NLP & Text Analytics Pipeline")
    parser.add_argument("--file", type=str, default="data.csv",
                        help="Path to CSV dataset (default: data.csv)")
    parser.add_argument("--text_column", type=str, default=None,
                        help="Specify the text column name (optional)")
    parser.add_argument("--sep", type=str, default=None,
                        help="CSV separator (auto-detect if not provided)")
    args = parser.parse_args() if sys.argv[0].endswith(".py") or len(sys.argv) > 0 else argparse.Namespace(
        file="data.csv", text_column=None, sep=None
    )

    # If a well-known path is present alongside defaults, prioritize it
    # (helpful for provided datasets)
    default_candidates = [
        args.file,
        "SemEval2017-task4-dev.subtask-A.english.INPUT.csv"
    ]
    dataset_path = next((p for p in default_candidates if os.path.exists(p)), args.file)

    print("=" * 80)
    print("NLP & TEXT ANALYTICS PIPELINE")
    print("=" * 80)

    # ---------------------------- 1) Text Data Loading and Overview ---------------------------- #
    print("\n" + "#" * 80)
    print("1) TEXT DATA LOADING AND OVERVIEW")
    print("#" * 80)

    # Attempt to load with pandas; try common separators
    sep_candidates = [args.sep] if args.sep is not None else [",", "\t", ";", "|"]
    last_err = None
    df = None
    for sep in sep_candidates:
        try:
            df = pd.read_csv(dataset_path, sep=sep)
            break
        except Exception as e:
            last_err = e
            continue
    if df is None:
        print(f"Failed to read CSV with common separators. Last error: {last_err}")
        sys.exit(1)

    print(f"\nLoaded dataset from: {dataset_path}")
    print("\nDataset Shape:", df.shape)
    print("\nColumn Names:", list(df.columns))
    print("\nData Types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing Values per Column:")
    print(df.isna().sum())

    # Duplicate rows
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {dup_count}")

    # Sample rows
    print("\nSample Rows:")
    print(df.head(5))

    # Determine text column
    if args.text_column and args.text_column in df.columns:
        text_col = args.text_column
    else:
        text_col = detect_text_column(df)
    print(f"\nUsing text column: '{text_col}'")

    # Clean text column (ensure string)
    df[text_col] = df[text_col].astype(str)
    n_texts = df[text_col].notna().sum()
    print(f"\nNumber of text entries: {n_texts}")

    # Length stats
    df["__len_chars__"] = df[text_col].str.len()
    df["__len_words__"] = df[text_col].str.split().apply(len)
    avg_chars = df["__len_chars__"].mean()
    avg_words = df["__len_words__"].mean()
    print(f"Average length: {avg_words:.2f} words, {avg_chars:.2f} characters")

    # Histogram of text lengths (words)
    print("\nPlotting histogram of text lengths (in words)...")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["__len_words__"], bins=30, kde=False)
    plt.title("Distribution of Text Lengths (Words)")
    plt.xlabel("Words per Text")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # ---------------------------- 2) Requirements ---------------------------- #
    print("\n" + "#" * 80)
    print("2) SENTIMENT ANALYSIS & TOPIC MODELING")
    print("#" * 80)

    # Ensure NLTK resources
    ensure_nltk_resources()

    # ---------------- Sentiment Analysis ---------------- #
    print("\n[Sentiment Analysis] Running VADER sentiment analysis...")
    sia = SentimentIntensityAnalyzer()
    scores = df[text_col].apply(lambda x: sia.polarity_scores(x)["compound"])
    df["sentiment_score"] = scores

    def label_from_compound(c):
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment_label"] = df["sentiment_score"].apply(label_from_compound)

    # Distribution statistics
    print("\nSentiment Distribution (counts):")
    sent_counts = df["sentiment_label"].value_counts().sort_index()
    print(sent_counts)

    print("\nSentiment Distribution (percent):")
    sent_perc = (sent_counts / len(df) * 100).round(2)
    print(sent_perc.astype(str) + "%")

    print("\nSample Sentiment Results:")
    print(df[[text_col, "sentiment_score", "sentiment_label"]].head(10))

    # Visualization: bar chart of sentiment labels
    print("\nPlotting bar chart of sentiment label distribution...")
    plt.figure(figsize=(6, 4))
    sns.countplot(x="sentiment_label", data=df, order=["negative", "neutral", "positive"])
    plt.title("Sentiment Label Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Visualization: histogram of sentiment scores
    print("\nPlotting histogram of sentiment scores (compound)...")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["sentiment_score"], bins=30, kde=False)
    plt.title("Distribution of Sentiment Scores (Compound)")
    plt.xlabel("Compound Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # ---------------- Topic Modeling (Gensim LDA) ---------------- #
    print("\n[Topic Modeling] Preparing data for LDA...")
    # Use sklearn stop words extended + NLTK stopwords
    sklearn_sw = set(sklearn_text.ENGLISH_STOP_WORDS)
    try:
        nltk_sw = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk_sw = set(stopwords.words("english"))
    extra_sw = sklearn_sw | nltk_sw

    tokenized_docs = tokenize_for_gensim(df[text_col], extra_stopwords=extra_sw)
    # Remove empty docs after tokenization
    non_empty_mask = [len(toks) > 0 for toks in tokenized_docs]
    if not any(non_empty_mask):
        print("All documents are empty after tokenization; skipping topic modeling.")
        topic_modeling_done = False
    else:
        tokenized_docs_filtered = [t for t in tokenized_docs if len(t) > 0]
        idx_map = [i for i, keep in enumerate(non_empty_mask) if keep]

        dictionary = corpora.Dictionary(tokenized_docs_filtered)
        # Filter extremes to reduce noise
        dictionary.filter_extremes(no_below=max(2, int(0.005 * len(tokenized_docs_filtered))),
                                   no_above=0.5, keep_n=5000)

        corpus = [dictionary.doc2bow(text) for text in tokenized_docs_filtered]

        num_topics = choose_num_topics(len(tokenized_docs_filtered))
        print(f"Training LDA with {num_topics} topics on {len(tokenized_docs_filtered)} documents...")
        lda = LdaModel(corpus=corpus,
                       id2word=dictionary,
                       num_topics=num_topics,
                       random_state=42,
                       passes=10,
                       chunksize=200,
                       alpha="auto",
                       eta="auto",
                       per_word_topics=False)

        # Print top keywords per topic
        print("\nTop Keywords per Topic:")
        topic_keywords = {}
        for t in range(num_topics):
            terms = lda.show_topic(t, topn=10)
            keywords = [w for w, _ in terms]
            topic_keywords[t] = keywords
            print(f" - Topic {t}: {', '.join(keywords)}")

        # Assign dominant topic to each document
        print("\nAssigning dominant topic to each text entry...")
        dominant_topics = []
        topic_probs = []
        for bow in corpus:
            topics = lda.get_document_topics(bow, minimum_probability=0.0)
            if topics:
                t_idx, t_prob = max(topics, key=lambda x: x[1])
            else:
                t_idx, t_prob = (None, 0.0)
            dominant_topics.append(t_idx)
            topic_probs.append(t_prob)

        # Map back to the full dataframe
        df["topic_id"] = np.nan
        df["topic_prob"] = np.nan
        for mapped_row_idx, original_df_idx in enumerate(idx_map):
            df.at[original_df_idx, "topic_id"] = dominant_topics[mapped_row_idx]
            df.at[original_df_idx, "topic_prob"] = topic_probs[mapped_row_idx]

        # Topic distribution
        topic_counts = df["topic_id"].dropna().astype(int).value_counts().sort_index()
        print("\nTopic Distribution (counts):")
        for t in range(num_topics):
            count = int(topic_counts.get(t, 0))
            print(f"Topic {t}: {count}")

        # Sample texts per topic
        print("\nSample Texts per Topic:")
        for t in range(num_topics):
            print(f"\n--- Topic {t} | Keywords: {', '.join(topic_keywords[t])} ---")
            sample_rows = df[df["topic_id"] == t][text_col].head(3)
            for i, row in enumerate(sample_rows, 1):
                print(f"[{i}] {row[:300]}{'...' if len(row)>300 else ''}")

        # Visualization: bar plot of topic frequencies
        print("\nPlotting bar chart of topic frequencies...")
        plt.figure(figsize=(8, 5))
        topics_order = list(range(num_topics))
        counts_ordered = [int(topic_counts.get(t, 0)) for t in topics_order]
        sns.barplot(x=[f"T{t}" for t in topics_order], y=counts_ordered)
        plt.title("Topic Frequencies")
        plt.xlabel("Topic")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        # Optional visualization: pyLDAvis
        topic_modeling_done = True
        try:
            import pyLDAvis
            import pyLDAvis.gensim_models as gensimvis
            print("\nAttempting pyLDAvis visualization (if environment supports HTML rendering)...")
            vis_data = gensimvis.prepare(lda, corpus, dictionary)
            # In a pure script, we cannot render inline HTML easily.
            # Save to HTML for manual viewing.
            out_html = "pyLDAvis_topics.html"
            pyLDAvis.save_html(vis_data, out_html)
            print(f"pyLDAvis saved to: {os.path.abspath(out_html)}")
        except Exception as e:
            print(f"pyLDAvis not available or failed to render: {e}")

    # ---------------------------- Summary ---------------------------- #
    print("\n" + "#" * 80)
    print("SUMMARY")
    print("#" * 80)

    mean_score = df["sentiment_score"].mean()
    pos_ratio = (df["sentiment_label"] == "positive").mean()
    neu_ratio = (df["sentiment_label"] == "neutral").mean()
    neg_ratio = (df["sentiment_label"] == "negative").mean()

    if 'topic_id' in df.columns and df['topic_id'].notna().any():
        used_topics = int(df['topic_id'].nunique())
        nontrivial_topics = used_topics
    else:
        used_topics = 0
        nontrivial_topics = 0

    overall_sent = "positive" if mean_score >= 0.05 else ("negative" if mean_score <= -0.05 else "neutral")
    print(f"Overall Sentiment Profile: {overall_sent.upper()} "
          f"(mean compound score={mean_score:.3f}; "
          f"{pos_ratio*100:.1f}% positive, {neu_ratio*100:.1f}% neutral, {neg_ratio*100:.1f}% negative)")

    if nontrivial_topics > 0:
        print(f"Thematic Diversity: {nontrivial_topics} topics identified across the corpus.")
    else:
        print("Thematic Diversity: Topic modeling not available or insufficient signal in the data.")

    # ---------------------------- 3) Time Measurement ---------------------------- #
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "#" * 80)
    print("3) TIME MEASUREMENT")
    print("#" * 80)
    print(f"Total Simulated Runtime: {minutes} minutes and {seconds} seconds")

    # ---------------------------- (Optional) Save Results ---------------------------- #
    out_csv = "nlp_analysis_output.csv"
    save_cols = [c for c in [text_col, "sentiment_score", "sentiment_label", "topic_id", "topic_prob"] if c in df.columns]
    if save_cols:
        df[save_cols].to_csv(out_csv, index=False)
        print(f"\nResults saved to: {os.path.abspath(out_csv)}")

    print("\nPipeline completed successfully.")

# ---------------------------- Entry ---------------------------- #
if __name__ == "__main__":
    main()
