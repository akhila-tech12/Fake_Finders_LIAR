"""
naive_bayes.py
==============
Naive Bayes text classifier for fake news detection.

Algorithm:
    Multinomial Naive Bayes with Laplace smoothing.
    Uses log-probabilities to prevent numerical underflow.

    For each document, computes:
        score(class) = log P(class) + sum(log P(word | class))
    Predicts the class with the highest score.

Why log probabilities?
    Multiplying many small probabilities (e.g. 0.001 * 0.002 * ...)
    causes floating point underflow → becomes 0.0.
    Adding log probabilities avoids this completely.

Author : Akhila Pavithran, Rajana
Project: Fake Finders — NLP Master Project, University of Bamberg
"""

import re
import math
import time
import os
import sys

# ── Make src importable regardless of working directory ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

from data_loader          import load_dataset, preview_dataset
from classification_evaluator import ClassificationEvaluator, print_report


# ══════════════════════════════════════════════════════════════════════════════
# Text Processing
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list:
    """
    Converts raw text into a list of clean lowercase tokens.

    Steps:
        1. Lowercase
        2. Keep only alphabetic characters and spaces
        3. Split on whitespace

    Args:
        text: Raw input string.

    Returns:
        List of lowercase word tokens.

    Example:
        tokenize("Hello, World!") → ["hello", "world"]
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def build_vocab(dataset: list) -> set:
    """
    Builds vocabulary set from a dataset.

    Args:
        dataset: List of (text, label) tuples.

    Returns:
        Set of all unique tokens seen in the dataset.
    """
    vocab = set()
    for text, _ in dataset:
        vocab.update(tokenize(text))
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def build_counts(dataset: list) -> tuple:
    """
    Counts word frequencies and document counts per class.

    Args:
        dataset: List of (text, label) tuples.

    Returns:
        Tuple of (total_words, class_counts, word_counts):
            total_words  : {label: total word count}
            class_counts : {label: document count}
            word_counts  : {label: {word: count}}
    """
    total_words  = {}
    class_counts = {}
    word_counts  = {}

    for text, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
            total_words[label]  = 0
            word_counts[label]  = {}

        class_counts[label] += 1

        for word in tokenize(text):
            total_words[label] += 1
            word_counts[label][word] = word_counts[label].get(word, 0) + 1

    return total_words, class_counts, word_counts


def compute_priors(class_counts: dict) -> dict:
    """
    Computes log prior probability for each class.

    P(class) = count(class) / total_documents
    Stored as log P(class) to avoid underflow.

    Args:
        class_counts: {label: document count}

    Returns:
        {label: log P(label)}
    """
    total = sum(class_counts.values())
    return {
        label: math.log(count / total)
        for label, count in class_counts.items()
    }


def compute_likelihoods(
    word_counts : dict,
    total_words : dict,
    vocab       : set
) -> dict:
    """
    Computes log P(word | class) with Laplace smoothing.

    Laplace smoothing prevents zero probabilities for unseen words:
        P(word | class) = (count(word, class) + 1)
                          / (total_words(class) + vocab_size)

    Stored as log probabilities to avoid underflow.

    Args:
        word_counts : {label: {word: count}}
        total_words : {label: total word count}
        vocab       : Full vocabulary set

    Returns:
        {label: {word: log P(word | label)}}
    """
    vocab_size  = len(vocab)
    likelihoods = {}

    for label in word_counts:
        likelihoods[label] = {}
        for word in vocab:
            count = word_counts[label].get(word, 0)
            likelihoods[label][word] = math.log(
                (count + 1) / (total_words[label] + vocab_size)
            )

    return likelihoods


def train(dataset: list) -> tuple:
    """
    Trains the Naive Bayes classifier on the given dataset.

    Args:
        dataset: List of (text, label) tuples.

    Returns:
        Tuple of (priors, likelihoods, vocab):
            priors      : {label: log P(label)}
            likelihoods : {label: {word: log P(word|label)}}
            vocab       : set of all known words
    """
    vocab                              = build_vocab(dataset)
    total_words, class_counts, word_counts = build_counts(dataset)
    priors                             = compute_priors(class_counts)
    likelihoods                        = compute_likelihoods(
                                             word_counts, total_words, vocab
                                         )
    return priors, likelihoods, vocab


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def score(
    text        : str,
    label       : int,
    priors      : dict,
    likelihoods : dict,
    vocab       : set
) -> float:
    """
    Computes log P(label | text) ∝ log P(label) + Σ log P(word | label).

    Unknown words (not in vocab) are silently ignored.

    Args:
        text        : Input document string.
        label       : Class label to score.
        priors      : Log prior probabilities.
        likelihoods : Log word likelihoods per class.
        vocab       : Known vocabulary set.

    Returns:
        Log probability score (higher = more likely).
    """
    result = priors[label]
    for word in tokenize(text):
        if word in vocab:
            result += likelihoods[label][word]
    return result


def predict(
    text        : str,
    priors      : dict,
    likelihoods : dict,
    vocab       : set
) -> int:
    """
    Predicts the most likely class label for a given text.

    Args:
        text        : Input document string.
        priors      : Log prior probabilities.
        likelihoods : Log word likelihoods per class.
        vocab       : Known vocabulary set.

    Returns:
        Predicted label (+1 for fake, -1 for real).
    """
    return max(
        priors.keys(),
        key=lambda label: score(text, label, priors, likelihoods, vocab)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Error Analysis
# ══════════════════════════════════════════════════════════════════════════════

def show_errors(
    test_data   : list,
    priors      : dict,
    likelihoods : dict,
    vocab       : set,
    limit       : int = 5
) -> None:
    """
    Displays misclassified examples for error analysis.

    Required for midterm presentation.
    Connects model errors back to model design decisions.

    Args:
        test_data   : List of (text, label) tuples.
        priors      : Trained prior probabilities.
        likelihoods : Trained likelihoods.
        vocab       : Known vocabulary.
        limit       : Max number of errors to display.
    """
    label_names = {+1: "FAKE", -1: "REAL"}
    shown       = 0

    print(f"\n{'=' * 55}")
    print("  MISCLASSIFIED EXAMPLES — ERROR ANALYSIS")
    print(f"{'=' * 55}")

    for text, y in test_data:
        yhat = predict(text, priors, likelihoods, vocab)
        if yhat != y:
            print(f"  TEXT : {text[:120]}")
            print(f"  TRUE : {label_names[y]}")
            print(f"  PRED : {label_names[yhat]}")
            print(f"  {'─' * 50}")
            shown += 1
            if shown >= limit:
                break

    if shown == 0:
        print("  No misclassified examples found.")

    print(f"{'=' * 55}")


def show_top_words(
    priors      : dict,
    likelihoods : dict,
    vocab       : set,
    n           : int = 15
) -> None:
    """
    Displays top n words most associated with each class.

    Computed as difference in log likelihoods:
        score(word) = log P(word | FAKE) - log P(word | REAL)

    Args:
        priors      : Trained prior probabilities.
        likelihoods : Trained likelihoods.
        vocab       : Known vocabulary.
        n           : Number of top words to show per class.
    """
    labels = list(priors.keys())
    if len(labels) != 2:
        return

    pos, neg = +1, -1

    word_scores = sorted(
        ((w, likelihoods[pos][w] - likelihoods[neg][w]) for w in vocab),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n  Top {n} words → FAKE news:")
    for word, sc in word_scores[:n]:
        print(f"    {word:25s} {sc:+.3f}")

    print(f"\n  Top {n} words → REAL news:")
    for word, sc in word_scores[-n:][::-1]:
        print(f"    {word:25s} {sc:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Helper
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    test_data   : list,
    priors      : dict,
    likelihoods : dict,
    vocab       : set
) -> tuple:
    """
    Runs predictions on test data and returns y_true, y_pred as 0/1.

    Args:
        test_data   : List of (text, label) tuples.
        priors      : Trained prior probabilities.
        likelihoods : Trained likelihoods.
        vocab       : Known vocabulary.

    Returns:
        Tuple of (y_true, y_pred) — both as 0/1 integer lists.
    """
    label_map      = {+1: 1, -1: 0}
    y_true, y_pred = [], []

    for text, y in test_data:
        y_true.append(label_map[y])
        y_pred.append(label_map[predict(text, priors, likelihoods, vocab)])

    return y_true, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Load data
    train_data, valid_data, test_data = load_dataset()
    preview_dataset(train_data, split_name="Train")

    # 2. Train
    print("\n=== Training Naive Bayes ===")
    start  = time.time()
    priors, likelihoods, vocab = train(train_data)
    elapsed = time.time() - start

    print(f"  Vocabulary size : {len(vocab):,}")
    print(f"  Training time   : {elapsed:.2f} sec")
    print(f"  Log priors      : { {k: round(v, 3) for k, v in priors.items()} }")

    # 3. Evaluate on validation set
    print("\n=== Validation Set ===")
    y_true, y_pred = evaluate(valid_data, priors, likelihoods, vocab)
    ev = ClassificationEvaluator(y_true, y_pred)
    print_report("Naive Bayes — Validation", ev.binary_report())

    # 4. Evaluate on test set
    print("\n=== Test Set ===")
    y_true, y_pred = evaluate(test_data, priors, likelihoods, vocab)
    ev = ClassificationEvaluator(y_true, y_pred)
    print_report("Naive Bayes — Test Results", ev.binary_report())

    # 5. Error analysis
    show_errors(test_data, priors, likelihoods, vocab, limit=5)

    # 6. Top words
    show_top_words(priors, likelihoods, vocab, n=15)

    # 7. Summary
    print(f"\n{'=' * 55}")
    print("  SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Model     : Naive Bayes (from scratch)")
    print(f"  Dataset   : LIAR (train.tsv / test.tsv)")
    print(f"  Task      : Binary fake vs real classification")
    print(f"  Train     : {len(train_data)} samples")
    print(f"  Test      : {len(test_data)} samples")
    print(f"  Vocab     : {len(vocab):,} words")
    print(f"  Time      : {elapsed:.2f} sec")
    print(f"{'=' * 55}")