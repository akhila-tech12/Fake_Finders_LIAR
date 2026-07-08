"""
perceptron.py
=============
Perceptron classifier for fake news detection on the LIAR dataset.

Algorithm:
    Single-layer Perceptron with binary Bag-of-Words representation.
    Learns iteratively by correcting mistakes only.

    For each training example (x, y):
        score   = w · x + b
        y_hat   = sign(score)
        if y_hat != y:
            w = w + y * x
            b = b + y

    Stops early when zero errors in a full epoch (convergence).

Why Perceptron over Naive Bayes?
    Naive Bayes assumes word independence — words contribute
    independently. Perceptron learns a discriminative boundary
    directly from mistakes, giving each word a weight that
    reflects its importance for the final decision.

Limitations:
    - Only linearly separable problems guaranteed to converge.
    - Vocabulary size limits scalability (one weight per word).
    - No probability output — only hard decisions.

Author : Akhila Pavithran, Rajana
Project: Fake Finders — NLP Master Project, University of Bamberg
"""

import re
import time
import os
import sys

# ── Make src importable regardless of working directory ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

from data_loader              import load_dataset, preview_dataset
from classification_evaluator import ClassificationEvaluator, print_report


# ══════════════════════════════════════════════════════════════════════════════
# Text Processing
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list:
    """
    Converts raw text into clean lowercase tokens.

    Steps:
        1. Lowercase
        2. Remove non-alphabetic characters
        3. Split on whitespace

    Args:
        text: Raw input string.

    Returns:
        List of lowercase word tokens.

    Example:
        tokenize("Free Money!") → ["free", "money"]
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def build_vocab(dataset: list) -> dict:
    """
    Builds a word-to-index dictionary from training data.

    Each unique word gets a unique integer index.
    Used to map words to positions in the weight vector.

    Args:
        dataset: List of (text, label) tuples.

    Returns:
        Dict of {word: index} — deterministic ordering.

    Note:
        Build vocab on TRAINING data only.
        Never include validation or test data in vocab.
    """
    vocab = {}
    idx   = 0
    for text, _ in dataset:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def vectorize(text: str, vocab: dict) -> list:
    """
    Converts text to a binary Bag-of-Words vector.

    Each position in the vector corresponds to one word in vocab.
    Value is 1 if the word appears in text, 0 otherwise.

    Args:
        text  : Input document string.
        vocab : Word-to-index mapping.

    Returns:
        Binary vector of length len(vocab).

    Example:
        vocab = {"free": 0, "money": 1, "call": 2}
        vectorize("free money") → [1, 1, 0]
    """
    vector = [0] * len(vocab)
    for word in tokenize(text):
        if word in vocab:
            vector[vocab[word]] = 1
    return vector


# ══════════════════════════════════════════════════════════════════════════════
# Perceptron Core — directly from professor's slides
# ══════════════════════════════════════════════════════════════════════════════

def sign(score: float) -> int:
    """
    Sign activation function.

    y_hat = sign(w · x + b)

    Returns:
        +1 if score > 0  (fake)
        -1 otherwise     (real)
    """
    return +1 if score > 0 else -1


def predict(x: list, w: list, b: float) -> int:
    """
    Predicts class label for a single input vector.

    y_hat = sign(w · x + b)

    Args:
        x : Input feature vector.
        w : Weight vector.
        b : Bias term.

    Returns:
        +1 (fake) or -1 (real).
    """
    return sign(sum(wi * xi for wi, xi in zip(w, x)) + b)


def update(w: list, b: float, x: list, y: int) -> tuple:
    """
    Updates weights when prediction is wrong.

    Update rule (from professor's slides):
        w = w + y * x
        b = b + y

    Single responsibility:
        This function ONLY updates weights.
        It does NOT predict — that is predict()'s job.

    Args:
        w : Current weight vector.
        b : Current bias.
        x : Input feature vector.
        y : True label (+1 or -1).

    Returns:
        Updated (w, b) tuple.
    """
    return [wi + y * xi for wi, xi in zip(w, x)], b + y


def train(train_data: list, vocab: dict, epochs: int = 10) -> tuple:
    """
    Trains the Perceptron on training data.

    Training loop (from professor's slides):
        For each epoch:
            For each (x, y):
                score = w · x + b
                if prediction is wrong:
                    w = w + y * x
                    b = b + y
        Stop early if zero errors in a full epoch.

    Why subset of data?
        Perceptron stores one weight per vocabulary word.
        Full dataset → 10,000+ weights × 8,000 articles
        → memory and time intensive on a standard laptop.
        Subset still achieves strong performance.

    Args:
        train_data : List of (text, label) tuples.
        vocab      : Word-to-index mapping (built from train only).
        epochs     : Maximum number of training epochs.

    Returns:
        Tuple of (w, b) — trained weight vector and bias.
    """
    w, b = [0] * len(vocab), 0

    print(f"\n=== Training Perceptron ({epochs} epochs max) ===")

    for epoch in range(epochs):
        errors = 0
        for text, y in train_data:
            x = vectorize(text, vocab)
            if predict(x, w, b) != y:
                w, b = update(w, b, x, y)
                errors += 1

        print(f"  Epoch {epoch + 1:02d}/{epochs} | Errors: {errors}")

        if errors == 0:
            print("  Converged early — zero errors!")
            break

    return w, b


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    test_data : list,
    vocab     : dict,
    w         : list,
    b         : float
) -> tuple:
    """
    Runs predictions on test data.

    Args:
        test_data : List of (text, label) tuples.
        vocab     : Word-to-index mapping.
        w         : Trained weight vector.
        b         : Trained bias.

    Returns:
        Tuple of (y_true, y_pred) as 0/1 integer lists
        for ClassificationEvaluator compatibility.
    """
    label_map      = {+1: 1, -1: 0}
    y_true, y_pred = [], []

    for text, y in test_data:
        x = vectorize(text, vocab)
        y_true.append(label_map[y])
        y_pred.append(label_map[predict(x, w, b)])

    return y_true, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# Error Analysis
# ══════════════════════════════════════════════════════════════════════════════

def show_errors(
    test_data : list,
    vocab     : dict,
    w         : list,
    b         : float,
    limit     : int = 5
) -> None:
    """
    Displays misclassified examples for error analysis.

    Required for midterm presentation.
    Shows concrete examples of model failures and connects
    them back to model design (linear boundary, BOW representation).

    Args:
        test_data : List of (text, label) tuples.
        vocab     : Word-to-index mapping.
        w         : Trained weight vector.
        b         : Trained bias.
        limit     : Max number of errors to display.
    """
    label_names = {+1: "FAKE", -1: "REAL"}
    shown       = 0

    print(f"\n{'=' * 55}")
    print("  MISCLASSIFIED EXAMPLES — ERROR ANALYSIS")
    print(f"{'=' * 55}")

    for text, y in test_data:
        x    = vectorize(text, vocab)
        yhat = predict(x, w, b)
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


def show_top_words(vocab: dict, w: list, n: int = 15) -> None:
    """
    Displays words with highest and lowest learned weights.

    High positive weight → strongly associated with FAKE
    High negative weight → strongly associated with REAL

    Args:
        vocab : Word-to-index mapping.
        w     : Trained weight vector.
        n     : Number of top words to display per class.
    """
    word_weights = sorted(
        ((word, w[idx]) for word, idx in vocab.items()),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n  Top {n} words → FAKE news:")
    for word, weight in word_weights[:n]:
        print(f"    {word:25s} {weight:+.0f}")

    print(f"\n  Top {n} words → REAL news:")
    for word, weight in word_weights[-n:][::-1]:
        print(f"    {word:25s} {weight:+.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Load data
    train_data, valid_data, test_data = load_dataset()
    preview_dataset(train_data, split_name="Train")

    # 2. Build vocab from training data only
    print("\n=== Building Vocabulary ===")
    vocab = build_vocab(train_data)
    print(f"  Vocabulary size : {len(vocab):,} words")

    # 3. Train
    start   = time.time()
    w, b    = train(train_data, vocab, epochs=10)
    elapsed = time.time() - start
    print(f"  Training time   : {elapsed:.2f} sec")

    # 4. Evaluate on validation set
    print("\n=== Validation Set ===")
    y_true, y_pred = evaluate(valid_data, vocab, w, b)
    ev = ClassificationEvaluator(y_true, y_pred)
    print_report("Perceptron — Validation", ev.binary_report())

    # 5. Evaluate on test set
    print("\n=== Test Set ===")
    y_true, y_pred = evaluate(test_data, vocab, w, b)
    ev = ClassificationEvaluator(y_true, y_pred)
    print_report("Perceptron — Test Results", ev.binary_report())

    # 6. Error analysis
    show_errors(test_data, vocab, w, b, limit=5)

    # 7. Top words
    show_top_words(vocab, w, n=15)

    # 8. Summary
    print(f"\n{'=' * 55}")
    print("  SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Model     : Perceptron (from scratch)")
    print(f"  Dataset   : LIAR (train.tsv / test.tsv)")
    print(f"  Task      : Binary fake vs real classification")
    print(f"  Train     : {len(train_data)} samples")
    print(f"  Valid     : {len(valid_data)} samples")
    print(f"  Test      : {len(test_data)} samples")
    print(f"  Vocab     : {len(vocab):,} words")
    print(f"  Bias      : {b}")
    print(f"  Time      : {elapsed:.2f} sec")
    print(f"{'=' * 55}")