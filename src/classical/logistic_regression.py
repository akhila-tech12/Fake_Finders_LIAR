"""
logistic_regression.py
======================
Logistic Regression classifier for fake news detection on LIAR dataset.

Algorithm:
    Binary Logistic Regression with Sigmoid activation
    and Gradient Descent optimization.

    Forward pass:
        score     = w · x + b
        prob      = sigmoid(score) = 1 / (1 + exp(-score))
        y_hat     = 1 if prob >= 0.5 else 0

    Loss function (Binary Cross Entropy):
        loss = -[y * log(prob) + (1-y) * log(1-prob)]

    Gradient Descent update:
        error = prob - y
        w     = w - lr * error * x
        b     = b - lr * error

Why Logistic Regression over Perceptron?
    Perceptron makes hard decisions (+1/-1) based on sign.
    Logistic Regression outputs probabilities (0.0 to 1.0)
    giving a confidence score for each prediction.
    It also uses gradient descent which is smoother and more
    stable than perceptron's hard update rule.

Why sigmoid?
    Squashes any real number into range (0, 1).
    Allows probabilistic interpretation of predictions.

Limitations:
    - Still linear — cannot capture non-linear patterns.
    - Bag-of-Words ignores word order and context.
    - Short LIAR statements provide limited signal.

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

from data_loader              import load_dataset, preview_dataset
from classification_evaluator import ClassificationEvaluator, print_report


# ══════════════════════════════════════════════════════════════════════════════
# Text Processing
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list:
    """
    Converts raw text into clean lowercase tokens.

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
    Builds word-to-index dictionary from training data only.

    Args:
        dataset: List of (text, label) tuples.

    Returns:
        Dict of {word: index}.
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
    Converts text to binary Bag-of-Words vector.

    Args:
        text  : Input document string.
        vocab : Word-to-index mapping.

    Returns:
        Binary vector of length len(vocab).
    """
    vector = [0] * len(vocab)
    for word in tokenize(text):
        if word in vocab:
            vector[vocab[word]] = 1
    return vector


# ══════════════════════════════════════════════════════════════════════════════
# Logistic Regression Core
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(score: float) -> float:
    """
    Sigmoid activation function.

    Squashes any real number into range (0, 1).
    Interpreted as probability of being FAKE.

    sigmoid(z) = 1 / (1 + exp(-z))

    Clipped to avoid numerical overflow for very large/small scores.

    Args:
        score: Raw linear score (w · x + b).

    Returns:
        Probability in range (0, 1).
    """
    score = max(-500, min(500, score))
    return 1.0 / (1.0 + math.exp(-score))


def predict_proba(x: list, w: list, b: float) -> float:
    """
    Computes probability of input being FAKE.

    prob = sigmoid(w · x + b)

    Args:
        x : Input feature vector.
        w : Weight vector.
        b : Bias term.

    Returns:
        Probability in range (0, 1).
    """
    return sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b)


def predict(x: list, w: list, b: float, threshold: float = 0.5) -> int:
    """
    Predicts binary class label.

    y_hat = 1 (FAKE) if prob >= threshold else 0 (REAL)

    Args:
        x         : Input feature vector.
        w         : Weight vector.
        b         : Bias term.
        threshold : Decision boundary (default 0.5).

    Returns:
        1 for FAKE, 0 for REAL.
    """
    return 1 if predict_proba(x, w, b) >= threshold else 0


def compute_loss(
    dataset : list,
    vocab   : dict,
    w       : list,
    b       : float
) -> float:
    """
    Computes average Binary Cross Entropy loss over dataset.

    BCE Loss = -[y * log(p) + (1-y) * log(1-p)]

    Lower loss = better model.

    Args:
        dataset : List of (text, label) tuples with 0/1 labels.
        vocab   : Word-to-index mapping.
        w       : Weight vector.
        b       : Bias term.

    Returns:
        Average loss value (float).
    """
    total_loss = 0.0
    eps        = 1e-15  # prevent log(0)

    for text, y in dataset:
        x    = vectorize(text, vocab)
        prob = predict_proba(x, w, b)
        prob = max(eps, min(1 - eps, prob))
        total_loss += -(y * math.log(prob) + (1 - y) * math.log(1 - prob))

    return total_loss / len(dataset)


def train(
    train_data : list,
    vocab      : dict,
    epochs     : int   = 20,
    lr         : float = 0.1
) -> tuple:
    """
    Trains Logistic Regression using Gradient Descent.

    Update rule per example:
        error = prob - y          (prediction error)
        w     = w - lr * error * x  (weight update)
        b     = b - lr * error    (bias update)

    Why Gradient Descent?
        Unlike Perceptron which only updates on mistakes,
        Gradient Descent updates weights on every example
        proportionally to the error magnitude.
        This leads to smoother, more stable convergence.

    Args:
        train_data : List of (text, label) tuples.
                     Labels must be 0/1 (not +1/-1).
        vocab      : Word-to-index mapping.
        epochs     : Number of full passes through data.
        lr         : Learning rate — controls step size.

    Returns:
        Tuple of (w, b) — trained weights and bias.
    """
    w, b = [0.0] * len(vocab), 0.0

    print(f"\n=== Training Logistic Regression ===")
    print(f"  Epochs        : {epochs}")
    print(f"  Learning rate : {lr}")
    print(f"  Vocab size    : {len(vocab):,}")
    print()

    for epoch in range(epochs):
        for text, y in train_data:
            x     = vectorize(text, vocab)
            prob  = predict_proba(x, w, b)
            error = prob - y

            # Gradient descent update
            w = [wi - lr * error * xi for wi, xi in zip(w, x)]
            b = b - lr * error

        # Compute loss every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            loss = compute_loss(train_data, vocab, w, b)
            print(f"  Epoch {epoch + 1:02d}/{epochs} | Loss: {loss:.4f}")

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
        test_data : List of (text, label) tuples with 0/1 labels.
        vocab     : Word-to-index mapping.
        w         : Trained weight vector.
        b         : Trained bias.

    Returns:
        Tuple of (y_true, y_pred) as 0/1 integer lists.
    """
    y_true, y_pred = [], []

    for text, y in test_data:
        x = vectorize(text, vocab)
        y_true.append(y)
        y_pred.append(predict(x, w, b))

    return y_true, y_pred


def convert_labels(dataset: list) -> list:
    """
    Converts +1/-1 labels to 1/0 for Logistic Regression.

    Logistic Regression uses sigmoid → outputs 0 to 1.
    So labels must be 0/1 not +1/-1.

    Args:
        dataset: List of (text, label) with +1/-1 labels.

    Returns:
        List of (text, label) with 1/0 labels.
    """
    return [(text, 1 if y == +1 else 0) for text, y in dataset]


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
    Displays misclassified examples with confidence scores.

    Shows not just what was wrong but HOW confident the model was.
    High confidence wrong predictions = interesting error cases.

    Args:
        test_data : List of (text, label) tuples with 0/1 labels.
        vocab     : Word-to-index mapping.
        w         : Trained weight vector.
        b         : Trained bias.
        limit     : Max number of errors to display.
    """
    label_names = {1: "FAKE", 0: "REAL"}
    shown       = 0

    print(f"\n{'=' * 55}")
    print("  MISCLASSIFIED EXAMPLES — ERROR ANALYSIS")
    print(f"{'=' * 55}")

    for text, y in test_data:
        x    = vectorize(text, vocab)
        yhat = predict(x, w, b)
        prob = predict_proba(x, w, b)

        if yhat != y:
            confidence = prob if yhat == 1 else 1 - prob
            print(f"  TEXT       : {text[:120]}")
            print(f"  TRUE       : {label_names[y]}")
            print(f"  PRED       : {label_names[yhat]}")
            print(f"  CONFIDENCE : {confidence:.2%}")
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
        n     : Number of top words per class.
    """
    word_weights = sorted(
        ((word, w[idx]) for word, idx in vocab.items()),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n  Top {n} words → FAKE news:")
    for word, weight in word_weights[:n]:
        print(f"    {word:25s} {weight:+.4f}")

    print(f"\n  Top {n} words → REAL news:")
    for word, weight in word_weights[-n:][::-1]:
        print(f"    {word:25s} {weight:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Load data
    train_data, valid_data, test_data = load_dataset()
    preview_dataset(train_data, split_name="Train")

    # 2. Convert labels from +1/-1 to 1/0
    train_data = convert_labels(train_data)
    valid_data = convert_labels(valid_data)
    test_data  = convert_labels(test_data)

    # 3. Build vocab from training data only
    print("\n=== Building Vocabulary ===")
    vocab = build_vocab(train_data)
    print(f"  Vocabulary size : {len(vocab):,} words")

    # 4. Train
    start   = time.time()
    w, b    = train(train_data, vocab, epochs=20, lr=0.1)
    elapsed = time.time() - start
    print(f"\n  Training time   : {elapsed:.2f} sec")

    # 5. Evaluate on validation set
    print("\n=== Validation Set ===")
    y_true, y_pred = evaluate(valid_data, vocab, w, b)
    ev = ClassificationEvaluator(y_true, y_pred)
    print_report("Logistic Regression — Validation", ev.binary_report())

    # 6. Evaluate on test set
    print("\n=== Test Set ===")
    y_true, y_pred = evaluate(test_data, vocab, w, b)
    ev = ClassificationEvaluator(y_true, y_pred)
    print_report("Logistic Regression — Test Results", ev.binary_report())

    # 7. Error analysis
    show_errors(test_data, vocab, w, b, limit=5)

    # 8. Top words
    show_top_words(vocab, w, n=15)

    # 9. Summary
    print(f"\n{'=' * 55}")
    print("  SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Model     : Logistic Regression (from scratch)")
    print(f"  Dataset   : LIAR (train.tsv / test.tsv)")
    print(f"  Task      : Binary fake vs real classification")
    print(f"  Train     : {len(train_data)} samples")
    print(f"  Valid     : {len(valid_data)} samples")
    print(f"  Test      : {len(test_data)} samples")
    print(f"  Vocab     : {len(vocab):,} words")
    print(f"  Epochs    : 20")
    print(f"  LR        : 0.1")
    print(f"  Bias      : {b:.4f}")
    print(f"  Time      : {elapsed:.2f} sec")
    print(f"{'=' * 55}")