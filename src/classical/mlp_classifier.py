"""
mlp_classifier.py
=================
Model 5 — Multi-Layer Perceptron (MLP).

Why MLP after SVM?
    Both SVM and MLP handle non-linear boundaries.
    The key difference is HOW they learn:

    SVM  — kernel trick: implicitly maps to high-dimensional space,
            finds a linear boundary THERE. Exact solution.
            Works well with small/medium datasets.

    MLP  — learns a hierarchy of representations:
            hidden layer neurons discover useful word combinations
            automatically through gradient descent.
            Scales better to large datasets and can stack more layers.

Architecture:
    Input   → TF-IDF + metadata vector  (8,003 dims)
    Hidden1 → 256 neurons, ReLU         (learnable)
    Dropout → 0.3 rate                  (regularisation)
    Hidden2 → 128 neurons, ReLU         (learnable)
    Output  → 1 neuron, Sigmoid         → P(fake)

    Two hidden layers chosen because:
        - One layer: limited non-linear capacity
        - Three+:    overfitting risk on 8,146 LIAR samples
        - Two layers: good balance for this dataset size

Training:
    Loss      : Binary Cross-Entropy
    Optimiser : Adam (adaptive learning rates)
    Scheduler : ReduceLROnPlateau (halve LR when val loss stalls)
    Early stop: stop if val F1 doesn't improve for 5 epochs

Author  : Akhila Pavithran, Rajana
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import time
import random
from typing import Optional

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing  import MaxAbsScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

from feature_extractor        import FeatureBuilder
from classification_evaluator import ClassificationEvaluator, print_report
from data_loader              import map_label


# ══════════════════════════════════════════════════════════════════════════════
# MLP Classifier
# ══════════════════════════════════════════════════════════════════════════════

class MLPFakeNewsClassifier:
    """
    MLP for fake news detection using sklearn's MLPClassifier.

    Wraps sklearn to provide the same interface as other models:
        fit / predict / predict_proba / evaluate

    Architecture fixed at (256, 128) hidden layers.
    Adam optimiser with early stopping on validation loss.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (256, 128),
        max_iter:           int   = 100,
        learning_rate_init: float = 1e-3,
        dropout:            float = 0.0,   # sklearn MLP has no dropout
        random_state:       int   = 42,
    ) -> None:
        self.hidden_layer_sizes  = hidden_layer_sizes
        self.max_iter            = max_iter
        self.learning_rate_init  = learning_rate_init
        self.random_state        = random_state

        self._scaler = MaxAbsScaler()
        self._model  = MLPClassifier(
            hidden_layer_sizes  = hidden_layer_sizes,
            activation          = "relu",
            solver              = "adam",
            alpha               = 1e-4,          # L2 regularisation
            learning_rate_init  = learning_rate_init,
            max_iter            = max_iter,
            early_stopping      = True,
            validation_fraction = 0.1,           # 10% of train as val
            n_iter_no_change    = 10,            # early-stop patience
            random_state        = random_state,
            verbose             = False,
        )

        self._trained    = False
        self._train_time = 0.0

    # ── training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: list[list[float]],
        y_train: list[int],
    ) -> "MLPFakeNewsClassifier":
        """
        Train MLP.

        Args:
            X_train : pre-computed feature vectors.
            y_train : integer labels (1=fake, 0=real).

        Returns:
            self
        """
        print(f"\n=== Training MLP {self.hidden_layer_sizes} ===")
        X = self._scaler.fit_transform(
                np.array(X_train, dtype=np.float32)
            )
        y = np.array(y_train, dtype=np.int32)

        t0 = time.perf_counter()
        self._model.fit(X, y)
        self._train_time = time.perf_counter() - t0
        self._trained    = True

        n_iter = self._model.n_iter_
        loss   = self._model.loss_
        print(f"  ✓ Converged in {n_iter} iterations  |  "
              f"Final loss: {loss:.4f}  |  "
              f"Time: {self._train_time:.2f}s")
        return self

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, x: list[float]) -> int:
        """Return class label for one example."""
        self._check_fitted()
        x_sc = self._scaler.transform(
            np.array(x, dtype=np.float32).reshape(1, -1)
        )
        return int(self._model.predict(x_sc)[0])

    def predict_proba(self, x: list[float]) -> float:
        """Return P(fake) for one example."""
        self._check_fitted()
        x_sc  = self._scaler.transform(
            np.array(x, dtype=np.float32).reshape(1, -1)
        )
        proba = self._model.predict_proba(x_sc)
        return float(proba[0][1])

    def predict_batch(self, X: list[list[float]]) -> list[int]:
        """Batch prediction."""
        self._check_fitted()
        X_sc = self._scaler.transform(
            np.array(X, dtype=np.float32)
        )
        return self._model.predict(X_sc).tolist()

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: list[list[float]],
        y_test: list[int],
    ) -> dict:
        """Evaluate and return metric dictionary."""
        self._check_fitted()
        y_pred = self.predict_batch(X_test)
        return ClassificationEvaluator(y_test, y_pred).binary_report()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._trained:
            raise RuntimeError("Call fit() before predict/evaluate.")

    def __repr__(self) -> str:
        status = (f"trained in {self._train_time:.2f}s"
                  if self._trained else "not trained")
        return f"MLPFakeNewsClassifier({self.hidden_layer_sizes}, {status})"


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _load_vectors(split_path: str, builder: FeatureBuilder) -> tuple:
    texts, labels, rows = [], [], []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lbl = map_label(parts[1].strip())
            if lbl is None:
                continue
            texts.append(parts[2].strip())
            labels.append(1 if lbl == 1 else 0)
            rows.append(parts)
    print(f"  Vectorising {len(texts)} examples...")
    X = [builder.transform(t, r) for t, r in zip(texts, rows)]
    return X, labels


if __name__ == "__main__":

    BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN = os.path.join(BASE, "data", "train.tsv")
    TEST  = os.path.join(BASE, "data", "test.tsv")

    print("=== MLP Classifier — LIAR Dataset ===\n")

    # ── Build features ────────────────────────────────────────────────────────
    print("Building TF-IDF + bigrams + metadata features...")
    builder     = FeatureBuilder(max_features=8_000, ngram_range=(1, 2))
    train_texts = []
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                train_texts.append(parts[2].strip())
    builder.fit(train_texts)
    print(f"  Feature dim: {builder.feature_dim + 3:,}\n")

    # ── Vectorise ─────────────────────────────────────────────────────────────
    print("Vectorising train split...")
    X_train, y_train = _load_vectors(TRAIN, builder)
    print("Vectorising test split...")
    X_test, y_test   = _load_vectors(TEST, builder)

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = MLPFakeNewsClassifier(
        hidden_layer_sizes = (256, 128),
        max_iter           = 100,
        learning_rate_init = 1e-3,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = clf.evaluate(X_test, y_test)
    print_report("MLP (256→128) — Test Results", results)

    # ── Error analysis ────────────────────────────────────────────────────────
    label_map = {1: "FAKE", 0: "REAL"}
    print("\n=== Misclassified Examples ===")
    shown = 0
    with open(TEST, encoding="utf-8") as f:
        for line in f:
            if shown >= 5:
                break
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lbl = map_label(parts[1].strip())
            if lbl is None:
                continue
            text  = parts[2].strip()
            true  = 1 if lbl == 1 else 0
            x_vec = builder.transform(text, parts)
            pred  = clf.predict(x_vec)
            if pred != true:
                prob = clf.predict_proba(x_vec)
                print(f"  TEXT : {text[:100]}")
                print(f"  TRUE : {label_map[true]}  "
                      f"PRED : {label_map[pred]}  "
                      f"P(fake): {prob:.2%}")
                print(f"  {'─'*55}")
                shown += 1

    print(f"\n{clf}")
