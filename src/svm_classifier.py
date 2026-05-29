"""
svm_classifier.py
=================
Model 4 — Support Vector Machine (SVM) with RBF kernel.

Why SVM?
    Our Perceptron (Model 2) never converged on LIAR after 10 epochs.
    This proves LIAR is NOT linearly separable — fake and real
    statements share too much vocabulary for a straight-line boundary.

    SVM with RBF kernel solves this:
        Linear models  → draw one straight line between classes
        SVM + RBF      → maps data into higher-dimensional space
                          where a linear boundary DOES exist
                       → effectively draws a CURVED boundary
                          in the original space

    Additionally, SVM finds the MAXIMUM MARGIN boundary:
        Other models   → find any separating boundary
        SVM            → finds the boundary that maximises the gap
                          between the closest fake and real examples
                       → more robust when classes overlap (LIAR!)

Hyperparameters:
    C     : regularisation. High C = fit training data closely.
             Low C = allow more misclassifications for wider margin.
    gamma : RBF width. "scale" = 1 / (n_features × var(X)).

Author  : Akhila Pavithran, Rajana
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor       import FeatureBuilder
from classification_evaluator import ClassificationEvaluator, print_report
from data_loader              import load_dataset


# ══════════════════════════════════════════════════════════════════════════════
# SVM Classifier
# ══════════════════════════════════════════════════════════════════════════════

class SVMClassifier:
    """
    SVM with RBF kernel, wrapped around sklearn's SVC.

    Includes MaxAbsScaler because SVM is sensitive to feature scale.
    TF-IDF vectors are already sparse; MaxAbsScaler preserves sparsity
    while normalising each feature to [-1, 1].

    Public interface:
        fit(X_train, y_train)
        predict(x)            → int
        predict_proba(x)      → float (P(fake))
        evaluate(X_test, y_test) → dict
    """

    # Hyperparameters chosen by common practice for text classification
    _DEFAULT_C     = 1.0
    _DEFAULT_GAMMA = "scale"
    _DEFAULT_KERNEL= "rbf"

    def __init__(
        self,
        C:      float = _DEFAULT_C,
        kernel: str   = _DEFAULT_KERNEL,
        gamma:  str   = _DEFAULT_GAMMA,
    ) -> None:
        self.C      = C
        self.kernel = kernel
        self.gamma  = gamma

        # Pipeline: scale → SVM
        self._pipeline = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("svm",    SVC(
                C            = C,
                kernel       = kernel,
                gamma        = gamma,
                probability  = True,   # needed for predict_proba
                class_weight = "balanced",  # handles class imbalance
                random_state = 42,
            )),
        ])

        self._trained      = False
        self._train_time   = 0.0

    # ── training ─────────────────────────────────────────────────────────────

    def fit(self, X_train: list[list[float]], y_train: list[int]) -> "SVMClassifier":
        """
        Fit SVM on pre-computed feature vectors.

        Args:
            X_train : list of dense TF-IDF+metadata vectors.
            y_train : list of int labels (1=fake, 0=real).

        Returns:
            self
        """
        print(f"\n=== Training SVM (kernel={self.kernel}, C={self.C}) ===")
        X = np.array(X_train, dtype=np.float32)
        y = np.array(y_train, dtype=np.int32)

        t0 = time.perf_counter()
        self._pipeline.fit(X, y)
        self._train_time = time.perf_counter() - t0
        self._trained    = True

        print(f"  ✓ Trained in {self._train_time:.2f}s  |  "
              f"Support vectors: {self._pipeline['svm'].n_support_}")
        return self

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, x: list[float]) -> int:
        """Return class label (1=fake, 0=real) for one example."""
        self._check_fitted()
        return int(self._pipeline.predict(
            np.array(x, dtype=np.float32).reshape(1, -1)
        )[0])

    def predict_proba(self, x: list[float]) -> float:
        """Return P(fake) for one example."""
        self._check_fitted()
        proba = self._pipeline.predict_proba(
            np.array(x, dtype=np.float32).reshape(1, -1)
        )
        return float(proba[0][1])   # index 1 = positive (fake) class

    def predict_batch(self, X: list[list[float]]) -> list[int]:
        """Batch predict — much faster than looping predict()."""
        self._check_fitted()
        return self._pipeline.predict(
            np.array(X, dtype=np.float32)
        ).tolist()

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: list[list[float]],
        y_test: list[int],
    ) -> dict:
        """
        Evaluate on test set and return metric dictionary.

        Args:
            X_test : list of feature vectors.
            y_test : ground-truth labels.

        Returns:
            dict with TP/FP/FN/TN/accuracy/precision/recall/f1.
        """
        self._check_fitted()
        y_pred = self.predict_batch(X_test)
        ev     = ClassificationEvaluator(y_test, y_pred)
        return ev.binary_report()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._trained:
            raise RuntimeError("Call fit() before predict/evaluate.")

    def summary(self) -> str:
        status = (f"trained in {self._train_time:.2f}s"
                  if self._trained else "not trained")
        return (f"SVMClassifier("
                f"kernel={self.kernel}, C={self.C}, "
                f"gamma={self.gamma}, {status})")

    def __repr__(self) -> str:
        return self.summary()


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _load_vectors(split_path: str, builder: FeatureBuilder) -> tuple:
    """Helper: load LIAR split and vectorise with metadata."""
    from data_loader import map_label

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

    BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN    = os.path.join(BASE, "data", "train.tsv")
    TEST     = os.path.join(BASE, "data", "test.tsv")

    # ── 1. Feature builder ────────────────────────────────────────────────────
    print("=== SVM Classifier — LIAR Dataset ===\n")
    print("Building features (TF-IDF + bigrams + metadata)...")

    builder = FeatureBuilder(max_features=8_000, ngram_range=(1, 2))

    # Fit on training texts only
    train_texts: list[str] = []
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                train_texts.append(parts[2].strip())
    builder.fit(train_texts)
    print(f"  Feature dimension: {builder.feature_dim:,} (TF-IDF) "
          f"+ 3 (metadata) = {builder.feature_dim + 3:,} total\n")

    # ── 2. Vectorise ──────────────────────────────────────────────────────────
    print("Vectorising train split...")
    X_train, y_train = _load_vectors(TRAIN, builder)

    print("Vectorising test split...")
    X_test, y_test = _load_vectors(TEST, builder)

    # ── 3. Train & evaluate ───────────────────────────────────────────────────
    clf = SVMClassifier(C=1.0, kernel="rbf")
    clf.fit(X_train, y_train)

    results = clf.evaluate(X_test, y_test)
    print_report("SVM (RBF) — Test Results", results)

    # ── 4. Error analysis ─────────────────────────────────────────────────────
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
            from data_loader import map_label
            lbl = map_label(parts[1].strip())
            if lbl is None:
                continue
            text  = parts[2].strip()
            true  = 1 if lbl == 1 else 0
            x_vec = builder.transform(text, parts)
            pred  = clf.predict(x_vec)
            if pred != true:
                print(f"  TEXT : {text[:100]}")
                print(f"  TRUE : {label_map[true]}  "
                      f"PRED : {label_map[pred]}  "
                      f"P(fake): {clf.predict_proba(x_vec):.2%}")
                print(f"  {'─'*55}")
                shown += 1

    print(f"\n{clf}")
