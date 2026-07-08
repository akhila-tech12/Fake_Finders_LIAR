"""
ensemble_classifier.py
======================
Ensemble model combining ALL classifiers for Fake News Detection.

Why ensemble?
    Each model sees the problem differently:
    NB/Perc/LR  -> simple word patterns (BOW)
    SVM/MLP     -> TF-IDF + metadata (speaker credibility)
    BERT+FiLM   -> deep language + metadata modulation

    When models are DIVERSE, their errors don't overlap.
    Combining diverse models cancels out individual errors.

    Our 3-way test (NB+Perc+LR) FAILED because those models
    are too similar (all BOW). This full ensemble includes
    SVM, MLP, BERT which use completely different features
    -> genuine diversity -> real improvement expected!

Two voting strategies:
    1. Simple majority vote   -> each model gets 1 vote
    2. Weighted majority vote -> each model votes weighted by F1

Run WITHOUT BERT (Mac only, ~10 min):
    cd ~/fake-finders-liar
    source venv/bin/activate
    python src/ensemble_classifier.py

Run WITH BERT (after downloading from Kaggle):
    python src/ensemble_classifier.py --bert_path models/bert_film_v3

Author  : Akhila Pavithran, Rajana
Project : Fake Finders -- NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse

import numpy as np
from sklearn.svm            import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing  import MaxAbsScaler
from sklearn.pipeline       import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor        import FeatureBuilder, metadata_features
from classification_evaluator import ClassificationEvaluator, print_report
from data_loader              import map_label

import naive_bayes        as nb
import perceptron         as perc
import logistic_regression as lr


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_with_meta(path: str):
    texts, labels, rows = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lbl = map_label(parts[1].strip())
            if lbl is None:
                continue
            texts.append(parts[2].strip())
            labels.append(lbl)   # +1 or -1
            rows.append(parts)
    return texts, labels, rows


# ══════════════════════════════════════════════════════════════════════════════
# Model Wrappers — all return (pred_01, proba)
# ══════════════════════════════════════════════════════════════════════════════

class NBWrapper:
    name = "Naive Bayes"
    f1   = 0.6446

    def fit(self, train_dataset):
        self.priors, self.likelihoods, self.vocab = nb.train(train_dataset)

    def predict(self, text, row=None):
        pred   = nb.predict(text, self.priors, self.likelihoods, self.vocab)
        s_fake = nb.score(text, +1, self.priors, self.likelihoods, self.vocab)
        s_real = nb.score(text, -1, self.priors, self.likelihoods, self.vocab)
        m      = max(s_fake, s_real)
        proba  = math.exp(s_fake - m) / (
                     math.exp(s_fake - m) + math.exp(s_real - m))
        return (1 if pred == 1 else 0), proba


class PercWrapper:
    name = "Perceptron"
    f1   = 0.5791

    def fit(self, train_dataset):
        self.vocab     = perc.build_vocab(train_dataset)
        self.w, self.b = perc.train(train_dataset, self.vocab, epochs=10)

    def predict(self, text, row=None):
        x     = perc.vectorize(text, self.vocab)
        pred  = perc.predict(x, self.w, self.b)
        raw   = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        proba = 1 / (1 + math.exp(-max(-500, min(500, raw / 10))))
        return (1 if pred == 1 else 0), proba


class LRWrapper:
    name = "Logistic Reg"
    f1   = 0.6125

    def fit(self, train_dataset):
        self.vocab     = lr.build_vocab(train_dataset)
        train_01       = lr.convert_labels(train_dataset)
        self.w, self.b = lr.train(train_01, self.vocab, epochs=20, lr=0.1)

    def predict(self, text, row=None):
        x     = lr.vectorize(text, self.vocab)
        proba = lr.predict_proba(x, self.w, self.b)
        pred  = lr.predict(x, self.w, self.b)
        return pred, proba


class SVMWrapper:
    name = "SVM"
    f1   = 0.7271

    def fit(self, X_train, y_train):
        self.pipeline = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("svm",    SVC(C=1.0, kernel="rbf", probability=True,
                          class_weight="balanced", random_state=42)),
        ])
        self.pipeline.fit(
            np.array(X_train, dtype=np.float32),
            np.array(y_train, dtype=np.int32)
        )

    def predict_vec(self, x_vec):
        x     = np.array(x_vec, dtype=np.float32).reshape(1, -1)
        pred  = int(self.pipeline.predict(x)[0])
        proba = float(self.pipeline.predict_proba(x)[0][1])
        return pred, proba


class MLPWrapper:
    name = "MLP"
    f1   = 0.7542

    def fit(self, X_train, y_train):
        self.scaler = MaxAbsScaler()
        X_sc        = self.scaler.fit_transform(
                          np.array(X_train, dtype=np.float32))
        self.model  = MLPClassifier(
            hidden_layer_sizes=(256, 128), activation="relu",
            solver="adam", max_iter=100, early_stopping=True,
            random_state=42, verbose=False,
        )
        self.model.fit(X_sc, np.array(y_train, dtype=np.int32))

    def predict_vec(self, x_vec):
        x     = self.scaler.transform(
                    np.array(x_vec, dtype=np.float32).reshape(1, -1))
        pred  = int(self.model.predict(x)[0])
        proba = float(self.model.predict_proba(x)[0][1])
        return pred, proba


class BERTFiLMWrapper:
    """Optional -- only loaded if --bert_path is provided."""
    name = "BERT+FiLM"
    f1   = 0.7706

    def __init__(self, bert_path: str):
        import torch
        from transformers import BertTokenizerFast, BertModel
        import torch.nn as nn

        self.device    = torch.device("cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_path)
        self.max_len   = 128

        class FiLMLayer(nn.Module):
            def __init__(self, meta_dim=3, bert_dim=768, hidden=32):
                super().__init__()
                self.to_gamma = nn.Sequential(
                    nn.Linear(meta_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, bert_dim))
                self.to_beta = nn.Sequential(
                    nn.Linear(meta_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, bert_dim))
            def forward(self, cls, meta):
                return self.to_gamma(meta) * cls + self.to_beta(meta)

        class BERTWithFiLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert       = BertModel.from_pretrained(bert_path)
                self.film       = FiLMLayer()
                self.dropout    = nn.Dropout(0.1)
                self.classifier = nn.Linear(768, 2)
            def forward(self, input_ids, attention_mask,
                        token_type_ids, metadata):
                out    = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                pooled = out.pooler_output
                modded = self.film(pooled, metadata)
                modded = self.dropout(modded)
                return self.classifier(modded)

        self.model = BERTWithFiLM().to(self.device)
        weights    = os.path.join(bert_path, "model.pt")
        if os.path.exists(weights):
            import torch
            state = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"  Loaded BERT+FiLM from {weights}")
        else:
            print(f"  WARNING: model.pt not found at {weights}")
        self.model.eval()

    def predict(self, text: str, meta_vec: list) -> tuple:
        import torch
        enc    = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        meta_t = torch.tensor([meta_vec], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(
                enc["input_ids"], enc["attention_mask"],
                enc["token_type_ids"], meta_t,
            )
            probs  = torch.softmax(logits, dim=1)
            pred   = int(torch.argmax(probs).item())
            proba  = float(probs[0][1].item())
        return pred, proba


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleClassifier:
    """
    Combines all available models via majority voting.

    Simple   -> each model gets 1 vote, majority wins
    Weighted -> each model's vote is weighted by its known F1 score
    """

    # Known F1 scores -- used for weighted voting
    WEIGHTS = {
        "nb"  : 0.6446,
        "pc"  : 0.5791,
        "lr"  : 0.6125,
        "svm" : 0.7271,
        "mlp" : 0.7542,
        "bert": 0.7706,
    }

    def __init__(self, bert_path: str = None):
        self.bert_path = bert_path

    def fit(self, train_texts, train_labels_pm, train_rows,
            X_train_vec, y_train_01) -> None:

        print("\n[Ensemble] Training all models...")
        train_dataset = list(zip(train_texts, train_labels_pm))

        print("  Naive Bayes...")
        self.nb  = NBWrapper();  self.nb.fit(train_dataset)

        print("  Perceptron...")
        self.pc  = PercWrapper(); self.pc.fit(train_dataset)

        print("  Logistic Regression...")
        self.lr_ = LRWrapper();  self.lr_.fit(train_dataset)

        print("  SVM...")
        self.svm = SVMWrapper(); self.svm.fit(X_train_vec, y_train_01)

        print("  MLP...")
        self.mlp = MLPWrapper(); self.mlp.fit(X_train_vec, y_train_01)

        self.bert = None
        if self.bert_path and os.path.exists(self.bert_path):
            print(f"  BERT+FiLM from {self.bert_path}...")
            try:
                self.bert = BERTFiLMWrapper(self.bert_path)
            except Exception as e:
                print(f"  WARNING: Could not load BERT: {e}")

        n = 5 + (1 if self.bert else 0)
        print(f"\n  Ready with {n} models!")

    def _predict_all(self, text, x_vec, row):
        meta = metadata_features(row)
        p    = {}
        p["nb"]  = self.nb.predict(text)
        p["pc"]  = self.pc.predict(text)
        p["lr"]  = self.lr_.predict(text)
        p["svm"] = self.svm.predict_vec(x_vec)
        p["mlp"] = self.mlp.predict_vec(x_vec)
        if self.bert:
            p["bert"] = self.bert.predict(text, meta)
        return p

    def _vote(self, preds: dict, strategy: str) -> tuple:
        if strategy == "simple":
            votes      = [p for p, _ in preds.values()]
            fake_votes = sum(votes)
            pred       = 1 if fake_votes > len(votes) / 2 else 0
            conf       = fake_votes / len(votes)
        else:
            wfake = wsum = 0.0
            for key, (_, proba) in preds.items():
                w      = self.WEIGHTS.get(key, 0.65)
                wfake += w * proba
                wsum  += w
            conf = wfake / wsum
            pred = 1 if conf >= 0.5 else 0
        return pred, conf

    def predict(self, text: str, x_vec: list, row: list,
                strategy: str = "weighted") -> tuple:
        preds = self._predict_all(text, x_vec, row)
        return self._vote(preds, strategy)

    def evaluate(self, test_texts, test_labels_pm,
                 test_rows, X_test_vec) -> dict:
        labels_01      = [1 if l == 1 else 0 for l in test_labels_pm]
        simple_preds   = []
        weighted_preds = []

        print("\n  Running ensemble on test set...")
        for i, (text, row, x_vec) in enumerate(
                zip(test_texts, test_rows, X_test_vec), 1):
            all_p = self._predict_all(text, x_vec, row)
            simple_preds  .append(self._vote(all_p, "simple")[0])
            weighted_preds.append(self._vote(all_p, "weighted")[0])
            if i % 200 == 0:
                print(f"    {i}/{len(test_texts)} done...")

        return {
            "simple"  : ClassificationEvaluator(
                            labels_01, simple_preds).binary_report(),
            "weighted": ClassificationEvaluator(
                            labels_01, weighted_preds).binary_report(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bert_path", type=str, default=None,
        help="Path to downloaded BERT+FiLM model folder "
             "(e.g. ../models/bert_film_v3)"
    )
    args = parser.parse_args()

    BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_PATH = os.path.join(BASE, "data", "train.tsv")
    TEST_PATH  = os.path.join(BASE, "data", "test.tsv")
    RESULTS    = os.path.join(BASE, "results")
    os.makedirs(RESULTS, exist_ok=True)

    print("=" * 60)
    print("  ENSEMBLE CLASSIFIER -- Fake Finders")
    bert_status = f"WITH BERT ({args.bert_path})" \
                  if args.bert_path else "WITHOUT BERT (5 models)"
    print(f"  {bert_status}")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    train_texts, train_labels_pm, train_rows = load_with_meta(TRAIN_PATH)
    test_texts,  test_labels_pm,  test_rows  = load_with_meta(TEST_PATH)
    print(f"  Train: {len(train_texts)}  Test: {len(test_texts)}")

    # ── Features ──────────────────────────────────────────────────────────
    print("\n[2/4] Building TF-IDF + metadata features...")
    builder = FeatureBuilder(max_features=8_000, ngram_range=(1, 2))
    builder.fit(train_texts)
    X_train    = [builder.transform(t, r)
                  for t, r in zip(train_texts, train_rows)]
    X_test     = [builder.transform(t, r)
                  for t, r in zip(test_texts, test_rows)]
    y_train_01 = [1 if l == 1 else 0 for l in train_labels_pm]
    print(f"  Feature dim: {len(X_train[0]):,}")

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n[3/4] Training...")
    ensemble = EnsembleClassifier(bert_path=args.bert_path)
    ensemble.fit(train_texts, train_labels_pm, train_rows,
                 X_train, y_train_01)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating...")
    results = ensemble.evaluate(test_texts, test_labels_pm,
                                test_rows, X_test)

    print_report("Ensemble Simple Vote   -- Test", results["simple"])
    print_report("Ensemble Weighted Vote -- Test", results["weighted"])

    # ── Comparison table ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  COMPLETE COMPARISON -- Individual + Ensemble")
    print(f"{'='*62}")
    known = [
        ("1. Naive Bayes",         0.6446),
        ("2. Perceptron",          0.5791),
        ("3. Logistic Reg",        0.6125),
        ("4. SVM",                 0.7271),
        ("5. MLP",                 0.7542),
        ("6. BERT+FiLM",           0.7706),
        ("7. Ensemble (simple)",   results["simple"]["f1"]),
        ("8. Ensemble (weighted)", results["weighted"]["f1"]),
    ]
    best_f1 = max(f1 for _, f1 in known)
    for name, f1 in known:
        star = " ★" if f1 == best_f1 else "  "
        print(f"  {star} {name:<30} F1: {f1:.4f}")
    print(f"{'='*62}")

    # ── Save ───────────────────────────────────────────────────────────────
    out = os.path.join(RESULTS, "ensemble_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")