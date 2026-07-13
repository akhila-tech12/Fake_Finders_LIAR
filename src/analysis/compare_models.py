"""
compare_models.py
=================
Run ALL 7 models and produce the final comparison table.

Models compared:
    1. Naive Bayes        (baseline — first half)
    2. Perceptron         (baseline — first half)
    3. Logistic Regression(baseline — first half)
    4. SVM + RBF          (second half)
    5. MLP (256→128)      (second half)
    6. BERT fine-tuned    (second half — run on Colab)
    7. BERT + RAG         (second half — run on Colab)

All models use the same test split for fair comparison.
Models 4-5 use TF-IDF + bigrams + metadata features.
Models 6-7 use BERT tokenisation + optionally evidence.

Author  : Akhila Pavithran, Rajana
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

from feature_extractor        import FeatureBuilder
from classical.svm_classifier import SVMClassifier
from classical.mlp_classifier import MLPFakeNewsClassifier
from classification_evaluator import print_report
from data_loader              import map_label


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_with_meta(path: str) -> tuple:
    """Load LIAR split → (texts, labels, rows)."""
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
            labels.append(1 if lbl == 1 else 0)
            rows.append(parts)
    return texts, labels, rows


def print_comparison(results: list[dict]) -> None:
    """Pretty-print final comparison table."""
    sep = "=" * 75
    print(f"\n{sep}")
    print("  FINAL MODEL COMPARISON — LIAR TEST SET")
    print(sep)
    header = (f"  {'Model':<30} {'Accuracy':>9} "
              f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'Time':>8}")
    print(header)
    print(f"  {'-' * 71}")

    for r in results:
        flag = " ★" if r["f1"] == max(x["f1"] for x in results) else ""
        print(
            f"  {r['model']:<30} "
            f"{r['accuracy']:>9.4f} "
            f"{r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{r['time']:>7.1f}s"
            f"{flag}"
        )

    print(sep)
    best = max(results, key=lambda x: x["f1"])
    print(f"  Best F1: {best['model']} ({best['f1']:.4f})")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    BASE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TRAIN_PATH = os.path.join(BASE, "data", "train.tsv")
    TEST_PATH  = os.path.join(BASE, "data", "test.tsv")
    RESULTS    = os.path.join(BASE, "results")
    os.makedirs(RESULTS, exist_ok=True)

    all_results: list[dict] = []

    # ── Baseline results (from first half — already computed) ─────────────────
    baselines = [
        {"model": "1. Naive Bayes",
         "accuracy": 0.6201, "precision": 0.6604,
         "recall": 0.6295,   "f1": 0.6446, "time": 0.04},
        {"model": "2. Perceptron",
         "accuracy": 0.5965, "precision": 0.6746,
         "recall": 0.5072,   "f1": 0.5791, "time": 19.71},
        {"model": "3. Logistic Regression",
         "accuracy": 0.6102, "precision": 0.6717,
         "recall": 0.5629,   "f1": 0.6125, "time": 97.35},
    ]
    all_results.extend(baselines)

    # ── Build shared feature vectors (Models 4 + 5) ───────────────────────────
    print("=" * 60)
    print("  Building TF-IDF + bigrams + metadata features")
    print("=" * 60)

    builder     = FeatureBuilder(max_features=8_000, ngram_range=(1, 2))
    train_texts, train_labels, train_rows = load_with_meta(TRAIN_PATH)
    test_texts,  test_labels,  test_rows  = load_with_meta(TEST_PATH)

    builder.fit(train_texts)
    feat_dim = builder.feature_dim + 3   # +3 for metadata
    print(f"  Feature dim: {feat_dim:,}\n")

    print("  Vectorising train split...")
    X_train = [builder.transform(t, r)
               for t, r in zip(train_texts, train_rows)]

    print("  Vectorising test split...")
    X_test  = [builder.transform(t, r)
               for t, r in zip(test_texts, test_rows)]

    # ── Model 4: SVM ──────────────────────────────────────────────────────────
    svm = SVMClassifier(C=1.0, kernel="rbf")

    t0 = time.perf_counter()
    svm.fit(X_train, train_labels)
    svm_metrics = svm.evaluate(X_test, test_labels)
    svm_time    = time.perf_counter() - t0

    all_results.append({
        "model"    : "4. SVM (RBF)",
        "accuracy" : svm_metrics["accuracy"],
        "precision": svm_metrics["precision"],
        "recall"   : svm_metrics["recall"],
        "f1"       : svm_metrics["f1"],
        "time"     : svm_time,
    })
    print_report("SVM (RBF) — Test Results", svm_metrics)

    # ── Model 5: MLP ──────────────────────────────────────────────────────────
    mlp = MLPFakeNewsClassifier(
        hidden_layer_sizes = (256, 128),
        max_iter           = 100,
        learning_rate_init = 1e-3,
    )

    t0 = time.perf_counter()
    mlp.fit(X_train, train_labels)
    mlp_metrics = mlp.evaluate(X_test, test_labels)
    mlp_time    = time.perf_counter() - t0

    all_results.append({
        "model"    : "5. MLP (256→128)",
        "accuracy" : mlp_metrics["accuracy"],
        "precision": mlp_metrics["precision"],
        "recall"   : mlp_metrics["recall"],
        "f1"       : mlp_metrics["f1"],
        "time"     : mlp_time,
    })
    print_report("MLP (256→128) — Test Results", mlp_metrics)

    # ── Models 6+7: BERT and RAG (Colab results) ──────────────────────────────
    # Check if Colab results were saved
    bert_results_path = os.path.join(RESULTS, "bert_results.json")
    rag_results_path  = os.path.join(RESULTS, "rag_results.json")

    if os.path.exists(bert_results_path):
        with open(bert_results_path) as f:
            bert_r = json.load(f)
        all_results.append({
            "model"    : "6. BERT (fine-tuned)",
            "accuracy" : bert_r.get("accuracy",  0),
            "precision": bert_r.get("precision", 0),
            "recall"   : bert_r.get("recall",    0),
            "f1"       : bert_r.get("f1",        0),
            "time"     : bert_r.get("time",      0),
        })
    else:
        print("  BERT results not found — run bert_classifier.py on Colab")
        all_results.append({
            "model": "6. BERT (fine-tuned)",
            "accuracy": 0, "precision": 0,
            "recall": 0, "f1": 0, "time": 0,
        })

    if os.path.exists(rag_results_path):
        with open(rag_results_path) as f:
            rag_r = json.load(f)
        all_results.append({
            "model"    : "7. BERT + RAG",
            "accuracy" : rag_r.get("accuracy",  0),
            "precision": rag_r.get("precision", 0),
            "recall"   : rag_r.get("recall",    0),
            "f1"       : rag_r.get("f1",        0),
            "time"     : rag_r.get("time",      0),
        })
    else:
        print("  RAG results not found — run rag_classifier.py on Colab")
        all_results.append({
            "model": "7. BERT + RAG",
            "accuracy": 0, "precision": 0,
            "recall": 0, "f1": 0, "time": 0,
        })

    # ── Model 10: BERT + FiLM + RAG (Kaggle results) ──────────────────────────
    film_rag_path = os.path.join(RESULTS, "bert_film_rag_results.json")
    if os.path.exists(film_rag_path):
        with open(film_rag_path) as f:
            fr = json.load(f)
        all_results.append({
            "model"    : "10. BERT + FiLM + RAG",
            "accuracy" : fr.get("accuracy",  0),
            "precision": fr.get("precision", 0),
            "recall"   : fr.get("recall",    0),
            "f1"       : fr.get("f1",        0),
            "time"     : fr.get("time",      0),
        })
    else:
        print("  FiLM+RAG results not found — run bert_film_rag.py on Kaggle")

    # ── Print final table ─────────────────────────────────────────────────────
    print_comparison(all_results)

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS, "comparison_table.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {out_path}")
