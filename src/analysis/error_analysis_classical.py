"""
error_analysis_classical.py
============================
Error analysis for the FIRST-HALF models: Naive Bayes, Perceptron,
and Logistic Regression.

Why these models specifically?
    SVM and MLP (second half) already get the metadata + bigram +
    TF-IDF treatment. NB/Perceptron/LR were built in the first half
    using plain binary Bag-of-Words ONLY — they have never been
    error-analysed with the categorisation lens we now have
    (topic bias, name bias, statement length).

    This script answers: "WHERE do the simplest models fail, and
    is it for the SAME reasons SVM/MLP fail?"

Note on label formats (important — each model is different!):
    Naive Bayes  : dataset = [(text, +1/-1)], predict() -> +1/-1
    Perceptron   : train_data = [(text, +1/-1)], predict() -> +1/-1
    Logistic Reg : train_data = [(text, 0/1)]   (via convert_labels),
                   predict() -> 0/1, predict_proba() -> P(fake)

    We convert everything to 0/1 (1=fake, 0=real) for evaluation,
    matching the convention used by SVM/MLP/BERT.

What this script produces:
    - Accuracy/Precision/Recall/F1 for NB, Perceptron, LR
      (re-confirms first-half numbers as a sanity check)
    - Error rate by topic (LIAR column 3)        -> topic bias
    - Error rate by speaker credibility           -> name bias
    - Error rate by statement length              -> short-context bias
    - Top 5 most-confident WRONG predictions per model
    - 3-way agreement analysis (NB vs Perceptron vs LR)
      -> a sneak preview of how majority voting (ensemble) would do!

Run:
    cd ~/fake-finders-liar/src
    source ../venv/bin/activate
    python error_analysis_classical.py

Takes ~2 minutes (LR training is the slow part, ~100s).

Author  : Akhila Pavithran, Rajana
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import math
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

from data_loader               import map_label
from feature_extractor         import speaker_fake_rate
from classification_evaluator  import ClassificationEvaluator, print_report

from classical import naive_bayes        as nb
from classical import perceptron         as perc
from classical import logistic_regression as lr


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_with_meta(path: str):
    """
    Load a LIAR split, keeping the full row for metadata-based
    error categorisation (subject, speaker credibility).

    Returns:
        texts  : list of statement strings
        labels : list of int, +1=fake / -1=real (NB/Perceptron format)
        rows   : list of raw TSV rows (for metadata)
    """
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
            labels.append(lbl)          # +1 or -1
            rows.append(parts)
    return texts, labels, rows


# ══════════════════════════════════════════════════════════════════════════════
# Error Categorisation Helpers
# (same definitions as error_analysis.py — kept self-contained here)
# ══════════════════════════════════════════════════════════════════════════════

def get_subject(row: list) -> str:
    """Primary subject/topic from LIAR column 3 (first of comma-separated list)."""
    if len(row) > 3 and row[3]:
        return row[3].split(",")[0].strip().lower()
    return "unknown"


def credibility_bucket(row: list) -> str:
    """Bucket speaker by historical fake rate -> tests NAME BIAS."""
    rate = speaker_fake_rate(row)
    if rate < 0.35:
        return "low (reliable speaker)"
    elif rate < 0.65:
        return "medium"
    else:
        return "high (unreliable speaker)"


def length_bucket(text: str) -> str:
    """Bucket by word count -> tests SHORT-CONTEXT hypothesis."""
    n = len(text.split())
    if n <= 8:
        return "short (<=8 words)"
    elif n <= 15:
        return "medium (9-15 words)"
    else:
        return "long (16+ words)"


def error_rate_by_category(texts, labels_01, rows, preds_01, category_fn) -> dict:
    """Group test examples by category_fn(text, row) and compute error rate."""
    stats = defaultdict(lambda: {"total": 0, "errors": 0})
    for text, label, row, pred in zip(texts, labels_01, rows, preds_01):
        cat = category_fn(text, row)
        stats[cat]["total"] += 1
        if pred != label:
            stats[cat]["errors"] += 1
    for s in stats.values():
        s["error_rate"] = s["errors"] / s["total"] if s["total"] else 0.0
    return dict(stats)


def print_category_table(title: str, *named_stats) -> None:
    """
    Print a side-by-side error rate table for multiple models.

    named_stats: tuples of (model_name, stats_dict)
    """
    print(f"\n  --- {title} ---")
    header = f"  {'Bucket':<28}"
    for name, _ in named_stats:
        header += f"{name + ' err%':>14}"
    header += f"{'n':>6}"
    print(header)

    # union of all bucket keys, sorted for stable order
    all_buckets = set()
    for _, stats in named_stats:
        all_buckets.update(stats.keys())

    for bucket in sorted(all_buckets):
        row_str = f"  {bucket:<28}"
        n = 0
        for _, stats in named_stats:
            s = stats.get(bucket, {"error_rate": 0, "total": 0})
            row_str += f"{s['error_rate']*100:>13.1f}%"
            n = s["total"]
        row_str += f"{n:>6}"
        print(row_str)


# ══════════════════════════════════════════════════════════════════════════════
# Confident Errors
# ══════════════════════════════════════════════════════════════════════════════

def show_confident_errors(texts, labels_01, rows, preds_01, scores, model_name: str, n: int = 5) -> list:
    """
    Top-N most confident WRONG predictions.

    `scores` should be a "confidence-like" value where LARGER absolute
    distance from the decision boundary = more confident:
        - NB         : pseudo-probability P(fake) in [0,1] -> |p - 0.5|
        - Perceptron : raw decision score w.x+b (unbounded) -> |score|
        - LR         : P(fake) in [0,1] -> |p - 0.5|
    """
    errors = []
    for text, label, row, pred, sc in zip(texts, labels_01, rows, preds_01, scores):
        if pred != label:
            errors.append((abs(sc), text, label, row, pred, sc))

    errors.sort(key=lambda x: x[0], reverse=True)
    top = errors[:n]

    label_map = {1: "FAKE", 0: "REAL"}
    print(f"\n  Top {n} most confident WRONG predictions — {model_name}:")
    print(f"  {'-' * 56}")

    results = []
    for _, text, label, row, pred, sc in top:
        subject = get_subject(row)
        cred    = credibility_bucket(row)
        print(f"  TEXT    : {text[:90]}")
        print(f"  TRUE    : {label_map[label]}  PRED: {label_map[pred]}  "
              f"score: {sc:.3f}")
        print(f"  SUBJECT : {subject}  |  SPEAKER CREDIBILITY: {cred}")
        print(f"  {'-' * 56}")
        results.append({
            "text": text, "true": label_map[label], "pred": label_map[pred],
            "score": round(sc, 4), "subject": subject, "credibility": cred,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    BASE       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TRAIN_PATH = os.path.join(BASE, "data", "train.tsv")
    TEST_PATH  = os.path.join(BASE, "data", "test.tsv")
    RESULTS    = os.path.join(BASE, "results")
    os.makedirs(RESULTS, exist_ok=True)

    print("=" * 60)
    print("  ERROR ANALYSIS — Classical Models (NB, Perceptron, LR)")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    train_texts, train_labels_pm, train_rows = load_with_meta(TRAIN_PATH)
    test_texts,  test_labels_pm,  test_rows  = load_with_meta(TEST_PATH)

    train_dataset = list(zip(train_texts, train_labels_pm))  # [(text, +1/-1)]
    test_labels_01 = [1 if l == 1 else 0 for l in test_labels_pm]

    print(f"      Train: {len(train_texts)}  Test: {len(test_texts)}")

    # ── 2. Naive Bayes ─────────────────────────────────────────────────────────
    print("\n[2/5] Training Naive Bayes...")
    priors, likelihoods, vocab_nb = nb.train(train_dataset)

    nb_preds, nb_scores = [], []
    for text in test_texts:
        pred = nb.predict(text, priors, likelihoods, vocab_nb)
        s_fake = nb.score(text, +1, priors, likelihoods, vocab_nb)
        s_real = nb.score(text, -1, priors, likelihoods, vocab_nb)
        # convert log-scores to pseudo-probability via softmax
        m = max(s_fake, s_real)
        p_fake = math.exp(s_fake - m) / (math.exp(s_fake - m) + math.exp(s_real - m))

        nb_preds.append(1 if pred == 1 else 0)
        nb_scores.append(p_fake - 0.5)  # signed distance from 0.5

    # ── 3. Perceptron ──────────────────────────────────────────────────────────
    print("\n[3/5] Training Perceptron...")
    vocab_p   = perc.build_vocab(train_dataset)
    w_p, b_p  = perc.train(train_dataset, vocab_p, epochs=10)

    perc_preds, perc_scores = [], []
    for text in test_texts:
        x    = perc.vectorize(text, vocab_p)
        raw  = sum(wi * xi for wi, xi in zip(w_p, x)) + b_p
        pred = perc.predict(x, w_p, b_p)

        perc_preds.append(1 if pred == 1 else 0)
        perc_scores.append(raw)  # unbounded raw decision score

    # ── 4. Logistic Regression ────────────────────────────────────────────────
    print("\n[4/5] Training Logistic Regression...")
    vocab_lr  = lr.build_vocab(train_dataset)
    train_01  = lr.convert_labels(train_dataset)
    w_lr, b_lr = lr.train(train_01, vocab_lr, epochs=20, lr=0.1)

    lr_preds, lr_scores = [], []
    for text in test_texts:
        x     = lr.vectorize(text, vocab_lr)
        proba = lr.predict_proba(x, w_lr, b_lr)
        pred  = lr.predict(x, w_lr, b_lr)

        lr_preds.append(pred)
        lr_scores.append(proba - 0.5)  # signed distance from 0.5

    # ── 5. Evaluate + analyse ──────────────────────────────────────────────────
    print("\n[5/5] Evaluating and categorising errors...\n")

    models = {
        "Naive Bayes":  (nb_preds,   nb_scores),
        "Perceptron":   (perc_preds, perc_scores),
        "Logistic Reg": (lr_preds,   lr_scores),
    }

    metrics_summary = {}
    for name, (preds, scores) in models.items():
        ev = ClassificationEvaluator(test_labels_01, preds)
        report = ev.binary_report()
        print_report(f"{name} — Test Results", report)
        metrics_summary[name] = report

    # ── Error categorisation tables ───────────────────────────────────────────
    by_subject = {}
    by_cred    = {}
    by_length  = {}

    for name, (preds, _) in models.items():
        by_subject[name] = error_rate_by_category(
            test_texts, test_labels_01, test_rows, preds,
            category_fn=lambda t, r: get_subject(r)
        )
        by_cred[name] = error_rate_by_category(
            test_texts, test_labels_01, test_rows, preds,
            category_fn=lambda t, r: credibility_bucket(r)
        )
        by_length[name] = error_rate_by_category(
            test_texts, test_labels_01, test_rows, preds,
            category_fn=lambda t, r: length_bucket(t)
        )

    print_category_table(
        "Error rate by SPEAKER CREDIBILITY (name bias check)",
        ("NB", by_cred["Naive Bayes"]),
        ("Perc", by_cred["Perceptron"]),
        ("LR", by_cred["Logistic Reg"]),
    )

    print_category_table(
        "Error rate by STATEMENT LENGTH (short-context check)",
        ("NB", by_length["Naive Bayes"]),
        ("Perc", by_length["Perceptron"]),
        ("LR", by_length["Logistic Reg"]),
    )

    # Top-5 highest-error subjects for each model (only subjects with >=5 samples)
    print("\n  --- Top 5 highest-error SUBJECTS per model (topic bias check) ---")
    top_subjects = {}
    for name in models:
        sorted_subj = sorted(
            [(k, v) for k, v in by_subject[name].items() if v["total"] >= 5],
            key=lambda kv: kv[1]["error_rate"], reverse=True
        )[:5]
        top_subjects[name] = sorted_subj
        print(f"\n  {name}:")
        print(f"  {'Subject':<25} {'Error Rate':>10} {'n':>6}")
        for subj, v in sorted_subj:
            print(f"  {subj:<25} {v['error_rate']*100:>9.1f}% {v['total']:>6}")

    # ── Confident errors per model ────────────────────────────────────────────
    confident_errors = {}
    for name, (preds, scores) in models.items():
        confident_errors[name] = show_confident_errors(
            test_texts, test_labels_01, test_rows, preds, scores, name, n=5
        )

    # ── 3-way agreement analysis (preview of ensemble!) ───────────────────────
    print(f"\n{'=' * 60}")
    print("  3-WAY AGREEMENT — NB vs Perceptron vs LR")
    print("  (preview of majority-vote ensemble)")
    print(f"{'=' * 60}")

    all_correct   = 0  # all 3 right
    all_wrong     = 0  # all 3 wrong -> hardest cases
    majority_vote_preds = []

    for i, label in enumerate(test_labels_01):
        votes  = [nb_preds[i], perc_preds[i], lr_preds[i]]
        n_correct = sum(1 for v in votes if v == label)

        if n_correct == 3:
            all_correct += 1
        elif n_correct == 0:
            all_wrong += 1

        # majority vote: sum of votes >= 2 -> predict 1 (fake)
        majority_vote_preds.append(1 if sum(votes) >= 2 else 0)

    total = len(test_labels_01)
    print(f"  All 3 correct  : {all_correct:4d}  ({100*all_correct/total:.1f}%)")
    print(f"  All 3 wrong     : {all_wrong:4d}  ({100*all_wrong/total:.1f}%)  <- hardest cases")

    # quick majority-vote F1 preview
    mv_ev     = ClassificationEvaluator(test_labels_01, majority_vote_preds)
    mv_report = mv_ev.binary_report()
    print(f"\n  Majority vote (NB+Perceptron+LR) F1: {mv_report['f1']:.4f}")
    print(f"  (vs individual best of these 3: "
          f"{max(metrics_summary[m]['f1'] for m in models):.4f})")
    print(f"  -> full ensemble with SVM/MLP/BERT will be done in next step!")
    print(f"{'=' * 60}")

    # ── Save everything ────────────────────────────────────────────────────────
    summary = {
        "metrics": metrics_summary,
        "error_by_credibility": by_cred,
        "error_by_length": by_length,
        "top_subjects": {k: dict(v) for k, v in top_subjects.items()},
        "confident_errors": confident_errors,
        "three_way_agreement": {
            "all_correct": all_correct,
            "all_wrong": all_wrong,
            "total": total,
            "majority_vote_f1": round(mv_report["f1"], 4),
        },
    }
    out_path = os.path.join(RESULTS, "error_analysis_classical.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {out_path}")