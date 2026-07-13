"""
build_evidence.py
=================
Precompute Wikipedia evidence for every binary-mappable LIAR statement.

Produces ``data/evidence_all.json``:

    {"<liar_id>": "<evidence string>", ...}

covering train + valid + test (half-true rows skipped, matching
``data_loader.map_label``). The file serves two purposes:

    1. Phase 1 — fast RAG evaluation: rag_classifier.py reads evidence by id
       instead of hitting Wikipedia per example (3600s → minutes).
    2. Phase 2 — uploaded to Kaggle next to the TSVs so the BERT+FiLM+RAG
       training script (bert_film_rag.py) never needs network access.

Retrieval uses the semantic (MiniLM) reranker, so run this in venv_bert:

    venv_bert/bin/python src/bert/build_evidence.py --split valid --report
    venv_bert/bin/python src/bert/build_evidence.py            # all splits

Everything is cached in data/evidence_cache_v2.json — interrupted runs
resume where they stopped.

Author  : Akhila Pavithran, Rajana
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

from data_loader import map_label
from evidence_retriever import EvidenceRetriever

BASE        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(BASE, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "evidence_all.json")

SPLITS = ("train", "valid", "test")


def load_rows(split: str) -> list[list[str]]:
    """Raw TSV rows for one split, half-true rows dropped."""
    path = os.path.join(DATA_DIR, f"{split}.tsv")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if map_label(parts[1]) is None:
                continue
            rows.append(parts)
    return rows


def report(retriever: EvidenceRetriever, rows: list[list[str]],
           results: dict[str, str], n_samples: int = 10) -> None:
    """Print intrinsic retrieval stats + sampled statement→evidence pairs."""
    n         = len(results)
    non_empty = [ev for ev in results.values() if ev]
    lengths   = [len(ev) for ev in non_empty]

    print("\n── Intrinsic retrieval report ─────────────────────────────")
    print(f"  statements        : {n}")
    print(f"  non-empty evidence: {len(non_empty)}  ({100 * len(non_empty) / max(n, 1):.1f}%)")
    if lengths:
        print(f"  mean length       : {sum(lengths) / len(lengths):.0f} chars")
        print(f"  min/max length    : {min(lengths)}/{max(lengths)}")

    print(f"\n── {n_samples} sampled statement → evidence pairs ──────────────")
    random.seed(42)
    for row in random.sample(rows, min(n_samples, len(rows))):
        ev = results.get(row[0].strip(), "")
        print(f"\n  [{row[0]}] {row[2][:100]}")
        print(f"      → {ev[:180] if ev else '(no evidence)'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--split", choices=SPLITS, default=None,
                        help="single split (default: all three)")
    parser.add_argument("--report", action="store_true",
                        help="print intrinsic stats + sampled pairs")
    parser.add_argument("--reranker", default="auto",
                        choices=("auto", "semantic", "lexical"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    splits = [args.split] if args.split else list(SPLITS)

    print(f"=== build_evidence — splits: {', '.join(splits)} ===\n")
    retriever = EvidenceRetriever(reranker=args.reranker)
    print(f"  {retriever!r}\n")

    # Merge into any existing output so single-split runs are additive.
    evidence_all: dict[str, str] = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as fh:
            evidence_all = json.load(fh)
        print(f"  Existing {os.path.basename(OUTPUT_PATH)}: {len(evidence_all)} entries\n")

    for split in splits:
        rows = load_rows(split)
        print(f"  {split}: {len(rows)} statements — retrieving "
              f"({args.workers} workers)…")
        t0      = time.time()
        results = retriever.retrieve_batch(rows, max_workers=args.workers)
        elapsed = time.time() - t0
        hits    = sum(1 for ev in results.values() if ev)
        print(f"  {split}: done in {elapsed:.0f}s — "
              f"{hits}/{len(results)} with evidence "
              f"({100 * hits / max(len(results), 1):.1f}%)\n")
        evidence_all.update(results)

        if args.report:
            report(retriever, rows, results)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(evidence_all, fh, indent=1)
    print(f"✓ wrote {len(evidence_all)} entries → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
