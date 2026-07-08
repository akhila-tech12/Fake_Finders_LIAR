"""
feature_extractor.py
====================
Shared feature extraction utilities used by ALL models.

Responsibilities:
    - Tokenization with n-gram support
    - TF-IDF vectorisation
    - LIAR metadata feature computation
    - Wikipedia evidence retrieval

Design principles:
    - No model logic here — pure feature engineering
    - All models import from this single source of truth
    - Stateless functions where possible
    - TFIDFVectorizer is stateful (fit on train, transform test)

Author  : Akhila Pavithran, Rajana
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import re
import os
import sys
import math
import json
import urllib.request
import urllib.parse
from typing import Optional

# ── stdlib only — no external deps in this file ──────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

STOPWORDS: frozenset = frozenset({
    "the","a","an","is","are","was","were","will","has","have",
    "had","be","been","to","of","in","on","at","for","and","or",
    "but","not","with","by","from","that","this","it","he","she",
    "they","we","said","says","would","could","should","also",
    "just","very","so","then","than","more","about","into","up",
})

WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
USER_AGENT    = "FakeFinders-NLP/1.0 (Bamberg University)"


# ══════════════════════════════════════════════════════════════════════════════
# Tokenisation
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str, ngram_range: tuple[int, int] = (1, 1)) -> list[str]:
    """
    Convert raw text to a list of clean tokens with optional n-grams.

    Pipeline:
        1. Lowercase
        2. Remove non-alphabetic characters
        3. Split on whitespace
        4. Generate n-grams for each n in [min_n, max_n]

    Args:
        text        : Raw input string.
        ngram_range : (min_n, max_n). (1,1)=unigrams, (1,2)=uni+bigrams.

    Returns:
        Flat list of token strings.

    Examples:
        tokenize("Free Money!")            → ["free", "money"]
        tokenize("will not raise", (1,2))  → ["will","not","raise",
                                               "will not","not raise"]

    Why bigrams?
        "not raise taxes" with unigrams → {not, raise, taxes}
        "not raise taxes" with bigrams  → {not, raise, taxes, not raise,
                                           raise taxes}
        "not raise" is now ONE feature → model detects negation!
    """
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())
    words   = cleaned.split()

    tokens: list[str] = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        tokens.extend(
            " ".join(words[i : i + n])
            for i in range(len(words) - n + 1)
        )
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF Vectoriser
# ══════════════════════════════════════════════════════════════════════════════

class TFIDFVectorizer:
    """
    TF-IDF vectoriser — fit on training data, transform any split.

    Why TF-IDF over binary BOW?
        Binary BOW treats every word identically (1 or 0).
        TF-IDF weights words by how informative they are:
            "the"         → near 0   (appears everywhere, useless)
            "conspiracy"  → high     (rare, informative)
            "pants-fire"  → very high (domain-specific signal)

    Formulae:
        TF(w, doc)  = count(w, doc) / |doc|
        IDF(w)      = log(N / df(w) + 1)   — +1 prevents log(0)
        TF-IDF(w)   = TF(w, doc) × IDF(w)

    Usage:
        vec = TFIDFVectorizer(max_features=10_000, ngram_range=(1,2))
        vec.fit(train_texts)                # fit ONCE on train only!
        X_train = [vec.transform(t) for t in train_texts]
        X_test  = [vec.transform(t) for t in test_texts]
    """

    def __init__(
        self,
        max_features: int        = 10_000,
        ngram_range:  tuple      = (1, 2),
        min_df:       int        = 2,
    ) -> None:
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.min_df       = min_df
        self.vocab_:      dict[str, int]   = {}
        self.idf_:        dict[str, float] = {}
        self._fitted      = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "TFIDFVectorizer":
        """
        Learn vocabulary and IDF weights from a corpus.

        IMPORTANT: call this ONLY on training texts.
        Fitting on test data would be data leakage.

        Args:
            texts : list of raw training strings.

        Returns:
            self (for chaining)
        """
        N          = len(texts)
        doc_freq:  dict[str, int] = {}

        for text in texts:
            seen = set(tokenize(text, self.ngram_range))
            for token in seen:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Compute IDF; filter by min_df
        idf_all = {
            token: math.log(N / (df + 1))
            for token, df in doc_freq.items()
            if df >= self.min_df
        }

        # Keep top-k by IDF (highest IDF = most informative)
        top_tokens = sorted(
            idf_all.items(), key=lambda kv: kv[1], reverse=True
        )[: self.max_features]

        self.vocab_ = {tok: i for i, (tok, _)  in enumerate(top_tokens)}
        self.idf_   = {tok: sc for tok, sc      in top_tokens}
        self._fitted = True
        return self

    # ── transform ────────────────────────────────────────────────────────────

    def transform(self, text: str) -> list[float]:
        """
        Convert a single text to a TF-IDF vector.

        Args:
            text : raw string.

        Returns:
            Dense TF-IDF vector of length len(vocab_).

        Raises:
            RuntimeError if called before fit().
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        tokens    = tokenize(text, self.ngram_range)
        n_tokens  = max(len(tokens), 1)
        vector    = [0.0] * len(self.vocab_)

        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1

        for tok, cnt in counts.items():
            if tok in self.vocab_:
                tf                          = cnt / n_tokens
                vector[self.vocab_[tok]]    = tf * self.idf_[tok]

        return vector

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        return len(self.vocab_)

    def __repr__(self) -> str:
        status = f"{self.n_features:,} features" if self._fitted else "unfitted"
        return (
            f"TFIDFVectorizer("
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, "
            f"status={status})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# LIAR Metadata Features
# ══════════════════════════════════════════════════════════════════════════════

def speaker_fake_rate(row: list[str]) -> float:
    """
    Compute a speaker's historical dishonesty rate from LIAR metadata.

    LIAR column schema:
        row[8]  = barely_true_count
        row[9]  = false_count
        row[10] = half_true_count
        row[11] = mostly_true_count
        row[12] = pants_on_fire_count

    fake_rate = (barely_true + false + pants_fire) / total_statements

    Why this matters (directly from Prof Weber's feedback):
        "Name bias cannot be fixed by better word representations.
         The model needs verified information beyond the text."
        → Speaker credibility IS verified external information.

    Args:
        row : raw TSV columns from LIAR dataset.

    Returns:
        float in [0.0, 1.0]. 0.5 if speaker history unknown.
    """
    try:
        barely = int(row[8]  or 0)
        false_ = int(row[9]  or 0)
        half   = int(row[10] or 0)
        mostly = int(row[11] or 0)
        pants  = int(row[12] or 0)
    except (IndexError, ValueError):
        return 0.5

    total = barely + false_ + half + mostly + pants
    if total == 0:
        return 0.5

    return round((barely + false_ + pants) / total, 4)


def metadata_features(row: list[str]) -> list[float]:
    """
    Extract three numeric metadata features from a LIAR row.

    Features:
        [0] speaker_fake_rate  — historical dishonesty [0, 1]
        [1] party_code         — 0.0=dem, 1.0=rep, 0.5=other
        [2] experience         — normalised statement count [0, 1]

    Args:
        row : raw TSV columns.

    Returns:
        list of 3 floats.
    """
    fake_rate = speaker_fake_rate(row)

    party = (row[7] if len(row) > 7 else "").strip().lower()
    if   "democrat"   in party: party_code = 0.0
    elif "republican" in party: party_code = 1.0
    else:                        party_code = 0.5

    try:
        total      = sum(int(row[i] or 0) for i in range(8, 13) if len(row) > i)
        experience = min(total / 100.0, 1.0)
    except (ValueError, IndexError):
        experience = 0.0

    return [fake_rate, party_code, experience]


# ══════════════════════════════════════════════════════════════════════════════
# Wikipedia Evidence Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_wikipedia(query: str, max_chars: int = 400) -> str:
    """Internal: hit Wikipedia REST API with a single query."""
    slug = urllib.parse.quote(query.replace(" ", "_"))
    url  = WIKIPEDIA_API.format(slug)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
            return data.get("extract", "")[:max_chars]
    except Exception:
        return ""


def retrieve_evidence(statement: str, n_keywords: int = 3) -> str:
    """
    Retrieve Wikipedia evidence for a political statement.

    Strategy — cascade from broad to narrow:
        1. Search top-3 keywords
        2. If empty, search top-2
        3. If empty, search top-1

    Args:
        statement   : raw claim text.
        n_keywords  : max keywords to extract (default 3).

    Returns:
        Wikipedia evidence string, or "" if nothing found.

    Example:
        statement = "Unemployment doubled to 20 percent"
        returns   = "Unemployment in the United States was 3.8%..."
        → contradiction gives model a factual signal!
    """
    words    = re.sub(r"[^a-z\s]", " ", statement.lower()).split()
    # Priority words — numbers and specific nouns are most useful
    numbers  = [w for w in words if w.isdigit() or '%' in w]
    content  = [w for w in words if w not in STOPWORDS 
                and len(w) > 4 and not w.isdigit()]
    
    # Combine: content words first, then numbers
    keywords = (content + numbers)[:n_keywords]
    if not keywords:
        return ""

    for k in range(len(keywords), 0, -1):
        result = _fetch_wikipedia(" ".join(keywords[:k]))
        if result:
            return result

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Unified Feature Builder
# ══════════════════════════════════════════════════════════════════════════════

class FeatureBuilder:
    """
    One-stop feature construction for all models.

    Produces a single feature vector per document:
        [TF-IDF features | metadata features | evidence features]

    The metadata and evidence sections are optional:
        - All classical models:  TF-IDF only
        - Metadata-enhanced:     TF-IDF + metadata
        - Evidence-enhanced:     TF-IDF + metadata + evidence TF-IDF

    Usage:
        fb = FeatureBuilder(max_features=8_000)
        fb.fit(train_texts)
        x = fb.transform(text, row, use_evidence=False)
    """

    def __init__(
        self,
        max_features: int   = 8_000,
        ngram_range:  tuple = (1, 2),
    ) -> None:
        self.vectorizer   = TFIDFVectorizer(max_features, ngram_range)
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self._fitted      = False

    def fit(self, texts: list[str]) -> "FeatureBuilder":
        """Fit TF-IDF on training texts ONLY."""
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(
        self,
        text:         str,
        row:          Optional[list[str]] = None,
        use_evidence: bool                = False,
    ) -> list[float]:
        """
        Build feature vector for one example.

        Args:
            text         : raw statement string.
            row          : LIAR TSV row (for metadata). None = skip.
            use_evidence : whether to fetch Wikipedia evidence.
                           Adds latency (~0.5s/sample). Disable for
                           large batches; enable for final evaluation.

        Returns:
            Dense float vector.
        """
        vector = self.vectorizer.transform(text)

        if row is not None:
            vector += metadata_features(row)

        if use_evidence:
            evidence = retrieve_evidence(text)
            if evidence:
                ev_vec = self.vectorizer.transform(evidence)
                base_n = min(len(vector), len(ev_vec))
                vector[:base_n] = [
                    max(a, b)
                    for a, b in zip(vector[:base_n], ev_vec[:base_n])
                ]

        return vector

    @property
    def feature_dim(self) -> int:
        """Dimension of the base TF-IDF vector."""
        return self.vectorizer.n_features


# ══════════════════════════════════════════════════════════════════════════════
# Smoke Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== feature_extractor.py — smoke test ===\n")

    # Tokenisation
    stmt = "I will not raise taxes next year"
    print("Unigrams :", tokenize(stmt, (1, 1)))
    print("Bigrams  :", tokenize(stmt, (1, 2)))

    # TF-IDF
    corpus = [
        "the president will not raise taxes",
        "conspiracy theory about government plot",
        "reuters reported the federal reserve decision",
        "shocking secret about the deep state revealed",
    ]
    vec = TFIDFVectorizer(max_features=20, ngram_range=(1, 2))
    vec.fit(corpus)
    v = vec.transform("president not raise taxes")
    nonzero = [(tok, round(vec.idf_[tok], 3))
               for tok in vec.vocab_ if v[vec.vocab_[tok]] > 0][:5]
    print("\nTF-IDF non-zero features:", nonzero)

    # Metadata
    fake_row = ["", "false", "stmt", "", "", "", "", "republican",
                "2", "8", "1", "0", "3"]
    print("\nSpeaker fake rate:", speaker_fake_rate(fake_row))
    print("Metadata features:", metadata_features(fake_row))

    # Evidence
    print("\nSearching Wikipedia for 'unemployment rate'...")
    ev = retrieve_evidence("The unemployment rate doubled to 20 percent")
    print("Evidence:", ev[:150] + "..." if ev else "None found")

    print("\n✓ All checks passed")
