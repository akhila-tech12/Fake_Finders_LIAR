"""
rag_classifier.py
=================
Model 7 — BERT + RAG (Retrieval-Augmented Generation).

This is the most advanced model and directly addresses
Professor Weber's core feedback:

    "Topic and person bias will not be fixed by better word
     representations — the model needs verified information
     BEYOND the text to check what is said against facts."

What RAG adds on top of BERT (Model 6):
    BERT alone:
        Reads statement → contextual understanding → prediction
        Still suffers from name/topic bias learned from training data!

    BERT + RAG:
        Reads statement
        ↓
        Retrieves related VERIFIED FACTS from knowledge base
        (Wikipedia / ChromaDB / Google Fact Check)
        ↓
        Feeds [statement + evidence] to BERT
        ↓
        BERT now has real-world context, not just learned patterns
        ↓
        Better, less biased prediction

Why ChromaDB?
    Normal keyword search: "unemployment" → exact phrase only
    ChromaDB semantic search: "unemployment" → also finds
        "jobless rate", "labour market", "out of work"
    → finds MEANING not just exact words

Three evidence sources (in order of quality):
    1. LIAR metadata     — speaker credibility (verified, best signal)
    2. Wikipedia API     — factual context (free, general)
    3. Google Fact Check — specific verdict on political claims (best
                           but requires API key)

Author  : Akhila Pavithran, Rajana
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import urllib.request
import urllib.parse
from typing import Optional

# ── Dependency guards ─────────────────────────────────────────────────────────
try:
    import torch
    import numpy as np
    from transformers import BertTokenizerFast, BertForSequenceClassification
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

from classification_evaluator import ClassificationEvaluator, print_report
from feature_extractor        import retrieve_evidence, speaker_fake_rate

# ── LIAR label helpers ────────────────────────────────────────────────────────
_FAKE = frozenset({"pants-fire", "false", "barely-true"})
_REAL = frozenset({"mostly-true", "true"})


def _map_label(raw: str) -> Optional[int]:
    raw = raw.strip().lower()
    if raw in _FAKE: return 1
    if raw in _REAL: return 0
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Knowledge Base — ChromaDB vector store
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeBase:
    """
    Semantic vector store of verified facts.

    Uses ChromaDB + sentence-transformers to store and retrieve
    factual statements by MEANING, not just keywords.

    Example:
        kb.add(["US unemployment was 3.8% in 2024"])
        kb.retrieve("jobless rate doubled") 
        → returns the unemployment fact (semantic match!)

    Install: pip install chromadb sentence-transformers
    """

    _EMBED_MODEL = "all-MiniLM-L6-v2"   # 80MB, fast, good quality

    def __init__(self, collection: str = "verified_facts") -> None:
        if not _CHROMA_OK:
            raise ImportError(
                "Run: pip install chromadb sentence-transformers"
            )
        self._embedder   = SentenceTransformer(self._EMBED_MODEL)
        self._client     = chromadb.Client()
        try:
            self._client.delete_collection(collection)
        except Exception:
            pass
        self._collection = self._client.create_collection(collection)
        self._n_facts    = 0

    def add(self, facts: list[str]) -> None:
        """Embed and store a list of verified fact strings."""
        if not facts:
            return
        embeddings = self._embedder.encode(facts).tolist()
        ids        = [f"f{self._n_facts + i}" for i in range(len(facts))]
        self._collection.add(
            documents  = facts,
            embeddings = embeddings,
            ids        = ids,
        )
        self._n_facts += len(facts)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """
        Return the k most semantically similar facts to query.

        Args:
            query : the statement to find evidence for.
            k     : number of facts to return.

        Returns:
            list of fact strings (may be empty if KB is empty).
        """
        if self._n_facts == 0:
            return []
        q_emb   = self._embedder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings = q_emb,
            n_results        = min(k, self._n_facts),
        )
        docs = results.get("documents", [[]])
        return docs[0] if docs else []

    def populate_from_wikipedia(self, topics: list[str]) -> None:
        """Auto-populate KB by fetching Wikipedia summaries."""
        facts = []
        for topic in topics:
            text = retrieve_evidence(topic)
            if text:
                facts.append(text)
                print(f"    + {topic[:50]}")
        if facts:
            self.add(facts)
            print(f"  KB: {self._n_facts} facts stored\n")

    def __len__(self) -> int:
        return self._n_facts

    def __repr__(self) -> str:
        return f"KnowledgeBase({self._n_facts} facts)"


# ══════════════════════════════════════════════════════════════════════════════
# Google Fact Check API
# ══════════════════════════════════════════════════════════════════════════════

class FactCheckAPI:
    """
    Google Fact Check Tools API wrapper.

    Free for academic use. Returns verdicts from PolitiFact, Snopes, etc.
    Get key: https://developers.google.com/fact-check/tools/api

    Verdict → signal mapping:
        False / Misleading → +1  (supports fake prediction)
        True  / Accurate   → -1  (supports real prediction)
        Unknown            →  0  (no signal)
    """

    _ENDPOINT = (
        "https://factchecktools.googleapis.com"
        "/v1alpha1/claims:search"
    )
    _FAKE_WORDS = frozenset({
        "false", "fake", "wrong", "incorrect",
        "pants", "misleading", "inaccurate", "mostly false",
    })
    _REAL_WORDS = frozenset({
        "true", "correct", "accurate", "mostly true", "largely true",
    })

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key or os.environ.get(
            "GOOGLE_FACTCHECK_KEY", ""
        )

    def signal(self, claim: str) -> int:
        """
        Query the API and return a numeric signal.

        Returns:
             1  → fact-checkers say claim is FALSE (fake signal)
            -1  → fact-checkers say claim is TRUE  (real signal)
             0  → no result / uncertain
        """
        if not self.api_key:
            return 0

        encoded = urllib.parse.quote(claim)
        url     = (f"{self._ENDPOINT}"
                   f"?query={encoded}&key={self.api_key}"
                   f"&languageCode=en")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
                for c in data.get("claims", [])[:3]:
                    for review in c.get("claimReview", []):
                        rating = review.get("textualRating", "").lower()
                        if any(w in rating for w in self._FAKE_WORDS):
                            return 1
                        if any(w in rating for w in self._REAL_WORDS):
                            return -1
        except Exception:
            pass
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# RAG Classifier
# ══════════════════════════════════════════════════════════════════════════════

class RAGClassifier:
    """
    BERT + RAG fake-news classifier.

    Prediction pipeline for each statement:
        1. Retrieve Wikipedia evidence (keyword search)
        2. Retrieve ChromaDB facts (semantic search)
        3. Check speaker fake rate from LIAR metadata
        4. Optionally query Google Fact Check API
        5. Combine: [CLS] statement [SEP] evidence [SEP]
        6. Feed combined text to BERT → probability
        7. Apply metadata/fact-check override if confident

    The combination step is the key insight:
        BERT's attention now covers both the claim AND the evidence.
        When a claim contradicts the evidence, BERT's attention
        mechanism will capture this mismatch → better prediction.
    """

    def __init__(
        self,
        bert_dir:       str,               # path to saved fine-tuned BERT
        google_api_key: str   = "",
        use_chromadb:   bool  = True,
        max_len:        int   = 256,       # longer to fit evidence too
        device:         Optional[str] = None,
    ) -> None:
        if not _TORCH_OK:
            raise ImportError("pip install transformers torch")

        self.max_len   = max_len
        self.device    = torch.device(
            device if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Load fine-tuned BERT ──────────────────────────────────────────────
        print(f"  Loading BERT from {bert_dir}...")
        self._tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
        self._model     = BertForSequenceClassification.from_pretrained(
            bert_dir
        ).to(self.device)
        self._model.eval()

        # ── Evidence components ───────────────────────────────────────────────
        self._fact_check = FactCheckAPI(google_api_key)

        self._kb: Optional[KnowledgeBase] = None
        if use_chromadb and _CHROMA_OK:
            self._kb = KnowledgeBase()
            self._populate_kb()
        else:
            print("  ChromaDB disabled — using Wikipedia only")

    def _populate_kb(self) -> None:
        """Seed knowledge base with common political topics."""
        print("  Populating knowledge base...")
        topics = [
            "United States unemployment rate statistics",
            "United States crime rate FBI statistics",
            "United States federal budget spending",
            "United States healthcare cost",
            "United States military budget",
            "United States national debt",
            "United States immigration statistics",
            "United States tax policy",
        ]
        self._kb.populate_from_wikipedia(topics)

    # ── evidence retrieval ───────────────────────────────────────────────────

    def _get_evidence(self, statement: str) -> str:
        """
        Gather all available evidence for a statement.

        Combines Wikipedia + ChromaDB into one evidence string.
        """
        parts: list[str] = []

        # Wikipedia
        wiki = retrieve_evidence(statement)
        if wiki:
            parts.append(wiki)

        # ChromaDB
        if self._kb:
            kb_facts = self._kb.retrieve(statement, k=2)
            parts.extend(kb_facts)

        return " ".join(parts)[:400]   # cap at 400 chars for BERT window

    # ── BERT forward pass ─────────────────────────────────────────────────────

    def _bert_predict(self, statement: str, evidence: str) -> float:
        """
        Run BERT on [statement + evidence] and return P(fake).

        The input format follows the BERT NSP convention:
            [CLS] statement [SEP] evidence [SEP]

        BERT's attention now spans BOTH the claim and the evidence.
        Contradictions between them are captured in the attention matrix.
        """
        combined = (statement + " [SEP] " + evidence
                    if evidence else statement)

        enc = self._tokenizer(
            combined,
            truncation    = True,
            padding       = "max_length",
            max_length    = self.max_len,
            return_tensors= "pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self._model(**enc).logits
            probs  = torch.softmax(logits, dim=1)

        return float(probs[0][1].item())   # P(fake)

    # ── public interface ──────────────────────────────────────────────────────

    def predict_proba(self, statement: str,
                      row: Optional[list[str]] = None) -> float:
        """
        Full RAG pipeline — return P(fake).

        Args:
            statement : raw claim text.
            row       : LIAR TSV row for metadata (optional).

        Returns:
            float in [0, 1].
        """
        # 1. Retrieve evidence
        evidence = self._get_evidence(statement)

        # 2. BERT prediction with evidence
        prob = self._bert_predict(statement, evidence)

        # 3. Speaker credibility adjustment
        if row is not None:
            fake_rate = speaker_fake_rate(row)
            if fake_rate > 0.80:       # highly unreliable speaker
                prob = min(prob + 0.15, 0.99)
            elif fake_rate < 0.20:     # highly reliable speaker
                prob = max(prob - 0.10, 0.01)

        # 4. Fact-check override (only if very confident)
        fc_signal = self._fact_check.signal(statement)
        if fc_signal == 1 and prob < 0.4:
            prob = 0.65   # fact-check says fake, BERT uncertain → lean fake
        elif fc_signal == -1 and prob > 0.6:
            prob = 0.35   # fact-check says real, BERT uncertain → lean real

        return prob

    def predict(self, statement: str,
                row: Optional[list[str]] = None) -> int:
        """Return class label (1=fake, 0=real)."""
        return int(self.predict_proba(statement, row) >= 0.5)

    def evaluate(self, test_path: str) -> dict:
        """Evaluate on LIAR test split."""
        print(f"\n  Evaluating RAG on {os.path.basename(test_path)}...")

        y_true: list[int] = []
        y_pred: list[int] = []

        with open(test_path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]

        for i, line in enumerate(lines, 1):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lbl = _map_label(parts[1])
            if lbl is None:
                continue

            text = parts[2].strip()
            pred = self.predict(text, row=parts)

            y_true.append(lbl)
            y_pred.append(pred)

            if i % 100 == 0:
                print(f"    {i}/{len(lines)} processed...")

        return ClassificationEvaluator(y_true, y_pred).binary_report()

    def __repr__(self) -> str:
        kb_size = len(self._kb) if self._kb else 0
        return (f"RAGClassifier("
                f"bert={BertForSequenceClassification.__name__}, "
                f"kb_facts={kb_size}, "
                f"fact_check={'on' if self._fact_check.api_key else 'off'})")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if not _TORCH_OK:
        print("Install: pip install transformers torch")
        sys.exit(1)

    BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST      = os.path.join(BASE, "data", "test.tsv")
    BERT_DIR  = os.path.join(BASE, "results", "bert_model")

    if not os.path.exists(BERT_DIR):
        print(f"ERROR: Fine-tuned BERT not found at {BERT_DIR}")
        print("Run bert_classifier.py first to fine-tune and save BERT.")
        sys.exit(1)

    print("=== RAG Classifier — LIAR Dataset ===\n")

    clf     = RAGClassifier(bert_dir=BERT_DIR, use_chromadb=_CHROMA_OK)
    results = clf.evaluate(TEST)
    print_report("BERT + RAG — Test Results", results)

    # ── Demo ──────────────────────────────────────────────────────────────────
    print("\n=== Example Predictions with Evidence ===\n")
    demos = [
        "The unemployment rate has doubled to 20 percent this year.",
        "Hillary Clinton agrees with John McCain on Iran.",
        "Building a wall on the U.S.-Mexico border will take literally years.",
        "The crime rate in Chicago has increased 300 percent.",
    ]

    for stmt in demos:
        evidence = clf._get_evidence(stmt)
        prob     = clf.predict_proba(stmt)
        label    = "FAKE" if prob >= 0.5 else "REAL"
        print(f"  Statement : {stmt[:80]}")
        print(f"  Evidence  : {evidence[:120]}...")
        print(f"  Prediction: {label} (P(fake)={prob:.0%})")
        print()

    print(clf)
