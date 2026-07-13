"""
evidence_retriever.py
=====================
Improved Wikipedia evidence retrieval (v2) for RAG models.

Why a new module? The legacy ``feature_extractor.retrieve_evidence`` guesses a
Wikipedia page *title* from the statement's keywords, which frequently lands on
disambiguation pages ("Romney may refer to:") or generic articles. This module
replaces title-guessing with a proper pipeline:

    LIAR row (id, statement, subject, speaker)
        ↓
    1. Query formation — entity spans + content words + speaker/subject context
    2. MediaWiki full-text search  → top-k candidate page titles
    3. REST summary fetch per title, discarding disambiguation pages
    4. Rerank candidates against the statement (semantic or lexical)
    5. Assemble top passages, truncated at sentence boundaries (~900 chars)
        ↓
    evidence string, cached in data/evidence_cache_v2.json

Rerankers (pluggable):
    SemanticReranker — MiniLM (all-MiniLM-L6-v2) mean-pooling cosine, built on
        raw ``transformers`` so it runs in venv_bert without new dependencies.
    LexicalReranker  — stdlib IDF-weighted token-overlap cosine; used by the
        Streamlit app (venv/, no torch in-process).

Leakage guard: the retriever reads ONLY columns 0 (id), 2 (statement),
3 (subject) and 4 (speaker) of a LIAR row — never the label or the speaker
credit-history counts.

feature_extractor.py stays stdlib-only and untouched; this module lives in
src/bert/ because its best reranker needs venv_bert.

Author  : Akhila Pavithran, Rajana
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import threading
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

from feature_extractor import STOPWORDS, USER_AGENT, tokenize

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# One call returns the top-k search hits WITH their intro extracts and a
# disambiguation marker (pageprops) — 1 request per statement instead of
# 1 search + k summary fetches. Essential: Wikimedia 429-throttles bursts.
SEARCH_API = ("https://en.wikipedia.org/w/api.php"
              "?action=query&format=json&generator=search"
              "&gsrlimit={k}&gsrsearch={q}"
              "&prop=extracts|pageprops&exintro=1&explaintext=1&exlimit=max")

_BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_V2_PATH  = os.path.join(_BASE_DIR, "data", "evidence_cache_v2.json")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ══════════════════════════════════════════════════════════════════════════════
# HTTP fetch (injectable for tests)
# ══════════════════════════════════════════════════════════════════════════════

# Global spacing between API requests, shared across threads. Wikimedia
# 429-throttles unauthenticated clients IP-wide (all API families at once)
# and blocks repeat offenders for a while — observed even at ~3 req/s after
# an earlier burst. Stay at ~1 req/s and honour Retry-After.
_rate_lock    = threading.Lock()
_last_request = [0.0]
MIN_INTERVAL  = 1.0           # seconds between requests (Wikimedia etiquette)


def _rate_gate() -> None:
    with _rate_lock:
        wait = _last_request[0] + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()


class FetchError(Exception):
    """Network/HTTP failure — the caller must NOT cache the miss."""


def default_fetch(url: str, timeout: float = 8.0, retries: int = 2) -> bytes:
    """
    GET ``url`` and return the raw response body.

    Globally rate-limited (MIN_INTERVAL between requests across threads);
    retries honour Retry-After on HTTP 429 and back off briefly on 5xx.
    Raises FetchError on final failure so callers can distinguish
    "Wikipedia has nothing" (cacheable) from "the request failed" (not).
    """
    for attempt in range(retries + 1):
        _rate_gate()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code == 429 and attempt < retries:
                retry_after = err.headers.get("Retry-After", "")
                delay = (float(retry_after) if retry_after.isdigit()
                         else 30.0 * (attempt + 1))
                time.sleep(min(delay, 120.0))
                continue
            if err.code in (500, 502, 503, 504) and attempt < retries:
                time.sleep(1.0 + attempt)
                continue
            raise FetchError(f"HTTP {err.code} for {url}") from err
        except Exception as err:            # timeout, connection reset, …
            if attempt < retries:
                time.sleep(0.5)
                continue
            raise FetchError(str(err)) from err
    raise FetchError(f"retries exhausted for {url}")


# ══════════════════════════════════════════════════════════════════════════════
# Rerankers
# ══════════════════════════════════════════════════════════════════════════════

class LexicalReranker:
    """
    IDF-weighted token-overlap cosine — stdlib only.

    IDF is computed over the candidate passages themselves (plus the
    statement), so tokens shared by every candidate ("United", "States")
    contribute little while distinctive overlaps dominate.
    """

    name = "lexical"

    def scores(self, statement: str, passages: list[str]) -> list[float]:
        if not passages:
            return []

        docs      = [set(tokenize(p)) for p in passages]
        stmt_toks = set(tokenize(statement))
        corpus    = docs + [stmt_toks]
        n_docs    = len(corpus)

        vocab = set().union(*corpus)
        idf   = {}
        for tok in vocab:
            df       = sum(1 for d in corpus if tok in d)
            idf[tok] = math.log(n_docs / df) + 1.0

        def _norm(toks: set[str]) -> float:
            return math.sqrt(sum(idf[t] ** 2 for t in toks)) or 1.0

        stmt_norm = _norm(stmt_toks)
        out = []
        for doc in docs:
            shared = stmt_toks & doc
            dot    = sum(idf[t] ** 2 for t in shared)
            out.append(dot / (stmt_norm * _norm(doc)))
        return out


class SemanticReranker:
    """
    MiniLM sentence-embedding cosine similarity.

    Uses ``sentence-transformers/all-MiniLM-L6-v2`` through raw
    ``transformers`` (AutoModel + mean pooling) — no sentence-transformers
    package needed, so it works with the pinned stack in venv_bert.
    """

    name  = "semantic"
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        import torch                                   # noqa: F401 — venv_bert only
        from transformers import AutoModel, AutoTokenizer

        self._torch     = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self._model     = AutoModel.from_pretrained(self.MODEL)
        self._model.eval()

    def _embed(self, texts: list[str]):
        torch = self._torch
        enc   = self._tokenizer(texts, padding=True, truncation=True,
                                max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = self._model(**enc).last_hidden_state       # (B, T, 384)
        mask   = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        emb    = summed / counts                             # mean pooling
        return emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)

    def scores(self, statement: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        emb  = self._embed([statement] + passages)
        sims = emb[1:] @ emb[0]
        return [float(s) for s in sims]


def get_reranker(kind: str = "auto"):
    """
    Factory: "semantic" | "lexical" | "auto".

    "auto" tries the MiniLM reranker (needs torch — venv_bert) and silently
    falls back to the stdlib lexical one (venv/, Streamlit app).
    """
    if kind == "lexical":
        return LexicalReranker()
    if kind == "semantic":
        return SemanticReranker()
    if kind == "auto":
        try:
            return SemanticReranker()
        except Exception:
            return LexicalReranker()
    raise ValueError(f"Unknown reranker kind: {kind!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Query formation
# ══════════════════════════════════════════════════════════════════════════════

def _entity_phrases(statement: str) -> list[str]:
    """
    Extract capitalized spans from the original-case statement.

    A run of capitalized tokens forms one entity phrase ("John McCain").
    A sentence-initial capitalized token only counts when the following
    token is capitalized too ("Barack Obama said…" keeps "Barack Obama";
    "Building a wall…" drops "Building").
    """
    phrases: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(statement):
        words = sentence.split()
        run: list[str] = []
        for i, word in enumerate(words):
            core = word.strip(".,;:!?\"'()")
            is_cap = bool(core) and core[0].isupper()
            if is_cap and i == 0:
                nxt    = words[1].strip(".,;:!?\"'()") if len(words) > 1 else ""
                is_cap = bool(nxt) and nxt[0].isupper()
            if is_cap:
                run.append(core)
            else:
                if len(run) >= 1:
                    phrases.append(" ".join(run))
                run = []
        if run:
            phrases.append(" ".join(run))

    # Drop single-token phrases that are stopword-ish ("I", "The")
    return [p for p in phrases
            if len(p.split()) > 1 or p.lower() not in STOPWORDS]


def _content_words(statement: str, limit: int = 3) -> list[str]:
    """Top content words: ≥4 chars, non-stopword, plus bare numbers.

    The ≥4 threshold (not >4) matters: "moon", "wall", "jobs", "debt" are
    exactly four letters and often the most informative word in the claim.
    """
    words   = re.sub(r"[^a-z0-9%\s]", " ", statement.lower()).split()
    numbers = [w for w in words if w.isdigit() or "%" in w]
    content = [w for w in words
               if w not in STOPWORDS and len(w) >= 4 and not w.isdigit()]
    seen: list[str] = []
    for w in content + numbers:
        if w not in seen:
            seen.append(w)
    return seen[:limit]


_SAYS_PREFIX = re.compile(r"^says\s+(that\s+)?", re.IGNORECASE)


def build_queries(statement: str, subject: str = "", speaker: str = "") -> list[str]:
    """
    Build up to three search queries, best first:

        1. entity phrases + top content words
        2. the statement itself (truncated for URL sanity)
        3. speaker + first subject topic  (LIAR metadata context)

    LIAR's attribution prefix ("Says …") is stripped before entity
    extraction so it never pollutes queries.
    """
    queries: list[str] = []

    core     = _SAYS_PREFIX.sub("", statement).strip()
    entities = _entity_phrases(core)
    entity_toks = {w.lower() for p in entities for w in p.split()}
    content  = [w for w in _content_words(core) if w not in entity_toks]
    if entities:
        q = " ".join(entities[:2] + content[:2])
        queries.append(q)
    elif content:
        queries.append(" ".join(content))

    queries.append(statement[:300])

    speaker_clean = speaker.replace("-", " ").strip()
    topic         = subject.split(",")[0].replace("-", " ").strip()
    if speaker_clean or topic:
        queries.append(f"{speaker_clean} {topic}".strip())

    # De-duplicate, keep order, drop empties
    out: list[str] = []
    for q in queries:
        q = " ".join(q.split())
        if q and q.lower() not in {o.lower() for o in out}:
            out.append(q)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Evidence assembly
# ══════════════════════════════════════════════════════════════════════════════

def assemble_evidence(passages: list[str], budget: int = 900) -> str:
    """
    Join passages, truncating at sentence boundaries within ``budget`` chars.

    Whole sentences are accumulated until the next one would overflow the
    budget. If even the first sentence exceeds it, that sentence is cut hard
    so callers always get *something* on a hit.
    """
    picked: list[str] = []
    used = 0
    for passage in passages:
        for sentence in _SENTENCE_SPLIT.split(passage.strip()):
            if not sentence:
                continue
            extra = len(sentence) + (1 if picked else 0)
            if used + extra > budget:
                if not picked:                    # first sentence longer than budget
                    return sentence[:budget]
                return " ".join(picked)
            picked.append(sentence)
            used += extra
    return " ".join(picked)


# ══════════════════════════════════════════════════════════════════════════════
# Evidence Retriever
# ══════════════════════════════════════════════════════════════════════════════

class EvidenceRetriever:
    """
    Search-based Wikipedia evidence retrieval with reranking and caching.

    Usage:
        retriever = EvidenceRetriever(reranker="auto")
        evidence  = retriever.retrieve(row)            # LIAR TSV row
        evidence  = retriever.retrieve_text(statement) # bare statement (app)
        by_id     = retriever.retrieve_batch(rows)     # parallel, for eval

    Args:
        reranker        : "auto" | "semantic" | "lexical" or a reranker object.
        cache_path      : JSON cache location (default data/evidence_cache_v2.json).
                          None disables persistence (still caches in memory).
        fetch_fn        : ``fn(url) -> bytes`` — injectable for tests.
        n_candidates    : search results to fetch summaries for.
        n_passages      : top reranked passages kept.
        evidence_budget : max evidence chars (sentence-boundary truncation).
    """

    CACHE_VERSION = 2
    _FLUSH_EVERY  = 50

    def __init__(
        self,
        reranker                  = "auto",
        cache_path: Optional[str] = CACHE_V2_PATH,
        fetch_fn:   Optional[Callable[[str], bytes]] = None,
        n_candidates:    int = 5,
        n_passages:      int = 2,
        evidence_budget: int = 900,
    ) -> None:
        self.reranker = (get_reranker(reranker) if isinstance(reranker, str)
                         else reranker)
        self.fetch_fn        = fetch_fn or default_fetch
        self.n_candidates    = n_candidates
        self.n_passages      = n_passages
        self.evidence_budget = evidence_budget

        self._cache_path = cache_path
        self._lock       = threading.Lock()
        self._dirty      = 0
        self._cache: dict = self._load_cache()

    # ── cache ─────────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, encoding="utf-8") as fh:
                    cache = json.load(fh)
                if cache.get("_meta", {}).get("version") == self.CACHE_VERSION:
                    return cache
            except Exception:
                pass
        return {"_meta": {"version": self.CACHE_VERSION,
                          "reranker": self.reranker.name}}

    def flush(self) -> None:
        """Persist the cache to disk (no-op when cache_path is None)."""
        if not self._cache_path:
            return
        with self._lock:
            snapshot = dict(self._cache)
            self._dirty = 0
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            tmp = self._cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, indent=1)
            os.replace(tmp, self._cache_path)
        except Exception:
            pass  # non-fatal: next run re-fetches

    def _cache_put(self, key: str, entry: dict) -> None:
        with self._lock:
            self._cache[key] = entry
            self._dirty += 1
            should_flush = self._dirty >= self._FLUSH_EVERY
        if should_flush:
            self.flush()

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _search_candidates(self, query: str) -> Optional[list[dict]]:
        """
        One API call → [{"title", "extract"}, …] in search-relevance order.

        Disambiguation pages ("Romney may refer to:") carry a pageprops
        marker and are dropped here. Returns None on fetch failure —
        distinct from [] (genuinely no results) so callers don't cache
        transient errors as permanent misses.
        """
        url = SEARCH_API.format(k=self.n_candidates,
                                q=urllib.parse.quote(query))
        try:
            data = json.loads(self.fetch_fn(url))
        except FetchError:
            return None
        except Exception:
            return None
        pages = data.get("query", {}).get("pages", {})
        ranked = sorted(pages.values(), key=lambda p: p.get("index", 99))
        out = []
        for page in ranked:
            if "disambiguation" in page.get("pageprops", {}):
                continue
            extract = (page.get("extract") or "").strip()
            if extract:
                out.append({"title": page.get("title", ""), "extract": extract})
        return out

    # ── core pipeline ─────────────────────────────────────────────────────────

    def _gather_candidates(self, queries: list[str]) -> tuple[str, list[dict], bool]:
        """Try each query in order → (query_used, candidates, cacheable)."""
        all_failed = True
        for query in queries:
            candidates = self._search_candidates(query)
            if candidates is None:                 # fetch error
                continue
            all_failed = False
            if candidates:
                return query, candidates, True
        # No candidates: cacheable only if at least one search succeeded
        return (queries[0] if queries else ""), [], not all_failed

    def _retrieve(self, key: str, statement: str,
                  subject: str = "", speaker: str = "") -> str:
        with self._lock:
            cached = self._cache.get(key)
        if isinstance(cached, dict):
            return cached.get("evidence", "")

        queries = build_queries(statement, subject, speaker)
        query, candidates, cacheable = self._gather_candidates(queries)

        evidence = ""
        if candidates:
            scores = self.reranker.scores(statement,
                                          [c["extract"] for c in candidates])
            ranked = [c for _, c in sorted(zip(scores, candidates),
                                           key=lambda sc: sc[0], reverse=True)]
            evidence = assemble_evidence(
                [c["extract"] for c in ranked[: self.n_passages]],
                budget=self.evidence_budget,
            )

        if cacheable:      # never cache network failures as permanent misses
            self._cache_put(key, {
                "statement":  statement,
                "query":      query,
                "candidates": candidates,
                "evidence":   evidence,
            })
        return evidence

    # ── public interface ──────────────────────────────────────────────────────

    def retrieve(self, row: list[str]) -> str:
        """
        Retrieve evidence for one LIAR TSV row.

        Reads only columns 0 (id), 2 (statement), 3 (subject), 4 (speaker).
        The label (column 1) and credit-history counts are never consulted.
        """
        liar_id   = row[0].strip() if len(row) > 0 else ""
        statement = row[2].strip() if len(row) > 2 else ""
        subject   = row[3].strip() if len(row) > 3 else ""
        speaker   = row[4].strip() if len(row) > 4 else ""
        if not statement:
            return ""
        key = liar_id or f"text::{statement}"
        return self._retrieve(key, statement, subject, speaker)

    def retrieve_text(self, statement: str,
                      subject: str = "", speaker: str = "") -> str:
        """Retrieve evidence for a bare statement (Streamlit app path)."""
        statement = statement.strip()
        if not statement:
            return ""
        return self._retrieve(f"text::{statement}", statement, subject, speaker)

    def retrieve_batch(self, rows: list[list[str]],
                       max_workers: int = 8) -> dict[str, str]:
        """
        Parallel retrieval over many LIAR rows → {liar_id: evidence}.

        ≤8 workers by default (Wikipedia etiquette). Cached statements cost
        nothing, so re-runs after interruption resume where they stopped.
        """
        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.retrieve, row): row for row in rows}
            for future, row in futures.items():
                liar_id = row[0].strip() if row else ""
                try:
                    results[liar_id] = future.result()
                except Exception:
                    results[liar_id] = ""
        self.flush()
        return results

    def __repr__(self) -> str:
        n = max(len(self._cache) - 1, 0)     # minus _meta
        return (f"EvidenceRetriever(reranker={self.reranker.name}, "
                f"cached={n}, budget={self.evidence_budget})")


# ══════════════════════════════════════════════════════════════════════════════
# Smoke Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== evidence_retriever.py — smoke test ===\n")

    stmt = ("Says Barack Obama promised he would cut the deficit in half "
            "and instead he doubled it.")
    print("Entity phrases :", _entity_phrases(stmt))
    print("Content words  :", _content_words(stmt))
    print("Queries        :", build_queries(stmt, subject="deficit,federal-budget",
                                            speaker="mitt-romney"))

    print("\nLive retrieval (network, lexical reranker)…")
    retriever = EvidenceRetriever(reranker="lexical", cache_path=None)
    for s in [stmt, "Building a wall on the U.S.-Mexico border will take literally years."]:
        ev = retriever.retrieve_text(s)
        print(f"\n  Statement: {s[:70]}")
        print(f"  Evidence : {ev[:160] or '(none)'}")

    print("\n✓ smoke test complete")
