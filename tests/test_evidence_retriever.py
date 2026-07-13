"""
Tests for bert/evidence_retriever.py — all HTTP mocked via the injectable
fetch_fn; no network access. Runs under venv/ (py3.14, no torch): only the
stdlib LexicalReranker path is exercised; the MiniLM SemanticReranker needs
torch and is covered by the module's __main__ smoke test in venv_bert.
"""
import json
import urllib.error
import urllib.parse

import pytest

from bert.evidence_retriever import (
    EvidenceRetriever,
    LexicalReranker,
    assemble_evidence,
    build_queries,
)


# ── fake Wikipedia backend ────────────────────────────────────────────────────

class FakeWiki:
    """
    Stands in for fetch_fn against the generator=search API. ``search_map``
    maps a search query → list of page dicts ("*" is the fallback for any
    query). A page dict has "title", "extract", and optionally
    "disambiguation": True.
    """

    def __init__(self, search_map=None):
        self.search_map = search_map or {}
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        parsed = urllib.parse.urlparse(url)
        qs     = urllib.parse.parse_qs(parsed.query)
        query  = qs.get("gsrsearch", [""])[0]
        hits   = self.search_map.get(query, self.search_map.get("*", []))
        pages  = {}
        for i, hit in enumerate(hits):
            page = {"index": i + 1, "title": hit["title"],
                    "extract": hit.get("extract", "")}
            if hit.get("disambiguation"):
                page["pageprops"] = {"disambiguation": ""}
            pages[str(1000 + i)] = page
        return json.dumps({"query": {"pages": pages}}).encode()


MCCAIN_ROW = [
    "1234", "false",
    "John McCain has not led on nonproliferation issues.",
    "foreign-policy,military", "barack-obama", "President", "Illinois",
    "democrat", "70", "71", "160", "163", "9", "a speech",
]

MCCAIN_PAGE = {
    "title": "John McCain",
    "extract": ("John Sidney McCain III was an American politician. "
                "He served as a United States senator from Arizona. "
                "He was known for his work on nuclear nonproliferation."),
}

MCCAIN_DISAMBIG = {
    "title": "McCain",
    "extract": "McCain may refer to:",
    "disambiguation": True,
}


def make_retriever(wiki, tmp_path=None, **kwargs):
    cache = str(tmp_path / "cache_v2.json") if tmp_path else None
    kwargs.setdefault("reranker", "lexical")
    return EvidenceRetriever(cache_path=cache, fetch_fn=wiki, **kwargs)


# ── query formation ───────────────────────────────────────────────────────────

def test_build_queries_extracts_entities():
    queries = build_queries("John McCain has not led on nonproliferation issues.")
    assert queries[0].startswith("John McCain")
    assert "nonproliferation" in queries[0]


def test_build_queries_strips_says_prefix():
    queries = build_queries("Says Barack Obama doubled the deficit.")
    assert not any(q.lower().startswith("says") for q in [queries[0]])
    assert "Barack Obama" in queries[0]


def test_build_queries_uses_speaker_and_subject():
    queries = build_queries("the numbers went up", subject="economy,jobs",
                            speaker="mitt-romney")
    assert "mitt romney economy" in queries
    assert not any("-" in q for q in queries if "mitt" in q)


def test_build_queries_sentence_initial_non_entity_dropped():
    # "Building" is capitalized only because it starts the sentence — it must
    # not be treated as an entity phrase (first query falls back to content words).
    queries = build_queries("Building a wall will take years.")
    assert "Building" not in queries[0]
    assert "building" in queries[0]


# ── leakage guard ─────────────────────────────────────────────────────────────

def test_label_and_credit_columns_never_reach_queries():
    row = list(MCCAIN_ROW)
    row[1]  = "XXLABELLEAKXX"
    row[8]  = "88888"
    row[12] = "99999"
    wiki = FakeWiki(search_map={"*": [MCCAIN_PAGE]})
    retriever = make_retriever(wiki)
    retriever.retrieve(row)
    joined = " ".join(urllib.parse.unquote(u) for u in wiki.calls)
    assert "XXLABELLEAKXX" not in joined
    assert "88888" not in joined
    assert "99999" not in joined


# ── disambiguation filtering ──────────────────────────────────────────────────

def test_disambiguation_pages_are_discarded():
    wiki = FakeWiki(search_map={"*": [MCCAIN_DISAMBIG, MCCAIN_PAGE]})
    retriever = make_retriever(wiki)
    evidence = retriever.retrieve(MCCAIN_ROW)
    assert "may refer to" not in evidence
    assert "American politician" in evidence


def test_all_disambiguation_falls_through_to_empty():
    wiki = FakeWiki(search_map={"*": [MCCAIN_DISAMBIG]})
    retriever = make_retriever(wiki)
    assert retriever.retrieve(MCCAIN_ROW) == ""


# ── evidence assembly ─────────────────────────────────────────────────────────

def test_assemble_evidence_respects_budget_at_sentence_boundary():
    passages = ["First sentence here. Second sentence follows. Third one is long."]
    # "First sentence here." = 20 chars; adding " Second sentence follows."
    # needs 45 total, so only the first sentence fits in a 40-char budget.
    out = assemble_evidence(passages, budget=40)
    assert out == "First sentence here."


def test_assemble_evidence_hard_cuts_oversized_first_sentence():
    out = assemble_evidence(["x" * 200 + "."], budget=50)
    assert len(out) == 50


def test_assemble_evidence_joins_multiple_passages():
    out = assemble_evidence(["One two. Three four.", "Five six."], budget=900)
    assert out == "One two. Three four. Five six."


# ── lexical reranker ──────────────────────────────────────────────────────────

def test_lexical_reranker_prefers_on_topic_passage():
    statement = "The unemployment rate doubled to 20 percent this year."
    on_topic  = ("The unemployment rate in the United States measures the share "
                 "of workers without jobs. The rate rose during the recession.")
    off_topic = ("The bald eagle is a bird of prey found in North America. "
                 "It is the national bird of the United States.")
    scores = LexicalReranker().scores(statement, [off_topic, on_topic])
    assert scores[1] > scores[0]


def test_lexical_reranker_empty_passages():
    assert LexicalReranker().scores("anything", []) == []


# ── caching ───────────────────────────────────────────────────────────────────

def test_cache_second_call_makes_zero_fetches(tmp_path):
    wiki = FakeWiki(search_map={"*": [MCCAIN_PAGE]})
    retriever = make_retriever(wiki, tmp_path)
    first = retriever.retrieve(MCCAIN_ROW)
    n_calls = len(wiki.calls)
    second = retriever.retrieve(MCCAIN_ROW)
    assert second == first
    assert len(wiki.calls) == n_calls


def test_cache_persists_across_instances(tmp_path):
    wiki = FakeWiki(search_map={"*": [MCCAIN_PAGE]})
    retriever = make_retriever(wiki, tmp_path)
    first = retriever.retrieve(MCCAIN_ROW)
    retriever.flush()

    fresh_wiki = FakeWiki()          # would 404 on any fetch
    reloaded   = make_retriever(fresh_wiki, tmp_path)
    assert reloaded.retrieve(MCCAIN_ROW) == first
    assert fresh_wiki.calls == []


def test_cache_stores_candidates_for_later_reranking(tmp_path):
    wiki = FakeWiki(search_map={"*": [MCCAIN_PAGE]})
    retriever = make_retriever(wiki, tmp_path)
    retriever.retrieve(MCCAIN_ROW)
    retriever.flush()

    with open(tmp_path / "cache_v2.json", encoding="utf-8") as fh:
        cache = json.load(fh)
    assert cache["_meta"]["version"] == 2
    entry = cache["1234"]
    assert entry["candidates"][0]["title"] == "John McCain"
    assert entry["evidence"]


# ── batch retrieval ───────────────────────────────────────────────────────────

def test_retrieve_batch_matches_per_item_results(tmp_path):
    other_row = list(MCCAIN_ROW)
    other_row[0] = "5678"
    other_row[2] = "The unemployment rate doubled this year."

    def build(tp):
        wiki = FakeWiki(search_map={"*": [MCCAIN_PAGE]})
        return make_retriever(wiki, tp)

    batch_dir = tmp_path / "batch"; batch_dir.mkdir()
    item_dir  = tmp_path / "item";  item_dir.mkdir()

    batch = build(batch_dir).retrieve_batch([MCCAIN_ROW, other_row], max_workers=2)
    single = build(item_dir)
    assert batch == {"1234": single.retrieve(MCCAIN_ROW),
                     "5678": single.retrieve(other_row)}


def test_empty_search_returns_empty_string():
    retriever = make_retriever(FakeWiki(search_map={"*": []}))
    assert retriever.retrieve(MCCAIN_ROW) == ""


def test_fetch_errors_are_not_cached(tmp_path):
    # A throttled/failed request must not poison the cache: after the
    # network recovers, the same statement is retried and succeeds.
    from bert.evidence_retriever import FetchError

    calls = []
    def broken_fetch(url):
        calls.append(url)
        raise FetchError("HTTP 429")

    retriever = make_retriever(broken_fetch, tmp_path)
    assert retriever.retrieve(MCCAIN_ROW) == ""
    assert len(calls) > 0

    healthy = FakeWiki(search_map={"*": [MCCAIN_PAGE]})
    retriever.fetch_fn = healthy
    evidence = retriever.retrieve(MCCAIN_ROW)
    assert "American politician" in evidence
    assert len(healthy.calls) > 0          # it re-fetched — no poisoned entry


def test_genuine_no_results_is_cached(tmp_path):
    # An empty-but-successful search IS a deterministic miss — cache it.
    wiki = FakeWiki(search_map={"*": []})
    retriever = make_retriever(wiki, tmp_path)
    retriever.retrieve(MCCAIN_ROW)
    n_calls = len(wiki.calls)
    retriever.retrieve(MCCAIN_ROW)
    assert len(wiki.calls) == n_calls      # second call served from cache


def test_empty_statement_returns_empty_string():
    retriever = make_retriever(FakeWiki())
    row = ["id", "false", "", "", ""]
    assert retriever.retrieve(row) == ""
