"""
Pytest port of the smoke checks in feature_extractor.py.
Network-dependent Wikipedia retrieval is NOT tested here (kept in the
module's __main__ smoke test).
"""
import pytest

from feature_extractor import (
    TFIDFVectorizer,
    metadata_features,
    speaker_fake_rate,
    tokenize,
)


# ── tokenize ──────────────────────────────────────────────────────────────────

def test_tokenize_unigrams():
    assert tokenize("Free Money!") == ["free", "money"]


def test_tokenize_bigrams_capture_negation():
    tokens = tokenize("will not raise", (1, 2))
    assert "not raise" in tokens
    assert "will not" in tokens
    assert set(tokens) >= {"will", "not", "raise"}


def test_tokenize_strips_non_alpha():
    assert tokenize("Taxes-2024, up 5%!") == ["taxes", "up"]


# ── TFIDFVectorizer ───────────────────────────────────────────────────────────

CORPUS = [
    "the president will not raise taxes",
    "conspiracy theory about government plot",
    "reuters reported the federal reserve decision",
    "shocking secret about the deep state revealed",
    "the president talked about the taxes decision",
]


def test_tfidf_requires_fit_before_transform():
    vec = TFIDFVectorizer()
    with pytest.raises(RuntimeError):
        vec.transform("anything")


def test_tfidf_vector_shape_and_nonzero():
    vec = TFIDFVectorizer(max_features=50, ngram_range=(1, 2), min_df=2)
    vec.fit(CORPUS)
    v = vec.transform("president not raise taxes")
    assert len(v) == vec.n_features
    assert any(x > 0 for x in v)


def test_tfidf_vocab_keeps_most_frequent_terms():
    """Regression test for the inverted vocab-selection bug: with a small
    max_features budget, common qualifying terms must survive selection."""
    vec = TFIDFVectorizer(max_features=3, ngram_range=(1, 1), min_df=2)
    vec.fit(CORPUS)
    # 'the' (df=4) and 'about'/'president'/'taxes'/'decision' (df=2) qualify;
    # the top-3 slots must include the most frequent term.
    assert "the" in vec.vocab_


def test_tfidf_min_df_filters_rare_terms():
    vec = TFIDFVectorizer(max_features=100, ngram_range=(1, 1), min_df=2)
    vec.fit(CORPUS)
    assert "conspiracy" not in vec.vocab_  # appears once only


# ── metadata features ─────────────────────────────────────────────────────────

def fake_row(party="republican", counts=("2", "8", "1", "0", "3")):
    row = ["", "false", "stmt", "", "", "", "", party]
    row.extend(counts)
    return row


def test_speaker_fake_rate():
    # (2 barely + 8 false + 3 pants) / 14 total
    assert speaker_fake_rate(fake_row()) == pytest.approx(13 / 14, abs=1e-4)


def test_speaker_fake_rate_unknown_history():
    assert speaker_fake_rate(fake_row(counts=("0",) * 5)) == 0.5
    assert speaker_fake_rate(["", "false", "stmt"]) == 0.5  # short row


def test_metadata_features_party_encoding():
    assert metadata_features(fake_row(party="republican"))[1] == 1.0
    assert metadata_features(fake_row(party="democrat"))[1] == 0.0
    assert metadata_features(fake_row(party="independent"))[1] == 0.5


def test_metadata_features_shape_and_range():
    feats = metadata_features(fake_row())
    assert len(feats) == 3
    assert all(0.0 <= f <= 1.0 for f in feats)
