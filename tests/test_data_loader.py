"""
Tests for the LIAR label mapping and split loading.
"""
import os

import pytest

from data_loader import TEST_PATH, TRAIN_PATH, VALID_PATH, load_split, map_label


# ── map_label ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["pants-fire", "false", "barely-true"])
def test_fake_labels(raw):
    assert map_label(raw) == +1


@pytest.mark.parametrize("raw", ["mostly-true", "true"])
def test_real_labels(raw):
    assert map_label(raw) == -1


def test_half_true_is_skipped():
    assert map_label("half-true") is None


def test_label_is_case_and_whitespace_insensitive():
    assert map_label("  FALSE ") == +1
    assert map_label("True") == -1


def test_unknown_label_returns_none():
    assert map_label("banana") is None


# ── load_split (uses the real dataset if present) ─────────────────────────────

needs_data = pytest.mark.skipif(
    not os.path.exists(TRAIN_PATH), reason="LIAR dataset not present in data/"
)


@needs_data
def test_load_split_returns_binary_labels_only():
    data = load_split(TEST_PATH)
    assert len(data) > 0
    assert all(label in (+1, -1) for _, label in data)
    assert all(isinstance(text, str) and text for text, _ in data)


@needs_data
def test_all_splits_nonempty():
    for path in (TRAIN_PATH, VALID_PATH, TEST_PATH):
        assert len(load_split(path)) > 100


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_split("/nonexistent/path.tsv")
