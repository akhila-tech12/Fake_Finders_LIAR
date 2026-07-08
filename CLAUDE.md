# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Fake Finders" — a University of Bamberg NLP master project (SS 2026) that benchmarks a
progression of models for **binary fake-news detection** on the **LIAR dataset**. Every
classifier is implemented and evaluated in `src/`, and the results feed a Streamlit demo UI.

The task is deliberately binary: the six LIAR labels are collapsed to `FAKE` / `REAL` and
the ambiguous `half-true` class is **dropped entirely** (see `data_loader.map_label`). Any
new model or data path must apply the same mapping to stay comparable with existing results.

## Commands

There is no build system, test runner, or `requirements.txt`. Everything is run as a plain
script against one of the virtual environments.

**Virtual environments (two, with different Python versions — do not mix):**
- `venv/` (Python 3.14) — classical models, sklearn, Streamlit. Also has torch. Use this for
  everything except BERT inference. (`src/venv/` is a stale duplicate — ignore it.)
- `venv_bert/` (Python 3.12) — torch + transformers only. Used **as a subprocess** for BERT
  inference so the Streamlit app doesn't need to import torch in-process.

**Run one model** (each script is self-contained: loads data, trains, evaluates, prints a report):
```bash
venv/bin/python src/naive_bayes.py        # also: perceptron, logistic_regression, svm_classifier, mlp_classifier
```

**Run the full comparison table** (baselines are hardcoded; trains SVM + MLP live; pulls BERT/RAG from results/*.json):
```bash
venv/bin/python src/compare_models.py
```

**Run the ensemble** (majority + F1-weighted voting):
```bash
venv/bin/python src/ensemble_classifier.py                              # classical only (Mac, ~10 min)
venv/bin/python src/ensemble_classifier.py --bert_path models/bert_film_v3
```

**Run the demo UI:**
```bash
venv/bin/streamlit run src/app_beautiful.py    # current UI (app.py is the older variant)
```

**BERT inference locally** (CPU, called by the app as a subprocess):
```bash
venv_bert/bin/python src/bert_predict.py "statement text" models/bert_film_v3 "0.5,0.5,0.5"
```

**"Tests"** — several core modules run assertion-based self-checks or smoke tests via `__main__`:
```bash
venv/bin/python src/classification_evaluator.py   # asserts on metric edge cases
venv/bin/python src/feature_extractor.py          # tokenizer/TF-IDF/metadata/Wikipedia smoke test
venv/bin/python src/data_loader.py                # prints split sizes + class distribution
```

**BERT/FiLM training is NOT run locally.** `bert_classifier.py`, `rag_classifier.py`,
`bert_metadata_classifier.py`, `metadatarachna.py`, and `bert_film_*.py` are written to run on
**Google Colab / Kaggle GPUs**. The trained artifact is committed to `models/bert_film_v3/`
(`model.pt` + tokenizer) and consumed at inference time.

## Architecture

Three shared modules are the single source of truth; every model imports from them. Each script
starts with `sys.path.insert(0, dirname(__file__))` so imports resolve regardless of CWD, and
paths are derived from the file location, so scripts can be run from anywhere.

- **`data_loader.py`** — reads the LIAR `.tsv` splits and applies the binary label mapping.
  `map_label` returns `+1` (FAKE), `-1` (REAL), or `None` (half-true → skip).
- **`feature_extractor.py`** — all feature engineering (stdlib only, no external deps):
  `tokenize` (with n-gram support), `TFIDFVectorizer` (fit on train only — fitting on test is
  leakage), `metadata_features` / `speaker_fake_rate` (LIAR credit-history columns 8–12 → 3
  numeric features), `retrieve_evidence` (Wikipedia REST lookup for RAG), and `FeatureBuilder`
  which composes `[TF-IDF | metadata | evidence]` into one vector.
- **`classification_evaluator.py`** — all metrics implemented from scratch (no sklearn.metrics).
  `binary_report()` / `multiclass_report()` return dicts; `print_report()` formats them.

**⚠️ Two label encodings coexist — this is the most common source of bugs:**
- `data_loader` produces `+1` / `-1`.
- `ClassificationEvaluator` and the sklearn models expect `0` / `1` (`_binary_check` enforces
  `{0,1}`). Models remap at the boundary (e.g. `naive_bayes.evaluate` uses `{+1: 1, -1: 0}`).
  When wiring a new model, convert to `0/1` before evaluating.

**Model tiers** (in rough project chronology; numbering matches `compare_models.py`):
1. **From-scratch classical** — `naive_bayes.py`, `perceptron.py`, `logistic_regression.py`.
   Pure Python, binary bag-of-words, no sklearn. These are function-based (train/predict/evaluate),
   not classes.
2. **sklearn-wrapped** — `svm_classifier.py` (`SVMClassifier`, RBF) and `mlp_classifier.py`
   (`MLPFakeNewsClassifier`). Class-based; consume `FeatureBuilder` vectors (TF-IDF + bigrams + metadata).
3. **BERT family** (Colab/Kaggle) — plain `bert_classifier.py`; metadata via concatenation
   (`bert_metadata_classifier.py`, `metadatarachna.py`); metadata via **FiLM** modulation of the
   CLS vector (`bert_film_classifier.py`, `bert_film_v3_earlystop.py`); retrieval-augmented
   (`rag_classifier.py`). `bert_film_v3` is the deployed model.
4. **Ensemble** — `ensemble_classifier.py` combines the above by majority and F1-weighted vote.

**Metadata is the project's central research thread.** The advisor's feedback ("name/topic bias
cannot be fixed by better word representations — the model needs verified information beyond the
text") drives the metadata features and the FiLM/RAG experiments. When touching feature or model
code, preserve the metadata pathway.

**App flow** (`app_beautiful.py` / `app.py`): loads the classical + sklearn models in-process
(cached with `@st.cache_resource`), and shells out to `venv_bert/bin/python bert_predict.py` for
the BERT+FiLM prediction. `BASE` is hardcoded to `~/fake-finders-liar` and BERT is gated behind
`models/bert_film_v3/` + `venv_bert/` existing.

## Data & results

- `data/{train,valid,test}.tsv` are required and must stay in `data/` (paths are relative to repo
  root). `valid` is for development; `test` is touched only for final numbers.
- Scripts persist metrics to `results/*.json` (e.g. `comparison_table.json`, `ensemble_results.json`,
  `bert_results.json`, `rag_results.json`). `compare_models.py` reads BERT/RAG numbers from these
  files rather than recomputing them.
