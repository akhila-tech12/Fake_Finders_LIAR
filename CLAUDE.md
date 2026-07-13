# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Fake Finders" — a University of Bamberg NLP master project (SS 2026) that benchmarks a
progression of models for **binary fake-news detection** on the **LIAR dataset**. Every
classifier lives in `src/`, and the results feed a Streamlit demo UI.

The task is deliberately binary: the six LIAR labels are collapsed to `FAKE` / `REAL` and
the ambiguous `half-true` class is **dropped entirely** (see `data_loader.map_label`). Any
new model or data path must apply the same mapping to stay comparable with existing results.

## Commands

**Virtual environments (two, with different Python versions — do not mix):**
- `venv/` (Python 3.14) — classical models, sklearn, Streamlit, pytest. Install:
  `pip install -r requirements.txt`. Use this for everything except BERT inference.
- `venv_bert/` (Python 3.12) — torch + transformers only (`requirements_bert.txt`). Used
  **as a subprocess** for BERT inference so the Streamlit app doesn't import torch in-process.

**Tests:**
```bash
venv/bin/python -m pytest tests/              # full suite
venv/bin/python -m pytest tests/test_feature_extractor.py -k tfidf   # single file / pattern
```

**Run one model** (each script is self-contained: loads data, trains, evaluates, prints a report):
```bash
venv/bin/python src/classical/naive_bayes.py   # also: perceptron, logistic_regression,
                                               #       svm_classifier, mlp_classifier
```

**Analysis:**
```bash
venv/bin/python src/analysis/compare_models.py       # full comparison table (trains SVM+MLP live)
venv/bin/python src/analysis/ensemble_classifier.py  # add --bert_path models/bert_film_v3 for BERT
```

**Demo UI:**
```bash
venv/bin/streamlit run src/app/app.py
```

**BERT inference locally** (CPU, called by the app as a subprocess):
```bash
venv_bert/bin/python src/bert/bert_predict.py "statement text" models/bert_film_v3 "0.5,0.5,0.5"
# 4th arg = evidence string; if the model dir has model_config.json with
# pair_encoding, statement+evidence are encoded as a BERT text pair
```

**Evidence retrieval (v2 RAG path)** — locally run, venv_bert for the semantic reranker:
```bash
venv_bert/bin/python src/bert/build_evidence.py --split valid --report  # intrinsic stats
venv_bert/bin/python src/bert/build_evidence.py                         # all splits → data/evidence_all.json
venv_bert/bin/python src/bert/rag_classifier.py --split valid --legacy  # A/B baseline arm
venv_bert/bin/python src/bert/rag_classifier.py --split valid --evidence data/evidence_all.json
```
Wikipedia requests are globally rate-limited (~1 req/s; Wikimedia 429-blocks the whole IP
across all API families otherwise). Cache: `data/evidence_cache_v2.json` (fetch failures
are never cached — interrupted builds resume cleanly).

Shared modules also run standalone smoke tests via `__main__`
(e.g. `venv/bin/python src/feature_extractor.py`).

**BERT/FiLM training is NOT run locally.** Everything in `src/bert/` except `bert_predict.py`,
`evidence_retriever.py` and `build_evidence.py` is written for **Google Colab / Kaggle GPUs**.
Trained artifacts live in `models/bert_film_v3/` (deployed champion) and `models/bert_film_rag/`
(model 10, needs `data/evidence_all.json` uploaded to Kaggle; local dry run:
`MAX_SAMPLES=50 EPOCHS=1 venv_bert/bin/python src/bert/bert_film_rag.py`). Weights are
gitignored (~840 MB each, obtained from Kaggle) and consumed at inference time.

## Architecture

Three shared modules at the top of `src/` are the single source of truth; everything else
imports them. Entry scripts insert `src/` into `sys.path`
(`sys.path.insert(0, dirname(dirname(abspath(__file__))))`), so imports resolve regardless
of CWD, and all data/model paths are derived from file locations — scripts run from anywhere.

- **`src/data_loader.py`** — reads the LIAR `.tsv` splits and applies the binary label mapping.
  `map_label` returns `+1` (FAKE), `-1` (REAL), or `None` (half-true → skip).
- **`src/feature_extractor.py`** — all feature engineering (stdlib only): `tokenize` (n-gram
  support), `TFIDFVectorizer` (fit on train only — fitting on test is leakage; vocab keeps the
  *most document-frequent* qualifying terms), `metadata_features` / `speaker_fake_rate` (LIAR
  credit-history columns 8–12 → 3 numeric features), `retrieve_evidence` (Wikipedia REST lookup,
  cached in `data/evidence_cache.json`), and `FeatureBuilder` which composes
  `[TF-IDF | metadata | evidence]` into one vector.
- **`src/classification_evaluator.py`** — all metrics implemented from scratch (no
  sklearn.metrics). `binary_report()` / `multiclass_report()` return dicts.

**⚠️ Two label encodings coexist — the most common source of bugs:**
- `data_loader` produces `+1` / `-1`.
- `ClassificationEvaluator` and the sklearn models expect `0` / `1` (`_binary_check` enforces
  `{0,1}`). Models remap at the boundary (e.g. `naive_bayes.evaluate` uses `{+1: 1, -1: 0}`).
  When wiring a new model, convert to `0/1` before evaluating.

**Model packages** (numbering matches `compare_models.py` / the README table):
- **`src/classical/`** — models 1–5. NB/Perceptron/LR are pure-Python function-based
  (train/predict/evaluate, binary BOW); SVM/MLP are class-based sklearn wrappers
  (`SVMClassifier`, `MLPFakeNewsClassifier`) consuming `FeatureBuilder` vectors.
- **`src/bert/`** — models 6–9 (Colab/Kaggle training scripts): plain fine-tune
  (`bert_classifier.py`), metadata-concat variants (`bert_metadata_classifier.py`,
  `metadatarachna.py` — two team variants of Model 6b, both kept), FiLM fusion
  (`bert_film_classifier.py` v1, `bert_film_v3_earlystop.py` — the deployed winner), RAG
  (`rag_classifier.py`), and `bert_predict.py` (the only locally-run file: standalone CPU
  inference, no repo imports, prints JSON to stdout).
- **`src/analysis/`** — cross-model tooling. Imports classical models via
  `from classical import naive_bayes as nb` etc. `compare_models.py` hardcodes the baseline
  numbers and reads BERT/RAG metrics from `results/*.json` rather than recomputing.
- **`src/app/app.py`** — Streamlit UI. Loads classical models in-process (cached with
  `@st.cache_resource`, trains at first startup ~3 min) and shells out to
  `venv_bert/bin/python src/bert/bert_predict.py` for BERT+FiLM. `BASE` (repo root) is derived
  from `__file__`; BERT options only appear if `models/bert_film_v3/` and `venv_bert/` exist.
- **`archive/`** — superseded code (old UI). Don't extend; git history has the context.

**Metadata is the project's central research thread.** The advisor's feedback ("name/topic bias
cannot be fixed by better word representations — the model needs verified information beyond the
text") drives the metadata features and the FiLM/RAG experiments. When touching feature or model
code, preserve the metadata pathway.

## Data & results

- `data/{train,valid,test}.tsv` are required (paths derived from repo root). `valid` is for
  development; `test` is touched only for final numbers.
- `data/evidence_cache.json` caches Wikipedia lookups — delete it to force fresh retrieval.
- Scripts persist metrics to `results/*.json`; `compare_models.py` and the README table are the
  canonical summaries. If a model is retrained and numbers change, update both.
