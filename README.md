# 🔍 Fake Finders — Fake News Detection on the LIAR Dataset

NLP Master Project · University of Bamberg · SS 2026
**Team:** Akhila Pavithran · Rajana — **Supervisor:** Prof. Sabine Weber

Binary fake-news classification (`FAKE` vs `REAL`) on the
[LIAR benchmark](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
(Wang, ACL 2017) — 10,198 political statements from PolitiFact. The six LIAR
labels are collapsed to binary; the ambiguous `half-true` class is excluded.

## Results (LIAR test set)

| # | Model | F1 | Type | Metadata |
|---|-------|-----|------|----------|
| 1 | Naive Bayes (from scratch) | 64.46% | Statistical | ❌ |
| 2 | Perceptron (from scratch) | 57.91% | Neural (1 layer) | ❌ |
| 3 | Logistic Regression (from scratch) | 61.25% | Statistical | ❌ |
| 4 | SVM (RBF) | 72.71% | Kernel | ✅ |
| 5 | MLP (256→128) | 75.42% | Neural (3 layers) | ✅ |
| 6 | BERT fine-tuned | 67.24% | Transformer | ❌ |
| 7 | BERT + RAG (Wikipedia) | 68.64% | Transformer | ❌ |
| 8 | BERT + metadata concat | 67.43% | Transformer | ✅ |
| 9 | **BERT + FiLM (differential LR)** | **78.18%** 🏆 | Transformer | ✅ |

**Key finding:** metadata (speaker credibility history) matters more than model
capacity — but *how* it is fused matters even more. Simple concatenation adds
+0.19%; FiLM (Feature-wise Linear Modulation) with differential learning rates
adds +10.94% over text-only BERT.

## Project structure

```
├── data/                LIAR train/valid/test TSVs (+ cached Wikipedia evidence)
├── models/              trained BERT+FiLM weights (not in git — see below)
├── results/             saved metrics (JSON)
├── src/
│   ├── data_loader.py               shared: LIAR loading + binary label mapping
│   ├── feature_extractor.py         shared: tokenizer, TF-IDF, metadata, evidence
│   ├── classification_evaluator.py  shared: all metrics from scratch
│   ├── classical/       models 1–5 (NB, Perceptron, LR, SVM, MLP)
│   ├── bert/            models 6–9 — trained on Colab/Kaggle GPUs + bert_predict.py
│   ├── analysis/        compare_models, ensemble, error analysis
│   └── app/app.py       Streamlit demo UI
├── tests/               pytest suite for the shared modules
└── archive/             superseded code kept for reference
```

## Setup

Two environments (BERT inference runs as a subprocess with its own venv):

```bash
# main environment (classical models + UI) — Python 3.14
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# BERT inference environment — Python 3.12
python3.12 -m venv venv_bert
venv_bert/bin/pip install -r requirements_bert.txt
```

The trained BERT+FiLM weights (`models/bert_film_v3/`, ~840 MB) are not in git.
Train `src/bert/bert_film_v3_earlystop.py` on Colab/Kaggle and place the output
in `models/bert_film_v3/`. The app works without them (classical models only).

## Usage

```bash
# run any single model (trains + evaluates + prints report)
venv/bin/python src/classical/naive_bayes.py

# full comparison table
venv/bin/python src/analysis/compare_models.py

# ensemble (majority + F1-weighted voting)
venv/bin/python src/analysis/ensemble_classifier.py --bert_path models/bert_film_v3

# demo UI
venv/bin/streamlit run src/app/app.py

# tests
venv/bin/python -m pytest tests/
```

## Data

LIAR dataset © original sources, research use only
(Wang, "Liar, Liar Pants on Fire", ACL 2017). See `data/README`.
