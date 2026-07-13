"""
bert_film_rag.py
================
Model 10 — BERT + FiLM + RAG: evidence-augmented FiLM fusion.

Combines the two research threads that were previously separate:

    Model 7 (rag_classifier.py)      : Wikipedia evidence, no metadata fusion
                                       → F1 = 0.6864
    Model 9 (bert_film_v3_earlystop) : FiLM metadata fusion, no evidence
                                       → F1 = 0.7818  (deployed champion)

This script trains a single model that gets BOTH signals:

    [CLS] statement [SEP] evidence [SEP]     ← text-PAIR encoding
              |
        BERT body (segment embeddings distinguish claim from evidence)
              |
        pooler [CLS] (768)      metadata (3: fake_rate, party, experience)
              |                          |
              +———— FiLM (gamma*cls+beta)+
              |
        dropout → Linear(768, 2)

Differences vs bert_film_v3_earlystop.py (everything else is the proven
v3 recipe — differential LRs, identity-init FiLM, early stopping):

    1. Evidence per statement is read from EVIDENCE_PATH
       (data/evidence_all.json — precomputed locally by build_evidence.py,
       keyed by LIAR id; Kaggle has no reliable network access).
    2. Text-PAIR tokenisation with truncation="only_second": real segment
       embeddings instead of a literal " [SEP] " string, and truncation
       eats the evidence, never the claim. Statements with no evidence
       are encoded alone — the model also learns the statement-only case.
    3. max_len = 256 (evidence needs the room; v3 used 128).
    4. save() also writes the BERT body via save_pretrained() and a
       model_config.json so bert_predict.py knows to use pair encoding.

Run on Google Colab / Kaggle (GPU required):
    os.environ['TRAIN_PATH']    = '/kaggle/working/train.tsv'
    os.environ['VALID_PATH']    = '/kaggle/working/valid.tsv'
    os.environ['TEST_PATH']     = '/kaggle/working/test.tsv'
    os.environ['EVIDENCE_PATH'] = '/kaggle/working/evidence_all.json'
    os.environ['OUTPUT_DIR']    = '/kaggle/working/bert_film_rag'
    !python bert_film_rag.py

Local CPU dry run (validates the script end-to-end before burning GPU):
    MAX_SAMPLES=50 EPOCHS=1 venv_bert/bin/python src/bert/bert_film_rag.py

Author  : Akhila Pavithran, Rajana
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import time
import json
import copy
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        BertTokenizerFast,
        BertModel,
        get_linear_schedule_with_warmup,
    )
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERR = str(_e)

# sklearn is present on Kaggle/Colab but deliberately NOT in venv_bert
# (torch+transformers only). The local MAX_SAMPLES dry run falls back to
# the repo's own from-scratch metrics.
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        matthews_corrcoef,
    )
    _SK_OK = True
except ImportError:
    _SK_OK = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

from classification_evaluator import print_report
from feature_extractor         import metadata_features


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading — LIAR rows + precomputed evidence
# ══════════════════════════════════════════════════════════════════════════════

_FAKE = frozenset({"pants-fire", "false", "barely-true"})
_REAL = frozenset({"mostly-true", "true"})


def _map_label(raw: str) -> Optional[int]:
    raw = raw.strip().lower()
    if raw in _FAKE: return 1
    if raw in _REAL: return 0
    return None


def _load_liar_with_rag(path: str, evidence_map: dict[str, str],
                        max_samples: int = 0):
    """
    Load a LIAR split → texts, evidences, labels, metadata vectors.

    Evidence is looked up by LIAR id (column 0); a missing id yields ""
    so the model gracefully degrades to statement-only for that example.
    """
    texts, evidences, labels, metadata = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lbl = _map_label(parts[1])
            if lbl is None:
                continue
            texts.append(parts[2].strip())
            evidences.append(evidence_map.get(parts[0].strip(), ""))
            labels.append(lbl)
            metadata.append(metadata_features(parts))
            if max_samples and len(texts) >= max_samples:
                break
    return texts, evidences, labels, metadata


class LIARRAGDataset(Dataset):
    """
    Tokenised (statement, evidence) pairs + metadata + label.

    Pair encoding gives BERT real segment embeddings (token_type_ids
    0 for the claim, 1 for the evidence) — the format it was pretrained
    with for NSP — instead of the literal " [SEP] " string concat used
    by the old RAG model. truncation="only_second" cuts evidence first,
    never the claim.
    """

    def __init__(self, texts, evidences, labels, metadata,
                 tokenizer, max_len: int = 256):
        self.labels   = labels
        self.metadata = metadata

        ids, masks, types = [], [], []
        for text, ev in zip(texts, evidences):
            if ev:
                enc = tokenizer(text, ev,
                                truncation = "only_second",
                                padding    = "max_length",
                                max_length = max_len)
            else:
                enc = tokenizer(text,
                                truncation = True,
                                padding    = "max_length",
                                max_length = max_len)
            ids.append(enc["input_ids"])
            masks.append(enc["attention_mask"])
            types.append(enc["token_type_ids"])

        self.input_ids      = torch.tensor(ids)
        self.attention_mask = torch.tensor(masks)
        self.token_type_ids = torch.tensor(types)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "metadata":       torch.tensor(self.metadata[idx], dtype=torch.float32),
            "labels":         torch.tensor(self.labels[idx],   dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# FiLM Layer + Model — identical architecture to bert_film_v3_earlystop.py
# ══════════════════════════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: modulated = gamma*cls + beta,
    identity-initialised (gamma=1, beta=0). See bert_film_v3_earlystop.py
    for the full design rationale."""

    def __init__(self, metadata_dim: int = 3, bert_dim: int = 768, hidden_dim: int = 32):
        super().__init__()
        self.to_gamma = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bert_dim),
        )
        self.to_beta = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bert_dim),
        )
        self._init_as_identity()

    def _init_as_identity(self) -> None:
        nn.init.zeros_(self.to_gamma[-1].weight)
        nn.init.ones_(self.to_gamma[-1].bias)    # gamma starts at 1.0
        nn.init.zeros_(self.to_beta[-1].weight)
        nn.init.zeros_(self.to_beta[-1].bias)    # beta starts at 0.0

    def forward(self, cls_vector: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(metadata)
        beta  = self.to_beta(metadata)
        return gamma * cls_vector + beta


class BERTWithFiLM(nn.Module):
    """BERT body + FiLM modulation + classification head (same as v3;
    only the INPUT differs — statement+evidence pairs at max_len 256)."""

    def __init__(
        self,
        bert_model_name: str   = "bert-base-uncased",
        metadata_dim:    int   = 3,
        film_hidden_dim: int   = 32,
        num_labels:      int   = 2,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        self.bert       = BertModel.from_pretrained(bert_model_name)
        hidden_size     = self.bert.config.hidden_size
        self.film       = FiLMLayer(metadata_dim, hidden_size, film_hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, metadata, labels=None):
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        pooled    = outputs.pooler_output
        modulated = self.dropout(self.film(pooled, metadata))
        logits    = self.classifier(modulated)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


# ══════════════════════════════════════════════════════════════════════════════
# Classifier wrapper — v3 training recipe on paired inputs
# ══════════════════════════════════════════════════════════════════════════════

class BERTFiLMRAGClassifier:
    """
    Fine-tunes BERTWithFiLM on LIAR statements paired with precomputed
    Wikipedia evidence. Training recipe (differential LRs, warmup, early
    stopping on val F1, best-checkpoint restore) is the proven v3 one.
    """

    MODEL_NAME = "bert-base-uncased"

    def __init__(
        self,
        evidence_path: str,
        epochs:       int   = 100,
        batch_size:   int   = 16,
        lr:           float = 2e-5,
        film_lr:      float = 1e-3,
        max_len:      int   = 256,     # room for evidence (v3 used 128)
        warmup_ratio: float = 0.1,
        patience:     int   = 5,
        max_samples:  int   = 0,       # >0 = dry-run subset
        device:       Optional[str] = None,
    ) -> None:
        if not _DEPS_OK:
            raise ImportError(f"Missing dependency: {_DEPS_ERR}")

        with open(evidence_path, encoding="utf-8") as fh:
            self.evidence_map: dict[str, str] = json.load(fh)

        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.film_lr      = film_lr
        self.max_len      = max_len
        self.warmup_ratio = warmup_ratio
        self.patience     = patience
        self.max_samples  = max_samples
        self.device       = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._tokenizer:    Optional[BertTokenizerFast] = None
        self._model:        Optional[BERTWithFiLM]      = None
        self._best_weights: Optional[dict]              = None
        self._trained       = False
        self._train_time    = 0.0

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _metrics(y_true, y_pred, y_proba) -> dict:
        if _SK_OK:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            return {
                "accuracy" : round(accuracy_score(y_true, y_pred), 4),
                "precision": round(float(prec), 4),
                "recall"   : round(float(rec),  4),
                "f1"       : round(float(f1),   4),
                "auc_roc"  : round(roc_auc_score(y_true, y_proba), 4),
                "mcc"      : round(matthews_corrcoef(y_true, y_pred), 4),
            }

        # Fallback (local dry run): repo metrics + manual AUC/MCC
        from classification_evaluator import ClassificationEvaluator
        report = ClassificationEvaluator(y_true, y_pred).binary_report()

        # AUC-ROC via the rank-sum (Mann-Whitney U) formulation
        ranked = sorted(range(len(y_proba)), key=lambda i: y_proba[i])
        ranks  = [0.0] * len(y_proba)
        i = 0
        while i < len(ranked):                      # average ranks over ties
            j = i
            while j + 1 < len(ranked) and y_proba[ranked[j + 1]] == y_proba[ranked[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[ranked[k]] = avg
            i = j + 1
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        if n_pos and n_neg:
            rank_sum = sum(r for r, t in zip(ranks, y_true) if t == 1)
            auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        else:
            auc = 0.0

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc   = (tp * tn - fp * fn) / denom if denom else 0.0

        return {
            "accuracy" : round(report["accuracy"],  4),
            "precision": round(report["precision"], 4),
            "recall"   : round(report["recall"],    4),
            "f1"       : round(report["f1"],        4),
            "auc_roc"  : round(auc, 4),
            "mcc"      : round(mcc, 4),
        }

    def _load_split(self, path: str):
        return _load_liar_with_rag(path, self.evidence_map, self.max_samples)

    def _make_loader(self, path: str, shuffle: bool, batch_mult: int = 1):
        texts, evidences, labels, meta = self._load_split(path)
        ds = LIARRAGDataset(texts, evidences, labels, meta,
                            self._tokenizer, self.max_len)
        dl = DataLoader(ds, batch_size=self.batch_size * batch_mult,
                        shuffle=shuffle, num_workers=0)
        with_ev = sum(1 for e in evidences if e)
        return dl, labels, with_ev

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, train_path: str, val_path: str) -> "BERTFiLMRAGClassifier":
        print(f"\n{'='*60}")
        print("  BERT + FiLM + RAG FINE-TUNING — LIAR Dataset")
        print(f"{'='*60}")
        print(f"  Device    : {self.device}")
        print(f"  Max epochs: {self.epochs} (early stopping patience={self.patience})")
        print(f"  Batch     : {self.batch_size}")
        print(f"  Max len   : {self.max_len} (statement [SEP] evidence pair)")
        print(f"  BERT LR   : {self.lr}     (pretrained body — slow)")
        print(f"  FiLM LR   : {self.film_lr}  (new layers — fast)")
        print(f"  Evidence  : {len(self.evidence_map)} precomputed entries\n")

        self._tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)
        self._model     = BERTWithFiLM(self.MODEL_NAME, metadata_dim=3).to(self.device)

        train_dl, train_labels, train_ev = self._make_loader(train_path, shuffle=True)
        val_dl,   val_labels,   val_ev   = self._make_loader(val_path, shuffle=False,
                                                             batch_mult=2)
        print(f"  Train: {len(train_labels)} ({train_ev} with evidence, "
              f"{100 * train_ev / max(len(train_labels), 1):.0f}%)")
        print(f"  Val  : {len(val_labels)} ({val_ev} with evidence)")

        bert_params = list(self._model.bert.parameters())
        film_params = (list(self._model.film.parameters())
                       + list(self._model.classifier.parameters()))
        optimizer = torch.optim.AdamW([
            {"params": bert_params, "lr": self.lr},
            {"params": film_params, "lr": self.film_lr},
        ], weight_decay=0.01, eps=1e-8)

        total_steps  = len(train_dl) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_f1           = 0.0
        best_epoch        = 0
        epochs_no_improve = 0
        t0                = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_dl, 1):
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                loss   = self._model(**batch)["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                if step % 50 == 0:
                    print(f"  Epoch {epoch}/{self.epochs}  Step {step}/{len(train_dl)}  "
                          f"Loss: {total_loss/step:.4f}")

            # ── Validation ────────────────────────────────────────────────────
            self._model.eval()
            all_preds, all_probas = [], []
            with torch.no_grad():
                for batch in val_dl:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    probs = torch.softmax(self._model(**batch)["logits"], dim=1)
                    all_preds .extend(torch.argmax(probs, dim=1).cpu().tolist())
                    all_probas.extend(probs[:, 1].cpu().tolist())

            metrics = self._metrics(val_labels, all_preds, all_probas)
            f1      = metrics["f1"]
            print(f"\n  Epoch {epoch}: train loss {total_loss/len(train_dl):.4f}  "
                  f"val F1 {f1:.4f}  acc {metrics['accuracy']:.4f}  "
                  f"AUC {metrics['auc_roc']:.4f}")

            if f1 > best_f1:
                best_f1, best_epoch, epochs_no_improve = f1, epoch, 0
                self._best_weights = copy.deepcopy(self._model.state_dict())
                print(f"    ✓ New best! Saved checkpoint (epoch {epoch})")
            else:
                epochs_no_improve += 1
                print(f"    No improvement for {epochs_no_improve}/{self.patience} epochs")
                if epochs_no_improve >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(best: epoch {best_epoch}, val F1 {best_f1:.4f})")
                    break

        self._train_time = time.perf_counter() - t0
        self._trained    = True
        if self._best_weights is not None:
            self._model.load_state_dict(self._best_weights)
            print(f"\n  ✓ Restored best model from epoch {best_epoch}")
        print(f"  Training complete — {self._train_time/60:.1f} min  "
              f"best val F1: {best_f1:.4f}")
        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, text: str, metadata: list, evidence: str = "") -> float:
        """Return P(fake) for one statement (+ optional evidence)."""
        self._check_fitted()
        if evidence:
            enc = self._tokenizer(text, evidence, truncation="only_second",
                                  padding="max_length", max_length=self.max_len,
                                  return_tensors="pt")
        else:
            enc = self._tokenizer(text, truncation=True, padding="max_length",
                                  max_length=self.max_len, return_tensors="pt")
        enc["metadata"] = torch.tensor([metadata], dtype=torch.float32)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(self._model(**enc)["logits"], dim=1)
        return float(probs[0][1].item())

    def predict(self, text: str, metadata: list, evidence: str = "") -> int:
        return int(self.predict_proba(text, metadata, evidence) >= 0.5)

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, test_path: str) -> dict:
        self._check_fitted()
        dl, labels, with_ev = self._make_loader(test_path, shuffle=False,
                                                batch_mult=2)
        print(f"\n  Test: {len(labels)} ({with_ev} with evidence)")

        self._model.eval()
        all_preds, all_probas = [], []
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                probs = torch.softmax(self._model(**batch)["logits"], dim=1)
                all_preds .extend(torch.argmax(probs, dim=1).cpu().tolist())
                all_probas.extend(probs[:, 1].cpu().tolist())

        return self._metrics(labels, all_preds, all_probas)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """
        Replicates the models/bert_film_v3/ layout exactly, so
        bert_predict.py loads it the same way:

            config.json + pytorch_model.bin  ← BERT body (save_pretrained)
            model.pt                         ← full FiLM state dict
            tokenizer files
            model_config.json                ← NEW: tells bert_predict.py to
                                               use pair encoding at max_len 256
        """
        self._check_fitted()
        os.makedirs(directory, exist_ok=True)
        self._model.bert.save_pretrained(directory, safe_serialization=False)
        torch.save(self._model.state_dict(), os.path.join(directory, "model.pt"))
        self._tokenizer.save_pretrained(directory)
        with open(os.path.join(directory, "model_config.json"), "w") as fh:
            json.dump({"max_len": self.max_len,
                       "pair_encoding": True,
                       "uses_evidence": True}, fh, indent=2)
        print(f"  Saved BERT+FiLM+RAG model to {directory}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError("Call fit() before inference.")

    def __repr__(self) -> str:
        status = (f"trained in {self._train_time/60:.1f}min"
                  if self._trained else "not trained")
        return f"BERTFiLMRAGClassifier({self.MODEL_NAME}, FiLM+RAG, {status})"


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point — run on Google Colab / Kaggle (GPU required)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if not _DEPS_OK:
        print(f"Missing dependency: {_DEPS_ERR}")
        print("Run: pip install transformers torch scikit-learn")
        sys.exit(1)

    BASE     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TRAIN    = os.environ.get("TRAIN_PATH",    os.path.join(BASE, "data", "train.tsv"))
    VALID    = os.environ.get("VALID_PATH",    os.path.join(BASE, "data", "valid.tsv"))
    TEST     = os.environ.get("TEST_PATH",     os.path.join(BASE, "data", "test.tsv"))
    EVIDENCE = os.environ.get("EVIDENCE_PATH", os.path.join(BASE, "data", "evidence_all.json"))
    SAVE     = os.environ.get("OUTPUT_DIR",    os.path.join(BASE, "models", "bert_film_rag"))

    MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))   # dry-run knob
    EPOCHS      = int(os.environ.get("EPOCHS", "100"))

    print("=== BERT + FiLM + RAG — LIAR Dataset ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if MAX_SAMPLES:
        print(f"DRY RUN: max {MAX_SAMPLES} samples/split, {EPOCHS} epoch(s)")

    if not os.path.exists(EVIDENCE):
        print(f"\nERROR: evidence file not found: {EVIDENCE}")
        print("Build it locally first:")
        print("  venv_bert/bin/python src/bert/build_evidence.py")
        sys.exit(1)

    clf = BERTFiLMRAGClassifier(
        evidence_path = EVIDENCE,
        epochs        = EPOCHS,
        batch_size    = 16,      # fits P100 at max_len 256; on T4 OOM → 8
        lr            = 2e-5,
        film_lr       = 1e-3,
        patience      = 5,
        max_samples   = MAX_SAMPLES,
    )
    clf.fit(TRAIN, VALID)

    results = clf.evaluate(TEST)
    print_report("BERT + FiLM + RAG — Test Results", results)

    clf.save(SAVE)

    # ── Comparison against the deployed champion ─────────────────────────────
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    print(f"  BERT text-only          : F1 = 0.6724")
    print(f"  BERT + RAG (v1)         : F1 = 0.6864")
    print(f"  BERT + FiLM v3          : F1 = 0.7818  ← deployed champion")
    print(f"  BERT + FiLM + RAG (this): F1 = {results['f1']:.4f}")
    delta = results["f1"] - 0.7818
    print(f"  vs champion             : {'+' if delta >= 0 else ''}{delta:.4f}")
    print(f"{'='*60}")

    results_path = os.path.join(SAVE, "bert_film_rag_results.json")
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"\n{clf}")
