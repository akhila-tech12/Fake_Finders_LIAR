"""
bert_with_metadata.py
=====================
Model 6b — BERT fine-tuned on LIAR + metadata features fused at the classifier head.

WHY ADD METADATA?
-----------------
The existing bert_classifier.py uses ONLY the statement text (column 2).
The LIAR dataset has 14 columns — rich speaker/context metadata is ignored.

Available metadata columns (from data_loader.py schema):
    0  : ID
    1  : label
    2  : statement           ← BERT already uses this
    3  : subject             ← NEW: topic domain
    4  : speaker             ← encoded as historical fake rate
    5  : speaker job title   ← encoded as experience proxy
    6  : state info
    7  : party affiliation   ← 0=dem, 1=rep, 0.5=other
    8  : barely true counts  ┐
    9  : false counts        │ speaker's historical
    10 : half true counts    ├ track record → fake_rate
    11 : mostly true counts  │
    12 : pants on fire count ┘
    13 : context / venue

KEY IDEA — Late Fusion Architecture:
    ┌─────────────┐     ┌───────────────┐
    │   Statement │     │   Metadata    │
    │   (text)    │     │  (5 features) │
    └──────┬──────┘     └──────┬────────┘
           │                   │
    ┌──────▼──────┐            │
    │    BERT     │            │
    │  [CLS] rep  │            │
    │  (768 dim)  │            │
    └──────┬──────┘            │
           │                   │
           └────────┬──────────┘
                    │ concat (768 + 5 = 773 dim)
             ┌──────▼──────┐
             │  Classifier │  Linear(773 → 2)
             └─────────────┘

This is called "late fusion" — each modality processed separately,
combined only at the final classification layer. Metadata is lightweight
(5 floats) so it doesn't overwhelm BERT's learned representation.

Metadata features (mirrors feature_extractor.py):
    1. speaker_fake_rate  — historical dishonesty rate [0, 1]
    2. party_code         — 0.0=democrat, 1.0=republican, 0.5=other
    3. experience         — normalized log(total_statements + 1)
    4. subject_code       — hash-based topic category [0, 1]
    5. has_context        — 1.0 if venue/context is known, else 0.0

DATA USAGE SUMMARY:
===================
Current bert_classifier.py:
    BERT text-only   → uses column 2 (statement text) ONLY

This file (bert_with_metadata.py):
    BERT + Metadata  → uses column 2 (statement) via BERT
                     + column 3  (subject)        → subject_code
                     + columns 4+5 (speaker/job)  → experience
                     + column 7  (party)           → party_code
                     + columns 8-12 (vote counts)  → speaker_fake_rate
                     + column 13 (context)         → has_context

Author  : Akhila Pavithran, Rajana (extended with metadata fusion)
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import time
import math
from typing import Optional

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        BertTokenizerFast,
        BertModel,
        AdamW,
        get_linear_schedule_with_warmup,
    )
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
    )
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERR = str(_e)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

FAKE_LABELS = frozenset({"pants-fire", "false", "barely-true"})
REAL_LABELS = frozenset({"mostly-true", "true"})
N_META_FEATURES = 5   # number of numeric metadata features


# ══════════════════════════════════════════════════════════════════════════════
# Metadata Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def speaker_fake_rate(parts: list[str]) -> float:
    """
    Compute historical dishonesty rate from speaker's past rulings.

    LIAR columns 8-12 contain the speaker's cumulative ruling counts:
        8  : barely_true_count
        9  : false_count
        10 : half_true_count
        11 : mostly_true_count
        12 : pants_on_fire_count

    fake_rate = (false + barely_true + pants_on_fire) / total
    Returns 0.5 (neutral) if counts are missing or zero.
    """
    try:
        bt   = float(parts[8])   # barely-true
        fl   = float(parts[9])   # false
        ht   = float(parts[10])  # half-true
        mt   = float(parts[11])  # mostly-true
        pof  = float(parts[12])  # pants-on-fire
        total = bt + fl + ht + mt + pof
        if total == 0:
            return 0.5
        return (fl + bt + pof) / total
    except (IndexError, ValueError):
        return 0.5


def extract_metadata(parts: list[str]) -> list[float]:
    """
    Extract 5 numeric metadata features from a raw LIAR TSV row.

    Features:
        [0] speaker_fake_rate  : historical dishonesty [0, 1]
        [1] party_code         : 0.0=dem, 1.0=rep, 0.5=other
        [2] experience         : log-normalized total rulings [0, ~1]
        [3] subject_code       : hash of topic domain → [0, 1]
        [4] has_context        : 1.0 if venue/context is non-empty

    All features are in [0, 1] so no further normalization is needed.
    """
    # ── feature 0: speaker historical fake rate ────────────────────────────
    fake_rate = speaker_fake_rate(parts)

    # ── feature 1: party affiliation ──────────────────────────────────────
    party = (parts[7] if len(parts) > 7 else "").strip().lower()
    if "democrat" in party:
        party_code = 0.0
    elif "republican" in party:
        party_code = 1.0
    else:
        party_code = 0.5

    # ── feature 2: experience (log-normalized total statement count) ───────
    try:
        counts = [float(parts[c]) for c in [8, 9, 10, 11, 12] if len(parts) > c]
        total  = sum(counts)
        # log(total+1) / log(1000+1) → clamp at 1.0 for very prolific speakers
        experience = min(math.log(total + 1) / math.log(1001), 1.0)
    except (ValueError, IndexError):
        experience = 0.0

    # ── feature 3: subject/topic code ─────────────────────────────────────
    subject = (parts[3] if len(parts) > 3 else "").strip().lower()
    # Deterministic hash → bucket in [0, 1]
    subject_code = (hash(subject) % 100) / 100.0 if subject else 0.5

    # ── feature 4: has context / venue ────────────────────────────────────
    context     = (parts[13] if len(parts) > 13 else "").strip()
    has_context = 1.0 if context else 0.0

    return [fake_rate, party_code, experience, subject_code, has_context]


# ══════════════════════════════════════════════════════════════════════════════
# LIAR Data Loader
# ══════════════════════════════════════════════════════════════════════════════

def _map_label(raw: str) -> Optional[int]:
    raw = raw.strip().lower()
    if raw in FAKE_LABELS: return 1
    if raw in REAL_LABELS: return 0
    return None   # half-true → skip


def load_liar_with_metadata(
    path: str,
) -> tuple[list[str], list[list[float]], list[int]]:
    """
    Load LIAR .tsv file and return texts, metadata feature vectors, and labels.

    Returns:
        texts    : list of statement strings (column 2)
        meta     : list of 5-float metadata vectors
        labels   : list of binary labels (1=fake, 0=real)
    """
    texts, meta, labels = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lbl = _map_label(parts[1])
            if lbl is None:
                continue
            texts.append(parts[2].strip())
            meta.append(extract_metadata(parts))
            labels.append(lbl)
    return texts, meta, labels


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class LIARMetaDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenised text + numeric metadata for LIAR.

    Each item returns:
        input_ids      : (max_len,)
        attention_mask : (max_len,)
        token_type_ids : (max_len,)
        meta           : (N_META_FEATURES,)  ← NEW vs text-only dataset
        labels         : scalar
    """

    def __init__(
        self,
        texts:     list[str],
        meta:      list[list[float]],
        labels:    list[int],
        tokenizer: "BertTokenizerFast",
        max_len:   int = 128,
    ) -> None:
        self.labels = labels
        self.meta   = torch.tensor(meta, dtype=torch.float)   # (N, 5)

        self.enc = tokenizer(
            texts,
            truncation     = True,
            padding        = "max_length",
            max_length     = max_len,
            return_tensors = "pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "token_type_ids": self.enc["token_type_ids"][idx],
            "meta":           self.meta[idx],                  # ← metadata
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# BERT + Metadata Fusion Model (nn.Module)
# ══════════════════════════════════════════════════════════════════════════════

class BertWithMetadata(nn.Module):
    """
    Late-fusion model: BERT [CLS] representation + numeric metadata vector.

    Architecture:
        BERT encoder  → 768-dim [CLS] embedding
        Metadata MLP  → 5 → 32 → 32 (optional lightweight transform)
        Concat        → 768 + 32 = 800 dim
        Dropout(0.3)
        Linear        → 2 logits (binary classification)

    Why a small MLP on metadata?
        Raw metadata values live on different scales / distributions.
        A 2-layer MLP with ReLU lets the model learn a better representation
        before combining with BERT. It only adds ~200 parameters — negligible.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        n_meta:          int = N_META_FEATURES,
        meta_hidden:     int = 32,
        dropout:         float = 0.3,
        num_labels:      int = 2,
    ) -> None:
        super().__init__()

        # BERT backbone — returns hidden states; we use [CLS] (index 0)
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim  = self.bert.config.hidden_size  # 768 for bert-base

        # Metadata projection: 5 → 32 → 32
        self.meta_proj = nn.Sequential(
            nn.Linear(n_meta, meta_hidden),
            nn.ReLU(),
            nn.Linear(meta_hidden, meta_hidden),
            nn.ReLU(),
        )

        # Final classifier
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_dim + meta_hidden, num_labels)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        meta:           torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids, attention_mask, token_type_ids : BERT inputs
            meta    : (batch, N_META_FEATURES) float tensor
            labels  : optional — if provided, loss is computed and returned

        Returns:
            dict with keys: logits, (optionally) loss
        """
        # ── BERT forward pass ─────────────────────────────────────────────
        bert_out    = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        # [CLS] token embedding → shape (batch, 768)
        cls_embed = bert_out.last_hidden_state[:, 0, :]

        # ── Metadata projection ───────────────────────────────────────────
        meta_embed = self.meta_proj(meta)   # (batch, 32)

        # ── Late fusion: concatenate ──────────────────────────────────────
        combined = torch.cat([cls_embed, meta_embed], dim=1)  # (batch, 800)
        combined = self.dropout(combined)

        # ── Classification head ───────────────────────────────────────────
        logits = self.classifier(combined)  # (batch, 2)

        result = {"logits": logits}

        if labels is not None:
            loss_fn     = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# High-Level Classifier (matches interface of BERTFakeNewsClassifier)
# ══════════════════════════════════════════════════════════════════════════════

class BERTWithMetadataClassifier:
    """
    BERT + metadata late-fusion classifier for LIAR fake news detection.

    Public interface mirrors bert_classifier.BERTFakeNewsClassifier:
        fit(train_path, val_path)
        predict(text, meta_row)
        predict_proba(text, meta_row)
        evaluate(test_path)
        save(directory)
        load(directory)

    Difference vs text-only BERT:
        - predict() / predict_proba() now also accept a raw TSV parts list
          so that metadata features can be extracted for single-statement
          inference (useful in the Streamlit app).
    """

    MODEL_NAME = "bert-base-uncased"

    def __init__(
        self,
        epochs:       int   = 3,
        batch_size:   int   = 16,
        lr:           float = 2e-5,
        max_len:      int   = 128,
        warmup_ratio: float = 0.1,
        meta_hidden:  int   = 32,
        dropout:      float = 0.3,
        device:       Optional[str] = None,
    ) -> None:
        if not _DEPS_OK:
            raise ImportError(
                f"Missing dependency: {_DEPS_ERR}\n"
                "Run: pip install transformers torch scikit-learn"
            )

        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.max_len      = max_len
        self.warmup_ratio = warmup_ratio
        self.meta_hidden  = meta_hidden
        self.dropout      = dropout
        self.device       = (
            torch.device(device) if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._tokenizer: Optional["BertTokenizerFast"] = None
        self._model:     Optional[BertWithMetadata]    = None
        self._trained    = False
        self._train_time = 0.0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _init_model(self) -> None:
        print(f"  Loading {self.MODEL_NAME} + metadata fusion head…")
        self._tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)
        self._model     = BertWithMetadata(
            bert_model_name = self.MODEL_NAME,
            n_meta          = N_META_FEATURES,
            meta_hidden     = self.meta_hidden,
            dropout         = self.dropout,
        ).to(self.device)

    @staticmethod
    def _metrics(y_true: list, y_pred: list) -> dict:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        return {
            "accuracy" : round(accuracy_score(y_true, y_pred), 4),
            "precision": round(float(prec), 4),
            "recall"   : round(float(rec),  4),
            "f1"       : round(float(f1),   4),
        }

    def _make_loader(
        self,
        texts:  list[str],
        meta:   list[list[float]],
        labels: list[int],
        shuffle: bool = False,
        multiplier: int = 1,
    ) -> DataLoader:
        ds = LIARMetaDataset(
            texts, meta, labels, self._tokenizer, self.max_len
        )
        return DataLoader(
            ds,
            batch_size = self.batch_size * multiplier,
            shuffle    = shuffle,
            num_workers= 0,
        )

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, train_path: str, val_path: str) -> "BERTWithMetadataClassifier":
        """
        Fine-tune BERT+metadata on LIAR training split.

        What this does differently vs text-only BERT:
            1. Loads metadata from all 14 TSV columns (not just column 2)
            2. Passes metadata tensor alongside BERT inputs each forward pass
            3. The fusion model concatenates [CLS] + metadata before classifying

        Args:
            train_path : path to train.tsv
            val_path   : path to valid.tsv or test.tsv
        """
        print(f"\n{'='*62}")
        print("  BERT + METADATA FUSION — LIAR Dataset")
        print(f"{'='*62}")
        print(f"  Device      : {self.device}")
        print(f"  Epochs      : {self.epochs}")
        print(f"  Batch size  : {self.batch_size}")
        print(f"  LR          : {self.lr}")
        print(f"  Max len     : {self.max_len} tokens")
        print(f"  Meta feats  : {N_META_FEATURES}  (fake_rate, party, experience, subject, context)")
        print(f"  Meta hidden : {self.meta_hidden}\n")

        self._init_model()

        # ── Load data (text + metadata + labels) ──────────────────────────
        tr_texts, tr_meta, tr_labels = load_liar_with_metadata(train_path)
        vl_texts, vl_meta, vl_labels = load_liar_with_metadata(val_path)
        print(f"  Train: {len(tr_texts)} · Val: {len(vl_texts)}")
        print(f"  Metadata features per sample: {N_META_FEATURES}\n")

        train_dl = self._make_loader(tr_texts, tr_meta, tr_labels, shuffle=True)
        val_dl   = self._make_loader(vl_texts, vl_meta, vl_labels, multiplier=2)

        # ── Optimizer + scheduler ─────────────────────────────────────────
        total_steps  = len(train_dl) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = AdamW(
            self._model.parameters(),
            lr           = self.lr,
            weight_decay = 0.01,
            eps          = 1e-8,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = warmup_steps,
            num_training_steps = total_steps,
        )

        # ── Training loop ─────────────────────────────────────────────────
        best_f1   = 0.0
        t0        = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_dl, 1):
                # Move all tensors to device (including 'meta')
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                output = self._model(**batch)   # dict with 'loss' and 'logits'
                loss   = output["loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                if step % 50 == 0:
                    print(f"  Epoch {epoch}/{self.epochs}  "
                          f"Step {step}/{len(train_dl)}  "
                          f"Loss: {total_loss/step:.4f}")

            avg_loss = total_loss / len(train_dl)

            # ── Validation ────────────────────────────────────────────────
            self._model.eval()
            all_preds, all_labels_list = [], []

            with torch.no_grad():
                for batch in val_dl:
                    batch   = {k: v.to(self.device) for k, v in batch.items()}
                    output  = self._model(**batch)
                    preds   = torch.argmax(output["logits"], dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels_list.extend(batch["labels"].cpu().tolist())

            metrics = self._metrics(all_labels_list, all_preds)
            f1      = metrics["f1"]

            print(f"\n  Epoch {epoch} summary:")
            print(f"    Train loss : {avg_loss:.4f}")
            print(f"    Val F1     : {f1:.4f}  Acc: {metrics['accuracy']:.4f}\n")

            if f1 > best_f1:
                best_f1 = f1

        self._train_time = time.perf_counter() - t0
        self._trained    = True
        print(f"  ✓ Training complete — {self._train_time/60:.1f} min  "
              f"Best val F1: {best_f1:.4f}")
        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        tsv_parts: Optional[list[str]] = None,
    ) -> int:
        """Return 1=fake / 0=real for one statement."""
        return int(self.predict_proba(text, tsv_parts) >= 0.5)

    def predict_proba(
        self,
        text: str,
        tsv_parts: Optional[list[str]] = None,
    ) -> float:
        """
        Return P(fake) for one statement.

        Args:
            text      : the statement string
            tsv_parts : raw TSV columns for metadata extraction.
                        If None, zeros are used (neutral metadata).
        """
        self._check_fitted()

        # Build metadata vector
        if tsv_parts is not None:
            meta_vec = extract_metadata(tsv_parts)
        else:
            meta_vec = [0.5, 0.5, 0.0, 0.5, 0.0]  # neutral defaults

        enc = self._tokenizer(
            text,
            truncation     = True,
            padding        = "max_length",
            max_length     = self.max_len,
            return_tensors = "pt",
        )
        enc["meta"]   = torch.tensor([meta_vec], dtype=torch.float)
        enc["labels"] = torch.tensor([0], dtype=torch.long)   # dummy label

        enc = {k: v.to(self.device) for k, v in enc.items()}

        self._model.eval()
        with torch.no_grad():
            logits = self._model(**enc)["logits"]
            probs  = torch.softmax(logits, dim=1)

        return float(probs[0][1].item())

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, test_path: str) -> dict:
        """Evaluate on a LIAR split file (uses full metadata)."""
        self._check_fitted()
        texts, meta, labels = load_liar_with_metadata(test_path)
        dl = self._make_loader(texts, meta, labels, multiplier=2)

        self._model.eval()
        all_preds: list[int] = []

        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                preds = torch.argmax(self._model(**batch)["logits"], dim=1)
                all_preds.extend(preds.cpu().tolist())

        return self._metrics(labels, all_preds)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save model weights + tokeniser to directory."""
        import json
        os.makedirs(directory, exist_ok=True)
        torch.save(self._model.state_dict(),
                   os.path.join(directory, "bert_meta_weights.pt"))
        self._tokenizer.save_pretrained(directory)
        # Save config so we can reconstruct model architecture on load
        config = {
            "meta_hidden": self.meta_hidden,
            "dropout":     self.dropout,
            "max_len":     self.max_len,
        }
        with open(os.path.join(directory, "meta_config.json"), "w") as f:
            json.dump(config, f)
        print(f"  Saved BERT+metadata model to {directory}")

    @classmethod
    def load(cls, directory: str, **kwargs) -> "BERTWithMetadataClassifier":
        """Load a previously saved model."""
        import json
        with open(os.path.join(directory, "meta_config.json")) as f:
            config = json.load(f)
        kwargs.setdefault("meta_hidden", config["meta_hidden"])
        kwargs.setdefault("dropout",     config["dropout"])
        kwargs.setdefault("max_len",     config["max_len"])

        inst = cls(**kwargs)
        inst._tokenizer = BertTokenizerFast.from_pretrained(directory)
        inst._model     = BertWithMetadata(
            bert_model_name = cls.MODEL_NAME,
            n_meta          = N_META_FEATURES,
            meta_hidden     = config["meta_hidden"],
            dropout         = config["dropout"],
        ).to(inst.device)
        inst._model.load_state_dict(
            torch.load(
                os.path.join(directory, "bert_meta_weights.pt"),
                map_location=inst.device,
            )
        )
        inst._trained = True
        print(f"  Loaded BERT+metadata model from {directory}")
        return inst

    # ── misc ─────────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError("Call fit() or load() before inference.")

    def __repr__(self) -> str:
        status = (f"trained in {self._train_time/60:.1f}min"
                  if self._trained else "not trained")
        return f"BERTWithMetadataClassifier({self.MODEL_NAME}, {status})"


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point — run on Google Colab T4
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if not _DEPS_OK:
        print("Install dependencies first:")
        print("  pip install transformers torch scikit-learn")
        sys.exit(1)

    BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN = os.path.join(BASE, "data", "train.tsv")
    VALID = os.path.join(BASE, "data", "valid.tsv")
    TEST  = os.path.join(BASE, "data", "test.tsv")
    SAVE  = os.path.join(BASE, "results", "bert_meta_model")

    print("=== BERT + Metadata Classifier — LIAR Dataset ===")
    print("(Run this script on Google Colab with T4 GPU)")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    clf = BERTWithMetadataClassifier(epochs=3, batch_size=16, lr=2e-5)
    clf.fit(TRAIN, VALID)

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = clf.evaluate(TEST)
    print("\n=== BERT + Metadata — Test Results ===")
    for k, v in results.items():
        print(f"  {k:12s}: {v}")

    # ── Save ──────────────────────────────────────────────────────────────
    clf.save(SAVE)

    # ── Demo (text only — neutral metadata) ───────────────────────────────
    label_map = {1: "FAKE", 0: "REAL"}
    print("\n=== Example Predictions (text-only metadata defaults) ===")
    demos = [
        "Building a wall on the U.S.-Mexico border will take literally years.",
        "We know there are more Democrats in Georgia than Republicans.",
        "The federal government paid out 601 million in benefits to deceased employees.",
    ]
    for text in demos:
        pred = clf.predict(text)
        prob = clf.predict_proba(text)
        print(f"  [{label_map[pred]} {prob:.0%}] {text[:80]}")

    # ── Comparison ────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("  MODEL COMPARISON (expected approximate results)")
    print(f"{'='*58}")
    print(f"  Text-only BERT (bert_classifier.py) : F1 ≈ 0.66–0.69")
    print(f"  BERT + Metadata (this file)          : F1 = {results['f1']:.4f}")
    improvement = results["f1"] - 0.67
    sign = "+" if improvement >= 0 else ""
    print(f"  Δ vs text-only BERT (approx)         : {sign}{improvement:.4f}")
    print(f"{'='*58}")
    print(f"\n{clf}")
