"""
bert_classifier.py
==================
Model 6 — BERT fine-tuned on LIAR dataset.

Run on Google Colab (free T4 GPU):
    Runtime → Change Runtime Type → T4 GPU → Save
    !pip install transformers datasets torch scikit-learn

Why BERT is fundamentally different from Models 1-5?
    Models 1-5 all use Bag-of-Words (BOW):
        - read words INDEPENDENTLY
        - "NOT raise taxes" ≈ "raise taxes"  ← wrong!
        - ignore word order completely

    BERT uses ATTENTION MECHANISM:
        - reads the ENTIRE sentence simultaneously
        - every word attends to every other word
        - "NOT" directly modifies "raise" in the attention matrix
        - true contextual language understanding

Pre-training vs Fine-tuning:
    Pre-training (Google, done for us):
        - Trained on 3.3B words (Wikipedia + BookCorpus)
        - Two tasks: masked language model + next sentence prediction
        - 110M parameters, weeks on TPUs, millions of dollars
        - Result: model understands grammar, facts, language

    Fine-tuning (us, 30-60 min on Colab):
        - Load pre-trained weights
        - Add classification head (linear layer on top of [CLS] token)
        - Train 3 epochs on LIAR with small learning rate (2e-5)
        - Model learns: fake vs real political statements
        - Key: small LR prevents forgetting pre-trained knowledge

Author  : Akhila Pavithran, Rajana
Project : Fake Finders — NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import time
import json
from typing import Optional

# ── Check dependencies before importing ──────────────────────────────────────
try:
    import torch
    import numpy as np
    from torch.utils.data                  import Dataset, DataLoader
    from transformers                      import (
        BertTokenizerFast,
        BertForSequenceClassification,
        AdamW,
        get_linear_schedule_with_warmup,
    )
    from sklearn.metrics                   import (
        accuracy_score,
        precision_recall_fscore_support,
    )
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERR = str(_e)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

# classification_evaluator works without torch
from classification_evaluator import print_report


# ══════════════════════════════════════════════════════════════════════════════
# LIAR Dataset Wrapper
# ══════════════════════════════════════════════════════════════════════════════

FAKE_LABELS = frozenset({"pants-fire", "false", "barely-true"})
REAL_LABELS = frozenset({"mostly-true", "true"})


def _map_label(raw: str) -> Optional[int]:
    raw = raw.strip().lower()
    if raw in FAKE_LABELS: return 1
    if raw in REAL_LABELS:  return 0
    return None


def _load_liar(path: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            lbl = _map_label(parts[1])
            if lbl is None:
                continue
            texts.append(parts[2].strip())
            labels.append(lbl)
    return texts, labels


class LIARTorchDataset(Dataset):
    """PyTorch Dataset over tokenised LIAR statements."""

    def __init__(
        self,
        texts:     list[str],
        labels:    list[int],
        tokenizer: "BertTokenizerFast",
        max_len:   int = 128,
    ) -> None:
        self.labels = labels
        self.enc    = tokenizer(
            texts,
            truncation    = True,
            padding       = "max_length",
            max_length    = max_len,
            return_tensors= "pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "token_type_ids": self.enc["token_type_ids"][idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# BERT Classifier
# ══════════════════════════════════════════════════════════════════════════════

class BERTFakeNewsClassifier:
    """
    BERT (bert-base-uncased) fine-tuned on LIAR for binary fake-news detection.

    Training loop is explicit (no Trainer API) so every step is transparent:
        for epoch in epochs:
            for batch in dataloader:
                forward pass → loss → backward → optimizer step

    Public interface (matches all other classifiers):
        fit(train_path, test_path)
        predict(text)          → int
        predict_proba(text)    → float
        evaluate(test_path)    → dict
        save(path) / load(path)
    """

    MODEL_NAME = "bert-base-uncased"

    def __init__(
        self,
        epochs:        int   = 3,
        batch_size:    int   = 16,
        lr:            float = 2e-5,
        max_len:       int   = 128,
        warmup_ratio:  float = 0.1,
        device:        Optional[str] = None,
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
        self.device       = (
            torch.device(device) if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._tokenizer: Optional["BertTokenizerFast"] = None
        self._model:     Optional["BertForSequenceClassification"] = None
        self._trained    = False
        self._train_time = 0.0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _init_model(self) -> None:
        """Load tokeniser + model from HuggingFace Hub."""
        print(f"  Loading {self.MODEL_NAME}...")
        self._tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)
        self._model     = BertForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels = 2,
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

    # ── training ─────────────────────────────────────────────────────────────

    def fit(self, train_path: str, val_path: str) -> "BERTFakeNewsClassifier":
        """
        Fine-tune BERT on LIAR training split.

        Args:
            train_path : path to train.tsv
            val_path   : path to valid.tsv or test.tsv (for monitoring)

        Returns:
            self
        """
        print(f"\n{'='*60}")
        print("  BERT FINE-TUNING — LIAR Dataset")
        print(f"{'='*60}")
        print(f"  Device    : {self.device}")
        print(f"  Epochs    : {self.epochs}")
        print(f"  Batch     : {self.batch_size}")
        print(f"  LR        : {self.lr}")
        print(f"  Max len   : {self.max_len} tokens\n")

        self._init_model()

        # ── Data ──────────────────────────────────────────────────────────────
        train_texts, train_labels = _load_liar(train_path)
        val_texts,   val_labels   = _load_liar(val_path)
        print(f"  Train: {len(train_texts)} · Val: {len(val_texts)}")

        train_ds = LIARTorchDataset(train_texts, train_labels,
                                    self._tokenizer, self.max_len)
        val_ds   = LIARTorchDataset(val_texts,   val_labels,
                                    self._tokenizer, self.max_len)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size * 2,
                              shuffle=False, num_workers=0)

        # ── Optimiser + Scheduler ─────────────────────────────────────────────
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

        # ── Training loop ─────────────────────────────────────────────────────
        best_f1   = 0.0
        t0        = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            # ── Train epoch ───────────────────────────────────────────────────
            self._model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_dl, 1):
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                output = self._model(**batch)
                loss   = output.loss

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

            # ── Validation epoch ──────────────────────────────────────────────
            self._model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in val_dl:
                    batch   = {k: v.to(self.device) for k, v in batch.items()}
                    output  = self._model(**batch)
                    preds   = torch.argmax(output.logits, dim=1)
                    all_preds  .extend(preds.cpu().tolist())
                    all_labels .extend(batch["labels"].cpu().tolist())

            metrics = self._metrics(all_labels, all_preds)
            f1      = metrics["f1"]

            print(f"\n  Epoch {epoch} summary:")
            print(f"    Train loss : {avg_loss:.4f}")
            print(f"    Val F1     : {f1:.4f}  "
                  f"Acc: {metrics['accuracy']:.4f}\n")

            if f1 > best_f1:
                best_f1 = f1

        self._train_time = time.perf_counter() - t0
        self._trained    = True
        print(f"  ✓ Training complete — {self._train_time/60:.1f} min  "
              f"Best val F1: {best_f1:.4f}")
        return self

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, text: str) -> int:
        """Return 1=fake / 0=real for one statement."""
        return int(self.predict_proba(text) >= 0.5)

    def predict_proba(self, text: str) -> float:
        """Return P(fake) for one statement."""
        self._check_fitted()
        enc = self._tokenizer(
            text,
            truncation    = True,
            padding       = "max_length",
            max_length    = self.max_len,
            return_tensors= "pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs  = torch.softmax(logits, dim=1)
        return float(probs[0][1].item())

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, test_path: str) -> dict:
        """Evaluate on a LIAR split file."""
        self._check_fitted()
        texts, labels = _load_liar(test_path)

        ds = LIARTorchDataset(texts, labels,
                              self._tokenizer, self.max_len)
        dl = DataLoader(ds, batch_size=self.batch_size * 2,
                        shuffle=False, num_workers=0)

        self._model.eval()
        all_preds: list[int] = []

        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                preds = torch.argmax(self._model(**batch).logits, dim=1)
                all_preds.extend(preds.cpu().tolist())

        return self._metrics(labels, all_preds)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save fine-tuned model + tokeniser to directory."""
        os.makedirs(directory, exist_ok=True)
        self._model    .save_pretrained(directory)
        self._tokenizer.save_pretrained(directory)
        print(f"  Saved BERT model to {directory}")

    @classmethod
    def load(cls, directory: str, **kwargs) -> "BERTFakeNewsClassifier":
        """Load a previously saved fine-tuned model."""
        inst = cls(**kwargs)
        inst._tokenizer = BertTokenizerFast.from_pretrained(directory)
        inst._model     = BertForSequenceClassification.from_pretrained(
            directory
        ).to(inst.device)
        inst._trained = True
        print(f"  Loaded BERT model from {directory}")
        return inst

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError("Call fit() or load() before inference.")

    def __repr__(self) -> str:
        status = (f"trained in {self._train_time/60:.1f}min"
                  if self._trained else "not trained")
        return f"BERTFakeNewsClassifier({self.MODEL_NAME}, {status})"


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point — run on Google Colab!
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
    SAVE  = os.path.join(BASE, "results", "bert_model")

    print("=== BERT Classifier — LIAR Dataset ===")
    print("(Run this script on Google Colab with T4 GPU)")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = BERTFakeNewsClassifier(epochs=3, batch_size=16, lr=2e-5)
    clf.fit(TRAIN, VALID)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = clf.evaluate(TEST)
    print_report("BERT (fine-tuned) — Test Results", results)

    # ── Save ──────────────────────────────────────────────────────────────────
    clf.save(SAVE)

    # ── Demo predictions ──────────────────────────────────────────────────────
    label_map = {1: "FAKE", 0: "REAL"}
    print("\n=== Example Predictions ===")
    demos = [
        "Building a wall on the U.S.-Mexico border will take literally years.",
        "We know there are more Democrats in Georgia than Republicans.",
        "Hillary Clinton agrees with John McCain by voting to give George Bush the benefit of the doubt on Iran.",
        "The federal government paid out 601 million in benefits to deceased employees.",
    ]
    for text in demos:
        pred = clf.predict(text)
        prob = clf.predict_proba(text)
        print(f"  [{label_map[pred]} {prob:.0%}] {text[:80]}")

    # ── Comparison ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  BASELINE COMPARISON")
    print(f"{'='*55}")
    print(f"  Naive Bayes (BOW)         : F1 = 0.6446")
    print(f"  BERT (fine-tuned, ~3 ep)  : F1 = {results['f1']:.4f}")
    improvement = results['f1'] - 0.6446
    sign = "+" if improvement >= 0 else ""
    print(f"  Improvement               : {sign}{improvement:.4f}")
    print(f"{'='*55}")
    print(f"\n{clf}")
