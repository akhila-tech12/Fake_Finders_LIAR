"""
bert_metadata_classifier.py
============================
BERT + Metadata — Model 6b.

Directly implements the agreed next step from our email to
Professor Weber: "Integrate metadata with BERT via concatenating
to the CLS output vector before the classification head."

THE CORE IDEA
=============
Standard BERT (bert_classifier.py):

    text -> BERT body -> [CLS] (768 numbers) -> Linear(768,2) -> FAKE/REAL
                                                  ^^^^^^^^^^^^^
                                          this is the "classification head"

BERT + Metadata (this file):

    text     -> BERT body -> [CLS] (768 numbers) ---+
                                                       concat -> Linear(771,2) -> FAKE/REAL
    metadata (3 numbers: fake_rate, party, exp) -----+

We do NOT modify BERT's 12 transformer layers at all -- those stay
exactly as pre-trained by Google. We only replace the final
classification head with one that accepts 771 inputs instead of 768.

WHY THIS IS THE RIGHT PLACE TO ADD METADATA
============================================
The [CLS] token's 768 numbers represent BERT's complete understanding
of the TEXT. Metadata (speaker credibility, party, experience) is
information ABOUT THE TEXT that BERT cannot get from reading the
words themselves -- it is "verified information beyond the text"
(Prof Weber's exact phrase). Concatenating it right before the final
decision means the classifier sees BOTH signals together and can
learn how to weigh them against each other.

WHY NOT MODIFY THE 768 NUMBERS THEMSELVES?
    We could add metadata as extra "tokens" fed into BERT's input,
    but that would require re-training the entire 12-layer body
    (expensive, and BERT was never designed for numeric features
    as tokens). Concatenation at the end is the standard, lightweight
    approach used in multimodal fusion literature.

EXPERIMENT WE ARE RUNNING
==========================
    BERT text-only        : F1 = 67.24%  (already measured)
    BERT + metadata        : F1 = ?       (this script)
    MLP (TF-IDF+metadata)  : F1 = 75.42%  (already measured, current best)

If BERT+metadata beats MLP -> metadata helps MORE when combined with
deep language understanding.
If BERT+metadata is close to BERT text-only -> the 768 numbers may
already be "saturating" the classifier and 3 extra numbers get lost
in the noise (a real possible finding worth discussing!).

Run on Google Colab / Kaggle (GPU required):
    os.environ['TRAIN_PATH'] = '/kaggle/working/train.tsv'
    os.environ['VALID_PATH'] = '/kaggle/working/valid.tsv'
    os.environ['TEST_PATH']  = '/kaggle/working/test.tsv'
    os.environ['OUTPUT_DIR'] = '/kaggle/working/bert_meta_model'
    !python bert_metadata_classifier.py

Takes ~5-6 minutes on Tesla P100 (same as plain BERT -- metadata adds
negligible compute).

Author  : Akhila Pavithran, Rajana
Project : Fake Finders -- NLP Master Project, Bamberg SS 2026
"""

from __future__ import annotations

import os
import sys
import time
import json
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
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        matthews_corrcoef,
    )
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _DEPS_ERR = str(_e)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add src/ to path

from classification_evaluator import print_report
from feature_extractor         import metadata_features


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

_FAKE = frozenset({"pants-fire", "false", "barely-true"})
_REAL = frozenset({"mostly-true", "true"})


def _map_label(raw: str) -> Optional[int]:
    raw = raw.strip().lower()
    if raw in _FAKE: return 1
    if raw in _REAL: return 0
    return None


def _load_liar_with_meta(path: str):
    """
    Load a LIAR split, returning texts, labels, AND metadata vectors.

    Returns:
        texts    : list of statement strings
        labels   : list of int (1=fake, 0=real)
        metadata : list of [fake_rate, party_code, experience] (floats)
    """
    texts, labels, metadata = [], [], []
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
            metadata.append(metadata_features(parts))
    return texts, labels, metadata


class LIARMetadataDataset(Dataset):
    """PyTorch Dataset that returns tokenised text + metadata + label."""

    def __init__(self, texts, labels, metadata, tokenizer, max_len: int = 128):
        self.labels   = labels
        self.metadata = metadata
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
            "metadata":       torch.tensor(self.metadata[idx], dtype=torch.float32),
            "labels":         torch.tensor(self.labels[idx],   dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Model -- BERT body + custom (768 + metadata)-dim classification head
# ══════════════════════════════════════════════════════════════════════════════

class BERTWithMetadata(nn.Module):
    """
    BERT body (unchanged, pre-trained) + custom classification head
    that accepts BERT's [CLS] output CONCATENATED with metadata.

    Architecture:
        input_ids, attention_mask, token_type_ids
                    |
                BERT body (12 layers, pre-trained, 768-dim output)
                    |
              pooler_output (768 numbers, the [CLS] representation)
                    |
                 dropout
                    |
        concat with metadata (3 numbers) --> 771 numbers
                    |
            Linear(771 -> 2)   <-- THIS is the new classification head,
                                    trained from scratch on LIAR
                    |
                 logits (2 numbers: [score_real, score_fake])
    """

    def __init__(
        self,
        bert_model_name: str   = "bert-base-uncased",
        metadata_dim:    int   = 3,
        num_labels:      int   = 2,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        self.bert         = BertModel.from_pretrained(bert_model_name)
        self.dropout      = nn.Dropout(dropout)
        hidden_size       = self.bert.config.hidden_size  # 768 for bert-base
        self.classifier   = nn.Linear(hidden_size + metadata_dim, num_labels)
        self.metadata_dim = metadata_dim

    def forward(self, input_ids, attention_mask, token_type_ids, metadata, labels=None):
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        pooled = outputs.pooler_output          # (batch, 768) -- the [CLS] vector
        pooled = self.dropout(pooled)

        combined = torch.cat([pooled, metadata], dim=1)  # (batch, 768+3)
        logits   = self.classifier(combined)             # (batch, 2)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


# ══════════════════════════════════════════════════════════════════════════════
# Classifier wrapper -- matches the interface of bert_classifier.py
# ══════════════════════════════════════════════════════════════════════════════

class BERTMetadataClassifier:
    """
    Fine-tunes BERTWithMetadata on LIAR.

    Same public interface as BERTFakeNewsClassifier (bert_classifier.py)
    so results are directly comparable.
    """

    MODEL_NAME = "bert-base-uncased"

    def __init__(
        self,
        epochs:       int   = 3,
        batch_size:   int   = 16,
        lr:           float = 2e-5,
        max_len:      int   = 128,
        warmup_ratio: float = 0.1,
        device:       Optional[str] = None,
    ) -> None:
        if not _DEPS_OK:
            raise ImportError(f"Missing dependency: {_DEPS_ERR}")

        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.max_len      = max_len
        self.warmup_ratio = warmup_ratio
        self.device       = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._tokenizer: Optional[BertTokenizerFast]  = None
        self._model:     Optional[BERTWithMetadata]   = None
        self._trained    = False
        self._train_time = 0.0

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _metrics(y_true, y_pred, y_proba) -> dict:
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

    # -- training ---------------------------------------------------------------

    def fit(self, train_path: str, val_path: str) -> "BERTMetadataClassifier":
        print(f"\n{'='*60}")
        print("  BERT + METADATA FINE-TUNING -- LIAR Dataset")
        print(f"{'='*60}")
        print(f"  Device  : {self.device}")
        print(f"  Epochs  : {self.epochs}")
        print(f"  Batch   : {self.batch_size}")
        print(f"  LR      : {self.lr}")
        print(f"  Metadata: speaker_fake_rate, party_code, experience (3 dims)\n")

        self._tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)
        self._model     = BERTWithMetadata(self.MODEL_NAME, metadata_dim=3).to(self.device)

        # -- Data ------------------------------------------------------------
        train_texts, train_labels, train_meta = _load_liar_with_meta(train_path)
        val_texts,   val_labels,   val_meta   = _load_liar_with_meta(val_path)
        print(f"  Train: {len(train_texts)} - Val: {len(val_texts)}")

        train_ds = LIARMetadataDataset(train_texts, train_labels, train_meta, self._tokenizer, self.max_len)
        val_ds   = LIARMetadataDataset(val_texts,   val_labels,   val_meta,   self._tokenizer, self.max_len)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size,     shuffle=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size * 2, shuffle=False, num_workers=0)

        # -- Optimiser + Scheduler ----------------------------------------------
        total_steps  = len(train_dl) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=0.01, eps=1e-8
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # -- Training loop -----------------------------------------------------
        best_f1 = 0.0
        t0      = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_dl, 1):
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                output = self._model(**batch)
                loss   = output["loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                if step % 50 == 0:
                    print(f"  Epoch {epoch}/{self.epochs}  Step {step}/{len(train_dl)}  "
                          f"Loss: {total_loss/step:.4f}")

            avg_loss = total_loss / len(train_dl)

            # -- Validation ------------------------------------------------------
            self._model.eval()
            all_preds, all_labels, all_probas = [], [], []

            with torch.no_grad():
                for batch in val_dl:
                    batch  = {k: v.to(self.device) for k, v in batch.items()}
                    output = self._model(**batch)
                    probs  = torch.softmax(output["logits"], dim=1)
                    preds  = torch.argmax(probs, dim=1)

                    all_preds .extend(preds.cpu().tolist())
                    all_probas.extend(probs[:, 1].cpu().tolist())
                    all_labels.extend(batch["labels"].cpu().tolist())

            metrics = self._metrics(all_labels, all_preds, all_probas)
            f1      = metrics["f1"]

            print(f"\n  Epoch {epoch} summary:")
            print(f"    Train loss : {avg_loss:.4f}")
            print(f"    Val F1     : {f1:.4f}  Acc: {metrics['accuracy']:.4f}  "
                  f"AUC: {metrics['auc_roc']:.4f}\n")

            if f1 > best_f1:
                best_f1 = f1

        self._train_time = time.perf_counter() - t0
        self._trained    = True
        print(f"  Training complete -- {self._train_time/60:.1f} min  Best val F1: {best_f1:.4f}")
        return self

    # -- inference ----------------------------------------------------------

    def predict_proba(self, text: str, metadata: list) -> float:
        """Return P(fake) for one statement given its metadata vector."""
        self._check_fitted()
        enc = self._tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        enc["metadata"] = torch.tensor([metadata], dtype=torch.float32)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        self._model.eval()
        with torch.no_grad():
            logits = self._model(**enc)["logits"]
            probs  = torch.softmax(logits, dim=1)
        return float(probs[0][1].item())

    def predict(self, text: str, metadata: list) -> int:
        return int(self.predict_proba(text, metadata) >= 0.5)

    # -- evaluation -----------------------------------------------------------

    def evaluate(self, test_path: str) -> dict:
        self._check_fitted()
        texts, labels, metadata = _load_liar_with_meta(test_path)

        ds = LIARMetadataDataset(texts, labels, metadata, self._tokenizer, self.max_len)
        dl = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False, num_workers=0)

        self._model.eval()
        all_preds, all_probas = [], []

        with torch.no_grad():
            for batch in dl:
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                output = self._model(**batch)
                probs  = torch.softmax(output["logits"], dim=1)
                preds  = torch.argmax(probs, dim=1)

                all_preds .extend(preds.cpu().tolist())
                all_probas.extend(probs[:, 1].cpu().tolist())

        return self._metrics(labels, all_preds, all_probas)

    # -- persistence ----------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(directory, "model.pt"))
        self._tokenizer.save_pretrained(directory)
        print(f"  Saved BERT+Metadata model to {directory}")

    # -- helpers ----------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError("Call fit() before inference.")

    def __repr__(self) -> str:
        status = (f"trained in {self._train_time/60:.1f}min" if self._trained else "not trained")
        return f"BERTMetadataClassifier({self.MODEL_NAME}, +3 metadata dims, {status})"


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point -- run on Google Colab / Kaggle (GPU required)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if not _DEPS_OK:
        print(f"Missing dependency: {_DEPS_ERR}")
        print("Run: pip install transformers torch scikit-learn")
        sys.exit(1)

    # Paths -- override via env vars for Colab/Kaggle (see docstring)
    BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else "."
    TRAIN = os.environ.get("TRAIN_PATH", os.path.join(BASE, "data", "train.tsv"))
    VALID = os.environ.get("VALID_PATH", os.path.join(BASE, "data", "valid.tsv"))
    TEST  = os.environ.get("TEST_PATH",  os.path.join(BASE, "data", "test.tsv"))
    SAVE  = os.environ.get("OUTPUT_DIR", os.path.join(BASE, "results", "bert_meta_model"))

    print("=== BERT + Metadata Classifier -- LIAR Dataset ===")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    # -- Train ---------------------------------------------------------------
    clf = BERTMetadataClassifier(epochs=3, batch_size=16, lr=2e-5)
    clf.fit(TRAIN, VALID)

    # -- Evaluate ---------------------------------------------------------------
    results = clf.evaluate(TEST)
    print_report("BERT + Metadata -- Test Results", results)

    # -- Save ---------------------------------------------------------------
    clf.save(SAVE)

    # -- Demo predictions ---------------------------------------------------------
    label_map = {1: "FAKE", 0: "REAL"}
    print("\n=== Example Predictions ===")
    demos = [
        # (text, [fake_rate, party_code, experience])
        ("Building a wall on the U.S.-Mexico border will take literally years.", [0.5, 0.5, 0.5]),
        ("We know there are more Democrats in Georgia than Republicans.",          [0.5, 0.5, 0.5]),
        ("Hillary Clinton agrees with John McCain by voting to give George Bush the benefit of the doubt on Iran.", [0.45, 0.0, 0.9]),
        ("The federal government paid out 601 million in benefits to deceased employees.", [0.5, 0.5, 0.5]),
    ]
    for text, meta in demos:
        pred = clf.predict(text, meta)
        prob = clf.predict_proba(text, meta)
        print(f"  [{label_map[pred]} {prob:.0%}] meta={meta}  {text[:70]}")

    # -- Comparison ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  COMPARISON -- Does metadata help BERT?")
    print(f"{'='*60}")
    print(f"  BERT text-only          : F1 = 0.6724")
    print(f"  BERT + metadata (this)  : F1 = {results['f1']:.4f}")
    diff = results['f1'] - 0.6724
    sign = "+" if diff >= 0 else ""
    print(f"  Difference              : {sign}{diff:.4f}")
    print(f"  {'-'*56}")
    print(f"  MLP (TF-IDF + metadata) : F1 = 0.7542  <- current best")
    diff2 = results['f1'] - 0.7542
    sign2 = "+" if diff2 >= 0 else ""
    print(f"  BERT+meta vs MLP        : {sign2}{diff2:.4f}")
    print(f"{'='*60}")

    # -- Save results JSON ---------------------------------------------------------
    results_path = os.path.join(SAVE, "bert_meta_results.json")
    os.makedirs(SAVE, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"\n{clf}")