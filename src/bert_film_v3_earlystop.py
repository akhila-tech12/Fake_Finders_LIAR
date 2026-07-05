"""
bert_film_classifier.py
========================
BERT + FiLM (Feature-wise Linear Modulation) for metadata fusion.

Follows Professor Weber's suggestion to try FiLM instead of simple
concatenation for combining BERT's text representation with our
3 metadata features.

THE PROBLEM WE FOUND (bert_metadata_classifier.py)
====================================================
Simple concatenation barely helped:
    BERT text-only         : F1 = 0.6724
    BERT + metadata (concat): F1 = 0.6743  (+0.19%, negligible)

Our hypothesis: 768 BERT numbers dominate a 771-dim linear layer.
3 metadata numbers are too small a fraction to influence training.

THE FiLM IDEA
=============
Instead of metadata SITTING ALONGSIDE BERT's 768 numbers,
metadata CONTROLS how those 768 numbers are used. A small network
maps the 3 metadata values to two 768-dimensional vectors:

    gamma (scale) and beta (shift)

and we apply them to BERT's [CLS] vector elementwise:

    modulated_cls = gamma * cls_vector + beta

Concatenation (previous experiment):
    text -> BERT -> [CLS] (768) --+
                                    concat -> Linear(771,2) -> FAKE/REAL
    metadata (3) ------------------+
    (metadata = 3 extra numbers, drowned out among 768)

FiLM (this experiment):
    metadata (3) -> small network -> gamma (768), beta (768)
    text -> BERT -> [CLS] (768) -> (gamma * cls + beta) -> Linear(768,2)
    (metadata RESHAPES all 768 dimensions, not just adding 3 more)

WHY THIS MIGHT FIX THE DIMENSIONAL IMBALANCE
==============================================
With FiLM, metadata's "vote" isn't 3 numbers out of 771 -- it
directly multiplies and shifts EVERY one of BERT's 768 numbers.
A speaker with a high fake_rate could, in principle, learn to
amplify whichever BERT dimensions correlate with deceptive
language patterns, rather than being just 3 passive extra inputs.

EXPERIMENT
==========
    BERT text-only            : F1 = 0.6724
    BERT + metadata (concat)  : F1 = 0.6743
    BERT + FiLM (this script) : F1 = ?
    MLP (TF-IDF + metadata)   : F1 = 0.7542  <- current best

Run on Google Colab / Kaggle (GPU required):
    os.environ['TRAIN_PATH'] = '/kaggle/working/train.tsv'
    os.environ['VALID_PATH'] = '/kaggle/working/valid.tsv'
    os.environ['TEST_PATH']  = '/kaggle/working/test.tsv'
    os.environ['OUTPUT_DIR'] = '/kaggle/working/bert_film_model'
    !python bert_film_classifier.py

Takes ~5-6 minutes on Tesla P100 (same as plain BERT).

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classification_evaluator import print_report
from feature_extractor         import metadata_features


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading  (identical to bert_metadata_classifier.py)
# ══════════════════════════════════════════════════════════════════════════════

_FAKE = frozenset({"pants-fire", "false", "barely-true"})
_REAL = frozenset({"mostly-true", "true"})


def _map_label(raw: str) -> Optional[int]:
    raw = raw.strip().lower()
    if raw in _FAKE: return 1
    if raw in _REAL: return 0
    return None


def _load_liar_with_meta(path: str):
    """Load a LIAR split, returning texts, labels, AND metadata vectors."""
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
# FiLM Layer -- the new component vs bert_metadata_classifier.py
# ══════════════════════════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

    Maps a small metadata vector to two 768-dimensional vectors,
    gamma (scale) and beta (shift), then applies them elementwise
    to BERT's [CLS] vector:

        modulated = gamma * cls_vector + beta

    Why a small hidden layer instead of a single Linear(3, 768)?
        A single linear layer from 3 -> 768 has very limited
        capacity to express interesting scale/shift PATTERNS --
        it can only produce linear combinations of 3 numbers.
        A small hidden layer (3 -> 32 -> 768) lets the network
        learn non-linear interactions between the 3 metadata
        features before deciding how to modulate BERT.

    Initialisation detail (important!):
        We initialise gamma's output bias to 1.0 and weights near
        zero, so that at the START of training, gamma ~= 1 and
        beta ~= 0 -- meaning modulation starts as the IDENTITY
        function (modulated ~= cls_vector, same as plain BERT).
        Training then gradually learns how much to deviate from
        identity based on what helps classification. Without this,
        random initial gamma/beta would scramble BERT's pretrained
        representations from step one, making training unstable.
    """

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
        """Start gamma~=1, beta~=0 so modulation begins as identity."""
        final_gamma_layer = self.to_gamma[-1]
        final_beta_layer  = self.to_beta[-1]

        nn.init.zeros_(final_gamma_layer.weight)
        nn.init.ones_(final_gamma_layer.bias)   # gamma starts at 1.0

        nn.init.zeros_(final_beta_layer.weight)
        nn.init.zeros_(final_beta_layer.bias)   # beta starts at 0.0

    def forward(self, cls_vector: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_vector : (batch, 768) -- BERT's [CLS] output
            metadata   : (batch, 3)   -- [fake_rate, party, experience]

        Returns:
            (batch, 768) -- modulated CLS vector
        """
        gamma = self.to_gamma(metadata)   # (batch, 768)
        beta  = self.to_beta(metadata)    # (batch, 768)
        return gamma * cls_vector + beta


# ══════════════════════════════════════════════════════════════════════════════
# Model -- BERT body + FiLM modulation + classification head
# ══════════════════════════════════════════════════════════════════════════════

class BERTWithFiLM(nn.Module):
    """
    BERT body (unchanged, pre-trained) + FiLM layer that lets metadata
    reshape the [CLS] vector + standard classification head.

    Architecture:
        input_ids, attention_mask, token_type_ids
                    |
                BERT body (12 layers, pre-trained, 768-dim output)
                    |
              pooler_output (768 numbers, the [CLS] representation)
                    |                              metadata (3 numbers)
                    |                                    |
                    +-------------- FiLM layer ----------+
                    |   modulated = gamma * cls + beta
                    v
            modulated_cls (768 numbers, SAME shape as before,
                            but now reshaped by metadata)
                    |
                 dropout
                    |
            Linear(768 -> 2)   <-- classification head, same SIZE
                                    as plain BERT (no dimension growth!)
                    |
                 logits (2 numbers: [score_real, score_fake])
    """

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
        hidden_size     = self.bert.config.hidden_size  # 768 for bert-base
        self.film       = FiLMLayer(metadata_dim, hidden_size, film_hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)  # still 768 -> 2!

    def forward(self, input_ids, attention_mask, token_type_ids, metadata, labels=None):
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        pooled = outputs.pooler_output              # (batch, 768) -- the [CLS] vector

        modulated = self.film(pooled, metadata)      # (batch, 768) -- metadata-reshaped
        modulated = self.dropout(modulated)

        logits = self.classifier(modulated)           # (batch, 2)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


# ══════════════════════════════════════════════════════════════════════════════
# Classifier wrapper -- matches the interface of bert_metadata_classifier.py
# ══════════════════════════════════════════════════════════════════════════════

class BERTFiLMClassifier:
    """
    Fine-tunes BERTWithFiLM on LIAR.

    Same public interface as BERTMetadataClassifier so results are
    directly comparable across all three BERT variants:
        bert_classifier.py          -> text only
        bert_metadata_classifier.py -> concatenation fusion
        bert_film_classifier.py     -> FiLM fusion (this file)
    """

    MODEL_NAME = "bert-base-uncased"

    def __init__(
        self,
        epochs:       int   = 100,
        batch_size:   int   = 16,
        lr:           float = 2e-5,
        film_lr:      float = 1e-3,
        max_len:      int   = 128,
        warmup_ratio: float = 0.1,
        patience:     int   = 5,
        device:       Optional[str] = None,
    ) -> None:
        """
        Args:
            epochs    : maximum number of epochs (100 by default per
                        Professor Weber's suggestion). Early stopping
                        will stop training before this if val F1 stops
                        improving, so in practice training usually
                        stops well before epoch 100.
            lr        : learning rate for BERT's pretrained body (slow --
                        don't disturb pretrained knowledge too fast)
            film_lr   : learning rate for FiLM layers + classification head
                        (fast -- these are randomly initialized and need
                        to learn quickly within just a few epochs)
            patience  : early stopping patience -- stop training if val F1
                        does not improve for this many consecutive epochs.
                        patience=5 means: if epochs 11,12,13,14,15 all
                        have lower val F1 than epoch 10, stop at epoch 15
                        and return the epoch-10 model weights.

        Why early stopping + 100 epochs?
            Previous BERT experiments peaked at epoch 1-3 then declined.
            With a fixed 3-epoch budget we might stop too early (haven't
            peaked yet) or too late (already declining). Early stopping
            lets the training run find the optimal point automatically,
            up to a maximum of 100 epochs. The BEST checkpoint is always
            saved and restored at the end, not the final epoch's weights.
        """
        if not _DEPS_OK:
            raise ImportError(f"Missing dependency: {_DEPS_ERR}")

        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.film_lr      = film_lr
        self.max_len      = max_len
        self.warmup_ratio = warmup_ratio
        self.patience     = patience
        self.device       = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._tokenizer:    Optional[BertTokenizerFast] = None
        self._model:        Optional[BERTWithFiLM]      = None
        self._best_weights: Optional[dict]              = None
        self._trained       = False
        self._train_time    = 0.0

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

    def fit(self, train_path: str, val_path: str) -> "BERTFiLMClassifier":
        print(f"\n{'='*60}")
        print("  BERT + FiLM FINE-TUNING -- LIAR Dataset")
        print(f"{'='*60}")
        print(f"  Device   : {self.device}")
        print(f"  Max epochs: {self.epochs} (early stopping patience={self.patience})")
        print(f"  Batch    : {self.batch_size}")
        print(f"  BERT LR  : {self.lr}     (pretrained body -- slow)")
        print(f"  FiLM LR  : {self.film_lr}  (new layers -- fast, {self.film_lr/self.lr:.0f}x BERT LR)")
        print(f"  Metadata : speaker_fake_rate, party_code, experience (3 dims)")
        print(f"  Fusion   : FiLM (gamma*cls + beta), NOT concatenation\n")

        self._tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)
        self._model     = BERTWithFiLM(self.MODEL_NAME, metadata_dim=3).to(self.device)

        # -- Data ------------------------------------------------------------
        train_texts, train_labels, train_meta = _load_liar_with_meta(train_path)
        val_texts,   val_labels,   val_meta   = _load_liar_with_meta(val_path)
        print(f"  Train: {len(train_texts)} - Val: {len(val_texts)}")

        train_ds = LIARMetadataDataset(train_texts, train_labels, train_meta, self._tokenizer, self.max_len)
        val_ds   = LIARMetadataDataset(val_texts,   val_labels,   val_meta,   self._tokenizer, self.max_len)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size,     shuffle=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size * 2, shuffle=False, num_workers=0)

        # -- Optimiser with DIFFERENTIAL learning rates --------------------------
        # BERT's pretrained body gets the small, standard fine-tuning rate.
        # FiLM layers + classification head are randomly initialized and
        # need a much higher rate to move meaningfully away from their
        # identity initialization within just a few epochs.
        bert_params  = list(self._model.bert.parameters())
        film_params  = (
            list(self._model.film.parameters())
            + list(self._model.classifier.parameters())
        )

        optimizer = torch.optim.AdamW([
            {"params": bert_params, "lr": self.lr},
            {"params": film_params, "lr": self.film_lr},
        ], weight_decay=0.01, eps=1e-8)

        n_bert_params = sum(p.numel() for p in bert_params)
        n_film_params = sum(p.numel() for p in film_params)
        print(f"  BERT body params : {n_bert_params:,} (lr={self.lr})")
        print(f"  FiLM+head params : {n_film_params:,} (lr={self.film_lr})")

        total_steps  = len(train_dl) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # -- Training loop with early stopping --------------------------------
        best_f1          = 0.0
        epochs_no_improve = 0          # patience counter
        best_epoch       = 0
        t0               = time.perf_counter()

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

            # -- track how far FiLM has moved from identity (gamma=1, beta=0) ----
            sample_meta = torch.tensor([[0.9, 0.5, 0.8]], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                g = self._model.film.to_gamma(sample_meta)[0]
                b = self._model.film.to_beta(sample_meta)[0]
            gamma_drift = float((g - 1.0).abs().mean().item())
            beta_drift  = float(b.abs().mean().item())

            print(f"\n  Epoch {epoch} summary:")
            print(f"    Train loss  : {avg_loss:.4f}")
            print(f"    Val F1      : {f1:.4f}  Acc: {metrics['accuracy']:.4f}  "
                  f"AUC: {metrics['auc_roc']:.4f}")
            print(f"    FiLM drift  : |gamma-1|={gamma_drift:.4f}  |beta|={beta_drift:.4f}  "
                  f"(0.0000 = still identity)\n")

            if f1 > best_f1:
                best_f1           = f1
                best_epoch        = epoch
                epochs_no_improve = 0
                # Save best model weights in memory
                import copy
                self._best_weights = copy.deepcopy(self._model.state_dict())
                print(f"    ✓ New best! Saved checkpoint (epoch {epoch})")
            else:
                epochs_no_improve += 1
                print(f"    No improvement for {epochs_no_improve}/{self.patience} epochs")
                if epochs_no_improve >= self.patience:
                    print(f"\n  Early stopping triggered at epoch {epoch}!")
                    print(f"  Best epoch was {best_epoch} with val F1 = {best_f1:.4f}")
                    break

        self._train_time = time.perf_counter() - t0
        self._trained    = True

        # Restore the BEST checkpoint (not the last epoch!)
        if self._best_weights is not None:
            self._model.load_state_dict(self._best_weights)
            print(f"\n  ✓ Restored best model from epoch {best_epoch}")

        print(f"  Training complete -- {self._train_time/60:.1f} min  "
              f"Best val F1: {best_f1:.4f} at epoch {best_epoch}/{epoch}")
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

    # -- inspect learned modulation (nice for discussion!) ---------------------

    def inspect_film(self, metadata: list) -> dict:
        """
        Show what gamma/beta a given metadata vector produces.
        Useful to demonstrate to the professor HOW metadata
        is actually reshaping BERT's representation.
        """
        self._check_fitted()
        meta_t = torch.tensor([metadata], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            gamma = self._model.film.to_gamma(meta_t)[0]
            beta  = self._model.film.to_beta(meta_t)[0]
        return {
            "metadata":   metadata,
            "gamma_mean": float(gamma.mean().item()),
            "gamma_std":  float(gamma.std().item()),
            "beta_mean":  float(beta.mean().item()),
            "beta_std":   float(beta.std().item()),
        }

    # -- persistence ----------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(directory, "model.pt"))
        self._tokenizer.save_pretrained(directory)
        print(f"  Saved BERT+FiLM model to {directory}")

    # -- helpers ----------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError("Call fit() before inference.")

    def __repr__(self) -> str:
        status = (f"trained in {self._train_time/60:.1f}min" if self._trained else "not trained")
        return f"BERTFiLMClassifier({self.MODEL_NAME}, FiLM fusion, {status})"


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
    SAVE  = os.environ.get("OUTPUT_DIR", os.path.join(BASE, "results", "bert_film_model"))

    print("=== BERT + FiLM Classifier -- LIAR Dataset ===")
    print(f"CUDA available: {torch.cuda.is_available()}\n")

    # -- Train ---------------------------------------------------------------
    clf = BERTFiLMClassifier(
        epochs     = 100,   # max epochs (Professor Weber's suggestion)
        batch_size = 16,
        lr         = 2e-5,  # BERT body -- slow
        film_lr    = 1e-3,  # FiLM + head -- fast (differential LR)
        patience   = 5,     # stop if no improvement for 5 epochs
    )
    clf.fit(TRAIN, VALID)

    # -- Evaluate ---------------------------------------------------------------
    results = clf.evaluate(TEST)
    print_report("BERT + FiLM -- Test Results", results)

    # -- Save ---------------------------------------------------------------
    clf.save(SAVE)

    # -- Demo predictions + FiLM inspection ---------------------------------------
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

    # -- Inspect how different metadata reshapes BERT (nice for the meeting!) -----
    print("\n=== FiLM Modulation Inspection ===")
    print("  (shows HOW different metadata values reshape BERT's 768 numbers)")
    test_metas = [
        ("Reliable speaker (low fake rate)",   [0.1, 0.5, 0.8]),
        ("Neutral / unknown speaker",          [0.5, 0.5, 0.5]),
        ("Unreliable speaker (high fake rate)",[0.9, 0.5, 0.8]),
    ]
    for label, meta in test_metas:
        info = clf.inspect_film(meta)
        print(f"  {label}: metadata={meta}")
        print(f"    gamma: mean={info['gamma_mean']:.3f} std={info['gamma_std']:.3f}  "
              f"(1.0 = no scaling)")
        print(f"    beta : mean={info['beta_mean']:.3f} std={info['beta_std']:.3f}  "
              f"(0.0 = no shift)")

    # -- Comparison ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  COMPARISON -- Differential LR vs uniform LR FiLM")
    print(f"{'='*60}")
    print(f"  BERT text-only                  : F1 = 0.6724")
    print(f"  BERT + metadata (concat)        : F1 = 0.6743")
    print(f"  BERT + FiLM (uniform lr, 3ep)   : F1 = 0.7029  <- previous run")
    print(f"  BERT + FiLM (diff lr, this run) : F1 = {results['f1']:.4f}")
    diff_vs_film1 = results['f1'] - 0.7029
    sign1 = "+" if diff_vs_film1 >= 0 else ""
    print(f"  Diff-LR vs uniform-LR FiLM      : {sign1}{diff_vs_film1:.4f}")
    print(f"  {'-'*56}")
    print(f"  MLP (TF-IDF + metadata)         : F1 = 0.7542  <- current best")
    diff_vs_mlp = results['f1'] - 0.7542
    sign2 = "+" if diff_vs_mlp >= 0 else ""
    print(f"  This run vs MLP                 : {sign2}{diff_vs_mlp:.4f}")
    print(f"{'='*60}")

    # -- Save results JSON ---------------------------------------------------------
    results_path = os.path.join(SAVE, "bert_film_results.json")
    os.makedirs(SAVE, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"\n{clf}")