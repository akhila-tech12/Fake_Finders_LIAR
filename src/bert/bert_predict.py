"""
bert_predict.py
===============
Standalone BERT+FiLM prediction script.
Called as subprocess from app.py using venv_bert Python.

Usage:
    /path/to/venv_bert/bin/python bert_predict.py \
        "statement text here" \
        "model_path" \
        "0.5,0.5,0.5"

Outputs JSON to stdout:
    {"label": "FAKE", "confidence": 0.87, "is_fake": true}
"""

import sys
import json
import os

def predict(statement, model_path, meta_str):
    import torch
    import torch.nn as nn
    from transformers import BertTokenizerFast, BertModel

    meta = [float(x) for x in meta_str.split(",")]

    class FiLMLayer(nn.Module):
        def __init__(self, meta_dim=3, bert_dim=768, hidden=32):
            super().__init__()
            self.to_gamma = nn.Sequential(
                nn.Linear(meta_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, bert_dim))
            self.to_beta = nn.Sequential(
                nn.Linear(meta_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, bert_dim))
        def forward(self, cls, meta):
            return self.to_gamma(meta) * cls + self.to_beta(meta)

    class BERTWithFiLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert       = BertModel.from_pretrained(model_path)
            self.film       = FiLMLayer()
            self.dropout    = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, 2)
        def forward(self, input_ids, attention_mask,
                    token_type_ids, metadata):
            out    = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            pooled = out.pooler_output
            modded = self.film(pooled, metadata)
            return self.classifier(self.dropout(modded))

    device    = torch.device("cpu")
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model     = BERTWithFiLM().to(device)

    weights = os.path.join(model_path, "model.pt")
    if os.path.exists(weights):
        model.load_state_dict(
            torch.load(weights, map_location=device)
        )
    model.eval()

    # If evidence provided, combine with statement (RAG style)
    evidence = sys.argv[4] if len(sys.argv) > 4 else ""

    # model_config.json (written by bert_film_rag.py) switches on text-PAIR
    # encoding to match how that model was trained. Absent (bert_film_v3),
    # the original string-concat path runs unchanged.
    cfg_path = os.path.join(model_path, "model_config.json")
    cfg      = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    max_len = int(cfg.get("max_len", 256))

    if cfg.get("pair_encoding") and evidence:
        enc = tokenizer(
            statement, evidence, truncation="only_second",
            padding="max_length", max_length=max_len, return_tensors="pt",
        )
    else:
        combined = f"{statement} [SEP] {evidence}" if evidence else statement
        enc = tokenizer(
            combined, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
    meta_t = torch.tensor([meta], dtype=torch.float32)

    with torch.no_grad():
        logits = model(
            enc["input_ids"], enc["attention_mask"],
            enc["token_type_ids"], meta_t,
        )
        probs = torch.softmax(logits, dim=1)
        pred  = int(torch.argmax(probs).item())
        conf  = float(probs[0][pred].item())

    return {
        "label"     : "FAKE" if pred == 1 else "REAL",
        "is_fake"   : pred == 1,
        "confidence": conf,
    }


if __name__ == "__main__":
    statement  = sys.argv[1]
    model_path = sys.argv[2]
    meta_str   = sys.argv[3] if len(sys.argv) > 3 else "0.5,0.5,0.5"

    result = predict(statement, model_path, meta_str)
    print(json.dumps(result))