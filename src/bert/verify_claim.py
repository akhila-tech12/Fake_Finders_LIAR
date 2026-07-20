"""
verify_claim.py -- standalone FEVER-style claim verification (CLI).

Pipeline: Wikipedia search (v2 retriever, top-10 candidate pages) -> NLI
entailment between every candidate sentence (premise) and the claim
(hypothesis) -> SUPPORTED / REFUTED / NOT ENOUGH INFO. This is the reasoning
step the LIAR classifiers lack: they pattern-match claim style, this actually
compares the claim against evidence.

Run with venv_bert (needs torch, transformers, sentencepiece):

    venv_bert/bin/python src/bert/verify_claim.py "Pakistan landed on the moon"
    venv_bert/bin/python src/bert/verify_claim.py "..." --evidence "own text"
    venv_bert/bin/python src/bert/verify_claim.py "..." --json

The NLI model (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, ~370 MB) is
downloaded to the HuggingFace cache on first use; override with $NLI_MODEL.

Design notes (standard FEVER shape: retrieval -> sentence selection -> NLI):
  * Sentence selection is essential, not an optimisation: NLI-scoring *every*
    sentence saturates max-contradiction at ~100% because some off-topic
    sentence always "contradicts" (water boils at 100 °C vs the Fahrenheit
    page's 212 °F). Only the TOP_SENTENCES most claim-relevant sentences
    (lexical overlap) are scored.
  * Entailment wins near-ties against contradiction (MARGIN): NLI models
    produce pseudo-contradictions from non-exclusive facts ("the first crewed
    landing was Apollo 11" does not exclude "India landed"), while a sentence
    that fully entails a claim with its named entity is much stronger evidence.
  * If nothing clears THRESHOLD the verdict is NOT ENOUGH INFO — the honest
    answer, never a guess.

Known failure mode (say it in the defense, don't hide it): claims that hinge
on numeric conversion or contrast clauses ("water boils at 100 °C" vs a
premise adding "...but at 93.4 °C at altitude", or 212 °F = 100 °C) can be
mis-refuted — base-size NLI models do not do arithmetic. A larger NLI model
(export NLI_MODEL=MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
reduces but does not eliminate this.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

NLI_MODEL     = os.environ.get("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
THRESHOLD     = 0.60   # min probability for a SUPPORTED/REFUTED verdict
MARGIN        = 0.15   # entailment beats contradiction within this margin
N_CANDIDATES  = 10     # Wikipedia pages to scan
TOP_SENTENCES = 12     # claim-relevant sentences that reach the NLI model
MIN_SENT_LEN  = 25

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_nli():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.eval()
    return tok, model, torch


def _premises(claim: str, candidates: list[dict]) -> list[tuple[str, str]]:
    """
    Sentence selection: the TOP_SENTENCES sentences most lexically relevant
    to the claim, drawn from all candidate pages. The page title is prepended
    to each sentence for scoring so "It landed in 2023" on the Chandrayaan-3
    page still counts as being about Chandrayaan.
    """
    from bert.evidence_retriever import get_reranker
    pool: list[tuple[str, str]] = []
    for cand in candidates:
        title = cand.get("title", "")
        for sent in _SENT_SPLIT.split(cand.get("extract", "").strip()):
            if len(sent) >= MIN_SENT_LEN:
                pool.append((title, sent))
    if not pool:
        return []
    scores = get_reranker("lexical").scores(
        claim, [f"{title}. {sent}" for title, sent in pool])
    ranked = sorted(zip(scores, range(len(pool))), reverse=True)
    return [pool[i] for _, i in ranked[:TOP_SENTENCES]]


def verify(claim: str, candidates: list[dict], tok, model, torch) -> dict:
    """
    NLI-score the claim against every candidate sentence; the strongest
    entailment or contradiction decides (entailment favoured by MARGIN).
    Returns the verdict plus the deciding sentence and its source page, so a
    demo can show *why* ("refuted by: 'India ... never participated ...'").
    """
    premises = _premises(claim, candidates)
    if not premises:
        return {"verdict": "NOT ENOUGH INFO", "confidence": 0.0,
                "max_entailment": 0.0, "max_contradiction": 0.0,
                "deciding_evidence": "", "source_page": ""}

    label_ids = {lbl.lower(): i for i, lbl in model.config.id2label.items()}
    scored: list[tuple[float, float, str, str]] = []   # (ent, con, sent, title)

    for start in range(0, len(premises), 16):          # mini-batches for CPU
        batch = premises[start:start + 16]
        enc = tok([p for _, p in batch], [claim] * len(batch),
                  return_tensors="pt", padding=True,
                  truncation=True, max_length=512)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        for (title, sent), row in zip(batch, probs):
            scored.append((float(row[label_ids["entailment"]]),
                           float(row[label_ids["contradiction"]]), sent, title))

    def _pick(idx: int) -> tuple[float, str, str]:
        """Best (prob, sentence, page) for one label; among near-max ties,
        prefer the sentence that shares the most claim words — 'India ...
        never participated in the FIFA World Cup' explains a REFUTED verdict,
        a same-scored sentence about England does not."""
        mx = max(s[idx] for s in scored)
        near = [s for s in scored if s[idx] >= mx - 0.05]
        toks = {w for w in re.sub(r"[^a-z0-9\s]", " ", claim.lower()).split()
                if len(w) >= 4}
        overlap = lambda s: sum(1 for w in toks if w in f"{s[3]} {s[2]}".lower())
        p_, c_, sent, title = max(near, key=lambda s: (overlap(s), s[idx]))
        return mx, sent, title

    ent, ent_sent, ent_page = _pick(0)
    con, con_sent, con_page = _pick(1)
    if ent >= THRESHOLD and ent >= con - MARGIN:
        verdict, score, sent, page = "SUPPORTED", ent, ent_sent, ent_page
    elif con >= THRESHOLD:
        verdict, score, sent, page = "REFUTED", con, con_sent, con_page
    else:
        verdict, score, sent, page = "NOT ENOUGH INFO", max(ent, con), "", ""

    return {"verdict": verdict, "confidence": round(score, 3),
            "max_entailment": round(ent, 3), "max_contradiction": round(con, 3),
            "deciding_evidence": sent[:300], "source_page": page}


def main() -> None:
    ap = argparse.ArgumentParser(description="FEVER-style claim verification")
    ap.add_argument("claim")
    ap.add_argument("--evidence", default=None,
                    help="skip retrieval, verify against this text")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if args.evidence is not None:
        candidates = [{"title": "(provided)", "extract": args.evidence}]
    else:
        from bert.evidence_retriever import EvidenceRetriever
        retriever  = EvidenceRetriever(reranker="lexical", cache_path=None,
                                       n_candidates=N_CANDIDATES)
        candidates = retriever.retrieve_candidates(args.claim)

    tok, model, torch = load_nli()
    result = verify(args.claim, candidates, tok, model, torch)
    result["pages_scanned"] = [c.get("title", "") for c in candidates]

    if args.json:
        print(json.dumps(result))
        return

    print(f"\nCLAIM   : {args.claim}")
    print(f"PAGES   : {', '.join(result['pages_scanned']) or '(none found)'}")
    print(f"\nVERDICT : {result['verdict']}  (confidence {result['confidence']:.0%})")
    print(f"          entailment {result['max_entailment']:.0%} | "
          f"contradiction {result['max_contradiction']:.0%}")
    if result["deciding_evidence"]:
        print(f"BECAUSE : \"{result['deciding_evidence']}\"")
        print(f"          — Wikipedia: {result['source_page']}")


if __name__ == "__main__":
    main()
