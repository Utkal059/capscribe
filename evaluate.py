"""Offline evaluation harness for extraction quality.

Closes the "RAG / extraction evaluation" gap. Compares an extracted JSON
against a hand-labelled gold set on the event identity key (type + date),
reporting precision, recall and F1. Runs fully offline — no API, no model
downloads — so it can live in CI and gate regressions.

Usage:
    python evaluate.py output/sample_extracted.json fixtures/gold_events.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _key(ev: dict) -> tuple:
    return (ev.get("event_type"), ev.get("date"))


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def _extraction_method(ev: dict) -> str:
    """Where an event came from: table | text | llm | ocr (default 'llm')."""
    prov = ev.get("source_provenance") or {}
    return prov.get("extraction_method", "llm")


def per_type_breakdown(pred: list[dict], gold: list[dict]) -> dict:
    """Precision/recall/F1 per event_type, so new types don't dilute the headline."""
    types = sorted({e.get("event_type") for e in (pred + gold) if e.get("event_type")})
    out: dict[str, dict] = {}
    for t in types:
        p_keys = {_key(e) for e in pred if e.get("event_type") == t}
        g_keys = {_key(e) for e in gold if e.get("event_type") == t}
        out[t] = {
            "gold": len(g_keys),
            "predicted": len(p_keys),
            **_prf(len(p_keys & g_keys), len(p_keys - g_keys), len(g_keys - p_keys)),
        }
    return out


def per_method_breakdown(pred: list[dict], gold_keys: set) -> dict:
    """How many predicted events each extraction method produced, and its precision."""
    out: dict[str, dict] = {}
    for ev in pred:
        m = _extraction_method(ev)
        bucket = out.setdefault(m, {"predicted": 0, "true_positives": 0, "false_positives": 0})
        bucket["predicted"] += 1
        if _key(ev) in gold_keys:
            bucket["true_positives"] += 1
        else:
            bucket["false_positives"] += 1
    for bucket in out.values():
        denom = bucket["true_positives"] + bucket["false_positives"]
        bucket["precision"] = round(bucket["true_positives"] / denom, 3) if denom else 0.0
    return out


def evaluate(pred_path: str, gold_path: str) -> dict:
    pred = json.loads(Path(pred_path).read_text(encoding="utf-8")).get("capital_events", [])
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8")).get("capital_events", [])

    pred_keys = {_key(e) for e in pred}
    gold_keys = {_key(e) for e in gold}

    tp = len(pred_keys & gold_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)

    headline = _prf(tp, fp, fn)
    return {
        "predicted": len(pred),
        "gold": len(gold),
        **headline,
        # backward-compatible headline aliases
        "precision": headline["precision"],
        "recall": headline["recall"],
        "f1": headline["f1"],
        "per_event_type": per_type_breakdown(pred, gold),
        "by_extraction_method": per_method_breakdown(pred, gold_keys),
        "missed": sorted(str(k) for k in (gold_keys - pred_keys)),
        "spurious": sorted(str(k) for k in (pred_keys - gold_keys)),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python evaluate.py <predicted.json> <gold.json>")
        raise SystemExit(1)
    report = evaluate(sys.argv[1], sys.argv[2])
    print(json.dumps(report, indent=2))
