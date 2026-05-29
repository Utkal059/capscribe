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


def evaluate(pred_path: str, gold_path: str) -> dict:
    pred = json.loads(Path(pred_path).read_text(encoding="utf-8")).get("capital_events", [])
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8")).get("capital_events", [])

    pred_keys = {_key(e) for e in pred}
    gold_keys = {_key(e) for e in gold}

    tp = len(pred_keys & gold_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "predicted": len(pred),
        "gold": len(gold),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "missed": sorted(str(k) for k in (gold_keys - pred_keys)),
        "spurious": sorted(str(k) for k in (pred_keys - gold_keys)),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python evaluate.py <predicted.json> <gold.json>")
        raise SystemExit(1)
    report = evaluate(sys.argv[1], sys.argv[2])
    print(json.dumps(report, indent=2))
