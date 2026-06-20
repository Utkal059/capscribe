"""Multi-filing evaluation harness — the headline robustness deliverable.

One real filing isn't enough evidence. This runs extraction across a corpus of
real DRHPs and reports per-filing AND aggregate precision / recall / F1, so a
regression on filing B shows up even if filing A still scores 1.0.

How it works:
  1. Always scores the committed baseline pair (the Ola Electric DRHP:
     fixtures/ola_drhp_extracted.json vs fixtures/ola_drhp_gold.json).
  2. Extracts every PDF found in fixtures/real_drhps/<name>.pdf, writing
     <name>.predicted.json beside it.
  3. If fixtures/real_drhps/<name>.gold.json exists (evaluate.py format:
     {"capital_events": [{"event_type", "date"}, ...]}), it is scored; if not,
     the filing is listed as "no gold set" rather than scored — numbers are
     never invented.

Usage:
    python eval_corpus.py

Add a filing: drop <name>.pdf into fixtures/real_drhps/, run this once to
generate <name>.predicted.json, hand-verify it against the filing, then save
the verified events as <name>.gold.json and re-run to get a real score.
"""
from __future__ import annotations

import json
from pathlib import Path

from evaluate import evaluate
from table_extractor import TableExtractor

ROOT = Path(__file__).parent
CORPUS = ROOT / "fixtures" / "real_drhps"

# (label, predicted_json, gold_json) pairs that are committed and always scored.
BASELINE = [
    (
        "Ola Electric DRHP (baseline)",
        ROOT / "fixtures" / "ola_drhp_extracted.json",
        ROOT / "fixtures" / "ola_drhp_gold.json",
    ),
]


def _extract_to_json(doc: Path) -> Path:
    """Extract events from a filing (PDF or Markdown) and write
    <name>.predicted.json beside it."""
    if doc.suffix.lower() in (".md", ".markdown"):
        from markdown_extractor import extract_events_from_md_file
        events = extract_events_from_md_file(doc)
    else:
        events = TableExtractor().extract_events(doc)
    out = doc.with_suffix(".predicted.json")
    out.write_text(
        json.dumps({"source_file": doc.name, "capital_events": events},
                   indent=2, default=str),
        encoding="utf-8",
    )
    return out


def _n_events(pred_json: Path) -> int:
    return len(json.loads(pred_json.read_text(encoding="utf-8")).get("capital_events", []))


def main() -> int:
    rows: list[tuple[str, int, str, str]] = []  # (label, n_pred, n_gold, metrics)
    agg = {"tp": 0, "fp": 0, "fn": 0}
    scored = 0

    def score(label: str, pred: Path, gold: Path | None) -> None:
        nonlocal scored
        n_pred = _n_events(pred)
        if gold is None or not gold.exists():
            rows.append((label, n_pred, "—", "no gold set"))
            return
        r = evaluate(str(pred), str(gold))
        agg["tp"] += r["true_positives"]
        agg["fp"] += r["false_positives"]
        agg["fn"] += r["false_negatives"]
        scored += 1
        rows.append((label, n_pred, str(r["gold"]),
                     f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}"))

    # 1) committed baselines
    for label, pred, gold in BASELINE:
        if pred.exists():
            score(label, pred, gold)

    # 2) any real DRHP filings the user has dropped in (PDF or Markdown)
    docs = sorted(
        p for ext in ("*.pdf", "*.md", "*.markdown")
        for p in (CORPUS.glob(ext) if CORPUS.exists() else [])
    )
    for doc in docs:
        print(f"extracting {doc.name} …")
        pred = _extract_to_json(doc)
        score(doc.stem, pred, doc.with_suffix(".gold.json"))

    # 3) report
    print(f"\n{'filing':40} {'pred':>5} {'gold':>5}  metrics")
    print("-" * 78)
    for label, n_pred, n_gold, metrics in rows:
        print(f"{label[:40]:40} {n_pred:>5} {n_gold:>5}  {metrics}")
    print("-" * 78)
    if agg["tp"] + agg["fp"] and agg["tp"] + agg["fn"]:
        p = agg["tp"] / (agg["tp"] + agg["fp"])
        rec = agg["tp"] / (agg["tp"] + agg["fn"])
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        print(f"AGGREGATE over {scored} gold-labelled filing(s): "
              f"P={p:.3f} R={rec:.3f} F1={f1:.3f}")
    else:
        print("No gold sets present yet — add fixtures/real_drhps/<name>.gold.json to score.")
    if not docs:
        print(f"\n(Tip: drop real DRHP PDFs or .md files into {CORPUS} to evaluate beyond the baseline.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
