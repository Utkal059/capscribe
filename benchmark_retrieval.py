"""Retrieval benchmark: dense-only vs hybrid (BM25 + dense + RRF).

Reports nDCG@5 and MRR for a fixed query set whose relevant answers are
known from the gold fixture. "Dense" routes through the hybrid retriever
with ``alpha=1.0`` (pure vector lean) and "hybrid" uses the auto alpha
heuristic, so both share identical plumbing and only the fusion differs.

Run:
    python benchmark_retrieval.py

Uses the real local embedding model, so the first run downloads the ONNX
MiniLM (~80 MB) and then runs fully offline.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from retrieval import EventStore, HybridRetriever, load_events

# (query, target (event_type, date)) — relevance judged against the gold key.
BENCH_QUERIES: list[tuple[str, tuple[str, str]]] = [
    ("promoter allotment in 2017", ("allotment", "2017-02-03")),
    ("preferential allotment at Rs. 154.20", ("allotment", "2019-06-21")),
    ("bonus issue ratio 5:1", ("bonus_issue", "2021-08-01")),
    ("rights issue 1:4", ("rights_issue", "2022-03-15")),
    ("increase in authorised capital", ("authorised_capital_change", "2023-01-10")),
    ("ESOP allotment in 2023", ("allotment", "2023-09-30")),
]


def _hit_key(hit) -> tuple:
    return (hit.event.get("event_type"), hit.event.get("date"))


def first_relevant_rank(hits, target: tuple) -> int | None:
    """1-indexed rank of the first relevant hit, or None if absent."""
    for rank, hit in enumerate(hits, start=1):
        if _hit_key(hit) == target:
            return rank
    return None


def ndcg_at_k(rank: int | None, k: int = 5) -> float:
    """Binary-relevance nDCG@k for a single relevant item at ``rank``."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)  # IDCG = 1 for a single relevant item


def mrr(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def run(alpha: float | None) -> dict:
    store = EventStore(in_memory=True)
    store.index_events(load_events("fixtures/sample_events.json"))
    hybrid = HybridRetriever(store)
    ndcgs, rrs = [], []
    for query, target in BENCH_QUERIES:
        hits = hybrid.search(query, k=5, alpha=alpha)
        rank = first_relevant_rank(hits, target)
        ndcgs.append(ndcg_at_k(rank, 5))
        rrs.append(mrr(rank))
    return {
        "ndcg@5": round(sum(ndcgs) / len(ndcgs), 4),
        "mrr": round(sum(rrs) / len(rrs), 4),
    }


def main() -> None:
    dense = run(alpha=1.0)   # pure vector
    hybrid = run(alpha=None)  # auto BM25+vector fusion
    report = {
        "queries": len(BENCH_QUERIES),
        "dense": dense,
        "hybrid": hybrid,
        "delta": {
            "ndcg@5": round(hybrid["ndcg@5"] - dense["ndcg@5"], 4),
            "mrr": round(hybrid["mrr"] - dense["mrr"], 4),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
