# CapScribe — Capital Event Intelligence

> Structured capital event extraction from DRHP/IPO filings using Claude AI.

[![Tests](https://github.com/Utkal059/capscribe/actions/workflows/tests.yml/badge.svg)](https://github.com/Utkal059/capscribe/actions)

CapScribe parses dense regulatory PDF documents (DRHPs, IPO prospectuses, annual reports) and extracts structured capital event data — allotments, bonus issues, rights issues, authorised-capital changes, dividends, buybacks, and warrant exercises — into clean, machine-readable JSON. Every event is traceable back to the exact page and verbatim text it came from. Built for analysts, quant researchers, and fintech pipelines that need reliable signal from unstructured filings.

## Demo

### Semantic Search
Natural language query matching across capital event types — no keyword syntax required.

![Semantic Search](docs/demo_search.png)

### Ask the Filing — LLM Mode
Structured investor intelligence synthesised from raw DRHP events via Claude.

![Ask the Filing](docs/demo_ask.png)

## What it extracts

| Event Type | Fields Captured |
|---|---|
| Allotments | Date, shares, face value, issue price, consideration, allottee category |
| Bonus Issues | Date, ratio, pre/post share count |
| Rights Issues | Date, ratio, price, record date |
| Authorised Capital Changes | Date, from/to amount, resolution type |
| Dividends | Date, amount per share, record/payment date, total outflow |
| Share Repurchases | Date, shares bought back, remaining authority |
| Warrant Exercises | Date, warrants exercised, exercise price |

## Architecture

- **Extraction** — PDF parser producing structured JSON events per `schema.py`
- **Tables** — direct `pdfplumber` table extraction (`table_extractor.py`), merged-cell aware, preferred over LLM events on a fuzzy match
- **OCR** — scanned-page fallback (`ocr.py`) via tesseract, degrades gracefully when the binary is absent
- **Retrieval** — hybrid BM25 + ChromaDB vectors fused with reciprocal rank fusion (`retrieval.py`); an auto-alpha heuristic leans toward BM25 for numeric queries
- **Agent** — observable LangGraph state machine `retrieve → grade → synthesize → validate` (`agent.py`)
- **Verification** — deterministic contradiction checks: timeline / capital-continuity / bonus-arithmetic (`verification.py`)
- **Report** — source-backed capital-history brief with inline page citations (`report.py`)
- **API** — FastAPI service (`api.py`)
- **Frontend** — financial terminal UI (dark theme, semantic search, extractive + LLM QA, citation pills)

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness; reports retrieval mode and `ocr_available` |
| `GET /stats` | event counts by type |
| `GET /events` | list events (optional `?event_type=` & `?limit=`) |
| `POST /search` | hybrid search `{query, k, alpha}` |
| `POST /ask` | agentic RAG `{question, mode}` (extractive / llm) |
| `POST /verify` | full-corpus contradiction report |
| `POST /report` | source-backed capital-history brief `{mode, title}` |
| `POST /ingest` | PDF upload → OCR fallback → table extraction (background job; `?llm=true` adds a bounded Claude pass) |
| `GET /ingest/status/{job_id}` | poll a running ingest job for its result |
| `POST /ingest/{job_id}/index` | promote an ingested filing to the live corpus (search / ask / verify / report) |
| `POST /index` | rebuild the index from a different extracted JSON |

## Evaluation

**Real document — Ola Electric DRHP** (allotment history tables, hand-verified gold set in `fixtures/ola_drhp_gold.json`):

| Metric | Score |
|---|---|
| Precision | 1.000 |
| Recall | 0.857 |
| F1 | 0.923 |

```bash
python evaluate.py fixtures/ola_drhp_extracted.json fixtures/ola_drhp_gold.json
```

Zero false positives — acquisition/transfer tables are correctly *not* typed as allotments (a mixed "date of allotment/transfer" table is filtered by its `nature of transaction` column). The single miss is a genuine allotment whose share count appears only in prose (no numeric column); the optional `POST /ingest?llm=true` pass recovers that class of event.

**Synthetic fixture** (full event-type coverage, known answers): Precision **1.000** · Recall **0.938** · F1 **0.968** via `python evaluate.py fixtures/sample_events.json fixtures/gold_events.json`.

Per-event-type and per-extraction-method breakdowns are emitted by `evaluate.py`; retrieval quality (nDCG@5, MRR — dense vs hybrid) by `benchmark_retrieval.py`.

## Quickstart

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY (only needed for llm modes)
pip install -r requirements.txt -r requirements-api.txt
uvicorn api:app --reload
# open http://localhost:8000
```

The index builds on startup from `fixtures/sample_events.json` using local embeddings, so search / ask / verify / report all run with **zero API spend**. Only `mode="llm"` calls Claude.

## Tests

```bash
pip install -r requirements.txt -r requirements-api.txt -r requirements-dev.txt
pytest -q   # 76 passed
```

## Eval & Benchmarks

```bash
python evaluate.py fixtures/sample_events.json fixtures/gold_events.json
python benchmark_retrieval.py          # nDCG@5 + MRR, dense vs hybrid
```

## License

MIT
