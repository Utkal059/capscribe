# CapScribe — Capital Event Intelligence

> Structured capital event extraction from DRHP/IPO filings using Claude AI.

[![Tests](https://github.com/Utkal059/capscribe/actions/workflows/tests.yml/badge.svg)](https://github.com/Utkal059/capscribe/actions)

CapScribe parses dense regulatory PDF documents (DRHPs, IPO prospectuses) and extracts structured capital event data — allotments, bonus issues, rights issues, and authorised capital changes — into clean, machine-readable JSON. Built for analysts, quant researchers, and fintech pipelines that need reliable signal from unstructured filings.

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

## Architecture

- **Extraction** — PDF parser producing structured JSON events per `schema.py`
- **Retrieval** — ChromaDB vector store with all-MiniLM-L6-v2 embeddings
- **Agent** — LangGraph ReAct loop over `search_events` / `get_event_detail` tools
- **API** — FastAPI service (`/health`, `/stats`, `/search`, `/ask`)
- **Frontend** — Financial terminal UI (dark theme, semantic search, extractive + LLM QA)

## Evaluation (sample DRHP)

| Metric | Score |
|---|---|
| Precision | 1.000 |
| Recall | 0.857 |
| F1 | 0.923 |

## Quickstart

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
pip install -r requirements-api.txt
uvicorn api:app --reload
# open http://localhost:8000
```

## Tests

```bash
pytest -q   # 15 passed
```

## Eval Harness

```bash
python evaluate.py fixtures/sample_events.json fixtures/gold_events.json
```

## License

MIT