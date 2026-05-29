# CapScribe — API, Retrieval & Eval Layer (setup)

This adds the four things the S45 JD asks for that the repo was missing:
a **FastAPI** service, a **vector store + semantic retrieval** layer (Chroma),
an **agentic RAG** ask-flow (LangGraph), and an offline **evaluation** harness +
**pytest** suite. The whole thing demos on your already-extracted
`fixtures/sample_events.json` with **zero API spend**. Only live PDF extraction
and `/ask` in `llm` mode call Claude.

## New files

| File | Purpose |
|---|---|
| `schema.py` | Pydantic models for events / extraction output |
| `config.py` | Typed settings (model, paths, top_k) from `.env` |
| `retrieval.py` | Chroma vector store + semantic search (local embeddings) |
| `agent.py` | LangGraph flow: retrieve → grade → synthesize → validate |
| `api.py` | FastAPI: `/health /stats /events /search /ask /index` |
| `evaluate.py` | Offline precision/recall/F1 vs a gold set |
| `frontend/index.html` | Single-page demo UI (search + ask) |
| `tests/` | Hermetic offline test suite (no API, no network) |
| `fixtures/` | Sample + gold event sets |
| `Dockerfile`, `Makefile`, `.github/workflows/tests.yml` | ops |

## Install (VS Code terminal)

```bat
pip install -r requirements-api.txt -r requirements-dev-additions.txt
```

(or merge those lines into your existing `requirements.txt` /
`requirements-dev.txt`, then `pip install -r requirements.txt`.)

## Run the tests — free, offline

```bat
pytest -q
```

## Run the evaluation — free, offline

```bat
python evaluate.py fixtures\sample_events.json fixtures\gold_events.json
```

## Start the API + open the UI — free (serves the sample file)

```bat
uvicorn api:app --reload --port 8000
```

Then open http://localhost:8000/ in your browser. Search and the
"extractive" ask mode cost nothing. Switch the UI toggle to `llm` only if you
want Claude to write prose (small credit cost).

## Point it at a real extraction (still free to serve)

```bat
curl -X POST http://localhost:8000/index -H "Content-Type: application/json" -d "{\"events_path\": \"output/sample_extracted.json\"}"
```

The first server start downloads a small local embedding model (~80 MB) once;
after that everything is offline except live extraction / llm answers.
