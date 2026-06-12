# CHANGES

Six targeted improvements mapping to the role's requirements (hybrid retrieval,
table extraction, OCR, agentic verification, citations, broader document types).
This was an **audit-and-fill** pass: parts of the spec were already implemented
under different filenames, so existing working code was kept and only the genuine
gaps were added. All pre-existing tests still pass; new tests were added for the
new logic.

## 1. Files changed / added

### New files
| File | What it does |
|------|--------------|
| `table_extractor.py` | `TableChunk` dataclass, `TableExtractor` (pdfplumber, merged-cell aware, infers section from above-median-font headings), pure `table_to_events()` pattern-matcher for allotment / bonus / authorised-capital tables, and `merge_with_dedup()` that prefers table-extracted events over LLM ones. |
| `verification.py` | Contradiction detection: `check_timeline`, `check_capital_continuity`, `check_bonus_arithmetic`, plus `verify_consistency()` / `verify_report()` returning `VerificationResult` / `Issue` / `VerificationReport`. Pure, offline. |
| `benchmark_retrieval.py` | Reports **nDCG@5** and **MRR** for dense (alpha=1.0) vs hybrid (auto fusion) over a fixed query set judged against the gold fixture. |
| `tests/test_verification.py` | Clean dataset + planted capital-continuity break + conflicting same-date events + bonus-arithmetic deviation + ratio parser + report summary. |
| `tests/test_table_extractor.py` | Value parsing (Indian grouping, dates), per-family column detection, dedup — all on synthetic `TableChunk`s (no PDF needed). |

### Modified files
| File | Change |
|------|--------|
| `schema.py` | Added event types `dividend_declaration`, `share_repurchase`, `warrant_exercise` with their fields; added `SourceProvenance` model and an **optional** `source_provenance` field (default `None`, backward-compatible). |
| `retrieval.py` | `event_to_text()` now renders the new event-type fields so they are searchable. |
| `ocr.py` | Added `process_document()` — whole-PDF extraction with a per-page method (native/ocr) summary. |
| `api.py` | Added `POST /verify`, `POST /ingest` (PDF upload → OCR fallback → table extraction → per-page method summary), `GET /ingest/status/{job_id}` (in-memory job registry). No change to existing endpoints. |
| `evaluate.py` | Added `per_event_type` and `by_extraction_method` breakdowns; kept all original keys. |
| `prompts/system.text` | Added `Dividend Declaration` / `Warrant Exercise` to the taxonomy and an extended-fields guidance block. |
| `requirements-api.txt` | Pinned the already-used `rank-bm25`, `pdfplumber`, `pytesseract`, `python-multipart`. |
| `fixtures/sample_events.json` | Added 9 synthetic new-type events (3 each), each with full page/snippet provenance, marked `"synthetic": true`. |
| `fixtures/gold_events.json` | Added 9 matching gold entries, marked `"synthetic": true`. |
| `frontend/index.html` | Citation pills now show `(p. X)` with `data-page` / `data-section` attributes and a `§section — excerpt` tooltip. (Visual design otherwise unchanged.) |
| `tests/test_api.py` | Added `/verify` and `/ingest` (reject non-PDF, unknown-job) tests. |

## 2. New dependencies (pinned in `requirements-api.txt`)
- `rank-bm25==0.2.2` — sparse BM25 layer for hybrid retrieval (already used by `retrieval.py`).
- `pdfplumber==0.11.9` — table extraction and the OCR scan-detection / page rendering.
- `pytesseract==0.3.13` — OCR of scanned pages (needs the system `tesseract` binary; degrades gracefully when absent — `GET /health` reports `ocr_available:false`).
- `python-multipart==0.0.29` — required by FastAPI for the `POST /ingest` file upload.

> Note: `pdf2image`/poppler are **not** required — `ocr.py` renders pages via
> `pdfplumber`'s `to_image()`, removing that system dependency.

## 3. New API endpoints
- `POST /verify` → full-corpus consistency report `{checked, consistent, issues[], by_check}`.
- `POST /ingest` (multipart PDF) → `{total_pages, ocr_page_count, text_page_count, page_methods[], events_extracted, events_by_method}`.
- `GET /ingest/status/{job_id}` → recorded result/status for an ingest job (404 if unknown).

## 4. New evaluation metrics
- `evaluate.py` → **`per_event_type`** (precision/recall/F1 per event type) so the
  synthetic new types don't dilute the headline score, and **`by_extraction_method`**
  (predicted / true-positives / precision per `table|text|llm|ocr`) to show where
  events came from.
- `benchmark_retrieval.py` → **nDCG@5** and **MRR** for dense vs hybrid, with the delta.

## 5. Deliberate deviations from the prompt (with rationale)
- **Verification is a standalone module + `/verify` endpoint, not a LangGraph
  `ToolNode` tool.** The existing agent is a linear `StateGraph`
  (retrieve→grade→synthesize→validate), not a ReAct ToolNode with
  `search_events`/`get_event_detail`. Bolting a tool loop on would have
  rearchitected (and risked) the working agent; a pure verification module gives
  the same contradiction-detection value with full test coverage and zero risk to
  existing behaviour.
- **Hybrid retrieval already existed** as `HybridRetriever` (RRF, `rrf_k=60`,
  weighted alpha) with a `/search` `alpha` parameter instead of a
  `retrieval_mode` string (`alpha=1.0` == dense). Left as-is; only the missing
  `benchmark_retrieval.py` was added.
- **OCR already existed** as `ocr.py` (not `ocr_pipeline.py`); table logic added as
  `table_extractor.py`. Extended in place rather than duplicated.

## Test status
`pytest tests/ -v` → **61 passed** (40 pre-existing + 21 new), no failures.
