"""
capscribe — extractor.py
Chunked DRHP capital event extraction using the Anthropic API.

Fixes applied (vs original):
  - model string updated to claude-sonnet-4-6
  - import io moved to top-level (was inside loop)
  - --out CLI arg now actually respected
  - Exponential-backoff retry via tenacity (Tier 2)
  - Per-chunk checkpoint files so a crashed run resumes (Tier 2)
  - Rate-limit / server-error handling (Tier 2)
  - max_tokens made configurable via MAX_TOKENS env var (Tier 2)
  - Overlapping chunks (OVERLAP pages) so boundary events aren't missed (Tier 3)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anthropic
import pypdf
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL = os.getenv("CAPSCRIBE_MODEL", "claude-sonnet-4-6")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2"))  # dense DRHP tables need small chunks
OVERLAP = int(os.getenv("OVERLAP", "1"))          # pages shared between chunks
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))  # was hardcoded 4096

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "system.text").read_text(encoding="utf-8")

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


# ── Retry decorator ────────────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError)
    ),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(2),
    reraise=True,
)
def call_claude(text: str) -> list[dict]:
    """Send one chunk of PDF text to Claude and return parsed events."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = response.content[0].text.strip()

    # Strip markdown fences if Claude wrapped the JSON
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    try:
        events = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [warn] JSON parse failed for chunk, skipping. Raw: {raw[:200]}")
        return []

    if isinstance(events, dict) and "capital_events" in events:
        return events["capital_events"]
    if isinstance(events, list):
        return events
    return []


# ── PDF helpers ────────────────────────────────────────────────────────────────
def extract_pages(pdf_path: Path) -> list[str]:
    """Return list of page text strings (one per page, 0-indexed)."""
    reader = pypdf.PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def make_chunks(pages: list[str], first_page: int = 1) -> list[tuple[int, int, str]]:
    """
    Yield (start_page, end_page, text) tuples with OVERLAP-page overlap
    so events straddling chunk boundaries aren't lost.

    Every page in the chunk text is prefixed with a `[PAGE n]` marker
    (n is the true 1-indexed PDF page, honouring `first_page`) so the
    model can attribute each extracted event to its source page.
    """
    chunks = []
    n = len(pages)
    start = 0
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        text = "\n".join(
            f"[PAGE {first_page + i}]\n{pages[i]}" for i in range(start, end)
        )
        chunks.append((first_page + start, first_page + end - 1, text))
        if end == n:
            break
        start = end - OVERLAP                    # overlap for next chunk
    return chunks


# ── Provenance validation ──────────────────────────────────────────────────────
SNIPPET_MATCH_THRESHOLD = 0.85
FALLBACK_CONFIDENCE = 0.7
FALLBACK_SNIPPET_CHARS = 200


def _normalise_ws(text: str) -> str:
    return " ".join(text.split())


def snippet_match_ratio(snippet: str, page_text: str) -> float:
    """Best fuzzy-match ratio of `snippet` against any window of `page_text`.

    Exact (whitespace-normalised) substrings score 1.0. Otherwise a sliding
    window of the snippet's length is compared with difflib.SequenceMatcher
    and the best ratio is returned.
    """
    from difflib import SequenceMatcher

    snippet_n = _normalise_ws(snippet).lower()
    page_n = _normalise_ws(page_text).lower()
    if not snippet_n or not page_n:
        return 0.0
    if snippet_n in page_n:
        return 1.0
    win = len(snippet_n)
    if win >= len(page_n):
        return SequenceMatcher(None, snippet_n, page_n).ratio()
    step = max(1, win // 2)
    best = 0.0
    for i in range(0, len(page_n) - win + 1, step):
        ratio = SequenceMatcher(None, snippet_n, page_n[i : i + win]).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.99:
                break
    return best


def attach_provenance(events: list[dict], pages_by_number: dict[int, str]) -> list[dict]:
    """Validate per-event provenance against the source pages, in place.

    - A `source_snippet` must genuinely occur on its claimed page
      (fuzzy ratio >= SNIPPET_MATCH_THRESHOLD). If it does not, the snippet
      is replaced with the first FALLBACK_SNIPPET_CHARS characters of the
      page text and `confidence` drops to FALLBACK_CONFIDENCE.
    - A `page_number` outside the known pages is removed (set to None)
      together with its snippet — page numbers are never invented.
    """
    for ev in events:
        page = ev.get("page_number")
        if page is None:
            continue
        try:
            page = int(page)
        except (TypeError, ValueError):
            ev["page_number"] = None
            ev.pop("source_snippet", None)
            continue
        page_text = pages_by_number.get(page)
        if page_text is None:
            ev["page_number"] = None
            ev.pop("source_snippet", None)
            continue
        ev["page_number"] = page
        snippet = ev.get("source_snippet")
        if not snippet or snippet_match_ratio(str(snippet), page_text) < SNIPPET_MATCH_THRESHOLD:
            ev["source_snippet"] = _normalise_ws(page_text)[:FALLBACK_SNIPPET_CHARS]
            ev["confidence"] = min(float(ev.get("confidence", 1.0)), FALLBACK_CONFIDENCE)
        else:
            ev.setdefault("confidence", 1.0)
    return events


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def checkpoint_path(out_dir: Path, chunk_index: int) -> Path:
    return out_dir / f".chunk_{chunk_index:04d}.json"


def load_checkpoint(out_dir: Path, chunk_index: int) -> list[dict] | None:
    p = checkpoint_path(out_dir, chunk_index)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_checkpoint(out_dir: Path, chunk_index: int, events: list[dict]) -> None:
    checkpoint_path(out_dir, chunk_index).write_text(
        json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clean_checkpoints(out_dir: Path, num_chunks: int) -> None:
    for i in range(num_chunks):
        p = checkpoint_path(out_dir, i)
        if p.exists():
            p.unlink()


# ── Main pipeline ──────────────────────────────────────────────────────────────
def extract(pdf_path: Path, out_dir: Path, max_pages: int = 10, start_page: int = 1) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {pdf_path.name}…")
    pages = extract_pages(pdf_path)
    pages = pages[start_page-1:]
    if max_pages and max_pages > 0:
        pages = pages[:max_pages]
    chunks = make_chunks(pages, first_page=start_page)
    total_pages = len(pages)
    pages_by_number = {start_page + i: text for i, text in enumerate(pages)}
    print(f"  {total_pages} pages → {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={OVERLAP})")

    all_events: list[dict] = []

    for idx, (start_pg, end_pg, text) in enumerate(chunks):
        cached = load_checkpoint(out_dir, idx)
        if cached is not None:
            print(f"  chunk {idx+1}/{len(chunks)} (pages {start_pg}–{end_pg}): loaded from checkpoint")
            all_events.extend(cached)
            continue

        print(f"  chunk {idx+1}/{len(chunks)} (pages {start_pg}–{end_pg}): calling API…", end=" ", flush=True)
        t0 = time.time()
        try:
            events = call_claude(text)
        except Exception as exc:
            print(f"\n  [error] chunk {idx+1} failed after retries: {exc}")
            print("  Partial results saved in checkpoints. Re-run to resume.")
            # Save progress so far and exit cleanly
            _write_partial(pdf_path, out_dir, all_events, total_pages)
            sys.exit(1)

        elapsed = time.time() - t0
        events = attach_provenance(events, pages_by_number)
        print(f"{len(events)} events  ({elapsed:.1f}s)")
        save_checkpoint(out_dir, idx, events)
        all_events.extend(events)

    out_file = out_dir / f"{pdf_path.stem}_extracted.json"
    payload = {
        "source_file": pdf_path.name,
        "total_pages": total_pages,
        "model": MODEL,
        "capital_events": all_events,
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. {len(all_events)} raw events → {out_file}")
    clean_checkpoints(out_dir, len(chunks))
    return out_file


def _write_partial(pdf_path: Path, out_dir: Path, events: list[dict], total_pages: int) -> None:
    out_file = out_dir / f"{pdf_path.stem}_partial.json"
    payload = {"source_file": pdf_path.name, "total_pages": total_pages, "capital_events": events}
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Partial output: {out_file}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract capital events from a DRHP PDF.")
    parser.add_argument("pdf", type=Path, help="Path to the DRHP PDF file")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)",
    )
    parser.add_argument("--start-page", type=int, default=1, help="First page to process (1-indexed)")
    parser.add_argument("--max-pages",
        type=int,
        default=10,
        help="Max pages to process (default: 10). Use 0 for full document.",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: file not found: {args.pdf}")
        sys.exit(1)

    extract(args.pdf, args.out, max_pages=args.max_pages, start_page=args.start_page)












