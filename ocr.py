"""OCR fallback for scanned / image-heavy PDFs.

Strategy: a page is "image-dominant" when it yields almost no extractable
text but embeds at least one image — the signature of a scan. Such pages
are rendered to a 300-DPI bitmap and run through tesseract.

Graceful degradation: pytesseract (and the tesseract binary) are optional.
When unavailable the pipeline still completes — scanned pages simply come
back empty with ``ocr_used=True, confidence=0.0`` and a logged warning,
and ``GET /health`` reports ``"ocr_available": false``.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber

from schema import PageText

if TYPE_CHECKING:  # pragma: no cover - typing only
    import PIL.Image

logger = logging.getLogger("capscribe.ocr")

# A digital text page comfortably exceeds this; a scan yields ~0 chars.
MIN_TEXT_CHARS = 50
OCR_RENDER_DPI = 300

try:
    import pytesseract

    _PYTESSERACT_INSTALLED = True
except ImportError:  # pragma: no cover - depends on environment
    pytesseract = None  # type: ignore[assignment]
    _PYTESSERACT_INSTALLED = False
    logger.warning("pytesseract not installed — scanned PDFs will not be OCR'd")


@lru_cache(maxsize=1)
def ocr_available() -> bool:
    """True when both the pytesseract package and the tesseract binary work."""
    if not _PYTESSERACT_INSTALLED:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # binary missing or broken install
        logger.warning("tesseract binary not found — OCR disabled")
        return False


def is_page_image_dominant(page: "pdfplumber.page.Page") -> bool:
    """Heuristic scan detector: almost no text but at least one embedded image."""
    text = (page.extract_text() or "").strip()
    return len(text) < MIN_TEXT_CHARS and len(page.images) >= 1


def ocr_page(page_image: "PIL.Image.Image") -> tuple[str, float]:
    """Run tesseract on a rendered page image.

    Returns ``(text, confidence)`` where confidence is the mean word-level
    confidence from tesseract's TSV output, scaled to 0..1. Returns
    ``("", 0.0)`` when OCR is unavailable.
    """
    if not ocr_available():
        return "", 0.0
    data = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT)
    words = [w for w, c in zip(data["text"], data["conf"]) if str(w).strip() and float(c) >= 0]
    confs = [float(c) for w, c in zip(data["text"], data["conf"]) if str(w).strip() and float(c) >= 0]
    text = " ".join(words)
    confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return text, round(confidence, 4)


def extract_text_with_ocr_fallback(pdf_path: str | Path) -> list[PageText]:
    """Per-page text extraction with OCR fallback for scanned pages.

    For each page:
      - try pdfplumber text extraction;
      - if the page is image-dominant, render at 300 DPI and OCR it;
      - return :class:`PageText` with provenance (``ocr_used``, confidence).
    """
    results: list[PageText] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                if is_page_image_dominant(page):
                    if ocr_available():
                        image = page.to_image(resolution=OCR_RENDER_DPI).original
                        text, confidence = ocr_page(image)
                        # A 300-DPI page bitmap is ~25 MB; close it immediately so
                        # memory stays flat across a long scanned filing.
                        try:
                            image.close()
                        except Exception:
                            pass
                        logger.info("page %d: OCR used (confidence %.2f)", i, confidence)
                    else:
                        text, confidence = "", 0.0
                        logger.warning("page %d looks scanned but OCR is unavailable", i)
                    results.append(
                        PageText(page_number=i, text=text, ocr_used=True, confidence=confidence)
                    )
                else:
                    results.append(
                        PageText(
                            page_number=i,
                            text=page.extract_text() or "",
                            ocr_used=False,
                            confidence=1.0,
                        )
                    )
            finally:
                # pdfplumber caches per-page objects; flushing keeps a large
                # filing from accumulating memory page-by-page (OOM guard).
                _flush_page(page)
    return results


def _flush_page(page) -> None:
    """Release a pdfplumber page's cached objects, if the API is available."""
    flush = getattr(page, "flush_cache", None)
    if callable(flush):
        try:
            flush()
        except Exception:
            pass


def process_document(pdf_path: str | Path) -> dict:
    """Extract a whole PDF with OCR fallback and summarise the method mix.

    Returns a dict with ``total_pages``, ``ocr_page_count``,
    ``text_page_count``, a per-page method breakdown, and the per-page text
    keyed by page number (so downstream extractors can reuse it).
    """
    pages = extract_text_with_ocr_fallback(pdf_path)
    ocr_n = sum(1 for p in pages if p.ocr_used)
    return {
        "total_pages": len(pages),
        "ocr_page_count": ocr_n,
        "text_page_count": len(pages) - ocr_n,
        "pages": [
            {
                "page_num": p.page_number,
                "method": "ocr" if p.ocr_used else "native",
                "confidence": p.confidence,
                "chars": len(p.text),
            }
            for p in pages
        ],
        "text_by_page": {p.page_number: p.text for p in pages},
    }
