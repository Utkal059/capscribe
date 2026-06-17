"""OCR-fallback routing tests.

These do not require the tesseract binary: they verify the *decision* to route
a text-less, image-bearing page to OCR, and that OCR degrades to empty output
when the engine is unavailable — so the suite passes on machines without
tesseract installed.
"""
from __future__ import annotations

import ocr


class _StubPage:
    """Minimal stand-in for a pdfplumber page (only what the detector reads)."""

    def __init__(self, text: str, images: list):
        self._text = text
        self.images = images

    def extract_text(self) -> str:
        return self._text


def test_scanned_page_routes_to_ocr():
    scan = _StubPage("", [{"x0": 0, "x1": 100}])          # no text, has an image
    digital = _StubPage("Real extractable text " * 5, [])  # plenty of text
    assert ocr.is_page_image_dominant(scan) is True
    assert ocr.is_page_image_dominant(digital) is False


def test_text_only_page_with_no_image_is_not_scanned():
    # a near-empty page with no image is not a scan (avoids false OCR routing)
    assert ocr.is_page_image_dominant(_StubPage("", [])) is False


def test_ocr_unavailable_degrades_to_empty(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)
    text, confidence = ocr.ocr_page(object())  # image arg is unused when disabled
    assert text == ""
    assert confidence == 0.0
