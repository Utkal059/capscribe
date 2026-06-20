FROM python:3.12-slim

WORKDIR /app

# tesseract powers the OCR fallback for scanned filings (pytesseract needs the
# system binary). OCR degrades gracefully if this layer is removed.
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# The API + retrieval stack lives in requirements-api.txt (fastapi, uvicorn,
# chromadb, langgraph, pdfplumber, …). requirements.txt is the legacy
# extractor-only set; pypdf + tenacity are the only bits the optional llm
# ingest pass still needs, so add just those rather than dragging in streamlit.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt "pypdf>=4.0.0" "tenacity>=8.2.0"

COPY . .

# Render / Railway / Fly inject $PORT; fall back to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
