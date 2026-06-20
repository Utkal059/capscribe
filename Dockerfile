FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; chroma's default embedding model is pulled at
# first request and cached in the layer's volume.
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
