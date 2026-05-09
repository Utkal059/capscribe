# capscribe

> Structured capital event extraction from DRHP/IPO filings using Claude AI.

capscribe parses dense regulatory PDF documents (DRHPs, IPO prospectuses) and extracts structured capital event data — allotments, bonus issues, rights issues, and authorised capital changes — into clean, machine-readable JSON. Built for analysts, quant researchers, and fintech pipelines that need reliable signal from unstructured filings.

---

## What it extracts

| Event Type | Fields Captured |
|---|---|
| Allotments | Date, number of shares, face value, issue price, consideration, allottee category |
| Bonus Issues | Date, ratio, shares issued, pre/post capital |
| Rights Issues | Date, ratio, price, shares offered |
| Authorised Capital Changes | Date, old capital, new capital, resolution type |

---

## How it works

PDFs are split into overlapping page chunks to stay within context limits. Each chunk is passed to Claude with a structured prompt that enforces consistent JSON output. Results are merged, deduplicated, and post-processed into a single clean file per document.

```
DRHP PDF (444 pages)
      │
      ▼
  chunk (1–40 pages)  →  Claude API  →  raw JSON
  chunk (41–80 pages) →  Claude API  →  raw JSON
  ...                                      │
                                           ▼
                                    postprocess.py
                                           │
                                           ▼
                              output/<filename>_extracted.json
```

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/Utkal059/capscribe.git
cd capscribe
pip install -r requirements.txt
```

**2. Add your Anthropic API key**

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

---

## Usage

```bash
python extractor.py <path-to-pdf>
```

**Example:**

```bash
python extractor.py sample.pdf
```

Output is saved to:

```text
output/sample_extracted.json
```

---

## Output format

```json
{
  "source_file": "sample.pdf",
  "total_pages": 444,
  "extraction_date": "2026-05-10",
  "capital_events": [
    {
      "event_type": "allotment",
      "date": "2021-03-15",
      "shares": 500000,
      "face_value": 10,
      "issue_price": 10,
      "consideration": "cash",
      "allottee_category": "promoters"
    },
    {
      "event_type": "bonus_issue",
      "date": "2022-08-01",
      "ratio": "1:1",
      "shares_issued": 5000000,
      "pre_issue_capital": 5000000,
      "post_issue_capital": 10000000
    }
  ]
}
```

---

## Project structure

```
capscribe/
├── extractor.py          # Core extraction pipeline (PDF → chunks → Claude → JSON)
├── postprocess.py        # Deduplication, merging, field normalisation
├── prompts/
│   └── system.text       # Structured prompt engineering for capital event extraction
├── output/               # Extracted JSON files (git-ignored)
├── .github/
│   └── workflows/
│       └── lint.yml      # Ruff linting on push/PR
├── .env                  # API key (git-ignored)
├── requirements.txt
└── README.md
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Your Anthropic API key. |
| `CHUNK_SIZE` | 40 pages | Pages per API call. Reduce if hitting token limits. |

---

## Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for fast Python linting, enforced on every push and pull request via GitHub Actions.

```bash
ruff check .
```

---

## Roadmap

- [ ] Batch mode: process a directory of PDFs
- [ ] CSV export option alongside JSON
- [ ] Confidence scores per extracted event
- [ ] Support for SEBI LODR quarterly disclosures
- [ ] Web UI for non-technical analysts

---

## License

MIT