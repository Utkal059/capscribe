# Real DRHP evaluation corpus

Drop additional real DRHP/IPO PDFs here to evaluate extraction across many
filings, not just one. `eval_corpus.py` (in the repo root) consumes whatever is
present.

## Add a filing

1. Copy a public DRHP into this folder, e.g. `acme_drhp.pdf`.
   (PDFs are gitignored — they are large/copyright; only the JSON below is committed.)
2. Generate predictions:
   ```bash
   python eval_corpus.py
   ```
   This writes `acme_drhp.predicted.json` beside the PDF and lists it as
   **"no gold set"** (it is not scored yet — no invented numbers).
3. Hand-verify: open the PDF's capital-structure / "history of equity share
   capital" tables and confirm the predicted allotment/bonus/rights/authorised
   events. Save the *correct* set as a gold file:
   ```
   acme_drhp.gold.json
   ```
   in `evaluate.py` format (the identity key is `event_type` + `date`):
   ```json
   { "capital_events": [
       { "event_type": "allotment", "date": "2021-07-29" },
       { "event_type": "bonus_issue", "date": "2021-12-23" }
   ] }
   ```
   Exclude secondary transfers/acquisitions and reclassifications — only
   primary company actions count (see `fixtures/ola_drhp_gold.json` for a worked
   example and its notes).
4. Re-run `python eval_corpus.py` to get per-filing and aggregate
   precision / recall / F1.

## What's scored out of the box

Even with no PDFs here, `eval_corpus.py` always scores the committed Ola
Electric baseline (`fixtures/ola_drhp_extracted.json` vs
`fixtures/ola_drhp_gold.json`).
