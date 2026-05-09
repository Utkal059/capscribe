# capscribe

Extracts capital events (allotments, bonus issues, authorised capital changes) from DRHP/IPO filings using Claude AI.

## Setup
`
pip install -r requirements.txt
`

Add your API key to .env:
`
ANTHROPIC_API_KEY=your_key_here
`

## Usage
`
python extractor.py
`
Output saved to output/<filename>_extracted.json
