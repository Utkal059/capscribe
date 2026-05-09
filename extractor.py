import anthropic
import base64
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import pypdf

load_dotenv()

CHUNK_SIZE = 40  # pages per API call (safe buffer under 100)

def pdf_to_chunks(pdf_path: str, chunk_size: int = CHUNK_SIZE):
    reader = pypdf.PdfReader(pdf_path)
    total = len(reader.pages)
    print(f"Total pages: {total}")
    chunks = []
    for start in range(0, total, chunk_size):
        writer = pypdf.PdfWriter()
        for page in reader.pages[start:start + chunk_size]:
            writer.add_page(page)
        import io
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append((start, buf.getvalue()))
    return chunks

def extract_chunk(client, pdf_bytes: bytes, system_prompt: str, chunk_num: int) -> list:
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    print(f"  Calling API for chunk {chunk_num}...")
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {
                    "type": "text",
                    "text": "Extract all capital events from this filing."
                }
            ],
        }],
    )
    text = message.content[0].text.strip()
    # Strip markdown fences if model adds them
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        print(f"  Warning: chunk {chunk_num} returned non-JSON, skipping")
        return []

def extract_from_pdf(pdf_path: str) -> list:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    system_prompt = Path("prompts/system.txt").read_text(encoding="utf-8")
    print(f"Processing: {pdf_path}")

    chunks = pdf_to_chunks(pdf_path)
    all_events = []
    for i, (start_page, pdf_bytes) in enumerate(chunks):
        print(f"Chunk {i+1}/{len(chunks)} (pages {start_page+1}-{start_page+CHUNK_SIZE})")
        events = extract_chunk(client, pdf_bytes, system_prompt, i+1)
        all_events.extend(events)
        print(f"  Found {len(events)} events in this chunk")

    output_path = Path("output") / (Path(pdf_path).stem + "_extracted.json")
    output_path.write_text(json.dumps(all_events, indent=2), encoding="utf-8")
    print(f"\nTotal events: {len(all_events)}")
    print(f"Saved to: {output_path}")
    return all_events

if __name__ == "__main__":
    result = extract_from_pdf("sample.pdf")
    print(json.dumps(result, indent=2))
