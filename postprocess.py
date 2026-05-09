import json
import csv
import sys
from pathlib import Path

def deduplicate(events):
    seen = set()
    out = []
    for e in events:
        key = (e.get("event_type"), e.get("date"), str(e.get("amount")), str(e.get("securities_count")))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out

def to_csv(events, out_path):
    fields = ["event_id","event_type","date","date_confidence","amount",
              "normalized_amount_inr","securities_count","face_value_per_share","allottees"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for e in events:
            e["allottees"] = "; ".join(e.get("allottees") or [])
            w.writerow(e)
    print(f"CSV saved to {out_path}")

if __name__ == "__main__":
    json_path = sys.argv[1]
    events = json.load(open(json_path, encoding="utf-8"))
    events = deduplicate(events)
    csv_path = Path(json_path).with_suffix(".csv")
    to_csv(events, csv_path)
    print(f"{len(events)} unique events written.")