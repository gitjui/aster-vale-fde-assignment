"""Convert the sheet export (CSV) into input-only JSONL + a separate gold JSONL. Usage: python prepare_data.py LABELED.csv data/"""
import csv, json, sys, os
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
with open(src, newline="", encoding="utf-8") as f, open(f"{out}/inputs.jsonl", "w", encoding="utf-8") as fi, open(f"{out}/gold.jsonl", "w", encoding="utf-8") as fg:
    for r in csv.DictReader(f):
        fi.write(json.dumps({k: r[k] for k in ("transcript_id", "call_started_at", "timezone", "transcript")}, ensure_ascii=False) + "\n")
        nc = None if r["next_call_at"].strip() == "null" else r["next_call_at"].strip()
        fg.write(json.dumps({"transcript_id": r["transcript_id"], "disposition": r["disposition"], "sub_disposition": r["sub_disposition"],
                             "lead_quality": r["lead_quality"], "next_call_at": nc}) + "\n")
