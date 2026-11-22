#!/usr/bin/env python3
"""python run.py INPUT.jsonl OUTPUT.jsonl  -- reads GROQ_API_KEY from the environment."""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, time as dtime

import combine, llm, prompts


CACHE_DIR = None  # --cache: per-record extraction cache, keyed by model + prompts + input (lets interrupted runs resume)
DEADLINE = None   # --max-seconds: stop starting new records after this time (cached ones still count)


class Skipped(Exception):
    pass


def _cache_path(rec, user):
    h = hashlib.sha256("\x00".join([llm.MODEL, json.dumps(llm.SETTINGS, sort_keys=True), prompts.SYSTEM, user]).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{rec['transcript_id']}.{h}.json")


def predict(rec: dict) -> dict:
    """Uses ONLY transcript_id, call_started_at, timezone, transcript."""
    now = combine.parse_call_start(rec["call_started_at"])
    user = prompts.USER_TEMPLATE.format(call_started_at=combine.fmt(now), weekday=now.strftime("%A"), transcript=rec["transcript"])
    cp = _cache_path(rec, user) if CACHE_DIR else None
    if cp and os.path.exists(cp):
        tr = json.load(open(cp))
    else:
        if DEADLINE and time.time() > DEADLINE:
            raise Skipped()
        t0 = time.time()
        tr = {"x": llm.chat_json(prompts.SYSTEM, user), "seconds": round(time.time() - t0, 1)}
        if cp:
            json.dump(tr, open(cp, "w"))
        print(f"[done] {rec['transcript_id']} {tr['seconds']}s", file=sys.stderr, flush=True)
    a, b = prompts.to_combine_inputs(tr["x"])
    return combine.combine(a, b, rec["call_started_at"], rec["transcript_id"]), tr


def safe_predict(rec, strict=False):
    try:
        return predict(rec)
    except (Skipped, llm.QuotaDeferred):
        raise Skipped()
    except Exception as e:  # keep one row per input id so a single failure cannot zero the whole file
        if strict: raise
        print(f"[warn] {rec.get('transcript_id')}: {e!r} -> conservative default", file=sys.stderr)
        now = combine.parse_call_start(rec["call_started_at"])
        d = combine.fmt(datetime.combine(now.date() + timedelta(days=1), dtime(11, 0), combine.IST))
        return {"transcript_id": str(rec["transcript_id"]), "disposition": "Follow-up Required", "sub_disposition": "Further Discussion",
                "lead_quality": "Medium", "next_call_at": d}, {"error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--strict", action="store_true")
    ap.add_argument("--trace", help="optional JSONL of intermediate LLM extractions for error analysis")
    ap.add_argument("--cache", help="directory for per-record extraction cache (resume interrupted runs)")
    ap.add_argument("--hard-seconds", type=float, help="defer (not fail) any record whose rate-limit wait would end after N seconds")
    ap.add_argument("--max-seconds", type=float, help="stop starting new uncached records after N seconds; output is written only when complete")
    args = ap.parse_args()
    global CACHE_DIR, DEADLINE
    if args.cache:
        CACHE_DIR = args.cache; os.makedirs(CACHE_DIR, exist_ok=True)
    if args.hard_seconds:
        llm.HARD_DEADLINE = time.time() + args.hard_seconds
    if args.max_seconds:
        DEADLINE = time.time() + args.max_seconds
    recs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(safe_predict, r, args.strict) for r in recs]
        results, skipped = [], 0
        for fu in futs:
            try:
                results.append(fu.result())
            except Skipped:
                skipped += 1
    if skipped:
        print(f"INCOMPLETE: {skipped} records not yet processed (cached progress kept); re-run the same command to resume", file=sys.stderr)
        sys.exit(3)
    with open(args.output, "w", encoding="utf-8") as f:
        for out, _ in results:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    if args.trace:
        with open(args.trace, "w", encoding="utf-8") as f:
            for (out, tr) in results:
                f.write(json.dumps({"transcript_id": out["transcript_id"], **tr}, ensure_ascii=False) + "\n")
    print(f"{len(recs)} records in {time.time()-t0:.0f}s | model={llm.MODEL} | tokens in={llm.USAGE['prompt_tokens']} out={llm.USAGE['completion_tokens']} calls={llm.USAGE['calls']} retries={llm.USAGE['retries']}", file=sys.stderr)


if __name__ == "__main__":
    main()
