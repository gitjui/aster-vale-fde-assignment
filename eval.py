"""python eval.py PRED.jsonl GOLD.jsonl  -> per-field accuracy, whole-record match, confusion list."""
import json, sys, collections
load = lambda p: {j["transcript_id"]: j for j in map(json.loads, open(p, encoding="utf-8")) if j}
P, G = load(sys.argv[1]), load(sys.argv[2])
if len(sys.argv) > 3:  # optional: IDS_FILE [--exclude] to score a subset (e.g. the held-out validation split)
    S = set(open(sys.argv[3]).read().split())
    G = {i: g for i, g in G.items() if (i in S) != ("--exclude" in sys.argv)}
F = ["disposition", "sub_disposition", "lead_quality", "next_call_at"]
hit = collections.Counter(); exact = 0; errs = collections.defaultdict(list)
for i, g in G.items():
    p = P.get(i)
    ok = {f: bool(p) and p[f] == g[f] for f in F}
    ok["pair"] = ok["disposition"] and ok["sub_disposition"]  # scoring: sub credit needs correct parent
    for f in F + ["pair"]: hit[f] += ok[f]
    exact += all(ok[f] for f in F)
    for f in F:
        if not ok[f]: errs[f].append((i, g[f], p[f] if p else None))
n = len(G)
print(f"records={n} predicted={len(P)}")
for f in ["disposition", "pair", "lead_quality", "next_call_at"]: print(f"{f:14s} {hit[f]/n:6.1%} ({hit[f]}/{n})")
print(f"{'sub (raw)':14s} {hit['sub_disposition']/n:6.1%}\nexact match    {exact/n:6.1%} ({exact}/{n})")
w = {"disposition": 15, "pair": 20, "lead_quality": 15, "next_call_at": 20}
print("weighted label score /70:", round(sum(w[f]*hit[f]/n for f in w), 1))
for f in F:
    print(f"\n-- {f} errors ({len(errs[f])})")
    for e in errs[f]: print("  ", *e)
