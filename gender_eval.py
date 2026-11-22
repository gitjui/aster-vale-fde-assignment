"""python gender_eval.py PRED.jsonl REF.jsonl -> accuracy per speaker (on calls where the reference is decidable) and coverage."""
import json, sys
P = {j["transcript_id"]: j for j in map(json.loads, open(sys.argv[1]))}
R = {j["transcript_id"]: j for j in map(json.loads, open(sys.argv[2]))}
for role in ("agent", "customer"):
    k = f"{role}_gender"
    dec = [i for i in R if R[i][k] in ("male", "female", "mixed")]
    ok = sum((P[i][k] == R[i][k]) or (R[i][k] == "mixed" and P[i][role].get("mixed_speakers")) for i in dec)
    said = [i for i in dec if P[i][k] != "unknown"]
    wrong = [(i, R[i][k], P[i][k]) for i in said if not ((P[i][k] == R[i][k]) or (R[i][k] == "mixed" and P[i][role].get("mixed_speakers")))]
    abst = [i for i in dec if P[i][k] == "unknown"]
    undec_said = [(i, P[i][k]) for i in R if i not in dec and P[i][k] != "unknown"]
    print(f"{role:8s} decidable={len(dec)}/50  correct={ok}/{len(dec)}  wrong={len(wrong)}  said-unknown-but-decidable={len(abst)}  guessed-on-undecidable={len(undec_said)}")
    for w in wrong: print("   WRONG", *w)
    for a in abst: print("   UNKNOWN (reference has an answer)", a, R[a][k])
    for u in undec_said: print("   GUESSED where reference is unknown", *u)
