#!/usr/bin/env python3
"""Speaker-gender classifier built on a BiLSTM (no transformer), for BOTH speakers of a call: the clinic agent and the customer.

Model (hierarchical BiLSTM with attention, trained from scratch on this data):
  * one example = (call, role) with role in {agent, customer}; its input is every line of the call
  * each line -> tokens [<ROLE>, <BY>|<TO>, words...]: <BY> = said by this speaker (self-introductions, "my wife"),
    <TO> = said by the other speaker, i.e. addressed to this one ("ma'am", "Mr. Kiran")
  * word embedding -> BiLSTM over the line -> max-pool = line vector
  * additive attention over the lines -> call vector -> linear -> P(male), P(female)
  * "unknown" when max probability < --threshold (default 0.75); attention weights give the evidence lines
Labels: there is no supplied gender ground truth. Training labels come from data/gender_reference.jsonl (hand labels written by
reading each transcript); "unknown" and "mixed" speakers are excluded from training.

Usage (run with the project venv: .venv/bin/python):
  python gender_bilstm.py cv       [--ref data/gender_reference.jsonl] [--folds 5]   # grouped-by-call cross-validation
  python gender_bilstm.py train    [--ref ...] [--model models/gender_bilstm.pt]     # train on all labeled speakers
  python gender_bilstm.py predict  INPUT.jsonl OUTPUT.jsonl [--model ...]            # label both speakers of each call
"""
import argparse, json, os, random, re, sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

ROLES = ("agent", "customer")
LABELS = ("male", "female")
MAX_TOK, MAX_LINES = 40, 200
SPECIAL = ["<pad>", "<unk>", "<agent>", "<customer>", "<by>", "<to>"]


# ----------------------------------------------------------------------------- data
def norm_word(w: str) -> str:
    w = w.lower()
    if w in ("mam", "maam", "ma'am", "madam", "madame"): return "ma'am"
    return w


def tokenize(text: str):
    return [norm_word(w) for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[?!]", text)][:MAX_TOK - 2]


def parse_lines(transcript: str):
    out = []
    for ln in transcript.split("\n"):
        m = re.match(r"\s*(Agent|Customer)\s*:\s*(.*)", ln)
        if m and m.group(2).strip():
            out.append(("agent" if m.group(1) == "Agent" else "customer", m.group(2)))
    return out


def make_bag(transcript: str, role: str):
    """Token lists for every line, seen from `role`'s side."""
    bag = []
    for spk, text in parse_lines(transcript)[:MAX_LINES]:
        bag.append([f"<{role}>", "<by>" if spk == role else "<to>"] + tokenize(text))
    return bag or [[f"<{role}>", "<to>"]]


class Vocab:
    def __init__(self, bags=None, min_freq=2):
        self.itos = list(SPECIAL)
        if bags:
            c = Counter(t for bag in bags for line in bag for t in line[2:])
            self.itos += sorted(t for t, n in c.items() if n >= min_freq)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, bag):
        return [[self.stoi.get(t, 1) for t in line] for line in bag]


def to_tensors(enc_bag):
    lens = torch.tensor([len(l) for l in enc_bag])
    x = torch.zeros(len(enc_bag), int(lens.max()), dtype=torch.long)
    for i, l in enumerate(enc_bag):
        x[i, :len(l)] = torch.tensor(l)
    return x, lens


# ----------------------------------------------------------------------------- model
class HierBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=64, att=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.att_w = nn.Linear(2 * hid, att)
        self.att_v = nn.Linear(att, 1, bias=False)
        self.out = nn.Linear(2 * hid, len(LABELS))

    def forward(self, x, lens):
        e = self.drop(self.emb(x))                                      # lines x tokens x emb
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.clamp(min=1), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(-1)
        line_vec = h.masked_fill(~mask, -1e4).max(dim=1).values          # max-pool over tokens: lines x 2hid
        a = torch.softmax(self.att_v(torch.tanh(self.att_w(line_vec))).squeeze(-1), dim=0)   # attention over lines
        call_vec = self.drop((a.unsqueeze(-1) * line_vec).sum(0))
        return self.out(call_vec), a


# ----------------------------------------------------------------------------- training / inference
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train_model(examples, seed=7, epochs=25, lr=2e-3, line_keep=(0.6, 1.0)):
    """examples: list of (bag_tokens, label_idx). Returns (model, vocab)."""
    seed_all(seed)
    vocab = Vocab([b for b, _ in examples])
    model = HierBiLSTM(len(vocab.itos))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    enc = [(vocab.encode(b), y) for b, y in examples]
    for _ in range(epochs):
        model.train()
        random.shuffle(enc)
        for bag, y in enc:
            keep = random.uniform(*line_keep)                            # augmentation: random subset of lines
            sub = [l for l in bag if random.random() < keep] or bag
            x, lens = to_tensors(sub)
            logits, _ = model(x, lens)
            loss = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([y]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    return model, vocab


@torch.no_grad()
def predict_bag(models, bag, threshold=0.75, top_k=3):
    """Average the ensemble's probabilities; return label, probs and the most-attended lines."""
    probs, att = [], None
    for model, vocab in models:
        x, lens = to_tensors(vocab.encode(bag))
        logits, a = model(x, lens)
        probs.append(torch.softmax(logits, -1).numpy())
        att = a.numpy() if att is None else att + a.numpy()
    p = np.mean(probs, 0)
    k = int(p.argmax())
    label = LABELS[k] if p[k] >= threshold else "unknown"
    top = [" ".join(bag[i][1:])[:120] for i in np.argsort(-att)[:top_k]]
    return {"gender": label, "p_male": round(float(p[0]), 3), "p_female": round(float(p[1]), 3), "evidence_lines": top}


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def labeled_examples(records, ref):
    ex = []
    for r in records:
        g = ref.get(r["transcript_id"])
        if not g: continue
        for role in ROLES:
            lab = g[f"{role}_gender"]
            if lab in LABELS:
                ex.append((r["transcript_id"], role, make_bag(r["transcript"], role), LABELS.index(lab)))
    return ex


# ----------------------------------------------------------------------------- commands
def cmd_cv(args):
    recs = load_jsonl(args.inputs); ref = {j["transcript_id"]: j for j in load_jsonl(args.ref)}
    ex = labeled_examples(recs, ref)
    ids = sorted({e[0] for e in ex}); random.Random(args.seed).shuffle(ids)
    fold_of = {i: k % args.folds for k, i in enumerate(ids)}
    rows = []
    for f in range(args.folds):
        tr = [(b, y) for i, _, b, y in ex if fold_of[i] != f]
        models = [train_model(tr, seed=args.seed + s, epochs=args.epochs) for s in range(args.ensemble)]
        for i, role, b, y in ex:
            if fold_of[i] == f:
                p = predict_bag(models, b, args.threshold)
                rows.append((i, role, LABELS[y], p["gender"], max(p["p_male"], p["p_female"])))
        print(f"fold {f + 1}/{args.folds} done", file=sys.stderr, flush=True)
    for role in ROLES + ("all",):
        rr = [r for r in rows if role in ("all", r[1])]
        decided = [r for r in rr if r[3] != "unknown"]
        correct = sum(r[2] == r[3] for r in decided)
        print(f"{role:9s} n={len(rr):3d}  answered={len(decided):3d} ({len(decided)/len(rr):.0%})  "
              f"accuracy when answering={correct}/{len(decided)} ({correct/max(1,len(decided)):.1%})  "
              f"accuracy over all={correct}/{len(rr)} ({correct/len(rr):.1%})")
    wrong = [r for r in rows if r[3] not in ("unknown", r[2])]
    for r in wrong: print("   WRONG", r[0], r[1], "gold", r[2], "pred", r[3], f"p={r[4]:.2f}")
    if args.out:
        with open(args.out, "w") as f:
            for r in rows: f.write(json.dumps(dict(zip(["transcript_id", "role", "reference", "predicted", "confidence"], r))) + "\n")


def cmd_train(args):
    recs = load_jsonl(args.inputs); ref = {j["transcript_id"]: j for j in load_jsonl(args.ref)}
    ex = [(b, y) for _, _, b, y in labeled_examples(recs, ref)]
    models = [train_model(ex, seed=args.seed + s, epochs=args.epochs) for s in range(args.ensemble)]
    os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
    torch.save({"threshold": args.threshold, "members": [{"state": m.state_dict(), "itos": v.itos} for m, v in models]}, args.model)
    print(f"trained {args.ensemble} BiLSTM(s) on {len(ex)} labeled speakers -> {args.model}", file=sys.stderr)


def load_models(path):
    ck = torch.load(path, weights_only=False)
    models = []
    for mem in ck["members"]:
        v = Vocab(); v.itos = mem["itos"]; v.stoi = {t: i for i, t in enumerate(v.itos)}
        m = HierBiLSTM(len(v.itos)); m.load_state_dict(mem["state"]); m.eval()
        models.append((m, v))
    return models, ck["threshold"]


def cmd_predict(args):
    models, thr = load_models(args.model)
    with open(args.output, "w", encoding="utf-8") as o:
        for r in load_jsonl(args.input):
            res = {role: predict_bag(models, make_bag(r["transcript"], role), thr) for role in ROLES}
            o.write(json.dumps({"transcript_id": r["transcript_id"], "agent_gender": res["agent"]["gender"],
                                "customer_gender": res["customer"]["gender"], **res}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cv", "train"):
        p = sub.add_parser(name)
        p.add_argument("--inputs", default="data/inputs.jsonl"); p.add_argument("--ref", default="data/gender_reference.jsonl")
        p.add_argument("--epochs", type=int, default=25); p.add_argument("--ensemble", type=int, default=3)
        p.add_argument("--seed", type=int, default=7); p.add_argument("--threshold", type=float, default=0.75)
    sub.choices["cv"].add_argument("--folds", type=int, default=5); sub.choices["cv"].add_argument("--out")
    sub.choices["train"].add_argument("--model", default="models/gender_bilstm.pt")
    p = sub.add_parser("predict"); p.add_argument("input"); p.add_argument("output"); p.add_argument("--model", default="models/gender_bilstm.pt")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    {"cv": cmd_cv, "train": cmd_train, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
