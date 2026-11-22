# Aster Vale transcript labeling (policy assignment-v2)

**Submitted:** `predictions.jsonl` (all 50 calls, model `qwen/qwen3.8-27b`). Score on the 15 held-out calls: 64.7/70 (80% exact
match); on all 50: 62.8/70 (72% exact). Details, iterations, failure patterns and cost are in `ERROR_ANALYSIS.md`.
Do not submit `.env` (it holds the API key) or `results/` (local caches and logs).

## Run (one command)
```
export GROQ_API_KEY=...            # or put GROQ_API_KEY=... in a local .env; never commit it
python run.py INPUT.jsonl OUTPUT.jsonl            # add --trace trace.jsonl to keep intermediate extractions
```
Input rows: `transcript_id, call_started_at, timezone, transcript` (only these are used). Output rows: exactly
`transcript_id, disposition, sub_disposition, lead_quality, next_call_at` (`null` is real JSON null; timestamps `...+05:30`).

Resume / free-tier friendly options: `--cache DIR` stores each record's extraction (keyed by model + settings + prompt + input) so an
interrupted run resumes without re-paying tokens; `--max-seconds N` stops starting new records after N s (exit code 3, no output written
until every record is done); `--strict` raises instead of writing a fallback row. `--workers` (default 4) sets parallelism.

Reproduce the local evaluation on the 50 labeled calls:
```
python prepare_data.py "<50 Labeled Calls>.csv" data/     # writes data/inputs.jsonl (input fields only) + data/gold.jsonl
ASTER_MODEL=qwen/qwen3.8-27b python run.py data/inputs.jsonl results/pred.jsonl --trace results/trace.jsonl --cache results/cache
python eval.py results/pred.jsonl data/gold.jsonl                         # all 50
python eval.py results/pred.jsonl data/gold.jsonl data/val_ids.txt        # 15 held-out validation calls only
python eval.py results/pred.jsonl data/gold.jsonl data/val_ids.txt --exclude   # 35 tuning calls only
python -m pytest -q tests                                  # 29 offline tests, no API key needed
```
`data/val_ids.txt` = IDs ending in 3, 6 or 9 (15 calls), fixed before tuning; they are the held-out validation split.

## Design
One LLM extraction call per transcript, then deterministic Python (`combine.py`) produces the four labels.
* `prompts.SYSTEM` asks only evidence questions: is there a pain need / meaningful dialogue, invalidity flags, final refusal or
  do-not-contact, was a consultation day AND time mutually confirmed (+online / replacement), the final primary blocker(s), info review,
  curiosity / >30-day deferral, the final acknowledged callback (day/time words copied into a structure, never computed), and whether the
  customer explicitly postponed. `prompts.to_combine_inputs` maps that JSON to the combiner's inputs.
* `combine.py`: outcome precedence (D6), valid parent/child pairs, blocker and invalid-reason tie-breaks, lead-quality order (LQ), all
  next_call_at arithmetic (F1-F12: relative days, weekdays, this/next week, month/day inference, default 11:00, hour 1-7 = PM, dayparts,
  past-time rejection, +3 vague fallback, +1 other fallback, suppression), and schema validation.
* No gold labels, IDs, hashes or per-record logic are used at inference. No few-shot examples are embedded (token budget).
* Iteration history: v1 = two LLM passes (`prompts_v1_two_pass.py`), replaced because it cost ~7.7k tokens/call and could not complete
  on the Groq free tier (200k tokens/day/model); v2 = single pass (~3.8k tokens/call); v3 = v2 + wording fixes from the error analysis.
  See ERROR_ANALYSIS.md.

## Settings (recorded for the submitted results)
* Provider: Groq (OpenAI-compatible REST). Final model ID: `qwen/qwen3.8-27b` (set via `ASTER_MODEL`; the code default is listed in
  `llm.py`). Also evaluated: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`.
* temperature 0, top_p 1, seed 7, max_tokens 4000, `response_format={"type":"json_object"}`; Qwen: `reasoning_format="hidden"`;
  gpt-oss: `reasoning_effort="low"` (env `ASTER_REASONING_EFFORT`).
* Timeout 90 s per HTTP call; up to 6 attempts; honours `retry-after` on 429/5xx, else exponential backoff; JSON parsed with
  fence-stripping and `{...}` salvage.
* If a record still fails after retries, a conservative valid default row is written (Follow-up Required / Further Discussion / Medium /
  next day 11:00) with a warning; `--strict` raises instead.
* Dependencies: Python 3.9+ (tested 3.13.9), `requests==2.32.5`; `pytest==8.4.2` for tests.
* Free-tier limits observed (per model, per organisation): 8k tokens/min (Qwen: 7k input tokens/min), 1000 requests/day,
  200k tokens/day. One full 50-call run is ~190k tokens, so expect ~25 min when the daily quota is fresh.

## Task 2: speaker gender with a BiLSTM (no transformer)
`gender_bilstm.py` labels the gender of BOTH speakers in each call (the clinic agent and the customer) with a neural sequence
model built from scratch in PyTorch: a hierarchical **BiLSTM with attention**. No transformer, no pretrained model, no API.

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python gender_bilstm.py cv                       # 5-fold cross-validation, grouped by call (~10 min on CPU)
.venv/bin/python gender_bilstm.py train                    # trains on all labeled speakers -> models/gender_bilstm.pt (~3 min)
.venv/bin/python gender_bilstm.py predict data/inputs.jsonl data/gender_bilstm_predictions.jsonl
```
* **Example** = one (call, speaker) pair. Input = every line of the call, each tagged `<by>` (said by that speaker, e.g. "I am
  Mehul", "my wife") or `<to>` (said to them, e.g. "ma'am", "Thank you, Mr. Kiran"), plus a role token.
* **Model**: word embeddings (64) -> BiLSTM (64 per direction) over each line -> max-pool -> additive attention over the lines ->
  linear -> male/female. Trained with Adam (lr 2e-3), 25 epochs, dropout 0.3, random line-dropout augmentation; an ensemble of 3
  seeds (7, 8, 9) whose probabilities are averaged. Output is `unknown` when the top probability is below 0.75. The most-attended
  lines are returned as evidence.
* **Labels**: there is no supplied gender ground truth. `data/gender_reference.jsonl` holds hand labels written by reading each
  transcript (92 usable speakers: 49 customers + 43 agents; 7 undecidable agents and the two-customer call AVT-008 are excluded).
* **Cross-validated result** (5 folds grouped by call, so a call is never scored by a model trained on it):
  agent 91.7% accurate on the 84% it answers; customer 100% on the 80% it answers; overall 96.0% when answering, 78.3% counting
  abstentions as misses. All three errors are female agents (AVT-007, 022, 030) whose only cue is a name unseen in training; the
  model then falls back to the majority class (male) with high confidence.
* **Outputs**: `data/gender_bilstm_predictions.jsonl` (labels, probabilities, evidence lines) and `data/gender_labels.csv` (one row
  per call). Final labels on the 50 calls: agent 30 male / 18 female / 2 unknown; customer 34 male / 14 female / 2 unknown.
* **Limitations**: 92 training examples is very small for a neural model. Unlike the rules, it guesses on 5 of the 7 agents whose
  gender the transcript does not settle (mostly "male", the majority class), and its confidence is not calibrated.
* **Baseline**: `gender.py` is a rule-based version (address terms, self-introductions, kin words, a name lexicon). It is kept for
  comparison only (`rules_*` columns in the CSV). Its rules were developed on these same calls, so it has no out-of-sample figure.
