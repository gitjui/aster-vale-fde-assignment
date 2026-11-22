# Aster Vale call labeling: results and error analysis

Policy: assignment-v2. Submitted model: **`qwen/qwen3.8-27b` on Groq**. Predictions for all 50 supplied calls: `predictions.jsonl`.

## 1. Results

Scored with `eval.py` against the 50 labeled calls. "Pair" gives sub_disposition credit only when the parent disposition is also
correct, as the brief scores it. The /70 column applies the brief's weights (disposition 15, pair 20, lead_quality 15, next_call_at 20).

| Split | Calls | Disposition | Pair | Lead quality | next_call_at | Exact match | Score /70 |
|---|---|---|---|---|---|---|---|
| **Held-out validation** | 15 | 100% | 80.0% | 100% | 93.3% | 80.0% | **64.7** |
| Tuning | 35 | 97.1% | 82.9% | 91.4% | 85.7% | 68.6% | 62.0 |
| All | 50 | 98.0% | 82.0% | 94.0% | 88.0% | 72.0% | 62.8 |

The validation split (`data/val_ids.txt`, the IDs ending in 3, 6 or 9) was fixed before any tuning. Every prompt and rule change
below was motivated by failures on calls I inspected, and the tuning figure is optimistic for that reason. The validation figure is
the better estimate for unseen calls, although 15 calls is a small sample: one call is worth about 1.3 points.

## 2. How the system works

Each call is processed in two steps.

1. **One LLM call** (`prompts.SYSTEM`) answers evidence questions only. Does the patient have a pain need? Is the lead invalid, and
   why? Was there a final refusal or a do-not-contact request? Was a consultation day *and* time mutually confirmed? What is the
   customer's final primary blocker? Was a callback agreed, and what day and time words were used? Did the customer explicitly
   postpone? The model copies spoken day and time words into a structure. It never computes a date.
2. **Deterministic Python** (`combine.py`) turns those answers into the four labels. It applies outcome precedence, the valid
   disposition/sub-disposition pairs, the lead-quality order and all of the next_call_at rules F1–F12 against `call_started_at`.
   It then validates the output schema. Twenty-nine offline unit tests cover the policy's boundary examples EX1–EX6 and the date rules.

No gold label, transcript ID, hash or per-record rule is used at inference time.

## 3. Iterations

| Version | Change | Reason | Effect |
|---|---|---|---|
| v1 | Two LLM passes (outcome facts, then callback facts), followed by the Python rules | Separates the two concerns | Cost about 7.7k tokens per call. One 50-call run needs about 385k tokens, but the Groq free tier allows 200k tokens per model per day, so v1 could never complete a run. Kept in `prompts_v1_two_pass.py`. |
| v2 | A single pass with a compact prompt and no evidence quotes; gpt-oss uses `reasoning_effort=low` | Fit the token budget | About 3.7k tokens per call (−51%). gpt-oss-120b scored 57.3/70 on the first 28 calls. |
| v2 + rules | `combine.py`: (a) "the 26th" with no month resolves to the next future occurrence of that day; (b) "no need" from the model invalidates a lead only when the explicit no-pain-need flag is set, and Unable to Assess requires that no meaningful dialogue took place | Callers who were merely "not ready" were labelled Invalid or Unable to Assess (AVT-007, AVT-022). A "call on the 26th" callback was dropped (AVT-017). | 57.3 → 60.0 on those 28 calls, with no extra tokens |
| v3 prompt | `need` describes the condition, not the customer's willingness. A reason given for postponing is the blocker. An absent escort or being out of town counts as travel. "A month or two" counts as more than 30 days. A vague deferral requires the customer's own postponing words. Dayparts are always filled. | The failure patterns in section 4 | Qwen v3: 61.2/70 on all 50 calls |
| v3 + rules | (c) An unresolved condition voids only a coarse callback timing such as "next week after the function". An exact time such as "in 15 minutes" or "on the 21st" is still a commitment. (d) With several blockers, the customer's primary (first-listed) reason wins. The fixed order Family > Cost > Travel > Current treatment > Treatment concern applies only to ties. | (c) AVT-047's "call in 15 minutes" became +3 days. (d) In 3 of 5 multi-blocker calls the fixed order overrode the reason the customer stressed. | 61.2 → 62.8 on all 50. Validation was unchanged at 64.7, so this gain comes from tuning calls only. |
| Runner | Per-record cache keyed by model, settings, prompt and input; `--max-seconds` and `--hard-seconds` time boxes; retries raised from 6 to 10 | Free-tier quotas forced runs across several sessions. One call fell back to the default row after six short 429 responses. | No call is paid for twice, and the submitted predictions contain no fallback rows |

## 4. Remaining failure patterns (final system, 14 call-level errors)

| Pattern | Calls | Diagnosis |
|---|---|---|
| **Choosing among co-occurring reasons** (largest group) | 013 (gold Information Requested), 014, 015, 026, 030, 032, 047, 049 | Travel is over-predicted: an office-hours remark is often picked over the customer's real final reason (treatment doubts, an ongoing cataract follow-up, reviewing profiles). Telling an open question from a blocker is also a judgement call (AVT-015: the call cut off, so gold is Further Discussion). |
| **Who acknowledged the callback** | 041, 048 | An agent's "I'll call you Wednesday" without a clear customer acknowledgement should fall back to +1 day. The reverse case, "I'll call you back tomorrow" answered with "Alright, alright", was missed. |
| **Visit intention read as postponement** | 050, 013 | "I'll meet you on the 21st" is a visit plan rather than a request to defer the conversation, so gold is +1, not +3. |
| **Event-qualified "next week"** | 040 | "After Diwali", followed by the customer accepting "call in between, next week", gives gold = next Monday. The model kept the Diwali condition, which also triggered the >30-day Low. |
| **Conditional "forget it" read as refusal** | 042 | "Only when you tell me will I want it… forget it" (customer to a bot) is a treatment concern, not an unconditional refusal. This is the only disposition error, and it also costs three other fields. |
| **Low vs Medium intent** | 022 | "I'll call you myself if I want to come" shows no current intent, so gold is Low. |

Likely next improvements, none applied because each needs a paid re-run of all 50 calls:
- State that office or clinic hours count as travel only when the customer gives them as the reason.
- Add one short few-shot pair for callback acknowledgement.
- Add a sentence distinguishing a visit plan from a request to defer.

## 5. Models

Three Groq models were tried, and the comparison was limited by the free-tier daily quota (section 6).

| Model | Why it was tried | Result |
|---|---|---|
| `openai/gpt-oss-120b` | Largest reasoning model available | v2 prompt only, on the 26 calls completed by every run: 59.2/70 |
| `openai/gpt-oss-20b` | Same family about 6× smaller, which isolates the effect of scale | v3, same 26 calls: 58.7/70 |
| **`qwen/qwen3.8-27b`** | A different family at a similar size, which checks whether the prompt depends on one model | v3, same 26 calls: **64.8/70**. Chosen and completed on all 50. |

Qwen led on the two 20-point fields (pair 81% vs 69%; next_call_at 96% vs 77–88%) and produces short JSON without long reasoning
output. The 120b row uses the older v2 prompt, so the 120b-vs-Qwen comparison is not purely a model comparison. Two more models
were unavailable: `llama-3.3-70b-versatile`, the original default, is no longer served to the account, and `allam-2-7b` has only a
4k context.

## 6. Runtime, tokens and cost

* **Tokens per call:** about 3,350 input (system prompt about 1,390; transcript about 1,930 on average, 458–2,875) plus about 350
  output. A 50-call run is about 168k input and 17k output tokens.
* **Cost** at Groq's listed price for `qwen/qwen3.8-27b` ($0.80 per million input tokens, $4.00 per million output): about
  **$0.20 per 50-call run**, or $0.004 per call. All of the runs here were made on the free tier, so the actual spend was $0. For
  comparison, gpt-oss-120b would cost about $0.04 per run.
* **Latency:** about 1–4 s per call when unthrottled. Wall-clock time is set by rate limits: the free tier allows about 7–8k tokens
  per minute (about 2 calls per minute) and 200k tokens per day per model. A full run on a fresh daily quota takes about 25 minutes.
  Once the daily quota is exhausted it refills at about 139 tokens per minute, or about 2 calls per hour. The final 12 calls ran in
  277 s with 2 workers.
* Because of these quotas, the work was spread over several sessions and two free-tier Groq keys. Results are cached by model,
  settings, prompt and input, so the key used has no effect on the predictions.

## 7. Known risks for the private set

* The 50 labeled calls contain no Invalid Lead, Unable to Assess, Do Not Contact, Online Consultation or Rescheduled cases. Those
  branches are covered only by unit tests on `combine.py` and by the prompt definitions.
* Sub-disposition is the noisiest field (82% here). It is worth 20 points, and it depends on picking the customer's primary reason.
* Rule (d) relies on the model listing the primary blocker first, but the prompt does not say so explicitly. Its gain appeared only
  on tuning calls.
* At temperature 0 a re-run on Groq is nearly but not exactly deterministic. Cached runs re-score exactly.
