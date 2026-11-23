# Aster Vale — Transcript Labeling & Speaker Gender Classification

## Overview

Aster Vale processes customer consultation transcripts and produces structured call labels using an LLM-based extraction pipeline and deterministic Python business logic.

The project has two components:

1. **Transcript labeling:** Extracts customer intent, lead quality, blockers, call outcomes and follow-up dates from conversations.
2. **Speaker gender classification:** Uses a BiLSTM neural network to classify the clinic agent and customer in each call, without transformers or pretrained models.

---

# Part 1: Transcript Labeling

## 1. Running the pipeline

### Requirements
- Python 3.9+
- Groq API key
- `requests==2.32.5`

Set your API key:

```bash
export GROQ_API_KEY=...
```

Alternatively, add `GROQ_API_KEY` to a local `.env` file. Never commit credentials.

Run the pipeline:

```bash
python run.py INPUT.jsonl OUTPUT.jsonl
```

To retain intermediate LLM extractions for debugging:

```bash
python run.py INPUT.jsonl OUTPUT.jsonl --trace trace.jsonl
```

### Input format

Each JSONL row contains:

| Field | Description |
|---|---|
| `transcript_id` | Unique call identifier |
| `call_started_at` | Call start timestamp |
| `timezone` | Local timezone |
| `transcript` | Full conversation transcript |

Only these four input fields are used.

### Output format

Each output row contains exactly five fields:

```text
transcript_id
disposition
sub_disposition
lead_quality
next_call_at
```

- Missing callback timestamps are represented by JSON `null`.
- Timestamps include their UTC offset (e.g. `...+05:30`).

## 2. System architecture

The pipeline uses **one LLM extraction call per transcript**, followed by deterministic Python processing to generate the four final labels.

The design separates language understanding from business-rule execution.

### Processing flow

```text
Customer Call Transcript
          |
          v
    prompts.SYSTEM
          |
          v
   Single LLM Call
          |
          v
 Structured JSON Extraction
          |
          v
prompts.to_combine_inputs
          |
          v
      combine.py
          |
          v
   Business Rule Engine
          |
          v
Schema Validation
          |
          v
 Four Final Labels
```

### A. LLM extraction — `prompts.SYSTEM`

The LLM identifies evidence from the conversation rather than directly deciding the final business outcome.

It extracts:

- Whether the customer expresses a genuine need or engages in meaningful dialogue.
- Invalidity flags and reasons.
- Final refusal or explicit do-not-contact requests.
- Whether a consultation day AND time were mutually confirmed, including online or replacement arrangements.
- Final primary blocker(s).
- Whether the customer needs to review information.
- Curiosity or deferral beyond 30 days.
- The final mutually acknowledged callback request.
- Whether the customer explicitly postponed.

**Important:** Callback dates and times are extracted as spoken. For example, "next Monday at 4 PM" is captured as day and time text. The LLM does not calculate the calendar date.

### B. Input mapping — `prompts.to_combine_inputs`

Maps the LLM's structured JSON extraction into the input format expected by `combine.py`.

This keeps extraction and business-rule processing separate.

### C. Deterministic processing — `combine.py`

Python applies the business rules and produces the final four labels.

Its responsibilities include:

| Rule group | Responsibility |
|---|---|
| D6 | Outcome precedence when multiple conditions apply |
| Parent/child validation | Ensures valid disposition and sub-disposition combinations |
| Blocker tie-breaks | Selects the final primary blocker |
| Invalid-reason tie-breaks | Resolves competing invalidity reasons |
| LQ | Applies the lead-quality hierarchy |
| F1–F12 | Calculates `next_call_at` |
| Schema validation | Validates the final structured output |

Callback calculation rules (F1–F12) cover:

- Relative dates and days.
- Weekday interpretation.
- This week / next week.
- Month and day inference.
- Default callback time of 11:00.
- Interpreting hours 1–7 as PM.
- Daypart interpretation.
- Rejection of past callback times.
- Three-day fallback for vague requests.
- One-day fallback for other unspecified cases.
- Callback suppression under applicable conditions.

### Design decisions

- No gold labels, IDs, hashes or per-record logic are used during inference.
- No few-shot examples are embedded in the prompt, keeping token consumption lower.
- LLM extraction is limited to understanding conversation evidence.
- Business decisions and date arithmetic are handled through deterministic Python rules.

This makes the system easier to debug: extraction errors can be distinguished from business-rule errors.

## 3. Evaluation and reproducibility

The local evaluation uses 50 manually labelled calls.

### Prepare the dataset

```bash
python prepare_data.py "<50 Labeled Calls>.csv" data/
```

Generates:
- `data/inputs.jsonl` — input fields only.
- `data/gold.jsonl` — reference labels for evaluation.

### Run inference

```bash
ASTER_MODEL=qwen/qwen3.8-27b \
python run.py data/inputs.jsonl results/pred.jsonl \
  --trace results/trace.jsonl \
  --cache results/cache
```

### Evaluate predictions

All 50 calls:

```bash
python eval.py results/pred.jsonl data/gold.jsonl
```

Held-out validation set (15 calls):

```bash
python eval.py results/pred.jsonl data/gold.jsonl data/val_ids.txt
```

Tuning set (35 calls):

```bash
python eval.py results/pred.jsonl data/gold.jsonl data/val_ids.txt --exclude
```

### Run offline tests

```bash
python -m pytest -q tests
```

The test suite contains 29 offline tests and does not require an API key.

### Validation split

`data/val_ids.txt` contains 15 calls whose IDs end in 3, 6 or 9.

The split was fixed before tuning:
- 35 calls for tuning.
- 15 calls held out for validation.

## 4. Model configuration

### Provider and models

Provider: Groq (OpenAI-compatible REST API).

Final model:

```text
qwen/qwen3.8-27b
```

Set through `ASTER_MODEL`. The default model is specified in `llm.py`.

Other evaluated models:
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`

### Inference parameters

| Parameter | Setting |
|---|---|
| Temperature | 0 |
| Top P | 1 |
| Seed | 7 |
| Maximum output tokens | 4,000 |
| Response format | JSON object |
| Qwen reasoning format | Hidden |
| GPT-OSS reasoning effort | Low |

GPT-OSS reasoning effort can be configured through `ASTER_REASONING_EFFORT`.

### Reliability and error handling

- HTTP timeout: 90 seconds per call.
- Maximum attempts: 6.
- Respects `retry-after` on HTTP 429/5xx responses.
- Uses exponential backoff otherwise.
- JSON parsing includes code-fence stripping and `{...}` salvage.

If a record still fails after retries, the pipeline writes a conservative valid default:

```text
Disposition: Follow-up Required
Sub-disposition: Further Discussion
Lead Quality: Medium
Next Call: Next day at 11:00
```

A warning is generated. Use `--strict` to raise an error instead of writing a fallback row.

### Groq free-tier constraints

Observed limits per model and organisation:

| Limit | Value |
|---|---:|
| Tokens per minute | 8,000 |
| Qwen input tokens per minute | 7,000 |
| Requests per day | 1,000 |
| Tokens per day | 200,000 |

One complete 50-call run consumes approximately 190,000 tokens.

Expect approximately 25 minutes for a full run when the daily quota is fresh.

## 5. Architecture iteration history

The pipeline evolved through three versions.

| Version | Architecture | Token usage | Change |
|---|---|---:|---|
| v1 | Two sequential LLM passes | ~7,700/call | Replaced due to token consumption and free-tier limitations |
| v2 | Single LLM pass | ~3,800/call | Consolidated extraction into one call |
| v3 | Single LLM pass | ~3,800/call | Refined prompt wording based on error analysis |

**Final implementation: v3.**

The two-pass implementation is retained in `prompts_v1_two_pass.py` for reference.

The final version reduces token consumption by approximately 50% compared with v1 while retaining the single-pass extraction architecture.

See `ERROR_ANALYSIS.md` for the identified errors and prompt changes.

---

# Part 2: Speaker Gender Classification — BiLSTM

## 1. Overview

`gender_bilstm.py` classifies the gender of both speakers in each call:

- Clinic agent
- Customer

It uses a hierarchical Bidirectional LSTM (BiLSTM) with attention, implemented from scratch in PyTorch.

**Model constraints:**
- No transformers.
- No pretrained models.
- No external LLM API calls.

## 2. Setup and execution

Create a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Five-fold cross-validation

```bash
.venv/bin/python gender_bilstm.py cv
```

- Grouped by call to prevent speaker leakage between training and validation.
- Approximately 10 minutes on CPU.

### Train the final model

```bash
.venv/bin/python gender_bilstm.py train
```

Trains on all labelled speakers and saves:

```text
models/gender_bilstm.pt
```

Approximately 3 minutes on CPU.

### Generate predictions

```bash
.venv/bin/python gender_bilstm.py predict \
  data/inputs.jsonl \
  data/gender_bilstm_predictions.jsonl
```

## 3. Data representation

One training example represents a `(call, speaker)` pair.

The model receives every line in the transcript associated with that speaker.

Each line is tagged according to its relationship to the speaker:

- `<by>` — words spoken by that speaker.
- `<to>` — words spoken to that speaker.

Examples:

```text
<by> I am Mehul
<by> my wife
<to> ma'am
<to> Thank you, Mr. Kiran
```

A role token distinguishes the clinic agent from the customer.

## 4. Model architecture

The model uses a hierarchical BiLSTM with attention.

```text
Transcript Lines
       |
       v
   Word Embeddings
       |
       v
   BiLSTM (64 per direction)
       |
       v
    Max Pooling
       |
       v
  Attention Across Lines
       |
       v
   Linear Classifier
       |
       v
   Male / Female
```

### Training configuration

| Parameter | Setting |
|---|---|
| Embedding dimension | 64 |
| BiLSTM hidden size | 64 per direction |
| Optimizer | Adam |
| Learning rate | 2e-3 |
| Epochs | 25 |
| Dropout | 0.3 |
| Augmentation | Random line dropout |
| Ensemble | 3 seeds: 7, 8, 9 |

The three models' predicted probabilities are averaged.

The model returns `unknown` when its highest predicted probability is below 0.75.

The most-attended transcript lines are also returned as evidence.

## 5. Labels and reference data

No ground-truth gender labels were supplied with the original dataset.

`data/gender_reference.jsonl` contains manually assigned labels created by reading the transcripts.

Usable reference dataset:
- 92 speakers.
- 49 customers.
- 43 agents.
- 7 undecidable agents excluded.
- The two-customer call AVT-008 excluded.

## 6. Cross-validation results

Five-fold cross-validation is grouped by call. A call is never evaluated by a model trained on that same call.

| Metric | Agent | Customer |
|---|---:|---:|
| Accuracy when answering | 91.7% | 100% |
| Coverage | 84% | 80% |

Overall:
- Accuracy when answering: 96.0%.
- Accuracy counting abstentions as misses: 78.3%.

All three classification errors are female agents:

- AVT-007
- AVT-022
- AVT-030

In these cases, the only available cue is a name unseen during training. The model falls back to the majority class (male) with high confidence.

## 7. Outputs

The pipeline generates:

| File | Contents |
|---|---|
| `data/gender_bilstm_predictions.jsonl` | Predictions, probabilities and evidence lines |
| `data/gender_labels.csv` | One row per call with speaker labels |

Final predictions across the 50 calls:

| Speaker | Male | Female | Unknown |
|---|---:|---:|---:|
| Clinic agent | 30 | 18 | 2 |
| Customer | 34 | 14 | 2 |

## 8. Limitations

The reference dataset contains only 92 usable training examples, which is very small for a neural network.

Additional limitations:

- The model sometimes predicts a gender even when the transcript does not provide sufficient evidence.
- It guesses on 5 of the 7 undecidable agents, mostly predicting the majority class (male).
- Confidence scores are not calibrated.
- Unseen names can lead to confident but incorrect predictions.

The `unknown` threshold provides an abstention mechanism, but does not eliminate these limitations.

## 9. Rule-based baseline

`gender.py` implements a rule-based alternative using:

- Address terms.
- Self-introductions.
- Kinship words.
- A name lexicon.

It is retained for comparison only. Its results appear in the `rules_*` columns of the CSV.

The rules were developed using the same 50 calls, so there is no out-of-sample performance figure for this baseline.

---

# Project Summary

Aster Vale combines two different approaches to transcript analysis:

| Component | Approach | Purpose |
|---|---|---|
| Transcript labeling | LLM + deterministic Python | Extract call outcomes and follow-up information |
| Speaker gender | Hierarchical BiLSTM with attention | Classify the two speakers without pretrained models |

The transcript-labeling pipeline prioritises consistent business-rule execution and efficient LLM usage.

The gender-classification pipeline explores a lightweight neural sequence model, with explicit evaluation, abstention and documented limitations.
