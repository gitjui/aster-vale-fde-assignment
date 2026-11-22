#!/usr/bin/env bash
# Finish the v3 runs for all three models on the Groq free tier, unattended.
# Each model has its own 200k tokens/day bucket, so the three loops run in parallel.
# Progress is cached per record (results/cache_v3); safe to stop (Ctrl-C) and restart at any time.
# Usage:  bash run_free_tier.sh        progress:  grep -c '^\[done\]' results/log_v3_*.txt
cd "$(dirname "$0")"
run_model () {  # $1 = Groq model id, $2 = short tag used in file names
  until ASTER_MODEL="$1" python3 run.py data/inputs.jsonl "results/pred_v3_$2.jsonl" \
        --trace "results/trace_v3_$2.jsonl" --cache results/cache_v3 --workers 1 --strict \
        2>> "results/log_v3_$2.txt"; do
    sleep 30   # a record failed after its retries (usually a long quota wait); resume from cache
  done
  echo "$(date) DONE $1"
}
run_model qwen/qwen3.8-27b   qwen &
run_model openai/gpt-oss-20b 20b  &
run_model openai/gpt-oss-120b 120b &
wait
echo "$(date) ALL DONE"
