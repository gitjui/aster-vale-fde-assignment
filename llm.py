"""Minimal Groq chat-completions client (OpenAI-compatible REST) with retries, timeouts and JSON repair."""
import json, os, re, sys, time
import requests

_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if "GROQ_API_KEY" not in os.environ and os.path.exists(_ENV):  # optional local .env (never commit it)
    for _line in open(_ENV):
        _k, _, _v = _line.strip().partition("=")
        if _k.strip() == "GROQ_API_KEY" and _v.strip():
            os.environ["GROQ_API_KEY"] = _v.strip().strip('"').strip("'")

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("ASTER_MODEL", "qwen/qwen3.8-27b")  # exact model ID, recorded in README/results
SETTINGS = dict(temperature=0, top_p=1, seed=7, max_tokens=4000, response_format={"type": "json_object"})
if MODEL.startswith("openai/gpt-oss"):
    SETTINGS["reasoning_effort"] = os.environ.get("ASTER_REASONING_EFFORT", "low")  # reasoning tokens count toward max_tokens
elif MODEL.startswith("qwen/"):
    SETTINGS["reasoning_format"] = "hidden"  # keep <think> text out of the JSON content
TIMEOUT_S, MAX_RETRIES = 90, 10
HARD_DEADLINE = None  # epoch seconds; set by run.py --hard-seconds so a rate-limit wait never overruns a time-boxed chunk


class QuotaDeferred(Exception):
    """Raised instead of sleeping when the provider's retry-after would cross HARD_DEADLINE (resume later from cache)."""


USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "retries": 0}


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def chat_json(system: str, user: str, key: str | None = None) -> dict:
    key = key or os.environ["GROQ_API_KEY"]
    body = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **SETTINGS}
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=TIMEOUT_S)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("retry-after", 0) or 0) or min(60, 2 ** attempt * 2)
                last = f"HTTP {r.status_code}"; USAGE["retries"] += 1
                if HARD_DEADLINE and time.time() + wait > HARD_DEADLINE:
                    time.sleep(max(0.0, HARD_DEADLINE - time.time()))  # use the chunk's remaining time to let quota refill
                    raise QuotaDeferred(f"{last}, retry-after {wait:.0f}s exceeds chunk deadline")
                print(f"[retry] {last}, waiting {wait:.0f}s", file=sys.stderr, flush=True); time.sleep(wait); continue
            if r.status_code >= 400:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
            j = r.json()
            u = j.get("usage", {}); USAGE["prompt_tokens"] += u.get("prompt_tokens", 0)
            USAGE["completion_tokens"] += u.get("completion_tokens", 0); USAGE["calls"] += 1
            return parse_json(j["choices"][0]["message"]["content"])
        except (requests.RequestException, ValueError, KeyError) as e:  # network, bad JSON, bad shape
            last = repr(e); USAGE["retries"] += 1; time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Groq call failed after {MAX_RETRIES} attempts: {last}")
