import json, os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import llm, run, combine

def fake_chat(system, user, key=None):
    return {"need": "yes", "meaningful": True, "blockers": ["travel_constraint"],
            "callback": {"agreed": True, "conditional": False, "day": {"kind": "tomorrow"}, "time": {"hour": 4}}, "vague": False}

def test_runner_plumbing(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "chat_json", fake_chat)
    inp = os.path.join(os.path.dirname(__file__), "..", "data", "inputs.jsonl")
    out = tmp_path / "pred.jsonl"
    monkeypatch.setattr(sys, "argv", ["run.py", inp, str(out), "--strict"])
    run.main()
    rows = [json.loads(l) for l in open(out)]
    ids = [json.loads(l)["transcript_id"] for l in open(inp)]
    assert [r["transcript_id"] for r in rows] == ids and len(set(ids)) == 50
    for r in rows:
        combine.validate(r)

def test_single_pass_mapping():
    a, b = __import__("prompts").to_combine_inputs(fake_chat("", ""))
    r = combine.combine(a, b, "2026-11-09T14:00:00+05:30", "x")
    assert (r["sub_disposition"], r["next_call_at"]) == ("Travel Constraint", "2026-11-10T16:00:00+05:30")

def test_failure_falls_back_to_valid_row(monkeypatch):
    def boom(*a, **k): raise RuntimeError("down")
    monkeypatch.setattr(llm, "chat_json", boom)
    rec = {"transcript_id": "Z1", "call_started_at": "2026-10-17T20:42:51+05:30", "timezone": "Asia/Kolkata", "transcript": "x"}
    out, _ = run.safe_predict(rec)
    combine.validate(out); assert out["next_call_at"] == "2026-10-18T11:00:00+05:30"
