import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from combine import combine, resolve_callback, parse_call_start

NOW = "2026-11-09T14:00:00+05:30"  # Monday
def cb(day, time=None, **kw):
    d = dict(agreed=True, acknowledged=True, day=day, time=time or {"kind": "none"}); d.update(kw); return d
def nxt(c, now=NOW):
    return resolve_callback(c, parse_call_start(now))[0]
def iso(c, now=NOW):
    v = nxt(c, now); return v.strftime("%Y-%m-%dT%H:%M:%S+05:30") if v else None
FU = dict(meaningful_discussion=True, need_established="yes")

def test_ex1_tomorrow_at_4():
    assert iso(cb({"kind": "tomorrow"}, {"kind": "clock", "hour": 4})) == "2026-11-10T16:00:00+05:30"
def test_ex2_vague_plus3():
    r = combine({**FU, "final_primary_blockers": []}, {"vague_postponement": True}, NOW, "x")
    assert r["next_call_at"] == "2026-11-12T11:00:00+05:30" and r["sub_disposition"] == "Vague Deferral"
def test_ex3_appointment_not_callback():
    a = {**FU, "appointment_confirmed": True, "appointment_time_stated": True, "appointment_modality": "clinic",
         "appointment_day": {"kind": "tomorrow"}}
    r = combine(a, {}, NOW, "x")
    assert (r["disposition"], r["sub_disposition"], r["next_call_at"], r["lead_quality"]) == ("Appointment Scheduled", "Clinic Visit", None, "High")
def test_ex4_unspecified_agent_call_plus1():
    r = combine(FU, {}, NOW, "x"); assert r["next_call_at"] == "2026-11-10T11:00:00+05:30" and r["sub_disposition"] == "Further Discussion"
def test_ex5_do_not_contact():
    r = combine({**FU, "final_refusal": "do_not_contact"}, cb({"kind": "tomorrow"}), NOW, "x")
    assert (r["sub_disposition"], r["lead_quality"], r["next_call_at"]) == ("Do Not Contact", "Low", None)
def test_ex6_day7_is_medium_day6_high():
    mk = lambda n: {**FU, "appointment_confirmed": True, "appointment_time_stated": True, "appointment_day": {"kind": "in_n_days", "n": n}}
    assert combine(mk(6), {}, NOW, "x")["lead_quality"] == "High"
    assert combine(mk(7), {}, NOW, "x")["lead_quality"] == "Medium"
def test_blocker_blocks_high():
    a = {**FU, "appointment_confirmed": True, "appointment_time_stated": True, "appointment_day": {"kind": "tomorrow"}, "final_primary_blockers": ["cost_concern"]}
    assert combine(a, {}, NOW, "x")["lead_quality"] == "Medium"

def test_weekday_rules():  # Monday call
    assert iso(cb({"kind": "weekday", "weekday": "monday"}, {"kind": "clock", "hour": 10, "meridiem": "am"})) == "2026-11-16T10:00:00+05:30"
    assert iso(cb({"kind": "this_weekday", "weekday": "monday"}, {"kind": "clock", "hour": 5, "meridiem": "pm"})) == "2026-11-09T17:00:00+05:30"
    assert iso(cb({"kind": "this_weekday", "weekday": "monday"}, {"kind": "clock", "hour": 1, "meridiem": "pm"})) is None
    assert iso(cb({"kind": "next_weekday", "weekday": "wednesday"})) == "2026-11-18T11:00:00+05:30"
    assert iso(cb({"kind": "next_week"})) == "2026-11-16T11:00:00+05:30"
def test_vague_week_and_event():
    assert nxt(cb({"kind": "this_week"})) is None
    assert nxt(cb({"kind": "next_week"}, conditional_only=True)) is None
def test_time_only_today_or_tomorrow():
    assert iso(cb({"kind": "none"}, {"kind": "clock", "hour": 6})) == "2026-11-09T18:00:00+05:30"
    assert iso(cb({"kind": "none"}, {"kind": "clock", "hour": 11})) == "2026-11-10T11:00:00+05:30"
def test_hour_defaults_and_dayparts():
    assert iso(cb({"kind": "tomorrow"}, {"kind": "clock", "hour": 9})) == "2026-11-10T09:00:00+05:30"
    assert iso(cb({"kind": "tomorrow"}, {"kind": "clock", "hour": 12})) == "2026-11-10T12:00:00+05:30"
    assert iso(cb({"kind": "tomorrow"}, {"kind": "daypart", "daypart": "evening"})) == "2026-11-10T18:00:00+05:30"
    assert iso(cb({"kind": "tomorrow"}, {"kind": "clock", "hour": 7, "daypart": "morning"})) == "2026-11-10T07:00:00+05:30"
def test_absolute_and_months():
    assert iso(cb({"kind": "date", "month": 11, "dom": 8})) == "2027-11-08T11:00:00+05:30"
    assert iso(cb({"kind": "date", "month": 11, "dom": 8, "year": 2026})) is None
    assert iso(cb({"kind": "in_n_months", "n": 1}), "2026-01-31T10:00:00+05:30") == "2026-02-28T11:00:00+05:30"
    assert iso(cb({"kind": "next_month"})) == "2026-12-01T11:00:00+05:30"
def test_minutes_keep_seconds():
    assert iso(cb({"kind": "in_n_minutes", "n": 30}), "2026-10-17T20:42:51+05:30") == "2026-10-17T21:12:51+05:30"
def test_unacknowledged_is_ignored():
    assert nxt(cb({"kind": "tomorrow"}, acknowledged=False)) is None
def test_specific_callback_beats_fallback_and_coexists_with_appt():
    r = combine(FU, cb({"kind": "tomorrow"}, {"kind": "clock", "hour": 4}), NOW, "x")
    assert r["sub_disposition"] == "Specific Callback" and r["next_call_at"] == "2026-11-10T16:00:00+05:30"
    a = {**FU, "appointment_confirmed": True, "appointment_time_stated": True, "appointment_day": {"kind": "in_n_days", "n": 2}}
    assert combine(a, cb({"kind": "tomorrow"}, {"kind": "clock", "hour": 4}), NOW, "x")["next_call_at"] == "2026-11-10T16:00:00+05:30"
def test_precedence_and_tiebreaks():
    r = combine({**FU, "wrong_contact": True, "test_enquiry": True, "final_refusal": "refusal"}, {}, NOW, "x")
    assert r["sub_disposition"] == "Test Enquiry" and r["lead_quality"] == "Not Qualified" and r["next_call_at"] is None
    r = combine({**FU, "final_primary_blockers": ["travel_constraint", "family_discussion"]}, {}, NOW, "x")
    assert r["sub_disposition"] == "Travel Constraint"  # primary (first-listed) reason wins

def test_condition_does_not_void_exact_interval():
    c = {"agreed": True, "acknowledged": True, "conditional_only": True, "day": {"kind": "in_n_minutes", "n": 15}}
    assert iso(c) == "2026-11-09T14:15:00+05:30"
    assert nxt({**c, "day": {"kind": "next_week"}}) is None
def test_rescheduled_precedes_modality():
    a = {**FU, "appointment_confirmed": True, "appointment_time_stated": True, "appointment_modality": "online",
         "appointment_replaces_previous": True, "appointment_day": {"kind": "tomorrow"}}
    assert combine(a, {}, NOW, "x")["sub_disposition"] == "Rescheduled"
def test_unable_to_assess():
    r = combine({"meaningful_discussion": False}, {}, NOW, "x")
    assert (r["disposition"], r["lead_quality"], r["next_call_at"]) == ("Unable to Assess", "Not Assessable", None)
