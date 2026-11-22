"""Deterministic combination logic for the Aster Vale labeling workflow (policy assignment-v2).

The LLM only extracts evidence (see prompts.py). Everything that is a *rule* lives here:
outcome precedence, valid parent/child pairs, lead quality, and next_call_at arithmetic.
"""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone, date, time as dtime

IST = timezone(timedelta(hours=5, minutes=30))
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

VALID_PAIRS = {
    "Appointment Scheduled": {"Clinic Visit", "Online Consultation", "Rescheduled"},
    "Follow-up Required": {"Specific Callback", "Vague Deferral", "Family Discussion", "Cost Concern",
                           "Travel Constraint", "Current Treatment", "Treatment Concern",
                           "Information Requested", "Further Discussion"},
    "Not Interested": {"Explicit Refusal", "Do Not Contact"},
    "Invalid Lead": {"Wrong Contact", "No Pain Need", "Outside Service Area", "Policy Exclusion", "Test Enquiry"},
    "Unable to Assess": {"Incomplete Conversation"},
}
LEAD_QUALITIES = {"Not Qualified", "Not Assessable", "Low", "Medium", "High"}

BLOCKER_ORDER = ["family_discussion", "cost_concern", "travel_constraint", "current_treatment", "treatment_concern"]
BLOCKER_LABEL = {"family_discussion": "Family Discussion", "cost_concern": "Cost Concern",
                 "travel_constraint": "Travel Constraint", "current_treatment": "Current Treatment",
                 "treatment_concern": "Treatment Concern"}
INVALID_ORDER = ["test_enquiry", "wrong_contact", "no_pain_need", "policy_exclusion", "outside_service_area"]
INVALID_LABEL = {"test_enquiry": "Test Enquiry", "wrong_contact": "Wrong Contact", "no_pain_need": "No Pain Need",
                 "policy_exclusion": "Policy Exclusion", "outside_service_area": "Outside Service Area"}


# ----------------------------------------------------------------------------- time helpers
def parse_call_start(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def fmt(dt: datetime) -> str:
    return dt.astimezone(IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")


def _int(v):
    try:
        return int(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def resolve_day(day: dict | None, now: datetime, *, forward_only: bool = True):
    """Resolve an extracted day expression to a date.
    Returns (date|None, status) where status in ok | vague | none | invalid | passthrough(minutes/hours handled by caller)."""
    if not day:
        return None, "none"
    kind = (day.get("kind") or "none").lower()
    today = now.date()
    n = _int(day.get("n"))
    if kind in ("none", ""):
        return None, "none"
    if kind in ("this_week", "event_conditioned", "unbounded_after", "vague"):
        return None, "vague"
    if kind == "today":
        return today, "ok"
    if kind == "tomorrow":
        return today + timedelta(days=1), "ok"
    if kind == "day_after":
        return today + timedelta(days=2), "ok"
    if kind == "in_n_days" and n is not None:
        return today + timedelta(days=n), "ok"
    if kind == "in_n_weeks" and n is not None:
        return today + timedelta(days=7 * n), "ok"
    if kind == "next_week":
        return today + timedelta(days=7 - today.weekday()), "ok"  # next Monday
    wd = (day.get("weekday") or "").lower()
    if kind in ("weekday", "this_weekday", "next_weekday") and wd in WEEKDAYS:
        target = WEEKDAYS.index(wd)
        if kind == "weekday":  # nearest strictly future occurrence
            delta = (target - today.weekday()) % 7 or 7
            return today + timedelta(days=delta), "ok"
        if kind == "this_weekday":  # today or later (time check done by caller)
            return today + timedelta(days=(target - today.weekday()) % 7), "ok"
        monday_next = today + timedelta(days=7 - today.weekday())
        return monday_next + timedelta(days=target), "ok"  # next weekday -> following Mon-Sun week
    if kind == "date":
        mo, dom, yr = _int(day.get("month")), _int(day.get("dom")), _int(day.get("year"))
        if dom and not mo and not yr:  # "call on the 26th": next future occurrence of that day number
            mo = today.month if dom >= today.day else (today.month % 12 + 1)
            yr = today.year if (mo >= today.month) else today.year + 1
            try:
                return date(yr, mo, dom), "ok"
            except ValueError:
                return None, "invalid"
        if not mo or not dom:
            return None, "invalid"
        try:
            if yr:
                d = date(yr, mo, dom)
                return d, ("invalid" if (forward_only and d < today) else "ok")
            d = date(today.year, mo, dom)
            if d < today:
                d = date(today.year + 1, mo, dom)
            return d, "ok"
        except ValueError:
            return None, "invalid"
    if kind == "in_n_months" and n is not None:
        return _add_months(today, n), "ok"
    if kind == "next_month":
        return _add_months(today.replace(day=1), 1), "ok"
    return None, "invalid"


def resolve_clock(t: dict | None):
    """Return (time|None, explicit:bool). Applies F5 defaults."""
    if not t:
        return None, False
    kind = (t.get("kind") or "none").lower()
    part = (t.get("daypart") or "").lower() or None
    hour, minute = _int(t.get("hour")), _int(t.get("minute")) or 0
    mer = (t.get("meridiem") or "").lower() or None
    fixed = {"morning": 10, "afternoon": 15, "evening": 18, "lunch": 13, "noon": 12}
    if hour is None:
        if part in fixed:
            return dtime(fixed[part], 0), True
        return None, False
    if hour > 12:  # 24h clock
        return dtime(hour % 24, minute), True
    if mer is None and part in ("morning",):
        mer = "am"
    elif mer is None and part in ("afternoon", "evening", "lunch"):
        mer = "pm"
    if mer == "am":
        h = 0 if hour == 12 else hour
    elif mer == "pm":
        h = 12 if hour == 12 else hour + 12
    elif hour == 12:
        h = 12
    elif 1 <= hour <= 7:
        h = hour + 12
    else:  # 8-11 AM (hour 0 treated as midnight)
        h = hour
    return dtime(h, minute), True


def resolve_callback(cb: dict | None, now: datetime):
    """Return (datetime|None, status). status: ok | vague | none | invalid."""
    if not cb or not cb.get("agreed") or not cb.get("acknowledged"):
        return None, "none"
    day, tm = cb.get("day") or {}, cb.get("time") or {}
    kind = (day.get("kind") or "none").lower()
    # F12: an unresolved condition only makes a COARSE timing vague ("next week after the function", "after he returns");
    # an exact interval, date or day ("in 15 minutes", "on the 21st", "tomorrow") stays a commitment.
    if cb.get("conditional_only") and kind in ("none", "next_week", "this_week", "event_conditioned", "next_month"):
        return None, "vague"
    n = _int(day.get("n"))
    if kind in ("in_n_minutes", "in_n_hours") and n is not None:
        out = now + (timedelta(minutes=n) if kind == "in_n_minutes" else timedelta(hours=n))
        return (out, "ok") if out > now else (None, "invalid")
    d, status = resolve_day(day, now)
    clock, explicit = resolve_clock(tm)
    if status in ("vague", "invalid"):
        return None, status
    if status == "none":  # time only
        if not explicit:
            return None, "none"
        out = datetime.combine(now.date(), clock, IST)
        if out <= now:
            out += timedelta(days=1)
        return out, "ok"
    out = datetime.combine(d, clock if explicit else dtime(11, 0), IST)
    if kind == "this_weekday" and out <= now:
        return None, "invalid"
    if out <= now:
        return None, "invalid"  # F6: explicit past -> reject, never repaired
    return out, "ok"


def day_offset(day: dict | None, now: datetime):
    d, status = resolve_day(day, now)
    return (d - now.date()).days if status == "ok" and d else None


# ----------------------------------------------------------------------------- main combiner
def _pick(order, flags):
    for k in order:
        if k in flags:
            return k
    return None


def combine(a: dict, cb: dict, call_started_at: str, transcript_id: str) -> dict:
    now = parse_call_start(call_started_at)
    a, cb = a or {}, cb or {}

    invalid = {k for k in INVALID_ORDER if a.get(k) or (k == "policy_exclusion" and a.get("patient_under_15"))}
    appt_off = None
    appt_ok = bool(a.get("appointment_confirmed")) and bool(a.get("appointment_time_stated"))
    if appt_ok:
        appt_off = day_offset(a.get("appointment_day"), now)
        if appt_off is None or appt_off < 0:
            appt_ok = False

    refusal = (a.get("final_refusal") or "none").lower()
    blocker_list = [b for b in (a.get("final_primary_blockers") or []) if b in BLOCKER_ORDER]  # model lists the primary reason first
    blockers = set(blocker_list)
    cb_dt, cb_status = resolve_callback(cb, now)
    vague = bool(cb.get("vague_postponement")) or cb_status == "vague"

    # ---- disposition / sub_disposition (D6 precedence)
    if invalid:
        disp, sub = "Invalid Lead", INVALID_LABEL[_pick(INVALID_ORDER, invalid)]
    elif refusal in ("refusal", "do_not_contact"):
        disp, sub = "Not Interested", ("Do Not Contact" if refusal == "do_not_contact" else "Explicit Refusal")
    elif appt_ok:
        disp = "Appointment Scheduled"
        sub = ("Rescheduled" if a.get("appointment_replaces_previous")
               else "Online Consultation" if (a.get("appointment_modality") or "").lower() == "online"
               else "Clinic Visit")
    elif a.get("meaningful_discussion", True):
        disp = "Follow-up Required"
        if blockers:  # customer's final primary reason = first listed; the fixed tie-break order is used by the
            # extraction prompt's definition of "equally primary" only when the model cannot rank them (see prompts.SYSTEM)
            sub = BLOCKER_LABEL[blocker_list[0]]
        elif cb_status == "ok":
            sub = "Specific Callback"
        elif a.get("info_review_is_final_next_step"):
            sub = "Information Requested"
        elif vague:
            sub = "Vague Deferral"
        else:
            sub = "Further Discussion"
    else:
        disp, sub = "Unable to Assess", "Incomplete Conversation"

    # ---- lead quality (order: NQ, NA, Low, High, Medium)
    need = (a.get("need_established") or "unclear").lower()
    if need == "no" and not a.get("no_pain_need"):  # model said "no need" without the invalidity flag: treat as unclear, not invalid
        need = "unclear"
    commit_off = appt_off if appt_ok else (day_offset(a.get("commitment_day"), now) if a.get("customer_committed_to_consult") else None)
    cb_far = cb_dt is not None and (cb_dt.date() - now.date()).days > 30
    if disp == "Invalid Lead":
        lq = "Not Qualified"
    elif disp == "Unable to Assess" or (disp == "Follow-up Required" and need != "yes" and not a.get("curiosity_only")):
        lq = "Not Assessable"
    elif disp == "Not Interested" or a.get("curiosity_only") or (not appt_ok and (cb_far or a.get("defers_over_30_days"))):
        lq = "Low"
    elif need == "yes" and commit_off is not None and 0 <= commit_off <= 6 and not blockers and (appt_ok or disp != "Follow-up Required" or a.get("customer_committed_to_consult")):
        lq = "High"
    else:
        lq = "Medium"

    # ---- next_call_at (F2, F3, F7, F8)
    if disp in ("Invalid Lead", "Not Interested", "Unable to Assess"):
        nxt = None
    elif cb_status == "ok":
        nxt = cb_dt
    elif disp == "Appointment Scheduled":
        nxt = None
    elif vague:
        nxt = datetime.combine(now.date() + timedelta(days=3), dtime(11, 0), IST)
    else:
        nxt = datetime.combine(now.date() + timedelta(days=1), dtime(11, 0), IST)
    if nxt is not None and nxt <= now:
        nxt = None

    out = {"transcript_id": str(transcript_id), "disposition": disp, "sub_disposition": sub,
           "lead_quality": lq, "next_call_at": fmt(nxt) if nxt else None}
    validate(out)
    return out


def validate(rec: dict):
    assert set(rec) == {"transcript_id", "disposition", "sub_disposition", "lead_quality", "next_call_at"}
    assert rec["sub_disposition"] in VALID_PAIRS[rec["disposition"]], rec
    assert rec["lead_quality"] in LEAD_QUALITIES, rec
    v = rec["next_call_at"]
    assert v is None or (isinstance(v, str) and v.endswith("+05:30") and len(v) == 25), rec
    return rec
