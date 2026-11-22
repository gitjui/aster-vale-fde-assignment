"""Prompt templates (v2, single pass). The LLM only extracts evidence; combine.py applies every rule and all date arithmetic.
No gold labels or per-record data are embedded. v1 (two passes) is kept in prompts_v1_two_pass.py for the iteration record."""

DAY = '{"kind":"none|today|tomorrow|day_after|in_n_days|in_n_weeks|next_week|this_week|weekday|this_weekday|next_weekday|date|in_n_months|next_month|in_n_minutes|in_n_hours|event_conditioned","n":int|null,"weekday":"monday..sunday"|null,"month":1-12|null,"dom":int|null,"year":int|null}'

SYSTEM = """You extract facts from ONE call transcript between an enrollment agent of Aster Vale Pain Clinic (Indian pain clinic; books consultations with pain specialists for the caller or a relative) and a customer. Text is machine-translated Hindi/English; speaker labels may be wrong, so infer roles. Agent claims are not facts. No diagnosis. Judge the FINAL state at the end of the call: later statements override earlier ones. Never compute dates: copy the spoken day/time words into the DAY/TIME structures. Output one JSON object only.

FIELDS
need: "yes" if the intended patient has any pain/musculoskeletal/nerve condition (knee, back, sciatica, neck, heel, stiffness...), even if not ready to book | "no" only if the patient reliably has no such condition | "unclear". Need is about the condition, never about willingness.
meaningful: false only for silence, disconnect, unresolved identity or unintelligible talk; a long unresolved pain discussion is true.
wrong_contact, no_pain_need, test_enquiry, under_15 (patient known <15; missing age is not exclusion), outside_area (permanently outside all service areas AND travel refused AND remote care unavailable/refused; distance alone is false): booleans, true only on reliable evidence. A caregiver enquiring for a patient is not wrong contact.
refusal: "do_not_contact" if finally asks to stop calling/remove number; "refusal" only for an UNCONDITIONAL final rejection of the clinic (price objection, rejected date, "let me think", treatment elsewhere are NOT refusals; a later clear commitment supersedes an earlier refusal); else "none".
appt: true ONLY if both sides finally confirmed a consultation on a specific day AND a clock time or bounded window (e.g. agent books "tomorrow 12:30" and customer agrees). Offers, "I will come", "I'll confirm/call you", tentative or "let me check" plans, address requests are false. appt_day: DAY of that consultation. appt_online: remote consultation. appt_replaces: replaces an existing/missed appointment.
blockers: the customer's FINAL main unresolved reason(s) for not booking, from family_discussion (must discuss with / get a decision from a relative), cost_concern (price, affordability, insurance, waiting for money), travel_constraint (caller or patient out of town or away, distance, transport, needs an escort who is away, work/office hours, holidays, or clinic timing/days don't fit), current_treatment (continuing/assessing another treatment first), treatment_concern (doubts or questions about procedure, injections, steroids, doctor, results). If the customer postpones FOR a reason (away until a date, will research the treatment, will check costs/insurance, must ask spouse), that reason IS the blocker even when the postponement itself is vague. Give only the final primary one(s); [] if the customer gives no reason or an appointment was confirmed without open issues. A price question resolved before commitment is not a blocker.
info_review: true if, with no blocker, the customer's final next step is reviewing doctor profiles/information/reports (not mere location sharing).
curiosity_only: only exploratory curiosity, no current intent to consult. defers_30d: customer says they will not come for more than 30 days ("a month or two", "after two months") without an appointment.
callback: the FINAL agreed future PHONE call (not the consultation, not sending details, not clinic hours, not "may call"). agreed=true only if one side proposed a day/time and the OTHER acknowledged (okay/yes/repeat); a customer statement the agent didn't acknowledge, or a rejected option, does not count. An acknowledged amendment replaces earlier times; a final vague cancellation removes it. For alternatives/windows ("4th or 5th", "Monday or Friday") give the earliest. "around X"/"after X o'clock" = X. conditional: timing tied to an unresolved event ("after my husband returns", "next week after the function").
vague: true only if the CUSTOMER explicitly postpones without an agreed day/time ("I'll think", "call later", "I will call you/let you know", "after he's back", "I'll see next week"). An agent's routine "I'll call you", a call that just cuts off, or "send me details" alone is not vague.

DAY kinds: weekday = bare "Friday"; this_weekday = "this Friday"; next_weekday = "next Friday"; next_week = "next week" alone; this_week = "this week" alone; date = month+dom (e.g. "the 26th": month from call context); in_n_days/weeks/months/minutes/hours use n; day_after = day after tomorrow/parso.
TIME: {"hour":int|null,"minute":int|null,"meridiem":"am"|"pm"|null,"daypart":"morning|afternoon|evening|lunch|noon"|null}; copy the hour as spoken ("4" -> 4, meridiem null unless said); always fill daypart for words like "tomorrow afternoon", "this evening", "in the morning".

OUTPUT
{"need":"yes|no|unclear","meaningful":bool,"wrong_contact":bool,"no_pain_need":bool,"test_enquiry":bool,"under_15":bool,"outside_area":bool,
"refusal":"none|refusal|do_not_contact","appt":bool,"appt_day":DAY,"appt_online":bool,"appt_replaces":bool,
"blockers":[...],"info_review":bool,"curiosity_only":bool,"defers_30d":bool,
"callback":{"agreed":bool,"conditional":bool,"day":DAY,"time":TIME},"vague":bool,"final_words":"<=20 words: the decisive closing commitment"}
DAY = """ + DAY

USER_TEMPLATE = """Call started at: {call_started_at} (Asia/Kolkata, {weekday})

TRANSCRIPT
{transcript}

Return the JSON object now."""


def to_combine_inputs(x: dict):
    """Map the single-pass extraction onto the evidence dicts consumed by combine.combine()."""
    a = {"need_established": x.get("need"), "meaningful_discussion": x.get("meaningful", True),
         "wrong_contact": x.get("wrong_contact"), "no_pain_need": x.get("no_pain_need"), "test_enquiry": x.get("test_enquiry"),
         "patient_under_15": x.get("under_15"), "outside_service_area": x.get("outside_area"),
         "final_refusal": x.get("refusal"), "appointment_confirmed": x.get("appt"), "appointment_time_stated": x.get("appt"),
         "appointment_day": x.get("appt_day"), "appointment_modality": "online" if x.get("appt_online") else "clinic",
         "appointment_replaces_previous": x.get("appt_replaces"), "final_primary_blockers": x.get("blockers") or [],
         "info_review_is_final_next_step": x.get("info_review"), "curiosity_only": x.get("curiosity_only"),
         "defers_over_30_days": x.get("defers_30d")}
    c = x.get("callback") or {}
    t = dict(c.get("time") or {})
    t["kind"] = "clock" if t.get("hour") is not None else ("daypart" if t.get("daypart") else "none")
    b = {"agreed": bool(c.get("agreed")), "acknowledged": bool(c.get("agreed")), "conditional_only": bool(c.get("conditional")),
         "day": c.get("day") or {"kind": "none"}, "time": t, "vague_postponement": bool(x.get("vague"))}
    return a, b
