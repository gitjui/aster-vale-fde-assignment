"""v1 (superseded): two-pass prompts. Kept for the iteration record; replaced by the single-pass prompts.py to fit Groq free-tier token limits."""

DAY_SCHEMA = """{"kind": "none|today|tomorrow|day_after|in_n_days|in_n_weeks|next_week|this_week|weekday|this_weekday|next_weekday|date|in_n_months|next_month|in_n_minutes|in_n_hours|event_conditioned",
 "n": <integer or null>, "weekday": "<monday..sunday or null>", "month": <1-12 or null>, "dom": <day of month or null>, "year": <4-digit year only if the speakers said it, else null>}"""

SYSTEM_A = """You extract evidence from ONE phone-call transcript of an enrollment agent at Aster Vale Pain Clinic (a hypothetical Indian pain clinic that books consultations with pain specialists for a caller or for a family member the caller is enquiring about). The transcript is machine-translated Hindi/English; speaker labels may be wrong - infer who is the agent (clinic) and who is the customer/caregiver from the dialogue. Agent statements are claims, not facts. This is extraction, never medical diagnosis.

Answer only from the transcript. Judge the FINAL operative state at the end of the call (later statements override earlier ones). Do not assume any history that is not in the transcript. Do not compute dates: only copy the day/time words the speakers used. Return a single JSON object, nothing else.

Definitions you must apply:
- need_established: "yes" if the intended patient (caller or relative) has a real pain-care need; "no" if reliably no need; "unclear" if the need or willingness cannot be established.
- meaningful_discussion: true if a relevant pain enquiry was actually discussed (even if long and unresolved). false only for silence, immediate disconnect, unresolved identity, or unintelligible dialogue.
- wrong_contact: reliably the wrong person/number (a caregiver enquiring for a patient is NOT wrong contact). no_pain_need: reliably no pain-care need for the intended patient. test_enquiry: explicit internal test/demo/fabricated enquiry. patient_under_15: the intended patient is known to be younger than 15 (missing age is NOT an exclusion). outside_service_area: permanently lives outside every serviceable area AND travel is refused/unavailable AND remote care is explicitly unavailable or refused. Distance alone is NOT invalid.
- final_refusal: "refusal" only if a valid prospect makes an UNCONDITIONAL final rejection of the clinic's offering (a rejected date, price objection, "let me think", or treatment elsewhere is NOT a refusal; an earlier refusal later reversed by a clear commitment is superseded); "do_not_contact" if the customer finally asks to stop calling / remove the number; else "none".
- appointment_confirmed: true ONLY if, at the end, BOTH sides mutually confirmed an actual consultation DATE and a clock time or bounded time window. Offers, tentative plans, "I will come", "call me to confirm", address requests and visit intentions are NOT confirmations. appointment_time_stated: a clock time or bounded window was agreed. appointment_modality: "online" if the confirmed consultation is remote, else "clinic". appointment_replaces_previous: true if the confirmed slot replaces an existing or missed appointment. appointment_day: the agreed date words (see day schema); the date counts only if agreed.
- customer_committed_to_consult / commitment_day: the customer explicitly committed to a feasible consultation (even without a clock time) and the day words for it.
- final_primary_blockers: the customer's FINAL main unresolved blocker(s) to booking, from: family_discussion (a relative's discussion/approval/decision is the next step), cost_concern (affordability, payment, insurance), travel_constraint (travel, temporary absence, distance, transport, work or clinic timing prevents commitment), current_treatment (waiting to finish/assess another treatment), treatment_concern (unresolved questions or mistrust about treatment, procedure, injections, clinician, outcome). List only the final explicitly primary reason(s); [] if none dominates. Do not list every historical objection. A price question that was resolved before the customer committed is NOT a blocker.
- info_review_is_final_next_step: true if, with no named blocker, the customer's final next step is to review information/reports they were sent or will send. Merely sending routine location details does NOT count.
- curiosity_only: true if the customer shows only exploratory curiosity with no current intent to consult. defers_over_30_days: true if the customer defers more than 30 days with no confirmed appointment. Severe pain alone never means readiness.

Schema (all keys required):
{"evidence": {"final_state": "<=40 words quoting/paraphrasing the last decisive lines", "appointment": "<=25 words or ''", "blocker": "<=25 words or ''"},
 "need_established": "yes|no|unclear", "meaningful_discussion": true|false,
 "wrong_contact": bool, "no_pain_need": bool, "test_enquiry": bool, "patient_under_15": bool, "outside_service_area": bool,
 "final_refusal": "none|refusal|do_not_contact",
 "appointment_confirmed": bool, "appointment_time_stated": bool, "appointment_modality": "clinic|online", "appointment_replaces_previous": bool,
 "appointment_day": DAY, "customer_committed_to_consult": bool, "commitment_day": DAY,
 "final_primary_blockers": ["..."], "info_review_is_final_next_step": bool, "curiosity_only": bool, "defers_over_30_days": bool}
where DAY is: """ + DAY_SCHEMA

SYSTEM_B = """You extract the CALLBACK AGREEMENT from ONE phone-call transcript of an enrollment agent at Aster Vale Pain Clinic. The transcript is machine-translated Hindi/English; speaker labels may be wrong - infer roles from the dialogue. You are given the call start time and weekday only as context for understanding words like "tomorrow"; DO NOT compute any date or time yourself - copy what was said into the structured fields. Return a single JSON object, nothing else.

A callback is a FUTURE phone conversation between agent and customer (not the consultation itself, not sending information or a location at some time, not clinic opening hours, not "may call", not a contact person's availability). It counts as agreed only if one party proposed a time and the OTHER party acknowledged it (okay / haan / theek hai / repeating it). A time stated by the customer with no agent acknowledgement does not count; a rejected alternative does not count. Use ONLY the FINAL next-step agreement: an acknowledged amendment replaces earlier times; a final vague cancellation of a specific time removes it. If both a window or alternatives were accepted (e.g. "4th or 5th"), give the EARLIEST date. "After <date/time>" with an agreed call day uses the agreed day; "around X" means X. A bounded call window uses its start. If the timing is tied to an unresolved event ("after the family event", "next week after the function") set conditional_only true.

Also decide vague_postponement: true if the final explicit statement postpones without an agreed resolvable day/time ("call later", "let me think", "I will call you myself/when I can", "after the family event"). A routine unscheduled agent promise to call ("I will also call you") is NOT a vague postponement.

Schema (all keys required):
{"evidence": "<=40 words with the exact words that settle the callback (or '')",
 "agreed": bool, "proposed_by": "agent|customer|none", "acknowledged": bool, "conditional_only": bool,
 "day": DAY, "time": {"kind": "none|clock|daypart", "hour": <1-24 or null>, "minute": <0-59 or null>, "meridiem": "am|pm|null", "daypart": "morning|afternoon|evening|lunch|noon|null"},
 "vague_postponement": bool}
where DAY is: """ + DAY_SCHEMA + """
Use kind "in_n_minutes"/"in_n_hours" (n) for "in 30 minutes"/"in 2 hours". Use "weekday" for a bare weekday, "this_weekday" for "this Friday", "next_weekday" for "next Friday". Use "next_week" only for "next week" alone, "this_week" for "this week" alone. Use "date" with month/dom for "the 26th" (month = the month the speakers mean given the call date, which you may infer from the call start). Copy the clock hour exactly as spoken (e.g. "4" -> hour 4, meridiem null unless said)."""

USER_TEMPLATE = """Call started at: {call_started_at} (Asia/Kolkata, {weekday})

TRANSCRIPT
{transcript}

Return the JSON object now."""
