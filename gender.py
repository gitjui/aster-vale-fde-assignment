#!/usr/bin/env python3
"""Rule-based (no transformer, no learned model) identification of the gender of BOTH speakers in a call transcript:
the CUSTOMER (identify_gender) and the clinic AGENT (identify_agent_gender); identify_speakers() returns both.

"Customer" = the person on the line (the caller), not necessarily the patient: a son calling for his mother is male.
Evidence families (log-odds style weights; positive = male, negative = female):
  1. Vocative address terms the AGENT uses towards the customer (sir / ma'am / madam / bhai / didi ...).
     Third-person uses ("told ma'am about...", "the madam will tell you") are excluded.
  2. Titles + name in agent lines ("Mr. Pranav"), and self-introductions in customer lines ("my name is X").
  3. Kin self-reference in customer lines ("I am Renu's husband", "my wife", "my husband").
  4. Transliterated Hindi gendered verb forms in customer lines (raha hoon / rahi hoon, gaya / gayi ...).
  5. First-name lexicon + conservative suffix heuristic (weak).
Agent evidence: self-introductions ("I am Mehul", "Pooja speaking from", "my name is Saira"), the customer's address terms
towards the agent (sir / ma'am / madam / brother / bhai), and titles the customer uses ("Mr. Kiran").
Names are de-identified substitutes in this dataset and may not preserve gender, so a name that conflicts with strong
address-term evidence yields "unknown" with conflict=true rather than a guess.
Usage:  python gender.py INPUT.jsonl OUTPUT.jsonl      or      from gender import identify_speakers
"""
import json, math, re, sys

MALE_VOC = r"sir|saab|sahab|sahib|bhaiya|bhai|brother|uncle|mr"
FEMALE_VOC = r"ma'?am|mam|madam|madame|miss|didi|sister|aunty|auntie|ms|mrs"
NON_VOC_PREV = {"if", "the", "told", "tell", "to", "with", "for", "that", "about", "of", "call", "by", "say", "ask", "asked", "and", "or", "from"}
NON_VOC_NEXT = {"has", "is", "was", "will", "can", "should", "did", "does", "may", "had", "have", "who", "also", "said", "says", "wants", "needs", "ji"}  # "ma'am has..." = third person

FEMALE_NAMES = set("""aarti aditi aishwarya akanksha alka amrita ananya anita anjali anju ankita anna anuradha anushka archana asha ashwini bhavna
chitra deepa deepika devi diya divya farina fatima geeta gita gauri hema indu isha jaya jyoti kajal kalpana kamala kavita kavya komal kriti lakshmi lata
leela lalita madhura mala malini manisha mansi maya meena meera megha mira mona mrinal nalini namrata nandini nandita neelam neha nidhi nisha nita nupur
pallavi pooja poonam prabha pratibha preeti priya priyanka purnima radha rachna rani rashmi rekha renu reshma rita riya roshni ruchi rupa sakshi sangeeta
sapna sarita savita seema shalini shanti sharmila shweta simran sita smita sneha sonali sonia sudha sujata sunita supriya surbhi sushma swati tara tanvi
uma usha vandana vanita varsha vidya vimala yamini zoya saira shabana ayesha sana
reva alina farzana pilasha kusuma parveen ila mala""".split())
MALE_NAMES = set("""aakash abhay abhishek aditya ajay ajit akash alok amit amar anand anil ankit anurag arjun arun ashok ashwin atul ayaan
bharat chetan deepak dev dinesh farhan ganesh gaurav girish gopal govind hari harish hemant imran jai jayesh karan kishore krishna kunal lalit mahesh
manoj mohan mukesh murali nalin naren naresh nikhil nitin om pankaj prakash pranav prasad pratik rahul raj rajesh rakesh ramesh ranjit ravi rohan rohit
sachin sagar sameer sandeep sanjay santosh satish shyam siddharth subhash sudhir sunil suresh sushil tarun umesh vijay vikas vikram vinay vinod vipin vishal
vivek yash yogesh zayan zubair rizwan salman aman arif
mehul ishan adil aamir neel aadi riaan rian nilesh veer vedant tushar ritesh hamid sajid rayan devendra""".split())
FEMALE_SUFFIX = ("a", "i", "ee", "ni", "ita", "ika", "aya", "li")   # weak; many male names end in -a/-i too (Krishna, Rajeshwari...)
MALE_EXCEPT_A = {"krishna", "harsha", "ravindra", "surya", "vishva", "rama", "indra", "shiva", "siva", "kiran", "sameer"}

HI_MALE = re.compile(r"\b(?:main|mai|mein)\b[^.?!]{0,30}\b(?:raha|gaya|aaya|karta|chahta|sakta|bola|kehta|soch raha)\b|\b(?:raha hoon|raha hun|gaya hoon|aaya hoon|chahta hoon|karta hoon|sakta hoon)\b", re.I)
HI_FEMALE = re.compile(r"\b(?:main|mai|mein)\b[^.?!]{0,30}\b(?:rahi|gayi|gai|aayi|karti|chahti|sakti|boli|kehti|soch rahi)\b|\b(?:rahi hoon|rahi hun|gayi hoon|aayi hoon|chahti hoon|karti hoon|sakti hoon)\b", re.I)

def _lines(t):
    ag, cu = [], []
    for ln in t.split("\n"):
        if ln.startswith("Agent:"): ag.append(ln[6:].strip())
        elif ln.startswith("Customer:"): cu.append(ln[9:].strip())
    return ag, cu

def _vocatives(line: str):
    """Yield 'M'/'F' for address terms used as vocatives in one agent line."""
    for m in re.finditer(rf"\b({MALE_VOC}|{FEMALE_VOC})\b\.?", line, re.I):
        term = m.group(1).lower().rstrip(".")
        if term in ("mr", "ms", "mrs", "miss") and line[m.end() - 1:m.end()] == ".":
            continue  # titles handled by the title rule
        before = re.findall(r"[A-Za-z']+", line[:m.start()])[-1:] or [""]
        after = re.findall(r"[A-Za-z']+", line[m.end():])[:1] or [""]
        tail = line[m.end():m.end() + 2].strip()
        punct_next = tail == "" or tail[0] in ",.?!;:-"
        if before[0].lower() in NON_VOC_PREV and not punct_next: continue
        if before[0].lower() in {"the", "told", "tell", "if", "with", "to", "for", "about"}: continue
        if after[0].lower() in NON_VOC_NEXT and not punct_next: continue
        if not (punct_next or m.start() == 0 or before[0].lower() in {"yes", "okay", "ok", "no", "see", "so", "hello", "hi", "look", "right", "sure", "alright", "correct", "thank", "you"}): continue
        yield "M" if re.fullmatch(MALE_VOC, term) else "F"

AMBIGUOUS_NAMES = {"kiran", "jyoti", "sonu", "guddu", "pinky", "lucky", "sunny", "shashi"}  # used for both genders


def _name_gender(name: str):
    n = name.lower()
    if n in AMBIGUOUS_NAMES: return 0.0
    if n in FEMALE_NAMES: return -1.6
    if n in MALE_NAMES: return 1.6
    if len(n) > 3 and n not in MALE_EXCEPT_A and n.endswith(FEMALE_SUFFIX): return -0.5
    if len(n) > 3: return 0.3
    return 0.0

def identify_gender(transcript: str, threshold: float = 1.0) -> dict:
    ag, cu = _lines(transcript)
    sig, score = [], 0.0

    # 1. agent vocatives (strongest, diminishing returns)
    m_n = f_n = 0
    for ln in ag:
        for g in _vocatives(ln):
            m_n += g == "M"; f_n += g == "F"
    def vw(n): return 0.0 if n == 0 else min(4.5, 1.2 + 0.6 * (n - 1))
    if m_n or f_n:
        score += vw(m_n) - vw(f_n); sig.append(f"agent_vocative M={m_n} F={f_n}")

    # 2a. title + name in agent lines ("Mr. Pranav"); counted once per call
    for ln in ag:
        m = re.search(r"\b(Mr|Mrs|Ms|Miss|Smt|Shri|Shrimati)\.?\s+([A-Z][a-z]+)", ln)
        if m:
            t, name = m.groups(); score += 2.5 if t in ("Mr", "Shri") else -2.5; sig.append(f"title {t} {name}")
            break
    # 2b. the customer's OWN name only ("my name is X", "I am X"); never "her name is" / "patient's name is" (that is the patient)
    STOP = {"calling", "sorry", "fine", "okay", "not", "going", "looking", "just", "from", "actually", "also", "really", "trying", "ok", "aster", "east", "west", "north", "south"}
    for ln in cu:
        m = re.search(r"\b[Mm]y name is ([A-Z][a-z]+)", ln) or re.search(r"\bI(?: am|'m) ([A-Z][a-z]+)\b(?!'s)(?! from| calling| speaking to)", ln) or re.search(r"^([A-Z][a-z]+) (?:here|speaking)\b", ln)
        if m and m.group(1).lower() not in STOP:
            w = _name_gender(m.group(1))
            if w: score += w; sig.append(f"name {m.group(1)} ({w:+.1f})")
            break

    # 3. kin self-reference
    txt = " ".join(cu)
    if re.search(r"\bI(?: am|'m) [A-Z][a-z]+'s (?:husband|son|father|brother|nephew)\b", txt): score += 3.0; sig.append("self=X's husband/son")
    if re.search(r"\bI(?: am|'m) [A-Z][a-z]+'s (?:wife|daughter|mother|sister|niece)\b", txt): score -= 3.0; sig.append("self=X's wife/daughter")
    hus, wif = len(re.findall(r"\bmy husband\b", txt, re.I)), len(re.findall(r"\bmy wife\b", txt, re.I))
    if hus: score -= min(2.5, 1.5 + 0.5 * (hus - 1)); sig.append(f"my husband x{hus}")
    if wif: score += min(2.5, 1.5 + 0.5 * (wif - 1)); sig.append(f"my wife x{wif}")

    # 4. Hindi gendered verbs
    hm, hf = len(HI_MALE.findall(txt)), len(HI_FEMALE.findall(txt))
    if hm or hf: score += min(2.5, hm) - min(2.5, hf); sig.append(f"hindi_verbs M={hm} F={hf}")

    mixed = (min(m_n, f_n) >= 2) or (any(x.startswith("self=X") for x in sig) and bool(hus or wif))
    label = "male" if score >= threshold else "female" if score <= -threshold else "unknown"
    return {"gender": label, "score": round(score, 2), "confidence": round(1 / (1 + math.exp(-abs(score))), 3) if not mixed else 0.5, "mixed_speakers": bool(mixed), "signals": sig}

AGENT_INTRO = [
    r"\b(?:I am|I'm|this is|This is|my name is|My name is)\s+([A-Z][a-z]+)",   # "I am Mehul", "my name is Saira"
    r"\b([A-Z][a-z]+)(?:'s)? speaking\b",  # "Aamir speaking from", "...good afternoon. Ishan's speaking from"
]
NOT_A_NAME = {"aster", "vale", "dr", "doctor", "calling", "speaking", "from", "sorry", "the", "your", "this", "here", "aruna", "varuna",
              "pineford", "cedar", "okay", "yes", "hello", "hi", "sir", "ma'am", "madam", "good", "very", "just", "not"}


def identify_agent_gender(transcript: str, threshold: float = 1.0) -> dict:
    """Gender of the clinic agent. Evidence: agent self-introduction (name lexicon), customer address terms toward the agent,
    and a title + name the customer uses. Names are weighted below many consistent address terms (names are substituted)."""
    ag, cu = _lines(transcript)
    sig, name_score, voc_score = [], 0.0, 0.0
    names = []
    for ln in ag:
        if re.search(r"\byour (?:good )?name\b|\bname is .* right\b|\b(?:patient|mother|father|her|his)'?s? name\b", ln, re.I):
            continue  # the agent repeating the customer's or patient's name
        for pat in AGENT_INTRO:
            m = re.search(pat, ln)
            if m and m.group(1).lower() not in NOT_A_NAME:
                names.append(m.group(1))
                break
    for n in dict.fromkeys(names):  # distinct names, in order (one call can contain two agents)
        w = _name_gender(n)
        if abs(w) >= 1.0:  # lexicon names only; the suffix guess is too weak for agents
            name_score += 2.5 if w > 0 else -2.5
            sig.append(f"agent_intro {n} ({'M' if w > 0 else 'F'})")
    name_score = max(-2.5, min(2.5, name_score))
    m_n = f_n = 0
    for ln in cu:
        for g in _vocatives(ln):
            m_n += g == "M"; f_n += g == "F"
        m_n += len(re.findall(r"(?:^|[,.!?]\s*)(?:brother|bhai|boss)\b", ln, re.I))
    def vw(n): return 0.0 if n == 0 else min(3.0, 1.0 + 0.4 * (n - 1))
    if m_n or f_n:
        voc_score = vw(m_n) - vw(f_n); sig.append(f"customer_vocative M={m_n} F={f_n}")
    for ln in cu:
        # only a title used to ADDRESS the agent ("Thank you, Mr. Kiran"), not "my father's name is Mr. Hamid"
        m = re.search(r"(?:^|\b(?:thank you|thanks|hello|hi|okay|ok|yes|bye|sorry)\b,?\s*)(Mr|Mrs|Ms|Miss|Madam)\.?\s+([A-Z][a-z]+)", ln, re.I)
        if m:
            name_score += 2.0 if m.group(1).lower() == "mr" else -2.0; sig.append(f"customer_title {m.group(1)} {m.group(2)}")
            break
    score = name_score + voc_score
    conflict = name_score * voc_score < 0 and abs(name_score) >= 2 and abs(voc_score) >= 1.5
    label = "unknown" if conflict else "male" if score >= threshold else "female" if score <= -threshold else "unknown"
    return {"gender": label, "score": round(score, 2), "confidence": 0.5 if label == "unknown" else round(1 / (1 + math.exp(-abs(score))), 3),
            "conflict": bool(conflict), "names": list(dict.fromkeys(names)), "signals": sig}


def identify_speakers(transcript: str) -> dict:
    """Both speakers: {"agent": {...}, "customer": {...}}."""
    return {"agent": identify_agent_gender(transcript), "customer": identify_gender(transcript)}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
        for line in f:
            if line.strip():
                r = json.loads(line)
                sp = identify_speakers(r["transcript"])
                o.write(json.dumps({"transcript_id": r["transcript_id"], "agent_gender": sp["agent"]["gender"],
                                    "customer_gender": sp["customer"]["gender"], **sp}) + "\n")

if __name__ == "__main__":
    main()
