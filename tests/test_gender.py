import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gender import identify_gender as g

def T(*lines): return "\n".join(lines)

def test_agent_vocatives():
    assert g(T("Agent: Yes ma'am, tell me.", "Customer: Hello.", "Agent: Okay ma'am, noted."))["gender"] == "female"
    assert g(T("Agent: Hello sir.", "Customer: Yes.", "Agent: Thank you, sir."))["gender"] == "male"
def test_customer_saying_sir_does_not_count():
    assert g(T("Customer: Sir, how much is it?", "Customer: Yes sir."))["gender"] == "unknown"
def test_third_person_ma_am_is_ignored():
    r = g(T("Agent: Alright, sir.", "Agent: I already told ma'am about the treatment.", "Agent: If ma'am has pain, the madam will tell you."))
    assert r["gender"] == "male"
def test_caregiver_kin_cues():
    assert g(T("Customer: My wife has knee pain."))["gender"] == "male"
    assert g(T("Customer: My husband has back pain."))["gender"] == "female"
    assert g(T("Customer: I am Renu's husband speaking."))["gender"] == "male"
def test_patient_name_not_used_as_customer_name():
    assert g(T("Customer: Her name is Mira Vardhan."))["gender"] == "unknown"
def test_title_and_self_name():
    assert g(T("Agent: So see here Mr. Pranav, we do procedures."))["gender"] == "male"
    assert g(T("Customer: My name is Farina."))["gender"] == "female"
def test_unknown_when_no_evidence():
    assert g("Agent: Hello.\nCustomer: Hello.")["gender"] == "unknown"

from gender import identify_agent_gender as ag, identify_speakers

def test_agent_self_introduction():
    assert ag(T("Agent: Hi, I am Mehul speaking from Aster Vale Pain Clinic."))["gender"] == "male"
    assert ag(T("Agent: Yes, ma'am. My name is Saira."))["gender"] == "female"
    assert ag(T("Agent: Hi Mahesh sir, good afternoon. Ishan's speaking from Aster Vale Care."))["gender"] == "male"
def test_agent_repeating_customer_name_is_ignored():
    r = ag(T("Agent: You said your name is Leela, right?", "Agent: And your name is Pooja, right?"))
    assert r["names"] == [] and r["gender"] == "unknown"
def test_agent_from_customer_address_terms():
    assert ag(T("Customer: Okay ma'am.", "Customer: Yes ma'am, tell me.", "Customer: Thank you ma'am."))["gender"] == "female"
    assert ag(T("Customer: Sir, how much is it?", "Customer: Okay sir."))["gender"] == "male"
def test_agent_title_only_when_addressing():
    assert ag(T("Customer: Thank you, Mr. Kiran. Bye-bye."))["gender"] == "male"
    assert ag(T("Customer: My father's name is Mr. Hamid Qureshi."))["gender"] == "unknown"
def test_agent_name_vs_many_address_terms_is_conflict():
    r = ag(T("Agent: I am Tara speaking from Aster Vale Care.", *["Customer: Yes sir, I see, sir."] * 6))
    assert r["gender"] == "unknown" and r["conflict"]
def test_both_speakers():
    r = identify_speakers(T("Agent: Hi, I am Pooja from the doctor's team.", "Customer: Yes ma'am.", "Agent: Okay sir, noted.", "Agent: Thank you, sir."))
    assert (r["agent"]["gender"], r["customer"]["gender"]) == ("female", "male")
