import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
torch = pytest.importorskip("torch")  # run with .venv/bin/python -m pytest
import gender_bilstm as g

T = "Agent: Hi, I am Pooja from the doctor's team.\nCustomer: Yes ma'am.\nAgent: Okay sir, noted."

def test_bag_marks_speaker_side():
    bag = g.make_bag(T, "customer")
    assert bag[0][:2] == ["<customer>", "<to>"] and bag[1][:2] == ["<customer>", "<by>"]
    assert "ma'am" in bag[1]

def test_model_forward_and_attention_shape():
    v = g.Vocab([g.make_bag(T, "agent")], min_freq=1)
    m = g.HierBiLSTM(len(v.itos)); m.eval()
    x, lens = g.to_tensors(v.encode(g.make_bag(T, "agent")))
    logits, att = m(x, lens)
    assert logits.shape == (2,) and att.shape == (3,) and abs(float(att.sum()) - 1) < 1e-5

def test_tiny_training_learns_address_terms():
    ex = [(g.make_bag("Agent: Yes ma'am.\nCustomer: Hello.", "customer"), 1),
          (g.make_bag("Agent: Yes sir.\nCustomer: Hello.", "customer"), 0)] * 6
    m = g.train_model(ex, epochs=15, line_keep=(1.0, 1.0))
    assert g.predict_bag([m], g.make_bag("Agent: Okay ma'am.\nCustomer: Hi.", "customer"), 0.5)["gender"] == "female"
    assert g.predict_bag([m], g.make_bag("Agent: Okay sir.\nCustomer: Hi.", "customer"), 0.5)["gender"] == "male"
