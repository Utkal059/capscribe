from agent import CapScribeAgent
from retrieval import event_to_text


def test_extractive_ask_is_grounded(store):
    agent = CapScribeAgent(store)
    q = event_to_text({"event_type": "rights_issue", "date": "2022-03-15", "ratio": "1:4"})
    resp = agent.ask(q, mode="extractive")
    assert resp.mode == "extractive"
    assert resp.citations  # answer is backed by retrieved events
    assert "capital events" in resp.answer.lower()


def test_agent_runs_full_graph(store):
    agent = CapScribeAgent(store)
    resp = agent.ask("bonus issue ratio", mode="extractive")
    assert isinstance(resp.answer, str) and resp.answer
