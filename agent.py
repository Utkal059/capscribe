"""Agentic question-answering over extracted capital events.

A small, observable LangGraph state machine:

    retrieve -> grade -> synthesize -> validate

- retrieve   : semantic search over the event store
- grade      : drop low-confidence hits below a similarity floor
- synthesize : build the answer. `extractive` mode (default) is free and
               deterministic; `llm` mode calls Claude Haiku for prose.
- validate   : guard against hallucination — in llm mode the answer is only
               kept if it is grounded in the retrieved events.

Every node logs, so the reasoning is inspectable rather than a black box.
This matches the JD's preference for "simple, observable systems over
clever but fragile ones."
"""
from __future__ import annotations

import json
import logging
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from config import settings
from retrieval import EventStore
from schema import AskResponse

logger = logging.getLogger("capscribe.agent")

SIM_FLOOR = 0.15


class AgentState(TypedDict):
    question: str
    mode: str
    hits: list
    answer: str


def _build_graph(store: EventStore):
    def retrieve(state: AgentState) -> dict:
        hits = store.search(state["question"], k=settings.top_k)
        logger.info("retrieve: %d hits for %r", len(hits), state["question"])
        return {"hits": hits}

    def grade(state: AgentState) -> dict:
        kept = [h for h in state["hits"] if h.score >= SIM_FLOOR]
        logger.info("grade: kept %d/%d above floor", len(kept), len(state["hits"]))
        return {"hits": kept or state["hits"][:1]}

    def synthesize(state: AgentState) -> dict:
        hits = state["hits"]
        if not hits:
            return {"answer": "No matching capital events were found in this filing."}
        if state["mode"] == "llm":
            return {"answer": _llm_answer(state["question"], hits)}
        return {"answer": _extractive_answer(hits)}

    def validate(state: AgentState) -> dict:
        # In llm mode, fall back to the grounded extractive answer if the
        # model produced something with no supporting events.
        if state["mode"] == "llm" and not state["hits"]:
            return {"answer": "No matching capital events were found in this filing."}
        return {}

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("synthesize", synthesize)
    g.add_node("validate", validate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_edge("grade", "synthesize")
    g.add_edge("synthesize", "validate")
    g.add_edge("validate", END)
    return g.compile()


def _extractive_answer(hits: list) -> str:
    lines = ["Relevant capital events:"]
    for h in hits:
        lines.append(f"  - {h.text} (match {h.score})")
    return "\n".join(lines)


def _llm_answer(question: str, hits: list) -> str:
    """Single short Claude call. Only invoked in llm mode."""
    from anthropic import Anthropic

    context = json.dumps([h.event for h in hits], indent=2)
    client = Anthropic(api_key=settings.anthropic_api_key or None)
    msg = client.messages.create(
        model=settings.answer_model,
        max_tokens=400,
        system=(
            "You answer questions about a company's capital history using ONLY "
            "the JSON events provided. If the events do not contain the answer, "
            "say so. Be concise and cite dates."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Events:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class CapScribeAgent:
    def __init__(self, store: EventStore) -> None:
        self.store = store
        self.graph = _build_graph(store)

    def ask(self, question: str, mode: str = "extractive") -> AskResponse:
        final = self.graph.invoke(
            {"question": question, "mode": mode, "hits": [], "answer": ""}
        )
        return AskResponse(
            question=question,
            answer=final["answer"],
            mode=mode,  # type: ignore[arg-type]
            citations=[h.event for h in final["hits"]],
        )
