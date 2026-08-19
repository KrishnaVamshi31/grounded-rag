"""
Step 3 — The loop.

    question
       |
    rewrite ------> retrieve ------> grade
       ^                               |
       |                     any relevant chunks?
       |                        /            \
       +--- no, retry once ----+              +---> answer (with citations)
                                |
                                +---> refuse ("not in these documents")

The point of every part of this: the system should be able to come back
empty-handed and say so, instead of writing a confident paragraph from four
irrelevant chunks.

    python graph.py "why are attention scores scaled"
"""

import json
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

import llm
from index import Index

MAX_ATTEMPTS = 2      # one initial pass plus one retry
RETRIEVE_K = 6        # chunks pulled per search query


class State(TypedDict):
    question: str
    queries: list
    retrieved: list
    relevant: list
    attempts: int
    status: Optional[str]
    answer: Optional[dict]
    rejected_citations: list


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

REWRITE_SYSTEM = """You turn a user's question into search queries for a \
keyword+embedding search over a technical document.

Rules:
- Output 1-3 queries as a JSON list of strings.
- Use terminology the document would use, not the user's phrasing.
- If the question has multiple parts, give one query per part.

Format: ["query one", "query two"]"""


def rewrite(state: State) -> dict:
    """Turn the question into search queries.

    On a retry we tell the model the first attempt found nothing useful and
    show it what it already tried, so it varies the wording instead of
    producing the same queries again.
    """
    if state["attempts"] == 0:
        user = state["question"]
    else:
        user = (
            f"Question: {state['question']}\n"
            f"These queries returned nothing relevant: {state['queries']}\n"
            "Write different queries. Try synonyms, more general wording, "
            "or the underlying concept rather than the surface phrasing."
        )

    queries = llm.complete_json(REWRITE_SYSTEM, user)
    if not isinstance(queries, list):
        queries = [state["question"]]
    return {"queries": [str(q) for q in queries][:3],
            "attempts": state["attempts"] + 1}


def retrieve(state: State, index: Index) -> dict:
    """Hybrid search for every query, deduped by chunk id."""
    seen, chunks = set(), []
    for query in state["queries"]:
        for chunk in index.search(query, method="hybrid", top_k=RETRIEVE_K):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                chunks.append(chunk)
    return {"retrieved": chunks}


GRADE_SYSTEM = """You judge whether retrieved passages can help answer a \
question.

For each passage, decide if it contains information that contributes to an \
answer. Be strict: same topic is NOT the same as answers the question.

Return JSON: {"relevant": ["c00012", "c00045"]}
Return an empty list if none qualify."""


def grade(state: State) -> dict:
    """Filter retrieved chunks down to the ones that actually help.

    One call for all chunks rather than one per chunk: cheaper, and the model
    grades better when it can compare passages against each other.
    """
    if not state["retrieved"]:
        return {"relevant": []}

    passages = "\n\n".join(
        f"[{c['chunk_id']}] ({c['heading']}) {c['text']}"
        for c in state["retrieved"]
    )
    verdict = llm.complete_json(
        GRADE_SYSTEM,
        f"Question: {state['question']}\n\nPassages:\n{passages}",
    )
    keep = set(verdict.get("relevant", []) if isinstance(verdict, dict) else [])
    return {"relevant": [c for c in state["retrieved"]
                         if c["chunk_id"] in keep]}


ANSWER_SYSTEM = """Answer the question using ONLY the passages provided.

Rules:
- Every claim must come from a passage. Never add outside knowledge.
- Cite the chunk id for each claim.
- If the passages don't fully answer the question, say what is missing.

Return JSON:
{"answer": "...", "citations": ["c00012"], "complete": true}"""


def answer(state: State) -> dict:
    """Generate the answer, then verify its citations in code.

    The prompt asks the model to cite only what it was given. Prompts are not
    guarantees, so we check: any citation naming a chunk that wasn't in the
    graded set is stripped, and if nothing survives, the answer is discarded.
    """
    passages = "\n\n".join(
        f"[{c['chunk_id']}] (p{c['page']}, {c['heading']}) {c['text']}"
        for c in state["relevant"]
    )
    result = llm.complete_json(
        ANSWER_SYSTEM,
        f"Question: {state['question']}\n\nPassages:\n{passages}",
    )

    allowed = {c["chunk_id"] for c in state["relevant"]}
    cited = [c for c in result.get("citations", []) if c in allowed]
    rejected = [c for c in result.get("citations", []) if c not in allowed]

    if not cited:
        # The model answered but grounded it in chunk ids it was never given.
        # Discard the answer entirely rather than showing an ungrounded one.
        return {"status": "insufficient",
                "answer": {
                    "text": "These documents don't contain enough information "
                            "to answer that.",
                    "citations": [], "pages": [], "complete": False,
                },
                "rejected_citations": rejected}

    pages = sorted({c["page"] for c in state["relevant"]
                    if c["chunk_id"] in cited})
    return {
        "status": "answered",
        "answer": {
            "text": result.get("answer", ""),
            "citations": cited,
            "pages": pages,
            "complete": bool(result.get("complete", True)),
        },
        "rejected_citations": rejected,
    }


def refuse(state: State) -> dict:
    return {
        "status": "insufficient",
        "answer": {
            "text": "These documents don't contain enough information to "
                    "answer that.",
            "citations": [],
            "pages": [],
            "complete": False,
        },
    }


def route(state: State) -> str:
    """After grading: answer, try again, or give up."""
    if state["relevant"]:
        return "answer"
    if state["attempts"] < MAX_ATTEMPTS:
        return "rewrite"
    return "refuse"


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

def build_graph(index: Index):
    graph = StateGraph(State)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", lambda s: retrieve(s, index))
    graph.add_node("grade", grade)
    graph.add_node("answer", answer)
    graph.add_node("refuse", refuse)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route,
                                {"answer": "answer",
                                 "rewrite": "rewrite",
                                 "refuse": "refuse"})
    graph.add_edge("answer", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def ask(app, question: str) -> dict:
    return app.invoke({
        "question": question,
        "queries": [],
        "retrieved": [],
        "relevant": [],
        "attempts": 0,
        "status": None,
        "answer": None,
        "rejected_citations": [],
    })


def main():
    import sys

    if len(sys.argv) < 2:
        print('usage: python graph.py "your question"')
        return

    index = Index.load("index")
    app = build_graph(index)
    result = ask(app, sys.argv[1])

    print(f"\nstatus:   {result['status']}")
    print(f"attempts: {result['attempts']}")
    print(f"queries:  {result['queries']}")
    print(f"\n{result['answer']['text']}\n")
    if result["answer"]["citations"]:
        print(f"cited:    {result['answer']['citations']}")
        print(f"pages:    {result['answer']['pages']}")
    if result["rejected_citations"]:
        print(f"REJECTED (hallucinated ids): {result['rejected_citations']}")


if __name__ == "__main__":
    main()
