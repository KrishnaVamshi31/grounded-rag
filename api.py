"""
FastAPI service.

    uvicorn api:app --reload

Endpoints
    GET  /              the UI
    GET  /health        index status, model names
    POST /ingest        upload PDFs, rebuild the index
    POST /ask           Server-Sent Events: one event per pipeline stage

The stream sends stage events rather than tokens. That is deliberate: the
interesting thing about this system is *how* it decides, and a user watching
"rewriting -> searching -> checking evidence -> answering" learns more than
they would from a paragraph appearing one word at a time.
"""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import llm
from graph import GRADE_SYSTEM, MAX_ATTEMPTS, RETRIEVE_K, answer, grade, rewrite
from index import Index

load_dotenv(override=True)

INDEX_DIR = os.environ.get("INDEX_DIR", "index")
CHUNKS_PATH = os.environ.get("CHUNKS_PATH", "chunks.jsonl")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Grounded PDF Q&A")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_index = None


def get_index():
    global _index
    if _index is None:
        if not Path(INDEX_DIR).exists():
            raise HTTPException(503, "No index yet — upload PDFs first.")
        _index = Index.load(INDEX_DIR)
    return _index


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    indexed = Path(INDEX_DIR).exists()
    chunks = len(Index.load(INDEX_DIR).chunks) if indexed else 0
    return {
        "indexed": indexed,
        "chunks": chunks,
        "model": llm.model_name(),
        "dense": indexed and Path(INDEX_DIR, "vectors.npy").exists(),
    }


@app.post("/ingest")
async def ingest(files: list[UploadFile]):
    """Accept PDFs, chunk them, rebuild the index."""
    from ingest import ingest_pdf

    global _index
    records, count = [], 0

    with tempfile.TemporaryDirectory() as tmp:
        for upload in files:
            if not upload.filename.lower().endswith(".pdf"):
                raise HTTPException(400, f"{upload.filename} is not a PDF.")
            path = Path(tmp) / upload.filename
            with open(path, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            for record in ingest_pdf(str(path)):
                record["chunk_id"] = f"c{count:05d}"
                records.append(record)
                count += 1

    if not records:
        raise HTTPException(400, "No text found in those PDFs.")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    index = await asyncio.to_thread(Index.build, CHUNKS_PATH)
    await asyncio.to_thread(index.save, INDEX_DIR)
    _index = index

    return {"chunks": len(records),
            "documents": sorted({r["source"] for r in records})}


def event(name, payload):
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


async def run_pipeline(question: str):
    """The graph, unrolled, so each stage can be reported as it happens.

    graph.py stays the reference implementation and the nodes are imported
    from it — this is the same logic with yields between the steps.
    """
    index = get_index()
    state = {"question": question, "queries": [], "retrieved": [], "relevant": [],
             "attempts": 0, "status": None, "answer": None,
             "rejected_citations": []}

    while True:
        yield event("stage", {"stage": "rewriting", "attempt": state["attempts"] + 1})
        state.update(await asyncio.to_thread(rewrite, state))
        yield event("queries", {"queries": state["queries"]})

        yield event("stage", {"stage": "searching"})
        seen, chunks = set(), []
        for query in state["queries"]:
            for chunk in index.search(query, method="hybrid", top_k=RETRIEVE_K):
                if chunk["chunk_id"] not in seen:
                    seen.add(chunk["chunk_id"])
                    chunks.append(chunk)
        state["retrieved"] = chunks
        yield event("retrieved", {"count": len(chunks),
                                  "pages": sorted({c["page"] for c in chunks})})

        yield event("stage", {"stage": "checking evidence"})
        state.update(await asyncio.to_thread(grade, state))
        yield event("graded", {"kept": len(state["relevant"]),
                               "of": len(state["retrieved"])})

        if state["relevant"]:
            break
        if state["attempts"] >= MAX_ATTEMPTS:
            yield event("done", {
                "status": "insufficient",
                "text": "These documents don't contain enough information to "
                        "answer that.",
                "citations": [], "attempts": state["attempts"],
            })
            return
        yield event("stage", {"stage": "retrying with different wording"})

    yield event("stage", {"stage": "answering"})
    state.update(await asyncio.to_thread(answer, state))

    by_id = {c["chunk_id"]: c for c in state["relevant"]}
    citations = [
        {"chunk_id": cid, "page": by_id[cid]["page"],
         "source": by_id[cid]["source"], "heading": by_id[cid]["heading"],
         "text": by_id[cid]["text"]}
        for cid in state["answer"]["citations"] if cid in by_id
    ]
    yield event("done", {
        "status": state["status"],
        "text": state["answer"]["text"],
        "citations": citations,
        "attempts": state["attempts"],
        "rejected": state["rejected_citations"],
    })


@app.post("/ask")
async def ask_endpoint(payload: dict):
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "Ask a question first.")
    return StreamingResponse(run_pipeline(question),
                             media_type="text/event-stream")
