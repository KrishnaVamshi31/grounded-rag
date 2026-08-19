# Grounded — PDF question answering that refuses

A document Q&A system built around one property: **when the documents don't
contain the answer, it says so instead of writing a confident paragraph from
the four least-irrelevant chunks.**

Naive RAG can't do this. It retrieves a fixed top-k, stuffs them into a
prompt, and generates — so every question produces an answer, whether or not
the evidence exists. This system decides what to search for, checks whether
what it found is actually relevant, searches again with different wording if
not, and returns nothing when nothing qualifies.

---

## Measured results

Evaluated on two documents with hand-verified question sets:

- **Book A** — *Natural Language Processing with Transformers*, 107 pages,
  296 chunks, 14 questions (10 single-fact, 2 cross-section, 2 unanswerable)
- **Book B** — *Speech and Language Processing* ch. 7, 30 pages, 141 chunks,
  31 questions (26 single-fact, 2 cross-section, 3 unanswerable)

### Retrieval — recall@k

**Book A** — 14 questions

| method | r@1 | r@3 | r@5 | r@10 |
|--------|-----|-----|-----|------|
| BM25   | 0.50 | 0.67 | 0.83 | **1.00** |
| Dense  | 0.50 | 0.83 | 0.83 | 0.92 |
| Hybrid | **0.67** | **0.83** | **0.92** | 0.92 |

**Book B** — 31 questions

| method | r@1 | r@3 | r@5 | r@10 |
|--------|-----|-----|-----|------|
| BM25   | **0.86** | **0.89** | **0.93** | **1.00** |
| Dense  | 0.68 | 0.82 | 0.86 | 0.93 |
| Hybrid | 0.82 | **0.89** | **0.93** | **1.00** |

**Hybrid does not win uniformly, and the exception is informative.** On Book A
it lifts r@1 from 0.50 to 0.67. On Book B, BM25 alone beats it at r@1
(0.86 vs 0.82).

The difference is question phrasing. Book B's questions were written by
paraphrasing the source closely, so they reuse the chapter's own rare terms —
*in-context learning*, *teacher forcing*, *data contamination*, *The Pile*.
Exact rare terms are what BM25 is built for and what embeddings smear
together. Book A's questions diverge more from the source wording, and hybrid
wins there.

So the defensible claim is conditional: **fusing dense retrieval with BM25
helps when the question is worded differently from the document, and costs a
little when the question uses the document's own vocabulary.** Since real
users don't phrase questions in the document's terms, Book A is the closer
analogue to production and Book B is closer to a worst case for hybrid.

Hybrid also trails BM25 at r@10 on Book A (0.92 vs 1.00). That is a property
of reciprocal rank fusion, not noise: a chunk only one retriever finds gets
outvoted by chunks both agree on and can fall past rank 10. Acceptable when
retrieving at k=5 with a loop that can re-query, and worth knowing.

On the cross-section questions of Book A — answers split across distant pages —
BM25 plateaus at 0.50 while dense and hybrid reach 1.00 by r@3.

### End-to-end (Book A)

| metric | result |
|--------|--------|
| answered when it should | 12/12 |
| refused when it should | 2/2 |
| cited a correct page | 10/12 |

Groundedness is judged by a **different model family** than the one writing
the answers (Qwen judging GPT-OSS). Judged by itself, the system scored 12/12
— including on a question where it asserted a fact about softmax saturation
appearing in neither cited passage. Cross-family judging caught it.
Self-judged eval numbers are not evidence.

---

## How it works

```
question
   │
rewrite ──► retrieve ──► grade ──┬─► answer ──► verify citations ──► response
   ▲                             │
   └──── retry once ─────────────┴─► refuse
```

**Ingestion** (`ingest.py`) — PyMuPDF, page-bounded chunks so citations point
somewhere exact. Headings come from the PDF outline when it has one, falling
back to font-size heuristics. Running headers and footers are detected by
finding lines that repeat in the page margins across the document and
dropped; on the test book that removed contamination from 15% of chunks.

**Index** (`index.py`) — BM25 implemented directly (~40 lines) alongside
dense embeddings from all-MiniLM-L6-v2, merged with reciprocal rank fusion.
RRF rather than score blending because BM25 scores are unbounded while cosine
sits in [-1,1]; fusing on rank position avoids normalizing across
incompatible scales.

**Loop** (`graph.py`) — LangGraph. Rewriting turns the user's phrasing into
the document's phrasing. Grading is one call over all retrieved passages
rather than one per passage — cheaper, and models judge better when passages
can be compared side by side. Retry is capped at one.

**Citation verification** — the answer prompt asks the model to cite only
supplied passages; the code then checks. Citations naming passages that
weren't in the graded set are stripped, and if none survive, the answer is
discarded and the system refuses. A prompt is a request; this is a guarantee.

**Service** (`api.py`) — FastAPI. `/ask` streams Server-Sent Events for each
pipeline stage, so the UI shows the queries it tried, how many passages it
kept, and whether it retried.

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add your key
python ingest.py yourdoc.pdf --out chunks.jsonl
python index.py build --chunks chunks.jsonl
uvicorn api:app --reload
```

Docker:

```bash
docker compose up --build
```

Evaluate:

```bash
python eval_retrieval.py --eval evalset.jsonl              # recall@k per method
python eval_retrieval.py --index index_jm7 --eval evalset_jm7.jsonl
python eval_e2e.py --judge               # decision, citation, groundedness
python inspect_retrieval.py --method hybrid   # per-question diagnosis
```

---

## Limitations

- **Small samples.** 14 and 31 questions. One question moves a metric by
  several points. These show direction, not precision.
- **The eval sets are LLM-generated**, spot-checked by hand. Two biases
  follow. Answer quality is flattered slightly, since the generator and the
  answerer share a lineage — retrieval recall is unaffected, as a chunk
  either surfaces or it doesn't. More importantly, questions written by
  paraphrasing source text inherit that text's vocabulary, which
  systematically favors keyword retrieval. That is most of the Book A / Book
  B gap above, and it means neither number transfers cleanly to real user
  questions.
- **Fusion weights untuned.** BM25 and dense contribute equally to the RRF
  score. Weighting them is the obvious next experiment, but tuning weights
  against 31 questions would fit the eval set rather than the problem.
- **Citations land on the demonstration, not the explanation.** Both misses
  in the 10/12 cited a page showing the concept in code rather than the page
  explaining it. The grader counts both as relevant, correctly; ranking
  explanation above demonstration is unsolved here.
- **Retry fires only on zero relevant passages.** Thin-but-nonzero evidence
  should also trigger it.
- **No table or figure handling.** Text only; tables are flattened into prose
  and degrade accordingly.
