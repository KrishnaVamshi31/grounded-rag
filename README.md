# Grounded — PDF question answering

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

The repo ships a **prebuilt index** (`index_jm7/` — 141 chunks from a 30-page
chapter), so you can go from clone to a working UI without supplying a PDF or
running ingestion. Note that `index/`, `chunks.jsonl`, and the source PDFs are
gitignored and will *not* be in your clone; `index_jm7/` is the one that is.

**You need:** Python 3.12 or newer (tested on 3.12 in the Docker image and
3.14 locally) and an OpenAI-compatible API key. `.env.example` points at Groq,
whose free tier is enough to try this.

### 1. Install

```bash
git clone https://github.com/KrishnaVamshi31/grounded-rag.git
cd grounded-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows the activate line is `.venv\Scripts\activate` instead.

### 2. Add a key

```bash
cp .env.example .env
```

Then edit `.env` and set `LLM_API_KEY`. A [Groq](https://console.groq.com) key
works with the base URL and model already in the file. `JUDGE_MODEL` only
matters if you run the end-to-end eval — see step 5.

### 3. Start it

```bash
INDEX_DIR=index_jm7 uvicorn api:app --reload
```

PowerShell:

```powershell
$env:INDEX_DIR = "index_jm7"; uvicorn api:app --reload
```

Open **http://localhost:8000**. `GET /health` should return
`{"indexed":true,"chunks":141,"model":"...","dense":true}`.

The first question takes an extra minute: sentence-transformers downloads
all-MiniLM-L6-v2 (~90MB) on first use and caches it. The Docker image bakes it
in and skips this.

Questions that work against the shipped index:

- *What is teacher forcing?* — answers, cites page 16
- *What is the difference between few-shot and zero-shot prompting?*
- *Why is greedy decoding not used in practice with large language models?*
- *What was Acme Corp's 2024 revenue?* — retries once, then refuses

That last one is the point of the project. Ask it something the chapter
doesn't cover and watch the stream retry with different wording, keep zero
passages, and decline.

### 4. Use your own PDF

Click **Upload PDFs** in the UI, or do it from the command line:

```bash
python ingest.py yourdoc.pdf --out chunks.jsonl
python index.py build --chunks chunks.jsonl --out index
uvicorn api:app --reload            # INDEX_DIR defaults to ./index
```

Uploading through the UI overwrites `chunks.jsonl` and `index/`; `index_jm7/`
is left alone, so the quickstart above keeps working.

### 5. Evaluate

Only the Book B index ships with the repo, so these two run against a fresh
clone as-is:

```bash
python eval_retrieval.py --index index_jm7 --eval evalset_jm7.jsonl
python inspect_retrieval.py --index index_jm7 --eval evalset_jm7.jsonl --method hybrid
```

The Book A numbers need that book ingested into `index/` first (step 4), after
which the defaults apply:

```bash
python eval_retrieval.py --eval evalset.jsonl        # recall@k per method
python eval_e2e.py --eval evalset.jsonl --judge      # decision, citation, groundedness
```

`--judge` needs `JUDGE_MODEL` set to a *different* model family than
`LLM_MODEL`, or the groundedness score is self-flattering — see the note under
"Measured results".

### Docker

```bash
docker compose up --build
```

The compose file bind-mounts `./index` and `./chunks.jsonl`, so build an index
locally (step 4) before using it — otherwise Docker creates an empty directory
where `chunks.jsonl` should be and the container starts with nothing indexed.

### If it stops mid-answer

Groq's free tier allows 8,000 tokens per minute. One question is comfortably
inside that; several in quick succession is not, and a 429 currently ends the
event stream without a `done` event, so the UI just stops. Wait a minute and
ask again, or move to a paid tier.

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
