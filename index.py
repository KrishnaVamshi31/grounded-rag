"""
Step 2 — The index.

Three retrievers over the same chunks:
  bm25   keyword matching (exact terms, model numbers, rare words)
  dense  embedding similarity (paraphrase, synonyms, "what is this about")
  hybrid the two fused with reciprocal rank fusion

Build once, then query:
    python index.py build --chunks chunks.jsonl
    python index.py query "why scale the attention scores" --method hybrid
"""

import argparse
import functools
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

K1 = 1.5        # BM25 term-frequency saturation
B = 0.75        # BM25 length normalization
RRF_K = 60      # reciprocal rank fusion constant (60 is the value from the paper)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "does", "do",
}


def tokenize(text):
    """Lowercase, split on non-alphanumerics, drop stopwords.

    Numbers survive on purpose — '768' and 'k=3' are exactly the kind of thing
    dense retrieval smears away and BM25 nails.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25:
    """Okapi BM25. ~40 lines, no dependency, and you can explain every term."""

    def __init__(self, corpus_tokens):
        self.docs = corpus_tokens
        self.n = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean())

        # postings: term -> {doc_index: term frequency}
        self.postings = defaultdict(dict)
        for i, tokens in enumerate(corpus_tokens):
            for term, freq in Counter(tokens).items():
                self.postings[term][i] = freq

        # idf: rare terms are worth more
        self.idf = {}
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens):
        out = np.zeros(self.n, dtype=np.float32)
        for term in query_tokens:
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for doc_index, freq in posting.items():
                norm = 1 - B + B * self.doc_len[doc_index] / self.avg_len
                out[doc_index] += idf * (freq * (K1 + 1)) / (freq + K1 * norm)
        return out


@functools.lru_cache(maxsize=1)
def _load_model(model_name):
    """Load the embedding model once per process, not once per call.

    Without the cache, every query rebuilds a 90MB model from disk — which
    turns a 2-second eval into a several-minute one.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts, model_name=EMBED_MODEL, progress=False):
    """Dense vectors, L2-normalized so a dot product is cosine similarity."""
    vectors = _load_model(model_name).encode(
        texts, batch_size=32, show_progress_bar=progress,
        normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def rank(scores, top_k):
    """Indices of the top_k highest scores, best first, zeros excluded."""
    if top_k >= len(scores):
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, top_k)[:top_k]
        order = part[np.argsort(-scores[part])]
    return [int(i) for i in order if scores[i] > 0][:top_k]


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Merge ranked lists by rank position, not by score.

    The two retrievers produce scores on completely different scales — BM25
    is unbounded, cosine sits in [-1, 1]. Normalizing them to compare is
    fiddly and dataset-dependent. RRF sidesteps it: only position counts.
    A document at rank 1 contributes 1/61, at rank 2 contributes 1/62, and
    a document both retrievers like beats one that only a single retriever
    ranks highly.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for position, doc_index in enumerate(ranking):
            fused[doc_index] += 1.0 / (k + position + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


class Index:
    def __init__(self, chunks, tokens, vectors):
        self.chunks = chunks
        self.tokens = tokens
        self.bm25 = BM25(tokens)
        self.vectors = vectors

    @classmethod
    def build(cls, chunks_path, skip_dense=False):
        chunks = [json.loads(line) for line in open(chunks_path, encoding="utf-8")]

        # Prepend the section heading to the text before indexing: it is real
        # context ("Self-Attention" tells you what the paragraph is about) and
        # it costs nothing.
        texts = [f"{c['heading']}\n{c['text']}" if c.get("heading") else c["text"]
                 for c in chunks]

        tokens = [tokenize(t) for t in texts]
        vectors = None if skip_dense else embed(texts, progress=True)
        return cls(chunks, tokens, vectors)

    def save(self, directory="index"):
        path = Path(directory)
        path.mkdir(exist_ok=True)
        with open(path / "index.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "tokens": self.tokens}, f)
        if self.vectors is not None:
            np.save(path / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, directory="index"):
        path = Path(directory)
        with open(path / "index.pkl", "rb") as f:
            data = pickle.load(f)
        vectors_path = path / "vectors.npy"
        vectors = np.load(vectors_path) if vectors_path.exists() else None
        return cls(data["chunks"], data["tokens"], vectors)

    def search(self, query, method="hybrid", top_k=5):
        pool = max(top_k * 4, 20)   # fuse deep lists, return a shallow one

        bm25_ranking = rank(self.bm25.scores(tokenize(query)), pool)

        dense_ranking = []
        if method in ("dense", "hybrid") and self.vectors is not None:
            query_vector = embed([query])[0]
            dense_ranking = rank(self.vectors @ query_vector, pool)

        if method == "bm25":
            chosen = bm25_ranking
        elif method == "dense":
            chosen = dense_ranking
        else:
            chosen = reciprocal_rank_fusion([bm25_ranking, dense_ranking])

        return [self.chunks[i] for i in chosen[:top_k]]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--chunks", default="chunks.jsonl")
    build.add_argument("--out", default="index")
    build.add_argument("--skip-dense", action="store_true")

    query = sub.add_parser("query")
    query.add_argument("text")
    query.add_argument("--method", default="hybrid",
                       choices=["bm25", "dense", "hybrid"])
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--index", default="index")

    args = parser.parse_args()

    if args.command == "build":
        index = Index.build(args.chunks, skip_dense=args.skip_dense)
        index.save(args.out)
        print(f"indexed {len(index.chunks)} chunks -> {args.out}/")
    else:
        index = Index.load(args.index)
        for i, chunk in enumerate(index.search(args.text, args.method, args.top_k), 1):
            print(f"{i}. p{chunk['page']} [{chunk['heading']}]")
            print(f"   {chunk['text'][:160]}...\n")


if __name__ == "__main__":
    main()
