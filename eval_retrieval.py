"""
Measure retrieval, before any LLM is involved.

    python eval_retrieval.py --index index --eval evalset.jsonl

Metric: recall@k — for each answerable question, did at least one chunk from
a gold page appear in the top k? That is the ceiling on answer quality. If
the evidence never gets retrieved, no amount of prompting recovers it.
"""

import argparse
import json

from index import Index

KS = [1, 3, 5, 10]


def recalls(index, questions, method, ks):
    """One search per question at max(ks); every k is read off that ranking."""
    depth = max(ks)
    hits = {k: 0 for k in ks}
    for q in questions:
        results = index.search(q["question"], method=method, top_k=depth)
        ranks = [i for i, c in enumerate(results, 1)
                 if c["page"] in q["gold_pages"]]
        best = min(ranks) if ranks else None
        for k in ks:
            if best is not None and best <= k:
                hits[k] += 1
    return {k: hits[k] / len(questions) for k in ks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index")
    parser.add_argument("--eval", default="evalset.jsonl")
    parser.add_argument("--methods", nargs="+",
                        default=["bm25", "dense", "hybrid"])
    args = parser.parse_args()

    index = Index.load(args.index)
    questions = [json.loads(line) for line in open(args.eval, encoding="utf-8")]
    answerable = [q for q in questions if q["gold_pages"]]
    cross = [q for q in answerable if q["type"] == "cross_section"]

    print(f"{len(answerable)} answerable questions "
          f"({len(cross)} of them cross-section)\n")

    header = "method  " + "  ".join(f"r@{k}".rjust(6) for k in KS)
    print(header)
    print("-" * len(header))
    for method in args.methods:
        scores = recalls(index, answerable, method, KS)
        print(f"{method:<8}" + "  ".join(f"{scores[k]:.2f}".rjust(6) for k in KS))

    print("\ncross-section questions only")
    print(header)
    print("-" * len(header))
    for method in args.methods:
        scores = recalls(index, cross, method, KS)
        print(f"{method:<8}" + "  ".join(f"{scores[k]:.2f}".rjust(6) for k in KS))


if __name__ == "__main__":
    main()
