"""
Per-question retrieval diagnostic.

Shows, for every answerable question, the pages the retriever returned and
where the gold page landed. Use it whenever a recall number moves and you
want to know which question caused it.

    python inspect_retrieval.py --method hybrid
"""

import argparse
import json

from index import Index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index")
    parser.add_argument("--eval", default="evalset.jsonl")
    parser.add_argument("--method", default="hybrid",
                        choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()

    index = Index.load(args.index)
    questions = [json.loads(line) for line in open(args.eval, encoding="utf-8")]

    for q in questions:
        if not q["gold_pages"]:
            continue
        results = index.search(q["question"], method=args.method,
                               top_k=args.depth)
        pages = [c["page"] for c in results]
        found = next((i for i, p in enumerate(pages, 1)
                      if p in q["gold_pages"]), None)
        verdict = f"rank {found}" if found else f"MISS beyond {args.depth}"

        print(f"{q['id']}  gold={q['gold_pages']}  {verdict}")
        print(f"      returned: {pages}")
        print(f"      {q['question'][:70]}\n")


if __name__ == "__main__":
    main()
