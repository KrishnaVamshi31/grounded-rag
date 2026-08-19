"""
Step 4 — End-to-end evaluation.

Three things get measured, and they fail independently:

  decision     answered when it should, refused when it should
  citation     do the cited pages overlap the gold pages
  groundedness does every claim actually follow from the cited passages

The third one needs a judge call and only runs with --judge. It is the one
that catches the interesting failure: a real citation attached to a claim the
passage never made.

    python eval_e2e.py                 # decision + citation only
    python eval_e2e.py --judge         # adds groundedness (1 extra call/question)
    python eval_e2e.py --limit 5       # short run while iterating
"""

import argparse
import json
import os
import time

import llm
from graph import ask, build_graph
from index import Index

JUDGE_SYSTEM = """You check whether an answer is grounded in its source passages.

Read the passages, then the answer. Identify any claim in the answer that is
not supported by the passages — including claims that are factually true but
absent from the sources.

Return JSON:
{"grounded": true/false, "unsupported": ["claim one", "claim two"]}"""


def judge(question, answer_text, chunks, model):
    """Judge with a DIFFERENT model than the one that wrote the answer.

    A model grading its own output is lenient in a way that is hard to see
    and easy to quote. Cross-family judging is the cheapest real check.
    """
    passages = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    verdict = llm.complete_json(
        JUDGE_SYSTEM,
        f"Question: {question}\n\nPassages:\n{passages}\n\nAnswer: {answer_text}",
        model=model,
    )
    return _normalize(verdict)


def _normalize(verdict):
    """Coerce whatever shape the judge returned into (grounded, unsupported).

    Models ignore the schema often enough that this belongs in code. Seen in
    practice: the object as specified, a bare list of unsupported claims, and
    the object wrapped in a single-element list.
    """
    if isinstance(verdict, list):
        if len(verdict) == 1 and isinstance(verdict[0], dict):
            verdict = verdict[0]                      # [{...}]
        else:
            claims = [str(v) for v in verdict]        # ["claim", "claim"]
            return (not claims), claims

    if not isinstance(verdict, dict):
        return None, []                               # unusable, don't count it

    unsupported = verdict.get("unsupported") or []
    if isinstance(unsupported, str):
        unsupported = [unsupported]
    grounded = verdict.get("grounded")
    if grounded is None:
        grounded = not unsupported                    # infer if the key is absent
    return bool(grounded), [str(c) for c in unsupported]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index")
    parser.add_argument("--eval", default="evalset.jsonl")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model",
                        default=os.environ.get("JUDGE_MODEL",
                                               "qwen/qwen3.6-27b"),
                        help="must differ from LLM_MODEL to be meaningful")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="pause between questions to stay under rate limits")
    args = parser.parse_args()

    if args.judge:
        answerer = os.environ.get("LLM_MODEL", "")
        print(f"answerer: {answerer}\njudge:    {args.judge_model}")
        if args.judge_model == answerer:
            print("WARNING: judge and answerer are the same model — "
                  "groundedness will be flattered")
        print()

    index = Index.load(args.index)
    app = build_graph(index)
    questions = [json.loads(line) for line in open(args.eval, encoding="utf-8")]
    if args.limit:
        questions = questions[:args.limit]

    rows = []
    for q in questions:
        result = ask(app, q["question"])
        answered = result["status"] == "answered"
        should_answer = bool(q["gold_pages"])

        cited_pages = result["answer"]["pages"] if result["answer"] else []
        page_hit = bool(set(cited_pages) & set(q["gold_pages"]))

        grounded, unsupported = None, []
        if args.judge and answered:
            try:
                grounded, unsupported = judge(q["question"],
                                              result["answer"]["text"],
                                              result["relevant"],
                                              args.judge_model)
            except Exception as exc:
                # One flaky judge response must not destroy a 14-question run.
                print(f"    judge failed on {q['id']}: "
                      f"{type(exc).__name__}: {str(exc)[:100]}")

        rows.append({
            "id": q["id"], "type": q["type"],
            "should_answer": should_answer, "answered": answered,
            "gold": q["gold_pages"], "cited": cited_pages,
            "page_hit": page_hit, "attempts": result["attempts"],
            "grounded": grounded, "unsupported": unsupported,
        })

        mark = "ok " if answered == should_answer else "BAD"
        if grounded is None and args.judge and answered:
            extra_note = "  judge:skipped"
        else:
            extra_note = ""
        extra = ""
        if answered:
            extra = f" cited={cited_pages} gold={q['gold_pages']}"
            if grounded is False:
                extra += "  UNGROUNDED"
        print(f"{mark} {q['id']:<4} {result['status']:<12} "
              f"attempts={result['attempts']}{extra}{extra_note}")
        time.sleep(args.sleep)

    # ---------------- summary ----------------
    answerable = [r for r in rows if r["should_answer"]]
    unanswerable = [r for r in rows if not r["should_answer"]]

    print("\n" + "=" * 46)
    if answerable:
        answered = [r for r in answerable if r["answered"]]
        print(f"answered when it should      "
              f"{len(answered)}/{len(answerable)}")
        print(f"cited a correct page         "
              f"{sum(r['page_hit'] for r in answerable)}/{len(answerable)}")
    if unanswerable:
        print(f"refused when it should       "
              f"{sum(not r['answered'] for r in unanswerable)}/{len(unanswerable)}")
    if args.judge:
        checked = [r for r in rows if r["grounded"] is not None]
        if checked:
            print(f"grounded answers             "
                  f"{sum(r['grounded'] for r in checked)}/{len(checked)}")
            for r in checked:
                for claim in r["unsupported"]:
                    print(f"    {r['id']}: {claim[:80]}")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote eval_results.json")


if __name__ == "__main__":
    main()
