"""
Thin LLM wrapper. Everything that talks to a model goes through here, so
swapping providers is a two-line change and the retry/parse logic lives in
exactly one place.

Set these before running (PowerShell):
    $env:LLM_BASE_URL = "https://..."     # your AICredits endpoint
    $env:LLM_API_KEY  = "sk-..."
    $env:LLM_MODEL    = "claude-3-5-haiku-20241022"
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)   # .env wins over stale shell variables

_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_BASE_URL") or None,
        )
    return _client


def model_name():
    return os.environ.get("LLM_MODEL", "claude-3-5-haiku-20241022")


def complete(system, user, temperature=0.0, max_tokens=1024, model=None):
    response = client().chat.completions.create(
        model=model or model_name(),
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return response.choices[0].message.content


def _strip_reasoning(text):
    """Remove <think>...</think> and similar blocks reasoning models emit."""
    return re.sub(r"<(think|thinking|reasoning)>.*?</\1>", "", text,
                  flags=re.S | re.I).strip()


def _first_json(text):
    r"""Scan for the first balanced {...} or [...] and return it.

    A regex like [\[{].*[\]}] fails on reasoning output because it latches
    onto the first bracket anywhere in the prose. Counting depth and ignoring
    brackets inside strings is the only version that survives real models.
    """
    starts = [i for i, ch in enumerate(text) if ch in "{["]
    for start in starts:
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None


def complete_json(system, user, **kwargs):
    """Get JSON out of a model, whatever wrapper it decides to add.

    Handles: clean JSON, fenced JSON, JSON after a preamble, and reasoning
    models that narrate before answering. Pass model="..." to override the
    default for a single call.
    """
    raw = complete(system + "\n\nRespond with JSON only. No prose, no fences.",
                   user, **kwargs)
    text = _strip_reasoning(raw.strip())

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    parsed = _first_json(text)
    if parsed is not None:
        return parsed

    raise ValueError(
        f"model did not return JSON (model={kwargs.get('model') or model_name()}).\n"
        f"first 300 chars:\n{raw[:300]}"
    )
