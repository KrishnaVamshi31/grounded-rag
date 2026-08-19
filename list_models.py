"""
Ask the provider what models this key can actually use.

    python list_models.py

Run this whenever you get a model_not_found — provider model IDs change,
and guessing is slower than asking.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

base_url = os.environ.get("LLM_BASE_URL")
print(f"endpoint: {base_url}\n")

client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=base_url or None)

for model in sorted(client.models.list().data, key=lambda m: m.id):
    print(" ", model.id)
