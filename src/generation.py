import os
import time

from google import genai
from google.genai import types

from src import config

SYSTEM_PROMPT = """You are a careful research assistant answering questions about India's Periodic Labour Force Survey (PLFS) Annual Report 2025.

Rules:
1. Use ONLY the numbered context passages provided. Do not use outside knowledge.
2. Cite every factual claim with the passage number in square brackets, e.g. [1] or [2].
3. If the passages do not contain the answer, reply exactly: "I could not find this in the provided PLFS reports."
4. Quote numbers exactly as written, with their units and reference period (for example 2025, usual status ps+ss, age group).
5. Many passages are table fragments. If you cannot tell which row or column a number belongs to, do not guess: say the table layout is unclear and cite the passage.
6. If the question can be read several ways (usual status vs current weekly status, rural vs urban, male vs female), give each relevant figure briefly, clearly labelled.
7. Keep the answer short and precise."""

RETRY_CODES = {429, 500, 502, 503, 504}   # rate limit / server overloaded: worth retrying


def format_context(passages):
    blocks = []
    for number, p in enumerate(passages, start=1):
        where = f"PDF page {p['page']}"
        if p.get("printed_page"):
            where += f", printed page {p['printed_page']}"
        blocks.append(f"[{number}] ({p['report']}, {where})\n{p['text']}")
    return "\n\n".join(blocks)


_client = None

def get_client(api_key=None):
    """Create the Gemini client once and reuse it."""
    global _client
    key = api_key or os.environ.get("GeminiAPIkey") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("No API key found. Set GeminiAPIkey or GEMINI_API_KEY.")
    if _client is None:
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30_000))  # 30 s, in milliseconds
    return _client


def build_user_prompt(question, passages):
    return f"Context passages:\n\n{format_context(passages)}\n\nQuestion: {question}"


def _call_model(client, model, prompt):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=config.MAX_ANSWER_TOKENS,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text or "The model returned no text. Please try again."


def generate_answer(question, passages, api_key=None, retries=2):
    client = get_client(api_key)
    prompt = build_user_prompt(question, passages)
    models = [m for m in (config.LLM_MODEL, config.FALLBACK_MODEL) if m]
    for model in models:
        for attempt in range(retries):
            try:
                return _call_model(client, model, prompt)
            except Exception as e:
                code = getattr(e, "code", None)
                timed_out = "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower()
                if code not in RETRY_CODES and not timed_out:
                    raise                              # a real bug: show it
                wait = 5 * 2 ** attempt
                print(f"{model}: {'timeout' if timed_out else code}; retrying in {wait}s")
                time.sleep(wait)
        print(f"{model} still busy, trying the next model")
    return "The Gemini service is busy right now. Please try again in a minute."
