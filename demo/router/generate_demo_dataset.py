"""Generate the Router Demo dataset by comparing a premium and a budget model.

For each prompt both models answer, the prompt is embedded, and a strong model
judges each answer 0.0-1.0 (strictly against a reference answer when one is
given). The prompts are organized into semantic clusters on purpose: an easy
general-knowledge region where both models succeed (so the router can go cheap)
and harder code and math/puzzle regions where the budget model falls down (so
the router must reach for the premium model). The kNN router routes by semantic
neighborhood, so keeping coherent clusters is what makes routing visible.

The whole set is shipped (no spread-based curation) so each region keeps enough
members for the kNN neighbors. The result is written as a DemoData JSON the web
app replays offline (demo mode) and uses to drive the "See it route" view.

Run (the key is read from the environment, never written to the output):

    OPENAI_API_KEY=sk-... uv run python demo/router/generate_demo_dataset.py

Output: demo/router/src/demo_prompts.json
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any, cast

from any_llm import acompletion, aembedding
from any_llm.types.completion import ChatCompletion

# --- model tiers -------------------------------------------------------------
# Prices are illustrative per-tier estimates (USD per 1M tokens); verify against
# current OpenAI pricing before quoting real numbers in a demo.
JUDGE_MODEL = "gpt-5.4"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 256

# A four-tier ladder with a deliberately wide capability gap so routing has a real
# spectrum to choose from: easy prompts route to the cheapest tier, hard ones to
# the cheapest tier that is still good enough, and only the hardest to the top.
# Prices are ILLUSTRATIVE and chosen to increase monotonically with capability so
# no tier is strictly dominated (in reality gpt-4o-mini is cheaper than
# gpt-3.5-turbo, which would make gpt-3.5 dead weight). Verify against current
# OpenAI pricing before quoting real numbers.
MODELS: list[dict[str, Any]] = [
    {"id": "openai:gpt-5.4", "model": "gpt-5.4", "label": "gpt-5.4", "in": 5.00, "out": 15.00},
    {"id": "openai:gpt-4o", "model": "gpt-4o", "label": "gpt-4o", "in": 2.50, "out": 10.00},
    {"id": "openai:gpt-4o-mini", "model": "gpt-4o-mini", "label": "gpt-4o-mini", "in": 0.60, "out": 1.80},
    {"id": "openai:gpt-3.5-turbo", "model": "gpt-3.5-turbo", "label": "gpt-3.5-turbo", "in": 0.20, "out": 0.60},
]
# Proxy $/1k calls assuming ~500 input + ~300 output tokens, for the cost story.
AVG_IN, AVG_OUT = 500, 300


def proxy_price(m: dict[str, Any]) -> float:
    return round((float(m["in"]) * AVG_IN + float(m["out"]) * AVG_OUT) / 1_000_000 * 1000, 4)


# --- prompts, grouped into semantic clusters ---------------------------------
# Cluster 1 (general-knowledge): both models succeed -> the router can route
# cheap and save. Clusters 2 and 3 (hard code, hard math/puzzles): the budget
# model tends to fail -> the router routes to the premium model. Because the kNN
# routes by neighborhood, each cluster needs enough members to dominate its
# neighbors. Objective prompts carry a `ref` ground truth for strict grading.
CANDIDATES: list[dict[str, str]] = [
    # --- easy: general knowledge (route cheap) -------------------------------
    {"task": "general-knowledge", "prompt": "What is the capital of Australia?", "ref": "Canberra"},
    {"task": "general-knowledge", "prompt": "In what year did the Berlin Wall fall?", "ref": "1989"},
    {"task": "general-knowledge", "prompt": "What is the chemical symbol for gold?", "ref": "Au"},
    {"task": "general-knowledge", "prompt": "What is the largest planet in our solar system?", "ref": "Jupiter"},
    {
        "task": "general-knowledge",
        "prompt": "Who wrote the play Romeo and Juliet?",
        "ref": "William Shakespeare",
    },
    {"task": "general-knowledge", "prompt": "How many continents are there on Earth?", "ref": "Seven (7)"},
    {"task": "general-knowledge", "prompt": "List the first five prime numbers.", "ref": "2, 3, 5, 7, 11"},
    {
        "task": "general-knowledge",
        "prompt": "Convert 72 degrees Fahrenheit to Celsius. Give just the number rounded to one decimal.",
        "ref": "22.2",
    },
    {
        "task": "general-knowledge",
        "prompt": "Translate the phrase 'good morning' into Spanish.",
        "ref": "Buenos días",
    },
    # --- hard: algorithmic code (route strong) -------------------------------
    {
        "task": "hard-code",
        "prompt": (
            "Implement an LRU cache in Python supporting get and put in O(1) time. "
            "Briefly explain the data structures you use."
        ),
    },
    {
        "task": "hard-code",
        "prompt": (
            "Implement Dijkstra's shortest-path algorithm in Python using a binary heap. "
            "State the time complexity and why."
        ),
    },
    {
        "task": "hard-code",
        "prompt": "Detect whether a singly linked list has a cycle, in O(1) extra space, in Python. Explain the method.",
    },
    {
        "task": "hard-code",
        "prompt": (
            "Compute the length of the longest strictly increasing subsequence of "
            "[10, 9, 2, 5, 3, 7, 101, 18] and explain the method."
        ),
        "ref": "4",
    },
    {
        "task": "hard-code",
        "prompt": "Write a regular expression that matches a valid IPv4 address (each octet 0 to 255). Explain it.",
    },
    {
        "task": "hard-code",
        "prompt": "Write an iterative (non-recursive) in-order traversal of a binary tree in Python.",
    },
    # --- hard: math and puzzles (route strong) -------------------------------
    {
        "task": "hard-math",
        "prompt": (
            "You have 25 horses and a track that races 5 at a time, with no timer. What is the minimum number of "
            "races needed to find the 3 fastest horses?"
        ),
        "ref": "7",
    },
    {
        "task": "hard-math",
        "prompt": (
            "Four people must cross a bridge at night with one torch that must be carried; at most two cross at a "
            "time and a pair moves at the slower person's pace. Their times are 1, 2, 5, and 10 minutes. What is "
            "the minimum total time for all four to cross?"
        ),
        "ref": "17 minutes",
    },
    {
        "task": "hard-math",
        "prompt": "A fair coin is flipped 10 times. What is the probability of exactly 3 heads? Give an exact fraction.",
        "ref": "15/128 (which is 120/1024)",
    },
    {
        "task": "hard-math",
        "prompt": "What is the determinant of the matrix [[2, 5, 3], [1, -2, -1], [1, 3, 4]]?",
        "ref": "-20",
    },
    {
        "task": "hard-math",
        "prompt": "How many trailing zeros are there in 100 factorial (100!)?",
        "ref": "24",
    },
    {
        "task": "hard-math",
        "prompt": "Compute 7^100 mod 13.",
        "ref": "9",
    },
]

KEY = os.environ.get("OPENAI_API_KEY")
ANSWER_MAX_TOKENS = 700
SEM = asyncio.Semaphore(6)


async def _answer(model: str, prompt: str) -> str:
    async with SEM:
        r = cast(
            ChatCompletion,
            await acompletion(
                model=model,
                provider="openai",
                api_key=KEY,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=ANSWER_MAX_TOKENS,
                temperature=0,
            ),
        )
        return (r.choices[0].message.content or "").strip()


async def _judge(prompt: str, answer: str, ref: str | None) -> float:
    if ref:
        grading = (
            "You are strictly grading an AI assistant's answer to a question for CORRECTNESS.\n"
            "Score 0.0 to 1.0 where 1.0 is fully correct and 0.0 is wrong. Be strict: if the "
            "final answer is factually or numerically wrong, score 0.2 or below even if the "
            "reasoning looks plausible. Judge correctness against the reference, ignore style.\n\n"
            f"QUESTION:\n{prompt}\n\n"
            f"REFERENCE ANSWER (ground truth):\n{ref}\n\n"
            f"ASSISTANT ANSWER:\n{answer}\n\n"
            "Respond with ONLY a number between 0 and 1."
        )
    else:
        grading = (
            "You are grading an AI assistant's answer. Given the user prompt and the answer, "
            "rate the answer's correctness and overall quality from 0.0 to 1.0, where 1.0 is "
            "fully correct and high quality and 0.0 is wrong or unhelpful. "
            "Respond with ONLY a number between 0 and 1.\n\n"
            f"PROMPT:\n{prompt}\n\nANSWER:\n{answer}"
        )
    async with SEM:
        r = cast(
            ChatCompletion,
            await acompletion(
                model=JUDGE_MODEL,
                provider="openai",
                api_key=KEY,
                messages=[{"role": "user", "content": grading}],
                max_tokens=10,
                temperature=0,
            ),
        )
    raw = (r.choices[0].message.content or "").strip()
    try:
        return max(0.0, min(1.0, float(raw.split()[0])))
    except (ValueError, IndexError):
        return 0.0


async def _embed(prompt: str) -> list[float]:
    async with SEM:
        r = await aembedding(model=EMBED_MODEL, provider="openai", api_key=KEY, inputs=prompt)
    # Truncate to EMBED_DIMS and L2-renormalize: text-embedding-3-small is
    # Matryoshka-trained, so this approximates the reduced-dim model and keeps
    # the bundled JSON small. (any-llm mishandles the OpenAI `dimensions` arg.)
    vec = [float(x) for x in r.data[0].embedding][:EMBED_DIMS]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _spread(scores: list[float]) -> float:
    return max(scores) - min(scores)


async def build_item(idx: int, candidate: dict[str, str]) -> dict[str, Any]:
    task, prompt, ref = candidate["task"], candidate["prompt"], candidate.get("ref")
    answers = await asyncio.gather(*[_answer(m["model"], prompt) for m in MODELS])
    scores = await asyncio.gather(*[_judge(prompt, a, ref) for a in answers])
    embedding = await _embed(prompt)
    responses = [
        {"model": m["id"], "label": m["label"], "text": a, "score": round(s, 3)}
        for m, a, s in zip(MODELS, answers, scores)
    ]
    summary = "  ".join(f"{m['label']}={s:.2f}" for m, s in zip(MODELS, scores))
    print(f"[{idx + 1:>2}/{len(CANDIDATES)}] {task:<18} spread={_spread(scores):.2f}  {summary}  | {prompt[:38]}")
    return {"id": f"p{idx:02d}", "task": task, "prompt": prompt, "responses": responses, "embedding": embedding}


async def main() -> None:
    if not KEY:
        raise SystemExit("Set OPENAI_API_KEY in the environment.")
    items = []
    for idx, candidate in enumerate(CANDIDATES):
        items.append(await build_item(idx, candidate))
    diverge = sum(1 for it in items if _spread([r["score"] for r in it["responses"]]) >= 0.2)
    print(f"\n{diverge}/{len(items)} prompts diverge (>= 0.2 spread); the rest are agree-cheap regions.")
    data = {
        "models": [{"id": m["id"], "label": m["label"], "price": proxy_price(m)} for m in MODELS],
        "items": items,
    }
    out = Path(__file__).resolve().parent / "src" / "demo_prompts.json"
    out.write_text(json.dumps(data, indent=1))
    print(f"Wrote {out} ({len(items)} prompts, {len(MODELS)} models)")


if __name__ == "__main__":
    asyncio.run(main())
