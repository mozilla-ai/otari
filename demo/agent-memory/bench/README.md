# Automatic recall and consolidation benches

A memory bigger than the demo's, with questions whose right answer is known, to measure what recall hands over
in each mode, and a set of proposals to measure how well a model consolidates. The recall bench uses the demo's
workspace and **erases its memory**: afterwards, `../terminal/prepare.sh` puts it back.

- `dataset.py`: 63 entries of a made-up team (staging and production deploys, incidents, decisions, meetings,
  each person's work and hobbies, billing, the local environment, a few in Spanish and two private to Alice)
  and 45 cases: direct questions, near misses, follow-ups, questions about people, changes of subject,
  cross-language questions, questions memory has no answer to, and privacy.
- `bench.py`: seeds, evaluates and adds filler.
- `query_variants.py`: tries, outside the gateway, what query to send the reranker.
- `overhead.py`: what automatic recall adds to a request's time.
- `consolidation_cases.py`, `consolidation_bench.py`: 34 proposals whose right decision is known, and how well
  a model decides them.

## Recall

```bash
cd demo/agent-memory/bench

# 1. Comment out memory_consolidation_model in ../otari-config.yml and restart the gateway (../start.sh), so each
#    entry is kept as written, without a model merging similar ones.
python3 bench.py seed

# 2. Evaluate. Modes: similarity (0.65, the default), broad (0.60), rerank (0.8, the default), or with a
#    threshold: rerank:0.9
python3 bench.py eval similarity broad rerank rerank:0.9

# 3. A crowded memory: 1,000 entries about made-up services, and evaluate again
python3 bench.py filler 1000
python3 bench.py eval broad rerank:0.9

# 4. When done: uncomment memory_consolidation_model, restart the gateway, and ../terminal/prepare.sh
```

What it measures:

- **found**: of the entries that answer each question, how many were handed over.
- **precision**: of what was handed over, how much answered (or was acceptable beside the answer, `ok` in the
  case).
- **nothing**: questions memory has no answer to, where nothing was handed over.
- **privacy**: Alice's private entries that reached Bob (must be 0).
- **latency**: of the turn's embedding (`/v1/memory/recall`) and of the rerank (`/v1/memory/rerank`), and how
  many recalls went past the 3 s budget.

Each `eval` writes the case-by-case detail to `results-<date>.json` (gitignored).

Overhead: `python3 overhead.py rerank:0.9 15` alternates the same request with memory declined and with it.

## Consolidation

It imports the consolidator's prompt and guard from the gateway, so it runs from the repo root with `uv`:

```bash
uv run python demo/agent-memory/bench/consolidation_bench.py \
  fireworks:accounts/fireworks/models/deepseek-v4p1-flash ollama:qwen2.5:7b
```

`ollama pull qwen2.5:7b` first for the local model. With the organization's prompt-injection guardrail on, pause
it in the dashboard while the bench runs: it asks the models through `/v1/chat/completions`, and the decide
prompt can be flagged.
