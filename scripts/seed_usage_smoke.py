"""Seed synthetic usage_logs for smoke-testing the Usage & analytics page.

Usage:
    uv run otari init-db --database-url sqlite:///./smoke.db
    uv run python scripts/seed_usage_smoke.py sqlite:///./smoke.db
    uv run otari serve --database-url sqlite:///./smoke.db --master-key sk-smoke --port 8000

Spreads rows across ~50 days (so the current and previous 30-day windows both have
data and the deltas render), over several models / users / API keys, with a mix of
success and error rows, cache tokens, and latencies.
"""

import random
import sys
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from gateway.models.entities import APIKey, UsageLog, User

URL = sys.argv[1] if len(sys.argv) > 1 else "sqlite:///./smoke.db"
rng = random.Random(303)  # deterministic

MODELS = [
    ("gpt-5.6", "openai", 0.020, 900),
    ("claude-sonnet-5", "anthropic", 0.012, 1300),
    ("gemini-2.5-flash", "google", 0.002, 600),
    ("gpt-4o-mini", "openai", 0.001, 500),
]
USERS = ["alice", "bob", "carol", "dave"]
KEYS = [("key-prod", "alice"), ("key-staging", "bob"), ("key-batch", "carol")]

engine = create_engine(URL)
Session = sessionmaker(bind=engine)
db = Session()

# Users + keys the logs reference (FKs are ON DELETE SET NULL, but present here).
for uid in USERS:
    if db.query(User).filter(User.user_id == uid).first() is None:
        db.add(User(user_id=uid, alias=uid.capitalize(), spend=0.0, blocked=False))
for kid, owner in KEYS:
    if db.query(APIKey).filter(APIKey.id == kid).first() is None:
        db.add(APIKey(id=kid, key_hash=f"hash-{kid}", key_name=kid, user_id=owner, is_active=True))
db.flush()

now = datetime.now(UTC)
n = 0
for _ in range(1500):
    # Weighted toward recent so the "24h" and "7d" presets have plenty; tail reaches
    # ~50 days back so the previous-30d comparison window is populated too.
    age_days = rng.random() ** 2 * 50
    ts = now - timedelta(days=age_days, hours=rng.random() * 24)
    model, provider, unit_cost, base_latency = rng.choice(MODELS)
    is_error = rng.random() < 0.04
    kid, owner = rng.choice(KEYS)
    prompt = rng.randint(200, 4000)
    completion = 0 if is_error else rng.randint(50, 1500)
    cache_read = rng.choice([0, 0, 0, rng.randint(100, 2000)])
    db.add(
        UsageLog(
            id=str(uuid.uuid4()),
            user_id=owner,
            api_key_id=kid,
            timestamp=ts,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cache_read_tokens=cache_read or None,
            cache_write_tokens=None,
            cost=None if is_error else round((prompt + completion) / 1000 * unit_cost, 6),
            status="error" if is_error else "success",
            error_message="provider quota exceeded" if is_error else None,
            latency_ms=None if is_error else base_latency + rng.randint(-200, 800),
        )
    )
    n += 1

db.commit()
print(f"Seeded {n} usage rows into {URL}")
