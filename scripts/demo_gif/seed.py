"""Seed a realistic-looking standalone gateway for the README dashboard GIF.

Populates providers, budgets, users (assigned to budgets, with spend), named API
keys, model pricing, aliases, and ~1500 usage_logs spread over ~50 days so the
Overview / Usage / Activity pages all render with live-looking data. Everything
is deterministic (seeded RNG, fixed reference time passed in) so re-running
produces the same GIF.

Usage:
    uv run otari migrate --config scripts/demo_gif/otari.yml
    OTARI_SECRET_KEY=<fernet> uv run python scripts/demo_gif/seed.py sqlite:///./scripts/demo_gif/demo.db

The reference "now" is fixed so the seed is reproducible; the usage window is
anchored to it. See scripts/demo_gif/record.sh for the full pipeline.
"""

import math
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from gateway.models.entities import (
    APIKey,
    Budget,
    ModelAlias,
    ModelPricing,
    UsageLog,
    User,
)

URL = sys.argv[1] if len(sys.argv) > 1 else "sqlite:///./scripts/demo_gif/demo.db"
rng = random.Random(4242)  # deterministic values across runs

# Anchor the usage window to the real clock so "today" and the 30-day windows
# always have fresh data whenever the GIF is regenerated. Token/cost values stay
# deterministic (seeded RNG); only the timestamps track the current date.
NOW = datetime.now(UTC)
MONTH = 30 * 24 * 3600

# The providers themselves are declared in scripts/demo_gif/otari.yml (config
# providers with a `models:` list, so the gateway reports them healthy without a
# live upstream). model_key here is `<provider instance>:<model>` and MUST match
# the instances/models declared there.
#
# model_key -> (provider, input $/M, output $/M, base latency ms)
MODELS = {
    "openai:gpt-5.6": ("openai", 1.25, 10.00, 950),
    "openai:gpt-4o-mini": ("openai", 0.15, 0.60, 480),
    "anthropic:claude-sonnet-5": ("anthropic", 3.00, 15.00, 1300),
    "google:gemini-2.5-flash": ("google", 0.30, 2.50, 620),
    "groq:llama-3.3-70b": ("groq", 0.59, 0.79, 320),
    "mistral:mistral-large": ("mistral", 2.00, 6.00, 700),
}

# budget name -> (target utilization, duration_sec). The dollar limit is derived
# from actual seeded spend so the utilization bars land near these targets
# regardless of the RNG draw.
BUDGETS = {
    "Engineering": (0.44, MONTH),
    "Research": (0.61, MONTH),
    "Data Science": (0.76, MONTH),
    "Personal": (0.90, MONTH),
}

# user_id -> (display alias, budget name)
USERS = {
    "alice@acme.ai": ("Alice Nguyen", "Engineering"),
    "bob@acme.ai": ("Bob Martins", "Engineering"),
    "carol@acme.ai": ("Carol Diaz", "Research"),
    "dave@acme.ai": ("Dave Okafor", "Data Science"),
    "erin@acme.ai": ("Erin Cole", "Data Science"),
    "frank@acme.ai": ("Frank Li", "Personal"),
}

# key name -> owner user_id
KEYS = {
    "prod-api": "alice@acme.ai",
    "alice-notebook": "alice@acme.ai",
    "ci-pipeline": "bob@acme.ai",
    "research": "carol@acme.ai",
    "etl-batch": "dave@acme.ai",
    "dashboards": "erin@acme.ai",
    "personal-cli": "frank@acme.ai",
}

ALIASES = {
    "gpt": "openai:gpt-5.6",
    "sonnet": "anthropic:claude-sonnet-5",
    "flash": "google:gemini-2.5-flash",
    "cheap": "openai:gpt-4o-mini",
    "fast": "groq:llama-3.3-70b",
}

engine = create_engine(URL)
Session = sessionmaker(bind=engine)
db = Session()

# --- Budgets -----------------------------------------------------------------
# Created with a placeholder limit; the real max_budget is derived from seeded
# spend after the usage rows exist (see "Derive budget limits" below).
budgets: dict[str, Budget] = {}
budget_ids: dict[str, str] = {}
for name, (_target, duration) in BUDGETS.items():
    bid = str(uuid.uuid4())
    budget_ids[name] = bid
    b = Budget(budget_id=bid, name=name, max_budget=0.0, budget_duration_sec=duration)
    budgets[name] = b
    db.add(b)

# --- Users -------------------------------------------------------------------
for uid, (alias, budget_name) in USERS.items():
    if db.get(User, uid) is not None:
        continue
    started = NOW - timedelta(days=rng.randint(6, 20))
    db.add(
        User(
            user_id=uid,
            alias=alias,
            spend=0.0,  # set from generated usage below
            reserved=0.0,
            blocked=False,
            budget_id=budget_ids[USERS[uid][1]],
            budget_started_at=started,
            next_budget_reset_at=started + timedelta(seconds=MONTH),
        )
    )

# --- API keys ----------------------------------------------------------------
for kname, owner in KEYS.items():
    kid = f"key-{kname}"
    if db.get(APIKey, kid) is not None:
        continue
    db.add(
        APIKey(
            id=kid,
            key_hash=f"demo-hash-{kname}",
            key_prefix="otari-",
            key_name=kname,
            user_id=owner,
            is_active=True,
            last_used_at=NOW - timedelta(hours=rng.randint(0, 40)),
        )
    )

# --- Pricing -----------------------------------------------------------------
# (model_key, effective_at) is the composite PK; anchor effective_at in the past.
pricing_effective = NOW - timedelta(days=45)
for model_key, (_provider, in_price, out_price, _lat) in MODELS.items():
    db.add(
        ModelPricing(
            model_key=model_key,
            effective_at=pricing_effective,
            input_price_per_million=in_price,
            output_price_per_million=out_price,
        )
    )

# --- Aliases -----------------------------------------------------------------
for name, target in ALIASES.items():
    if db.get(ModelAlias, name) is not None:
        continue
    db.add(ModelAlias(name=name, target=target))

db.flush()

# --- Usage logs --------------------------------------------------------------
# Weight callers toward a handful of "heavy" keys so the per-user/per-model
# breakdowns are not uniform. Each key prefers a couple of models.
key_names = list(KEYS.keys())
model_keys = list(MODELS.keys())
key_model_affinity = {k: rng.sample(model_keys, k=rng.randint(2, 4)) for k in key_names}
# A weighting so a few keys dominate volume (looks like real prod traffic).
key_weights = [6, 2, 4, 3, 5, 3, 1][: len(key_names)]

period_spend: dict[str, float] = {uid: 0.0 for uid in USERS}
rows = 0
for _ in range(4000):
    # Recency-weighted so 24h/7d presets are dense; tail reaches ~50 days so the
    # previous-30d comparison window is populated and deltas render.
    age_days = rng.random() ** 2 * 50
    ts = NOW - timedelta(days=age_days, hours=rng.random() * 24)
    kname = rng.choices(key_names, weights=key_weights, k=1)[0]
    owner = KEYS[kname]
    model_key = rng.choice(key_model_affinity[kname])
    provider, in_price, out_price, base_latency = MODELS[model_key]
    model = model_key.split(":", 1)[1]

    is_error = rng.random() < 0.035
    prompt = rng.randint(1500, 90000)
    completion = 0 if is_error else rng.randint(400, 16000)
    cache_read = rng.choice([0, 0, 0, rng.randint(1000, 40000)])
    cost = None if is_error else round(prompt / 1e6 * in_price + completion / 1e6 * out_price, 6)

    if cost and ts >= NOW - timedelta(seconds=MONTH):
        period_spend[owner] += cost

    db.add(
        UsageLog(
            id=str(uuid.uuid4()),
            user_id=owner,
            api_key_id=f"key-{kname}",
            timestamp=ts,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cache_read_tokens=cache_read or None,
            cache_write_tokens=None,
            cost=cost,
            status="error" if is_error else "success",
            error_message="provider rate limit exceeded" if is_error else None,
            latency_ms=None if is_error else base_latency + rng.randint(-150, 700),
        )
    )
    rows += 1

# Reflect each user's current-period spend on their row so the Users page shows
# spend-vs-budget utilization consistent with the Usage totals.
for uid, spent in period_spend.items():
    user = db.get(User, uid)
    if user is not None:
        user.spend = round(spent, 2)


# Derive each budget's dollar limit from the highest member's spend and the
# budget's target utilization, rounded up to a tidy figure. Keeps every user
# comfortably under budget with a varied, healthy set of utilization bars.
def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 25.0
    step = 5 if value < 30 else 10 if value < 100 else 25 if value < 300 else 50
    return float(math.ceil(value / step) * step)


for name, (util_target, _duration) in BUDGETS.items():
    members = [uid for uid, (_alias, bname) in USERS.items() if bname == name]
    peak = max((period_spend[uid] for uid in members), default=0.0)
    budgets[name].max_budget = _nice_ceiling(peak / util_target)

db.commit()

print(f"Seeded: {len(BUDGETS)} budgets, {len(USERS)} users, {len(KEYS)} keys, "
      f"{len(MODELS)} priced models, {len(ALIASES)} aliases, {rows} usage rows -> {URL}")
print("(providers are declared in scripts/demo_gif/otari.yml)")
for uid, spent in sorted(period_spend.items()):
    bname = USERS[uid][1]
    limit = budgets[bname].max_budget or 1.0
    print(f"  {uid:18s} ${spent:8.2f} / ${limit:7.0f}  ({spent / limit:4.0%})  {bname}")
