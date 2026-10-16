"""Seed a realistic-looking standalone gateway for the README dashboard GIF.

Populates an organization with members (each linked to the gateway user their
keys and budget attach to), budgets, named API keys, model pricing, and ~26k
usage_logs over ~60 days, part of them routed through the policies declared in
otari.yml, so Overview / Usage / Activity / Members / Budgets all render with
live-looking data.

Values come from a seeded RNG and timestamps are anchored to the current wall
clock, so the demo always looks fresh whenever the GIF is regenerated. A run's
shape (growth, mix, budget utilization) is reproducible; its exact figures are
not, because how many of today's rows fall in the future, and so every draw after
them, depends on the time of day the seed runs.

Usage:
    uv run otari migrate --config scripts/demo_gif/otari.yml
    OTARI_SECRET_KEY=<fernet> uv run python scripts/demo_gif/seed.py sqlite:///./scripts/demo_gif/demo.db

See scripts/demo_gif/record.sh for the full pipeline.
"""

import math
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import col

from gateway.models.api_keys import APIKey
from gateway.models.budgets import Budget
from gateway.models.money import to_usd
from gateway.models.pricing import ModelPricing
from gateway.models.tenancy import Organization, OrganizationMember, Workspace, WorkspaceMember
from gateway.models.tenancy import User as Identity
from gateway.models.usage import UsageLog
from gateway.models.users import User
from gateway.services.tenancy.provisioning_service import DEFAULT_WORKSPACE_NAME

URL = sys.argv[1] if len(sys.argv) > 1 else "sqlite:///./scripts/demo_gif/demo.db"
rng = random.Random(4242)  # deterministic values across runs

NOW = datetime.now(UTC)
MONTH = 30 * 24 * 3600
DAYS = 60
# The newest minutes of traffic are the hand-written RECENT rows, not random ones.
RECENT_WINDOW_MIN = 10

ORGANIZATION_NAME = "Acme AI"

# The providers themselves are declared in scripts/demo_gif/otari.yml (config
# providers with a `models:` list, so the gateway reports them healthy without a
# live upstream). model_key here is `<provider instance>:<model>` and MUST match
# the instances/models declared there. Ids are real models.dev ids, so the
# Models page shows their vendor, modalities, and context length.
#
# model_key -> (input $/M, output $/M, base latency ms)
MODELS = {
    "openai:gpt-6.1-sol": (2.00, 10.00, 900),
    "openai:gpt-6-luna": (0.10, 0.50, 380),
    "openai:gpt-6-astra": (10.00, 50.00, 2100),
    "anthropic:claude-opus-5-5": (4.00, 20.00, 1500),
    "anthropic:claude-sonnet-5-5": (2.00, 10.00, 950),
    "google:gemini-3.8-flash": (0.75, 3.75, 520),
    "google:gemini-3.5-flash-lite": (0.30, 2.50, 340),
    "groq:openai/gpt-oss-120b": (0.15, 0.60, 210),
    "groq:qwen/qwen3.8-27b": (0.80, 4.00, 260),
    "mistral:mistral-medium-latest": (1.50, 7.50, 700),
    "mistral:devstral-2512": (0.40, 2.00, 640),
    "deepseek:deepseek-v4-pro": (0.435, 0.87, 1100),
    "deepseek:deepseek-v4-flash": (0.15, 0.60, 480),
    "xai:grok-4.7": (2.00, 6.00, 820),
}

# Mirrors the `routing:` block in otari.yml: the model a policy starts on (with
# weights for a weighted router) and what it falls back to.
POLICIES = {
    "smart": ({"openai:gpt-6.1-sol": 60, "anthropic:claude-sonnet-5-5": 40}, "google:gemini-3.8-flash"),
    "thrifty": ({"openai:gpt-6-luna": 1}, "groq:openai/gpt-oss-120b"),
    "coding": ({"anthropic:claude-opus-5-5": 1}, "openai:gpt-6.1-sol"),
    "fast": ({"groq:openai/gpt-oss-120b": 1}, "google:gemini-3.5-flash-lite"),
}

# budget name -> target utilization. The dollar limit is derived from actual
# seeded spend so the utilization bars land near these targets regardless of
# the RNG draw.
BUDGETS = {
    "Engineering": 0.58,
    "Research": 0.71,
    "Data Science": 0.46,
    "Interns": 0.88,
}

# email -> (full name, organization role, budget name)
MEMBERS = {
    "alice@acme.ai": ("Alice Nguyen", "admin", "Engineering"),
    "bob@acme.ai": ("Bob Martins", "member", "Engineering"),
    "hiro@acme.ai": ("Hiro Tanaka", "member", "Engineering"),
    "carol@acme.ai": ("Carol Diaz", "member", "Research"),
    "grace@acme.ai": ("Grace Park", "member", "Research"),
    "dave@acme.ai": ("Dave Okafor", "admin", "Data Science"),
    "erin@acme.ai": ("Erin Cole", "member", "Data Science"),
    "frank@acme.ai": ("Frank Li", "member", "Interns"),
}

# key name -> (owner email, traffic weight, policy or None, models used directly)
KEYS = {
    "prod-api": ("alice@acme.ai", 9, "smart", ["openai:gpt-6.1-sol", "anthropic:claude-sonnet-5-5"]),
    "support-bot": ("erin@acme.ai", 7, "thrifty", ["openai:gpt-6-luna", "deepseek:deepseek-v4-flash"]),
    "coding-agent": ("hiro@acme.ai", 5, "coding", ["anthropic:claude-opus-5-5", "mistral:devstral-2512"]),
    "ci-pipeline": ("bob@acme.ai", 4, "fast", ["groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b"]),
    "research-notebook": ("carol@acme.ai", 3, None, ["openai:gpt-6-astra", "deepseek:deepseek-v4-pro", "xai:grok-4.7"]),
    "eval-harness": (
        "grace@acme.ai",
        3,
        None,
        ["google:gemini-3.8-flash", "mistral:mistral-medium-latest", "xai:grok-4.7"],
    ),
    "etl-batch": ("dave@acme.ai", 4, None, ["google:gemini-3.5-flash-lite", "deepseek:deepseek-v4-flash"]),
    "frank-cli": ("frank@acme.ai", 1, None, ["openai:gpt-6-luna", "google:gemini-3.8-flash"]),
}

engine = create_engine(URL)
Session = sessionmaker(bind=engine)
db = Session()


def default_tenancy() -> tuple[Organization, Workspace]:
    """The default organization (renamed for the demo) and its default workspace.

    The workspace keeps its default name: the gateway finds its default
    workspace by that name at boot, so renaming it would leave this data in a
    workspace the operator never opens. ``api_keys``, ``usage_logs`` and the
    membership rows carry a NOT NULL workspace or organization id, and the ORM
    sends an explicit NULL for a column it was given no value for, so the
    migration's ``server_default`` does not cover a writer like this one.
    """
    organization = db.query(Organization).filter(col(Organization.slug) == "default").one()
    organization.name = ORGANIZATION_NAME
    workspace = (
        db.query(Workspace)
        .filter(col(Workspace.organization_id) == organization.id, col(Workspace.name) == DEFAULT_WORKSPACE_NAME)
        .first()
    )
    if workspace is None:
        workspace = Workspace(name=DEFAULT_WORKSPACE_NAME, organization_id=organization.id)
        db.add(workspace)
    db.flush()
    return organization, workspace


ORGANIZATION, WORKSPACE = default_tenancy()

# --- Budgets -----------------------------------------------------------------
# Created with a placeholder limit; the real max_budget is derived from seeded
# spend once the usage rows exist, at the end of this file.
budgets: dict[str, Budget] = {}
for name in BUDGETS:
    budgets[name] = Budget(budget_id=str(uuid.uuid4()), name=name, max_budget=0.0, budget_duration_sec=MONTH)
    db.add(budgets[name])

# --- Members -----------------------------------------------------------------
# Each member is an identity in the organization and its workspace, plus the
# gateway ``users`` row keyed by that identity's id, which is what keys, budgets,
# and usage attach to (the Members page joins the two on it).
gateway_user_id: dict[str, str] = {}
gateway_users: dict[str, User] = {}
budget_started: dict[str, datetime] = {}
for email, (full_name, role, budget_name) in MEMBERS.items():
    identity = Identity(email=email, full_name=full_name, active_organization_id=ORGANIZATION.id)
    db.add(identity)
    db.flush()
    db.add(OrganizationMember(organization_id=ORGANIZATION.id, user_id=identity.id, role=role))
    db.add(WorkspaceMember(workspace_id=WORKSPACE.id, user_id=identity.id, role=role))
    uid = str(identity.id)
    gateway_user_id[email] = uid
    started = NOW - timedelta(days=rng.randint(12, 22), hours=rng.random() * 24)
    budget_started[uid] = started
    gateway_users[uid] = User(
        user_id=uid,
        alias=full_name,
        spend=0.0,  # set from generated usage below
        reserved=0.0,
        blocked=False,
        budget_id=budgets[budget_name].budget_id,
        budget_started_at=started,
        next_budget_reset_at=started + timedelta(seconds=MONTH),
    )
    db.add(gateway_users[uid])

# --- API keys ----------------------------------------------------------------
api_keys: dict[str, APIKey] = {}
for kname, (owner, *_rest) in KEYS.items():
    api_keys[kname] = APIKey(
        id=f"key-{kname}",
        workspace_id=WORKSPACE.id,
        key_hash=f"demo-hash-{kname}",
        key_prefix="otari-",
        key_name=kname,
        user_id=gateway_user_id[owner],
        is_active=True,
        created_at=NOW - timedelta(days=rng.randint(40, 70)),
    )
    db.add(api_keys[kname])

# --- Pricing -----------------------------------------------------------------
# (model_key, effective_at) is the composite PK; anchor effective_at in the past.
pricing_effective = NOW - timedelta(days=DAYS + 15)
for model_key, (in_price, out_price, _lat) in MODELS.items():
    db.add(
        ModelPricing(
            model_key=model_key,
            effective_at=pricing_effective,
            input_price_per_million=in_price,
            output_price_per_million=out_price,
        )
    )

db.flush()

# --- Usage logs --------------------------------------------------------------
key_names = list(KEYS)
key_weights = [KEYS[k][1] for k in key_names]
period_spend: dict[str, float] = dict.fromkeys(gateway_user_id.values(), 0.0)
last_used: dict[str, datetime] = {}
rows = 0


def add_row(
    *, ts: datetime, kname: str, model_key: str, ok: bool, status_code: int | None = None, **routing: str | int
) -> None:
    global rows
    owner = gateway_user_id[KEYS[kname][0]]
    provider, model = model_key.split(":", 1)
    in_price, out_price, base_latency = MODELS[model_key]
    prompt = int(rng.lognormvariate(10.4, 0.8))  # median ~33k tokens (agentic contexts)
    completion = int(rng.lognormvariate(7.5, 0.7)) if ok else 0
    cache_read = int(prompt * rng.uniform(0.3, 0.8)) if ok and rng.random() < 0.35 else 0
    cost = round(prompt / 1e6 * in_price + completion / 1e6 * out_price, 6) if ok else None
    latency = base_latency + int(completion * rng.uniform(0.6, 1.4)) if ok else rng.randint(80, 900)
    if cost and ts >= budget_started[owner]:
        period_spend[owner] += cost
    last_used[kname] = max(last_used.get(kname, ts), ts)
    db.add(
        UsageLog(
            id=str(uuid.uuid4()),
            workspace_id=WORKSPACE.id,
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
            cost=cost,
            status="success" if ok else "error",
            status_code=status_code,
            error_message=None
            if ok
            else ("provider rate limit exceeded" if status_code == 429 else "upstream unavailable"),
            latency_ms=latency,
            ttft_ms=int(base_latency * rng.uniform(0.25, 0.5)) if ok else None,
            **routing,
        )
    )
    rows += 1


def hour_weight(hour: int) -> float:
    """Diurnal traffic: a working-day hump peaking mid-afternoon UTC."""
    return 0.35 + math.exp(-(((hour - 15) / 4.5) ** 2))


hour_weights = [hour_weight(h) for h in range(24)]
for day in range(DAYS, -1, -1):
    start = (NOW - timedelta(days=day)).replace(hour=0, minute=0, second=0, microsecond=0)
    # Steady adoption: volume roughly doubles over the window, with quieter
    # weekends and a little day-to-day noise.
    growth = 1 + 1.2 * (DAYS - day) / DAYS
    weekend = 0.55 if start.weekday() >= 5 else 1.0
    count = int(300 * growth * weekend * rng.uniform(0.85, 1.15))
    for _ in range(count):
        hour = rng.choices(range(24), weights=hour_weights, k=1)[0]
        ts = start + timedelta(hours=hour, seconds=rng.random() * 3600)
        if ts > NOW - timedelta(minutes=RECENT_WINDOW_MIN):
            continue
        kname = rng.choices(key_names, weights=key_weights, k=1)[0]
        _owner, _w, policy, direct = KEYS[kname]
        if policy is None or rng.random() < 0.25:
            failed = rng.random() < 0.006
            add_row(
                ts=ts,
                kname=kname,
                model_key=rng.choice(direct),
                ok=not failed,
                status_code=rng.choice([429, 503]) if failed else 200,
            )
            continue
        starts, fallback = POLICIES[policy]
        first = rng.choices(list(starts), weights=list(starts.values()), k=1)[0]
        reason = "router:weighted" if len(starts) > 1 else "default"
        group = str(uuid.uuid4())
        if rng.random() < 0.015:
            # The first provider failed and the policy's fallback served it.
            common: dict[str, str | int] = {"policy_name": policy, "request_group_id": group, "attempt_count": 2}
            add_row(
                ts=ts,
                kname=kname,
                model_key=first,
                ok=False,
                status_code=503,
                selection_reason=reason,
                attempt_position=1,
                **common,
            )
            add_row(
                ts=ts + timedelta(milliseconds=rng.randint(200, 900)),
                kname=kname,
                model_key=fallback,
                ok=True,
                status_code=200,
                selection_reason="on_failure",
                attempt_position=2,
                **common,
            )
        else:
            add_row(
                ts=ts,
                kname=kname,
                model_key=first,
                ok=True,
                status_code=200,
                policy_name=policy,
                selection_reason=reason,
                attempt_position=1,
                attempt_count=1,
                request_group_id=group,
            )

# The newest rows are written out rather than drawn, so the top of Activity
# (which the tour opens) shows each policy at work, and one request a fallback
# rescued.
RECENT = [
    # (minutes ago, key, model_key, policy, selection reason)
    (0.4, "prod-api", "openai:gpt-6.1-sol", "smart", "router:weighted"),
    (0.9, "coding-agent", "anthropic:claude-opus-5-5", "coding", "default"),
    (1.6, "support-bot", "openai:gpt-6-luna", "thrifty", "default"),
    (2.3, "research-notebook", "openai:gpt-6-astra", None, None),
    (3.1, "ci-pipeline", "groq:openai/gpt-oss-120b", "fast", "default"),
    (4.4, "eval-harness", "mistral:mistral-medium-latest", None, None),
    (5.2, "prod-api", "anthropic:claude-sonnet-5-5", "smart", "router:weighted"),
    (6.7, "etl-batch", "google:gemini-3.5-flash-lite", None, None),
    (7.9, "coding-agent", "anthropic:claude-opus-5-5", "coding", "default"),
    (9.5, "support-bot", "openai:gpt-6-luna", "thrifty", "default"),
]
for minutes, kname, model_key, recent_policy, recent_reason in RECENT:
    ts = NOW - timedelta(minutes=minutes)
    routing: dict[str, str | int] = {}
    if recent_policy is not None and recent_reason is not None:
        routing = {
            "policy_name": recent_policy,
            "selection_reason": recent_reason,
            "attempt_position": 1,
            "attempt_count": 1,
            "request_group_id": str(uuid.uuid4()),
        }
    add_row(ts=ts, kname=kname, model_key=model_key, ok=True, status_code=200, **routing)
# OpenAI answered 503 and the policy's fallback served the request.
rescued: dict[str, str | int] = {"policy_name": "smart", "request_group_id": str(uuid.uuid4()), "attempt_count": 2}
rescued_at = NOW - timedelta(minutes=1.2)
add_row(
    ts=rescued_at,
    kname="prod-api",
    model_key="openai:gpt-6.1-sol",
    ok=False,
    status_code=503,
    selection_reason="router:weighted",
    attempt_position=1,
    **rescued,
)
add_row(
    ts=rescued_at + timedelta(milliseconds=640),
    kname="prod-api",
    model_key="google:gemini-3.8-flash",
    ok=True,
    status_code=200,
    selection_reason="on_failure",
    attempt_position=2,
    **rescued,
)

for kname, ts in last_used.items():
    api_keys[kname].last_used_at = ts

# Reflect each member's current-period spend on their row so the Members and
# Budgets pages show spend-vs-budget utilization consistent with the usage rows.
for uid, spent in period_spend.items():
    gateway_users[uid].spend = to_usd(round(spent, 2))


# Derive each budget's dollar limit from the highest member's spend and the
# budget's target utilization, rounded up to a tidy figure. Keeps every member
# under budget with a varied, healthy set of utilization bars.
def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 25.0
    step = 5 if value < 30 else 10 if value < 100 else 25 if value < 300 else 50
    return float(math.ceil(value / step) * step)


for name, util_target in BUDGETS.items():
    members = [gateway_user_id[e] for e, (_n, _r, bname) in MEMBERS.items() if bname == name]
    peak = max(period_spend[uid] for uid in members)
    budgets[name].max_budget = to_usd(_nice_ceiling(peak / util_target))

db.commit()

print(
    f"Seeded: {len(MEMBERS)} members, {len(BUDGETS)} budgets, {len(KEYS)} keys, "
    f"{len(MODELS)} priced models, {rows} usage rows -> {URL}"
)
print("(providers and routing policies are declared in scripts/demo_gif/otari.yml)")
for email, (_name, _role, bname) in MEMBERS.items():
    spent = period_spend[gateway_user_id[email]]
    limit = float(budgets[bname].max_budget or 1.0)
    print(f"  {email:16s} ${spent:8.2f} / ${limit:7.0f}  ({spent / limit:4.0%})  {bname}")
