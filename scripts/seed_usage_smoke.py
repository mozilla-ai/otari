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
from sqlmodel import col

from gateway.models.entities import APIKey, ModelPricing, UsageLog, User
from gateway.models.tenancy import Organization, Workspace
from gateway.services.pricing_service import gateway_tool_pricing_key
from gateway.services.tool_usage import TOOL_METER_NAMESPACE

URL = sys.argv[1] if len(sys.argv) > 1 else "sqlite:///./smoke.db"
rng = random.Random(303)  # deterministic

MODELS = [
    ("gpt-5.6", "openai", 0.020, 900),
    ("claude-sonnet-5", "anthropic", 0.012, 1300),
    ("gemini-2.5-flash", "google", 0.002, 600),
    ("gpt-4o-mini", "openai", 0.001, 500),
]
USERS = ["alice", "bob", "carol", "dave"]

# Gateway-run tools, with the per-call rate the seeded rows are billed at. Priced so
# the dashboard shows a real charge; `otari:code_execution` is deliberately left
# unpriced so the "unpriced tool" treatment is visible too.
TOOL_UNIT_RATES = {"web_search": 0.01, "code_execution": None}
KEYS = [("key-prod", "alice"), ("key-staging", "bob"), ("key-batch", "carol")]

engine = create_engine(URL)
Session = sessionmaker(bind=engine)
db = Session()


def default_workspace_id() -> uuid.UUID:
    """The workspace a directly-built request-plane row belongs to.

    ``api_keys`` and ``usage_logs`` carry a NOT NULL ``workspace_id``, and the
    ORM sends an explicit NULL for a column it was given no value for, so the
    migration's ``server_default`` does not cover a writer like this one. The
    migration seeds this row; the fallback covers a database built some other
    way.
    """
    workspace = (
        db.query(Workspace)
        .join(Organization, col(Organization.id) == col(Workspace.organization_id))
        .filter(col(Organization.slug) == "default")
        .first()
    )
    if workspace is not None:
        return workspace.id

    organization = db.query(Organization).filter(col(Organization.slug) == "default").first()
    if organization is None:
        organization = Organization(name="Default organization", slug="default")
        db.add(organization)
        db.flush()
    workspace = Workspace(name="Default workspace", organization_id=organization.id)
    db.add(workspace)
    db.flush()
    return workspace.id


WORKSPACE_ID = default_workspace_id()

# Users + keys the logs reference (FKs are ON DELETE SET NULL, but present here).
for uid in USERS:
    if db.query(User).filter(User.user_id == uid).first() is None:
        db.add(User(user_id=uid, alias=uid.capitalize(), spend=0.0, blocked=False))
for kid, owner in KEYS:
    if db.query(APIKey).filter(APIKey.id == kid).first() is None:
        db.add(
            APIKey(
                id=kid,
                key_hash=f"hash-{kid}",
                key_name=kid,
                user_id=owner,
                is_active=True,
                workspace_id=WORKSPACE_ID,
            )
        )
db.flush()

# Price web search so its calls carry a cost (the stored convention is USD per
# million calls, so a cent per call is 10000).
if db.query(ModelPricing).filter(ModelPricing.model_key == gateway_tool_pricing_key("web_search")).first() is None:
    db.add(
        ModelPricing(
            model_key=gateway_tool_pricing_key("web_search"),
            # Narrowed for the type checker: this entry is priced by construction,
            # while code_execution is deliberately left unpriced (None) so the
            # dashboard's unpriced-tool treatment is visible in the seeded data.
            input_price_per_million=(TOOL_UNIT_RATES["web_search"] or 0.0) * 1_000_000,
            output_price_per_million=0.0,
            effective_at=datetime.now(UTC) - timedelta(days=60),
        )
    )
db.flush()

# Read the rate back rather than trusting TOOL_UNIT_RATES: the pricing row above is
# only created when absent, so an instance already priced differently (say through
# POST /v1/pricing) would otherwise get seeded rows whose unit_rate and cost
# contradict the pricing table the dashboard reads.
for tool in list(TOOL_UNIT_RATES):
    row = (
        db.query(ModelPricing)
        .filter(ModelPricing.model_key == gateway_tool_pricing_key(tool))
        .order_by(ModelPricing.effective_at.desc())
        .first()
    )
    TOOL_UNIT_RATES[tool] = (row.input_price_per_million / 1_000_000) if row else None
print(f"tool rates in effect: {TOOL_UNIT_RATES}")

now = datetime.now(UTC)
n = 0
tool_rows = 0
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

    # ~12% of requests run a gateway tool, the way a search-enabled deployment looks:
    # mostly web search, occasionally code execution, occasionally a failed call.
    tool_meters: dict[str, dict[str, float]] = {}
    tool_lines: list[dict[str, float | int | str]] = []
    tool_cost = 0.0
    if not is_error and rng.random() < 0.12:
        tool = "web_search" if rng.random() < 0.8 else "code_execution"
        billed = rng.randint(1, 4)
        errors = 1 if rng.random() < 0.15 else 0
        entry: dict[str, float] = {"billed": billed, "errors": errors}
        rate = TOOL_UNIT_RATES[tool]
        if rate is not None:
            entry["unit_rate"] = rate
            tool_cost = round(billed * rate, 6)
            tool_lines.append(
                {"meter": f"{tool}_calls", "units": billed, "unit_rate": rate, "cost": tool_cost}
            )
        tool_meters[tool] = entry
        tool_rows += 1

    token_cost = None if is_error else round((prompt + completion) / 1000 * unit_cost, 6)

    # Real priced rows carry the normalized token meters the pricing writer emits, and
    # the dashboard's token bar and the "needs pricing" predicate both read them, so
    # the seed writes them too rather than looking like a legacy unmetered row.
    token_meters: dict[str, float] = {}
    if token_cost is not None:
        token_meters = {
            "total_input_tokens": prompt,
            "fresh_input_tokens": prompt - cache_read,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": 0,
            "cache_write_1h_tokens": 0,
            "completion_tokens": completion,
        }
    db.add(
        UsageLog(
            id=str(uuid.uuid4()),
            workspace_id=WORKSPACE_ID,
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
            cost=token_cost if not tool_cost else round((token_cost or 0.0) + tool_cost, 6),
            billing_meters=(
                {**token_meters, **({TOOL_METER_NAMESPACE: tool_meters} if tool_meters else {})} or None
            ),
            pricing_breakdown=tool_lines or None,
            status="error" if is_error else "success",
            error_message="provider quota exceeded" if is_error else None,
            # A tool-loop request is slower: it made several provider round trips.
            latency_ms=None
            if is_error
            else base_latency + rng.randint(-200, 800) + (1800 if tool_meters else 0),
        )
    )
    n += 1

db.commit()
print(f"Seeded {n} usage rows into {URL} ({tool_rows} of them ran a gateway tool)")
