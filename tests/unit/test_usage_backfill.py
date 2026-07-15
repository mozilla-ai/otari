import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from gateway.core.config import GatewayConfig
from gateway.main import create_app
from gateway.models.entities import ModelPricing, UsageLog, User

BACKFILL_PATH = "/v1/usage/backfill"
MASTER = {"Authorization": "Bearer sk-test-master"}


def _app(tmp_path: Path) -> tuple[TestClient, sessionmaker[Session]]:
    db_path = tmp_path / "backfill-test.db"
    config = GatewayConfig(database_url=f"sqlite:///{db_path}", master_key="sk-test-master")
    client = TestClient(create_app(config))
    # A separate sync session on the same sqlite file for seeding/inspecting rows.
    factory = sessionmaker(bind=create_engine(f"sqlite:///{db_path}"))
    return client, factory


def _add_log(
    db: Session,
    *,
    user_id: str,
    cost: float | None,
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 5,
    model: str = "gpt-4",
    provider: str | None = "openai",
) -> None:
    db.add(
        UsageLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime(2025, 5, 1, 12, 0, tzinfo=UTC),
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
            cost=cost,
            status="success",
        )
    )


def test_backfill_requires_master_key(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    with client:
        assert client.post(BACKFILL_PATH, json={"model_key": "openai:gpt-4"}).status_code == 401


def test_backfill_unpriced_model_returns_404(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    with client:
        response = client.post(BACKFILL_PATH, headers=MASTER, json={"model_key": "openai:never-priced"})
    assert response.status_code == 404


def test_backfill_recomputes_empty_costs_and_spend(tmp_path: Path) -> None:
    client, factory = _app(tmp_path)
    with client:
        seed = factory()
        seed.add(User(user_id="bf-user", alias="bf-user", spend=0.0, blocked=False))
        seed.add(
            ModelPricing(
                model_key="openai:gpt-4",
                effective_at=datetime(2025, 1, 1, tzinfo=UTC),
                input_price_per_million=1.0,
                output_price_per_million=2.0,
            )
        )
        # Two empty rows to backfill, plus one already-costed row that must be left alone.
        _add_log(seed, user_id="bf-user", cost=None, prompt_tokens=10, completion_tokens=5)
        _add_log(seed, user_id="bf-user", cost=None, prompt_tokens=20, completion_tokens=0)
        _add_log(seed, user_id="bf-user", cost=0.5, prompt_tokens=1, completion_tokens=1)
        seed.commit()
        seed.close()

        response = client.post(BACKFILL_PATH, headers=MASTER, json={"model_key": "openai:gpt-4"})
        assert response.status_code == 200
        body = response.json()
        # row1: 10/1e6*1 + 5/1e6*2 = 2e-5; row2: 20/1e6*1 = 2e-5; total 4e-5.
        assert body["rows_updated"] == 2
        assert body["users_updated"] == 1
        assert abs(body["cost_added"] - 4e-5) < 1e-12

        check = factory()
        try:
            null_rows = check.query(UsageLog).filter(UsageLog.user_id == "bf-user", UsageLog.cost.is_(None)).count()
            assert null_rows == 0
            user = check.query(User).filter(User.user_id == "bf-user").one()
            assert abs(float(user.spend) - 4e-5) < 1e-12
        finally:
            check.close()

        # Re-running is a no-op now that the rows carry costs.
        second = client.post(BACKFILL_PATH, headers=MASTER, json={"model_key": "openai:gpt-4"})
        assert second.status_code == 200
        assert second.json()["rows_updated"] == 0
