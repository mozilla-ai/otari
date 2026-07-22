"""Unit tests for the runtime-settings validation/serialization helpers."""

import asyncio
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from gateway.core.config import GatewayConfig
from gateway.models.entities import Base, RuntimeSetting
from gateway.services.pricing_service import configure_default_pricing, default_pricing_enabled
from gateway.services.runtime_settings_service import (
    DEFAULT_PRICING,
    MODEL_CACHE_TTL_SECONDS,
    MODEL_DISCOVERY_NEGATIVE_TTL_SECONDS,
    MODEL_DISCOVERY_TIMEOUT_SECONDS,
    REJECT_USER_MISMATCH,
    REQUIRE_PRICING,
    STREAM_MISSING_USAGE_POLICY,
    VISION_DESCRIBE_MODEL,
    VISION_STRATEGY,
    apply_override,
    load_overrides,
    validate_value,
)


def test_validate_value_normalizes_stream_policy() -> None:
    assert validate_value(STREAM_MISSING_USAGE_POLICY, "FAIL") == "fail"


def test_validate_value_accepts_float() -> None:
    assert validate_value(MODEL_DISCOVERY_NEGATIVE_TTL_SECONDS, 2.5) == 2.5
    # int coerces to float for a float field.
    assert validate_value(MODEL_DISCOVERY_NEGATIVE_TTL_SECONDS, 3) == 3.0


def test_validate_value_treats_blank_nullable_string_as_none() -> None:
    assert validate_value(VISION_DESCRIBE_MODEL, "") is None
    assert validate_value(VISION_DESCRIBE_MODEL, None) is None
    assert validate_value(VISION_DESCRIBE_MODEL, "ollama/qwen2-vl") == "ollama/qwen2-vl"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (STREAM_MISSING_USAGE_POLICY, "bogus"),
        (VISION_STRATEGY, "nope"),
        (MODEL_CACHE_TTL_SECONDS, -1),
        (MODEL_CACHE_TTL_SECONDS, True),  # bool is not a valid int here
        (MODEL_DISCOVERY_TIMEOUT_SECONDS, 0),  # gt=0, so zero is rejected
        (REQUIRE_PRICING, "yes"),  # not a bool
        ("host", "10.0.0.1"),  # not a settable key at all
    ],
)
def test_validate_value_rejects_bad_input(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=".+"):
        validate_value(key, value)  # type: ignore[arg-type]


def test_apply_override_sets_config_attribute() -> None:
    config = GatewayConfig(reject_user_mismatch=True, model_cache_ttl_seconds=300)
    apply_override(config, REJECT_USER_MISMATCH, False)
    apply_override(config, MODEL_CACHE_TTL_SECONDS, 42)
    assert config.reject_user_mismatch is False
    assert config.model_cache_ttl_seconds == 42


def test_load_overrides_skips_unknown_and_invalid_rows(tmp_path: Path) -> None:
    # A stale key (no longer settable) and a stored value that no longer
    # validates must be skipped so a hand-edited or migrated row cannot crash
    # startup; the still-valid override is returned, parsed to its typed value.
    async def _run() -> dict[str, object]:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'overrides.db'}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            session.add(RuntimeSetting(key=MODEL_CACHE_TTL_SECONDS, value="45"))  # valid
            session.add(RuntimeSetting(key="not_a_setting", value="true"))  # unknown key
            session.add(RuntimeSetting(key=MODEL_DISCOVERY_NEGATIVE_TTL_SECONDS, value="-1"))  # ge=0 violated
            await session.commit()
        async with factory() as session:
            overrides = await load_overrides(session)
        await engine.dispose()
        return dict(overrides)

    overrides = asyncio.run(_run())
    assert overrides == {MODEL_CACHE_TTL_SECONDS: 45}


def test_apply_override_default_pricing_updates_process_flag() -> None:
    configure_default_pricing(False)
    config = GatewayConfig(default_pricing=False)
    try:
        apply_override(config, DEFAULT_PRICING, True)
        assert config.default_pricing is True
        assert default_pricing_enabled() is True
    finally:
        configure_default_pricing(False)
