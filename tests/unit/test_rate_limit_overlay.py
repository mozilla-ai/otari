"""Stored rate-limit rules added to ``config.rate_limits``."""

from gateway.core.config import GatewayConfig, RateLimitRule
from gateway.services.rate_limits._overlay import apply_stored_rules, config_file_rules


def _rule(name: str, rpm: int = 1) -> RateLimitRule:
    return RateLimitRule(name=name, per="key", rpm=rpm)


def test_stored_rules_follow_the_config_rules() -> None:
    config = GatewayConfig(rate_limits=[_rule("file")])

    apply_stored_rules(config, [_rule("stored")])

    assert [rule.name for rule in config.rate_limits] == ["file", "stored"]
    assert [rule.name for rule in config_file_rules(config)] == ["file"]


def test_a_config_rule_wins_a_name_collision() -> None:
    config = GatewayConfig(rate_limits=[_rule("shared", rpm=5)])

    skipped = apply_stored_rules(config, [_rule("shared", rpm=1)])

    assert skipped == ["shared"]
    assert [(rule.name, rule.rpm) for rule in config.rate_limits] == [("shared", 5)]


def test_applying_again_drops_a_rule_no_longer_stored() -> None:
    config = GatewayConfig(rate_limits=[_rule("file")])
    apply_stored_rules(config, [_rule("stored")])

    apply_stored_rules(config, [])

    assert [rule.name for rule in config.rate_limits] == ["file"]
