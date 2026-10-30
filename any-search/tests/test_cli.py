"""The any-search command."""

import json
import logging

import pytest

from any_search import AnySearch
from any_search._logging import RedactProviderUrls
from any_search.cli import main
from any_search.providers.fake import CANNED_HITS


def test_prints_the_canned_hits(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "latest python"]) == 0
    out = capsys.readouterr().out
    for number, item in enumerate(CANNED_HITS, start=1):
        assert f"{number}. {item['title']}" in out
        assert item["url"] in out


def test_json_prints_the_envelope_without_raw(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "q", "--json", "--max-results", "1", "-o", "cost=0.007"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed["provider"], printed["cost"], printed["cost_source"]) == ("fake", "0.007", "reported")
    assert len(printed["hits"]) == 1
    assert "raw" not in printed and "raw" not in printed["hits"][0]


def test_raw_prints_the_provider_response(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "q", "--raw", "-o", "account=acme"]) == 0
    assert json.loads(capsys.readouterr().out)["request"]["options"] == {"account": "acme"}


def test_json_and_raw_keep_raw(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "q", "--json", "--raw"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert "raw" in printed and "raw" in printed["hits"][0]


@pytest.mark.parametrize("option", ["error=rate_limit", "in_body_error=engine_unavailable", "delay=abc"])
def test_an_error_exits_1_without_the_query(option: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "sentinel-query", "-o", option]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: fake")
    assert "sentinel-query" not in captured.err


def test_a_provider_bug_exits_1_without_the_query(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "sentinel-query", "-o", "leak_query=true"]) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "error: fake failed unexpectedly (RuntimeError)"
    assert "sentinel-query" not in captured.err


def test_an_unknown_option_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "q", "-o", "color=red"]) == 1
    assert "'color'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv", [["nope", "q"], ["fake", "q", "--max-results", "0"], ["fake", "q", "-o", "max_results=2"]]
)
def test_a_usage_error_exits_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(argv)
    assert exited.value.code == 2


def test_a_count_that_is_not_a_number_names_the_problem(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["fake", "q", "--max-results", "five"])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "must be a whole number" in err
    assert "_positive" not in err


def _log_filter_installed() -> bool:
    return any(isinstance(existing, RedactProviderUrls) for existing in logging.getLogger("httpx").filters)


def test_the_command_installs_the_log_filter() -> None:
    assert not _log_filter_installed()
    assert main(["fake", "query"]) == 0
    assert _log_filter_installed()


def _describe(provider: str, capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as exited:
        main(["--describe", provider])
    assert exited.value.code == 0
    return capsys.readouterr().out


@pytest.mark.parametrize("provider", AnySearch.get_supported_providers())
def test_describe_lists_every_option_without_a_query(provider: str, capsys: pytest.CaptureFixture[str]) -> None:
    out = _describe(provider, capsys)
    metadata = AnySearch.get_provider_metadata(provider)
    assert out.startswith(f"{provider}: {metadata.doc_url}\n")
    assert f"Results: at most {metadata.max_results} per call" in out
    assert "Shared parameters: --max-results, --time-range" in out
    for option in metadata.options:
        assert f"  {option.name} ({option.type}" in out


def test_describe_shows_defaults_as_json_and_the_allowed_values(capsys: pytest.CaptureFixture[str]) -> None:
    out = _describe("exa", capsys)
    assert "API key: required, from EXA_API_KEY" in out
    assert "API base: https://api.exa.ai" in out
    assert '  type (string), default "auto"\n      One of: instant, fast, auto,' in out
    assert '  contents (object), default {"text": {"maxCharacters": 10000}}' in out
    assert "  moderation (boolean), default false" in out


def test_describe_marks_operator_only_options_and_the_test_tier(capsys: pytest.CaptureFixture[str]) -> None:
    out = _describe("fake", capsys)
    assert "API key: none" in out
    assert "For tests:" in out
    assert "  account (string, operator only)" in out
    assert "Operator only: a host lets only its operator set it" in out


def test_describe_says_when_the_key_and_the_query_travel_in_the_url(capsys: pytest.CaptureFixture[str]) -> None:
    out = _describe("mojeek", capsys)
    assert "API key: required, from MOJEEK_API_KEY" in out
    assert "The key travels in the request URL.\nThe query travels in the request URL." in out
    assert "  date (boolean), default true" in out
    assert "The key travels" not in _describe("exa", capsys)


def test_describe_says_the_query_travels_in_the_url_without_the_key(capsys: pytest.CaptureFixture[str]) -> None:
    out = _describe("serply", capsys)
    assert "API key: required, from SERPLY_API_KEY" in out
    assert "The query travels in the request URL." in out
    assert "The key travels" not in out


def test_describe_refuses_an_unknown_provider(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["--describe", "nope"])
    assert exited.value.code == 2
    assert "invalid choice: 'nope'" in capsys.readouterr().err
