"""The any-search command."""

import json
import logging

import pytest

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
