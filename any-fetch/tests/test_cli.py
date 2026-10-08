"""The any-fetch command."""

import json
import logging

import pytest

from any_fetch._logging import RedactProviderUrls
from any_fetch.cli import main
from any_fetch.providers.fake import CANNED_TEXT, CANNED_TITLE


def test_prints_the_canned_page(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "https://example.com/page"]) == 0
    out = capsys.readouterr().out
    assert CANNED_TITLE in out
    assert "https://example.com/page" in out
    assert CANNED_TEXT in out


def test_max_chars_cuts_the_text(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "https://example.com/", "--max-chars", "12"]) == 0
    out = capsys.readouterr().out
    assert CANNED_TEXT[:12] in out and CANNED_TEXT not in out
    assert "(truncated)" in out


def test_json_prints_the_page_without_raw(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "https://example.com/", "--json", "-o", "cost=0.001"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed["text"], printed["cost"], printed["cost_source"]) == (CANNED_TEXT, "0.001", "reported")
    assert "raw" not in printed


def test_raw_prints_the_provider_response(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "https://example.com/", "--raw", "-o", "account=acme"]) == 0
    assert json.loads(capsys.readouterr().out)["request"]["options"] == {"account": "acme"}


def test_json_and_raw_keep_raw(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "https://example.com/", "--json", "--raw"]) == 0
    assert "raw" in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("option", ["error=http_status", "in_body_error=crawl_not_found", "delay=abc"])
def test_an_error_exits_1_without_the_url(option: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "https://example.com/sentinel-url", "-o", option]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: fake")
    assert "sentinel-url" not in captured.err


def test_a_provider_bug_exits_1_without_the_url(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fake", "https://sentinel.example/", "-o", "leak_url=true"]) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "error: fake failed unexpectedly (RuntimeError)"
    assert "sentinel.example" not in captured.err


def test_builtin_without_a_host_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["builtin", "https://example.com/"]) == 1
    assert "register_builtin" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["nope", "https://example.com/"],
        ["fake", "https://example.com/", "--max-chars", "0"],
        ["fake", "https://example.com/", "-o", "max_chars=2"],
    ],
)
def test_a_usage_error_exits_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(argv)
    assert exited.value.code == 2


def test_a_count_that_is_not_a_number_names_the_problem(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["fake", "https://example.com/", "--max-chars", "twelve"])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "must be a whole number" in err
    assert "_positive" not in err


def _log_filter_installed() -> bool:
    return any(isinstance(existing, RedactProviderUrls) for existing in logging.getLogger("httpx").filters)


def test_the_command_installs_the_log_filter() -> None:
    assert not _log_filter_installed()
    assert main(["fake", "https://example.com/page"]) == 0
    assert _log_filter_installed()
