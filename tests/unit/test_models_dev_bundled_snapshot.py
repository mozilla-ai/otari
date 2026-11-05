"""The packaged models.dev snapshot."""

import http.client
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from gateway.services.pricing import (
    ModelsDevPriceIndex,
    active_generations,
    bundled_catalog,
    bundled_generation,
    load_bundled_index,
    reset_generations,
    set_accepted_generations,
)
from gateway.types.models_dev import CATALOG_FORMAT, read_catalog_document, snapshot_document


def _rates(input_: float, output: float, cache_read: float, cache_write: float) -> dict[str, Any]:
    return {"input": input_, "output": output, "cache_read": cache_read, "cache_write": cache_write}


# Anthropic's first-party rates per million tokens, as published at
# https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-16).
# The cache-read rate is not one ratio of input: Opus 5.5 and Sonnet 5.5 read at
# 0.05x and Fable 5.1 at 0.025x. The snapshot's cache_write is the 5m rate; it
# carries no 1h rate.
ANTHROPIC_PUBLISHED_COSTS: dict[str, dict[str, Any]] = {
    "claude-fable-5-1": _rates(10, 50, 0.25, 12.5),
    "claude-fable-5": _rates(10, 50, 1, 12.5),
    "claude-opus-5-5": _rates(4, 20, 0.2, 5),
    "claude-opus-5": _rates(5, 25, 0.5, 6.25),
    "claude-opus-4-8": _rates(5, 25, 0.5, 6.25),
    "claude-opus-4-7": _rates(5, 25, 0.5, 6.25),
    "claude-opus-4-6": _rates(5, 25, 0.5, 6.25),
    "claude-opus-4-5": _rates(5, 25, 0.5, 6.25),
    "claude-opus-4-5-20251101": _rates(5, 25, 0.5, 6.25),
    "claude-sonnet-5-5": _rates(2, 10, 0.1, 2.5),
    "claude-sonnet-5": _rates(2, 10, 0.2, 2.5),
    "claude-sonnet-4-6": _rates(3, 15, 0.3, 3.75),
    "claude-sonnet-4-5": _rates(3, 15, 0.3, 3.75),
    "claude-sonnet-4-5-20250929": _rates(3, 15, 0.3, 3.75),
    "claude-haiku-5-5": {
        **_rates(0.1, 0.5, 0.01, 0.125),
        "tiers": [{**_rates(0.5, 2.5, 0.05, 0.625), "tier": {"type": "context", "size": 100_000}}],
    },
    "claude-haiku-4-5": _rates(1, 5, 0.1, 1.25),
    "claude-haiku-4-5-20251001": _rates(1, 5, 0.1, 1.25),
}


def _bundled_text() -> str:
    return resources.files("gateway").joinpath("data", "models_dev_catalog.json").read_text("utf-8")


@cache
def _bundled_prices() -> dict[str, Any]:
    """The bundled catalog as the price snapshot a refresh would store."""
    return snapshot_document(bundled_catalog())


@pytest.mark.parametrize("model", ANTHROPIC_PUBLISHED_COSTS)
def test_the_bundled_snapshot_carries_anthropics_published_rates(model: str) -> None:
    assert _bundled_prices()["anthropic"]["models"][model]["cost"] == ANTHROPIC_PUBLISHED_COSTS[model]


def test_the_bundled_snapshot_prices_well_known_models() -> None:
    index = load_bundled_index()

    assert len(index) > 1000
    entry = index.resolve("openai", "gpt-4o")
    assert entry is not None and entry.priced and entry.context_window
    assert bundled_generation().index is index


def test_the_bundled_snapshot_is_the_whole_catalog_and_stays_bounded() -> None:
    raw = _bundled_text()
    # About 7.5 MB on 2026-11-02, for 9,311 models (0.7 MB gzipped, as in the wheel).
    assert len(raw.encode()) < 10_000_000
    document = json.loads(raw)
    assert document["_meta"]["format"] == CATALOG_FORMAT
    assert document["_meta"]["generated_at"]
    model = bundled_catalog().model_for("anthropic", "claude-sonnet-4-5")
    assert model is not None
    assert model.description and "image" in model.input_modalities and model.tool_call and model.max_output_tokens


def test_the_bundled_prices_are_those_a_stored_price_snapshot_of_it_gives() -> None:
    stored = ModelsDevPriceIndex.from_document(json.loads(json.dumps(_bundled_prices())))

    # A price snapshot is the v1 trim, which carries no cost block and so no slot read
    # out of one; the text prices are what it keeps.
    bundled = [replace(entry, pricing=None, slot_columns=None) for entry in load_bundled_index().entries()]
    assert sorted(bundled, key=lambda entry: entry.reference) == sorted(
        stored.entries(), key=lambda entry: entry.reference
    )


def test_the_generation_list_falls_back_to_the_bundled_snapshot() -> None:
    reset_generations()
    assert active_generations() == (bundled_generation(),)

    assert bundled_generation().baseline
    generation = replace(bundled_generation(), baseline=False)
    set_accepted_generations([generation])
    baseline, accepted = active_generations()
    assert baseline.baseline
    assert baseline.effective_at < accepted.effective_at
    assert accepted == generation
    reset_generations()


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "update_models_dev_snapshot.py"


def _cut(document: object, tmp_path: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "api.json"
    source.write_text(json.dumps(document), "utf-8")
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--input", str(source), "--output", str(tmp_path / "out.json"), *flags],
        capture_output=True,
        text=True,
        check=False,
    )


def _snapshot(tmp_path: Path) -> dict[str, Any]:
    """The cut's ``_meta``, and each model's price fields as a stored price snapshot keeps them."""
    document: dict[str, Any] = json.loads((tmp_path / "out.json").read_text("utf-8"))
    return {"_meta": document["_meta"], **snapshot_document(read_catalog_document(document))}


def test_the_release_script_cuts_a_snapshot_from_v2_and_names_a_damaged_model(tmp_path: Path) -> None:
    def cost(price: object) -> dict[str, Any]:
        return {"text": {"input": {"price": price}, "output": {"price": 2}}}

    models = {f"m{n}": {"cost": cost(1)} for n in range(4)} | {"bad": {"cost": cost("1")}}
    result = _cut({"p": {"models": models}}, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "models.dev v2 catalog read with" in result.stderr
    assert "p.models.bad.cost.text.input.price: expected a number" in result.stderr
    snapshot = _snapshot(tmp_path)
    assert snapshot["_meta"]["source"] == "https://models.dev/experimental/v2.0/api.json?type=all"
    assert set(snapshot["p"]["models"]) == {*models}
    assert snapshot["p"]["models"]["m0"]["cost"] == {"input": 1, "output": 2}


def test_the_release_script_reads_a_saved_v1_document(tmp_path: Path) -> None:
    result = _cut({"p": {"models": {"m": {"cost": {"input": 1, "output": 2}}}}}, tmp_path, "--input-version", "v1")

    assert result.returncode == 0, result.stderr
    snapshot = _snapshot(tmp_path)
    assert snapshot["_meta"]["source"] == "https://models.dev/api.json"
    assert snapshot["p"]["models"]["m"]["cost"] == {"input": 1, "output": 2}


def test_a_cut_off_download_is_a_failure_the_release_script_moves_past(capsys: pytest.CaptureFixture[str]) -> None:
    """A truncated body raises ``IncompleteRead``, which is no ``OSError``; the script must still try v1."""
    spec = importlib.util.spec_from_file_location("update_models_dev_snapshot", SCRIPT)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    v2, _v1 = script._load_document_module().MODELS_DEV_SOURCES

    def cut_off() -> object:
        raise http.client.IncompleteRead(b"{")

    assert script._read(v2, cut_off) is None
    assert "models.dev v2 catalog fetch failed: IncompleteRead" in capsys.readouterr().err


def test_the_release_script_refuses_a_document_it_cannot_read(tmp_path: Path) -> None:
    result = _cut({"p": {"models": ["m"]}}, tmp_path)

    assert result.returncode == 1
    assert "models.dev v2 catalog refused: the document lists no models" in result.stderr
    assert not (tmp_path / "out.json").exists()
