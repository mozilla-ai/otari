"""Normalize local eval artifacts into routing-policy score updates."""

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_ROW_LIST_KEYS = ("scores", "results", "items", "rows", "evals")
_MODEL_KEYS = ("model", "model_key", "candidate_model")
_PROVIDER_KEYS = ("provider", "vendor")
_SCORE_KEYS = ("score", "eval_score", "accuracy", "pass_rate", "mean_score")
_QUALITY_SCORE_KEYS = ("quality_score",)
_BENCHMARK_SCORE_KEYS = ("benchmark_score",)
_METRIC_KEYS = ("metric", "eval", "benchmark", "task")
_SAMPLE_COUNT_KEYS = ("sample_count", "samples", "n", "count")
_KNOWN_KEYS = {
    *_ROW_LIST_KEYS,
    *_MODEL_KEYS,
    *_PROVIDER_KEYS,
    *_SCORE_KEYS,
    *_QUALITY_SCORE_KEYS,
    *_BENCHMARK_SCORE_KEYS,
    *_METRIC_KEYS,
    *_SAMPLE_COUNT_KEYS,
    "metadata",
}


class EvalScorePipelineError(ValueError):
    """Raised when an eval artifact cannot be converted into score rows."""


def _is_present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _first_string(row: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(value)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("%"):
        normalized = normalized[:-1].strip()
    try:
        return float(normalized)
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    parsed = _coerce_float(value)
    if parsed is None or parsed < 1:
        return None
    return int(parsed)


def _first_float(row: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        parsed = _coerce_float(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _first_int(row: Mapping[str, Any], keys: Iterable[str]) -> int | None:
    for key in keys:
        parsed = _coerce_int(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _row_metadata(row: Mapping[str, Any], *, source: str | None, row_number: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    existing = row.get("metadata")
    if isinstance(existing, dict):
        metadata.update(existing)
    for key, value in row.items():
        if key not in _KNOWN_KEYS and _is_present(value):
            metadata[key] = value
    if source is not None:
        metadata.setdefault("source", source)
    metadata.setdefault("row_number", row_number)
    return metadata


def normalize_eval_score_row(
    row: Mapping[str, Any],
    *,
    default_metric: str | None = None,
    source: str | None = None,
    row_number: int = 1,
) -> dict[str, Any]:
    """Normalize one eval artifact row into the routing-policy API shape."""
    model = _first_string(row, _MODEL_KEYS)
    if model is None:
        raise EvalScorePipelineError(f"Eval score row {row_number} is missing a model")

    quality_score = _first_float(row, _QUALITY_SCORE_KEYS)
    benchmark_score = _first_float(row, _BENCHMARK_SCORE_KEYS)
    score = _first_float(row, _SCORE_KEYS)
    if quality_score is None and benchmark_score is None and score is None:
        raise EvalScorePipelineError(
            f"Eval score row {row_number} must include score, quality_score, benchmark_score, "
            "eval_score, accuracy, pass_rate, or mean_score"
        )

    item: dict[str, Any] = {"model": model}
    provider = _first_string(row, _PROVIDER_KEYS)
    if provider is not None:
        item["provider"] = provider
    if quality_score is not None:
        item["quality_score"] = quality_score
    elif benchmark_score is not None:
        item["benchmark_score"] = benchmark_score
    else:
        item["score"] = score

    metric = _first_string(row, _METRIC_KEYS) or default_metric
    if metric is not None and metric.strip():
        item["metric"] = metric.strip()
    sample_count = _first_int(row, _SAMPLE_COUNT_KEYS)
    if sample_count is not None:
        item["sample_count"] = sample_count

    metadata = _row_metadata(row, source=source, row_number=row_number)
    if metadata:
        item["metadata"] = metadata
    return item


def _rows_from_json_payload(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise EvalScorePipelineError("JSON eval artifact list must contain objects")
        return payload
    if not isinstance(payload, dict):
        raise EvalScorePipelineError("JSON eval artifact must be an object or list")
    for key in _ROW_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            if not all(isinstance(item, dict) for item in value):
                raise EvalScorePipelineError(f"JSON eval artifact '{key}' list must contain objects")
            return value
    if any(key in payload for key in _MODEL_KEYS):
        return [payload]
    raise EvalScorePipelineError("JSON eval artifact must contain scores, results, items, rows, evals, or model")


def load_eval_score_rows(path: Path) -> list[Mapping[str, Any]]:
    """Load raw eval score rows from JSON, JSONL/NDJSON, or CSV."""
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[Mapping[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise EvalScorePipelineError(f"JSONL line {line_number} must be an object")
                rows.append(parsed)
        return rows

    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    with path.open(encoding="utf-8") as handle:
        return _rows_from_json_payload(json.load(handle))


def build_eval_scores_payload(
    rows: Iterable[Mapping[str, Any]],
    *,
    default_metric: str | None = None,
    change_note: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Build the `/eval-scores` request payload for a set of raw eval rows."""
    scores = [
        normalize_eval_score_row(
            row,
            default_metric=default_metric,
            source=source,
            row_number=index,
        )
        for index, row in enumerate(rows, start=1)
    ]
    if not scores:
        raise EvalScorePipelineError("Eval artifact did not contain any score rows")
    payload: dict[str, Any] = {"scores": scores}
    if change_note is not None and change_note.strip():
        payload["change_note"] = change_note.strip()
    return payload
