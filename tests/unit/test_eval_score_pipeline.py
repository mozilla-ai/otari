import json
import subprocess
import sys
from pathlib import Path

import pytest

from gateway.services.eval_score_pipeline import (
    EvalScorePipelineError,
    build_eval_scores_payload,
    load_eval_score_rows,
    normalize_eval_score_row,
)


def test_eval_score_pipeline_normalizes_common_eval_result_fields() -> None:
    payload = build_eval_scores_payload(
        [
            {
                "model_key": "openai:gpt-4o",
                "accuracy": "91%",
                "samples": "12",
                "task": "mt_bench",
                "run_id": "nightly-001",
            },
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "quality_score": "0.72",
                "n": "4",
            },
        ],
        default_metric="nightly_eval",
        change_note="import nightly eval artifact",
        source="evals/nightly.jsonl",
    )

    assert payload["change_note"] == "import nightly eval artifact"
    assert payload["scores"] == [
        {
            "model": "openai:gpt-4o",
            "score": 91.0,
            "metric": "mt_bench",
            "sample_count": 12,
            "metadata": {
                "run_id": "nightly-001",
                "source": "evals/nightly.jsonl",
                "row_number": 1,
            },
        },
        {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "quality_score": 0.72,
            "metric": "nightly_eval",
            "sample_count": 4,
            "metadata": {
                "source": "evals/nightly.jsonl",
                "row_number": 2,
            },
        },
    ]


def test_eval_score_pipeline_loads_jsonl_csv_and_nested_json(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "scores.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"model": "openai:gpt-4o", "score": 0.9}),
                json.dumps({"model": "openai:gpt-4o-mini", "benchmark_score": 72}),
            ]
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text("model,score,sample_count\nopenai:gpt-4o,88,5\n", encoding="utf-8")
    json_path = tmp_path / "scores.json"
    json_path.write_text(
        json.dumps({"results": [{"model": "anthropic:claude-3-5-haiku-latest", "score": 0.66}]}),
        encoding="utf-8",
    )

    assert len(load_eval_score_rows(jsonl_path)) == 2
    assert load_eval_score_rows(csv_path) == [{"model": "openai:gpt-4o", "score": "88", "sample_count": "5"}]
    assert load_eval_score_rows(json_path) == [{"model": "anthropic:claude-3-5-haiku-latest", "score": 0.66}]


def test_eval_score_pipeline_rejects_rows_without_score() -> None:
    with pytest.raises(EvalScorePipelineError, match="must include score"):
        normalize_eval_score_row({"model": "openai:gpt-4o"})


def test_apply_eval_scores_script_dry_run_outputs_endpoint_payload(tmp_path: Path) -> None:
    input_path = tmp_path / "scores.json"
    input_path.write_text(
        json.dumps({"scores": [{"provider": "openai", "model": "gpt-4o", "pass_rate": 0.93}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/apply_eval_scores.py",
            "--input",
            str(input_path),
            "--policy-id",
            "policy_123",
            "--metric",
            "nightly_eval",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["scores"][0]["provider"] == "openai"
    assert payload["scores"][0]["model"] == "gpt-4o"
    assert payload["scores"][0]["score"] == 0.93
    assert payload["scores"][0]["metric"] == "nightly_eval"
