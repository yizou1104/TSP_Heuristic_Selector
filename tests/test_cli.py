"""CLI integration tests using the Click test runner."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tsp_heuristics.cli import predict_cli

ATT48 = str(Path(__file__).parent / "att48.tsp")

HEURISTICS = {"CH", "Greedy", "Insertion", "NN"}


@pytest.fixture
def runner():
    return CliRunner()


def test_default_output(runner):
    result = runner.invoke(predict_cli, [ATT48])
    assert result.exit_code == 0, result.output
    assert "Instance:" in result.output
    assert "Best heuristic:" in result.output
    assert "Predicted gap:" in result.output
    assert "%" in result.output
    assert "95% CI" in result.output


def test_json_output(runner):
    result = runner.invoke(predict_cli, [ATT48, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["instance"] == "att48"
    assert data["n_nodes"] == 48
    assert data["predicted_heuristic"] in HEURISTICS
    assert isinstance(data["predicted_gap_pct"], (int, float))
    assert data["gap_lower_95_pct"] is not None
    assert data["gap_upper_95_pct"] is not None


def test_no_intervals(runner):
    result = runner.invoke(predict_cli, [ATT48, "--no-intervals"])
    assert result.exit_code == 0, result.output
    assert "95% CI" not in result.output
    assert "Predicted gap:" in result.output


def test_model_xgb(runner):
    result = runner.invoke(predict_cli, [ATT48, "--model", "xgb"])
    assert result.exit_code == 0, result.output
    assert "XGBoost" in result.output


def test_model_lr(runner):
    result = runner.invoke(predict_cli, [ATT48, "--model", "lr"])
    assert result.exit_code == 0, result.output
    assert "Logistic Regression" in result.output
    # LR has no intervals
    assert "95% CI" not in result.output


def test_seed_2(runner):
    result = runner.invoke(predict_cli, [ATT48, "--seed", "2"])
    assert result.exit_code == 0, result.output
    assert "seed=2" in result.output


def test_xgb_json_no_intervals(runner):
    result = runner.invoke(predict_cli, [ATT48, "--model", "xgb", "--json", "--no-intervals"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["gap_lower_95_pct"] is None
    assert data["gap_upper_95_pct"] is None


def test_invalid_file(runner):
    result = runner.invoke(predict_cli, ["nonexistent.tsp"])
    assert result.exit_code != 0


def test_version(runner):
    result = runner.invoke(predict_cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_heuristic_is_valid(runner):
    for model in ("rf", "xgb", "lr"):
        result = runner.invoke(predict_cli, [ATT48, "--model", model, "--json"])
        assert result.exit_code == 0, f"model={model}: {result.output}"
        data = json.loads(result.output)
        assert data["predicted_heuristic"] in HEURISTICS, f"model={model}: {data}"
