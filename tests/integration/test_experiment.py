"""The experiment grid, aggregation, and saved artefacts."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import pytest

from mars_rover_q.experiment import (
    ExperimentConfig,
    aggregate_rows,
    json_safe,
    run_experiment,
)


@pytest.fixture
def tiny_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="pytest_smoke",
        scenarios=("safe_corridor",),
        reward_modes=("sparse", "potential"),
        seeds=(1, 2),
        episodes=2,
        eval_episodes=2,
        threshold_window=2,
    )


def test_default_config_covers_the_declared_grid() -> None:
    config = ExperimentConfig()
    assert len(config.scenarios) == 3
    assert set(config.reward_modes) == {"sparse", "naive_dense", "potential"}
    assert len(config.seeds) >= 5
    assert config.total_runs == 45


def test_bundled_configs_load() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    for path in sorted(root.glob("*.json")):
        config = ExperimentConfig.from_file(path)
        assert config.total_runs > 0


def test_ten_seed_config_is_available() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    config = ExperimentConfig.from_file(root / "reward_comparison_10seeds.json")
    assert len(config.seeds) == 10


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"episodes": 5, "learning_rat": 0.2}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown experiment config keys"):
        ExperimentConfig.from_file(path)


def test_aggregate_reports_mean_and_ci_over_every_seed() -> None:
    rows: list[dict[str, Any]] = [
        {
            "scenario": "s",
            "reward_mode": "sparse",
            "seed": seed,
            "eval_success_rate": value,
            "eval_mean_delivered_value": 0.0,
            "eval_mean_base_return": 0.0,
            "eval_mean_shaped_return": 0.0,
            "eval_mean_steps": 0.0,
            "eval_mean_energy_remaining_on_success": 0.0,
            "eval_repeated_edge_fraction": 0.0,
            "eval_max_undirected_edge_repeats": 0.0,
            "train_tied_state_fraction": 0.0,
            "env_steps_to_threshold": None,
            "episodes_to_threshold": None,
        }
        for seed, value in enumerate([0.2, 0.4, 0.6], start=1)
    ]
    aggregated = aggregate_rows(rows)
    assert len(aggregated) == 1
    entry = aggregated[0]
    assert entry["seeds"] == 3
    assert entry["seeds_reaching_threshold"] == 0
    assert entry["eval_success_rate_mean"] == pytest.approx(0.4)
    assert entry["eval_success_rate_ci_low"] < 0.4 < entry["eval_success_rate_ci_high"]
    assert math.isnan(entry["episodes_to_threshold_mean"])


def test_experiment_writes_every_declared_artifact(
    tmp_path: Path, tiny_experiment: ExperimentConfig
) -> None:
    payload = run_experiment(tiny_experiment, tmp_path, stream=io.StringIO())

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary_per_seed.csv").exists()
    assert (tmp_path / "summary_aggregated.csv").exists()
    for name in ("learning_curves.png", "reward_comparison.png", "failure_modes.png"):
        assert (tmp_path / "plots" / name).exists(), name

    assert len(payload["per_seed"]) == tiny_experiment.total_runs
    assert len(payload["aggregated"]) == 2
    assert payload["config"]["seeds"] == [1, 2]
    assert "git_commit" in payload["environment"]

    for row in payload["per_seed"]:
        run_dir = Path(row["run_dir"])
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "q_table.npy").exists()
        assert (run_dir / "evaluation.json").exists()


def test_experiment_records_the_teaching_state(
    tmp_path: Path, tiny_experiment: ExperimentConfig
) -> None:
    payload = run_experiment(tiny_experiment, tmp_path, stream=io.StringIO(), make_plots=False)
    assert payload["learning_is_meaningful"] == (not payload["pending_human_functions"])


def test_base_and_shaped_returns_stay_separate_in_the_summary(
    tmp_path: Path, tiny_experiment: ExperimentConfig
) -> None:
    payload = run_experiment(tiny_experiment, tmp_path, stream=io.StringIO(), make_plots=False)
    for row in payload["per_seed"]:
        assert "eval_mean_base_return" in row
        assert "eval_mean_shaped_return" in row


@pytest.fixture
def tiny_sweep() -> ExperimentConfig:
    return ExperimentConfig(
        name="pytest_sweep",
        scenarios=("safe_corridor",),
        reward_modes=("sparse",),
        seeds=(1, 2),
        episode_budgets=(2, 4),
        curriculum_fractions=(0.0, 0.5),
        eval_episodes=2,
        threshold_window=2,
        save_q_tables="max_budget",
    )


def test_sweep_writes_a_cell_per_budget_and_curriculum(
    tmp_path: Path, tiny_sweep: ExperimentConfig
) -> None:
    payload = run_experiment(tiny_sweep, tmp_path, stream=io.StringIO())

    assert len(payload["per_seed"]) == 8
    assert {row["train_episodes"] for row in payload["per_seed"]} == {2, 4}
    assert {row["curriculum_fraction"] for row in payload["per_seed"]} == {0.0, 0.5}
    # One aggregated entry per (budget, curriculum); budgets are never pooled.
    assert len(payload["aggregated"]) == 4
    assert len({Path(row["run_dir"]) for row in payload["per_seed"]}) == 8


def test_sweep_keeps_tables_only_at_the_largest_budget(
    tmp_path: Path, tiny_sweep: ExperimentConfig
) -> None:
    payload = run_experiment(tiny_sweep, tmp_path, stream=io.StringIO(), make_plots=False)
    for row in payload["per_seed"]:
        run_dir = Path(row["run_dir"])
        # Everything the analysis reads is always written.
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "training_metrics.csv").exists()
        assert (run_dir / "evaluation.json").exists()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        kept = row["train_episodes"] == 4
        assert (run_dir / "q_table.npy").exists() is kept
        assert (manifest["artifacts"]["q_table"] is not None) is kept


def test_sweep_writes_the_analysis_and_its_figures(
    tmp_path: Path, tiny_sweep: ExperimentConfig
) -> None:
    payload = run_experiment(tiny_sweep, tmp_path, stream=io.StringIO())

    assert (tmp_path / "sweep_analysis.json").exists()
    for name in ("budget_curves.png", "paired_difference.png"):
        assert (tmp_path / "plots" / name).exists(), name

    analysis = payload["sweep_analysis"]
    (group,) = analysis["groups"]
    assert analysis["metric"] == "eval_mean_base_return"
    assert {s["curriculum_fraction"] for s in group["series"]} == {0.0, 0.5}
    # One paired comparison per budget, for the one non-baseline arm.
    assert {entry["budget"] for entry in group["paired"]} == {2, 4}


def test_sweep_results_do_not_depend_on_the_worker_count(
    tmp_path: Path, tiny_sweep: ExperimentConfig
) -> None:
    """Every cell is independent and fully seeded; ``--jobs`` must be a pure speedup."""
    serial = run_experiment(tiny_sweep, tmp_path / "serial", stream=io.StringIO(), make_plots=False)
    parallel = run_experiment(
        tiny_sweep, tmp_path / "parallel", stream=io.StringIO(), make_plots=False, jobs=2
    )

    def comparable(payload: dict[str, Any]) -> Any:
        # Through ``json_safe`` so that an undefined mean compares equal to itself:
        # ``nan != nan``, and the interesting question is what lands in the CSVs.
        return json_safe(
            [{k: v for k, v in row.items() if k != "run_dir"} for row in payload["per_seed"]]
        )

    assert comparable(serial) == comparable(parallel)
