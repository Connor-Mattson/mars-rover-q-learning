"""Run-directory serialisation and the reward-comparison experiment grid.

Every declared seed is aggregated. Nothing here selects a best seed, and shaping
reward is never folded into the base mission-return comparison.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .agent import teaching_stub_status
from .evaluation import EvaluationResult, Trajectory, evaluate
from .metrics import (
    CSV_COLUMNS,
    EpisodeRecord,
    env_steps_to_threshold,
    episodes_to_threshold,
    greedy_actions,
    mean_ci,
)
from .rewards import RewardMode
from .scenario import Scenario, resolve_scenario
from .training import TrainConfig, TrainResult, train

MANIFEST_NAME = "manifest.json"
QTABLE_NAME = "q_table.npy"
POLICY_NAME = "policy.npy"
TRAINING_CSV = "training_metrics.csv"
EVAL_JSON = "evaluation.json"
BEST_EPISODE = "best_episode.json"


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` so output is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as strict JSON, with non-finite floats as ``null``."""
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[2],
        )
    except OSError:  # pragma: no cover - git absent
        return None
    return result.stdout.strip() or None


def environment_metadata() -> dict[str, Any]:
    """Package and platform versions recorded alongside every run."""
    import matplotlib
    import pygame

    return {
        "mars_rover_q": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "pygame": pygame.version.ver,
        "matplotlib": matplotlib.__version__,
        "git_commit": _git_commit(),
    }


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """The full reward-comparison grid."""

    name: str = "reward_comparison"
    scenarios: tuple[str, ...] = ("safe_corridor", "risk_value_tradeoff", "shaping_trap")
    reward_modes: tuple[str, ...] = tuple(str(mode) for mode in RewardMode)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    episodes: int = 4000
    eval_episodes: int = 300
    learning_rate: float = 0.2
    gamma: float = 0.99
    initial_q: float = 0.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.6
    potential_scale: float = 1.0
    success_threshold: float = 0.8
    threshold_window: int = 100
    log_every: int = 0

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        """Load an experiment config from JSON, ignoring unknown keys with an error."""
        resolved = Path(path)
        with resolved.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"{resolved}: unknown experiment config keys {unknown}")
        tuples = {"scenarios", "reward_modes", "seeds"}
        kwargs: dict[str, Any] = {k: (tuple(v) if k in tuples else v) for k, v in payload.items()}
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot, with tuples flattened to lists."""
        payload = asdict(self)
        for key in ("scenarios", "reward_modes", "seeds"):
            payload[key] = list(payload[key])
        return payload

    def train_config(self, scenario: str, reward_mode: str, seed: int) -> TrainConfig:
        """The :class:`TrainConfig` for one cell of the grid."""
        return TrainConfig(
            scenario=scenario,
            reward_mode=RewardMode(reward_mode),
            seed=seed,
            episodes=self.episodes,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            initial_q=self.initial_q,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_fraction=self.epsilon_decay_fraction,
            potential_scale=self.potential_scale,
            log_every=self.log_every,
        )

    @property
    def total_runs(self) -> int:
        """Number of ``(scenario, reward mode, seed)`` cells."""
        return len(self.scenarios) * len(self.reward_modes) * len(self.seeds)


@dataclass(slots=True)
class RunPaths:
    """Where one run's artefacts live."""

    root: Path

    @property
    def manifest(self) -> Path:
        """Path to the machine-readable run manifest."""
        return self.root / MANIFEST_NAME


def write_training_csv(path: Path, records: Sequence[EpisodeRecord]) -> None:
    """Write per-episode training metrics as CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_dict())


def read_training_csv(path: Path) -> list[EpisodeRecord]:
    """Read back per-episode training metrics written by :func:`write_training_csv`."""
    numeric_floats = {"base_return", "shaped_return", "repeated_edge_fraction", "epsilon"}
    records: list[EpisodeRecord] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            typed: dict[str, Any] = {}
            for key, value in row.items():
                typed[key] = (
                    float(value)
                    if key in numeric_floats
                    else (value if key == "outcome" else int(value))
                )
            records.append(EpisodeRecord(**typed))
    return records


def save_run(
    directory: Path,
    scenario: Scenario,
    result: TrainResult,
    evaluation: EvaluationResult | None,
    *,
    experiment: str | None = None,
) -> RunPaths:
    """Persist one training run: config, metrics, table, policy, and manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = RunPaths(directory)

    write_training_csv(directory / TRAINING_CSV, result.records)
    np.save(directory / QTABLE_NAME, result.q_table)
    policy = greedy_actions(result.q_table)
    np.save(directory / POLICY_NAME, policy)
    write_json(directory / "scenario.json", scenario.to_dict())

    if evaluation is not None:
        write_json(directory / EVAL_JSON, evaluation.as_dict())
        if evaluation.best_trajectory is not None:
            write_json(directory / BEST_EPISODE, evaluation.best_trajectory.as_dict())

    manifest = {
        "run_id": directory.name,
        "experiment": experiment,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scenario": scenario.name,
        "config": result.config.as_dict(),
        "environment": environment_metadata(),
        "total_env_steps": result.total_env_steps,
        "num_states": result.metadata.get("num_states"),
        "pending_human_functions": list(result.pending_human_functions),
        "learning_is_meaningful": result.learning_is_meaningful,
        "artifacts": {
            "training_metrics": TRAINING_CSV,
            "q_table": QTABLE_NAME,
            "policy": POLICY_NAME,
            "evaluation": EVAL_JSON if evaluation is not None else None,
            "best_episode": (
                BEST_EPISODE
                if evaluation is not None and evaluation.best_trajectory is not None
                else None
            ),
        },
    }
    write_json(paths.manifest, manifest)
    return paths


@dataclass(slots=True)
class LoadedRun:
    """A run directory read back from disk."""

    root: Path
    manifest: dict[str, Any]
    scenario: Scenario
    q_table: NDArray[np.float64]
    records: list[EpisodeRecord] = field(default_factory=list)
    best_trajectory: Trajectory | None = None

    @property
    def policy(self) -> NDArray[np.int64]:
        """The greedy policy implied by the stored table."""
        return greedy_actions(self.q_table)


def load_run(directory: str | Path) -> LoadedRun:
    """Read a run directory produced by :func:`save_run`."""
    root = Path(directory)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario = resolve_scenario(manifest["scenario"])
    q_table = np.load(root / QTABLE_NAME)
    records = read_training_csv(root / TRAINING_CSV) if (root / TRAINING_CSV).exists() else []
    best = None
    if (root / BEST_EPISODE).exists():
        best = Trajectory.from_dict(json.loads((root / BEST_EPISODE).read_text(encoding="utf-8")))
    return LoadedRun(root, manifest, scenario, q_table, records, best)


def run_cell(
    config: ExperimentConfig,
    scenario_name: str,
    reward_mode: str,
    seed: int,
    output_root: Path,
    *,
    stream: Any = None,
) -> tuple[TrainResult, EvaluationResult, dict[str, Any]]:
    """Train and evaluate one ``(scenario, reward mode, seed)`` cell."""
    scenario = resolve_scenario(scenario_name)
    train_config = config.train_config(scenario_name, reward_mode, seed)
    result = train(scenario, train_config, warn_on_stubs=False, stream=stream)
    evaluation = evaluate(
        result.q_table,
        scenario,
        reward_mode,
        episodes=config.eval_episodes,
        seed=seed,
        gamma=config.gamma,
        potential_scale=config.potential_scale,
    )
    run_dir = output_root / "runs" / f"{scenario_name}__{reward_mode}__seed{seed}"
    save_run(run_dir, scenario, result, evaluation, experiment=config.name)

    summary = evaluation.summary
    row: dict[str, Any] = {
        "scenario": scenario_name,
        "reward_mode": reward_mode,
        "seed": seed,
        "run_dir": str(run_dir),
        "train_episodes": config.episodes,
        "train_env_steps": result.total_env_steps,
        "episodes_to_threshold": episodes_to_threshold(
            result.records, config.success_threshold, config.threshold_window
        ),
        "env_steps_to_threshold": env_steps_to_threshold(
            result.records, config.success_threshold, config.threshold_window
        ),
        "eval_success_rate": summary.success_rate,
        "eval_mean_delivered_value": summary.mean_delivered_value,
        "eval_mean_base_return": summary.mean_base_return,
        "eval_mean_shaped_return": summary.mean_shaped_return,
        "eval_mean_steps": summary.mean_steps,
        "eval_mean_energy_remaining_on_success": summary.mean_energy_remaining_on_success,
        "eval_repeated_edge_fraction": summary.mean_repeated_edge_fraction,
        "eval_max_undirected_edge_repeats": summary.mean_max_undirected_edge_repeats,
        "failure_battery_depleted": summary.failure_reasons.get("battery_depleted", 0),
        "failure_step_limit": summary.failure_reasons.get("step_limit", 0),
        "learning_is_meaningful": result.learning_is_meaningful,
    }
    return result, evaluation, row


AGGREGATED_METRICS: tuple[str, ...] = (
    "eval_success_rate",
    "eval_mean_delivered_value",
    "eval_mean_base_return",
    "eval_mean_shaped_return",
    "eval_mean_steps",
    "eval_mean_energy_remaining_on_success",
    "eval_repeated_edge_fraction",
    "eval_max_undirected_edge_repeats",
    "env_steps_to_threshold",
    "episodes_to_threshold",
)


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed rows into mean and 95% CI per (scenario, reward mode)."""
    aggregated: list[dict[str, Any]] = []
    keys = sorted({(row["scenario"], row["reward_mode"]) for row in rows})
    for scenario, reward_mode in keys:
        cell = [r for r in rows if r["scenario"] == scenario and r["reward_mode"] == reward_mode]
        entry: dict[str, Any] = {
            "scenario": scenario,
            "reward_mode": reward_mode,
            "seeds": len(cell),
            "seeds_reaching_threshold": sum(
                1 for r in cell if r["episodes_to_threshold"] is not None
            ),
        }
        for metric in AGGREGATED_METRICS:
            values = [r[metric] for r in cell if r[metric] is not None]
            interval = mean_ci(values)
            entry[f"{metric}_mean"] = interval.mean
            entry[f"{metric}_ci_low"] = interval.low
            entry[f"{metric}_ci_high"] = interval.high
            entry[f"{metric}_n"] = interval.n
        aggregated.append(entry)
    return aggregated


def run_experiment(
    config: ExperimentConfig,
    output_root: Path,
    *,
    stream: Any = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Run the full grid, save every artefact, and return the summary payload."""
    import sys

    out = stream if stream is not None else sys.stderr
    output_root.mkdir(parents=True, exist_ok=True)
    pending = teaching_stub_status()

    rows: list[dict[str, Any]] = []
    index = 0
    for scenario_name in config.scenarios:
        for reward_mode in config.reward_modes:
            for seed in config.seeds:
                index += 1
                print(
                    f"[{index}/{config.total_runs}] {scenario_name} / {reward_mode} / seed {seed}",
                    file=out,
                )
                _result, _evaluation, row = run_cell(
                    config, scenario_name, reward_mode, seed, output_root, stream=out
                )
                rows.append(row)

    aggregated = aggregate_rows(rows)
    payload: dict[str, Any] = {
        "config": config.as_dict(),
        "environment": environment_metadata(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "pending_human_functions": list(pending),
        "learning_is_meaningful": not pending,
        "per_seed": rows,
        "aggregated": aggregated,
    }
    write_json(output_root / "summary.json", payload)
    _write_csv(output_root / "summary_per_seed.csv", rows)
    _write_csv(output_root / "summary_aggregated.csv", aggregated)

    if make_plots:
        from .plots import write_experiment_plots

        write_experiment_plots(output_root, rows, aggregated)
    return payload


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "AGGREGATED_METRICS",
    "ExperimentConfig",
    "LoadedRun",
    "RunPaths",
    "aggregate_rows",
    "environment_metadata",
    "load_run",
    "read_training_csv",
    "run_cell",
    "run_experiment",
    "save_run",
    "write_json",
    "write_training_csv",
]
