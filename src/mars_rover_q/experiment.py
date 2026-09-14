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
from .curriculum import DEFAULT_WEIGHT_EXPONENT, DEFAULT_WINDOW_FRACTION, CurriculumStrategy
from .evaluation import EvaluationResult, Trajectory, evaluate
from .metrics import (
    CHECKPOINT_CSV_COLUMNS,
    CSV_COLUMNS,
    CheckpointRecord,
    EpisodeRecord,
    canonical_start_records,
    env_steps_to_threshold,
    episodes_to_threshold,
    greedy_actions,
    mean_ci,
)
from .numerics import install_numeric_guard
from .rewards import RewardMode
from .scenario import Scenario, resolve_scenario
from .state import BatteryEncoding
from .training import TrainConfig, TrainResult, train

MANIFEST_NAME = "manifest.json"
QTABLE_NAME = "q_table.npy"
POLICY_NAME = "policy.npy"
VISITS_NAME = "visit_counts.npy"
TRAINING_CSV = "training_metrics.csv"
CHECKPOINT_CSV = "eval_checkpoints.csv"
EVAL_JSON = "evaluation.json"
BEST_EPISODE = "best_episode.json"

#: How each :class:`EpisodeRecord` column is parsed back out of the training CSV.
#: Anything not listed here is an integer. ``csv`` has no types of its own, so a
#: missing entry silently turns a bool into ``int("True")`` and raises.
_CSV_FLOAT_FIELDS: frozenset[str] = frozenset(
    {"base_return", "shaped_return", "repeated_edge_fraction", "epsilon"}
)
_CSV_BOOL_FIELDS: frozenset[str] = frozenset({"from_canonical_start"})
_CSV_TEXT_FIELDS: frozenset[str] = frozenset({"outcome"})


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
    """The experiment grid.

    Four factors are crossed: scenario, reward mode, seed, and -- for the
    curriculum budget sweep -- episode budget and curriculum fraction.

    ``episodes`` and ``curriculum_fraction`` are the single-value forms and remain
    the defaults, so every config written before the sweep existed still describes
    exactly the grid it always did. ``episode_budgets`` and
    ``curriculum_fractions`` are their plural forms: when non-empty they *replace*
    the scalar and become swept factors. See :attr:`budgets` and :attr:`curricula`.
    """

    name: str = "reward_comparison"
    scenarios: tuple[str, ...] = ("safe_corridor", "risk_value_tradeoff", "shaping_trap")
    reward_modes: tuple[str, ...] = tuple(str(mode) for mode in RewardMode)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    episodes: int = 4000
    eval_episodes: int = 300
    learning_rate: float = 0.2
    gamma: float = 0.99
    initial_q: float = 0.0
    #: How finely the tables resolve remaining charge. The grid does not sweep this
    #: -- an encoding change makes tables of different lengths, so an arm under one
    #: encoding cannot share a colour bar or a paired difference with an arm under
    #: another -- but a whole grid can be re-run under the dense encoding to compare
    #: the two as experiments rather than as planning bounds.
    battery_encoding: str = BatteryEncoding.AFFORDABILITY.value
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.6
    curriculum_fraction: float = 0.0
    #: How the swept curriculum arms draw from the ranked pool. The single-value
    #: form; :attr:`curriculum_strategies` is its plural.
    curriculum_strategy: str = CurriculumStrategy.GROWING.value
    curriculum_window_fraction: float = DEFAULT_WINDOW_FRACTION
    curriculum_weight_exponent: float = DEFAULT_WEIGHT_EXPONENT
    #: Swept episode budgets. Empty means "just ``episodes``". Each budget trains a
    #: fresh table for exactly that many episodes, so the epsilon and curriculum
    #: anneals -- both defined as fractions of the budget -- scale with it. A point
    #: on the budget axis is therefore a complete run at that budget, not a
    #: snapshot taken partway through a longer one's schedule.
    episode_budgets: tuple[int, ...] = ()
    #: Swept curriculum anneal fractions. Empty means "just ``curriculum_fraction``".
    curriculum_fractions: tuple[float, ...] = ()
    #: Swept sampling strategies. Empty means "just ``curriculum_strategy``". A
    #: strategy only changes anything when a curriculum is switched on, so the
    #: ``curriculum_fraction == 0`` cells are *not* multiplied by this factor --
    #: see :meth:`cells`.
    curriculum_strategies: tuple[str, ...] = ()
    #: Which cells keep their ``q_table.npy`` and ``policy.npy``: ``"all"``,
    #: ``"max_budget"`` (only the largest budget in the sweep), or ``"none"``.
    #: A swept grid is 300+ runs and the tables dominate the artifact size.
    save_q_tables: str = "all"
    #: Mid-training greedy checkpoints per run; ``0`` disables them. This is what
    #: makes the learning curve comparable across conditions -- see
    #: :class:`mars_rover_q.metrics.CheckpointRecord`.
    eval_checkpoints: int = 0
    #: Greedy episodes per checkpoint.
    checkpoint_episodes: int = 40
    #: Write policy GIFs for the largest budget after the grid finishes.
    write_gifs: bool = False
    #: Which seed the GIFs are rendered from.
    gif_seed: int = 1
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
        tuples = {
            "scenarios",
            "reward_modes",
            "seeds",
            "episode_budgets",
            "curriculum_fractions",
            "curriculum_strategies",
        }
        kwargs: dict[str, Any] = {k: (tuple(v) if k in tuples else v) for k, v in payload.items()}
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot, with tuples flattened to lists."""
        payload = asdict(self)
        for key in (
            "scenarios",
            "reward_modes",
            "seeds",
            "episode_budgets",
            "curriculum_fractions",
            "curriculum_strategies",
        ):
            payload[key] = list(payload[key])
        return payload

    @property
    def budgets(self) -> tuple[int, ...]:
        """The episode budgets actually swept, in ascending order."""
        return tuple(sorted(self.episode_budgets)) if self.episode_budgets else (self.episodes,)

    @property
    def curricula(self) -> tuple[float, ...]:
        """The curriculum anneal fractions actually swept, in ascending order."""
        return (
            tuple(sorted(self.curriculum_fractions))
            if self.curriculum_fractions
            else (self.curriculum_fraction,)
        )

    @property
    def strategies(self) -> tuple[str, ...]:
        """The curriculum sampling strategies actually swept, in declared order.

        Not sorted: these are names, and the order they are written in the config
        is the order a reader expects them in the legend. Every one is validated
        here rather than at training time, so a typo fails before 300 cells run.
        """
        names = self.curriculum_strategies or (self.curriculum_strategy,)
        return tuple(CurriculumStrategy(name).value for name in names)

    @property
    def is_swept(self) -> bool:
        """Whether more than one budget or curriculum condition is being compared."""
        return len(self.budgets) > 1 or len(self.curricula) > 1 or len(self.strategies) > 1

    def train_config(
        self,
        scenario: str,
        reward_mode: str,
        seed: int,
        *,
        episodes: int | None = None,
        curriculum_fraction: float | None = None,
        curriculum_strategy: str | None = None,
    ) -> TrainConfig:
        """The :class:`TrainConfig` for one cell of the grid."""
        return TrainConfig(
            scenario=scenario,
            reward_mode=RewardMode(reward_mode),
            seed=seed,
            episodes=self.episodes if episodes is None else episodes,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            initial_q=self.initial_q,
            battery_encoding=BatteryEncoding(self.battery_encoding),
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_fraction=self.epsilon_decay_fraction,
            curriculum_fraction=(
                self.curriculum_fraction if curriculum_fraction is None else curriculum_fraction
            ),
            curriculum_strategy=CurriculumStrategy(
                self.curriculum_strategy if curriculum_strategy is None else curriculum_strategy
            ),
            curriculum_window_fraction=self.curriculum_window_fraction,
            curriculum_weight_exponent=self.curriculum_weight_exponent,
            potential_scale=self.potential_scale,
            log_every=self.log_every,
            eval_checkpoints=self.eval_checkpoints,
            checkpoint_episodes=self.checkpoint_episodes,
        )

    def strategies_for(self, curriculum: float) -> tuple[str, ...]:
        """The strategies this curriculum level is actually run under.

        A disabled curriculum starts every episode at the lander whatever the
        sampler says, so running the ``curriculum_fraction == 0`` control once per
        strategy would train identical tables under different labels -- and then
        split one control arm into three in every comparison downstream. The
        control is run once, under the first declared strategy.
        """
        return self.strategies if curriculum > 0.0 else self.strategies[:1]

    @property
    def total_runs(self) -> int:
        """Number of ``(scenario, reward mode, budget, curriculum, strategy, seed)`` cells."""
        return len(self.cells())

    @property
    def total_train_episodes(self) -> int:
        """Total training episodes the whole grid will run, for cost reporting."""
        per_budget = sum(
            len(self.scenarios) * len(self.reward_modes) * len(self.strategies_for(curriculum))
            for curriculum in self.curricula
        ) * len(self.seeds)
        return per_budget * sum(self.budgets)

    def cells(self) -> list[tuple[str, str, int, float, str, int]]:
        """Every ``(scenario, reward mode, budget, curriculum, strategy, seed)`` cell.

        Ordered so that the expensive factor -- the episode budget -- varies
        slowest, which keeps a partially finished sweep useful: the cheap end of
        the budget axis is complete for every condition before any long run starts.
        """
        return [
            (scenario, reward_mode, budget, curriculum, strategy, seed)
            for budget in self.budgets
            for scenario in self.scenarios
            for reward_mode in self.reward_modes
            for curriculum in self.curricula
            for strategy in self.strategies_for(curriculum)
            for seed in self.seeds
        ]

    def run_name(
        self,
        scenario: str,
        reward_mode: str,
        seed: int,
        budget: int,
        curriculum: float,
        strategy: str = CurriculumStrategy.GROWING.value,
    ) -> str:
        """The run-directory name for one cell.

        Segments for the budget, curriculum, and strategy factors appear only when
        that factor actually varies, so a grid that does not sweep them produces
        exactly the directory names it produced before the sweep existed. The
        strategy segment is dropped from a disabled-curriculum cell for the reason
        :meth:`strategies_for` gives: there is only one such cell to name.
        """
        parts = [scenario, reward_mode]
        if len(self.curricula) > 1:
            parts.append(f"cf{curriculum:g}")
        if len(self.strategies) > 1 and curriculum > 0.0:
            parts.append(f"cs{strategy}")
        if len(self.budgets) > 1:
            parts.append(f"ep{budget}")
        parts.append(f"seed{seed}")
        return "__".join(parts)


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


def write_checkpoint_csv(path: Path, records: Sequence[CheckpointRecord]) -> None:
    """Write mid-training greedy-evaluation checkpoints as CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CHECKPOINT_CSV_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_dict())


def read_checkpoint_csv(path: Path) -> list[CheckpointRecord]:
    """Read back the checkpoints written by :func:`write_checkpoint_csv`."""
    records: list[CheckpointRecord] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                CheckpointRecord(
                    episode=int(row["episode"]),
                    env_steps=int(row["env_steps"]),
                    episodes=int(row["episodes"]),
                    success_rate=float(row["success_rate"]),
                    mean_base_return=float(row["mean_base_return"]),
                    mean_delivered_value=float(row["mean_delivered_value"]),
                    mean_steps=float(row["mean_steps"]),
                )
            )
    return records


def _typed_csv_value(key: str, value: str) -> Any:
    """Coerce one CSV cell back to the type :class:`EpisodeRecord` declares."""
    if key in _CSV_FLOAT_FIELDS:
        return float(value)
    if key in _CSV_BOOL_FIELDS:
        return value == "True"
    if key in _CSV_TEXT_FIELDS:
        return value
    return int(value)


def read_training_csv(path: Path) -> list[EpisodeRecord]:
    """Read back per-episode training metrics written by :func:`write_training_csv`."""
    records: list[EpisodeRecord] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            typed = {key: _typed_csv_value(key, value) for key, value in row.items()}
            records.append(EpisodeRecord(**typed))
    return records


def save_run(
    directory: Path,
    scenario: Scenario,
    result: TrainResult,
    evaluation: EvaluationResult | None,
    *,
    experiment: str | None = None,
    save_q_table: bool = True,
    save_eval_episodes: bool = True,
) -> RunPaths:
    """Persist one training run: config, metrics, table, policy, and manifest.

    ``save_q_table`` and ``save_eval_episodes`` exist for the budget sweep, which is
    300+ cells: the tables and the per-episode evaluation dumps are the two things
    that scale the artifact tree into gigabytes, and neither is needed to read the
    sweep's curves. Everything the analysis consumes -- the summary row, the
    training CSV, the manifest -- is written either way, and the manifest records
    which files were skipped so a reader is never left guessing.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = RunPaths(directory)

    write_training_csv(directory / TRAINING_CSV, result.records)
    if result.checkpoints:
        write_checkpoint_csv(directory / CHECKPOINT_CSV, result.checkpoints)
    if save_q_table:
        np.save(directory / QTABLE_NAME, result.q_table)
        np.save(directory / POLICY_NAME, greedy_actions(result.q_table))
        # Same size as the policy, and gated with it for the same reason: the sweep
        # writes 300+ cells and reads none of these back.
        if result.visit_counts.size:
            np.save(directory / VISITS_NAME, result.visit_counts)
    write_json(directory / "scenario.json", scenario.to_dict())

    if evaluation is not None:
        write_json(
            directory / EVAL_JSON,
            evaluation.as_dict()
            if save_eval_episodes
            else {"summary": evaluation.summary.as_dict()},
        )
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
        "tied_state_fraction": result.metadata.get("tied_state_fraction"),
        "learned_state_fraction": result.metadata.get("learned_state_fraction"),
        "visited_state_count": result.metadata.get("visited_state_count"),
        "curriculum": result.metadata.get("curriculum"),
        "pending_human_functions": list(result.pending_human_functions),
        "learning_is_meaningful": result.learning_is_meaningful,
        "artifacts": {
            "training_metrics": TRAINING_CSV,
            "eval_checkpoints": CHECKPOINT_CSV if result.checkpoints else None,
            "q_table": QTABLE_NAME if save_q_table else None,
            "policy": POLICY_NAME if save_q_table else None,
            "visit_counts": (VISITS_NAME if save_q_table and result.visit_counts.size else None),
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
    checkpoints: list[CheckpointRecord] = field(default_factory=list)
    #: Per-state Q-update counts, when the run was saved with them. Runs written
    #: before this was recorded, and sweep cells saved without their tables, have
    #: ``None`` here rather than a zero array, so a reader can tell "no experience"
    #: apart from "experience not recorded".
    visit_counts: NDArray[np.int64] | None = None

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
    table_path = root / QTABLE_NAME
    if not table_path.exists():
        raise FileNotFoundError(
            f"{root} has no {QTABLE_NAME}: it was saved without its table "
            "(save_q_tables was not 'all'). Its summary row and training CSV are "
            "intact; re-run this cell if you need the table itself."
        )
    q_table = np.load(table_path)
    records = read_training_csv(root / TRAINING_CSV) if (root / TRAINING_CSV).exists() else []
    checkpoints = (
        read_checkpoint_csv(root / CHECKPOINT_CSV) if (root / CHECKPOINT_CSV).exists() else []
    )
    best = None
    if (root / BEST_EPISODE).exists():
        best = Trajectory.from_dict(json.loads((root / BEST_EPISODE).read_text(encoding="utf-8")))
    visits = np.load(root / VISITS_NAME) if (root / VISITS_NAME).exists() else None
    return LoadedRun(root, manifest, scenario, q_table, records, best, checkpoints, visits)


def _keeps_table(config: ExperimentConfig, episodes: int) -> bool:
    """Whether this cell's Q-table is written to disk, per ``save_q_tables``."""
    if config.save_q_tables == "none":
        return False
    if config.save_q_tables == "max_budget":
        return episodes == max(config.budgets)
    return True


def run_cell(
    config: ExperimentConfig,
    scenario_name: str,
    reward_mode: str,
    seed: int,
    output_root: Path,
    *,
    episodes: int | None = None,
    curriculum_fraction: float | None = None,
    curriculum_strategy: str | None = None,
    stream: Any = None,
) -> tuple[TrainResult, EvaluationResult, dict[str, Any]]:
    """Train and evaluate one cell of the grid.

    ``episodes``, ``curriculum_fraction``, and ``curriculum_strategy`` override the
    config's scalars for this cell; the sweep passes the cell's own budget,
    curriculum level, and sampling strategy here. The
    evaluation is deliberately *not* scaled with the budget: every cell is judged by
    the same ``config.eval_episodes`` greedy episodes from the canonical lander
    start, so a point at 1k and a point at 20k are measured with the same ruler.
    """
    scenario = resolve_scenario(scenario_name)
    train_config = config.train_config(
        scenario_name,
        reward_mode,
        seed,
        episodes=episodes,
        curriculum_fraction=curriculum_fraction,
        curriculum_strategy=curriculum_strategy,
    )
    result = train(scenario, train_config, warn_on_stubs=False, stream=stream)
    evaluation = evaluate(
        result.q_table,
        scenario,
        reward_mode,
        episodes=config.eval_episodes,
        seed=seed,
        gamma=config.gamma,
        potential_scale=config.potential_scale,
        battery_encoding=train_config.battery_encoding,
    )
    run_dir = (
        output_root
        / "runs"
        / config.run_name(
            scenario_name,
            reward_mode,
            seed,
            train_config.episodes,
            train_config.curriculum_fraction,
            str(train_config.curriculum_strategy),
        )
    )
    save_run(
        run_dir,
        scenario,
        result,
        evaluation,
        experiment=config.name,
        save_q_table=_keeps_table(config, train_config.episodes),
        save_eval_episodes=not config.is_swept,
    )

    summary = evaluation.summary
    # Threshold metrics read the canonical-start episodes only: a curriculum run's
    # easy episodes would otherwise cross the threshold on a mission that is not the
    # one being compared. ``env_steps_before`` still counts every step taken, so
    # sample efficiency stays honest across conditions.
    canonical_records = canonical_start_records(result.records)
    row: dict[str, Any] = {
        "scenario": scenario_name,
        "reward_mode": reward_mode,
        "seed": seed,
        "run_dir": str(run_dir),
        "train_episodes": train_config.episodes,
        "train_env_steps": result.total_env_steps,
        "train_tied_state_fraction": result.metadata.get("tied_state_fraction"),
        "curriculum_fraction": train_config.curriculum_fraction,
        "curriculum_strategy": str(train_config.curriculum_strategy),
        "curriculum_active": result.curriculum_was_active,
        "episodes_to_threshold": episodes_to_threshold(
            canonical_records, config.success_threshold, config.threshold_window
        ),
        "env_steps_to_threshold": env_steps_to_threshold(
            canonical_records, config.success_threshold, config.threshold_window
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
    "train_tied_state_fraction",
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


#: The row keys that identify one experimental condition. Seeds are what gets
#: aggregated *over*; everything else here is a factor that must not be pooled.
#: ``.get`` is used against these so that a row from a grid predating the sweep --
#: which carries no budget or curriculum column -- still groups cleanly as a single
#: condition rather than raising.
CONDITION_KEYS: tuple[str, ...] = (
    "scenario",
    "reward_mode",
    "train_episodes",
    "curriculum_fraction",
    "curriculum_strategy",
)


def condition_of(row: dict[str, Any]) -> tuple[Any, ...]:
    """The condition a per-seed row belongs to."""
    return tuple(row.get(key) for key in CONDITION_KEYS)


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed rows into mean and 95% CI per condition.

    A condition is a full factor combination (see :data:`CONDITION_KEYS`), so a
    budget sweep produces one entry per (scenario, reward mode, budget, curriculum)
    rather than silently pooling budgets together.
    """
    aggregated: list[dict[str, Any]] = []
    keys = sorted({condition_of(row) for row in rows}, key=lambda k: [(v is None, v) for v in k])
    for key in keys:
        cell = [r for r in rows if condition_of(r) == key]
        entry: dict[str, Any] = {
            **dict(zip(CONDITION_KEYS, key, strict=True)),
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


def _run_one_cell(
    task: tuple[ExperimentConfig, str, str, int, float, str, int, Path],
) -> dict[str, Any]:
    """Run one cell and return only its summary row.

    The worker entry point for the process pool. It deliberately returns the row
    and nothing else: a :class:`TrainResult` carries the Q-table and every episode
    record, and pickling those back to the parent would cost more than the training
    did. The artefacts are already on disk by the time this returns.
    """
    config, scenario_name, reward_mode, budget, curriculum, strategy, seed, output_root = task
    install_numeric_guard()
    _result, _evaluation, row = run_cell(
        config,
        scenario_name,
        reward_mode,
        seed,
        output_root,
        episodes=budget,
        curriculum_fraction=curriculum,
        curriculum_strategy=strategy,
        stream=None,
    )
    return row


def run_experiment(
    config: ExperimentConfig,
    output_root: Path,
    *,
    stream: Any = None,
    make_plots: bool = True,
    jobs: int = 1,
) -> dict[str, Any]:
    """Run the full grid, save every artefact, and return the summary payload.

    Args:
        config: the grid to run.
        output_root: where ``runs/``, the summaries, and ``plots/`` are written.
        stream: progress destination; defaults to ``sys.stderr``.
        make_plots: whether to render the figures.
        jobs: worker processes. Every cell is independent and fully seeded, so the
            per-seed rows are identical for any ``jobs``; only the order in which
            progress is printed changes.
    """
    import sys

    out = stream if stream is not None else sys.stderr
    output_root.mkdir(parents=True, exist_ok=True)
    pending = teaching_stub_status()

    cells = config.cells()
    print(
        f"{len(cells)} cells, {config.total_train_episodes} training episodes total",
        file=out,
    )
    tasks = [
        (config, scenario, reward_mode, budget, curriculum, strategy, seed, output_root)
        for scenario, reward_mode, budget, curriculum, strategy, seed in cells
    ]

    rows: list[dict[str, Any]] = []
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for index, row in enumerate(pool.map(_run_one_cell, tasks), start=1):
                rows.append(row)
                print(f"[{index}/{len(tasks)}] {_cell_label(row)}", file=out)
    else:
        for index, task in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {_cell_label_for_task(task)}", file=out, flush=True)
            _result, _evaluation, row = run_cell(
                task[0],
                task[1],
                task[2],
                task[6],
                task[7],
                episodes=task[3],
                curriculum_fraction=task[4],
                curriculum_strategy=task[5],
                stream=out,
            )
            rows.append(row)

    # Ordered by cell so the CSVs are stable regardless of how the pool interleaved.
    order = {
        (scenario, reward_mode, budget, curriculum, strategy, seed): i
        for i, (scenario, reward_mode, budget, curriculum, strategy, seed) in enumerate(cells)
    }
    rows.sort(
        key=lambda r: order[
            (
                r["scenario"],
                r["reward_mode"],
                r["train_episodes"],
                r["curriculum_fraction"],
                r["curriculum_strategy"],
                r["seed"],
            )
        ]
    )

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

    if config.is_swept:
        from .sweep import analyze_sweep

        analysis = analyze_sweep(rows)
        payload["sweep_analysis"] = analysis
        write_json(output_root / "sweep_analysis.json", analysis)

    if make_plots:
        from .plots import write_experiment_plots

        write_experiment_plots(output_root, rows, aggregated, payload.get("sweep_analysis"))
    if config.write_gifs:
        print("writing policy animations", file=out)
        payload["gifs"] = [
            str(path) for path in write_policy_gifs(config, output_root, rows, stream=out)
        ]
    return payload


def write_policy_gifs(
    config: ExperimentConfig,
    output_root: Path,
    rows: Sequence[dict[str, Any]],
    *,
    stream: Any = None,
) -> list[Path]:
    """Animate one representative episode per condition at the largest budget.

    Only the largest budget is animated, and only ``config.gif_seed``: the point of
    these files is to show what each *condition* learned, and one policy per
    condition per map is what a reader can actually watch. The episode shown is the
    median-return one of a fresh greedy batch, not the best of it -- the best
    episode of a stochastic batch is a best case, and a reader shown only best
    cases is being flattered.

    Requires the cell's Q-table, so ``save_q_tables`` must have kept it. Cells
    whose table was not saved are skipped with a note rather than failing the run.
    """
    import sys

    from .animation import caption_for, gif_name, write_policy_gif

    out = stream if stream is not None else sys.stderr
    budget = max(config.budgets)
    gifs_dir = output_root / "gifs"
    written: list[Path] = []

    targets = [
        row
        for row in rows
        if int(row["train_episodes"]) == budget and int(row["seed"]) == config.gif_seed
    ]
    for row in sorted(
        targets,
        key=lambda r: (r["scenario"], r["curriculum_fraction"], r.get("curriculum_strategy", "")),
    ):
        run_dir = Path(row["run_dir"])
        if not (run_dir / QTABLE_NAME).exists():
            print(f"  skipping GIF for {run_dir.name}: no saved Q-table", file=out)
            continue
        run = load_run(run_dir)
        result = evaluate(
            run.q_table,
            run.scenario,
            row["reward_mode"],
            episodes=config.eval_episodes,
            seed=int(row["seed"]),
            gamma=config.gamma,
            potential_scale=config.potential_scale,
            battery_encoding=BatteryEncoding(config.battery_encoding),
            capture_best=False,
            capture_typical=True,
        )
        if result.typical_trajectory is None:  # pragma: no cover - eval_episodes > 0
            continue
        curriculum = float(row["curriculum_fraction"])
        strategy = str(row.get("curriculum_strategy", CurriculumStrategy.GROWING.value))
        path = gifs_dir / gif_name(row["scenario"], curriculum, budget, int(row["seed"]), strategy)
        write_policy_gif(
            path,
            run.scenario,
            result.typical_trajectory,
            caption_lines=caption_for(
                row["scenario"], curriculum, budget, int(row["seed"]), strategy
            ),
        )
        print(f"  wrote {path}", file=out)
        written.append(path)
    return written


def _cell_label(row: dict[str, Any]) -> str:
    return (
        f"{row['scenario']} / {row['reward_mode']} / ep{row['train_episodes']} "
        f"/ cf{row['curriculum_fraction']:g} / {row.get('curriculum_strategy', '-')} "
        f"/ seed {row['seed']}"
    )


def _cell_label_for_task(task: tuple[Any, ...]) -> str:
    _config, scenario, reward_mode, budget, curriculum, strategy, seed, _root = task
    return f"{scenario} / {reward_mode} / ep{budget} / cf{curriculum:g} / {strategy} / seed {seed}"


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
    "CHECKPOINT_CSV",
    "CONDITION_KEYS",
    "ExperimentConfig",
    "LoadedRun",
    "RunPaths",
    "aggregate_rows",
    "condition_of",
    "environment_metadata",
    "load_run",
    "read_checkpoint_csv",
    "read_training_csv",
    "run_cell",
    "run_experiment",
    "save_run",
    "write_checkpoint_csv",
    "write_json",
    "write_policy_gifs",
    "write_training_csv",
]
