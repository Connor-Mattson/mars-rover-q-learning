"""The training and evaluation loops, and run-directory serialisation.

These tests assert the *plumbing*: that the real learning path calls the
human-owned functions and that artefacts round-trip. They do not assert that
learning works, which is not yet true.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mars_rover_q import agent, evaluation, training
from mars_rover_q.evaluation import evaluate
from mars_rover_q.experiment import load_run, read_training_csv, save_run, write_training_csv
from mars_rover_q.rewards import RewardMode
from mars_rover_q.scenario import Scenario, resolve_scenario
from mars_rover_q.training import TrainConfig, split_rngs, train


@pytest.fixture
def quick_config() -> TrainConfig:
    return TrainConfig(
        scenario="safe_corridor",
        reward_mode=RewardMode.SPARSE,
        seed=1,
        episodes=3,
        log_every=0,
    )


@pytest.fixture
def corridor() -> Scenario:
    return resolve_scenario("safe_corridor")


def test_training_completes_and_records_every_episode(
    corridor: Scenario, quick_config: TrainConfig
) -> None:
    result = train(corridor, quick_config, warn_on_stubs=False, stream=io.StringIO())
    assert len(result.records) == quick_config.episodes
    assert [r.episode for r in result.records] == [0, 1, 2]
    assert result.total_env_steps > 0


def test_training_allocates_a_fresh_table_of_the_right_shape(
    corridor: Scenario, quick_config: TrainConfig
) -> None:
    result = train(corridor, quick_config, warn_on_stubs=False, stream=io.StringIO())
    expected_states = corridor.rows * corridor.cols * (corridor.battery_capacity + 1) * 4
    assert result.q_table.shape == (expected_states, 5)


def test_training_calls_every_human_owned_function(
    corridor: Scenario, quick_config: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards against the algorithm being quietly reimplemented somewhere else."""
    calls: dict[str, int] = dict.fromkeys(agent.HUMAN_OWNED_FUNCTIONS, 0)

    def counted(name: str) -> Any:
        original = getattr(training, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            return original(*args, **kwargs)

        return wrapper

    for name in agent.HUMAN_OWNED_FUNCTIONS:
        monkeypatch.setattr(training, name, counted(name))

    train(corridor, quick_config, warn_on_stubs=False, stream=io.StringIO())
    assert calls["initialize_q_table"] == 1
    for name in ("select_action", "calculate_target", "calculate_td_error", "update_q_value"):
        assert calls[name] > 0, f"{name} was never called by the training loop"


def test_training_is_reproducible_from_its_seed(
    corridor: Scenario, quick_config: TrainConfig
) -> None:
    first = train(corridor, quick_config, warn_on_stubs=False, stream=io.StringIO())
    second = train(corridor, quick_config, warn_on_stubs=False, stream=io.StringIO())
    assert [r.as_dict() for r in first.records] == [r.as_dict() for r in second.records]
    assert np.array_equal(first.q_table, second.q_table)


def test_seeds_derive_independent_reproducible_streams() -> None:
    """Environment and agent randomness come from separate, seed-reproducible streams."""
    env_rng_a, agent_rng_a = split_rngs(1)
    env_rng_b, agent_rng_b = split_rngs(1)
    env_rng_c, agent_rng_c = split_rngs(2)

    assert env_rng_a.random(8).tolist() == env_rng_b.random(8).tolist()
    assert agent_rng_a.random(8).tolist() == agent_rng_b.random(8).tolist()
    assert env_rng_a.random(8).tolist() != env_rng_c.random(8).tolist()
    assert agent_rng_a.random(8).tolist() != agent_rng_c.random(8).tolist()

    fresh_env, fresh_agent = split_rngs(1)
    assert fresh_env.random(8).tolist() != fresh_agent.random(8).tolist()


def test_training_warns_conspicuously_while_the_functions_are_stubs(
    corridor: Scenario, quick_config: TrainConfig
) -> None:
    stream = io.StringIO()
    result = train(corridor, quick_config, warn_on_stubs=True, stream=stream)
    if result.pending_human_functions:
        text = stream.getvalue()
        assert "TEACHING STATE" in text
        assert "NO LEARNING IS HAPPENING" in text
        assert not result.learning_is_meaningful
    else:
        assert result.learning_is_meaningful


def test_evaluation_disables_exploration(
    corridor: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float] = []
    original = evaluation.select_action

    def spy(q_table: Any, state: int, epsilon: float, rng: Any, mask: Any = None) -> int:
        seen.append(epsilon)
        return original(q_table, state, epsilon, rng, mask)

    monkeypatch.setattr(evaluation, "select_action", spy)
    table = np.zeros((corridor.rows * corridor.cols * (corridor.battery_capacity + 1) * 4, 5))
    evaluate(table, corridor, RewardMode.SPARSE, episodes=2, seed=1)
    assert seen
    assert set(seen) == {0.0}


def test_evaluation_summarises_and_captures_a_best_trajectory(corridor: Scenario) -> None:
    table = np.zeros((corridor.rows * corridor.cols * (corridor.battery_capacity + 1) * 4, 5))
    result = evaluate(table, corridor, RewardMode.SPARSE, episodes=3, seed=1)
    assert result.summary.episodes == 3
    assert result.best_trajectory is not None
    trajectory = result.best_trajectory
    assert len(trajectory.states) == len(trajectory.actions) + 1
    assert trajectory.outcome in {"success", "battery_depleted", "step_limit"}


def test_evaluation_rejects_a_nonpositive_episode_count(corridor: Scenario) -> None:
    table = np.zeros((10, 5))
    with pytest.raises(ValueError, match="episodes"):
        evaluate(table, corridor, RewardMode.SPARSE, episodes=0)


def test_training_metrics_round_trip_through_csv(
    tmp_path: Path, corridor: Scenario, quick_config: TrainConfig
) -> None:
    result = train(corridor, quick_config, warn_on_stubs=False, stream=io.StringIO())
    path = tmp_path / "metrics.csv"
    write_training_csv(path, result.records)
    assert read_training_csv(path) == result.records


def test_run_directory_round_trips(
    tmp_path: Path, corridor: Scenario, quick_config: TrainConfig
) -> None:
    result = train(corridor, quick_config, warn_on_stubs=False, stream=io.StringIO())
    evaluated = evaluate(result.q_table, corridor, RewardMode.SPARSE, episodes=2, seed=1)
    run_dir = tmp_path / "run"
    save_run(run_dir, corridor, result, evaluated)

    for name in (
        "manifest.json",
        "q_table.npy",
        "policy.npy",
        "training_metrics.csv",
        "evaluation.json",
        "best_episode.json",
        "scenario.json",
    ):
        assert (run_dir / name).exists(), name

    loaded = load_run(run_dir)
    assert loaded.scenario.name == corridor.name
    assert np.array_equal(loaded.q_table, result.q_table)
    assert loaded.records == result.records
    assert loaded.best_trajectory is not None
    assert loaded.policy.shape == (result.q_table.shape[0],)


def test_manifest_records_the_teaching_state_and_provenance(
    tmp_path: Path, corridor: Scenario, quick_config: TrainConfig
) -> None:
    result = train(corridor, quick_config, warn_on_stubs=False, stream=io.StringIO())
    save_run(tmp_path / "run", corridor, result, None)
    manifest = load_run(tmp_path / "run").manifest
    assert manifest["scenario"] == "safe_corridor"
    assert manifest["config"]["seed"] == 1
    assert "numpy" in manifest["environment"]
    assert manifest["learning_is_meaningful"] == result.learning_is_meaningful
    assert manifest["pending_human_functions"] == list(result.pending_human_functions)


def test_load_run_reports_a_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_run(tmp_path)
