"""Mid-training greedy checkpoints: schedule, isolation, and persistence."""

from __future__ import annotations

import io
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from mars_rover_q.experiment import read_checkpoint_csv, write_checkpoint_csv
from mars_rover_q.metrics import CheckpointRecord
from mars_rover_q.rewards import RewardMode
from mars_rover_q.scenario import Scenario
from mars_rover_q.training import TrainConfig, checkpoint_schedule, train


def _config(scenario: str, **overrides: object) -> TrainConfig:
    base: dict[str, object] = {
        "scenario": scenario,
        "reward_mode": RewardMode.SPARSE,
        "seed": 3,
        "episodes": 60,
        "log_every": 0,
    }
    return TrainConfig(**{**base, **overrides})  # type: ignore[arg-type]


# -- schedule --------------------------------------------------------------


def test_schedule_is_empty_when_checkpointing_is_off() -> None:
    assert checkpoint_schedule(4000, 0) == ()
    assert checkpoint_schedule(0, 50) == ()


def test_schedule_starts_at_zero_and_is_evenly_spaced() -> None:
    points = checkpoint_schedule(20000, 50)
    assert points[0] == 0
    assert len(points) == 50
    gaps = {b - a for a, b in pairwise(points)}
    assert gaps == {400}


def test_schedule_never_reaches_the_episode_count() -> None:
    """`train` adds the final checkpoint itself; the schedule must not duplicate it."""
    assert max(checkpoint_schedule(1000, 50)) < 1000


def test_schedule_degrades_when_checkpoints_outnumber_episodes() -> None:
    points = checkpoint_schedule(10, 50)
    assert points == tuple(range(10))


# -- isolation -------------------------------------------------------------


@pytest.mark.parametrize("curriculum_fraction", [0.0, 0.5])
def test_checkpointing_does_not_perturb_training(
    tiny_scenario: Scenario, curriculum_fraction: float
) -> None:
    """The measurement must be free.

    Checkpoint evaluation runs in its own seed space on its own environment, so a
    run with checkpoints must learn exactly what the same run without them learns.
    If this fails, the sweep's numbers depend on whether it was being observed.
    """
    plain = train(
        tiny_scenario,
        _config(tiny_scenario.name, curriculum_fraction=curriculum_fraction),
        warn_on_stubs=False,
        stream=io.StringIO(),
    )
    watched = train(
        tiny_scenario,
        _config(
            tiny_scenario.name,
            curriculum_fraction=curriculum_fraction,
            eval_checkpoints=6,
            checkpoint_episodes=3,
        ),
        warn_on_stubs=False,
        stream=io.StringIO(),
    )
    assert np.array_equal(plain.q_table, watched.q_table)
    assert [r.as_dict() for r in plain.records] == [r.as_dict() for r in watched.records]
    assert plain.checkpoints == []
    assert watched.checkpoints


def test_checkpoints_span_the_run_and_end_on_the_final_table(tiny_scenario: Scenario) -> None:
    result = train(
        tiny_scenario,
        _config(tiny_scenario.name, eval_checkpoints=6, checkpoint_episodes=3),
        warn_on_stubs=False,
        stream=io.StringIO(),
    )
    episodes = [c.episode for c in result.checkpoints]
    assert episodes[0] == 0
    assert episodes[-1] == 60
    assert episodes == sorted(episodes)
    assert len(set(episodes)) == len(episodes)
    assert all(c.episodes == 3 for c in result.checkpoints)


def test_both_arms_are_measured_at_the_same_training_episodes(tiny_scenario: Scenario) -> None:
    """Paired at every point on the x-axis, not just at the end."""
    arms = [
        train(
            tiny_scenario,
            _config(
                tiny_scenario.name,
                curriculum_fraction=fraction,
                eval_checkpoints=6,
                checkpoint_episodes=3,
            ),
            warn_on_stubs=False,
            stream=io.StringIO(),
        )
        for fraction in (0.0, 0.5)
    ]
    assert [c.episode for c in arms[0].checkpoints] == [c.episode for c in arms[1].checkpoints]


def test_checkpoints_are_reproducible(tiny_scenario: Scenario) -> None:
    config = _config(tiny_scenario.name, eval_checkpoints=6, checkpoint_episodes=3)
    first = train(tiny_scenario, config, warn_on_stubs=False, stream=io.StringIO())
    second = train(tiny_scenario, config, warn_on_stubs=False, stream=io.StringIO())
    assert [c.as_dict() for c in first.checkpoints] == [c.as_dict() for c in second.checkpoints]


# -- persistence -----------------------------------------------------------


def test_checkpoint_csv_round_trips(tmp_path: Path) -> None:
    records = [
        CheckpointRecord(
            episode=0,
            env_steps=0,
            episodes=40,
            success_rate=0.0,
            mean_base_return=-100.0,
            mean_delivered_value=0.0,
            mean_steps=60.0,
        ),
        CheckpointRecord(
            episode=400,
            env_steps=12345,
            episodes=40,
            success_rate=0.75,
            mean_base_return=12.5,
            mean_delivered_value=40.0,
            mean_steps=31.25,
        ),
    ]
    path = tmp_path / "eval_checkpoints.csv"
    write_checkpoint_csv(path, records)
    assert read_checkpoint_csv(path) == records
