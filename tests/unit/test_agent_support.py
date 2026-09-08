"""The non-human-owned parts of the agent module, and the teaching-state probe.

These assertions must hold both before and after Connor implements the five
functions, so they never test the placeholder behaviour itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from mars_rover_q.actions import NUM_ACTIONS
from mars_rover_q.agent import (
    HUMAN_OWNED_FUNCTIONS,
    EpsilonSchedule,
    QLearningConfig,
    initialize_q_table,
    select_action,
    teaching_stub_status,
)


def test_epsilon_decays_linearly_then_flattens() -> None:
    schedule = EpsilonSchedule(start=1.0, end=0.1, decay_episodes=10)
    assert schedule.value_at(0) == pytest.approx(1.0)
    assert schedule.value_at(5) == pytest.approx(0.55)
    assert schedule.value_at(10) == pytest.approx(0.1)
    assert schedule.value_at(999) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 0.1, "end": 0.5},
        {"start": 1.5, "end": 0.1},
        {"start": 1.0, "end": -0.1},
        {"decay_episodes": 0},
    ],
)
def test_epsilon_schedule_validates_its_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        EpsilonSchedule(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(("lr", "gamma"), [(0.0, 0.9), (1.5, 0.9), (0.2, 0.0), (0.2, 1.5)])
def test_q_learning_config_validates_hyperparameters(lr: float, gamma: float) -> None:
    with pytest.raises(ValueError):
        QLearningConfig(learning_rate=lr, gamma=gamma)


def test_default_config_is_usable() -> None:
    config = QLearningConfig()
    assert 0.0 < config.learning_rate <= 1.0
    assert 0.0 < config.gamma <= 1.0


def test_stub_probe_only_ever_names_human_owned_functions() -> None:
    assert set(teaching_stub_status()) <= set(HUMAN_OWNED_FUNCTIONS)


def test_placeholders_are_compile_safe_so_the_rest_of_the_repo_runs() -> None:
    """Whether stubbed or implemented, these calls must not explode."""
    table = initialize_q_table(4, NUM_ACTIONS)
    assert table.shape == (4, NUM_ACTIONS)
    action = select_action(table, 0, 0.0, np.random.default_rng(0))
    assert 0 <= int(action) < NUM_ACTIONS
