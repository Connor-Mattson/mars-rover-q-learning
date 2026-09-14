"""The non-human-owned parts of the start-state curriculum.

These assertions must hold both before and after Connor implements
``enumerate_start_states`` and ``sample_start_state``, so they never test the
placeholder behaviour itself -- they test the difficulty metric, the schedule
arithmetic, and the guarantees the wrapper makes regardless of the pool.
"""

from __future__ import annotations

import numpy as np
import pytest

from mars_rover_q.curriculum import (
    StartStateCurriculum,
    canonical_start_state,
    start_state_difficulty,
)
from mars_rover_q.scenario import UNREACHABLE, Scenario
from mars_rover_q.state import RoverState, SampleType


def test_canonical_start_is_the_lander_on_a_full_battery(tiny_scenario: Scenario) -> None:
    state = canonical_start_state(tiny_scenario)
    assert state.position == tiny_scenario.lander
    assert state.battery == tiny_scenario.battery_capacity
    assert state.carried is SampleType.NONE


def test_difficulty_of_a_carried_sample_is_the_drive_home(tiny_scenario: Scenario) -> None:
    # (1, 2) is one unit-cost tile from the lander at (1, 1).
    assert start_state_difficulty(tiny_scenario, RoverState(1, 2, 9, SampleType.BASALT)) == 1.0
    assert start_state_difficulty(tiny_scenario, RoverState(1, 3, 9, SampleType.BASALT)) == 2.0


def test_difficulty_empty_handed_is_the_cheapest_collect_and_return(
    tiny_scenario: Scenario,
) -> None:
    """From the lander: 2 out to the basalt, 1 to collect, 2 back."""
    assert start_state_difficulty(tiny_scenario, canonical_start_state(tiny_scenario)) == 5.0


def test_difficulty_ignores_which_sample_is_already_carried(tiny_scenario: Scenario) -> None:
    """Carrying anything means the mission is now just the drive home."""
    for carried in (SampleType.BASALT, SampleType.HYDRATED_MINERAL, SampleType.BIOSIGNATURE):
        assert start_state_difficulty(tiny_scenario, RoverState(3, 3, 9, carried)) == 4.0


def test_carrying_is_never_harder_than_being_empty_handed(bundled_scenario: Scenario) -> None:
    """The ordering the curriculum relies on: a full sample bay is progress."""
    for cell in bundled_scenario.traversable_cells():
        empty = start_state_difficulty(
            bundled_scenario, RoverState(cell[0], cell[1], 10, SampleType.NONE)
        )
        carried = start_state_difficulty(
            bundled_scenario, RoverState(cell[0], cell[1], 10, SampleType.BASALT)
        )
        assert carried <= empty, cell


def test_difficulty_is_finite_everywhere_on_a_validated_map(
    bundled_scenario: Scenario,
) -> None:
    """Scenario validation guarantees connectivity, so the sentinel never leaks out."""
    for cell in bundled_scenario.traversable_cells():
        state = RoverState(cell[0], cell[1], 10, SampleType.NONE)
        assert start_state_difficulty(bundled_scenario, state) < UNREACHABLE


def test_a_disabled_curriculum_never_leaves_the_lander(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=100, anneal_fraction=0.0)
    rng = np.random.default_rng(0)
    assert not curriculum.enabled
    assert not curriculum.active
    assert curriculum.ranked_pool == ()
    assert all(
        curriculum.start_state_for(episode, rng) == curriculum.canonical for episode in (0, 50, 99)
    )


def test_a_disabled_curriculum_skips_pool_construction(tiny_scenario: Scenario) -> None:
    """Enumeration is the expensive part; an unconfigured run must not pay it."""
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=100, anneal_fraction=0.0)
    assert curriculum.describe()["pool_size"] == 0
    assert curriculum.describe()["anneal_episodes"] == 0


def test_progress_runs_from_zero_to_one_over_the_anneal(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=100, anneal_fraction=0.5)
    assert curriculum.anneal_episodes == 50
    assert curriculum.progress_at(0) == 0.0
    assert curriculum.progress_at(25) == pytest.approx(0.5)
    assert curriculum.progress_at(50) == 1.0
    assert curriculum.progress_at(99) == 1.0


def test_progress_is_one_when_the_curriculum_is_disabled(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=100, anneal_fraction=0.0)
    assert curriculum.progress_at(0) == 1.0


def test_the_anneal_is_at_least_one_episode_long(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=10, anneal_fraction=0.001)
    assert curriculum.anneal_episodes == 1
    assert curriculum.progress_at(1) == 1.0


@pytest.mark.parametrize(
    ("total_episodes", "anneal_fraction"),
    [(0, 0.5), (-1, 0.5), (100, -0.1), (100, 1.5)],
)
def test_the_curriculum_validates_its_arguments(
    tiny_scenario: Scenario, total_episodes: int, anneal_fraction: float
) -> None:
    with pytest.raises(ValueError):
        StartStateCurriculum(
            tiny_scenario, total_episodes=total_episodes, anneal_fraction=anneal_fraction
        )


def test_describe_is_json_serialisable(tiny_scenario: Scenario) -> None:
    import json

    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=100, anneal_fraction=0.5)
    payload = json.loads(json.dumps(curriculum.describe()))
    assert payload["requested"] is True
    assert payload["anneal_fraction"] == 0.5
    assert payload["canonical_difficulty"] == 5.0
    assert payload["active"] == bool(payload["pool_size"])


def test_the_pool_is_sorted_easiest_first(tiny_scenario: Scenario) -> None:
    """Holds vacuously while the pool is a placeholder, and is the contract after."""
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=100, anneal_fraction=0.5)
    scores = [start_state_difficulty(tiny_scenario, s) for s in curriculum.ranked_pool]
    assert scores == sorted(scores)
