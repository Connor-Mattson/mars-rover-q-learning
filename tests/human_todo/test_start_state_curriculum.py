"""Contract tests for the two human-owned start-state curriculum functions.

These are EXPECTED TO FAIL until ``src/mars_rover_q/curriculum.py`` is implemented.
They are deselected from the default ``pytest`` run and are executed with::

    pytest -m human_todo tests/human_todo

Do not skip, weaken, xfail, or delete them. Every failure here should be a plain
assertion failure caused by a placeholder, never an import or fixture error.

The two functions are tested independently: the ``sample_start_state`` cases build
their own ranked pool rather than calling ``enumerate_start_states``, so finishing
one function turns its tests green without waiting for the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from mars_rover_q.curriculum import (
    canonical_start_state,
    enumerate_start_states,
    sample_start_state,
    start_state_difficulty,
)
from mars_rover_q.scenario import Scenario
from mars_rover_q.state import RoverState, SampleType, StateEncoder

pytestmark = pytest.mark.human_todo


# The tiny fixture is a 5x5 sandbox with unit energy costs and no slip, so every
# distance below is just a step count:
#
#     #####      lander    (1, 1)     battery capacity 20
#     #...#      basalt    (1, 3)     collect cost     1
#     #.#.#      hydrated  (3, 1)
#     #...#      biosig    (3, 3)
#     #####
#
# Cheapest route lander -> basalt is 2, basalt -> lander is 2, lander -> biosig is 4.


# -- enumerate_start_states -----------------------------------------------


def test_pool_is_not_empty(tiny_scenario: Scenario) -> None:
    assert enumerate_start_states(tiny_scenario)


def test_pool_contains_the_canonical_start(tiny_scenario: Scenario) -> None:
    """The real mission start is itself a physically reachable state."""
    assert canonical_start_state(tiny_scenario) in enumerate_start_states(tiny_scenario)


def test_pool_has_no_duplicates(tiny_scenario: Scenario) -> None:
    pool = enumerate_start_states(tiny_scenario)
    assert len(set(pool)) == len(pool)


def test_pool_states_all_encode(tiny_scenario: Scenario) -> None:
    """Every start state must be a real Q-table row, not an out-of-range triple."""
    encoder = StateEncoder(tiny_scenario.rows, tiny_scenario.cols, tiny_scenario.battery_capacity)
    for state in enumerate_start_states(tiny_scenario):
        encoder.encode(state)


def test_pool_excludes_walls_and_out_of_bounds(tiny_scenario: Scenario) -> None:
    for state in enumerate_start_states(tiny_scenario):
        assert tiny_scenario.is_traversable(state.position), state


def test_pool_excludes_a_flat_battery(tiny_scenario: Scenario) -> None:
    """A battery-depleted state is already terminal; an episode cannot begin there."""
    assert all(state.battery > 0 for state in enumerate_start_states(tiny_scenario))


def test_pool_excludes_a_completed_delivery(tiny_scenario: Scenario) -> None:
    """Carrying a sample on the lander is a success the environment scores at once."""
    pool = set(enumerate_start_states(tiny_scenario))
    assert RoverState(1, 1, 20, SampleType.BASALT) not in pool
    assert RoverState(1, 1, 5, SampleType.HYDRATED_MINERAL) not in pool


def test_pool_excludes_a_start_the_rover_could_not_have_driven_to(
    tiny_scenario: Scenario,
) -> None:
    """A full battery four tiles from the lander accounts for no energy spent."""
    pool = set(enumerate_start_states(tiny_scenario))
    assert RoverState(3, 3, 20, SampleType.NONE) not in pool
    assert RoverState(3, 3, 16, SampleType.NONE) in pool


def test_pool_charges_the_collect_and_the_detour_via_the_sample(
    tiny_scenario: Scenario,
) -> None:
    """Carrying basalt at its own tile costs 2 to drive there plus 1 to collect."""
    pool = set(enumerate_start_states(tiny_scenario))
    assert RoverState(1, 3, 18, SampleType.BASALT) not in pool
    assert RoverState(1, 3, 17, SampleType.BASALT) in pool


def test_pool_charges_the_return_leg_from_the_sample(tiny_scenario: Scenario) -> None:
    """Carrying basalt at (3, 3) means 2 to the sample, 1 to collect, 2 onward."""
    pool = set(enumerate_start_states(tiny_scenario))
    assert RoverState(3, 3, 16, SampleType.BASALT) not in pool
    assert RoverState(3, 3, 15, SampleType.BASALT) in pool


def test_pool_excludes_starts_that_cannot_reach_the_lander(
    tiny_scenario: Scenario,
) -> None:
    """One joule two tiles from home can only ever teach a battery failure."""
    pool = set(enumerate_start_states(tiny_scenario))
    assert RoverState(1, 3, 1, SampleType.BASALT) not in pool
    assert RoverState(1, 2, 1, SampleType.BASALT) in pool


def test_every_pooled_start_can_still_finish_the_mission(tiny_scenario: Scenario) -> None:
    for state in enumerate_start_states(tiny_scenario):
        assert start_state_difficulty(tiny_scenario, state) <= state.battery, state


def test_pool_is_built_for_every_bundled_scenario(bundled_scenario: Scenario) -> None:
    """The real maps are bigger and walled; the filter must survive them too."""
    pool = enumerate_start_states(bundled_scenario)
    assert canonical_start_state(bundled_scenario) in pool
    assert len(set(pool)) == len(pool)
    for state in pool:
        assert bundled_scenario.is_traversable(state.position)
        assert 0 < state.battery <= bundled_scenario.battery_capacity


# -- sample_start_state ---------------------------------------------------


def _ranked_pool(scenario: Scenario) -> list[RoverState]:
    """A hand-built pool sorted easiest-first; the sampler only sees the ordering."""
    states = [
        RoverState(row, col, battery, carried)
        for (row, col) in ((1, 2), (1, 3), (3, 3), (3, 1))
        for carried in (SampleType.BASALT, SampleType.NONE)
        for battery in range(1, 17)
    ]
    return sorted(states, key=lambda s: (start_state_difficulty(scenario, s), s.battery))


def test_the_anneal_ends_on_the_canonical_start(tiny_scenario: Scenario) -> None:
    """Training has to finish on the distribution it is evaluated on."""
    pool = _ranked_pool(tiny_scenario)
    canonical = canonical_start_state(tiny_scenario)
    rng = np.random.default_rng(0)
    assert all(sample_start_state(pool, canonical, 1.0, rng) == canonical for _ in range(50))


def test_every_draw_comes_from_the_pool_or_is_canonical(tiny_scenario: Scenario) -> None:
    pool = _ranked_pool(tiny_scenario)
    canonical = canonical_start_state(tiny_scenario)
    rng = np.random.default_rng(1)
    allowed = {*pool, canonical}
    for progress in (0.0, 0.3, 0.6, 0.9):
        for _ in range(50):
            assert sample_start_state(pool, canonical, progress, rng) in allowed


def test_the_anneal_starts_at_the_easy_end(tiny_scenario: Scenario) -> None:
    """At zero progress the support must not reach past the easiest quarter."""
    pool = _ranked_pool(tiny_scenario)
    canonical = canonical_start_state(tiny_scenario)
    rng = np.random.default_rng(2)
    ranks = [pool.index(sample_start_state(pool, canonical, 0.0, rng)) for _ in range(200)]
    assert max(ranks) < 0.25 * len(pool)


def test_the_support_widens_as_progress_grows(tiny_scenario: Scenario) -> None:
    pool = _ranked_pool(tiny_scenario)
    canonical = canonical_start_state(tiny_scenario)
    rng = np.random.default_rng(3)

    def hardest_rank(progress: float) -> int:
        # The canonical start is the hardest thing the sampler can return, so it
        # ranks just past the end of the pool.
        draws = [sample_start_state(pool, canonical, progress, rng) for _ in range(200)]
        return max(pool.index(s) if s in pool else len(pool) for s in draws)

    early, middle, late = hardest_rank(0.1), hardest_rank(0.5), hardest_rank(0.9)
    assert early <= middle <= late
    assert early < late, "the support must actually reach harder states as training runs"


def test_mid_anneal_draws_are_not_all_the_same_state(tiny_scenario: Scenario) -> None:
    """A curriculum that returns one state is a fixed start, not a distribution."""
    pool = _ranked_pool(tiny_scenario)
    canonical = canonical_start_state(tiny_scenario)
    rng = np.random.default_rng(4)
    draws = {sample_start_state(pool, canonical, 0.5, rng) for _ in range(200)}
    assert len(draws) > 1


def test_an_empty_pool_degrades_to_the_canonical_start(tiny_scenario: Scenario) -> None:
    """Nothing to offer must mean ordinary training, not a crash."""
    canonical = canonical_start_state(tiny_scenario)
    rng = np.random.default_rng(5)
    assert sample_start_state([], canonical, 0.0, rng) == canonical
    assert sample_start_state([], canonical, 0.5, rng) == canonical


@pytest.mark.parametrize("progress", [-0.1, -1.0, 1.01, 2.0])
def test_progress_outside_the_unit_interval_is_rejected(
    tiny_scenario: Scenario, progress: float
) -> None:
    pool = _ranked_pool(tiny_scenario)
    canonical = canonical_start_state(tiny_scenario)
    with pytest.raises(ValueError):
        sample_start_state(pool, canonical, progress, np.random.default_rng(6))


def test_sampling_is_reproducible_from_the_injected_generator(
    tiny_scenario: Scenario,
) -> None:
    """Every draw must come from ``rng``; a module-level draw breaks this."""
    pool = _ranked_pool(tiny_scenario)
    canonical = canonical_start_state(tiny_scenario)
    first = [sample_start_state(pool, canonical, 0.4, np.random.default_rng(7)) for _ in range(20)]
    second = [sample_start_state(pool, canonical, 0.4, np.random.default_rng(7)) for _ in range(20)]
    assert first == second

    shared = np.random.default_rng(8)
    streamed = [sample_start_state(pool, canonical, 0.4, shared) for _ in range(200)]
    assert len(set(streamed)) > 1, "successive draws from one generator must vary"


def test_sampling_does_not_reorder_or_mutate_the_pool(tiny_scenario: Scenario) -> None:
    pool = _ranked_pool(tiny_scenario)
    canonical = canonical_start_state(tiny_scenario)
    before = list(pool)
    rng = np.random.default_rng(9)
    for progress in (0.0, 0.5, 1.0):
        for _ in range(20):
            sample_start_state(pool, canonical, progress, rng)
    assert pool == before
