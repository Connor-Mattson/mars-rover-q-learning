"""Contract tests for the human-owned visit-weighted start-state sampler.

These are EXPECTED TO FAIL until ``sample_visit_weighted_start_state`` in
``src/mars_rover_q/curriculum.py`` is implemented. They are deselected from the
default ``pytest`` run and are executed with::

    pytest -m human_todo tests/human_todo

Do not skip, weaken, xfail, or delete them. Every failure here should be a plain
assertion failure caused by the placeholder, never an import or fixture error.

The pool below is synthetic on purpose: the sampler is a function of an *ordering*
and a count vector, and nothing it does depends on the states being reachable, so
these tests never call ``enumerate_start_states``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pytest

from mars_rover_q.curriculum import (
    growing_window_bounds,
    sample_visit_weighted_start_state,
)
from mars_rover_q.state import RoverState, SampleType

pytestmark = pytest.mark.human_todo

#: Eight ranked states standing in for a difficulty-sorted pool. Rank is the only
#: thing the sampler reads; the field values exist so that the states are distinct.
POOL: list[RoverState] = [RoverState(1, 1, rank + 1, SampleType.BASALT) for rank in range(8)]
CANONICAL = RoverState(3, 3, 20, SampleType.NONE)

#: Rank lookup that survives a draw from outside the pool -- the placeholder returns
#: the canonical start, and a test should report that as a failed assertion rather
#: than as a ``ValueError`` from ``list.index``.
RANK_OF: dict[RoverState, int] = {state: rank for rank, state in enumerate(POOL)}


def _zero_counts() -> np.ndarray:
    return np.zeros(len(POOL), dtype=np.int64)


def _draw(
    progress: float,
    counts: Sequence[int] | np.ndarray,
    *,
    draws: int = 2000,
    seed: int = 0,
    exponent: float = 1.0,
) -> Counter[RoverState]:
    """Frequency of each drawn state over one generator's stream."""
    rng = np.random.default_rng(seed)
    return Counter(
        sample_visit_weighted_start_state(POOL, CANONICAL, progress, rng, counts, exponent=exponent)
        for _ in range(draws)
    )


# -- the boundaries it shares with the other samplers ----------------------


def test_the_anneal_ends_on_the_canonical_start() -> None:
    rng = np.random.default_rng(0)
    assert all(
        sample_visit_weighted_start_state(POOL, CANONICAL, 1.0, rng, _zero_counts()) == CANONICAL
        for _ in range(20)
    )


def test_an_empty_pool_degrades_to_the_canonical_start() -> None:
    """Nothing to offer must mean ordinary training, not a crash."""
    rng = np.random.default_rng(0)
    empty = np.zeros(0, dtype=np.int64)
    assert sample_visit_weighted_start_state([], CANONICAL, 0.0, rng, empty) == CANONICAL
    assert sample_visit_weighted_start_state([], CANONICAL, 0.5, rng, empty) == CANONICAL


@pytest.mark.parametrize("progress", [-0.1, -1.0, 1.01, 2.0])
def test_progress_outside_the_unit_interval_is_rejected(progress: float) -> None:
    with pytest.raises(ValueError):
        sample_visit_weighted_start_state(
            POOL, CANONICAL, progress, np.random.default_rng(0), _zero_counts()
        )


@pytest.mark.parametrize("exponent", [-0.5, -1.0])
def test_a_negative_exponent_is_rejected(exponent: float) -> None:
    """Negative would tilt *toward* the states that need the episodes least."""
    with pytest.raises(ValueError):
        sample_visit_weighted_start_state(
            POOL, CANONICAL, 0.5, np.random.default_rng(0), _zero_counts(), exponent=exponent
        )


@pytest.mark.parametrize("counts", [[0] * 4, [0] * 12])
def test_counts_of_the_wrong_length_are_rejected(counts: list[int]) -> None:
    """A misaligned count vector silently reweights the wrong states."""
    with pytest.raises(ValueError):
        sample_visit_weighted_start_state(POOL, CANONICAL, 0.5, np.random.default_rng(0), counts)


def test_negative_counts_are_rejected() -> None:
    counts = _zero_counts()
    counts[0] = -1
    with pytest.raises(ValueError):
        sample_visit_weighted_start_state(POOL, CANONICAL, 0.5, np.random.default_rng(0), counts)


# -- the admission window --------------------------------------------------


@pytest.mark.parametrize("progress", [0.0, 0.13, 0.25, 0.5, 0.75, 0.99])
def test_the_admitted_band_is_exactly_the_growing_window(progress: float) -> None:
    """Same admission rule as ``sample_start_state``; only the weights differ."""
    low, high = growing_window_bounds(len(POOL), progress)
    drawn = _draw(progress, _zero_counts(), draws=300, seed=1)
    ranks = {RANK_OF.get(state, -1) for state in drawn}
    assert ranks <= set(range(low, high))


def test_the_window_widens_with_progress() -> None:
    early = {RANK_OF.get(s, -1) for s in _draw(0.25, _zero_counts(), draws=300, seed=2)}
    late = {RANK_OF.get(s, -1) for s in _draw(0.9, _zero_counts(), draws=300, seed=2)}
    assert max(early) < max(late)


# -- the weighting itself --------------------------------------------------


def test_untouched_states_are_drawn_about_uniformly() -> None:
    """Before any learning has happened there is nothing to tilt away from."""
    counts = _draw(0.5, _zero_counts(), draws=4000, seed=3)
    admitted = [POOL[rank] for rank in range(*growing_window_bounds(len(POOL), 0.5))]
    assert set(counts) == set(admitted)
    frequencies = [counts[state] for state in admitted]
    assert min(frequencies) > 0.5 * max(frequencies)


def test_a_zero_exponent_reproduces_the_uniform_draw() -> None:
    """The control for the weighting: same window, no tilt, whatever the counts."""
    counts = np.array([0, 500, 5000, 50_000, 0, 0, 0, 0], dtype=np.int64)
    drawn = _draw(0.5, counts, draws=4000, seed=4, exponent=0.0)
    frequencies = [drawn[POOL[rank]] for rank in range(4)]
    assert min(frequencies) > 0.5 * max(frequencies)


def test_a_well_learned_state_is_drawn_less_than_an_untouched_one() -> None:
    counts = np.array([100, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    drawn = _draw(0.25, counts, draws=2000, seed=5)
    assert drawn[POOL[0]] < drawn[POOL[1]]


def test_draw_frequency_falls_as_experience_rises() -> None:
    """The whole point: episodes follow the states that still have work to do."""
    counts = np.array([0, 9, 99, 999, 0, 0, 0, 0], dtype=np.int64)
    drawn = _draw(0.5, counts, draws=4000, seed=6)
    frequencies = [drawn[POOL[rank]] for rank in range(4)]
    assert frequencies == sorted(frequencies, reverse=True)
    assert frequencies[0] > 4 * frequencies[1]


def test_a_heavily_updated_state_is_never_starved_outright() -> None:
    """A state that can never be drawn again is one whose value can never be fixed.

    The tilt therefore has to fall as a *power* of the count rather than
    exponentially: at a hundred updates the state is rare, roughly one episode in a
    hundred, not absent.
    """
    counts = np.array([0, 100, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    drawn = _draw(0.25, counts, draws=2000, seed=7)
    assert drawn[POOL[1]] > 0


def test_a_plain_list_of_counts_is_accepted() -> None:
    """The trainer passes an array; the diagnostics and tests pass sequences."""
    drawn = _draw(0.5, [0, 0, 0, 900, 0, 0, 0, 0], draws=1000, seed=8)
    assert drawn[POOL[3]] < drawn[POOL[0]]


# -- determinism and purity ------------------------------------------------


def test_sampling_is_reproducible_from_the_injected_generator() -> None:
    counts = np.array([0, 5, 50, 0, 0, 0, 0, 0], dtype=np.int64)
    first = [
        sample_visit_weighted_start_state(POOL, CANONICAL, 0.6, np.random.default_rng(9), counts)
        for _ in range(20)
    ]
    second = [
        sample_visit_weighted_start_state(POOL, CANONICAL, 0.6, np.random.default_rng(9), counts)
        for _ in range(20)
    ]
    assert first == second

    shared = np.random.default_rng(10)
    streamed = [
        sample_visit_weighted_start_state(POOL, CANONICAL, 0.6, shared, counts) for _ in range(200)
    ]
    assert len(set(streamed)) > 1, "successive draws from one generator must vary"


def test_sampling_mutates_neither_the_pool_nor_the_counts() -> None:
    counts = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64)
    pool_before = list(POOL)
    counts_before = counts.copy()
    rng = np.random.default_rng(11)
    for progress in (0.0, 0.5, 1.0):
        for _ in range(20):
            sample_visit_weighted_start_state(POOL, CANONICAL, progress, rng, counts)
    assert pool_before == POOL
    assert np.array_equal(counts, counts_before)
