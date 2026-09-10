"""The sliding-window and visit-weighted machinery around Connor's sampler.

The window arithmetic, the dispatch, the count plumbing and the dry-run
diagnostic. The visit-weighted draw itself is covered by
``tests/human_todo/test_visit_weighted_curriculum.py``; the only assertion here
that reaches it is the stub probe, which must now come back empty.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mars_rover_q.curriculum import (
    CurriculumStrategy,
    StartStateCurriculum,
    curriculum_arm_label,
    curriculum_arm_slug,
    curriculum_stub_status,
    growing_window_bounds,
    rank_band_shares,
    sample_sliding_window_start_state,
    simulate_start_distribution,
    sliding_window_bounds,
    start_state_difficulty,
)
from mars_rover_q.scenario import Scenario
from mars_rover_q.state import RoverState, SampleType, StateEncoder

POOL = [RoverState(1, 1, rank + 1, SampleType.BASALT) for rank in range(20)]
CANONICAL = RoverState(3, 3, 20, SampleType.NONE)


# -- window arithmetic -----------------------------------------------------


@pytest.mark.parametrize(
    ("progress", "expected"),
    [(0.0, (0, 1)), (0.25, (0, 5)), (0.5, (0, 10)), (1.0, (0, 20))],
)
def test_the_growing_window_is_anchored_at_the_easy_end(
    progress: float, expected: tuple[int, int]
) -> None:
    assert growing_window_bounds(20, progress) == expected


def test_the_growing_window_of_an_empty_pool_is_empty() -> None:
    assert growing_window_bounds(0, 0.5) == (0, 0)


def test_the_sliding_window_keeps_its_width_and_moves_its_edges() -> None:
    widths = set()
    lows = []
    for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
        low, high = sliding_window_bounds(20, progress, 0.25)
        widths.add(high - low)
        lows.append(low)
    assert widths == {5}
    assert lows == sorted(lows)
    assert lows[0] == 0
    assert lows[-1] == 15


def test_the_sliding_window_retires_the_easy_end() -> None:
    """The difference from the growing window, stated as an assertion."""
    early = sliding_window_bounds(20, 0.0, 0.25)
    late = sliding_window_bounds(20, 1.0, 0.25)
    assert late[0] >= early[1]


def test_the_sliding_window_never_runs_off_a_small_pool() -> None:
    for pool_size in (1, 2, 3):
        for progress in (0.0, 0.5, 1.0):
            low, high = sliding_window_bounds(pool_size, progress, 0.25)
            assert 0 <= low < high <= pool_size


def test_a_full_width_sliding_window_is_the_whole_pool() -> None:
    assert sliding_window_bounds(20, 0.7, 1.0) == (0, 20)


@pytest.mark.parametrize("width", [0.0, -0.1, 1.5])
def test_the_sliding_window_validates_its_width(width: float) -> None:
    with pytest.raises(ValueError):
        sliding_window_bounds(20, 0.5, width)


# -- the sliding sampler ---------------------------------------------------


def test_the_sliding_anneal_ends_on_the_canonical_start() -> None:
    rng = np.random.default_rng(0)
    assert all(
        sample_sliding_window_start_state(POOL, CANONICAL, 1.0, rng) == CANONICAL for _ in range(20)
    )


def test_an_empty_pool_degrades_to_the_canonical_start() -> None:
    rng = np.random.default_rng(0)
    assert sample_sliding_window_start_state([], CANONICAL, 0.4, rng) == CANONICAL


@pytest.mark.parametrize("progress", [-0.1, 1.01])
def test_the_sliding_sampler_validates_progress(progress: float) -> None:
    with pytest.raises(ValueError):
        sample_sliding_window_start_state(POOL, CANONICAL, progress, np.random.default_rng(0))


@pytest.mark.parametrize("progress", [0.0, 0.3, 0.6, 0.9])
def test_every_sliding_draw_lands_inside_its_band(progress: float) -> None:
    low, high = sliding_window_bounds(len(POOL), progress, 0.25)
    rng = np.random.default_rng(1)
    ranks = {
        POOL.index(sample_sliding_window_start_state(POOL, CANONICAL, progress, rng))
        for _ in range(200)
    }
    assert ranks <= set(range(low, high))
    assert len(ranks) > 1


def test_the_sliding_band_leaves_the_easy_end_behind() -> None:
    rng = np.random.default_rng(2)
    early = [
        POOL.index(sample_sliding_window_start_state(POOL, CANONICAL, 0.0, rng)) for _ in range(200)
    ]
    late = [
        POOL.index(sample_sliding_window_start_state(POOL, CANONICAL, 0.9, rng)) for _ in range(200)
    ]
    assert max(early) < 0.25 * len(POOL)
    assert min(late) > max(early)


def test_sliding_draws_are_reproducible_and_leave_the_pool_alone() -> None:
    before = list(POOL)
    first = [
        sample_sliding_window_start_state(POOL, CANONICAL, 0.4, np.random.default_rng(3))
        for _ in range(20)
    ]
    second = [
        sample_sliding_window_start_state(POOL, CANONICAL, 0.4, np.random.default_rng(3))
        for _ in range(20)
    ]
    assert first == second
    assert before == POOL


# -- the curriculum wrapper ------------------------------------------------


def test_every_strategy_builds_the_same_pool(tiny_scenario: Scenario) -> None:
    """They differ in how the pool is drawn from, never in what is admissible."""
    pools = {
        strategy: StartStateCurriculum(
            tiny_scenario, total_episodes=100, anneal_fraction=0.5, strategy=strategy
        ).ranked_pool
        for strategy in CurriculumStrategy
    }
    assert len(set(pools.values())) == 1


def test_an_unknown_strategy_is_rejected(tiny_scenario: Scenario) -> None:
    with pytest.raises(ValueError):
        StartStateCurriculum(
            tiny_scenario, total_episodes=100, anneal_fraction=0.5, strategy="anti_curriculum"
        )


@pytest.mark.parametrize(
    ("window_fraction", "weight_exponent"), [(0.0, 1.0), (1.5, 1.0), (0.25, -1.0)]
)
def test_the_strategy_knobs_are_validated(
    tiny_scenario: Scenario, window_fraction: float, weight_exponent: float
) -> None:
    with pytest.raises(ValueError):
        StartStateCurriculum(
            tiny_scenario,
            total_episodes=100,
            anneal_fraction=0.5,
            window_fraction=window_fraction,
            weight_exponent=weight_exponent,
        )


def test_describe_records_the_strategy(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(
        tiny_scenario, total_episodes=100, anneal_fraction=0.5, strategy="sliding"
    )
    payload = json.loads(json.dumps(curriculum.describe()))
    assert payload["strategy"] == "sliding"
    assert payload["window_fraction"] == 0.25
    assert payload["weight_exponent"] == 1.0


def test_only_the_visit_weighted_strategy_reads_the_table(tiny_scenario: Scenario) -> None:
    """The open-loop schedules must not be charged for the gather."""
    for strategy in CurriculumStrategy:
        curriculum = StartStateCurriculum(
            tiny_scenario, total_episodes=100, anneal_fraction=0.5, strategy=strategy
        )
        assert curriculum.needs_update_counts == (strategy is CurriculumStrategy.VISIT_WEIGHTED)


def test_pool_update_counts_gathers_in_pool_order(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=100, anneal_fraction=0.5)
    encoder = StateEncoder(tiny_scenario.rows, tiny_scenario.cols, tiny_scenario.battery_capacity)
    visits = np.zeros(encoder.num_states, dtype=np.int64)
    visits[encoder.encode(curriculum.ranked_pool[3])] = 7
    gathered = curriculum.pool_update_counts(visits)
    assert gathered.shape == (len(curriculum.ranked_pool),)
    assert gathered[3] == 7
    assert gathered.sum() == 7


def test_pool_update_counts_of_nothing_is_zeros(tiny_scenario: Scenario) -> None:
    """Episode 0 has no experience to report, which is not an error."""
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=100, anneal_fraction=0.5)
    assert not curriculum.pool_update_counts(None).any()


def test_the_sliding_strategy_dispatches_to_its_band(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(
        tiny_scenario, total_episodes=100, anneal_fraction=1.0, strategy="sliding"
    )
    low, high = sliding_window_bounds(len(curriculum.ranked_pool), 0.5, 0.25)
    rng = np.random.default_rng(4)
    ranks = {curriculum.ranked_pool.index(curriculum.start_state_for(50, rng)) for _ in range(100)}
    assert ranks <= set(range(low, high))


def test_a_disabled_curriculum_ignores_its_strategy(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(
        tiny_scenario, total_episodes=100, anneal_fraction=0.0, strategy="visit_weighted"
    )
    rng = np.random.default_rng(5)
    assert curriculum.start_state_for(0, rng) == curriculum.canonical


# -- the dry-run diagnostic ------------------------------------------------


def test_the_simulation_records_one_draw_per_episode(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(
        tiny_scenario, total_episodes=200, anneal_fraction=0.5, strategy="sliding"
    )
    anneal = simulate_start_distribution(curriculum, np.random.default_rng(6), episodes=120)
    assert anneal.ranks.shape == anneal.difficulties.shape == (120,)
    assert anneal.is_canonical.shape == (120,)
    assert (anneal.ranks < anneal.pool_size).all()
    # Every drawn state is pooled, so the synthetic counter accounts for every episode.
    assert anneal.update_counts.sum() == 120


def test_the_simulation_runs_the_anneal_by_default(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=200, anneal_fraction=0.5)
    anneal = simulate_start_distribution(curriculum, np.random.default_rng(7))
    assert anneal.ranks.size == curriculum.anneal_episodes == 100


def test_the_simulation_sees_the_canonical_tail(tiny_scenario: Scenario) -> None:
    """Past the anneal every episode starts at the lander, and the dry run says so."""
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=200, anneal_fraction=0.5)
    anneal = simulate_start_distribution(curriculum, np.random.default_rng(8), episodes=200)
    assert anneal.is_canonical[-1]
    assert anneal.canonical_share == pytest.approx(0.5, abs=0.01)


def test_the_simulation_is_reproducible(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(
        tiny_scenario, total_episodes=200, anneal_fraction=0.5, strategy="sliding"
    )
    first = simulate_start_distribution(curriculum, np.random.default_rng(9), episodes=50)
    second = simulate_start_distribution(curriculum, np.random.default_rng(9), episodes=50)
    assert np.array_equal(first.ranks, second.ranks)


def test_the_simulation_rejects_a_non_positive_budget(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=200, anneal_fraction=0.5)
    with pytest.raises(ValueError):
        simulate_start_distribution(curriculum, np.random.default_rng(10), episodes=0)


def test_simulated_difficulty_matches_the_drawn_states(tiny_scenario: Scenario) -> None:
    curriculum = StartStateCurriculum(tiny_scenario, total_episodes=100, anneal_fraction=1.0)
    anneal = simulate_start_distribution(curriculum, np.random.default_rng(11), episodes=30)
    expected = [
        start_state_difficulty(tiny_scenario, curriculum.ranked_pool[rank]) for rank in anneal.ranks
    ]
    assert anneal.difficulties.tolist() == expected


# -- the oversampling diagnostic -------------------------------------------


def test_band_shares_of_a_uniform_spread_are_equal() -> None:
    assert rank_band_shares(list(range(100)), 100) == (0.25, 0.25, 0.25, 0.25)


def test_band_shares_report_a_bottom_heavy_draw() -> None:
    shares = rank_band_shares([0, 1, 2, 3, 90], 100)
    assert shares[0] == pytest.approx(0.8)
    assert shares[3] == pytest.approx(0.2)


def test_band_shares_exclude_draws_from_outside_the_pool() -> None:
    """Shares are of *episodes*, so an unpooled draw lowers them rather than vanishing."""
    assert sum(rank_band_shares([0, 1, 100, 100], 100)) == pytest.approx(0.5)


def test_band_shares_of_nothing_are_zero() -> None:
    assert rank_band_shares([], 100) == (0.0, 0.0, 0.0, 0.0)
    assert rank_band_shares([1, 2], 0) == (0.0, 0.0, 0.0, 0.0)


def test_band_count_is_validated() -> None:
    with pytest.raises(ValueError):
        rank_band_shares([1], 10, bands=0)


# -- arm vocabulary --------------------------------------------------------


def test_the_control_arm_has_no_strategy() -> None:
    """One control, not three: a disabled curriculum ignores the sampler entirely."""
    labels = {curriculum_arm_label(0.0, strategy) for strategy in CurriculumStrategy}
    assert labels == {"No Curriculum"}
    assert curriculum_arm_slug(0.0) == "no-curriculum"


def test_each_strategy_names_its_own_arm() -> None:
    labels = {curriculum_arm_label(0.5, strategy) for strategy in CurriculumStrategy}
    assert labels == {
        "Growing Window Curriculum",
        "Sliding Window Curriculum",
        "Visit-Weighted Curriculum",
    }
    assert curriculum_arm_slug(0.5, "sliding") == "sliding-window-curriculum"


def test_the_stub_probe_reports_nothing_pending() -> None:
    """The sampler is implemented, so the probe must clear.

    Asserted as equality rather than as a subset: the subset form held both before
    and after the assignment, which meant a probe that could never clear -- one that
    reseeded its generator per draw, say -- passed its own test.
    """
    assert curriculum_stub_status() == ()
