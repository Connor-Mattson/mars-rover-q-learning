"""Contract tests for the two human-owned budget-sweep analysis functions.

These are EXPECTED TO FAIL until ``paired_difference`` in
``src/mars_rover_q/metrics.py`` and ``budget_to_reach`` in
``src/mars_rover_q/sweep.py`` are implemented. They are deselected from the
default ``pytest`` run and are executed with::

    pytest -m human_todo tests/human_todo

Do not skip, weaken, xfail, or delete them. Every failure here should be a plain
assertion failure caused by a placeholder, never an import or fixture error.

The two functions are independent: neither set of tests calls the other function,
so finishing one turns its tests green without waiting for the other.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from mars_rover_q.metrics import paired_difference
from mars_rover_q.sweep import budget_to_reach

pytestmark = pytest.mark.human_todo


def row(seed: int, value: float | None, **extra: Any) -> dict[str, Any]:
    """One per-seed summary row carrying a single metric."""
    return {"seed": seed, "eval_mean_base_return": value, **extra}


# -- paired_difference ----------------------------------------------------


def test_paired_difference_averages_the_within_seed_deltas() -> None:
    treatment = [row(1, 30.0), row(2, 50.0), row(3, 70.0)]
    baseline = [row(1, 10.0), row(2, 10.0), row(3, 10.0)]
    result = paired_difference(treatment, baseline, "eval_mean_base_return")
    assert result.n == 3
    assert result.mean == pytest.approx(40.0)


def test_paired_difference_pairs_by_seed_not_by_position() -> None:
    """The two arms are not guaranteed to arrive in the same order.

    Differencing position-by-position would compare seed 3's curriculum run against
    seed 1's baseline, which is not a paired comparison at all. Note that it would
    still produce the right *mean*: the average difference equals the difference of
    the averages however the rows are lined up. Only the spread gives it away, which
    is the whole reason pairing is worth doing -- the mean was never the part that
    needed help.
    """
    treatment = [row(3, 300.0), row(1, 100.0), row(2, 200.0)]
    baseline = [row(1, 95.0), row(2, 195.0), row(3, 295.0)]
    result = paired_difference(treatment, baseline, "eval_mean_base_return")
    assert result.n == 3
    assert result.mean == pytest.approx(5.0)
    # Paired correctly every delta is exactly +5, so the spread is zero and the
    # interval is tight. Paired by position the deltas are 205, -95, -95.
    assert result.std == pytest.approx(0.0, abs=1e-9)
    assert result.low == pytest.approx(5.0, abs=1e-6)


def test_paired_difference_is_signed_treatment_minus_baseline() -> None:
    treatment = [row(1, 5.0), row(2, 5.0)]
    baseline = [row(1, 25.0), row(2, 25.0)]
    result = paired_difference(treatment, baseline, "eval_mean_base_return")
    assert result.mean == pytest.approx(-20.0)


def test_paired_difference_beats_unpaired_when_seeds_move_together() -> None:
    """The whole reason this function exists.

    Both arms of a seed share their RNG streams, so a seed that happens to get an
    easy roll scores high in *both* arms. Here the between-seed spread is huge and
    the within-seed effect is a constant +5; the paired interval must find it.
    """
    treatment = [row(seed, base + 5.0) for seed, base in enumerate([0.0, 100.0, 200.0], start=1)]
    baseline = [row(seed, base) for seed, base in enumerate([0.0, 100.0, 200.0], start=1)]
    result = paired_difference(treatment, baseline, "eval_mean_base_return")
    assert result.mean == pytest.approx(5.0)
    assert result.std == pytest.approx(0.0, abs=1e-9)
    assert result.low > 0.0


def test_paired_difference_drops_seeds_missing_from_either_arm() -> None:
    treatment = [row(1, 30.0), row(2, 40.0), row(9, 999.0)]
    baseline = [row(1, 10.0), row(2, 10.0)]
    result = paired_difference(treatment, baseline, "eval_mean_base_return")
    assert result.n == 2
    assert result.mean == pytest.approx(25.0)


def test_paired_difference_drops_pairs_whose_metric_is_none() -> None:
    """``episodes_to_threshold`` is legitimately ``None`` when never reached."""
    treatment = [row(1, 30.0), row(2, None), row(3, 50.0)]
    baseline = [row(1, 10.0), row(2, 10.0), row(3, 10.0)]
    result = paired_difference(treatment, baseline, "eval_mean_base_return")
    assert result.n == 2
    assert result.mean == pytest.approx(30.0)


def test_paired_difference_drops_non_finite_pairs() -> None:
    treatment = [row(1, 30.0), row(2, math.nan)]
    baseline = [row(1, 10.0), row(2, 10.0)]
    result = paired_difference(treatment, baseline, "eval_mean_base_return")
    assert result.n == 1
    assert result.mean == pytest.approx(20.0)


def test_paired_difference_with_no_overlap_is_an_empty_interval_not_an_error() -> None:
    result = paired_difference([row(1, 30.0)], [row(2, 10.0)], "eval_mean_base_return")
    assert result.n == 0
    assert math.isnan(result.mean)
    assert math.isnan(result.low)


def test_paired_difference_with_one_pair_has_an_undefined_interval() -> None:
    result = paired_difference([row(1, 30.0)], [row(1, 10.0)], "eval_mean_base_return")
    assert result.n == 1
    assert result.mean == pytest.approx(20.0)
    assert math.isnan(result.low)
    assert math.isnan(result.high)


def test_paired_difference_rejects_a_duplicated_pair_key() -> None:
    treatment = [row(1, 30.0), row(1, 40.0)]
    baseline = [row(1, 10.0)]
    with pytest.raises(ValueError):
        paired_difference(treatment, baseline, "eval_mean_base_return")


def test_paired_difference_honours_a_custom_pair_key() -> None:
    treatment = [{"replicate": "a", "m": 4.0}, {"replicate": "b", "m": 6.0}]
    baseline = [{"replicate": "a", "m": 1.0}, {"replicate": "b", "m": 1.0}]
    result = paired_difference(treatment, baseline, "m", pair_on="replicate")
    assert result.n == 2
    assert result.mean == pytest.approx(4.0)


def test_paired_difference_does_not_mutate_its_inputs() -> None:
    treatment = [row(1, 30.0), row(2, 40.0)]
    baseline = [row(1, 10.0), row(2, 20.0)]
    before = ([dict(r) for r in treatment], [dict(r) for r in baseline])
    paired_difference(treatment, baseline, "eval_mean_base_return")
    assert [dict(r) for r in treatment] == before[0]
    assert [dict(r) for r in baseline] == before[1]


# -- budget_to_reach ------------------------------------------------------

BUDGETS = [1000, 3000, 5000, 10000, 20000]


def test_budget_to_reach_returns_none_when_the_target_is_never_met() -> None:
    assert budget_to_reach(BUDGETS, [0.0, 10.0, 20.0, 30.0, 40.0], target=100.0) is None


def test_budget_to_reach_returns_the_first_budget_when_already_above() -> None:
    """Never extrapolate below the ladder; nothing was measured there."""
    assert budget_to_reach(BUDGETS, [50.0, 60.0, 70.0, 80.0, 90.0], target=40.0) == 1000.0


def test_budget_to_reach_returns_an_exactly_measured_budget_unchanged() -> None:
    assert budget_to_reach(BUDGETS, [0.0, 40.0, 70.0, 80.0, 90.0], target=40.0) == pytest.approx(
        3000.0
    )


def test_budget_to_reach_interpolates_in_log_budget_not_linear_budget() -> None:
    """The midpoint between 5k and 20k is 10k, not 12.5k.

    The ladder is log-spaced and the figure's x-axis is log-scaled, so a linear
    interpolation would put the answer somewhere the curve does not visibly cross
    the target line.
    """
    result = budget_to_reach([5000, 20000], [0.0, 100.0], target=50.0)
    assert result == pytest.approx(10000.0, rel=1e-6)


def test_budget_to_reach_lands_inside_the_bracketing_pair() -> None:
    result = budget_to_reach(BUDGETS, [0.0, 10.0, 20.0, 90.0, 95.0], target=50.0)
    assert result is not None
    assert 5000.0 < result < 10000.0


def test_budget_to_reach_takes_the_first_crossing_of_a_noisy_curve() -> None:
    """These curves are not monotonic at ten seeds.

    A dip after an early crossing must not push the answer out to the later,
    larger budget where the curve comes back up.
    """
    result = budget_to_reach(BUDGETS, [0.0, 60.0, 20.0, 30.0, 80.0], target=50.0)
    assert result is not None
    assert 1000.0 < result < 3000.0


def test_budget_to_reach_does_not_interpolate_from_a_nan_predecessor() -> None:
    """With nothing usable to interpolate *from*, the crossing budget is the answer."""
    result = budget_to_reach(BUDGETS, [0.0, math.nan, 80.0, 85.0, 90.0], target=50.0)
    assert result == pytest.approx(5000.0)


def test_budget_to_reach_skips_a_nan_that_is_not_a_crossing() -> None:
    result = budget_to_reach(BUDGETS, [0.0, math.nan, 10.0, 80.0, 90.0], target=50.0)
    assert result is not None
    assert 5000.0 < result < 10000.0


def test_budget_to_reach_handles_a_single_measured_budget() -> None:
    assert budget_to_reach([4000], [90.0], target=50.0) == 4000.0
    assert budget_to_reach([4000], [10.0], target=50.0) is None


@pytest.mark.parametrize(
    ("budgets", "values"),
    [
        ([1000, 3000], [1.0]),
        ([], []),
        ([3000, 1000], [1.0, 2.0]),
        ([1000, 1000], [1.0, 2.0]),
        ([0, 1000], [1.0, 2.0]),
        ([-1000, 1000], [1.0, 2.0]),
    ],
)
def test_budget_to_reach_rejects_malformed_ladders(budgets: list[int], values: list[float]) -> None:
    with pytest.raises(ValueError):
        budget_to_reach(budgets, values, target=1.5)


def test_budget_to_reach_does_not_mutate_its_inputs() -> None:
    budgets = [1000, 3000, 5000]
    values = [0.0, 60.0, 20.0]
    budget_to_reach(budgets, values, target=50.0)
    assert budgets == [1000, 3000, 5000]
    assert values == [0.0, 60.0, 20.0]
