"""Contract tests for the human-owned search objective and Pareto front.

These are EXPECTED TO FAIL until ``score_learning_curve`` and ``pareto_front`` in
``src/mars_rover_q/tuning.py`` are implemented. They are deselected from the default
``pytest`` run and are executed with::

    pytest -m human_todo tests/human_todo

Do not skip, weaken, xfail, or delete them. Every failure here should be a plain
assertion failure caused by the placeholder, never an import or fixture error.

**These tests deliberately do not pin a formula.** There is more than one defensible
way to collapse a learning curve into one number, and the repository does not need a
particular one -- it needs one that satisfies every property below. So the assertions
are invariants, anchors, and orderings: dominance, earliness, both rescalings, the two
endpoints, and the documented ``Raises:`` clauses. Any scoring rule meeting the
docstring passes all of them, and a rule that happens to match some reference
implementation but breaks an invariant fails.

The curves are synthetic. The objective is a function of an ordered sequence of
``(episodes, return)`` pairs and nothing else, so no test here trains anything.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from mars_rover_q.tuning import TrialOutcome, pareto_front, score_learning_curve

pytestmark = pytest.mark.human_todo

#: The y-axis scale used throughout: ``safe_corridor``'s best affordable sample.
REFERENCE = 160.0

#: The x-axis scale used throughout.
CAP = 1000


def curve(*points: tuple[int, float]) -> tuple[tuple[int, float], ...]:
    """A curve literal, for readability at the call sites."""
    return points


def flat(value: float, cap: int = CAP, points: int = 5) -> tuple[tuple[int, float], ...]:
    """A curve that sits at ``value`` from episode 0 onward."""
    step = cap // (points - 1)
    return tuple((index * step, value) for index in range(points))


def ramp(
    reaches: int, value: float, cap: int = CAP, points: int = 5
) -> tuple[tuple[int, float], ...]:
    """Zero until ``reaches``, then ``value`` for the rest of the budget.

    The cleanest shape for asking the question the objective exists to answer: two
    curves with the same plateau that arrive at it at different times.
    """
    step = cap // (points - 1)
    episodes = [index * step for index in range(points)]
    assert reaches in episodes, f"{reaches} is not a checkpoint of this curve"
    return tuple((episode, 0.0 if episode < reaches else value) for episode in episodes)


def outcome(
    number: int,
    *,
    episodes_to_best: int,
    best_return: float,
    score: float | None = 0.5,
) -> TrialOutcome:
    """A trial outcome carrying the two axes the front reads."""
    return TrialOutcome(
        number=number,
        params={},
        episodes_granted=CAP,
        episodes_run=CAP,
        score=score,
        best_return=best_return,
        episodes_to_best=episodes_to_best,
        final_return=best_return,
        final_success_rate=0.0,
        pruned=False,
    )


def numbers(front: Sequence[TrialOutcome]) -> list[int]:
    """Trial numbers of a front, in the order it was returned."""
    return [entry.number for entry in front]


# -- score_learning_curve: the two anchors ---------------------------------------


def test_a_curve_at_the_reference_from_the_first_point_scores_one() -> None:
    assert score_learning_curve(flat(REFERENCE), CAP, REFERENCE) == pytest.approx(1.0)


def test_a_curve_that_never_rises_above_zero_scores_zero() -> None:
    assert score_learning_curve(flat(0.0), CAP, REFERENCE) == pytest.approx(0.0)


def test_a_curve_of_pure_failure_scores_zero() -> None:
    """A policy that strands the rover is charged the battery penalty, not credited."""
    assert score_learning_curve(flat(-100.0), CAP, REFERENCE) == pytest.approx(0.0)


def test_every_score_lies_in_the_unit_interval() -> None:
    candidates = [
        flat(0.0),
        flat(-100.0),
        flat(REFERENCE),
        flat(REFERENCE * 4),
        ramp(500, 40.0),
        ramp(250, REFERENCE),
        curve((0, -100.0), (500, 40.0), (1000, REFERENCE)),
    ]
    for candidate in candidates:
        score = score_learning_curve(candidate, CAP, REFERENCE)
        assert 0.0 <= score <= 1.0, f"{score} out of range for {candidate}"


def test_a_curve_above_the_reference_does_not_exceed_one() -> None:
    """The reference is a ceiling on the scale, so overshoot cannot buy extra score."""
    assert score_learning_curve(flat(REFERENCE * 10), CAP, REFERENCE) == pytest.approx(1.0)


# -- score_learning_curve: dominance --------------------------------------------


def test_a_pointwise_better_curve_scores_strictly_higher() -> None:
    worse = curve((0, 0.0), (500, 40.0), (1000, 40.0))
    better = curve((0, 0.0), (500, 90.0), (1000, 90.0))
    assert score_learning_curve(better, CAP, REFERENCE) > score_learning_curve(
        worse, CAP, REFERENCE
    )


def test_an_improvement_at_a_single_checkpoint_scores_strictly_higher() -> None:
    worse = curve((0, 0.0), (250, 0.0), (500, 40.0), (750, 90.0), (1000, 90.0))
    better = curve((0, 0.0), (250, 0.0), (500, 90.0), (750, 90.0), (1000, 90.0))
    assert score_learning_curve(better, CAP, REFERENCE) > score_learning_curve(
        worse, CAP, REFERENCE
    )


def test_dominance_survives_the_region_below_zero() -> None:
    """Every trial starts at the battery penalty, so a floor-bound curve is the norm.

    ``ramp`` asks the dominance question above zero; this asks it below. A curve that
    recovers to the reference at its last checkpoint is strictly better than one that
    never leaves the floor, so it must score strictly higher. The ``0.0`` anchor is for
    a curve that *never rises above zero*, and this one does.
    """
    hopeless = flat(-100.0)
    recovers = curve(*hopeless[:-1], (hopeless[-1][0], REFERENCE))
    assert score_learning_curve(recovers, CAP, REFERENCE) > score_learning_curve(
        hopeless, CAP, REFERENCE
    )


def test_the_floor_is_a_floor_and_not_a_ranking_of_failures() -> None:
    """The complement of the test above: below zero there is nothing left to rank."""
    assert score_learning_curve(flat(-50.0), CAP, REFERENCE) == pytest.approx(0.0)
    assert score_learning_curve(flat(-100.0), CAP, REFERENCE) == pytest.approx(0.0)


def test_an_equal_curve_scores_equally() -> None:
    first = ramp(500, 90.0)
    second = ramp(500, 90.0)
    assert score_learning_curve(first, CAP, REFERENCE) == score_learning_curve(
        second, CAP, REFERENCE
    )


# -- score_learning_curve: earliness, the half that makes this a speed search ----


def test_reaching_the_same_plateau_sooner_scores_strictly_higher() -> None:
    early = ramp(250, 90.0)
    late = ramp(750, 90.0)
    assert score_learning_curve(early, CAP, REFERENCE) > score_learning_curve(late, CAP, REFERENCE)


def test_earliness_outranks_a_marginally_better_finish() -> None:
    """The search must not be a final-performance search wearing a speed label.

    ``fast`` holds 150 from a quarter of the budget in; ``slow`` spends three quarters
    of it at zero and then edges ahead at the very end. A pure end-of-run objective
    prefers ``slow``. This objective must not.
    """
    fast = ramp(250, 150.0)
    slow = curve((0, 0.0), (250, 0.0), (500, 0.0), (750, 0.0), (1000, REFERENCE))
    assert score_learning_curve(fast, CAP, REFERENCE) > score_learning_curve(slow, CAP, REFERENCE)


def test_a_curve_that_peaks_then_collapses_is_not_scored_as_if_it_held() -> None:
    """Read the whole curve, not its best point: a policy that degrades degraded."""
    held = flat(90.0)
    spiked = curve((0, 0.0), (250, 90.0), (500, 0.0), (750, 0.0), (1000, 0.0))
    assert score_learning_curve(held, CAP, REFERENCE) > score_learning_curve(spiked, CAP, REFERENCE)


# -- score_learning_curve: the two rescalings ------------------------------------


def test_the_score_is_invariant_to_rescaling_the_return_axis() -> None:
    """The same score on two maps whose payoffs differ fourfold."""
    base = ramp(250, 90.0)
    scaled = tuple((episode, value * 4.0) for episode, value in base)
    assert score_learning_curve(scaled, CAP, REFERENCE * 4.0) == pytest.approx(
        score_learning_curve(base, CAP, REFERENCE)
    )


def test_the_score_is_invariant_to_rescaling_the_episode_axis() -> None:
    """The same score at two caps an order of magnitude apart."""
    base = ramp(250, 90.0)
    stretched = tuple((episode * 10, value) for episode, value in base)
    assert score_learning_curve(stretched, CAP * 10, REFERENCE) == pytest.approx(
        score_learning_curve(base, CAP, REFERENCE)
    )


def test_the_two_rescalings_compose() -> None:
    base = ramp(500, 40.0)
    both = tuple((episode * 7, value * 0.25) for episode, value in base)
    assert score_learning_curve(both, CAP * 7, REFERENCE * 0.25) == pytest.approx(
        score_learning_curve(base, CAP, REFERENCE)
    )


# -- score_learning_curve: the documented Raises clauses -------------------------


def test_an_empty_curve_is_rejected() -> None:
    with pytest.raises(ValueError):
        score_learning_curve((), CAP, REFERENCE)


def test_a_curve_not_starting_at_episode_zero_is_rejected() -> None:
    with pytest.raises(ValueError):
        score_learning_curve(curve((100, 0.0), (1000, 90.0)), CAP, REFERENCE)


def test_a_non_ascending_curve_is_rejected() -> None:
    with pytest.raises(ValueError):
        score_learning_curve(curve((0, 0.0), (500, 90.0), (250, 90.0)), CAP, REFERENCE)


def test_a_repeated_episode_count_is_rejected() -> None:
    """Strictly ascending: two returns at the same episode count is not a curve."""
    with pytest.raises(ValueError):
        score_learning_curve(curve((0, 0.0), (500, 40.0), (500, 90.0)), CAP, REFERENCE)


def test_a_curve_reaching_past_the_cap_is_rejected() -> None:
    with pytest.raises(ValueError):
        score_learning_curve(curve((0, 0.0), (CAP + 1, 90.0)), CAP, REFERENCE)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_return_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        score_learning_curve(curve((0, 0.0), (1000, bad)), CAP, REFERENCE)


@pytest.mark.parametrize("cap", [0, -1, -1000])
def test_a_non_positive_cap_is_rejected(cap: int) -> None:
    with pytest.raises(ValueError):
        score_learning_curve(curve((0, 0.0)), cap, REFERENCE)


@pytest.mark.parametrize("reference", [0.0, -160.0, math.nan, math.inf])
def test_a_reference_return_that_is_not_positive_and_finite_is_rejected(reference: float) -> None:
    with pytest.raises(ValueError):
        score_learning_curve(flat(40.0), CAP, reference)


def test_a_single_point_curve_at_episode_zero_is_scored_not_rejected() -> None:
    """Degenerate but legal: one checkpoint, at episode 0, inside the cap."""
    score = score_learning_curve(curve((0, 40.0)), CAP, REFERENCE)
    assert 0.0 <= score <= 1.0


# -- pareto_front ----------------------------------------------------------------


def test_an_empty_pool_has_an_empty_front() -> None:
    assert pareto_front([]) == []


def test_a_lone_outcome_is_its_own_front() -> None:
    only = outcome(0, episodes_to_best=500, best_return=40.0)
    assert numbers(pareto_front([only])) == [0]


def test_a_dominated_outcome_is_dropped() -> None:
    cheap = outcome(0, episodes_to_best=100, best_return=50.0)
    strong = outcome(1, episodes_to_best=900, best_return=160.0)
    dominated = outcome(2, episodes_to_best=900, best_return=40.0)
    assert numbers(pareto_front([cheap, strong, dominated])) == [0, 1]


def test_domination_requires_being_no_worse_on_both_axes() -> None:
    """Neither of these dominates the other: one is cheaper, the other is better."""
    cheap = outcome(0, episodes_to_best=100, best_return=40.0)
    strong = outcome(1, episodes_to_best=900, best_return=160.0)
    assert numbers(pareto_front([cheap, strong])) == [0, 1]


def test_an_equal_return_at_a_lower_cost_dominates() -> None:
    fast = outcome(0, episodes_to_best=100, best_return=90.0)
    slow = outcome(1, episodes_to_best=900, best_return=90.0)
    assert numbers(pareto_front([slow, fast])) == [0]


def test_a_better_return_at_an_equal_cost_dominates() -> None:
    better = outcome(0, episodes_to_best=500, best_return=160.0)
    worse = outcome(1, episodes_to_best=500, best_return=90.0)
    assert numbers(pareto_front([worse, better])) == [0]


def test_outcomes_equal_on_both_axes_are_both_kept() -> None:
    """Domination needs a strict improvement somewhere, and a tie has none."""
    first = outcome(0, episodes_to_best=500, best_return=90.0)
    second = outcome(1, episodes_to_best=500, best_return=90.0)
    assert numbers(pareto_front([first, second])) == [0, 1]


def test_the_front_is_ordered_by_ascending_cost() -> None:
    pool = [
        outcome(0, episodes_to_best=900, best_return=160.0),
        outcome(1, episodes_to_best=100, best_return=40.0),
        outcome(2, episodes_to_best=500, best_return=90.0),
    ]
    assert numbers(pareto_front(pool)) == [1, 2, 0]


def test_the_order_of_the_input_does_not_change_the_front() -> None:
    pool = [
        outcome(0, episodes_to_best=900, best_return=160.0),
        outcome(1, episodes_to_best=100, best_return=40.0),
        outcome(2, episodes_to_best=500, best_return=90.0),
        outcome(3, episodes_to_best=600, best_return=45.0),
    ]
    forward = numbers(pareto_front(pool))
    backward = numbers(pareto_front(list(reversed(pool))))
    assert forward == backward == [1, 2, 0]


def test_ties_on_both_axes_are_ordered_by_trial_number() -> None:
    pool = [
        outcome(5, episodes_to_best=500, best_return=90.0),
        outcome(2, episodes_to_best=500, best_return=90.0),
    ]
    assert numbers(pareto_front(pool)) == [2, 5]


def test_a_front_of_many_is_the_full_staircase() -> None:
    pool = [
        outcome(index, episodes_to_best=index * 100, best_return=index * 10.0) for index in range(6)
    ]
    assert numbers(pareto_front(pool)) == [0, 1, 2, 3, 4, 5]


def test_every_outcome_dominated_by_one_leaves_a_front_of_one() -> None:
    best = outcome(0, episodes_to_best=100, best_return=160.0)
    pool = [best] + [
        outcome(index, episodes_to_best=100 + index * 50, best_return=160.0 - index * 10.0)
        for index in range(1, 5)
    ]
    assert numbers(pareto_front(pool)) == [0]


def test_the_returned_outcomes_are_the_objects_that_were_passed_in() -> None:
    """The caller reads parameters off the front, so it must carry the real outcomes."""
    cheap = outcome(0, episodes_to_best=100, best_return=40.0)
    strong = outcome(1, episodes_to_best=900, best_return=160.0)
    front = pareto_front([cheap, strong])
    assert front[0] is cheap
    assert front[1] is strong


def test_a_negative_cost_is_rejected_even_when_it_is_the_only_outcome() -> None:
    """A single outcome makes no comparisons, and the guard still has to fire."""
    with pytest.raises(ValueError):
        pareto_front([outcome(0, episodes_to_best=-1, best_return=40.0)])


def test_a_negative_cost_is_rejected_among_several() -> None:
    pool = [
        outcome(0, episodes_to_best=100, best_return=40.0),
        outcome(1, episodes_to_best=-5, best_return=90.0),
    ]
    with pytest.raises(ValueError):
        pareto_front(pool)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_return_is_rejected_even_when_it_is_the_only_outcome(bad: float) -> None:
    with pytest.raises(ValueError):
        pareto_front([outcome(0, episodes_to_best=100, best_return=bad)])


def test_a_zero_cost_outcome_is_legal() -> None:
    """A table that is best at its first checkpoint costs zero episodes, not an error."""
    free = outcome(0, episodes_to_best=0, best_return=-100.0)
    assert numbers(pareto_front([free])) == [0]
