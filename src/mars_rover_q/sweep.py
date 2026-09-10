"""Analysis of an episode-budget sweep: curriculum against no curriculum.

The sweep asks one question -- *how much of the episode budget does the start-state
curriculum buy back?* -- and answers it three ways, in increasing order of how much
they commit to:

1. :func:`budget_curve_series` reshapes the flat per-seed rows into one curve per
   condition: mean and 95% CI of a metric against training budget. That is the
   figure.
2. :func:`mars_rover_q.metrics.paired_difference` differences the two arms *within
   each seed*. Both arms of a seed share their RNG streams, so this cancels the
   variance that is about the seed rather than about the curriculum, and it is the
   interval that decides whether the gap in the figure is real.
3. :func:`budget_to_reach` reads the same curves along the *x* axis instead of the
   *y*: the budget each condition needs to reach a shared performance target. This
   is the sample-efficiency claim -- "the curriculum needs N episodes for what the
   baseline needs M" -- and it is the number a reader remembers.

Everything here reads ``eval_mean_base_return`` by default: the mission return with
shaping excluded. Shaped return is not comparable across conditions and is never
the y-axis of a cross-condition figure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .curriculum import BASELINE_CURRICULUM as _BASELINE_CURRICULUM
from .curriculum import CurriculumStrategy, curriculum_arm_label
from .metrics import mean_ci, paired_difference

#: The default metric for every sweep figure and statistic: mission return with
#: shaping excluded.
DEFAULT_METRIC = "eval_mean_base_return"

#: The shared performance target for :func:`budget_to_reach`, as a fraction of the
#: best condition mean observed anywhere on the curve for that scenario. Deriving
#: the target from the data rather than fixing it per map keeps one number
#: meaningful across three scenarios whose achievable returns differ by 4x; using
#: the best value across *all* conditions keeps it symmetric, so neither arm is
#: measured against a bar set by itself.
DEFAULT_TARGET_FRACTION = 0.9

#: ``curriculum_fraction`` of the control arm. Re-exported from
#: :mod:`mars_rover_q.curriculum`, which is where the arm vocabulary lives.
BASELINE_CURRICULUM = _BASELINE_CURRICULUM


@dataclass(frozen=True, slots=True)
class BudgetSeries:
    """One condition's metric plotted against training budget.

    All five tuples are the same length and index-aligned with ``budgets``, which is
    ascending. A budget whose cell produced no usable value still appears, with
    ``nan`` in ``means`` and ``0`` in ``counts`` -- a gap in the data is drawn as a
    gap, never closed by dropping the point or by carrying the previous one forward.
    """

    scenario: str
    reward_mode: str
    curriculum_fraction: float
    curriculum_strategy: str
    metric: str
    budgets: tuple[int, ...]
    means: tuple[float, ...]
    ci_low: tuple[float, ...]
    ci_high: tuple[float, ...]
    counts: tuple[int, ...]

    @property
    def label(self) -> str:
        """Short human-readable name for a legend entry."""
        return curriculum_arm_label(self.curriculum_fraction, self.curriculum_strategy)

    @property
    def arm(self) -> tuple[float, str]:
        """The condition this curve belongs to, as a comparable key.

        A disabled curriculum ignores its strategy, so every control row collapses
        to one arm however many strategies the grid swept -- otherwise the control
        would be split three ways and each paired comparison would lose two-thirds
        of its seeds.
        """
        return arm_of_row(
            {
                "curriculum_fraction": self.curriculum_fraction,
                "curriculum_strategy": self.curriculum_strategy,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view."""
        return {
            "scenario": self.scenario,
            "reward_mode": self.reward_mode,
            "curriculum_fraction": self.curriculum_fraction,
            "curriculum_strategy": self.curriculum_strategy,
            "label": self.label,
            "metric": self.metric,
            "budgets": list(self.budgets),
            "means": list(self.means),
            "ci_low": list(self.ci_low),
            "ci_high": list(self.ci_high),
            "seeds": list(self.counts),
        }


def arm_of_row(row: dict[str, Any]) -> tuple[float, str]:
    """The ``(curriculum fraction, strategy)`` arm a per-seed row belongs to.

    Rows written before the strategy factor existed carry no ``curriculum_strategy``
    column and are read as the growing window, which is what they were. Control rows
    are normalised to a single empty strategy: see :attr:`BudgetSeries.arm`.
    """
    fraction = float(row.get("curriculum_fraction", 0.0) or 0.0)
    if fraction <= BASELINE_CURRICULUM:
        return (fraction, "")
    return (fraction, str(row.get("curriculum_strategy", CurriculumStrategy.GROWING.value)))


def _usable(value: Any) -> bool:
    """Whether a row's metric value can enter the arithmetic."""
    return value is not None and isinstance(value, int | float) and math.isfinite(float(value))


def budget_curve_series(
    rows: Sequence[dict[str, Any]], metric: str = DEFAULT_METRIC
) -> list[BudgetSeries]:
    """Fold per-seed rows into one budget curve per (scenario, reward mode, curriculum).

    Seeds are aggregated with :func:`mars_rover_q.metrics.mean_ci`, so each point
    carries a Student-t interval over however many seeds actually produced a usable
    value at that budget. Rows whose metric is ``None`` -- ``episodes_to_threshold``
    when the threshold was never met, for instance -- are excluded from that point's
    mean and reflected in its ``counts``, never coerced to a number.
    """
    series: list[BudgetSeries] = []
    conditions = sorted({(r["scenario"], r["reward_mode"], *arm_of_row(r)) for r in rows})
    for scenario, reward_mode, curriculum, strategy in conditions:
        cell = [
            r
            for r in rows
            if r["scenario"] == scenario
            and r["reward_mode"] == reward_mode
            and arm_of_row(r) == (curriculum, strategy)
        ]
        budgets = tuple(sorted({int(r["train_episodes"]) for r in cell}))
        means, lows, highs, counts = [], [], [], []
        for budget in budgets:
            values = [
                float(r[metric])
                for r in cell
                if int(r["train_episodes"]) == budget and _usable(r.get(metric))
            ]
            interval = mean_ci(values)
            means.append(interval.mean)
            lows.append(interval.low)
            highs.append(interval.high)
            counts.append(interval.n)
        series.append(
            BudgetSeries(
                scenario=scenario,
                reward_mode=reward_mode,
                curriculum_fraction=curriculum,
                curriculum_strategy=strategy,
                metric=metric,
                budgets=budgets,
                means=tuple(means),
                ci_low=tuple(lows),
                ci_high=tuple(highs),
                counts=tuple(counts),
            )
        )
    return series


def budget_to_reach(
    budgets: Sequence[int],
    values: Sequence[float],
    target: float,
) -> float | None:
    """The training budget at which a curve first reaches ``target``.

    This is the x-axis reading of the budget curve, and the statistic the whole
    sweep exists to produce: comparing it between the curriculum and baseline arms
    turns "the curriculum scores higher" into "the curriculum needs this many fewer
    episodes", which is the claim a sample-efficiency result is actually making.

    Because the budget ladder is log-spaced (1k, 3k, 5k, 10k, 20k), the answer
    almost never lands on a measured budget, so it is interpolated between the two
    that bracket the crossing. Interpolate in **log budget**: on a log-spaced ladder
    a linear interpolation between 10k and 20k would put the midpoint at 15k, which
    treats that gap as if it were the same size as the 1k-to-3k one. The figure's
    x-axis is log-scaled for the same reason, so this also makes the returned budget
    land where the curve visibly crosses the target line.

    Args:
        budgets: the measured training budgets, strictly ascending and all
            positive.
        values: the condition's mean metric at each budget, index-aligned with
            ``budgets`` and the same length. May contain ``nan`` where a budget
            produced no usable seeds.
        target: the performance level to reach.

    Returns:
        The interpolated budget at which the curve first attains ``target``, or
        ``None`` if it never does. ``None`` is a real result -- "did not reach
        within the budgets tested" -- and must be reported as such rather than
        coerced to the largest budget, exactly as
        :func:`mars_rover_q.metrics.episodes_to_threshold` treats its own
        never-reached case.

    Raises:
        ValueError: if the two sequences differ in length, are empty, if
            ``budgets`` is not strictly ascending, or if any budget is not
            positive (its logarithm has to exist).

    Invariants:
        * The **first** crossing is the answer. These curves are noisy at 10 seeds
          and are not guaranteed monotonic; a later, larger crossing must not
          shadow an earlier one.
        * Already at or above ``target`` at the smallest measured budget returns
          that budget itself. Never extrapolate below the ladder -- nothing was
          measured there.
        * The returned value lies within the bracketing pair, so it is always
          inside the measured range.
        * A ``nan`` value never contributes to an interpolation. When the point
          before a crossing is unusable there is nothing to interpolate *from*, and
          the crossing budget itself is the honest answer.
        * Neither input sequence is mutated.
    """
    if not values or not budgets:
        raise ValueError("Values and Budgets sequences must not be empty or None")
    if len(values) != len(budgets):
        raise ValueError("Values ant Budgets sequences must be the same length!")

    for i in range(len(budgets)):
        # Strictly ascending Budgets
        if i > 0 and budgets[i] <= budgets[i - 1]:
            raise ValueError("Budgets sequence must be strictly ascending")
        # Non negative
        if budgets[i] <= 0:
            raise ValueError(
                "Budgets sequence contained a negative or value. All elements must be positive."
            )

    # Edge case
    if values[0] >= target:
        return budgets[0]

    # Check for the crossing
    crossing_i = None
    for i in range(1, len(budgets)):
        if values[i] >= target:
            crossing_i = i
            break

    # Never Reached Target
    if crossing_i is None:
        return None

    # Check for the nan case
    if not _usable(values[crossing_i - 1]):
        return budgets[crossing_i]

    # Exact Match
    if values[crossing_i] == target:
        return budgets[crossing_i]

    # Interpolate
    frac_along_way = (target - values[crossing_i - 1]) / (
        values[crossing_i] - values[crossing_i - 1]
    )
    log_i_sub_one = math.log10(budgets[crossing_i - 1])
    log_i = math.log10(budgets[crossing_i])
    log_ret_budget = log_i_sub_one + (frac_along_way * (log_i - log_i_sub_one))
    ret_budget = math.pow(10, log_ret_budget)
    return ret_budget


def analyze_sweep(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str = DEFAULT_METRIC,
    target_fraction: float = DEFAULT_TARGET_FRACTION,
) -> dict[str, Any]:
    """The full curriculum-vs-baseline analysis for one swept experiment.

    Returns a JSON-serialisable payload with, per (scenario, reward mode):

    ``series``
        One :class:`BudgetSeries` per curriculum level -- the plotted curves.
    ``paired``
        Per budget, the within-seed ``curriculum - baseline`` difference with its
        paired 95% interval. This is the inferential result; the ``series``
        intervals overlapping does not refute a difference these intervals find.
    ``budget_to_target``
        Per curriculum level, the budget needed to reach ``target``, plus the
        ``speedup`` ratio between the baseline's and the curriculum's requirement.
    """
    series = budget_curve_series(rows, metric)
    groups = sorted({(s.scenario, s.reward_mode) for s in series})
    results: list[dict[str, Any]] = []

    for scenario, reward_mode in groups:
        group = [s for s in series if s.scenario == scenario and s.reward_mode == reward_mode]
        finite = [m for s in group for m in s.means if math.isfinite(m)]
        target = max(finite) * target_fraction if finite else math.nan

        baseline_series = next(
            (s for s in group if s.curriculum_fraction <= BASELINE_CURRICULUM), None
        )
        baseline_budget = (
            budget_to_reach(baseline_series.budgets, baseline_series.means, target)
            if baseline_series is not None and math.isfinite(target)
            else None
        )

        budget_to_target: list[dict[str, Any]] = []
        for entry in group:
            needed = (
                budget_to_reach(entry.budgets, entry.means, target)
                if math.isfinite(target)
                else None
            )
            budget_to_target.append(
                {
                    "curriculum_fraction": entry.curriculum_fraction,
                    "curriculum_strategy": entry.curriculum_strategy,
                    "label": entry.label,
                    "budget": needed,
                    # How many times fewer episodes than the control needed. Only
                    # defined when both arms actually got there.
                    "speedup_vs_baseline": (
                        baseline_budget / needed
                        if needed and baseline_budget and needed > 0.0
                        else None
                    ),
                }
            )

        paired: list[dict[str, Any]] = []
        baseline_rows = [
            r
            for r in rows
            if r["scenario"] == scenario
            and r["reward_mode"] == reward_mode
            and arm_of_row(r)[0] <= BASELINE_CURRICULUM
        ]
        for entry in group:
            if entry.curriculum_fraction <= BASELINE_CURRICULUM:
                continue
            for budget in entry.budgets:
                treatment_rows = [
                    r
                    for r in rows
                    if r["scenario"] == scenario
                    and r["reward_mode"] == reward_mode
                    and arm_of_row(r) == entry.arm
                    and int(r["train_episodes"]) == budget
                ]
                control_rows = [r for r in baseline_rows if int(r["train_episodes"]) == budget]
                interval = paired_difference(treatment_rows, control_rows, metric)
                paired.append(
                    {
                        "curriculum_fraction": entry.curriculum_fraction,
                        "curriculum_strategy": entry.curriculum_strategy,
                        "label": entry.label,
                        "budget": budget,
                        "metric": metric,
                        **interval.as_dict(),
                        # The claim is only "the curriculum helped at this budget"
                        # when the whole interval sits above zero.
                        "excludes_zero": bool(
                            math.isfinite(interval.low)
                            and math.isfinite(interval.high)
                            and (interval.low > 0.0 or interval.high < 0.0)
                        ),
                    }
                )

        results.append(
            {
                "scenario": scenario,
                "reward_mode": reward_mode,
                "metric": metric,
                "target": target,
                "target_fraction": target_fraction,
                "series": [s.as_dict() for s in group],
                "paired": paired,
                "budget_to_target": budget_to_target,
            }
        )

    return {"metric": metric, "target_fraction": target_fraction, "groups": results}


__all__ = [
    "BASELINE_CURRICULUM",
    "DEFAULT_METRIC",
    "DEFAULT_TARGET_FRACTION",
    "BudgetSeries",
    "analyze_sweep",
    "budget_curve_series",
    "budget_to_reach",
]
