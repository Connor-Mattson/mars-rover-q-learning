"""Episode records, aggregation, and the statistics used in the write-up.

Base mission return and shaped training reward are kept in separate fields
everywhere in this module. Mixing them would make the three reward conditions
incomparable.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .environment import Outcome

#: Two-sided 95% Student-t critical values by degrees of freedom (n - 1).
#: Used instead of a normal approximation because the seed count is small.
_T_CRITICAL_95: Final[dict[int, float]] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    24: 2.064,
    29: 2.045,
    39: 2.023,
    59: 2.001,
}


def t_critical_95(degrees_of_freedom: int) -> float:
    """Two-sided 95% t critical value, falling back to the normal 1.96."""
    if degrees_of_freedom <= 0:
        return math.nan
    if degrees_of_freedom in _T_CRITICAL_95:
        return _T_CRITICAL_95[degrees_of_freedom]
    known = [df for df in _T_CRITICAL_95 if df >= degrees_of_freedom]
    if known:
        return _T_CRITICAL_95[min(known)]
    return 1.96


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A mean with its 95% confidence interval and the sample size behind it."""

    mean: float
    low: float
    high: float
    n: int
    std: float

    @property
    def half_width(self) -> float:
        """Half the interval width; ``nan`` when undefined."""
        return (self.high - self.low) / 2.0

    def as_dict(self) -> dict[str, float | int]:
        """A flat, JSON-serialisable view."""
        return {
            "mean": self.mean,
            "ci_low": self.low,
            "ci_high": self.high,
            "n": self.n,
            "std": self.std,
        }


def mean_ci(values: Sequence[float] | NDArray[np.float64]) -> ConfidenceInterval:
    """Mean and 95% confidence interval across independent samples (seeds).

    With a single sample the interval is undefined and reported as ``nan``, never
    as a zero-width interval.
    """
    array = np.asarray(list(values), dtype=np.float64)
    n = int(array.size)
    if n == 0:
        return ConfidenceInterval(math.nan, math.nan, math.nan, 0, math.nan)
    mean = float(array.mean())
    if n == 1:
        return ConfidenceInterval(mean, math.nan, math.nan, 1, math.nan)
    std = float(array.std(ddof=1))
    margin = t_critical_95(n - 1) * std / math.sqrt(n)
    return ConfidenceInterval(mean, mean - margin, mean + margin, n, std)


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One completed episode, training or evaluation.

    ``base_return`` excludes shaping; ``shaped_return`` is what the agent actually
    optimised. Reports must never substitute one for the other.

    ``from_canonical_start`` is ``True`` for every evaluation episode and for any
    training episode that began at the lander with a full battery. Training episodes
    started elsewhere by the start-state curriculum are easier by construction, so a
    success rate that mixes the two is not a learning curve -- see
    :func:`canonical_start_records`.
    """

    episode: int
    steps: int
    base_return: float
    shaped_return: float
    outcome: str
    delivered_value: int
    battery_remaining: int
    energy_spent: int
    slips: int
    collisions: int
    invalid_collects: int
    max_directed_edge_repeats: int
    max_undirected_edge_repeats: int
    repeated_edge_fraction: float
    epsilon: float = 0.0
    env_steps_before: int = 0
    from_canonical_start: bool = True

    def as_dict(self) -> dict[str, Any]:
        """A flat, CSV/JSON-friendly view."""
        return asdict(self)


CSV_COLUMNS: Final[tuple[str, ...]] = tuple(EpisodeRecord.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Greedy performance on the canonical mission, measured mid-training.

    Why this exists: a training episode's own outcome is not a comparable measure
    of progress when the two conditions start their episodes in different places.
    A curriculum run spends its anneal starting mid-mission, so filtering to
    canonical-start training episodes leaves a thin, high-epsilon sample exactly
    where the comparison matters most -- and positioning those few episodes by
    their index within the filtered subsequence compresses them toward the origin,
    which flatters the curriculum.

    A checkpoint instead pauses training every ``cadence`` episodes and runs a
    fixed batch of greedy episodes from the canonical lander start, with
    exploration off. Both conditions are then measured on the same mission, at the
    same training-episode counts, with the same evaluation budget. ``episode`` is
    the true training-episode index, so these are directly plottable against
    training effort.
    """

    episode: int
    env_steps: int
    episodes: int
    success_rate: float
    mean_base_return: float
    mean_delivered_value: float
    mean_steps: float

    def as_dict(self) -> dict[str, Any]:
        """A flat, CSV/JSON-friendly view."""
        return asdict(self)


#: Column order for ``eval_checkpoints.csv``.
CHECKPOINT_CSV_COLUMNS: Final[tuple[str, ...]] = tuple(CheckpointRecord.__dataclass_fields__)


@dataclass(slots=True)
class EpisodeSummary:
    """Aggregate statistics over a batch of episodes."""

    episodes: int = 0
    success_rate: float = 0.0
    mean_base_return: float = 0.0
    mean_shaped_return: float = 0.0
    mean_delivered_value: float = 0.0
    mean_steps: float = 0.0
    mean_energy_remaining_on_success: float = math.nan
    mean_repeated_edge_fraction: float = 0.0
    mean_max_undirected_edge_repeats: float = 0.0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    delivered_sample_values: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """A flat, JSON-serialisable view."""
        return asdict(self)


def summarize_episodes(records: Iterable[EpisodeRecord]) -> EpisodeSummary:
    """Aggregate a batch of episodes into the reporting metrics."""
    items = list(records)
    summary = EpisodeSummary(episodes=len(items))
    if not items:
        return summary

    successes = [r for r in items if r.outcome == Outcome.SUCCESS.value]
    summary.success_rate = len(successes) / len(items)
    summary.mean_base_return = float(np.mean([r.base_return for r in items]))
    summary.mean_shaped_return = float(np.mean([r.shaped_return for r in items]))
    summary.mean_delivered_value = float(np.mean([r.delivered_value for r in items]))
    summary.mean_steps = float(np.mean([r.steps for r in items]))
    summary.mean_energy_remaining_on_success = (
        float(np.mean([r.battery_remaining for r in successes])) if successes else math.nan
    )
    summary.mean_repeated_edge_fraction = float(np.mean([r.repeated_edge_fraction for r in items]))
    summary.mean_max_undirected_edge_repeats = float(
        np.mean([r.max_undirected_edge_repeats for r in items])
    )
    summary.failure_reasons = dict(
        Counter(r.outcome for r in items if r.outcome != Outcome.SUCCESS.value)
    )
    summary.delivered_sample_values = dict(Counter(str(r.delivered_value) for r in successes))
    return summary


def paired_difference(
    treatment: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    pair_on: str = "seed",
) -> ConfidenceInterval:
    """Mean ``treatment - baseline`` difference in ``metric``, paired by ``pair_on``.

    Why paired: :func:`mars_rover_q.training.split_rngs` is deterministic, so the
    curriculum and no-curriculum arms of a given seed share the same environment,
    exploration, and start-state streams. The two arms are therefore *not*
    independent samples, and comparing their separate confidence intervals throws
    away the shared nuisance variance -- the part of a seed's outcome that is about
    which map rolls it got rather than about the curriculum. Differencing within a
    seed cancels it, and the interval that survives is usually far tighter than
    either arm's own.

    Args:
        treatment: per-seed rows for the condition under test, as produced by
            :func:`mars_rover_q.experiment.run_cell`. Read-only.
        baseline: per-seed rows for the control condition. Read-only.
        metric: the row key to difference, e.g. ``"eval_mean_base_return"``.
        pair_on: the row key identifying a pair. Defaults to ``"seed"``.

    Returns:
        A :class:`ConfidenceInterval` over the per-pair differences, whose ``n`` is
        the number of *pairs*, not the number of rows. An interval that excludes
        zero is the claim; one that straddles it is not.

    Raises:
        ValueError: if either side contains two rows with the same ``pair_on``
            value, which would make the pairing ambiguous.

    Invariants:
        * Only keys present on **both** sides contribute. A seed that ran in one
          arm and not the other is dropped, never treated as a zero difference.
        * A pair whose metric is ``None`` or non-finite on either side is dropped
          for that metric alone, exactly as :func:`mean_ci`'s callers do. This is
          how ``episodes_to_threshold`` -- which is legitimately ``None`` when the
          threshold was never met -- stays out of the arithmetic instead of being
          coerced to a number.
        * Zero pairs returns the same all-``nan`` interval :func:`mean_ci` returns
          for an empty sample; one pair returns the difference with a ``nan``
          interval. Neither is an error.
        * Neither argument is mutated or reordered.
    """
    # Check for key duplicates up front
    treatment_keys = {t[pair_on] for t in treatment}
    baseline_keys = {b[pair_on] for b in baseline}

    if len(treatment_keys) != len(treatment):
        raise ValueError("Treatment contains a duplicate key!")

    if len(baseline_keys) != len(baseline):
        raise ValueError("Treatment contains a duplicate key!")

    key_intersection = treatment_keys.intersection(baseline_keys)

    differences = []
    for k in key_intersection:
        usable_baseline, usable_treatment = False, False
        # Find baseline value
        for b in baseline:
            if b[pair_on] == k:
                base_val = b[metric]
                usable_baseline = True
                if base_val is None or not math.isfinite(float(base_val)):
                    usable_baseline = False
                    break

        # Find treatment value
        for t in treatment:
            if t[pair_on] == k:
                treat_val = t[metric]
                usable_treatment = True
                if treat_val is None or not math.isfinite(float(treat_val)):
                    usable_treatment = False
                    break

        if usable_treatment and usable_baseline:
            differences.append(treat_val - base_val)

    return mean_ci(differences)


def rolling_success_rate(records: Sequence[EpisodeRecord], window: int) -> NDArray[np.float64]:
    """Trailing success rate over a fixed window, aligned to each episode index."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    flags = np.array(
        [1.0 if r.outcome == Outcome.SUCCESS.value else 0.0 for r in records], dtype=np.float64
    )
    if flags.size == 0:
        return flags
    cumulative = np.concatenate(([0.0], np.cumsum(flags)))
    indices = np.arange(1, flags.size + 1)
    starts = np.maximum(0, indices - window)
    return (cumulative[indices] - cumulative[starts]) / (indices - starts)


def episodes_to_threshold(
    records: Sequence[EpisodeRecord], threshold: float, window: int = 100
) -> int | None:
    """First episode index whose trailing success rate reaches ``threshold``.

    Returns ``None`` when the threshold is never reached, which is a real result
    and must be reported as such rather than silently coerced to the episode count.
    """
    if not records:
        return None
    rates = rolling_success_rate(records, window)
    reached = np.flatnonzero(rates >= threshold)
    if reached.size == 0:
        return None
    return int(records[int(reached[0])].episode)


def env_steps_to_threshold(
    records: Sequence[EpisodeRecord], threshold: float, window: int = 100
) -> int | None:
    """Environment steps consumed before the success threshold is first met."""
    episode = episodes_to_threshold(records, threshold, window)
    if episode is None:
        return None
    for record in records:
        if record.episode == episode:
            return record.env_steps_before + record.steps
    return None


def canonical_start_records(records: Sequence[EpisodeRecord]) -> list[EpisodeRecord]:
    """The subset of ``records`` that began at the canonical lander start.

    Curriculum runs deliberately start most early episodes somewhere easier, so
    progress must be read from these episodes alone to stay comparable with a run
    that had no curriculum. ``env_steps_before`` is untouched by the filter, so
    sample-efficiency comparisons still count every step the agent actually took.
    """
    return [record for record in records if record.from_canonical_start]


def tied_state_fraction(q_table: NDArray[np.float64]) -> float:
    """Share of states whose action values are all identical.

    A state is "tied" when nothing has ever broken the symmetry between its
    actions -- typically because the state was never visited, so every entry still
    holds the table's initial value. :func:`mars_rover_q.agent.select_action` then
    picks uniformly at random among all of them, which is what makes the policy
    overlay flicker between arrows on repeated draws.

    This is the coverage diagnostic behind the start-state curriculum: a table that
    solves its map from the lander but leaves most of the state space tied has
    learned one corridor, not a policy.
    """
    if q_table.size == 0:
        return 0.0
    if q_table.shape[1] < 2:
        return 1.0
    return float(np.mean(np.ptp(q_table, axis=1) == 0.0))


def learned_state_mask(
    q_table: NDArray[np.float64], initial_value: float = 0.0
) -> NDArray[np.bool_]:
    """Which states hold at least one action value the trainer actually wrote.

    A row that still equals the table's initial value everywhere was never on the
    receiving end of an update: nothing was learned about that state, and its
    greedy action is whatever ``argmax`` happens to return for a flat row. The
    complement of this mask is therefore the honest denominator for "how much of
    the state space did this run touch".

    ``initial_value`` must be the run's ``initial_q``, not assumed zero --
    optimistic initialisation is a supported exploration device, and comparing an
    optimistically initialised table against ``0.0`` would report every state as
    learned.

    Related but not the same as :func:`tied_state_fraction`: a state that was
    visited can still end up tied, and a state whose row moved uniformly away from
    ``initial_value`` is learned but would not be counted here as tied.
    """
    if q_table.size == 0:
        return np.zeros(q_table.shape[0], dtype=np.bool_)
    return np.asarray(np.any(q_table != initial_value, axis=1), dtype=np.bool_)


def learned_state_fraction(q_table: NDArray[np.float64], initial_value: float = 0.0) -> float:
    """Share of states carrying a learned value; see :func:`learned_state_mask`."""
    if q_table.size == 0:
        return 0.0
    return float(np.mean(learned_state_mask(q_table, initial_value)))


def greedy_actions(q_table: NDArray[np.float64]) -> NDArray[np.int64]:
    """Highest-valued action per state, for policy export and the render overlay.

    This is a reporting utility for already-learned tables. It is deliberately *not*
    a substitute for :func:`mars_rover_q.agent.select_action`: it has no
    exploration and no random tie-breaking, and the training and evaluation loops
    never call it.
    """
    return np.asarray(np.argmax(q_table, axis=1), dtype=np.int64)


__all__ = [
    "CHECKPOINT_CSV_COLUMNS",
    "CSV_COLUMNS",
    "CheckpointRecord",
    "ConfidenceInterval",
    "EpisodeRecord",
    "EpisodeSummary",
    "canonical_start_records",
    "env_steps_to_threshold",
    "episodes_to_threshold",
    "greedy_actions",
    "learned_state_fraction",
    "learned_state_mask",
    "mean_ci",
    "paired_difference",
    "rolling_success_rate",
    "summarize_episodes",
    "t_critical_95",
    "tied_state_fraction",
]
