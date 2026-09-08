"""Episode records, aggregation, and the statistics used in the write-up.

Base mission return and shaped training reward are kept in separate fields
everywhere in this module. Mixing them would make the three reward conditions
incomparable.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
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

    def as_dict(self) -> dict[str, Any]:
        """A flat, CSV/JSON-friendly view."""
        return asdict(self)


CSV_COLUMNS: Final[tuple[str, ...]] = tuple(EpisodeRecord.__dataclass_fields__)


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


def greedy_actions(q_table: NDArray[np.float64]) -> NDArray[np.int64]:
    """Highest-valued action per state, for policy export and the render overlay.

    This is a reporting utility for already-learned tables. It is deliberately *not*
    a substitute for :func:`mars_rover_q.agent.select_action`: it has no
    exploration and no random tie-breaking, and the training and evaluation loops
    never call it.
    """
    return np.asarray(np.argmax(q_table, axis=1), dtype=np.int64)


__all__ = [
    "CSV_COLUMNS",
    "ConfidenceInterval",
    "EpisodeRecord",
    "EpisodeSummary",
    "env_steps_to_threshold",
    "episodes_to_threshold",
    "greedy_actions",
    "mean_ci",
    "rolling_success_rate",
    "summarize_episodes",
    "t_critical_95",
]
