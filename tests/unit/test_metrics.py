"""Aggregation, confidence intervals, and the threshold metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mars_rover_q.metrics import (
    EpisodeRecord,
    env_steps_to_threshold,
    episodes_to_threshold,
    greedy_actions,
    mean_ci,
    rolling_success_rate,
    summarize_episodes,
    t_critical_95,
)


def record(episode: int, outcome: str = "success", **kwargs: float) -> EpisodeRecord:
    defaults: dict[str, float] = {
        "steps": 10,
        "base_return": 40.0,
        "shaped_return": 42.0,
        "delivered_value": 40,
        "battery_remaining": 5,
        "energy_spent": 10,
        "slips": 0,
        "collisions": 0,
        "invalid_collects": 0,
        "max_directed_edge_repeats": 1,
        "max_undirected_edge_repeats": 1,
        "repeated_edge_fraction": 0.0,
        "epsilon": 0.1,
        "env_steps_before": episode * 10,
    }
    defaults.update(kwargs)
    return EpisodeRecord(episode=episode, outcome=outcome, **defaults)  # type: ignore[arg-type]


def test_mean_ci_of_a_constant_sample_has_zero_width() -> None:
    interval = mean_ci([3.0, 3.0, 3.0, 3.0])
    assert interval.mean == 3.0
    assert interval.low == interval.high == 3.0
    assert interval.n == 4


def test_mean_ci_widens_with_spread() -> None:
    tight = mean_ci([1.0, 1.1, 0.9, 1.0, 1.0])
    loose = mean_ci([1.0, 5.0, -3.0, 2.0, 0.0])
    assert loose.half_width > tight.half_width


def test_mean_ci_of_a_single_sample_is_undefined_not_zero_width() -> None:
    interval = mean_ci([2.0])
    assert interval.mean == 2.0
    assert math.isnan(interval.low)
    assert math.isnan(interval.high)


def test_mean_ci_of_no_samples_is_all_nan() -> None:
    interval = mean_ci([])
    assert interval.n == 0
    assert math.isnan(interval.mean)


def test_t_critical_is_wider_than_the_normal_approximation_for_small_samples() -> None:
    assert t_critical_95(4) > 1.96
    assert t_critical_95(4) > t_critical_95(20)
    assert t_critical_95(500) == pytest.approx(1.96)


def test_summarize_reports_success_rate_and_separate_returns() -> None:
    records = [
        record(0, "success", base_return=40.0, shaped_return=60.0),
        record(1, "battery_depleted", base_return=-100.0, shaped_return=-40.0, delivered_value=0),
    ]
    summary = summarize_episodes(records)
    assert summary.episodes == 2
    assert summary.success_rate == 0.5
    assert summary.mean_base_return == -30.0
    assert summary.mean_shaped_return == 10.0
    assert summary.failure_reasons == {"battery_depleted": 1}


def test_summarize_averages_energy_only_over_successes() -> None:
    records = [
        record(0, "success", battery_remaining=12),
        record(1, "step_limit", battery_remaining=0, delivered_value=0),
    ]
    assert summarize_episodes(records).mean_energy_remaining_on_success == 12.0


def test_summarize_of_no_episodes_is_empty_not_an_error() -> None:
    summary = summarize_episodes([])
    assert summary.episodes == 0
    assert summary.success_rate == 0.0


def test_rolling_success_rate_tracks_the_trailing_window() -> None:
    records = [record(i, "success" if i >= 2 else "step_limit") for i in range(6)]
    rates = rolling_success_rate(records, window=2)
    assert rates[0] == 0.0
    assert rates[-1] == 1.0
    assert rates.shape == (6,)


def test_episodes_to_threshold_finds_the_first_crossing() -> None:
    records = [record(i, "success" if i >= 3 else "step_limit") for i in range(10)]
    assert episodes_to_threshold(records, threshold=1.0, window=3) == 5


def test_episodes_to_threshold_returns_none_when_never_reached() -> None:
    records = [record(i, "step_limit") for i in range(10)]
    assert episodes_to_threshold(records, threshold=0.5, window=3) is None
    assert env_steps_to_threshold(records, threshold=0.5, window=3) is None


def test_env_steps_to_threshold_counts_environment_steps() -> None:
    records = [record(i, "success") for i in range(5)]
    assert env_steps_to_threshold(records, threshold=1.0, window=1) == 10


def test_rolling_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window"):
        rolling_success_rate([record(0)], window=0)


def test_greedy_actions_picks_the_highest_valued_column() -> None:
    table = np.array([[0.0, 1.0, 0.5], [2.0, -1.0, 0.0]])
    assert list(greedy_actions(table)) == [1, 0]
