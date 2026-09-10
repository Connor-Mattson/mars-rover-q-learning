"""The budget-sweep grid expansion, artefact policy, and series reshaping.

These cover the machinery around the two human-owned analysis functions and must
pass at all times. The contracts for ``paired_difference`` and ``budget_to_reach``
themselves live in ``tests/human_todo/test_sweep_analysis.py``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from mars_rover_q.experiment import (
    AGGREGATED_METRICS,
    ExperimentConfig,
    aggregate_rows,
    condition_of,
)
from mars_rover_q.sweep import BudgetSeries, analyze_sweep, arm_of_row, budget_curve_series


@pytest.fixture
def swept_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="sweep",
        scenarios=("safe_corridor",),
        reward_modes=("sparse",),
        seeds=(1, 2),
        episode_budgets=(3000, 1000),
        curriculum_fractions=(0.5, 0.0),
    )


# -- config expansion ------------------------------------------------------


def test_scalar_config_reports_a_single_level_per_swept_factor() -> None:
    config = ExperimentConfig(episodes=4000, curriculum_fraction=0.25)
    assert config.budgets == (4000,)
    assert config.curricula == (0.25,)
    assert not config.is_swept


def test_swept_factors_replace_the_scalars_and_are_sorted(
    swept_config: ExperimentConfig,
) -> None:
    assert swept_config.budgets == (1000, 3000)
    assert swept_config.curricula == (0.0, 0.5)
    assert swept_config.is_swept


def test_total_runs_and_episodes_cover_the_full_cross(swept_config: ExperimentConfig) -> None:
    assert swept_config.total_runs == 1 * 1 * 2 * 2 * 2
    # 2 curricula x 2 seeds x (1000 + 3000)
    assert swept_config.total_train_episodes == 16000
    assert len(swept_config.cells()) == swept_config.total_runs


def test_cells_vary_the_budget_slowest(swept_config: ExperimentConfig) -> None:
    """A partial sweep should still have the cheap end complete for every condition."""
    budgets = [budget for _s, _r, budget, _c, _strategy, _seed in swept_config.cells()]
    assert budgets == sorted(budgets)


def test_train_config_carries_the_cells_budget_and_curriculum(
    swept_config: ExperimentConfig,
) -> None:
    train_config = swept_config.train_config(
        "safe_corridor", "sparse", 1, episodes=3000, curriculum_fraction=0.5
    )
    assert train_config.episodes == 3000
    assert train_config.curriculum_fraction == 0.5


def test_epsilon_and_curriculum_anneals_scale_with_the_budget(
    swept_config: ExperimentConfig,
) -> None:
    """The point of training a fresh table per budget rather than checkpointing one."""
    short = swept_config.train_config("safe_corridor", "sparse", 1, episodes=1000)
    long = swept_config.train_config("safe_corridor", "sparse", 1, episodes=3000)
    assert short.agent_config().epsilon.decay_episodes == 600
    assert long.agent_config().epsilon.decay_episodes == 1800


def test_bundled_sweep_configs_load_and_declare_the_designed_grid() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    config = ExperimentConfig.from_file(root / "curriculum_budget_sweep.json")
    assert config.budgets == (1000, 3000, 5000, 10000, 20000)
    assert config.curricula == (0.0, 0.5)
    assert len(config.seeds) == 10
    assert config.reward_modes == ("sparse",)
    assert config.total_runs == 300


# -- run naming and artefact policy ---------------------------------------


def test_run_name_omits_segments_for_factors_that_do_not_vary() -> None:
    config = ExperimentConfig(scenarios=("safe_corridor",), reward_modes=("sparse",), seeds=(1,))
    assert (
        config.run_name("safe_corridor", "sparse", 1, 4000, 0.0) == "safe_corridor__sparse__seed1"
    )


def test_run_name_separates_every_swept_cell(swept_config: ExperimentConfig) -> None:
    names = {
        swept_config.run_name("safe_corridor", "sparse", seed, budget, curriculum, strategy)
        for _s, _r, budget, curriculum, strategy, seed in swept_config.cells()
    }
    assert len(names) == swept_config.total_runs
    assert "safe_corridor__sparse__cf0.5__ep3000__seed2" in names


def test_max_budget_policy_keeps_only_the_largest_budgets_tables(
    swept_config: ExperimentConfig,
) -> None:
    from mars_rover_q.experiment import _keeps_table

    config = ExperimentConfig(
        episode_budgets=swept_config.episode_budgets, save_q_tables="max_budget"
    )
    assert not _keeps_table(config, 1000)
    assert _keeps_table(config, 3000)


# -- aggregation keys ------------------------------------------------------


def _row(
    scenario: str,
    budget: int,
    curriculum: float,
    seed: int,
    value: float,
    strategy: str = "growing",
) -> dict[str, Any]:
    """A per-seed summary row shaped like the ones ``run_cell`` produces."""
    return {
        "scenario": scenario,
        "reward_mode": "sparse",
        "seed": seed,
        "train_episodes": budget,
        "curriculum_fraction": curriculum,
        "curriculum_strategy": strategy,
        "eval_mean_base_return": value,
        **{metric: 0.0 for metric in AGGREGATED_METRICS if metric != "eval_mean_base_return"},
        "episodes_to_threshold": None,
        "env_steps_to_threshold": None,
    }


def test_conditions_are_not_pooled_across_budget_or_curriculum() -> None:
    rows = [
        _row("safe_corridor", budget, curriculum, seed, 1.0)
        for budget in (1000, 3000)
        for curriculum in (0.0, 0.5)
        for seed in (1, 2)
    ]
    assert len({condition_of(r) for r in rows}) == 4
    aggregated = aggregate_rows(rows)
    assert len(aggregated) == 4
    assert all(entry["seeds"] == 2 for entry in aggregated)


def test_aggregation_still_works_on_rows_predating_the_sweep() -> None:
    """The reward-comparison grid's rows carry no budget or curriculum column."""
    rows = [
        {
            "scenario": "s",
            "reward_mode": "sparse",
            "seed": seed,
            "eval_mean_base_return": float(seed),
            **{metric: 0.0 for metric in AGGREGATED_METRICS if metric != "eval_mean_base_return"},
            "episodes_to_threshold": None,
            "env_steps_to_threshold": None,
        }
        for seed in (1, 2, 3)
    ]
    aggregated = aggregate_rows(rows)
    assert len(aggregated) == 1
    assert aggregated[0]["seeds"] == 3


# -- budget_curve_series ---------------------------------------------------


def test_series_are_split_by_condition_and_ordered_by_budget() -> None:
    rows = [
        _row("safe_corridor", budget, curriculum, seed, 10.0)
        for budget in (5000, 1000, 3000)
        for curriculum in (0.0, 0.5)
        for seed in (1, 2)
    ]
    series = budget_curve_series(rows)
    assert len(series) == 2
    for entry in series:
        assert entry.budgets == (1000, 3000, 5000)
        assert entry.counts == (2, 2, 2)


def test_series_carry_a_mean_and_interval_across_seeds() -> None:
    rows = [
        _row("safe_corridor", 1000, 0.0, seed, value)
        for seed, value in enumerate([10.0, 20.0, 30.0], start=1)
    ]
    (entry,) = budget_curve_series(rows)
    assert entry.means[0] == pytest.approx(20.0)
    assert entry.ci_low[0] < 20.0 < entry.ci_high[0]
    assert entry.counts[0] == 3


def test_series_report_a_missing_metric_as_a_gap_not_a_zero() -> None:
    """``episodes_to_threshold`` is ``None`` for a seed that never got there."""
    rows = [_row("safe_corridor", 1000, 0.0, seed, 5.0) for seed in (1, 2)]
    (entry,) = budget_curve_series(rows, "episodes_to_threshold")
    assert math.isnan(entry.means[0])
    assert entry.counts[0] == 0


def test_series_labels_name_the_condition() -> None:
    rows = [_row("s", 1000, 0.0, 1, 1.0), _row("s", 1000, 0.5, 1, 1.0)]
    labels = {entry.label for entry in budget_curve_series(rows)}
    # Named by the condition, not by the anneal parameter that switches it on.
    assert labels == {"No Curriculum", "Growing Window Curriculum"}


def test_series_round_trip_to_json_safe_primitives() -> None:
    rows = [_row("s", 1000, 0.5, 1, 1.0)]
    (entry,) = budget_curve_series(rows)
    payload = entry.as_dict()
    assert payload["budgets"] == [1000]
    assert payload["curriculum_fraction"] == 0.5
    assert isinstance(entry, BudgetSeries)


# -- the sampling-strategy factor ------------------------------------------


@pytest.fixture
def strategy_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="strategies",
        scenarios=("safe_corridor",),
        reward_modes=("sparse",),
        seeds=(1, 2),
        episode_budgets=(1000,),
        curriculum_fractions=(0.0, 0.5),
        curriculum_strategies=("growing", "sliding", "visit_weighted"),
    )


def test_strategies_are_reported_in_declared_order(strategy_config: ExperimentConfig) -> None:
    """Names, not numbers: the config's order is the legend's order."""
    assert strategy_config.strategies == ("growing", "sliding", "visit_weighted")
    assert ExperimentConfig().strategies == ("growing",)


def test_an_unknown_strategy_fails_before_the_grid_runs() -> None:
    config = ExperimentConfig(curriculum_strategies=("anti_curriculum",))
    with pytest.raises(ValueError):
        _ = config.strategies


def test_the_control_arm_is_run_once_not_once_per_strategy(
    strategy_config: ExperimentConfig,
) -> None:
    """Three identical no-curriculum tables would split one control arm three ways."""
    controls = [cell for cell in strategy_config.cells() if cell[3] == 0.0]
    assert {cell[4] for cell in controls} == {"growing"}
    assert len(controls) == len(strategy_config.seeds)
    # 2 seeds of control + 3 strategies x 2 seeds of treatment.
    assert strategy_config.total_runs == 8
    assert len(strategy_config.cells()) == 8
    assert strategy_config.total_train_episodes == 8 * 1000


def test_run_names_separate_the_strategies(strategy_config: ExperimentConfig) -> None:
    names = {
        strategy_config.run_name("safe_corridor", "sparse", seed, budget, curriculum, strategy)
        for _s, _r, budget, curriculum, strategy, seed in strategy_config.cells()
    }
    assert len(names) == strategy_config.total_runs
    assert "safe_corridor__sparse__cf0.5__cssliding__seed1" in names
    # The control has no strategy to name.
    assert "safe_corridor__sparse__cf0__seed1" in names


def test_run_names_ignore_the_strategy_when_only_one_is_swept(
    swept_config: ExperimentConfig,
) -> None:
    assert (
        swept_config.run_name("safe_corridor", "sparse", 1, 3000, 0.5, "growing")
        == "safe_corridor__sparse__cf0.5__ep3000__seed1"
    )


def test_conditions_are_not_pooled_across_strategies() -> None:
    rows = [
        _row("safe_corridor", 1000, 0.5, seed, 1.0, strategy)
        for strategy in ("growing", "sliding")
        for seed in (1, 2)
    ]
    assert len({condition_of(row) for row in rows}) == 2
    assert len(aggregate_rows(rows)) == 2


def test_the_arm_of_a_control_row_ignores_its_strategy() -> None:
    assert arm_of_row(_row("s", 1000, 0.0, 1, 1.0, "sliding")) == arm_of_row(
        _row("s", 1000, 0.0, 1, 1.0, "growing")
    )


def test_a_row_written_before_strategies_existed_reads_as_the_growing_window() -> None:
    legacy = _row("s", 1000, 0.5, 1, 1.0)
    del legacy["curriculum_strategy"]
    assert arm_of_row(legacy) == (0.5, "growing")


def test_each_strategy_gets_its_own_curve() -> None:
    rows = [
        _row("s", budget, 0.5, seed, 1.0, strategy)
        for strategy in ("growing", "sliding")
        for budget in (1000, 3000)
        for seed in (1, 2)
    ] + [_row("s", budget, 0.0, seed, 1.0) for budget in (1000, 3000) for seed in (1, 2)]
    series = budget_curve_series(rows)
    assert len(series) == 3
    assert {entry.label for entry in series} == {
        "No Curriculum",
        "Growing Window Curriculum",
        "Sliding Window Curriculum",
    }


def test_every_treatment_arm_is_paired_against_the_shared_control() -> None:
    rows = [
        _row("s", budget, 0.5, seed, 2.0, strategy)
        for strategy in ("growing", "sliding")
        for budget in (1000, 3000)
        for seed in (1, 2, 3)
    ] + [_row("s", budget, 0.0, seed, 1.0) for budget in (1000, 3000) for seed in (1, 2, 3)]
    (group,) = analyze_sweep(rows, metric="eval_mean_base_return")["groups"]
    strategies = {entry["curriculum_strategy"] for entry in group["paired"]}
    assert strategies == {"growing", "sliding"}
    # Every seed of the control is available to every treatment arm.
    assert all(entry["n"] == 3 for entry in group["paired"])
