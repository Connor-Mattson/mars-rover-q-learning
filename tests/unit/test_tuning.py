"""The machinery around the human-owned objective: ledger, space, curves, config.

Nothing here asserts anything about :func:`score_learning_curve` or
:func:`pareto_front` beyond the stub probe; their contracts live in
``tests/human_todo/test_tuning_objective.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mars_rover_q.curriculum import (
    DEFAULT_WEIGHT_EXPONENT,
    DEFAULT_WINDOW_FRACTION,
    CurriculumStrategy,
)
from mars_rover_q.numerics import install_numeric_guard
from mars_rover_q.rewards import RewardMode
from mars_rover_q.scenario import resolve_scenario
from mars_rover_q.state import BatteryEncoding
from mars_rover_q.tuning import (
    HUMAN_OWNED_FUNCTIONS,
    EpisodeLedger,
    TrialOutcome,
    TuningConfig,
    _make_study,
    average_curves,
    best_affordable_return,
    curve_peak,
    suggest_trial_params,
    train_config_for,
    tuning_stub_status,
)
from tests.conftest import make_scenario


class RecordingTrial:
    """A stand-in for ``optuna.trial.Trial`` that records what was asked of it.

    The search space is a specification of which parameters exist, on what ranges,
    and under what conditions -- all of which is observable from the calls alone. A
    real ``Trial`` would need a study, a sampler, and a storage backend to answer the
    same questions, and would answer them stochastically.
    """

    def __init__(self, choices: dict[str, Any] | None = None, *, position: float = 0.5) -> None:
        self.choices = choices or {}
        self.position = position
        self.floats: dict[str, tuple[float, float, bool]] = {}
        self.categoricals: dict[str, list[Any]] = {}
        self.order: list[str] = []

    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool = False, step: float | None = None
    ) -> float:
        self.floats[name] = (low, high, log)
        self.order.append(name)
        if name in self.choices:
            return float(self.choices[name])
        if log:
            return float(math.exp(math.log(low) + self.position * (math.log(high) - math.log(low))))
        return float(low + self.position * (high - low))

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        self.categoricals[name] = list(choices)
        self.order.append(name)
        if name in self.choices:
            return self.choices[name]
        return choices[0]


# -- EpisodeLedger ---------------------------------------------------------------


def test_a_fresh_ledger_has_its_whole_budget_available() -> None:
    ledger = EpisodeLedger(total=1000, per_trial=100)
    assert (ledger.remaining, ledger.exhausted, ledger.grant()) == (1000, False, 100)


def test_recording_a_trial_spends_what_it_ran() -> None:
    ledger = EpisodeLedger(total=1000, per_trial=100)
    ledger.record(40)
    assert (ledger.spent, ledger.remaining) == (40, 960)


def test_a_pruned_trial_returns_its_unspent_episodes_to_the_pool() -> None:
    """The point of pruning: a short trial leaves room for an extra full one."""
    ledger = EpisodeLedger(total=500, per_trial=100)
    for _ in range(5):
        ledger.grant()
        ledger.record(50)
    assert ledger.remaining == 250
    assert not ledger.exhausted


def test_a_ledger_is_exhausted_when_it_cannot_fund_a_full_trial() -> None:
    ledger = EpisodeLedger(total=250, per_trial=100)
    ledger.record(200)
    assert ledger.remaining == 50
    assert ledger.exhausted
    assert ledger.grant() == 0


def test_grants_are_never_partial() -> None:
    """A short grant would produce a trial whose x-axis nothing else shares."""
    ledger = EpisodeLedger(total=150, per_trial=100)
    ledger.record(100)
    assert ledger.grant() == 0


def test_the_ledger_cannot_be_overspent() -> None:
    ledger = EpisodeLedger(total=100, per_trial=100)
    with pytest.raises(ValueError, match="exceeds"):
        ledger.record(101)


def test_a_negative_charge_is_rejected() -> None:
    ledger = EpisodeLedger(total=100, per_trial=100)
    with pytest.raises(ValueError, match="negative"):
        ledger.record(-1)


@pytest.mark.parametrize(
    ("total", "per_trial"),
    [(100, 0), (100, -1), (50, 100)],
)
def test_an_unusable_ledger_is_rejected_at_construction(total: int, per_trial: int) -> None:
    with pytest.raises(ValueError):
        EpisodeLedger(total=total, per_trial=per_trial)


def test_the_ledger_serialises_what_a_reader_needs() -> None:
    ledger = EpisodeLedger(total=1000, per_trial=100)
    ledger.record(120)
    assert ledger.as_dict() == {
        "total": 1000,
        "per_trial": 100,
        "spent": 120,
        "remaining": 880,
        "exhausted": False,
    }


# -- TuningConfig ----------------------------------------------------------------


def test_a_trial_costs_the_cap_once_per_seed() -> None:
    config = TuningConfig(trial_episode_cap=1000, seeds=(1, 2, 3))
    assert config.episodes_per_trial == 3000


def test_fundable_trials_is_the_floor_not_the_expectation() -> None:
    config = TuningConfig(trial_episode_cap=1000, seeds=(1,), study_episode_budget=25_500)
    assert config.fundable_trials == 25


def test_the_default_budget_is_two_million_episodes() -> None:
    assert TuningConfig().study_episode_budget == 2_000_000


def test_the_default_trial_cap_sits_below_convergence() -> None:
    """The cap has to bite, or every converging setting scores the same."""
    assert TuningConfig().trial_episode_cap < 4000 * 2


def test_a_budget_too_small_for_one_trial_is_rejected() -> None:
    with pytest.raises(ValueError, match="single trial"):
        TuningConfig(trial_episode_cap=1000, seeds=(1, 2), study_episode_budget=1500)


def test_repeated_seeds_are_rejected() -> None:
    """Two identical seeds are one measurement charged twice."""
    with pytest.raises(ValueError, match="distinct"):
        TuningConfig(seeds=(1, 1))


def test_an_empty_seed_tuple_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TuningConfig(seeds=())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trial_episode_cap": 0},
        {"max_trials": 0},
        {"checkpoints": 1},
        {"checkpoint_episodes": 0},
        {"reference_return": 0.0},
        {"reference_return": -1.0},
        {"reference_return": math.inf},
    ],
)
def test_unusable_search_settings_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TuningConfig(**kwargs)


def test_the_config_round_trips_through_json(tmp_path: Path) -> None:
    config = TuningConfig(
        name="search",
        scenario="risk_value_tradeoff",
        reward_mode=RewardMode.POTENTIAL,
        seeds=(1, 2),
        trial_episode_cap=1000,
        study_episode_budget=100_000,
        battery_encoding=BatteryEncoding.DENSE,
    )
    path = tmp_path / "search.json"
    path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    assert TuningConfig.from_file(path) == config


@pytest.mark.parametrize(
    "name",
    [
        "configs/tuning/safe_corridor_search.json",
        "configs/tuning/safe_corridor_search_v2.json",
        "configs/tuning/risk_value_tradeoff_search_v2.json",
        "configs/tuning/shaping_trap_search_v2.json",
        "configs/tuning/smoke.json",
    ],
)
def test_every_shipped_search_config_loads(name: str) -> None:
    """A bundled config that no longer constructs is a broken documented workflow."""
    config = TuningConfig.from_file(name)
    assert config.fundable_trials >= 1
    assert config.episodes_per_trial == config.trial_episode_cap * len(config.seeds)


def test_the_smoke_config_can_fund_its_confirmation_run() -> None:
    """The plumbing check has to exercise the confirmation path, or it checks less.

    ``max_trials`` caps the spend below the budget on purpose: without it a study that
    prunes heavily funds extra trials until the ledger cannot pay for confirming the
    winner, and the one end-to-end command in the docs would skip that branch.
    """
    config = TuningConfig.from_file("configs/tuning/smoke.json")
    assert config.confirm_best
    assert config.max_trials is not None
    worst_case_spend = config.max_trials * config.episodes_per_trial
    assert config.study_episode_budget - worst_case_spend >= config.episodes_per_trial


def test_unknown_keys_in_a_config_file_are_ignored(tmp_path: Path) -> None:
    """``as_dict`` writes derived fields that are not constructor arguments."""
    path = tmp_path / "search.json"
    path.write_text(json.dumps({"scenario": "safe_corridor", "nonsense": 3}), encoding="utf-8")
    assert TuningConfig.from_file(path).scenario == "safe_corridor"


# -- the search space ------------------------------------------------------------


def test_the_agent_knobs_are_always_suggested() -> None:
    trial = RecordingTrial()
    params = suggest_trial_params(trial, TuningConfig())
    assert set(trial.floats) >= {
        "learning_rate",
        "gamma",
        "initial_q",
        "epsilon_start",
        "epsilon_end_ratio",
        "epsilon_decay_fraction",
    }
    assert set(params) >= set(trial.floats)


def test_the_multiplicative_parameters_are_sampled_on_a_log_axis() -> None:
    trial = RecordingTrial()
    suggest_trial_params(trial, TuningConfig())
    assert trial.floats["learning_rate"][2] is True
    assert trial.floats["epsilon_end_ratio"][2] is True


def test_the_additive_parameters_are_not_log_sampled() -> None:
    trial = RecordingTrial()
    suggest_trial_params(trial, TuningConfig())
    for name in ("gamma", "initial_q", "epsilon_start", "epsilon_decay_fraction"):
        assert trial.floats[name][2] is False, name


def test_optimistic_initialisation_can_exceed_the_maps_best_return() -> None:
    """Initial Q is an exploration knob, so its range has to reach past the payoff."""
    trial = RecordingTrial()
    suggest_trial_params(trial, TuningConfig())
    assert trial.floats["initial_q"][1] >= best_affordable_return(resolve_scenario("safe_corridor"))


def test_the_learning_rate_range_stays_inside_the_agents_contract() -> None:
    trial = RecordingTrial()
    suggest_trial_params(trial, TuningConfig())
    low, high, _ = trial.floats["learning_rate"]
    assert low > 0.0 and high <= 1.0


def test_the_gamma_range_stays_inside_the_agents_contract() -> None:
    trial = RecordingTrial()
    suggest_trial_params(trial, TuningConfig())
    low, high, _ = trial.floats["gamma"]
    assert low > 0.0 and high <= 1.0


def test_epsilon_floor_is_sampled_as_a_fraction_of_its_ceiling() -> None:
    """Sampling the two rates independently would put part of the space out of bounds."""
    trial = RecordingTrial()
    suggest_trial_params(trial, TuningConfig())
    assert "epsilon_end" not in trial.floats
    assert trial.floats["epsilon_end_ratio"][1] <= 1.0


def test_the_suggested_epsilon_schedule_is_always_constructible() -> None:
    for position in (0.0, 0.01, 0.5, 0.99, 1.0):
        trial = RecordingTrial(position=position)
        params = suggest_trial_params(trial, TuningConfig())
        config = train_config_for(TuningConfig(), params, seed=1, episodes=100)
        # Raises if end > start, which is the corner the ratio parameterisation removes.
        assert config.agent_config().epsilon.end <= config.agent_config().epsilon.start


def test_the_curriculum_is_skipped_entirely_when_it_is_not_searched() -> None:
    trial = RecordingTrial()
    params = suggest_trial_params(trial, TuningConfig(search_curriculum=False))
    assert not any(name.startswith("curriculum") for name in params)
    assert "use_curriculum" not in trial.categoricals


def test_the_no_curriculum_arm_gets_its_own_categorical_switch() -> None:
    """A continuous fraction would essentially never land on exactly zero."""
    trial = RecordingTrial({"use_curriculum": False})
    params = suggest_trial_params(trial, TuningConfig())
    assert trial.categoricals["use_curriculum"] == [False, True]
    assert params["use_curriculum"] is False
    assert "curriculum_fraction" not in params


def test_the_growing_window_suggests_neither_strategy_knob() -> None:
    trial = RecordingTrial(
        {"use_curriculum": True, "curriculum_strategy": CurriculumStrategy.GROWING.value}
    )
    params = suggest_trial_params(trial, TuningConfig())
    assert params["curriculum_strategy"] == CurriculumStrategy.GROWING.value
    assert "curriculum_window_fraction" not in params
    assert "curriculum_weight_exponent" not in params


def test_the_sliding_window_suggests_only_its_own_width() -> None:
    trial = RecordingTrial(
        {"use_curriculum": True, "curriculum_strategy": CurriculumStrategy.SLIDING.value}
    )
    params = suggest_trial_params(trial, TuningConfig())
    assert "curriculum_window_fraction" in params
    assert "curriculum_weight_exponent" not in params


def test_the_visit_weighted_schedule_suggests_only_its_exponent() -> None:
    trial = RecordingTrial(
        {"use_curriculum": True, "curriculum_strategy": CurriculumStrategy.VISIT_WEIGHTED.value}
    )
    params = suggest_trial_params(trial, TuningConfig())
    assert "curriculum_weight_exponent" in params
    assert "curriculum_window_fraction" not in params


def test_the_visit_exponent_range_includes_the_uniform_control() -> None:
    """``0.0`` is the uniform draw, and is the control for the weighting itself."""
    trial = RecordingTrial(
        {"use_curriculum": True, "curriculum_strategy": CurriculumStrategy.VISIT_WEIGHTED.value}
    )
    suggest_trial_params(trial, TuningConfig())
    assert trial.floats["curriculum_weight_exponent"][0] == 0.0


def test_every_curriculum_strategy_is_reachable() -> None:
    trial = RecordingTrial({"use_curriculum": True})
    suggest_trial_params(trial, TuningConfig())
    assert trial.categoricals["curriculum_strategy"] == [s.value for s in CurriculumStrategy]


# -- params -> TrainConfig -------------------------------------------------------


def test_a_suggested_setting_becomes_a_runnable_train_config() -> None:
    params = {
        "learning_rate": 0.3,
        "gamma": 0.97,
        "initial_q": 12.0,
        "epsilon_start": 0.8,
        "epsilon_end_ratio": 0.1,
        "epsilon_decay_fraction": 0.4,
    }
    config = train_config_for(TuningConfig(scenario="safe_corridor"), params, seed=7, episodes=500)
    assert (config.scenario, config.seed, config.episodes) == ("safe_corridor", 7, 500)
    assert (config.learning_rate, config.gamma, config.initial_q) == (0.3, 0.97, 12.0)
    assert config.epsilon_start == 0.8
    assert config.epsilon_end == pytest.approx(0.08)
    assert config.curriculum_fraction == 0.0


def test_the_checkpoint_schedule_comes_from_the_search_config() -> None:
    tuning = TuningConfig(checkpoints=7, checkpoint_episodes=11)
    config = train_config_for(tuning, {}, seed=1, episodes=500)
    assert (config.eval_checkpoints, config.checkpoint_episodes) == (7, 11)


def test_absent_conditional_params_fall_back_to_repository_defaults() -> None:
    """What makes the conditional space safe: an unsampled knob is simply the default."""
    params = {"use_curriculum": True, "curriculum_fraction": 0.5}
    config = train_config_for(TuningConfig(), params, seed=1, episodes=500)
    assert config.curriculum_window_fraction == DEFAULT_WINDOW_FRACTION
    assert config.curriculum_weight_exponent == DEFAULT_WEIGHT_EXPONENT
    assert config.curriculum_strategy is CurriculumStrategy.GROWING


def test_the_curriculum_switch_gates_its_fraction() -> None:
    """A fraction left over from a previous suggestion must not leak into the run."""
    params = {"use_curriculum": False, "curriculum_fraction": 0.7}
    config = train_config_for(TuningConfig(), params, seed=1, episodes=500)
    assert config.curriculum_fraction == 0.0


def test_the_battery_encoding_is_the_searchs_not_the_trials() -> None:
    tuning = TuningConfig(battery_encoding=BatteryEncoding.DENSE)
    assert train_config_for(tuning, {}, seed=1, episodes=10).battery_encoding is (
        BatteryEncoding.DENSE
    )


def test_the_battery_encoding_stays_out_of_the_space_by_default() -> None:
    """Off unless asked for: it changes the table, not the rule that writes into it.

    Every other study in the repository holds the representation fixed so that the
    factor it is varying is the only one moving. A search that silently added this
    axis would make those studies and this one incomparable.
    """
    trial = RecordingTrial()
    params = suggest_trial_params(trial, TuningConfig())
    assert "battery_encoding" not in params
    assert "battery_encoding" not in trial.categoricals


def test_the_searched_battery_encoding_offers_both_representations() -> None:
    trial = RecordingTrial()
    suggest_trial_params(trial, TuningConfig(search_battery_encoding=True))
    assert trial.categoricals["battery_encoding"] == [
        BatteryEncoding.AFFORDABILITY.value,
        BatteryEncoding.DENSE.value,
    ]


@pytest.mark.parametrize("encoding", list(BatteryEncoding))
def test_a_searched_encoding_overrides_the_study_default(encoding: BatteryEncoding) -> None:
    """The trial's draw wins, whichever way the study-level default points.

    Parametrising over both members is what makes this a real assertion: agreeing
    with the default by accident would pass a single-value version of this test.
    """
    trial = RecordingTrial({"battery_encoding": encoding.value})
    params = suggest_trial_params(trial, TuningConfig(search_battery_encoding=True))
    other = (
        BatteryEncoding.DENSE
        if encoding is BatteryEncoding.AFFORDABILITY
        else BatteryEncoding.AFFORDABILITY
    )
    config = TuningConfig(battery_encoding=other, search_battery_encoding=True)
    assert train_config_for(config, params, seed=1, episodes=10).battery_encoding is encoding


def test_a_searched_encoding_survives_the_csv_round_trip() -> None:
    """``as_row`` flattens params into columns, and this one is a string, not a float."""
    trial = RecordingTrial({"battery_encoding": BatteryEncoding.DENSE.value})
    params = suggest_trial_params(trial, TuningConfig(search_battery_encoding=True))
    outcome = TrialOutcome(
        number=0,
        params=params,
        episodes_granted=10,
        episodes_run=10,
        score=0.5,
        best_return=90.0,
        episodes_to_best=10,
        final_return=90.0,
        final_success_rate=1.0,
        pruned=False,
    )
    assert outcome.as_row()["param_battery_encoding"] == BatteryEncoding.DENSE.value


# -- curve helpers ---------------------------------------------------------------


def test_the_reference_return_is_the_best_affordable_sample() -> None:
    scenario = resolve_scenario("safe_corridor")
    assert best_affordable_return(scenario) == 160.0


def test_an_unaffordable_sample_is_not_the_reference() -> None:
    """A return the battery cannot pay for is not a ceiling the agent can be held to.

    On the tiny fixture the biosignature round trip costs 9 and the other two cost 5,
    so a battery of 6 leaves the 90-point sample as the best the map can pay.
    """
    assert best_affordable_return(make_scenario(battery_capacity=6)) == 90.0


def test_a_map_with_nothing_affordable_has_no_return_scale() -> None:
    with pytest.raises(ValueError, match="no sample"):
        best_affordable_return(make_scenario(battery_capacity=2))


def test_the_curve_peak_reads_the_best_value() -> None:
    assert curve_peak(((0, -100.0), (500, 40.0), (1000, 90.0))) == (1000, 90.0)


def test_the_curve_peak_reads_the_earliest_attainment() -> None:
    """A curve that plateaus was fast; its last point is not when it got there."""
    assert curve_peak(((0, 0.0), (500, 90.0), (1000, 90.0))) == (500, 90.0)


def test_the_curve_peak_of_an_empty_curve_is_an_error() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        curve_peak(())


def test_curves_average_pointwise() -> None:
    first = ((0, 0.0), (100, 40.0))
    second = ((0, 0.0), (100, 80.0))
    assert average_curves([first, second]) == ((0, 0.0), (100, 60.0))


def test_averaging_one_curve_returns_it() -> None:
    curve = ((0, 0.0), (100, 40.0))
    assert average_curves([curve]) == curve


def test_averaging_curves_on_different_episode_axes_is_an_error() -> None:
    with pytest.raises(ValueError, match="episode axis"):
        average_curves([((0, 0.0), (100, 40.0)), ((0, 0.0), (200, 40.0))])


def test_averaging_no_curves_is_an_error() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        average_curves([])


# -- the teaching probe ----------------------------------------------------------


def test_the_stub_probe_names_exactly_the_human_owned_functions(
    stubbed_tuning_objective: None,
) -> None:
    """Asserted as equality: a probe that can never clear would pass a subset check.

    The placeholders are installed by the fixture rather than read off the module, so
    this keeps testing the probe once the real implementations land.
    """
    assert tuning_stub_status() == HUMAN_OWNED_FUNCTIONS


def test_the_probe_reports_a_working_objective_as_done() -> None:
    """The probe must be satisfiable -- checked against a minimal honest scorer."""
    import mars_rover_q.tuning as tuning

    def area(
        curve: Sequence[tuple[int, float]], episode_cap: int, reference_return: float
    ) -> float:
        values = [max(0.0, min(1.0, value / reference_return)) for _, value in curve]
        xs = [episode / episode_cap for episode, _ in curve]
        if len(values) == 1:
            return values[0]
        return sum(
            (values[i] + values[i + 1]) / 2 * (xs[i + 1] - xs[i]) for i in range(len(values) - 1)
        ) / max(1e-12, xs[-1] - xs[0])

    def front(outcomes: Sequence[TrialOutcome]) -> list[TrialOutcome]:
        return [
            candidate
            for candidate in outcomes
            if not any(
                other.episodes_to_best <= candidate.episodes_to_best
                and other.best_return >= candidate.best_return
                and (
                    other.episodes_to_best < candidate.episodes_to_best
                    or other.best_return > candidate.best_return
                )
                for other in outcomes
            )
        ]

    original = (tuning.score_learning_curve, tuning.pareto_front)
    try:
        tuning.score_learning_curve = area
        tuning.pareto_front = front
        assert tuning.tuning_stub_status() == ()
    finally:
        tuning.score_learning_curve, tuning.pareto_front = original


def test_the_trial_outcome_row_flattens_its_parameters() -> None:
    outcome = TrialOutcome(
        number=3,
        params={"learning_rate": 0.2, "use_curriculum": True},
        episodes_granted=1000,
        episodes_run=1000,
        score=0.4,
        best_return=90.0,
        episodes_to_best=500,
        final_return=90.0,
        final_success_rate=1.0,
        pruned=False,
        curve=((0, 0.0), (500, 90.0)),
    )
    row = outcome.as_row()
    assert row["param_learning_rate"] == 0.2
    assert row["param_use_curriculum"] is True
    assert "curve" not in row
    assert outcome.as_dict()["curve"] == [[0, 0.0], [500, 90.0]]


# -- the sampler under the project's floating-point guard -------------------------


def _drive_sampler(trials: int) -> None:
    """Ask the real study for ``trials`` suggestions over the real search space.

    No training: what is under test is the sampler's own arithmetic, and a trial only
    has to be told *a* score for the sampler to model it.
    """
    import optuna

    config = TuningConfig()
    study = _make_study(config)
    for number in range(trials):
        trial = study.ask()
        suggest_trial_params(trial, config)
        # A spread with many ties and several pruned trials, which is what the real
        # study reports and what shrinks the sampler's kernels onto a few points.
        if number % 3 == 0:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial, 0.0 if number % 2 else 0.01 * (number % 5))


def test_the_sampler_survives_the_numeric_guard_past_its_startup_trials() -> None:
    """A regression guard: ``seterr(all="raise")`` used to kill the study at trial 10.

    TPE samples from the prior until it has ``startup_trials`` of history, then starts
    scoring candidates through the log-sum-exp identity -- which evaluates ``exp`` of
    large negative numbers and needs them to flush to zero. So the crash was invisible
    to every test short enough to stay inside the startup trials, and arrived a third
    of an hour into a real search.
    """
    previous = np.geterr()
    install_numeric_guard()
    try:
        _drive_sampler(60)
    finally:
        np.seterr(**previous)
