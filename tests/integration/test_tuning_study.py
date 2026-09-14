"""End-to-end tests for the hyper-parameter search: budget, pruning, artefacts.

These run real (tiny) training. The study's *ranking* is meaningless while the
objective is stubbed, and nothing here asserts otherwise -- what is asserted is that
the budget is honoured, that pruning returns episodes to the ledger, and that the
artefacts say plainly which state the search was run in.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import optuna
import pytest

from mars_rover_q.cli import build_parser, main
from mars_rover_q.metrics import CheckpointRecord
from mars_rover_q.scenario import resolve_scenario
from mars_rover_q.training import TrainConfig, train
from mars_rover_q.tuning import TuningConfig, run_study

#: A study small enough for a test suite: four funded trials of 60 episodes.
TINY = {
    "trial_episode_cap": 60,
    "study_episode_budget": 240,
    "checkpoints": 4,
    "checkpoint_episodes": 2,
    "eval_episodes": 2,
}


def tiny_config(**overrides: Any) -> TuningConfig:
    """A search config sized for tests."""
    return TuningConfig(name="tiny_search", **{**TINY, **overrides})


# -- the train() seam pruning is built on ----------------------------------------


def test_a_checkpoint_callback_sees_every_mid_training_checkpoint() -> None:
    scenario = resolve_scenario("safe_corridor")
    seen: list[CheckpointRecord] = []

    def watch(record: CheckpointRecord) -> bool:
        seen.append(record)
        return True

    result = train(
        scenario,
        TrainConfig(
            scenario="safe_corridor", episodes=40, eval_checkpoints=4, checkpoint_episodes=2
        ),
        warn_on_stubs=False,
        checkpoint_callback=watch,
    )
    assert [record.episode for record in seen] == [0, 10, 20, 30]
    # The final checkpoint is measured on the finished table and is not offered to the
    # callback: there is nothing left to decide by then.
    assert [c.episode for c in result.checkpoints] == [0, 10, 20, 30, 40]
    assert result.episodes_completed == 40
    assert not result.stopped_early


def test_a_callback_returning_false_stops_training_there() -> None:
    scenario = resolve_scenario("safe_corridor")
    result = train(
        scenario,
        TrainConfig(
            scenario="safe_corridor", episodes=40, eval_checkpoints=4, checkpoint_episodes=2
        ),
        warn_on_stubs=False,
        checkpoint_callback=lambda record: record.episode < 20,
    )
    assert result.stopped_early
    assert result.episodes_completed == 20
    # The curve ends on the checkpoint that made the decision, with no duplicate final
    # point measured at the same episode index.
    assert [c.episode for c in result.checkpoints] == [0, 10, 20]
    assert len(result.records) == 20


def test_not_stopping_leaves_the_run_identical_to_one_without_a_callback() -> None:
    """Pruning may change what a trial costs, never what it learns."""
    scenario = resolve_scenario("safe_corridor")
    config = TrainConfig(
        scenario="safe_corridor", episodes=30, eval_checkpoints=3, checkpoint_episodes=2
    )
    plain = train(scenario, config, warn_on_stubs=False)
    watched = train(scenario, config, warn_on_stubs=False, checkpoint_callback=lambda _: True)
    assert (plain.q_table == watched.q_table).all()
    assert plain.total_env_steps == watched.total_env_steps


# -- the study ------------------------------------------------------------------


def test_a_study_spends_its_budget_and_no_more(tmp_path: Path) -> None:
    payload = run_study(tiny_config(), tmp_path, make_plots=False)
    budget = payload["budget"]
    assert budget["total"] == 240
    assert budget["spent"] <= budget["total"]
    assert budget["exhausted"]
    assert len(payload["trials"]) >= 4


def test_a_trial_cap_bounds_the_study_before_the_ledger_does(tmp_path: Path) -> None:
    payload = run_study(tiny_config(max_trials=2), tmp_path, make_plots=False)
    assert len(payload["trials"]) == 2
    assert not payload["budget"]["exhausted"]


def test_every_trial_ran_the_episodes_it_was_granted(tmp_path: Path) -> None:
    payload = run_study(tiny_config(), tmp_path, make_plots=False)
    for trial in payload["trials"]:
        assert trial["episodes_granted"] == 60
        assert trial["episodes_run"] <= trial["episodes_granted"]
        assert trial["episodes_run"] == 60 or trial["pruned"]


def test_the_budget_ledger_equals_the_sum_of_what_the_trials_ran(tmp_path: Path) -> None:
    payload = run_study(tiny_config(confirm_best=False), tmp_path, make_plots=False)
    assert payload["budget"]["spent"] == sum(t["episodes_run"] for t in payload["trials"])


def test_a_study_is_reproducible_from_its_sampler_seed(tmp_path: Path) -> None:
    first = run_study(tiny_config(), tmp_path / "a", make_plots=False)
    second = run_study(tiny_config(), tmp_path / "b", make_plots=False)
    assert [t["params"] for t in first["trials"]] == [t["params"] for t in second["trials"]]
    assert [t["curve"] for t in first["trials"]] == [t["curve"] for t in second["trials"]]


def test_a_different_sampler_seed_searches_elsewhere(tmp_path: Path) -> None:
    first = run_study(tiny_config(sampler_seed=0), tmp_path / "a", make_plots=False)
    second = run_study(tiny_config(sampler_seed=7), tmp_path / "b", make_plots=False)
    assert [t["params"] for t in first["trials"]] != [t["params"] for t in second["trials"]]


def test_the_curve_of_a_completed_trial_spans_its_whole_budget(tmp_path: Path) -> None:
    payload = run_study(tiny_config(), tmp_path, make_plots=False)
    for trial in payload["trials"]:
        if trial["pruned"]:
            continue
        curve = trial["curve"]
        assert curve[0][0] == 0, "a curve must start at the untrained table"
        assert curve[-1][0] == 60, "a completed trial's curve must reach the cap"
        assert [point[0] for point in curve] == sorted({point[0] for point in curve})


def test_several_seeds_are_averaged_into_one_curve(tmp_path: Path) -> None:
    payload = run_study(
        tiny_config(seeds=(1, 2), study_episode_budget=240, max_trials=1),
        tmp_path,
        make_plots=False,
    )
    trial = payload["trials"][0]
    assert trial["episodes_granted"] == 120
    assert trial["episodes_run"] == 120
    assert len(trial["curve"]) == 5


def test_the_study_records_that_its_objective_is_stubbed(
    tmp_path: Path, stubbed_tuning_objective: None
) -> None:
    payload = run_study(tiny_config(), tmp_path, make_plots=False)
    assert payload["search_is_meaningful"] is False
    assert "score_learning_curve" in payload["pending_human_functions"]
    assert "pareto_front" in payload["pending_human_functions"]
    # The trials themselves are real training runs: it is the ranking that is not real.
    assert payload["learning_is_meaningful"] is True


def test_the_stubbed_search_warns_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], stubbed_tuning_objective: None
) -> None:
    run_study(tiny_config(max_trials=1), tmp_path, make_plots=False)
    assert "HYPERPARAMETER SEARCH STUBBED" in capsys.readouterr().err


def test_the_reference_return_is_the_maps_best_affordable_sample(tmp_path: Path) -> None:
    payload = run_study(tiny_config(max_trials=1), tmp_path, make_plots=False)
    assert payload["reference_return"] == 160.0


def test_an_explicit_reference_return_overrides_the_map(tmp_path: Path) -> None:
    payload = run_study(
        tiny_config(max_trials=1, reference_return=90.0), tmp_path, make_plots=False
    )
    assert payload["reference_return"] == 90.0


# -- artefacts ------------------------------------------------------------------


def test_a_study_writes_its_json_and_csv(tmp_path: Path) -> None:
    run_study(tiny_config(confirm_best=False), tmp_path, make_plots=False)
    payload = json.loads((tmp_path / "study.json").read_text(encoding="utf-8"))
    assert payload["config"]["scenario"] == "safe_corridor"
    assert payload["environment"]["python"]

    with (tmp_path / "trials.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(payload["trials"])
    assert "param_learning_rate" in rows[0]


def test_a_parameter_no_trial_sampled_leaves_an_empty_cell_not_a_zero(tmp_path: Path) -> None:
    """The conditional space means trials do not share columns; absent is not zero."""
    run_study(tiny_config(confirm_best=False), tmp_path, make_plots=False)
    with (tmp_path / "trials.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    conditional = [
        row.get("param_curriculum_window_fraction", "")
        for row in rows
        if row.get("param_curriculum_strategy") not in ("sliding",)
    ]
    assert all(cell == "" for cell in conditional)


def test_a_study_writes_its_figures(tmp_path: Path) -> None:
    payload = run_study(tiny_config(max_trials=2, confirm_best=False), tmp_path, make_plots=True)
    written = sorted(path.name for path in (tmp_path / "plots").iterdir())
    assert written == ["pareto_front.png", "search_progress.png", "top_trial_curves.png"]
    assert len(payload["plots"]) == 3


def test_the_best_trial_is_confirmed_as_a_full_run_directory(tmp_path: Path) -> None:
    payload = run_study(
        tiny_config(study_episode_budget=180, max_trials=2), tmp_path, make_plots=False
    )
    confirmation = payload["confirmation_run"]
    assert confirmation is not None
    assert Path(confirmation["run_dir"]).name == "best_run"
    assert (tmp_path / "best_run" / "manifest.json").exists()
    assert (tmp_path / "best_run" / "q_table.npy").exists()
    assert confirmation["eval_episodes"] == 2


def test_the_confirmation_run_is_charged_to_the_same_ledger(tmp_path: Path) -> None:
    """A search's cost includes confirming its own winner."""
    payload = run_study(
        tiny_config(study_episode_budget=180, max_trials=2), tmp_path, make_plots=False
    )
    trial_episodes = sum(t["episodes_run"] for t in payload["trials"])
    assert payload["budget"]["spent"] == trial_episodes + payload["confirmation_run"]["episodes"]


def test_an_exhausted_budget_skips_the_confirmation_run(tmp_path: Path) -> None:
    payload = run_study(tiny_config(), tmp_path, make_plots=False)
    assert payload["budget"]["exhausted"]
    assert payload["confirmation_run"] is None


def test_confirmation_can_be_switched_off(tmp_path: Path) -> None:
    payload = run_study(
        tiny_config(study_episode_budget=180, max_trials=1, confirm_best=False),
        tmp_path,
        make_plots=False,
    )
    assert payload["confirmation_run"] is None
    assert not (tmp_path / "best_run").exists()


# -- pruning --------------------------------------------------------------------


class _PruneEverything(optuna.pruners.BasePruner):
    """A pruner that cuts every trial as soon as it has reported twice.

    Going through Optuna's own pruner interface rather than stubbing
    ``trial.should_prune`` keeps the test on the path a real pruner's decision takes:
    the checkpoint callback asks, the pruner answers, training stops, and the study
    tells Optuna the trial was pruned.
    """

    def prune(self, study: optuna.study.Study, trial: Any) -> bool:
        return len(trial.intermediate_values) >= 2


def test_a_pruned_trial_costs_less_than_a_full_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mars_rover_q import tuning

    config = tiny_config(study_episode_budget=600)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.sampler_seed),
        pruner=_PruneEverything(),
    )
    monkeypatch.setattr(tuning, "_make_study", lambda _: study)
    payload = run_study(config, tmp_path, make_plots=False)

    assert all(trial["pruned"] for trial in payload["trials"])
    assert all(trial["score"] is None for trial in payload["trials"])
    assert all(trial["episodes_run"] < trial["episodes_granted"] for trial in payload["trials"])
    # The returned episodes buy extra trials: ten funded at full price, more in fact.
    assert len(payload["trials"]) > config.fundable_trials
    assert payload["best_trial"] is None


def test_a_study_where_nothing_completes_still_writes_its_artefacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mars_rover_q import tuning

    study = optuna.create_study(direction="maximize", pruner=_PruneEverything())
    monkeypatch.setattr(tuning, "_make_study", lambda _: study)
    payload = run_study(tiny_config(max_trials=2), tmp_path, make_plots=True)

    assert payload["best_trial"] is None
    assert payload["confirmation_run"] is None
    assert (tmp_path / "study.json").exists()
    assert (tmp_path / "plots" / "search_progress.png").exists()


# -- the CLI --------------------------------------------------------------------


def test_the_parser_accepts_the_documented_tune_workflow() -> None:
    args = build_parser().parse_args(
        [
            "tune",
            "--scenario",
            "safe_corridor",
            "--trial-episodes",
            "6000",
            "--total-episodes",
            "2000000",
            "--trials",
            "50",
        ]
    )
    assert args.command == "tune"
    assert callable(args.func)


def test_the_tune_command_runs_a_study(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], stubbed_tuning_objective: None
) -> None:
    code = main(
        [
            "tune",
            "--scenario",
            "safe_corridor",
            "--trial-episodes",
            "60",
            "--total-episodes",
            "180",
            "--checkpoints",
            "4",
            "--checkpoint-episodes",
            "2",
            "--eval-episodes",
            "2",
            "--no-confirm-best",
            "--no-plots",
            "--output",
            str(tmp_path / "study"),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "study written to" in captured.out
    assert "episodes spent" in captured.out
    assert "TEACHING STATE" in captured.err
    assert (tmp_path / "study" / "study.json").exists()


def test_the_tune_commands_teaching_line_tracks_the_payload_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run against whatever state the repository is in: the line and the flag must agree.

    Asserted as a coupling rather than a fixed expectation so the test keeps its meaning
    after the search objective lands -- the warning must disappear exactly when the
    ranking becomes real, and never be printed over a ranking that is.
    """
    code = main(
        [
            "tune",
            "--scenario",
            "safe_corridor",
            "--trial-episodes",
            "60",
            "--total-episodes",
            "180",
            "--checkpoints",
            "4",
            "--checkpoint-episodes",
            "2",
            "--eval-episodes",
            "2",
            "--no-confirm-best",
            "--no-plots",
            "--output",
            str(tmp_path / "study"),
        ]
    )
    assert code == 0
    warned = "the search objective is stubbed" in capsys.readouterr().err
    payload = json.loads((tmp_path / "study" / "study.json").read_text())
    assert warned is not payload["search_is_meaningful"]


def test_the_tune_command_honours_a_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "search.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "from_file",
                "scenario": "safe_corridor",
                "trial_episode_cap": 60,
                "study_episode_budget": 120,
                "checkpoints": 4,
                "checkpoint_episodes": 2,
                "eval_episodes": 2,
                "confirm_best": False,
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "tune",
                "--config",
                str(config_path),
                "--no-plots",
                "--output",
                str(tmp_path / "study"),
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "study" / "study.json").read_text(encoding="utf-8"))
    assert payload["config"]["name"] == "from_file"
    assert payload["budget"]["total"] == 120


def test_a_flag_overrides_the_config_file_and_leaves_the_rest_alone(tmp_path: Path) -> None:
    config_path = tmp_path / "search.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "from_file",
                "trial_episode_cap": 60,
                "study_episode_budget": 600,
                "checkpoints": 4,
                "checkpoint_episodes": 2,
                "sampler_seed": 3,
                "confirm_best": False,
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "tune",
                "--config",
                str(config_path),
                "--total-episodes",
                "120",
                "--no-plots",
                "--output",
                str(tmp_path / "study"),
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "study" / "study.json").read_text(encoding="utf-8"))
    assert payload["budget"]["total"] == 120
    assert payload["config"]["sampler_seed"] == 3
    assert payload["config"]["trial_episode_cap"] == 60
