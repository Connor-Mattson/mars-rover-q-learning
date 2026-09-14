"""CLI smoke tests. Pygame-driven commands are exercised through their parsers only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mars_rover_q.cli import build_parser, main
from mars_rover_q.scenario import resolve_scenario
from mars_rover_q.state import NUM_PAYLOAD_STATES


@pytest.mark.parametrize(
    "argv",
    [
        ["play", "--scenario", "safe_corridor"],
        ["train", "--scenario", "safe_corridor", "--reward", "sparse", "--seed", "1"],
        ["train", "--scenario", "safe_corridor", "--dense-battery"],
        ["play", "--scenario", "safe_corridor", "--dense-battery"],
        ["evaluate", "--run", "some/run", "--episodes", "500"],
        ["experiment", "--config", "configs/experiments/reward_comparison.json"],
        ["replay", "--run", "some/run", "--episode", "best"],
        ["curriculum", "--scenario", "safe_corridor", "--curriculum-fraction", "0.5"],
    ],
)
def test_parser_accepts_every_documented_workflow(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]
    assert callable(args.func)


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_scenarios_command_lists_all_three(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scenarios"]) == 0
    output = capsys.readouterr().out
    for name in ("safe_corridor", "risk_value_tradeoff", "shaping_trap"):
        assert name in output


def test_train_writes_a_run_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "run"
    code = main(
        [
            "train",
            "--scenario",
            "safe_corridor",
            "--reward",
            "sparse",
            "--seed",
            "1",
            "--episodes",
            "2",
            "--eval-episodes",
            "2",
            "--output",
            str(run_dir),
            "--quiet",
            "--no-figs",
        ]
    )
    assert code == 0
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "q_table.npy").exists()
    assert "run written to" in capsys.readouterr().out


def test_evaluate_reads_back_a_saved_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    main(
        [
            "train",
            "--scenario",
            "safe_corridor",
            "--episodes",
            "2",
            "--eval-episodes",
            "0",
            "--output",
            str(run_dir),
            "--quiet",
            "--no-figs",
        ]
    )
    capsys.readouterr()
    assert main(["evaluate", "--run", str(run_dir), "--episodes", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["episodes"] == 3
    assert "success_rate" in payload


def test_train_accepts_every_reward_mode(tmp_path: Path) -> None:
    for mode in ("sparse", "naive_dense", "potential"):
        run_dir = tmp_path / mode
        assert (
            main(
                [
                    "train",
                    "--scenario",
                    "safe_corridor",
                    "--reward",
                    mode,
                    "--episodes",
                    "1",
                    "--eval-episodes",
                    "0",
                    "--output",
                    str(run_dir),
                    "--quiet",
                    "--no-figs",
                ]
            )
            == 0
        )
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["config"]["reward_mode"] == mode


def test_unknown_reward_mode_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--reward", "dense_but_clever"])


def test_experiment_command_runs_a_tiny_grid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "smoke.json"
    config.write_text(
        json.dumps(
            {
                "name": "cli_smoke",
                "scenarios": ["safe_corridor"],
                "reward_modes": ["sparse"],
                "seeds": [1],
                "episodes": 2,
                "eval_episodes": 2,
                "threshold_window": 2,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert main(["experiment", "--config", str(config), "--output", str(out)]) == 0
    assert (out / "summary.json").exists()
    assert "experiment written to" in capsys.readouterr().out


def test_replay_requires_a_stored_episode(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    main(
        [
            "train",
            "--scenario",
            "safe_corridor",
            "--episodes",
            "1",
            "--eval-episodes",
            "0",
            "--output",
            str(run_dir),
            "--quiet",
            "--no-figs",
        ]
    )
    with pytest.raises(SystemExit, match="no stored best episode"):
        main(["replay", "--run", str(run_dir)])


def test_curriculum_command_reports_the_pool(capsys: pytest.CaptureFixture[str]) -> None:
    """Inspecting the curriculum must work whether or not the pool is populated."""
    assert main(["curriculum", "--scenario", "safe_corridor", "--episodes", "200"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out.split("\n\n")[0])
    assert payload["requested"] is True
    assert payload["anneal_fraction"] == 0.5
    assert payload["canonical_difficulty"] > 0
    if payload["pool_size"]:
        # Every strategy is dry-run side by side, which is the comparison the
        # command exists for.
        assert "simulated starts by anneal progress" in captured.out
        for strategy in ("growing", "sliding", "visit_weighted"):
            assert strategy in captured.out
    else:
        assert "no admissible start states" in captured.err


def test_curriculum_command_dry_runs_one_strategy_on_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "curriculum",
                "--scenario",
                "safe_corridor",
                "--episodes",
                "200",
                "--strategy",
                "sliding",
                "--window-fraction",
                "0.1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "window_fraction=0.1" in output
    assert "visit_weighted" not in output


def test_train_reports_state_coverage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "train",
            "--scenario",
            "safe_corridor",
            "--episodes",
            "2",
            "--eval-episodes",
            "0",
            "--curriculum-fraction",
            "0.5",
            "--output",
            str(tmp_path / "run"),
            "--quiet",
            "--no-figs",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "tied (never-decided) states:" in output
    assert "curriculum: strategy=growing pool=" in output

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["curriculum_fraction"] == 0.5
    assert manifest["curriculum"]["requested"] is True
    assert 0.0 <= manifest["tied_state_fraction"] <= 1.0


def _train_args(run_dir: Path, *extra: str) -> list[str]:
    return [
        "train",
        "--scenario",
        "safe_corridor",
        "--episodes",
        "2",
        "--eval-episodes",
        "0",
        "--output",
        str(run_dir),
        "--quiet",
        *extra,
    ]


def test_train_writes_the_per_run_figures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir, "--no-battery-figs")) == 0

    figs = sorted(path.name for path in (run_dir / "figs").iterdir())
    assert figs == ["experience_heatmaps.png", "state_coverage.png"]
    assert "figure written to" in capsys.readouterr().out


def test_train_writes_a_battery_frame_per_battery_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir)) == 0

    frames = sorted((run_dir / "figs" / "q_by_battery").iterdir())
    scenario = resolve_scenario("safe_corridor")
    assert len(frames) == scenario.battery_binning().levels == 4
    output = capsys.readouterr().out
    assert f"{len(frames)} battery frames written to" in output
    assert frames[0].name not in output


@pytest.mark.slow
def test_dense_battery_writes_a_frame_per_reading(tmp_path: Path) -> None:
    """``--dense-battery`` is the sixty-one-frame set the drain sequence was built on."""
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir, "--dense-battery")) == 0

    frames = sorted((run_dir / "figs" / "q_by_battery").iterdir())
    scenario = resolve_scenario("safe_corridor")
    assert len(frames) == scenario.battery_capacity + 1
    assert frames[0].name == "battery_00.png"
    assert frames[-1].name == f"battery_{scenario.battery_capacity}.png"


def test_train_defaults_to_the_binned_battery_axis(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir, "--no-figs")) == 0

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    scenario = resolve_scenario("safe_corridor")
    rows = scenario.rows * scenario.cols * 4 * NUM_PAYLOAD_STATES

    assert manifest["config"]["battery_encoding"] == "affordability"
    assert manifest["num_states"] == rows
    assert np.load(run_dir / "q_table.npy").shape == (rows, 5)


def test_dense_battery_restores_the_original_row_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir, "--no-figs", "--dense-battery")) == 0

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    scenario = resolve_scenario("safe_corridor")
    rows = scenario.rows * scenario.cols * (scenario.battery_capacity + 1) * NUM_PAYLOAD_STATES

    assert manifest["config"]["battery_encoding"] == "dense"
    assert manifest["num_states"] == rows
    assert np.load(run_dir / "q_table.npy").shape == (rows, 5)


def test_evaluate_and_figures_follow_the_run_s_own_encoding(tmp_path: Path) -> None:
    """A saved table has to be read back through the axis it was written under."""
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir, "--no-figs", "--dense-battery")) == 0

    assert main(["evaluate", "--run", str(run_dir), "--episodes", "2"]) == 0
    assert main(["figures", "--run", str(run_dir), "--no-battery-figs"]) == 0
    assert (run_dir / "figs" / "state_coverage.png").exists()


def test_a_run_saved_before_the_flag_existed_is_read_as_dense(tmp_path: Path) -> None:
    """Every run written before the encoding was configurable is a dense one."""
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir, "--no-figs", "--dense-battery")) == 0
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["config"]["battery_encoding"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert main(["evaluate", "--run", str(run_dir), "--episodes", "2"]) == 0
    assert main(["figures", "--run", str(run_dir), "--no-battery-figs"]) == 0


def test_train_can_skip_only_the_battery_frames(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir, "--no-battery-figs")) == 0

    assert (run_dir / "figs" / "state_coverage.png").exists()
    assert not (run_dir / "figs" / "q_by_battery").exists()


def test_train_can_skip_the_figures(tmp_path: Path) -> None:
    """The sweep's per-cell cost is the reason this switch exists."""
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir, "--no-figs")) == 0

    assert not (run_dir / "figs").exists()


def test_figures_subcommand_redraws_a_saved_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir, "--no-figs")) == 0
    assert main(["figures", "--run", str(run_dir), "--no-battery-figs"]) == 0

    assert (run_dir / "figs" / "state_coverage.png").exists()


def test_figures_subcommand_warns_when_visit_counts_are_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    assert main(_train_args(run_dir, "--no-figs")) == 0
    (run_dir / "visit_counts.npy").unlink()
    capsys.readouterr()

    assert main(["figures", "--run", str(run_dir), "--no-battery-figs"]) == 0
    assert "no visit_counts.npy" in capsys.readouterr().err
