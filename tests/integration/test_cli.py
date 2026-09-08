"""CLI smoke tests. Pygame-driven commands are exercised through their parsers only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mars_rover_q.cli import build_parser, main


@pytest.mark.parametrize(
    "argv",
    [
        ["play", "--scenario", "safe_corridor"],
        ["train", "--scenario", "safe_corridor", "--reward", "sparse", "--seed", "1"],
        ["evaluate", "--run", "some/run", "--episodes", "500"],
        ["experiment", "--config", "configs/experiments/reward_comparison.json"],
        ["replay", "--run", "some/run", "--episode", "best"],
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
                ]
            )
            == 0
        )
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["config"]["reward_mode"] == mode


def test_unknown_reward_mode_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--reward", "dense_but_clever"])


@pytest.mark.slow
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
        ]
    )
    with pytest.raises(SystemExit, match="no stored best episode"):
        main(["replay", "--run", str(run_dir)])
