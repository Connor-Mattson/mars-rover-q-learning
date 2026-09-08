"""Command-line entry points for play, train, evaluate, experiment, and replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .agent import teaching_stub_status
from .evaluation import evaluate
from .experiment import ExperimentConfig, json_safe, load_run, run_experiment, save_run, write_json
from .rewards import RewardMode
from .scenario import available_scenarios, resolve_scenario
from .training import TrainConfig, make_env, split_rngs, train

DEFAULT_ARTIFACT_ROOT = Path("artifacts")


def _teaching_banner(stream: Any) -> None:
    pending = teaching_stub_status()
    if pending:
        print(
            "note: human-owned functions still incomplete -> "
            + ", ".join(pending)
            + "  (see .teacher/current.md)",
            file=stream,
        )


def cmd_play(args: argparse.Namespace) -> int:
    """Drive the rover manually in a Pygame window."""
    from .renderer import run_manual_mission

    scenario = resolve_scenario(args.scenario)
    env_rng, _ = split_rngs(args.seed)
    env = make_env(
        scenario,
        RewardMode(args.reward),
        gamma=args.gamma,
        rng=env_rng,
    )
    print(f"{scenario.name}: {scenario.description}")
    print("arrows/WASD drive, SPACE collect, R restart, P policy overlay, Q quit")
    summary = run_manual_mission(env)
    if summary:
        print(json.dumps(summary, indent=2))
    env.close()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train one Q-table and write a run directory."""
    scenario = resolve_scenario(args.scenario)
    config = TrainConfig(
        scenario=scenario.name,
        reward_mode=RewardMode(args.reward),
        seed=args.seed,
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        initial_q=args.initial_q,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        log_every=args.log_every,
    )
    result = train(scenario, config, progress=not args.quiet)
    evaluation = (
        evaluate(
            result.q_table,
            scenario,
            config.reward_mode,
            episodes=args.eval_episodes,
            seed=args.seed,
            gamma=args.gamma,
        )
        if args.eval_episodes > 0
        else None
    )

    run_dir = (
        Path(args.output)
        if args.output
        else (DEFAULT_ARTIFACT_ROOT / "runs" / f"{scenario.name}__{args.reward}__seed{args.seed}")
    )
    save_run(run_dir, scenario, result, evaluation)
    print(f"run written to {run_dir}")
    if evaluation is not None:
        print(json.dumps(json_safe(evaluation.summary.as_dict()), indent=2))
    if not result.learning_is_meaningful:
        print(
            "TEACHING STATE: these numbers are placeholders, not a learning result.",
            file=sys.stderr,
        )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a saved run's Q-table with exploration disabled."""
    run = load_run(args.run)
    reward_mode = run.manifest["config"]["reward_mode"]
    result = evaluate(
        run.q_table,
        run.scenario,
        reward_mode,
        episodes=args.episodes,
        seed=args.seed,
        gamma=run.manifest["config"]["gamma"],
    )
    print(json.dumps(json_safe(result.summary.as_dict()), indent=2))
    if args.output:
        write_json(Path(args.output), result.as_dict())
    if run.manifest.get("pending_human_functions"):
        print(
            "TEACHING STATE: this table was produced with placeholder functions.",
            file=sys.stderr,
        )
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Run the full reward-comparison grid and write plots and summaries."""
    config = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    output_root = Path(args.output) if args.output else DEFAULT_ARTIFACT_ROOT / config.name
    _teaching_banner(sys.stderr)
    payload = run_experiment(config, output_root, make_plots=not args.no_plots)
    print(f"experiment written to {output_root}")
    print(json.dumps(json_safe(payload["aggregated"]), indent=2))
    if not payload["learning_is_meaningful"]:
        print(
            "TEACHING STATE: no learning happened; do not report these numbers.",
            file=sys.stderr,
        )
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay a stored episode in the Pygame renderer."""
    from .renderer import replay_trajectory

    run = load_run(args.run)
    if args.episode != "best":
        raise SystemExit("only --episode best is stored; re-run evaluate to capture others")
    if run.best_trajectory is None:
        raise SystemExit(f"{run.root} has no stored best episode; run evaluate first")
    policy = run.policy if args.policy_overlay else None
    replay_trajectory(run.scenario, run.best_trajectory, policy=policy, fps=args.fps)
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    """List the bundled scenarios and their headline numbers."""
    del args
    for name in available_scenarios():
        scenario = resolve_scenario(name)
        target = scenario.heuristic_target_sample()
        print(
            f"{name:<22} {scenario.rows}x{scenario.cols}  "
            f"battery={scenario.battery_capacity:<4} steps={scenario.max_steps:<4} "
            f"states={scenario.rows * scenario.cols * (scenario.battery_capacity + 1) * 4:<8} "
            f"subgoal={target.name.lower()}"
        )
        if scenario.description:
            print(f"{'':<22} {scenario.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="mars_rover_q",
        description="Mars Sample Return: reward design in tabular Q-learning.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="drive the rover manually in a Pygame window")
    play.add_argument("--scenario", default="safe_corridor")
    play.add_argument(
        "--reward", default=RewardMode.SPARSE.value, choices=[m.value for m in RewardMode]
    )
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--gamma", type=float, default=0.99)
    play.set_defaults(func=cmd_play)

    trainer = subparsers.add_parser("train", help="train one Q-table and save a run directory")
    trainer.add_argument("--scenario", default="safe_corridor")
    trainer.add_argument(
        "--reward", default=RewardMode.SPARSE.value, choices=[m.value for m in RewardMode]
    )
    trainer.add_argument("--seed", type=int, default=1)
    trainer.add_argument("--episodes", type=int, default=4000)
    trainer.add_argument("--learning-rate", type=float, default=0.2)
    trainer.add_argument("--gamma", type=float, default=0.99)
    trainer.add_argument("--initial-q", type=float, default=0.0)
    trainer.add_argument("--epsilon-start", type=float, default=1.0)
    trainer.add_argument("--epsilon-end", type=float, default=0.05)
    trainer.add_argument("--eval-episodes", type=int, default=200)
    trainer.add_argument("--log-every", type=int, default=200)
    trainer.add_argument("--output", default=None, help="run directory (default under artifacts/)")
    trainer.add_argument("--quiet", action="store_true")
    trainer.set_defaults(func=cmd_train)

    evaluator = subparsers.add_parser("evaluate", help="greedy-evaluate a saved run")
    evaluator.add_argument("--run", required=True, help="run directory produced by train")
    evaluator.add_argument("--episodes", type=int, default=500)
    evaluator.add_argument("--seed", type=int, default=1)
    evaluator.add_argument("--output", default=None, help="optional JSON output path")
    evaluator.set_defaults(func=cmd_evaluate)

    experiment = subparsers.add_parser("experiment", help="run the reward-comparison grid")
    experiment.add_argument("--config", default=None, help="experiment config JSON")
    experiment.add_argument("--output", default=None, help="output root (default under artifacts/)")
    experiment.add_argument("--no-plots", action="store_true")
    experiment.set_defaults(func=cmd_experiment)

    replay = subparsers.add_parser("replay", help="replay a stored episode in Pygame")
    replay.add_argument("--run", required=True)
    replay.add_argument("--episode", default="best")
    replay.add_argument("--fps", type=float, default=4.0)
    replay.add_argument("--policy-overlay", action="store_true")
    replay.set_defaults(func=cmd_replay)

    scenarios = subparsers.add_parser("scenarios", help="list bundled scenarios")
    scenarios.set_defaults(func=cmd_scenarios)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    np.seterr(all="raise")
    handler: Any = args.func
    result: int = handler(args)
    return result


if __name__ == "__main__":  # pragma: no cover - module execution path
    raise SystemExit(main())
