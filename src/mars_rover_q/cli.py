"""Command-line entry points for play, train, evaluate, experiment, tune, replay, and
curriculum inspection."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .agent import teaching_stub_status
from .curriculum import (
    CURRICULUM_STRATEGY_LABELS,
    DEFAULT_WEIGHT_EXPONENT,
    DEFAULT_WINDOW_FRACTION,
    CurriculumStrategy,
    SimulatedAnneal,
    StartStateCurriculum,
    rank_band_shares,
    simulate_start_distribution,
)
from .evaluation import evaluate
from .experiment import ExperimentConfig, json_safe, load_run, run_experiment, save_run, write_json
from .numerics import install_numeric_guard
from .rewards import RewardMode
from .scenario import Scenario, available_scenarios, resolve_scenario
from .state import NUM_PAYLOAD_STATES, BatteryEncoding
from .training import TrainConfig, make_env, split_rngs, train
from .tuning import (
    DEFAULT_CHECKPOINT_EPISODES,
    DEFAULT_CHECKPOINTS,
    DEFAULT_STUDY_EPISODE_BUDGET,
    DEFAULT_TRIAL_EPISODE_CAP,
    TuningConfig,
    run_study,
)

DEFAULT_ARTIFACT_ROOT = Path("artifacts")


def _battery_encoding(args: argparse.Namespace) -> BatteryEncoding:
    """Resolve the ``--dense-battery`` flag into a :class:`BatteryEncoding`."""
    return (
        BatteryEncoding.DENSE
        if getattr(args, "dense_battery", False)
        else BatteryEncoding.AFFORDABILITY
    )


def _run_battery_encoding(manifest: dict[str, Any]) -> BatteryEncoding:
    """The battery encoding a saved run's table was built under.

    Runs written before the encoding was configurable carry no such key, and every
    one of them is dense -- that was the only encoding there was -- so the fallback
    is ``DENSE`` rather than the current default. Reading an old run's table under
    the new default would index a 24,400-row table as if it had 1,600 rows.
    """
    recorded = manifest.get("config", {}).get("battery_encoding")
    return BatteryEncoding(recorded) if recorded else BatteryEncoding.DENSE


def _add_dense_battery_flag(parser: argparse.ArgumentParser) -> None:
    """Opt out of the binned battery axis, on the subcommands that build a table.

    The default bins the battery at the round-trip cost of each sample -- the only
    charge levels at which the best action can change -- which is one to two orders
    of magnitude fewer rows than a row per reading. ``--dense-battery`` restores the
    original encoding, which is the baseline the binning has to be measured against.
    """
    parser.add_argument(
        "--dense-battery",
        action="store_true",
        help=(
            "give every battery reading its own Q-table row instead of binning at "
            "the samples' round-trip costs (far more rows; the original encoding)"
        ),
    )


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
        battery_encoding=_battery_encoding(args),
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
        battery_encoding=_battery_encoding(args),
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        curriculum_fraction=args.curriculum_fraction,
        curriculum_strategy=CurriculumStrategy(args.curriculum_strategy),
        curriculum_window_fraction=args.curriculum_window_fraction,
        curriculum_weight_exponent=args.curriculum_weight_exponent,
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
            battery_encoding=config.battery_encoding,
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
    if not args.no_figs:
        _report_figures(
            _write_figures(
                run_dir,
                scenario,
                result.q_table,
                visit_counts=result.visit_counts,
                initial_q=config.initial_q,
                config=config.as_dict(),
                battery_frames=not args.no_battery_figs,
                battery_encoding=config.battery_encoding,
            )
        )
    # State coverage is the diagnostic that motivated the curriculum: a table whose
    # rows are still tied was never visited, and its greedy action is a coin toss.
    print(f"tied (never-decided) states: {result.metadata['tied_state_fraction']:.1%}")
    curriculum = result.metadata["curriculum"]
    if curriculum["requested"]:
        print(
            f"curriculum: strategy={curriculum['strategy']} pool={curriculum['pool_size']} "
            f"distinct_starts={curriculum['distinct_start_states']} "
            f"curriculum_episodes={curriculum['episodes_from_curriculum']}"
        )
    if evaluation is not None:
        print(json.dumps(json_safe(evaluation.summary.as_dict()), indent=2))
    if not result.learning_is_meaningful:
        print(
            "TEACHING STATE: these numbers are placeholders, not a learning result.",
            file=sys.stderr,
        )
    return 0


def _write_figures(
    run_dir: Path,
    scenario: Scenario,
    q_table: NDArray[np.float64],
    *,
    visit_counts: NDArray[np.int64] | None,
    initial_q: float,
    config: dict[str, Any],
    battery_frames: bool = True,
    battery_encoding: BatteryEncoding = BatteryEncoding.AFFORDABILITY,
) -> list[Path]:
    """Draw the per-run figures. Imported lazily: matplotlib is slow to import and
    every other subcommand gets by without it."""
    from .run_figures import run_subtitle, write_run_figures

    return write_run_figures(
        run_dir,
        scenario,
        q_table,
        visit_counts=visit_counts,
        initial_q=initial_q,
        subtitle=run_subtitle(config),
        battery_frames=battery_frames,
        binning=scenario.battery_binning(battery_encoding),
    )


def _report_figures(paths: list[Path]) -> None:
    """Name the summary figures; count the battery frames.

    The battery set is one figure per battery level, so under the dense encoding
    listing it a line at a time would bury the run summary underneath sixty-odd
    paths that differ in two characters. The directory is the useful address for a
    set.
    """
    from .run_figures import BATTERY_FIGS_DIRNAME

    frames = [path for path in paths if path.parent.name == BATTERY_FIGS_DIRNAME]
    for path in paths:
        if path not in frames:
            print(f"figure written to {path}")
    if frames:
        print(f"{len(frames)} battery frames written to {frames[0].parent}")


def cmd_figures(args: argparse.Namespace) -> int:
    """(Re)draw the per-run figures for a saved run."""
    run = load_run(args.run)
    config = run.manifest.get("config", {})
    if run.visit_counts is None:
        print(
            f"note: {run.root} has no visit_counts.npy (saved before they were "
            "recorded); the map figure falls back to learned battery levels per cell",
            file=sys.stderr,
        )
    _report_figures(
        _write_figures(
            Path(run.root),
            run.scenario,
            run.q_table,
            visit_counts=run.visit_counts,
            initial_q=float(config.get("initial_q", 0.0)),
            config=config,
            battery_frames=not args.no_battery_figs,
            battery_encoding=_run_battery_encoding(run.manifest),
        )
    )
    return 0


def _add_curriculum_strategy_flags(parser: argparse.ArgumentParser) -> None:
    """The three knobs that select and tune a start-state sampling strategy.

    Only one of the two numeric knobs is read by any given strategy, so they are
    grouped here rather than repeated: ``--curriculum-window-fraction`` is the
    sliding band's width and ``--curriculum-weight-exponent`` the strength of the
    visit-weighted tilt.
    """
    parser.add_argument(
        "--curriculum-strategy",
        default=CurriculumStrategy.GROWING.value,
        choices=[strategy.value for strategy in CurriculumStrategy],
        help=(
            "how the curriculum draws from the ranked pool: a growing window "
            "(uniform over the easiest k), a sliding window of fixed width, a "
            "growing window weighted against already-updated states, or exploring "
            "starts (uniform over the whole pool, no anneal)"
        ),
    )
    parser.add_argument(
        "--curriculum-window-fraction",
        type=float,
        default=DEFAULT_WINDOW_FRACTION,
        help="sliding-band width as a share of the pool (sliding strategy only)",
    )
    parser.add_argument(
        "--curriculum-weight-exponent",
        type=float,
        default=DEFAULT_WEIGHT_EXPONENT,
        help=(
            "strength of the visit tilt; 0 reproduces the uniform draw "
            "(visit_weighted strategy only)"
        ),
    )


def _add_battery_figs_flag(parser: argparse.ArgumentParser) -> None:
    """Opt out of the per-battery value maps, on both subcommands that draw figures.

    They are one figure per battery level -- sixty-one of them on ``safe_corridor``,
    a hundred and eighty-one on ``shaping_trap`` -- and a sweep redrawing figures
    across a few hundred runs has a real reason to want only the two summaries.
    """
    parser.add_argument(
        "--no-battery-figs",
        action="store_true",
        help="skip the per-battery value maps in <run>/figs/q_by_battery",
    )


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
        battery_encoding=_run_battery_encoding(run.manifest),
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
    payload = run_experiment(config, output_root, make_plots=not args.no_plots, jobs=args.jobs)
    print(f"experiment written to {output_root}")
    if "sweep_analysis" in payload:
        _print_sweep_analysis(payload["sweep_analysis"])
    else:
        print(json.dumps(json_safe(payload["aggregated"]), indent=2))
    if not payload["learning_is_meaningful"]:
        print(
            "TEACHING STATE: no learning happened; do not report these numbers.",
            file=sys.stderr,
        )
    return 0


def _tuning_config(args: argparse.Namespace) -> TuningConfig:
    """Build the search config from a JSON file, a set of flags, or both.

    A flag overrides the file only when it was actually passed: every override below
    defaults to ``None`` rather than to the config's own default, so loading a tuned
    search definition and changing one thing on the command line does not silently
    reset the rest of it.
    """
    config = TuningConfig.from_file(args.config) if args.config else TuningConfig()
    overrides: dict[str, Any] = {
        "scenario": args.scenario,
        "reward_mode": RewardMode(args.reward) if args.reward else None,
        "seeds": tuple(int(s) for s in args.seeds.split(",")) if args.seeds else None,
        "trial_episode_cap": args.trial_episodes,
        "study_episode_budget": args.total_episodes,
        "max_trials": args.trials,
        "checkpoints": args.checkpoints,
        "checkpoint_episodes": args.checkpoint_episodes,
        "eval_episodes": args.eval_episodes,
        "sampler_seed": args.sampler_seed,
        "battery_encoding": BatteryEncoding.DENSE if args.dense_battery else None,
        "search_curriculum": False if args.no_curriculum_search else None,
        "confirm_best": False if args.no_confirm_best else None,
    }
    return dataclasses.replace(
        config, **{key: value for key, value in overrides.items() if value is not None}
    )


def cmd_tune(args: argparse.Namespace) -> int:
    """Search hyper-parameters with Optuna under a bounded episode budget."""
    config = _tuning_config(args)
    output_root = Path(args.output) if args.output else DEFAULT_ARTIFACT_ROOT / config.name
    _teaching_banner(sys.stderr)
    payload = run_study(config, output_root, make_plots=not args.no_plots)
    print(f"\nstudy written to {output_root}")
    _print_study(payload, top=args.top)
    if not payload["search_is_meaningful"]:
        print(
            "TEACHING STATE: the search objective is stubbed; this ranking is not a "
            "ranking. Do not report these numbers.",
            file=sys.stderr,
        )
    return 0


def _print_study(payload: dict[str, Any], top: int = 5) -> None:
    """Print the budget actually spent, the top trials, and the Pareto front."""
    budget = payload["budget"]
    trials = payload["trials"]
    pruned = sum(1 for trial in trials if trial["pruned"])
    print(
        f"{len(trials)} trials ({pruned} pruned), "
        f"{budget['spent']:,}/{budget['total']:,} episodes spent"
    )

    scored = sorted(
        (t for t in trials if t["score"] is not None),
        key=lambda t: (-float(t["score"]), t["number"]),
    )
    if not scored:
        print("no trial completed: nothing to rank")
        return
    print(f"\ntop {min(top, len(scored))} by objective score")
    for trial in scored[:top]:
        print(
            f"  #{trial['number']:<3} score={float(trial['score']):0.4f}  "
            f"best={trial['best_return']:7.2f} @ {trial['episodes_to_best']:>6} ep  "
            f"final={trial['final_return']:7.2f}"
        )
        print(f"       {_format_params(trial['params'])}")

    front = payload["pareto_front"]
    by_number = {t["number"]: t for t in trials}
    print("\npareto front (cannot do better without spending more episodes)")
    if not front:
        print("  unavailable")
    for number in front:
        trial = by_number[number]
        print(
            f"  #{number:<3} best={trial['best_return']:7.2f} @ "
            f"{trial['episodes_to_best']:>6} episodes"
        )
    confirmation = payload.get("confirmation_run")
    if confirmation is not None:
        print(
            f"\nconfirmation run of trial #{confirmation['trial_number']}: "
            f"success={confirmation['eval_success_rate']:0.2f}  "
            f"base_return={confirmation['eval_mean_base_return']:0.2f}  "
            f"-> {confirmation['run_dir']}"
        )


def _format_params(params: dict[str, Any]) -> str:
    """One-line rendering of a trial's suggested parameters."""
    parts = []
    for key, value in sorted(params.items()):
        parts.append(f"{key}={value:0.4g}" if isinstance(value, float) else f"{key}={value}")
    return "  ".join(parts)


def _print_sweep_analysis(analysis: dict[str, Any]) -> None:
    """Print the budget sweep's headline numbers as a small table."""
    metric = analysis["metric"]
    print(f"\ncurriculum vs baseline, metric = {metric}")
    for group in analysis["groups"]:
        print(f"\n  {group['scenario']} / {group['reward_mode']}  (target {group['target']:.1f})")
        for entry in group["budget_to_target"]:
            budget = entry["budget"]
            reached = f"{budget:,.0f} episodes" if budget is not None else "did not reach"
            speedup = entry["speedup_vs_baseline"]
            suffix = f"  ({speedup:.2f}x fewer than baseline)" if speedup else ""
            print(f"    {entry['label']:<26} {reached}{suffix}")
        arm = None
        for entry in group["paired"]:
            # One block per treatment arm: with three strategies in the grid, an
            # unlabelled list of budgets is three interleaved comparisons.
            if entry.get("label") != arm:
                arm = entry.get("label")
                print(f"    {arm} vs No Curriculum")
            mark = "*" if entry["excludes_zero"] else " "
            low, high = entry["ci_low"], entry["ci_high"]
            interval = (
                f"[{low:+8.2f}, {high:+8.2f}]"
                if low is not None and high is not None
                else "[  undefined  ]"
            )
            print(
                f"    {mark} budget {entry['budget']:>6}  "
                f"paired delta {entry['mean']:+8.2f}  95% CI {interval}  n={entry['n']}"
            )
    print("\n  * = paired 95% interval excludes zero")


def cmd_analyze(args: argparse.Namespace) -> int:
    """Re-analyse and re-plot a finished experiment without re-running it."""
    from .plots import write_experiment_plots
    from .sweep import analyze_sweep

    summary_path = Path(args.summary)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload["per_seed"]
    output_root = summary_path.parent
    analysis = analyze_sweep(rows, metric=args.metric, target_fraction=args.target_fraction)
    write_json(output_root / "sweep_analysis.json", analysis)
    _print_sweep_analysis(analysis)
    if not args.no_plots:
        written = write_experiment_plots(output_root, rows, payload["aggregated"], analysis)
        for path in written:
            print(f"wrote {path}")
    if payload.get("pending_human_functions"):
        print(
            "TEACHING STATE: this summary came from placeholder functions.",
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
    replay_trajectory(
        run.scenario,
        run.best_trajectory,
        policy=policy,
        fps=args.fps,
        battery_encoding=_run_battery_encoding(run.manifest),
    )
    return 0


#: Progress buckets the ``curriculum`` command reports its dry run in.
CURRICULUM_PROBES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _print_progress_bucket(
    label: str, anneal: SimulatedAnneal, selected: NDArray[np.bool_]
) -> None:
    """One row of the ``curriculum`` dry-run table, over the selected episodes."""
    if not selected.any():
        return
    difficulty = anneal.difficulties[selected]
    bands = rank_band_shares(anneal.ranks[selected], anneal.pool_size)
    canonical = float(np.count_nonzero(anneal.is_canonical[selected]) / selected.sum())
    print(
        f"  {label}  "
        f"difficulty min/mean/max="
        f"{difficulty.min():6.1f}/{difficulty.mean():6.1f}/{difficulty.max():6.1f}  "
        "bands=" + "/".join(f"{share:0.2f}" for share in bands) + "  "
        f"canonical={canonical:0.0%}"
    )


def cmd_curriculum(args: argparse.Namespace) -> int:
    """Inspect the start-state curriculum for one scenario without training.

    Each requested strategy is dry-run over the full episode budget by
    :func:`~mars_rover_q.curriculum.simulate_start_distribution` and reported per
    progress bucket. The band shares are the column that matters: mean difficulty
    can rise monotonically while the easiest quarter of the pool still takes most
    of the episodes, which is exactly the oversampling the sliding and
    visit-weighted strategies exist to fix.
    """
    scenario = resolve_scenario(args.scenario)
    strategies = (
        tuple(CurriculumStrategy)
        if args.strategy == "all"
        else (CurriculumStrategy(args.strategy),)
    )
    curricula = {
        strategy: StartStateCurriculum(
            scenario,
            total_episodes=args.episodes,
            anneal_fraction=args.curriculum_fraction,
            strategy=strategy,
            window_fraction=args.window_fraction,
            weight_exponent=args.weight_exponent,
        )
        for strategy in strategies
    }
    # The pool is the same object of study for every strategy -- they differ only
    # in how it is drawn from -- so it is reported once rather than per section.
    shared = curricula[strategies[0]].describe()
    print(
        json.dumps(
            json_safe(
                {
                    key: shared[key]
                    for key in (
                        "requested",
                        "active",
                        "anneal_fraction",
                        "anneal_episodes",
                        "pool_size",
                        "canonical_difficulty",
                        "difficulty_quantiles",
                    )
                }
            ),
            indent=2,
        )
    )
    if not curricula[strategies[0]].active:
        print(
            "no admissible start states; the curriculum would fall back to the "
            "canonical lander start (see .teacher/current.md)",
            file=sys.stderr,
        )
        return 0

    for strategy in strategies:
        curriculum = curricula[strategy]
        anneal = simulate_start_distribution(
            curriculum, np.random.default_rng(args.seed), episodes=args.episodes
        )
        knob = (
            f"  window_fraction={curriculum.window_fraction:g}"
            if strategy is CurriculumStrategy.SLIDING
            else f"  weight_exponent={curriculum.weight_exponent:g}"
            if strategy is CurriculumStrategy.VISIT_WEIGHTED
            else ""
        )
        print(f"\n{strategy} -- {CURRICULUM_STRATEGY_LABELS[strategy]}{knob}")
        print(
            "  simulated starts by anneal progress "
            "(bands are the share of episodes in each quarter of the pool, easiest first):"
        )
        progress = np.minimum(1.0, anneal.episodes / curriculum.anneal_episodes)
        # Every bucket is half-open, so the whole report is about the anneal itself.
        # Episodes past it start at the lander by definition and would otherwise
        # dominate the bands with a single state -- on some maps a rather easy one.
        for low, high in pairwise(CURRICULUM_PROBES):
            _print_progress_bucket(
                f"progress={low:0.1f}-{high:0.1f}",
                anneal,
                (progress >= low) & (progress < high),
            )
        after = progress >= 1.0
        if after.any():
            _print_progress_bucket("after the anneal ", anneal, after)

        during = progress < 1.0
        distinct = int(np.unique(anneal.ranks[during]).size)
        repeats = int(np.bincount(anneal.ranks[during]).max()) if during.any() else 0
        print(
            f"  over the anneal: distinct starts={distinct}/{anneal.pool_size}  "
            f"easiest quarter="
            f"{rank_band_shares(anneal.ranks[during], anneal.pool_size)[0]:0.2f} "
            f"of episodes  most-repeated start={repeats}x"
        )
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    """List the bundled scenarios and their headline numbers."""
    del args
    for name in available_scenarios():
        scenario = resolve_scenario(name)
        target = scenario.heuristic_target_sample()
        cells = scenario.rows * scenario.cols * NUM_PAYLOAD_STATES
        binning = scenario.battery_binning()
        print(
            f"{name:<22} {scenario.rows}x{scenario.cols}  "
            f"battery={scenario.battery_capacity:<4} steps={scenario.max_steps:<4} "
            f"states={cells * binning.levels:<7} "
            f"dense={cells * (scenario.battery_capacity + 1):<7} "
            f"subgoal={target.name.lower()}"
        )
        if scenario.description:
            print(f"{'':<22} {scenario.description}")
        bins = "  ".join(binning.label(level) for level in range(binning.levels))
        print(f"{'':<22} battery bins: {bins}")
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
    _add_dense_battery_flag(play)
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
    _add_dense_battery_flag(trainer)
    trainer.add_argument("--epsilon-start", type=float, default=1.0)
    trainer.add_argument("--epsilon-end", type=float, default=0.05)
    trainer.add_argument(
        "--curriculum-fraction",
        type=float,
        default=0.0,
        help=(
            "share of episodes over which the start-state curriculum anneals from "
            "easy reachable states to the canonical lander start (0 disables)"
        ),
    )
    _add_curriculum_strategy_flags(trainer)
    trainer.add_argument("--eval-episodes", type=int, default=200)
    trainer.add_argument("--log-every", type=int, default=200)
    trainer.add_argument("--output", default=None, help="run directory (default under artifacts/)")
    trainer.add_argument("--quiet", action="store_true")
    trainer.add_argument(
        "--no-figs",
        action="store_true",
        help="skip the per-run figures written to <run>/figs",
    )
    _add_battery_figs_flag(trainer)
    trainer.set_defaults(func=cmd_train)

    figures = subparsers.add_parser("figures", help="(re)draw the per-run figures for a saved run")
    figures.add_argument("--run", required=True, help="run directory produced by train")
    _add_battery_figs_flag(figures)
    figures.set_defaults(func=cmd_figures)

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
    experiment.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "worker processes; every cell is independent and fully seeded, so the "
            "results do not depend on this"
        ),
    )
    experiment.set_defaults(func=cmd_experiment)

    tune = subparsers.add_parser(
        "tune",
        help="search hyper-parameters with Optuna under a bounded episode budget",
    )
    tune.add_argument("--config", default=None, help="search definition JSON")
    tune.add_argument("--scenario", default=None)
    tune.add_argument("--reward", default=None, choices=[m.value for m in RewardMode])
    tune.add_argument(
        "--seeds",
        default=None,
        help="comma-separated training seeds averaged within each trial (default: 1)",
    )
    tune.add_argument(
        "--trial-episodes",
        type=int,
        default=None,
        help=(
            "episode budget per trial per seed; deliberately below convergence so that "
            f"learning speed still separates settings (default: {DEFAULT_TRIAL_EPISODE_CAP})"
        ),
    )
    tune.add_argument(
        "--total-episodes",
        type=int,
        default=None,
        help=(
            "total training episodes the whole search may spend; the study stops when "
            f"what remains cannot fund another trial (default: {DEFAULT_STUDY_EPISODE_BUDGET})"
        ),
    )
    tune.add_argument(
        "--trials",
        type=int,
        default=None,
        help="optional hard cap on trial count, applied on top of the episode budget",
    )
    tune.add_argument(
        "--checkpoints",
        type=int,
        default=None,
        help=f"learning-curve points per trial (default: {DEFAULT_CHECKPOINTS})",
    )
    tune.add_argument(
        "--checkpoint-episodes",
        type=int,
        default=None,
        help=f"greedy episodes per checkpoint (default: {DEFAULT_CHECKPOINT_EPISODES})",
    )
    tune.add_argument("--eval-episodes", type=int, default=None)
    tune.add_argument("--sampler-seed", type=int, default=None)
    _add_dense_battery_flag(tune)
    tune.add_argument(
        "--no-curriculum-search",
        action="store_true",
        help="hold the start-state curriculum off and search the agent knobs only",
    )
    tune.add_argument(
        "--no-confirm-best",
        action="store_true",
        help="skip re-training the winning configuration into a full run directory",
    )
    tune.add_argument("--output", default=None, help="output root (default under artifacts/)")
    tune.add_argument("--no-plots", action="store_true")
    tune.add_argument("--top", type=int, default=5, help="how many trials to print")
    tune.set_defaults(func=cmd_tune)

    analyze = subparsers.add_parser(
        "analyze", help="re-analyse and re-plot a finished experiment from its summary.json"
    )
    analyze.add_argument("--summary", required=True, help="path to an experiment summary.json")
    analyze.add_argument(
        "--metric",
        default="eval_mean_base_return",
        help="row metric to analyse; the default is mission return with shaping excluded",
    )
    analyze.add_argument(
        "--target-fraction",
        type=float,
        default=0.9,
        help="performance target as a fraction of the best condition mean on the curve",
    )
    analyze.add_argument("--no-plots", action="store_true")
    analyze.set_defaults(func=cmd_analyze)

    replay = subparsers.add_parser("replay", help="replay a stored episode in Pygame")
    replay.add_argument("--run", required=True)
    replay.add_argument("--episode", default="best")
    replay.add_argument("--fps", type=float, default=4.0)
    replay.add_argument("--policy-overlay", action="store_true")
    replay.set_defaults(func=cmd_replay)

    scenarios = subparsers.add_parser("scenarios", help="list bundled scenarios")
    scenarios.set_defaults(func=cmd_scenarios)

    curriculum = subparsers.add_parser(
        "curriculum", help="inspect the start-state curriculum for a scenario"
    )
    curriculum.add_argument("--scenario", default="safe_corridor")
    curriculum.add_argument("--curriculum-fraction", type=float, default=0.5)
    curriculum.add_argument("--episodes", type=int, default=4000)
    curriculum.add_argument(
        "--strategy",
        default="all",
        choices=["all", *(s.value for s in CurriculumStrategy)],
        help="which sampling strategy to dry-run (default: every one, side by side)",
    )
    curriculum.add_argument("--window-fraction", type=float, default=DEFAULT_WINDOW_FRACTION)
    curriculum.add_argument("--weight-exponent", type=float, default=DEFAULT_WEIGHT_EXPONENT)
    curriculum.add_argument("--seed", type=int, default=1)
    curriculum.set_defaults(func=cmd_curriculum)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    install_numeric_guard()
    handler: Any = args.func
    result: int = handler(args)
    return result


if __name__ == "__main__":  # pragma: no cover - module execution path
    raise SystemExit(main())
