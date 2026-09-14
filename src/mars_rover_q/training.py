"""Training orchestration around the human-owned Q-learning functions.

The loop below is the *only* learning path in the repository, and it calls
:mod:`mars_rover_q.agent`'s five human-owned functions directly. While those
functions are still stubs the loop runs to completion but learns nothing, and
prints a conspicuous teaching-state warning saying so.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .agent import (
    EpsilonSchedule,
    QLearningConfig,
    calculate_target,
    calculate_td_error,
    initialize_q_table,
    select_action,
    teaching_stub_status,
    update_q_value,
)
from .curriculum import (
    DEFAULT_WEIGHT_EXPONENT,
    DEFAULT_WINDOW_FRACTION,
    CurriculumStrategy,
    StartStateCurriculum,
    curriculum_stub_status,
)
from .environment import MarsRoverEnv
from .metrics import (
    CheckpointRecord,
    EpisodeRecord,
    canonical_start_records,
    learned_state_fraction,
    tied_state_fraction,
)
from .rewards import RewardMode, make_reward_model
from .scenario import Scenario
from .state import BatteryEncoding, RoverState

TEACHING_WARNING = """
================================ TEACHING STATE ================================
The following human-owned functions in src/mars_rover_q/agent.py are still
placeholder stubs: {names}

Training will run to completion, but NO LEARNING IS HAPPENING and the resulting
Q-table is meaningless. Do not report these numbers as a result.

Implement the functions listed in .teacher/current.md, then re-run.
================================================================================
"""

CURRICULUM_STUB_WARNING = """
========================= CURRICULUM STRATEGY STUBBED ==========================
curriculum_strategy={strategy} was requested, but its sampler in
src/mars_rover_q/curriculum.py is still a placeholder: {names}

Training will run to completion and the Q-table will be a real one, but every
episode begins at the lander, so this run is a NO-CURRICULUM run wearing a
curriculum label. Do not compare it against one.

See .teacher/current.md.
================================================================================
"""

CURRICULUM_WARNING = """
=========================== CURRICULUM NOT ACTIVE ==============================
A start-state curriculum was requested (curriculum_fraction={fraction}), but
enumerate_start_states() returned no admissible states, so every episode will
start at the lander exactly as it would without a curriculum.

This run is NOT a curriculum run; do not report it as one. See
.teacher/current.md.
================================================================================
"""


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Everything needed to reproduce one training run.

    ``curriculum_fraction`` is the share of the episode budget over which the
    start-state curriculum anneals from easy, physically reachable states to the
    canonical lander start; ``0.0`` disables it and is the default, so the baseline
    condition is unchanged. ``curriculum_strategy`` chooses *how* the anneal draws
    from the ranked pool, and ``curriculum_window_fraction`` and
    ``curriculum_weight_exponent`` are that strategy's knobs. See
    :mod:`mars_rover_q.curriculum`.

    ``battery_encoding`` picks how finely the table resolves remaining charge. It is
    a representation choice, not an environment one -- the MDP is identical either
    way -- but it changes the table's row count by more than an order of magnitude,
    so it is recorded here and lands in the run manifest alongside the seed.
    """

    scenario: str
    reward_mode: RewardMode = RewardMode.SPARSE
    seed: int = 1
    episodes: int = 4000
    learning_rate: float = 0.2
    gamma: float = 0.99
    initial_q: float = 0.0
    battery_encoding: BatteryEncoding = BatteryEncoding.AFFORDABILITY
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.6
    curriculum_fraction: float = 0.0
    curriculum_strategy: CurriculumStrategy = CurriculumStrategy.GROWING
    #: Sliding-band width, as a share of the pool. Read only by ``sliding``.
    curriculum_window_fraction: float = DEFAULT_WINDOW_FRACTION
    #: Strength of the visit tilt. Read only by ``visit_weighted``; ``0.0`` there
    #: is the uniform draw and therefore the control for the weighting itself.
    curriculum_weight_exponent: float = DEFAULT_WEIGHT_EXPONENT
    potential_scale: float = 1.0
    max_steps: int | None = None
    log_every: int = 50
    #: How many times during training to pause and measure greedy performance on
    #: the canonical mission. ``0`` disables it. The cadence is derived from the
    #: episode budget, so every budget gets the same number of points on its curve.
    eval_checkpoints: int = 0
    #: Greedy episodes per checkpoint.
    checkpoint_episodes: int = 40

    def agent_config(self) -> QLearningConfig:
        """The hyper-parameters as an agent-side config object."""
        decay_episodes = max(1, int(self.episodes * self.epsilon_decay_fraction))
        return QLearningConfig(
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            initial_q=self.initial_q,
            epsilon=EpsilonSchedule(
                start=self.epsilon_start,
                end=self.epsilon_end,
                decay_episodes=decay_episodes,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot."""
        payload = asdict(self)
        payload["reward_mode"] = str(self.reward_mode)
        payload["curriculum_strategy"] = str(self.curriculum_strategy)
        payload["battery_encoding"] = str(self.battery_encoding)
        return payload


#: Offset applied to a training seed for mid-training checkpoint evaluation, so it
#: shares no stream with training (the seed itself) or with final evaluation
#: (``EVAL_SEED_OFFSET``). Checkpoint *k* always uses ``seed + this + k``, which is
#: what makes checkpoint *k* of the curriculum arm and of the baseline arm face an
#: identical sequence of slips: the two curves are paired at every point on the
#: x-axis, not merely at the end.
CHECKPOINT_SEED_OFFSET = 2_000_000


@dataclass(slots=True)
class TrainResult:
    """The artefacts of one completed training run."""

    config: TrainConfig
    q_table: NDArray[np.float64]
    records: list[EpisodeRecord]
    pending_human_functions: tuple[str, ...] = ()
    total_env_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[CheckpointRecord] = field(default_factory=list)
    #: How many Q-updates each state received, indexed by :class:`StateEncoder` row
    #: ID. This is the experience the table was built from, and it is *not*
    #: recoverable from the table afterwards: a state visited ten thousand times and
    #: a state visited once are both simply "not the initial value".
    visit_counts: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    #: Episodes actually run. This is ``config.episodes`` unless a
    #: ``checkpoint_callback`` asked :func:`train` to stop early, which is how the
    #: hyper-parameter search abandons a hopeless trial: the episodes it did not
    #: run are the ones the study gets to spend somewhere else, so the true count
    #: has to be reported rather than assumed from the config.
    episodes_completed: int = 0
    #: Whether a ``checkpoint_callback`` ended the run before its episode budget.
    stopped_early: bool = False

    @property
    def learning_is_meaningful(self) -> bool:
        """``False`` while any human-owned function is still a placeholder."""
        return not self.pending_human_functions

    @property
    def curriculum_was_active(self) -> bool:
        """Whether episodes actually started anywhere other than the lander."""
        curriculum = self.metadata.get("curriculum")
        return bool(curriculum and curriculum.get("episodes_from_curriculum", 0) > 0)


def make_env(
    scenario: Scenario,
    reward_mode: RewardMode | str,
    *,
    gamma: float,
    rng: np.random.Generator,
    potential_scale: float = 1.0,
    max_steps: int | None = None,
    render_mode: str | None = None,
    battery_encoding: BatteryEncoding | str = BatteryEncoding.AFFORDABILITY,
) -> MarsRoverEnv:
    """Build an environment with the reward model for one experimental condition."""
    reward_model = make_reward_model(reward_mode, scenario, gamma, potential_scale=potential_scale)
    return MarsRoverEnv(
        scenario,
        reward_model,
        render_mode=render_mode,
        max_steps=max_steps,
        rng=rng,
        battery_encoding=battery_encoding,
    )


def split_rngs(seed: int, count: int = 2) -> list[np.random.Generator]:
    """Derive independent generators from one integer seed.

    Environment stochasticity, agent exploration, and start-state sampling draw from
    separate streams, so changing one does not silently reshuffle the others.
    ``SeedSequence.spawn`` is prefix-stable: asking for a third stream leaves the
    first two byte-identical, so adding the curriculum did not move any existing
    baseline.
    """
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(count)]


def checkpoint_schedule(episodes: int, checkpoints: int) -> tuple[int, ...]:
    """Zero-based episode indices at which to measure greedy performance.

    The first checkpoint is always episode 0 -- the untrained table -- so every
    curve starts from a measured point rather than an assumed one. These are the
    *mid-training* points only; :func:`train` adds a final checkpoint on the
    finished table, which is the one every reported evaluation number describes.
    Returns an empty tuple when checkpointing is disabled.
    """
    if checkpoints <= 0 or episodes <= 0:
        return ()
    step = max(1, episodes // checkpoints)
    return tuple(sorted({min(i * step, episodes - 1) for i in range(checkpoints)}))


def train(
    scenario: Scenario,
    config: TrainConfig,
    *,
    progress: bool = False,
    warn_on_stubs: bool = True,
    stream: Any = None,
    checkpoint_callback: Callable[[CheckpointRecord], bool] | None = None,
) -> TrainResult:
    """Run tabular Q-learning for ``config.episodes`` episodes.

    Args:
        scenario: the fixed map to train on. A fresh Q-table is allocated here; a
            table is never carried across scenarios.
        config: hyper-parameters and seed.
        progress: print a short per-window progress line.
        warn_on_stubs: print the teaching-state banner when the human-owned
            functions are still placeholders.
        stream: where warnings and progress go; defaults to ``sys.stderr``.
        checkpoint_callback: called with each mid-training checkpoint as it is
            measured. Returning ``False`` stops training there, which is what lets
            the hyper-parameter search abandon a trial whose curve is already
            hopeless and spend the unused episodes on another one. It cannot change
            what the run learns -- checkpoints are measured on a disjoint seed space
            and on their own environment -- so a run that is never stopped is
            bit-identical with and without a callback.

    Returns:
        A :class:`TrainResult`. Its ``learning_is_meaningful`` flag is ``False``
        while any human-owned function is a stub, and ``episodes_completed`` is the
        episode count actually run, which is lower than ``config.episodes`` when the
        callback stopped the run.
    """
    out = stream if stream is not None else sys.stderr
    pending = teaching_stub_status()
    if pending and warn_on_stubs:
        print(TEACHING_WARNING.format(names=", ".join(pending)), file=out)

    agent_config = config.agent_config()
    env_rng, agent_rng, curriculum_rng = split_rngs(config.seed, 3)
    curriculum = StartStateCurriculum(
        scenario,
        total_episodes=config.episodes,
        anneal_fraction=config.curriculum_fraction,
        strategy=config.curriculum_strategy,
        window_fraction=config.curriculum_window_fraction,
        weight_exponent=config.curriculum_weight_exponent,
        battery_encoding=config.battery_encoding,
    )
    if curriculum.enabled and not curriculum.active and warn_on_stubs:
        print(CURRICULUM_WARNING.format(fraction=config.curriculum_fraction), file=out)
    # A stubbed sampler is not a broken one: it returns the canonical start, so the
    # run is well-formed and simply is not the condition it claims to be. That is
    # exactly the failure worth a banner rather than an exception.
    curriculum_pending = curriculum_stub_status() if curriculum.needs_update_counts else ()
    if curriculum_pending and curriculum.enabled and warn_on_stubs:
        print(
            CURRICULUM_STUB_WARNING.format(
                strategy=config.curriculum_strategy, names=", ".join(curriculum_pending)
            ),
            file=out,
        )
    env = make_env(
        scenario,
        config.reward_mode,
        gamma=config.gamma,
        rng=env_rng,
        potential_scale=config.potential_scale,
        max_steps=config.max_steps,
        battery_encoding=config.battery_encoding,
    )

    q_table = initialize_q_table(
        env.num_states,
        env.num_actions,
        initial_value=config.initial_q,
    )

    records: list[EpisodeRecord] = []
    checkpoints: list[CheckpointRecord] = []
    visit_counts = np.zeros(env.num_states, dtype=np.int64)
    total_env_steps = 0
    start_states: set[RoverState] = set()
    schedule = set(checkpoint_schedule(config.episodes, config.eval_checkpoints))
    checkpoint_index = 0
    episodes_completed = 0
    stopped_early = False

    for episode in range(config.episodes):
        # Measured *before* the episode runs, so checkpoint 0 is the untrained
        # table and every point reports performance after exactly `episode`
        # episodes of learning.
        if episode in schedule:
            checkpoints.append(
                _measure_checkpoint(
                    q_table, scenario, config, episode, total_env_steps, checkpoint_index
                )
            )
            checkpoint_index += 1
            if checkpoint_callback is not None and not checkpoint_callback(checkpoints[-1]):
                stopped_early = True
                break

        epsilon = agent_config.epsilon.value_at(episode)
        # The visit-weighted schedule reads the table's own experience back out;
        # the open-loop ones never look at it, so they are not charged the gather.
        start_state = curriculum.start_state_for(
            episode,
            curriculum_rng,
            visit_counts if curriculum.needs_update_counts else None,
        )
        start_states.add(start_state)
        observation, _ = env.reset(start_state=start_state)
        env_steps_before = total_env_steps
        terminated = truncated = False
        info: dict[str, Any] = {"outcome": "ongoing"}

        while not (terminated or truncated):
            action = select_action(
                q_table,
                observation,
                epsilon,
                agent_rng,
                env.action_mask(env.state),
            )
            next_observation, reward, terminated, truncated, info = env.step(action)

            target = calculate_target(
                reward,
                q_table[next_observation],
                agent_config.gamma,
                terminated,
                truncated,
            )
            td_error = calculate_td_error(float(q_table[observation, action]), target)
            update_q_value(q_table, observation, action, agent_config.learning_rate, td_error)

            # Counted on the state that was *updated*, not the one landed on: this
            # is a record of where learning happened, so a terminal state the agent
            # steps into but never bootstraps from does not accrue experience.
            visit_counts[observation] += 1
            observation = next_observation
            total_env_steps += 1

        records.append(
            _record_from_env(
                env,
                episode,
                epsilon,
                env_steps_before,
                info,
                from_canonical_start=start_state == curriculum.canonical,
            )
        )

        episodes_completed = episode + 1

        if progress and config.log_every > 0 and (episode + 1) % config.log_every == 0:
            _print_progress(records[-config.log_every :], episode, epsilon, config, out)

    # A stopped run broke out *immediately after* a checkpoint, so its curve already
    # ends on the table as it stands; measuring again at the same episode index would
    # append a duplicate point and charge the trial for an evaluation it did not need.
    if schedule and not stopped_early:
        # The final table is what every reported evaluation number describes, so the
        # curve has to end on it rather than on the second-to-last checkpoint.
        checkpoints.append(
            _measure_checkpoint(
                q_table, scenario, config, episodes_completed, total_env_steps, checkpoint_index
            )
        )

    env.close()
    canonical_episodes = len(canonical_start_records(records))
    return TrainResult(
        config=config,
        q_table=q_table,
        records=records,
        pending_human_functions=pending,
        total_env_steps=total_env_steps,
        checkpoints=checkpoints,
        visit_counts=visit_counts,
        episodes_completed=episodes_completed,
        stopped_early=stopped_early,
        metadata={
            "scenario": scenario.name,
            "num_states": env.num_states,
            "battery_encoding": str(config.battery_encoding),
            "battery_levels": env.encoder.battery_levels,
            "battery_bins": [
                env.encoder.binning.label(level) for level in range(env.encoder.battery_levels)
            ]
            if not env.encoder.binning.is_dense
            else None,
            "tied_state_fraction": tied_state_fraction(q_table),
            "learned_state_fraction": learned_state_fraction(q_table, config.initial_q),
            "visited_state_count": int(np.count_nonzero(visit_counts)),
            "curriculum": {
                **curriculum.describe(),
                "pending_human_functions": list(curriculum_pending),
                "distinct_start_states": len(start_states),
                "episodes_from_curriculum": episodes_completed - canonical_episodes,
            },
            "eval_checkpoints": len(checkpoints),
            "episodes_completed": episodes_completed,
            "stopped_early": stopped_early,
        },
    )


def _measure_checkpoint(
    q_table: NDArray[np.float64],
    scenario: Scenario,
    config: TrainConfig,
    episode: int,
    env_steps: int,
    index: int,
) -> CheckpointRecord:
    """Greedy performance on the canonical mission after ``episode`` episodes.

    Runs in a seed space disjoint from both training and final evaluation, on its
    own environment, so measuring costs the training run nothing: remove the
    checkpoints and the learned table is bit-identical. Checkpoint ``index`` uses
    the same seed in every condition, so the curriculum and baseline arms of a seed
    meet the same slips at the same point on the x-axis.
    """
    from .evaluation import evaluate

    result = evaluate(
        q_table,
        scenario,
        config.reward_mode,
        episodes=config.checkpoint_episodes,
        seed=CHECKPOINT_SEED_OFFSET + config.seed * 10_000 + index,
        gamma=config.gamma,
        potential_scale=config.potential_scale,
        max_steps=config.max_steps,
        battery_encoding=config.battery_encoding,
        capture_best=False,
    )
    summary = result.summary
    return CheckpointRecord(
        episode=episode,
        env_steps=env_steps,
        episodes=summary.episodes,
        success_rate=summary.success_rate,
        mean_base_return=summary.mean_base_return,
        mean_delivered_value=summary.mean_delivered_value,
        mean_steps=summary.mean_steps,
    )


def _print_progress(
    recent: list[EpisodeRecord],
    episode: int,
    epsilon: float,
    config: TrainConfig,
    out: Any,
) -> None:
    """One progress line per window.

    ``success`` covers every episode in the window; ``canonical`` covers only those
    that started at the lander. Under a curriculum the two differ a lot early on,
    and only the second one is the learning curve for the mission being evaluated.
    """
    rate = sum(1 for r in recent if r.outcome == "success") / len(recent)
    mean_base = float(np.mean([r.base_return for r in recent]))
    line = (
        f"episode {episode + 1:>6}/{config.episodes}  "
        f"eps={epsilon:0.3f}  success={rate:0.2f}  base_return={mean_base:8.2f}"
    )
    if config.curriculum_fraction > 0.0:
        canonical = canonical_start_records(recent)
        canonical_rate = (
            f"{sum(1 for r in canonical if r.outcome == 'success') / len(canonical):0.2f}"
            if canonical
            else "  n/a"
        )
        line += f"  canonical={canonical_rate} ({len(canonical)}/{len(recent)})"
    print(line, file=out)


def _record_from_env(
    env: MarsRoverEnv,
    episode: int,
    epsilon: float,
    env_steps_before: int,
    info: dict[str, Any],
    *,
    from_canonical_start: bool = True,
) -> EpisodeRecord:
    stats = env.stats
    return EpisodeRecord(
        episode=episode,
        steps=stats.steps,
        base_return=stats.base_return,
        shaped_return=stats.shaped_return,
        outcome=str(info["outcome"]),
        delivered_value=stats.delivered_value,
        battery_remaining=env.state.battery,
        energy_spent=stats.energy_spent,
        slips=stats.slips,
        collisions=stats.collisions,
        invalid_collects=stats.invalid_collects,
        max_directed_edge_repeats=stats.max_directed_edge_repeats,
        max_undirected_edge_repeats=stats.max_undirected_edge_repeats,
        repeated_edge_fraction=stats.repeated_edge_fraction,
        epsilon=epsilon,
        env_steps_before=env_steps_before,
        from_canonical_start=from_canonical_start,
    )


__all__ = [
    "CHECKPOINT_SEED_OFFSET",
    "CURRICULUM_STUB_WARNING",
    "CURRICULUM_WARNING",
    "TEACHING_WARNING",
    "TrainConfig",
    "TrainResult",
    "checkpoint_schedule",
    "make_env",
    "split_rngs",
    "train",
]
