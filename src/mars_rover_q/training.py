"""Training orchestration around the human-owned Q-learning functions.

The loop below is the *only* learning path in the repository, and it calls
:mod:`mars_rover_q.agent`'s five human-owned functions directly. While those
functions are still stubs the loop runs to completion but learns nothing, and
prints a conspicuous teaching-state warning saying so.
"""

from __future__ import annotations

import sys
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
from .environment import MarsRoverEnv
from .metrics import EpisodeRecord
from .rewards import RewardMode, make_reward_model
from .scenario import Scenario

TEACHING_WARNING = """
================================ TEACHING STATE ================================
The following human-owned functions in src/mars_rover_q/agent.py are still
placeholder stubs: {names}

Training will run to completion, but NO LEARNING IS HAPPENING and the resulting
Q-table is meaningless. Do not report these numbers as a result.

Implement the functions listed in .teacher/current.md, then re-run.
================================================================================
"""


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Everything needed to reproduce one training run."""

    scenario: str
    reward_mode: RewardMode = RewardMode.SPARSE
    seed: int = 1
    episodes: int = 4000
    learning_rate: float = 0.2
    gamma: float = 0.99
    initial_q: float = 0.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.6
    potential_scale: float = 1.0
    max_steps: int | None = None
    log_every: int = 50

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
        return payload


@dataclass(slots=True)
class TrainResult:
    """The artefacts of one completed training run."""

    config: TrainConfig
    q_table: NDArray[np.float64]
    records: list[EpisodeRecord]
    pending_human_functions: tuple[str, ...] = ()
    total_env_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def learning_is_meaningful(self) -> bool:
        """``False`` while any human-owned function is still a placeholder."""
        return not self.pending_human_functions


def make_env(
    scenario: Scenario,
    reward_mode: RewardMode | str,
    *,
    gamma: float,
    rng: np.random.Generator,
    potential_scale: float = 1.0,
    max_steps: int | None = None,
    render_mode: str | None = None,
) -> MarsRoverEnv:
    """Build an environment with the reward model for one experimental condition."""
    reward_model = make_reward_model(reward_mode, scenario, gamma, potential_scale=potential_scale)
    return MarsRoverEnv(
        scenario,
        reward_model,
        render_mode=render_mode,
        max_steps=max_steps,
        rng=rng,
    )


def split_rngs(seed: int, count: int = 2) -> list[np.random.Generator]:
    """Derive independent generators from one integer seed.

    Environment stochasticity and agent exploration draw from separate streams, so
    changing one does not silently reshuffle the other.
    """
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(count)]


def train(
    scenario: Scenario,
    config: TrainConfig,
    *,
    progress: bool = False,
    warn_on_stubs: bool = True,
    stream: Any = None,
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

    Returns:
        A :class:`TrainResult`. Its ``learning_is_meaningful`` flag is ``False``
        while any human-owned function is a stub.
    """
    out = stream if stream is not None else sys.stderr
    pending = teaching_stub_status()
    if pending and warn_on_stubs:
        print(TEACHING_WARNING.format(names=", ".join(pending)), file=out)

    agent_config = config.agent_config()
    env_rng, agent_rng = split_rngs(config.seed)
    env = make_env(
        scenario,
        config.reward_mode,
        gamma=config.gamma,
        rng=env_rng,
        potential_scale=config.potential_scale,
        max_steps=config.max_steps,
    )

    q_table = initialize_q_table(
        env.num_states,
        env.num_actions,
        initial_value=config.initial_q,
    )

    records: list[EpisodeRecord] = []
    total_env_steps = 0

    for episode in range(config.episodes):
        epsilon = agent_config.epsilon.value_at(episode)
        observation, _ = env.reset()
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

            observation = next_observation
            total_env_steps += 1

        records.append(_record_from_env(env, episode, epsilon, env_steps_before, info))

        if progress and config.log_every > 0 and (episode + 1) % config.log_every == 0:
            recent = records[-config.log_every :]
            rate = sum(1 for r in recent if r.outcome == "success") / len(recent)
            mean_base = float(np.mean([r.base_return for r in recent]))
            print(
                f"episode {episode + 1:>6}/{config.episodes}  "
                f"eps={epsilon:0.3f}  success={rate:0.2f}  base_return={mean_base:8.2f}",
                file=out,
            )

    env.close()
    return TrainResult(
        config=config,
        q_table=q_table,
        records=records,
        pending_human_functions=pending,
        total_env_steps=total_env_steps,
        metadata={"scenario": scenario.name, "num_states": env.num_states},
    )


def _record_from_env(
    env: MarsRoverEnv,
    episode: int,
    epsilon: float,
    env_steps_before: int,
    info: dict[str, Any],
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
    )


__all__ = ["TEACHING_WARNING", "TrainConfig", "TrainResult", "make_env", "split_rngs", "train"]
