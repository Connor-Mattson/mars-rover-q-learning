"""Greedy evaluation of a learned Q-table, and trajectory capture for replay.

Evaluation disables exploration by calling the same human-owned
:func:`mars_rover_q.agent.select_action` with ``epsilon = 0``, on a fixed
evaluation seed set that is separate from the training seeds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .actions import Action
from .agent import select_action
from .environment import Outcome
from .metrics import EpisodeRecord, EpisodeSummary, summarize_episodes
from .rewards import RewardMode
from .scenario import Scenario
from .training import make_env, split_rngs

#: Offset applied to a training seed so evaluation never reuses its stream.
EVAL_SEED_OFFSET = 1_000_000


@dataclass(slots=True)
class Trajectory:
    """A single recorded episode, replayable in the Pygame renderer."""

    scenario: str
    reward_mode: str
    seed: int
    states: list[list[int]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    base_rewards: list[float] = field(default_factory=list)
    shaping_rewards: list[float] = field(default_factory=list)
    movement_outcomes: list[str] = field(default_factory=list)
    outcome: str = Outcome.ONGOING.value
    base_return: float = 0.0
    delivered_value: int = 0

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Trajectory:
        """Rebuild a trajectory from :meth:`as_dict` output."""
        known = {k: payload[k] for k in cls.__dataclass_fields__ if k in payload}
        return cls(**known)


@dataclass(slots=True)
class EvaluationResult:
    """Greedy-evaluation outcome for one trained table."""

    records: list[EpisodeRecord]
    summary: EpisodeSummary
    best_trajectory: Trajectory | None = None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot."""
        return {
            "summary": self.summary.as_dict(),
            "episodes": [record.as_dict() for record in self.records],
        }


def evaluate(
    q_table: NDArray[np.float64],
    scenario: Scenario,
    reward_mode: RewardMode | str,
    *,
    episodes: int = 500,
    seed: int = 1,
    gamma: float = 0.99,
    potential_scale: float = 1.0,
    max_steps: int | None = None,
    capture_best: bool = True,
) -> EvaluationResult:
    """Run ``episodes`` greedy episodes and aggregate the reporting metrics.

    Exploration is off (``epsilon = 0``); the environment keeps its stochastic
    wheel slip, so the success rate reflects real mobility risk.
    """
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    env_rng, agent_rng = split_rngs(seed + EVAL_SEED_OFFSET)
    env = make_env(
        scenario,
        reward_mode,
        gamma=gamma,
        rng=env_rng,
        potential_scale=potential_scale,
        max_steps=max_steps,
    )

    records: list[EpisodeRecord] = []
    best: Trajectory | None = None

    for episode in range(episodes):
        observation, _ = env.reset()
        trajectory = Trajectory(scenario=scenario.name, reward_mode=str(reward_mode), seed=seed)
        terminated = truncated = False
        info: dict[str, Any] = {"outcome": Outcome.ONGOING.value}

        while not (terminated or truncated):
            state = env.state
            trajectory.states.append([state.row, state.col, state.battery, int(state.carried)])
            action = select_action(q_table, observation, 0.0, agent_rng, env.action_mask(state))
            observation, _reward, terminated, truncated, info = env.step(Action(action))
            trajectory.actions.append(int(action))
            trajectory.base_rewards.append(float(info["base_reward"]))
            trajectory.shaping_rewards.append(float(info["shaping_reward"]))
            trajectory.movement_outcomes.append(str(info["movement_outcome"]))

        final = env.state
        trajectory.states.append([final.row, final.col, final.battery, int(final.carried)])
        trajectory.outcome = str(info["outcome"])
        trajectory.base_return = env.stats.base_return
        trajectory.delivered_value = env.stats.delivered_value

        record = EpisodeRecord(
            episode=episode,
            steps=env.stats.steps,
            base_return=env.stats.base_return,
            shaped_return=env.stats.shaped_return,
            outcome=trajectory.outcome,
            delivered_value=env.stats.delivered_value,
            battery_remaining=final.battery,
            energy_spent=env.stats.energy_spent,
            slips=env.stats.slips,
            collisions=env.stats.collisions,
            invalid_collects=env.stats.invalid_collects,
            max_directed_edge_repeats=env.stats.max_directed_edge_repeats,
            max_undirected_edge_repeats=env.stats.max_undirected_edge_repeats,
            repeated_edge_fraction=env.stats.repeated_edge_fraction,
            epsilon=0.0,
        )
        records.append(record)

        if capture_best and (best is None or trajectory.base_return > best.base_return):
            best = trajectory

    env.close()
    return EvaluationResult(
        records=records,
        summary=summarize_episodes(records),
        best_trajectory=best,
    )


__all__ = ["EVAL_SEED_OFFSET", "EvaluationResult", "Trajectory", "evaluate"]
