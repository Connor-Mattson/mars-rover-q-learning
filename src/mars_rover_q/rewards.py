"""The three reward configurations under comparison.

Reward logic lives outside the agent so that an *identical* Q-learning agent can
be run under different reward definitions. Every model returns only the *shaping*
term; the base mission reward is produced by the environment and is always logged
separately, otherwise shaped agents cannot be compared honestly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .actions import Action
from .scenario import Scenario
from .state import RoverState, SampleType


class RewardMode(StrEnum):
    """The three experimental reward conditions."""

    SPARSE = "sparse"
    NAIVE_DENSE = "naive_dense"
    POTENTIAL = "potential"


REWARD_MODE_LABELS: Final[dict[RewardMode, str]] = {
    RewardMode.SPARSE: "Sparse mission reward",
    RewardMode.NAIVE_DENSE: "Naive dense shaping (intentionally flawed)",
    RewardMode.POTENTIAL: "Potential-based shaping",
}


class RewardModel(ABC):
    """Shaping term added on top of the environment's base mission reward.

    Implementations are pure functions of the transition and static scenario
    geometry. They never see learned Q-values and never mutate the environment.
    """

    mode: RewardMode

    def __init__(self, scenario: Scenario, gamma: float) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must lie in (0, 1], got {gamma}")
        self.scenario = scenario
        self.gamma = gamma

    def reset(self, state: RoverState) -> None:
        """Hook called by the environment at the start of every episode."""

    @abstractmethod
    def shaping(
        self,
        prev_state: RoverState,
        action: Action,
        next_state: RoverState,
        *,
        episode_over: bool,
    ) -> float:
        """Shaping reward for one transition.

        Args:
            prev_state: state before the transition.
            action: the commanded action.
            next_state: state after the transition.
            episode_over: ``True`` when ``next_state`` ends the episode, whether by
                termination (delivery, battery depletion) or truncation (step limit).

        Returns:
            The shaping term, to be added to the base mission reward.
        """


class SparseMissionReward(RewardModel):
    """No shaping at all: the base mission reward is the whole signal.

    Discounting already creates pressure to finish sooner, so no step penalty is
    added here.
    """

    mode = RewardMode.SPARSE

    def shaping(
        self,
        prev_state: RoverState,
        action: Action,
        next_state: RoverState,
        *,
        episode_over: bool,
    ) -> float:
        """Always ``0.0``."""
        return 0.0


class NaiveDenseShaping(RewardModel):
    """Intentionally flawed asymmetric progress bonus.

    Moving one shortest-path step closer to the current subgoal earns
    ``closer_bonus``; moving farther earns the smaller-magnitude
    ``farther_penalty``. Because the two magnitudes differ, a back-and-forth cycle
    has *positive* undiscounted shaped return, which is exactly the exploit the
    ``shaping_trap`` scenario is built to expose.

    This is an experimental control condition, not a recommended design.
    """

    mode = RewardMode.NAIVE_DENSE

    def __init__(self, scenario: Scenario, gamma: float) -> None:
        super().__init__(scenario, gamma)
        self.target_sample = scenario.heuristic_target_sample()
        self._to_sample = scenario.distances_to(scenario.samples[self.target_sample].position)
        self._to_lander = scenario.distances_to(scenario.lander)

    def _subgoal_distances(self, state: RoverState) -> NDArray[np.float64]:
        return self._to_lander if state.carried is not SampleType.NONE else self._to_sample

    def shaping(
        self,
        prev_state: RoverState,
        action: Action,
        next_state: RoverState,
        *,
        episode_over: bool,
    ) -> float:
        """Asymmetric progress bonus toward the stage's subgoal."""
        if prev_state.carried is not next_state.carried:
            # The subgoal changed mid-transition; no progress term is well defined.
            return 0.0
        distances = self._subgoal_distances(prev_state)
        before = float(distances[prev_state.position])
        after = float(distances[next_state.position])
        if after < before:
            return self.scenario.shaping.closer_bonus
        if after > before:
            return self.scenario.shaping.farther_penalty
        return 0.0


class PotentialBasedShaping(RewardModel):
    """Standard potential-based shaping ``F = gamma * Phi(s') - Phi(s)``.

    The stage-aware potential is the negative energy-weighted distance still to be
    travelled on the heuristic mission plan:

    * carrying nothing -> distance to the subgoal sample plus that sample's
      distance back to the lander;
    * carrying a sample -> distance back to the lander.

    Potential is zero in every episode-ending state, so the shaped return of a
    complete episode telescopes to ``-Phi(s_0)`` and a closed cycle contributes
    nothing when ``gamma == 1``. The potential is computed once from static
    geometry; learned values never enter it.

    Limits: the potential encodes one hand-chosen mission plan, so it can slow
    discovery of a *different* sample choice even though it cannot change the
    optimal policy of the base MDP.
    """

    mode = RewardMode.POTENTIAL

    def __init__(self, scenario: Scenario, gamma: float, scale: float = 1.0) -> None:
        super().__init__(scenario, gamma)
        if scale <= 0.0:
            raise ValueError(f"potential scale must be positive, got {scale}")
        self.scale = scale
        self.target_sample = scenario.heuristic_target_sample()
        sample_cell = scenario.samples[self.target_sample].position
        self._to_sample = scenario.distances_to(sample_cell)
        self._to_lander = scenario.distances_to(scenario.lander)
        self._sample_return_leg = float(self._to_lander[sample_cell])

    def potential(self, state: RoverState, *, episode_over: bool = False) -> float:
        """Potential of ``state``; zero for any episode-ending state."""
        if episode_over:
            return 0.0
        if state.carried is not SampleType.NONE:
            remaining = float(self._to_lander[state.position])
        else:
            remaining = float(self._to_sample[state.position]) + self._sample_return_leg
        return -self.scale * remaining

    def shaping(
        self,
        prev_state: RoverState,
        action: Action,
        next_state: RoverState,
        *,
        episode_over: bool,
    ) -> float:
        """The discounted potential difference for one transition."""
        return self.gamma * self.potential(next_state, episode_over=episode_over) - self.potential(
            prev_state
        )


def make_reward_model(
    mode: RewardMode | str,
    scenario: Scenario,
    gamma: float,
    *,
    potential_scale: float = 1.0,
) -> RewardModel:
    """Construct the reward model for one experimental condition."""
    resolved = RewardMode(mode)
    if resolved is RewardMode.SPARSE:
        return SparseMissionReward(scenario, gamma)
    if resolved is RewardMode.NAIVE_DENSE:
        return NaiveDenseShaping(scenario, gamma)
    return PotentialBasedShaping(scenario, gamma, scale=potential_scale)


__all__ = [
    "REWARD_MODE_LABELS",
    "NaiveDenseShaping",
    "PotentialBasedShaping",
    "RewardMode",
    "RewardModel",
    "SparseMissionReward",
    "make_reward_model",
]
