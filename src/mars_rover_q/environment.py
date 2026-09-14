"""The Mars sample-return MDP.

A small, typed, Gymnasium-*like* API implemented from scratch: no Gymnasium,
Stable-Baselines, RLlib, CleanRL or TorchRL, so the learning loop stays visible.

Observations are integer Q-table row IDs. Pixel observations are out of scope --
rendering is portfolio evidence, never a learning input.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .actions import ACTION_DELTAS, DEFLECTIONS, NUM_ACTIONS, Action
from .rewards import RewardModel
from .scenario import OUTCOMES, Scenario
from .state import RoverState, SampleType, StateEncoder


class Outcome(StrEnum):
    """How a step ended."""

    ONGOING = "ongoing"
    SUCCESS = "success"
    BATTERY_DEPLETED = "battery_depleted"
    STEP_LIMIT = "step_limit"


TERMINAL_OUTCOMES: Final[frozenset[Outcome]] = frozenset(
    {Outcome.SUCCESS, Outcome.BATTERY_DEPLETED}
)


@dataclass(slots=True)
class EpisodeStats:
    """Cumulative per-episode bookkeeping exposed through ``info``."""

    steps: int = 0
    base_return: float = 0.0
    shaped_return: float = 0.0
    energy_spent: int = 0
    slips: int = 0
    collisions: int = 0
    invalid_collects: int = 0
    delivered_value: int = 0
    directed_edges: Counter[tuple[tuple[int, int], tuple[int, int]]] = field(
        default_factory=Counter
    )
    undirected_edges: Counter[frozenset[tuple[int, int]]] = field(default_factory=Counter)

    @property
    def max_directed_edge_repeats(self) -> int:
        """How often the single most-repeated directed move was taken."""
        return max(self.directed_edges.values(), default=0)

    @property
    def max_undirected_edge_repeats(self) -> int:
        """How often the single most-repeated map edge was traversed either way."""
        return max(self.undirected_edges.values(), default=0)

    @property
    def repeated_edge_fraction(self) -> float:
        """Share of moves that re-traversed an already-used map edge.

        A back-and-forth shaping loop drives this toward 1.0; a direct mission
        route keeps it near 0.0.
        """
        total = sum(self.undirected_edges.values())
        if total == 0:
            return 0.0
        distinct = len(self.undirected_edges)
        return (total - distinct) / total


class MarsRoverEnv:
    """Fixed-map, fully observable sample-return MDP.

    Usage::

        observation, info = env.reset(seed=seed)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        env.close()

    Conventions:

    * ``terminated`` covers delivery and battery depletion; ``truncated`` covers
      the step limit. Q-learning in this project bootstraps across neither.
    * All randomness comes from an injected :class:`numpy.random.Generator`.
    * Energy is charged for the tile the rover *occupies after* the transition,
      so a slip, a wall collision, and an invalid ``COLLECT`` all still cost power.
    """

    metadata: Final[dict[str, Any]] = {"render_modes": [None, "human", "rgb_array"]}

    def __init__(
        self,
        scenario: Scenario,
        reward_model: RewardModel,
        *,
        render_mode: str | None = None,
        max_steps: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode {render_mode!r} not in {self.metadata['render_modes']}")
        self.scenario = scenario
        self.reward_model = reward_model
        self.render_mode = render_mode
        self.max_steps = int(max_steps) if max_steps is not None else scenario.max_steps
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        self.encoder = StateEncoder(scenario.rows, scenario.cols, scenario.battery_capacity)
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

        self._probability_table: dict[int, NDArray[np.float64]] = {
            int(terrain): spec.probability_vector() for terrain, spec in scenario.terrain.items()
        }
        self._state = RoverState(scenario.lander[0], scenario.lander[1], scenario.battery_capacity)
        self._stats = EpisodeStats()
        self._outcome = Outcome.ONGOING
        self._last_info: dict[str, Any] = {}
        self._renderer: Any = None
        self._closed = False

    # -- introspection ----------------------------------------------------

    @property
    def num_states(self) -> int:
        """Number of Q-table rows for this scenario."""
        return self.encoder.num_states

    @property
    def num_actions(self) -> int:
        """Number of Q-table columns."""
        return NUM_ACTIONS

    @property
    def state(self) -> RoverState:
        """The current semantic state."""
        return self._state

    @property
    def observation(self) -> int:
        """The current state's integer Q-table row ID."""
        return self.encoder.encode(self._state)

    @property
    def stats(self) -> EpisodeStats:
        """Cumulative statistics for the episode in progress."""
        return self._stats

    @property
    def last_info(self) -> dict[str, Any]:
        """The most recent ``info`` mapping, for the renderer's HUD."""
        return self._last_info

    def action_mask(self, state: RoverState | None = None) -> NDArray[np.bool_]:
        """Legal-action mask.

        Every action is always legal: illegal-looking choices (driving into a wall,
        collecting on an empty tile) are legal moves that simply waste energy, which
        is part of what the agent has to learn.
        """
        del state
        return np.ones(NUM_ACTIONS, dtype=np.bool_)

    # -- episode lifecycle ------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        start_state: RoverState | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Start a new episode, by default at the lander with a full battery.

        Args:
            seed: when given, replaces the environment generator.
            start_state: an explicit, non-terminal state to begin from. This is the
                seam the training start-state curriculum uses; evaluation and the
                real mission always leave it ``None``. The state is validated, not
                trusted -- an unreachable start would train values for a mission
                that cannot happen.

        Raises:
            ValueError: if ``start_state`` is off the map, on a wall, flat, or
                already a completed delivery.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._state = self._validated_start(start_state)
        self._stats = EpisodeStats()
        self._outcome = Outcome.ONGOING
        self.reward_model.reset(self._state)
        self._last_info = {
            "outcome": Outcome.ONGOING.value,
            "base_reward": 0.0,
            "shaping_reward": 0.0,
            "reward": 0.0,
            "battery": self._state.battery,
            "payload": self._state.carried.name,
            "position": self._state.position,
            "terrain": self.scenario.terrain_at(self._state.position).name,
            "commanded_action": None,
            "executed_displacement": (0, 0),
            "slipped": False,
            "collided": False,
            "collected": False,
            "invalid_collect": False,
            "reward_mode": self.reward_model.mode.value,
            "scenario": self.scenario.name,
            "episode": self._stats_snapshot(),
        }
        return self.observation, dict(self._last_info)

    def _validated_start(self, start_state: RoverState | None) -> RoverState:
        """Check an explicit start state, or build the canonical lander start."""
        if start_state is None:
            return RoverState(
                row=self.scenario.lander[0],
                col=self.scenario.lander[1],
                battery=self.scenario.battery_capacity,
                carried=SampleType.NONE,
            )
        self.encoder.encode(start_state)  # raises for out-of-range rows, cols, battery
        if not self.scenario.is_traversable(start_state.position):
            raise ValueError(f"start state {start_state} is not on a traversable tile")
        if start_state.battery <= 0:
            raise ValueError(f"start state {start_state} is already battery-depleted")
        if (
            start_state.carried is not SampleType.NONE
            and start_state.position == self.scenario.lander
        ):
            raise ValueError(f"start state {start_state} is already a completed delivery")
        return start_state

    def step(self, action: int | Action) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Apply one action and advance the mission by a single step."""
        if self._outcome is not Outcome.ONGOING:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        commanded = Action(int(action))
        previous = self._state

        if commanded is Action.COLLECT:
            next_state, event = self._apply_collect(previous)
        else:
            next_state, event = self._apply_move(previous, commanded)

        self._stats.steps += 1
        self._stats.energy_spent += previous.battery - next_state.battery
        if event["slipped"]:
            self._stats.slips += 1
        if event["collided"]:
            self._stats.collisions += 1
        if event["invalid_collect"]:
            self._stats.invalid_collects += 1
        if previous.position != next_state.position:
            self._stats.directed_edges[(previous.position, next_state.position)] += 1
            self._stats.undirected_edges[frozenset({previous.position, next_state.position})] += 1

        outcome = self._classify(next_state)
        terminated = outcome in TERMINAL_OUTCOMES
        truncated = outcome is Outcome.STEP_LIMIT
        base_reward = self._base_reward(outcome, next_state)
        shaping_reward = self.reward_model.shaping(
            previous,
            commanded,
            next_state,
            episode_over=terminated or truncated,
        )

        self._state = next_state
        self._outcome = outcome
        self._stats.base_return += base_reward
        self._stats.shaped_return += base_reward + shaping_reward
        if outcome is Outcome.SUCCESS:
            self._stats.delivered_value = self.scenario.samples[next_state.carried].value

        info: dict[str, Any] = {
            "outcome": outcome.value,
            "base_reward": base_reward,
            "shaping_reward": shaping_reward,
            "reward": base_reward + shaping_reward,
            "battery": next_state.battery,
            "payload": next_state.carried.name,
            "position": next_state.position,
            "terrain": self.scenario.terrain_at(next_state.position).name,
            "commanded_action": commanded.name,
            "executed_displacement": (
                next_state.row - previous.row,
                next_state.col - previous.col,
            ),
            "reward_mode": self.reward_model.mode.value,
            "scenario": self.scenario.name,
            **event,
            "episode": self._stats_snapshot(),
        }
        self._last_info = info
        if self.render_mode == "human":
            self.render()
        return self.observation, base_reward + shaping_reward, terminated, truncated, dict(info)

    # -- transition internals ---------------------------------------------

    def _apply_collect(self, state: RoverState) -> tuple[RoverState, dict[str, Any]]:
        spec = self.scenario.sample_at(state.position)
        valid = spec is not None and state.carried is SampleType.NONE
        battery = max(0, state.battery - self.scenario.collect_energy_cost)
        carried = spec.sample_type if (valid and spec is not None) else state.carried
        event = {
            "slipped": False,
            "collided": False,
            "collected": valid,
            "invalid_collect": not valid,
            "movement_outcome": "collect",
        }
        return RoverState(state.row, state.col, battery, carried), event

    def _apply_move(
        self, state: RoverState, commanded: Action
    ) -> tuple[RoverState, dict[str, Any]]:
        probabilities = self._probability_table[int(self.scenario.terrain_at(state.position))]
        outcome_name = OUTCOMES[int(self.rng.choice(len(OUTCOMES), p=probabilities))]
        left, right = DEFLECTIONS[commanded]
        executed = {
            "forward": commanded,
            "stay": None,
            "left": left,
            "right": right,
        }[outcome_name]

        collided = False
        target = state.position
        if executed is not None:
            dr, dc = ACTION_DELTAS[executed]
            candidate = (state.row + dr, state.col + dc)
            if self.scenario.is_traversable(candidate):
                target = candidate
            else:
                collided = True

        battery = max(0, state.battery - self.scenario.energy_cost(target))
        event = {
            "slipped": outcome_name != "forward",
            "collided": collided,
            "collected": False,
            "invalid_collect": False,
            "movement_outcome": outcome_name,
        }
        return RoverState(target[0], target[1], battery, state.carried), event

    def _classify(self, next_state: RoverState) -> Outcome:
        # Delivery is checked before depletion: arriving home on the last joule
        # still completes the mission.
        if (
            next_state.carried is not SampleType.NONE
            and next_state.position == self.scenario.lander
        ):
            return Outcome.SUCCESS
        if next_state.battery <= 0:
            return Outcome.BATTERY_DEPLETED
        if self._stats.steps >= self.max_steps:
            return Outcome.STEP_LIMIT
        return Outcome.ONGOING

    def _base_reward(self, outcome: Outcome, next_state: RoverState) -> float:
        if outcome is Outcome.SUCCESS:
            return float(self.scenario.samples[next_state.carried].value)
        if outcome is Outcome.BATTERY_DEPLETED:
            return self.scenario.battery_penalty
        if outcome is Outcome.STEP_LIMIT:
            return self.scenario.step_limit_penalty
        return 0.0

    def _stats_snapshot(self) -> dict[str, Any]:
        stats = self._stats
        return {
            "steps": stats.steps,
            "base_return": stats.base_return,
            "shaped_return": stats.shaped_return,
            "energy_spent": stats.energy_spent,
            "slips": stats.slips,
            "collisions": stats.collisions,
            "invalid_collects": stats.invalid_collects,
            "delivered_value": stats.delivered_value,
            "max_directed_edge_repeats": stats.max_directed_edge_repeats,
            "max_undirected_edge_repeats": stats.max_undirected_edge_repeats,
            "repeated_edge_fraction": stats.repeated_edge_fraction,
        }

    # -- rendering --------------------------------------------------------

    def render(self) -> NDArray[np.uint8] | None:
        """Render the mission-control view for the configured ``render_mode``.

        Returns an RGB array for ``render_mode="rgb_array"`` and ``None`` otherwise.
        Headless training with ``render_mode=None`` never imports or initialises
        Pygame.
        """
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .renderer import MissionRenderer

            self._renderer = MissionRenderer(self.scenario, mode=self.render_mode)
        result: NDArray[np.uint8] | None = self._renderer.draw(self)
        return result

    def close(self) -> None:
        """Release renderer resources. Safe to call more than once."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._closed = True


__all__ = ["TERMINAL_OUTCOMES", "EpisodeStats", "MarsRoverEnv", "Outcome"]
