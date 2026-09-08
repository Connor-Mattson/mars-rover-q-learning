"""Shared fixtures. Everything here is deterministic and Pygame-free by default."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mars_rover_q.environment import MarsRoverEnv
from mars_rover_q.rewards import RewardMode, make_reward_model
from mars_rover_q.scenario import Scenario, resolve_scenario, scenario_from_dict

#: A 5x5 sandbox with fully deterministic movement, used for transition tests.
TINY_SCENARIO: dict[str, Any] = {
    "name": "tiny",
    "description": "deterministic 5x5 test fixture",
    "grid": [
        "#####",
        "#...#",
        "#.#.#",
        "#...#",
        "#####",
    ],
    "lander": [1, 1],
    "samples": {
        "basalt": {"position": [1, 3], "value": 40},
        "hydrated_mineral": {"position": [3, 1], "value": 90},
        "biosignature": {"position": [3, 3], "value": 160},
    },
    "battery_capacity": 20,
    "max_steps": 30,
    "collect_energy_cost": 1,
    "battery_penalty": -100.0,
    "step_limit_penalty": -100.0,
    "terrain": {
        ".": {
            "energy_cost": 1,
            "move_probabilities": {"forward": 1.0, "stay": 0.0, "left": 0.0, "right": 0.0},
        }
    },
}


def make_scenario(**overrides: Any) -> Scenario:
    """Build the tiny fixture scenario with optional JSON-level overrides."""
    payload: dict[str, Any] = {**TINY_SCENARIO, **overrides}
    return scenario_from_dict(payload)


@pytest.fixture
def tiny_scenario() -> Scenario:
    """A deterministic 5x5 scenario: no slip, unit energy costs."""
    return make_scenario()


@pytest.fixture
def tiny_env(tiny_scenario: Scenario) -> MarsRoverEnv:
    """A sparse-reward environment over :func:`tiny_scenario`."""
    model = make_reward_model(RewardMode.SPARSE, tiny_scenario, 0.99)
    env = MarsRoverEnv(tiny_scenario, model, rng=np.random.default_rng(0))
    env.reset(seed=0)
    return env


@pytest.fixture(params=["safe_corridor", "risk_value_tradeoff", "shaping_trap"])
def bundled_scenario(request: pytest.FixtureRequest) -> Scenario:
    """Each bundled scenario in turn."""
    return resolve_scenario(str(request.param))
