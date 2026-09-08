"""Headless coverage of the two event-driven Pygame loops.

Real key events are scripted through a patched ``pygame.event.get`` so the manual
mission and replay loops run end to end under SDL's dummy driver.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mars_rover_q.actions import Action
from mars_rover_q.environment import MarsRoverEnv
from mars_rover_q.evaluation import Trajectory
from mars_rover_q.metrics import greedy_actions
from mars_rover_q.rewards import RewardMode, make_reward_model
from mars_rover_q.scenario import Scenario, resolve_scenario

pygame = pytest.importorskip("pygame")

from mars_rover_q.renderer import replay_trajectory, run_manual_mission  # noqa: E402


def scripted(frames: list[list[Any]]) -> Callable[[], list[Any]]:
    """A ``pygame.event.get`` stand-in that plays ``frames`` then quits forever."""
    remaining: Iterator[list[Any]] = iter(frames)

    def get() -> list[Any]:
        return next(remaining, [SimpleNamespace(type=pygame.QUIT)])

    return get


def key(code: int) -> SimpleNamespace:
    return SimpleNamespace(type=pygame.KEYDOWN, key=code)


@pytest.fixture
def corridor() -> Scenario:
    return resolve_scenario("safe_corridor")


def test_manual_mission_drives_collects_and_quits(
    corridor: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pygame.event,
        "get",
        scripted(
            [
                [key(pygame.K_UP)],
                [key(pygame.K_w)],
                [key(pygame.K_SPACE)],
                [key(pygame.K_p)],
                [],
                [key(pygame.K_q)],
            ]
        ),
    )
    model = make_reward_model(RewardMode.SPARSE, corridor, 0.99)
    env = MarsRoverEnv(corridor, model, rng=np.random.default_rng(0))
    summary = run_manual_mission(env, fps=0)

    assert summary["steps"] == 3
    assert summary["outcome"] == "ongoing"
    assert env.stats.invalid_collects == 1
    env.close()


def test_manual_mission_restart_resets_the_episode(
    corridor: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pygame.event,
        "get",
        scripted([[key(pygame.K_UP)], [key(pygame.K_UP)], [key(pygame.K_r)], [key(pygame.K_q)]]),
    )
    model = make_reward_model(RewardMode.SPARSE, corridor, 0.99)
    env = MarsRoverEnv(corridor, model, rng=np.random.default_rng(0))
    run_manual_mission(env, fps=0)

    assert env.stats.steps == 0
    assert env.state.position == corridor.lander
    assert env.state.battery == corridor.battery_capacity
    env.close()


def test_manual_mission_closing_the_window_stops_the_loop(
    corridor: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pygame.event, "get", scripted([[SimpleNamespace(type=pygame.QUIT)]]))
    model = make_reward_model(RewardMode.SPARSE, corridor, 0.99)
    env = MarsRoverEnv(corridor, model, rng=np.random.default_rng(0))
    assert run_manual_mission(env, fps=0) == {}
    env.close()


def test_replay_steps_through_a_recorded_trajectory(
    corridor: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = Trajectory(
        scenario=corridor.name,
        reward_mode=RewardMode.SPARSE.value,
        seed=1,
        actions=[int(Action.NORTH), int(Action.NORTH), int(Action.NORTH)],
        outcome="step_limit",
    )
    monkeypatch.setattr(
        pygame.event,
        "get",
        scripted(
            [
                [key(pygame.K_TAB)],  # pause
                [key(pygame.K_PERIOD)],  # single-step
                [key(pygame.K_RIGHTBRACKET)],
                [key(pygame.K_LEFTBRACKET)],
                [key(pygame.K_TAB)],  # resume
                [],
                [key(pygame.K_p)],
                [key(pygame.K_r)],  # restart
                [key(pygame.K_ESCAPE)],
            ]
        ),
    )
    policy = greedy_actions(np.zeros((10, 5)))
    replay_trajectory(corridor, trajectory, policy=policy, fps=1.0)
