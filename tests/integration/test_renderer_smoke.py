"""Headless renderer smoke tests using SDL's dummy video driver."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from collections.abc import Iterator

import numpy as np
import pytest

from mars_rover_q.actions import Action
from mars_rover_q.environment import MarsRoverEnv
from mars_rover_q.metrics import greedy_actions
from mars_rover_q.rewards import RewardMode, make_reward_model
from mars_rover_q.scenario import Scenario, resolve_scenario

pygame = pytest.importorskip("pygame")

from mars_rover_q.renderer import MissionRenderer  # noqa: E402


@pytest.fixture
def corridor() -> Scenario:
    return resolve_scenario("safe_corridor")


@pytest.fixture
def rgb_env(corridor: Scenario) -> Iterator[MarsRoverEnv]:
    model = make_reward_model(RewardMode.SPARSE, corridor, 0.99)
    env = MarsRoverEnv(corridor, model, render_mode="rgb_array", rng=np.random.default_rng(0))
    env.reset()
    yield env
    env.close()


def test_render_returns_an_rgb_frame(rgb_env: MarsRoverEnv) -> None:
    frame = rgb_env.render()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert frame.shape[0] > 100 and frame.shape[1] > 100


def test_rendering_is_deterministic_for_screenshot_tests(rgb_env: MarsRoverEnv) -> None:
    first = rgb_env.render()
    second = rgb_env.render()
    assert first is not None and second is not None
    assert np.array_equal(first, second)


def test_frame_changes_when_the_rover_moves(rgb_env: MarsRoverEnv) -> None:
    before = rgb_env.render()
    rgb_env.step(Action.NORTH)
    after = rgb_env.render()
    assert before is not None and after is not None
    assert not np.array_equal(before, after)


def test_render_mode_none_never_touches_pygame(corridor: Scenario) -> None:
    model = make_reward_model(RewardMode.SPARSE, corridor, 0.99)
    env = MarsRoverEnv(corridor, model, render_mode=None)
    env.reset()
    assert env.render() is None
    env.close()


def test_policy_overlay_draws_and_toggles(corridor: Scenario, rgb_env: MarsRoverEnv) -> None:
    renderer = MissionRenderer(corridor, mode="rgb_array")
    try:
        table = np.zeros((rgb_env.num_states, rgb_env.num_actions))
        table[:, int(Action.EAST)] = 1.0
        renderer.set_policy(greedy_actions(table), rgb_env.encoder)

        without = renderer.draw(rgb_env)
        assert renderer.toggle_policy_overlay() is True
        with_overlay = renderer.draw(rgb_env)
        assert without is not None and with_overlay is not None
        assert not np.array_equal(without, with_overlay)
        assert renderer.toggle_policy_overlay() is False
        restored = renderer.draw(rgb_env)
        assert restored is not None
        assert np.array_equal(restored, without)
    finally:
        renderer.close()


def test_renderer_rejects_an_unknown_mode(corridor: Scenario) -> None:
    with pytest.raises(ValueError, match="render mode"):
        MissionRenderer(corridor, mode="ansi")


def test_renderer_close_is_idempotent(corridor: Scenario) -> None:
    renderer = MissionRenderer(corridor, mode="rgb_array")
    renderer.close()
    renderer.close()


def test_every_bundled_scenario_renders(bundled_scenario: Scenario) -> None:
    model = make_reward_model(RewardMode.SPARSE, bundled_scenario, 0.99)
    env = MarsRoverEnv(
        bundled_scenario, model, render_mode="rgb_array", rng=np.random.default_rng(0)
    )
    env.reset()
    frame = env.render()
    assert frame is not None
    assert frame.shape[2] == 3
    env.close()
