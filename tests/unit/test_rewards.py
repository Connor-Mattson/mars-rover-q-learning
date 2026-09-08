"""Reward-condition behaviour, including the telescoping and cycle properties."""

from __future__ import annotations

import numpy as np
import pytest

from mars_rover_q.actions import Action
from mars_rover_q.environment import MarsRoverEnv
from mars_rover_q.rewards import (
    NaiveDenseShaping,
    PotentialBasedShaping,
    RewardMode,
    SparseMissionReward,
    make_reward_model,
)
from mars_rover_q.scenario import Scenario, resolve_scenario
from mars_rover_q.state import RoverState, SampleType
from tests.conftest import make_scenario


def env_for(scenario: Scenario, mode: RewardMode, gamma: float = 0.99) -> MarsRoverEnv:
    model = make_reward_model(mode, scenario, gamma)
    env = MarsRoverEnv(scenario, model, rng=np.random.default_rng(0))
    env.reset()
    return env


# -- sparse ---------------------------------------------------------------


def test_sparse_adds_no_shaping(tiny_scenario: Scenario) -> None:
    env = env_for(tiny_scenario, RewardMode.SPARSE)
    for action in (Action.EAST, Action.EAST, Action.COLLECT, Action.WEST):
        _obs, reward, _term, _trunc, info = env.step(action)
        assert info["shaping_reward"] == 0.0
        assert reward == info["base_reward"]


def test_sparse_has_no_hidden_step_penalty(tiny_scenario: Scenario) -> None:
    env = env_for(tiny_scenario, RewardMode.SPARSE)
    for _ in range(3):
        _obs, reward, _term, _trunc, _info = env.step(Action.EAST)
        assert reward == 0.0


def test_sparse_pays_the_sample_value_on_delivery(tiny_scenario: Scenario) -> None:
    env = env_for(tiny_scenario, RewardMode.SPARSE)
    for action in (Action.EAST, Action.EAST, Action.COLLECT, Action.WEST):
        env.step(action)
    _obs, reward, terminated, _trunc, _info = env.step(Action.WEST)
    assert terminated
    assert reward == 40.0


# -- naive dense ----------------------------------------------------------


def test_naive_dense_is_asymmetric() -> None:
    scenario = resolve_scenario("shaping_trap")
    model = NaiveDenseShaping(scenario, 0.99)
    target = scenario.samples[model.target_sample].position
    distances = scenario.distances_to(target)

    lander = scenario.lander
    neighbours = [(lander[0] + dr, lander[1] + dc) for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1))]
    closer = min(neighbours, key=lambda c: distances[c])
    assert distances[closer] < distances[lander]

    here = RoverState(lander[0], lander[1], 50)
    there = RoverState(closer[0], closer[1], 49)
    forward = model.shaping(here, Action.SOUTH, there, episode_over=False)
    backward = model.shaping(there, Action.NORTH, here, episode_over=False)
    assert forward == scenario.shaping.closer_bonus
    assert backward == scenario.shaping.farther_penalty
    assert forward + backward > 0.0


def test_naive_dense_pays_for_a_closed_cycle_on_the_trap_map() -> None:
    scenario = resolve_scenario("shaping_trap")
    env = env_for(scenario, RewardMode.NAIVE_DENSE, gamma=1.0)
    total = 0.0
    for _ in range(10):
        for action in (Action.SOUTH, Action.NORTH):
            _obs, _reward, _term, _trunc, info = env.step(action)
            total += float(info["shaping_reward"])
    assert env.state.position == scenario.lander
    assert total > 0.0


def test_naive_dense_leaves_the_base_return_untouched() -> None:
    scenario = resolve_scenario("shaping_trap")
    env = env_for(scenario, RewardMode.NAIVE_DENSE)
    for _ in range(6):
        env.step(Action.SOUTH)
        env.step(Action.NORTH)
    assert env.stats.base_return == 0.0
    assert env.stats.shaped_return > 0.0


def test_naive_dense_gives_no_progress_term_on_the_collection_step() -> None:
    scenario = make_scenario()
    model = NaiveDenseShaping(scenario, 0.99)
    before = RoverState(1, 3, 10, SampleType.NONE)
    after = RoverState(1, 3, 9, SampleType.BASALT)
    assert model.shaping(before, Action.COLLECT, after, episode_over=False) == 0.0


def test_naive_dense_gives_nothing_for_standing_still() -> None:
    scenario = make_scenario()
    model = NaiveDenseShaping(scenario, 0.99)
    here = RoverState(1, 1, 10)
    assert model.shaping(here, Action.NORTH, here, episode_over=False) == 0.0


# -- potential based ------------------------------------------------------


def test_potential_is_zero_in_episode_ending_states() -> None:
    scenario = make_scenario()
    model = PotentialBasedShaping(scenario, 0.99)
    state = RoverState(1, 1, 5, SampleType.BASALT)
    assert model.potential(state, episode_over=True) == 0.0
    assert model.potential(state, episode_over=False) <= 0.0


def test_potential_shaping_is_the_discounted_potential_difference() -> None:
    scenario = make_scenario()
    gamma = 0.9
    model = PotentialBasedShaping(scenario, gamma)
    before = RoverState(1, 1, 10)
    after = RoverState(1, 2, 9)
    expected = gamma * model.potential(after) - model.potential(before)
    assert model.shaping(before, Action.EAST, after, episode_over=False) == pytest.approx(expected)


def test_potential_shaping_telescopes_over_a_full_episode() -> None:
    """With gamma == 1 the shaped return of a finished episode is exactly -Phi(s0)."""
    scenario = make_scenario()
    env = env_for(scenario, RewardMode.POTENTIAL, gamma=1.0)
    model = env.reward_model
    assert isinstance(model, PotentialBasedShaping)
    initial_potential = model.potential(env.state)

    total_shaping = 0.0
    script = [
        Action.EAST,
        Action.EAST,
        Action.COLLECT,
        Action.WEST,
        Action.WEST,
    ]
    for action in script:
        _obs, _reward, terminated, truncated, info = env.step(action)
        total_shaping += float(info["shaping_reward"])
    assert terminated or truncated
    assert total_shaping == pytest.approx(-initial_potential)


def test_potential_shaping_cannot_manufacture_reward_from_a_closed_cycle() -> None:
    """The naive scheme's exploit does not exist under the potential construction."""
    scenario = resolve_scenario("shaping_trap")
    naive_env = env_for(scenario, RewardMode.NAIVE_DENSE, gamma=1.0)
    potential_env = env_for(scenario, RewardMode.POTENTIAL, gamma=1.0)

    naive_total = 0.0
    potential_total = 0.0
    for _ in range(12):
        for action in (Action.SOUTH, Action.NORTH):
            naive_total += float(naive_env.step(action)[4]["shaping_reward"])
            potential_total += float(potential_env.step(action)[4]["shaping_reward"])

    assert naive_env.state.position == scenario.lander
    assert potential_env.state.position == scenario.lander
    assert naive_total > 0.0
    assert potential_total == pytest.approx(0.0)


def test_potential_stage_change_reflects_the_new_subgoal() -> None:
    scenario = make_scenario()
    model = PotentialBasedShaping(scenario, 0.99)
    carrying = RoverState(1, 3, 10, SampleType.BASALT)
    empty = RoverState(1, 3, 10, SampleType.NONE)
    assert model.potential(carrying) > model.potential(empty)


def test_potential_scale_must_be_positive() -> None:
    scenario = make_scenario()
    with pytest.raises(ValueError, match="scale"):
        PotentialBasedShaping(scenario, 0.99, scale=0.0)


# -- factory --------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (RewardMode.SPARSE, SparseMissionReward),
        (RewardMode.NAIVE_DENSE, NaiveDenseShaping),
        (RewardMode.POTENTIAL, PotentialBasedShaping),
        ("sparse", SparseMissionReward),
    ],
)
def test_factory_builds_each_condition(mode: RewardMode | str, expected: type) -> None:
    scenario = make_scenario()
    assert isinstance(make_reward_model(mode, scenario, 0.99), expected)


def test_factory_rejects_unknown_modes() -> None:
    scenario = make_scenario()
    with pytest.raises(ValueError):
        make_reward_model("dense_but_clever", scenario, 0.99)


def test_gamma_is_validated() -> None:
    scenario = make_scenario()
    with pytest.raises(ValueError, match="gamma"):
        make_reward_model(RewardMode.SPARSE, scenario, 0.0)
