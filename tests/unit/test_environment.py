"""Environment transitions, termination, truncation, and reproducibility."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mars_rover_q.actions import Action
from mars_rover_q.environment import MarsRoverEnv, Outcome
from mars_rover_q.rewards import RewardMode, make_reward_model
from mars_rover_q.scenario import Scenario
from mars_rover_q.state import RoverState, SampleType
from tests.conftest import make_scenario


def build_env(scenario: Scenario, seed: int = 0, **kwargs: Any) -> MarsRoverEnv:
    model = make_reward_model(RewardMode.SPARSE, scenario, 0.99)
    env = MarsRoverEnv(scenario, model, rng=np.random.default_rng(seed), **kwargs)
    env.reset()
    return env


def deterministic_outcome(outcome: str) -> dict[str, Any]:
    """A terrain block whose movement always resolves to ``outcome``."""
    probabilities = dict.fromkeys(("forward", "stay", "left", "right"), 0.0)
    probabilities[outcome] = 1.0
    return {".": {"energy_cost": 1, "move_probabilities": probabilities}}


def test_reset_starts_at_the_lander_with_a_full_battery(tiny_env: MarsRoverEnv) -> None:
    state = tiny_env.state
    assert state.position == tiny_env.scenario.lander
    assert state.battery == tiny_env.scenario.battery_capacity
    assert state.carried is SampleType.NONE
    assert tiny_env.stats.steps == 0


def test_reset_returns_the_encoded_observation(tiny_env: MarsRoverEnv) -> None:
    observation, info = tiny_env.reset()
    assert observation == tiny_env.encoder.encode(tiny_env.state)
    assert info["outcome"] == Outcome.ONGOING.value


def test_move_costs_energy_and_changes_position(tiny_env: MarsRoverEnv) -> None:
    before = tiny_env.state.battery
    _obs, reward, terminated, truncated, info = tiny_env.step(Action.EAST)
    assert tiny_env.state.position == (1, 2)
    assert tiny_env.state.battery == before - 1
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert info["executed_displacement"] == (0, 1)
    assert info["slipped"] is False


def test_wall_collision_keeps_the_rover_and_still_costs_energy(tiny_env: MarsRoverEnv) -> None:
    before = tiny_env.state.battery
    _obs, _reward, _term, _trunc, info = tiny_env.step(Action.NORTH)
    assert tiny_env.state.position == (1, 1)
    assert tiny_env.state.battery == before - 1
    assert info["collided"] is True
    assert info["executed_displacement"] == (0, 0)


def test_out_of_bounds_move_is_treated_as_a_collision() -> None:
    scenario = make_scenario(
        grid=["...", "...", "..."],
        lander=[0, 0],
        samples={
            "basalt": {"position": [0, 1], "value": 40},
            "hydrated_mineral": {"position": [1, 0], "value": 90},
            "biosignature": {"position": [2, 2], "value": 160},
        },
    )
    env = build_env(scenario)
    _obs, _reward, _term, _trunc, info = env.step(Action.NORTH)
    assert env.state.position == (0, 0)
    assert info["collided"] is True


@pytest.mark.parametrize(
    ("outcome", "expected_cell", "slipped"),
    [
        ("forward", (1, 2), False),
        ("stay", (1, 1), True),
        ("left", (1, 1), True),  # north of the lander is a wall
        ("right", (2, 1), True),
    ],
)
def test_seeded_slip_and_deflection_outcomes(
    outcome: str, expected_cell: tuple[int, int], slipped: bool
) -> None:
    scenario = make_scenario(terrain=deterministic_outcome(outcome))
    env = build_env(scenario)
    _obs, _reward, _term, _trunc, info = env.step(Action.EAST)
    assert env.state.position == expected_cell
    assert info["slipped"] is slipped
    assert info["movement_outcome"] == outcome


def test_slip_still_costs_energy() -> None:
    scenario = make_scenario(terrain=deterministic_outcome("stay"))
    env = build_env(scenario)
    before = env.state.battery
    env.step(Action.EAST)
    assert env.state.battery == before - 1


def test_energy_is_charged_for_the_tile_actually_occupied() -> None:
    scenario = make_scenario(
        grid=["#####", "#.~.#", "#.#.#", "#...#", "#####"],
        terrain={
            ".": {
                "energy_cost": 1,
                "move_probabilities": {"forward": 1.0, "stay": 0.0, "left": 0.0, "right": 0.0},
            },
            "~": {
                "energy_cost": 5,
                "move_probabilities": {"forward": 1.0, "stay": 0.0, "left": 0.0, "right": 0.0},
            },
        },
    )
    env = build_env(scenario)
    before = env.state.battery
    env.step(Action.EAST)  # enters the rough tile at (1, 2)
    assert env.state.position == (1, 2)
    assert env.state.battery == before - 5


def test_collect_on_a_sample_tile_locks_the_payload(tiny_env: MarsRoverEnv) -> None:
    tiny_env.step(Action.EAST)
    tiny_env.step(Action.EAST)
    assert tiny_env.state.position == (1, 3)
    _obs, _reward, _term, _trunc, info = tiny_env.step(Action.COLLECT)
    assert tiny_env.state.carried is SampleType.BASALT
    assert info["collected"] is True
    assert info["invalid_collect"] is False


def test_a_second_collect_cannot_swap_the_payload(tiny_env: MarsRoverEnv) -> None:
    for action in (Action.EAST, Action.EAST, Action.COLLECT):
        tiny_env.step(action)
    assert tiny_env.state.carried is SampleType.BASALT
    for action in (Action.SOUTH, Action.SOUTH, Action.COLLECT):
        _obs, _reward, _term, _trunc, info = tiny_env.step(action)
    assert tiny_env.state.position == (3, 3)
    assert tiny_env.state.carried is SampleType.BASALT
    assert info["invalid_collect"] is True


def test_invalid_collect_costs_energy_and_a_step(tiny_env: MarsRoverEnv) -> None:
    before = tiny_env.state.battery
    _obs, reward, _term, _trunc, info = tiny_env.step(Action.COLLECT)
    assert tiny_env.state.carried is SampleType.NONE
    assert tiny_env.state.battery == before - tiny_env.scenario.collect_energy_cost
    assert tiny_env.stats.steps == 1
    assert info["invalid_collect"] is True
    assert reward == 0.0


def test_delivery_terminates_with_the_sample_value(tiny_env: MarsRoverEnv) -> None:
    script = [Action.EAST, Action.EAST, Action.COLLECT, Action.WEST, Action.WEST]
    for action in script[:-1]:
        _obs, _reward, terminated, truncated, _info = tiny_env.step(action)
        assert not (terminated or truncated)
    _obs, reward, terminated, truncated, info = tiny_env.step(script[-1])
    assert (terminated, truncated) == (True, False)
    assert reward == 40.0
    assert info["outcome"] == Outcome.SUCCESS.value
    assert tiny_env.stats.delivered_value == 40


def test_entering_the_lander_without_a_sample_does_not_end_the_episode(
    tiny_env: MarsRoverEnv,
) -> None:
    _obs, _reward, terminated, truncated, _info = tiny_env.step(Action.EAST)
    _obs, _reward, terminated, truncated, info = tiny_env.step(Action.WEST)
    assert (terminated, truncated) == (False, False)
    assert info["outcome"] == Outcome.ONGOING.value


def test_battery_depletion_terminates_with_the_failure_penalty() -> None:
    scenario = make_scenario(battery_capacity=3, max_steps=50)
    env = build_env(scenario)
    outcomes = [env.step(Action.EAST) for _ in range(3)]
    _obs, reward, terminated, truncated, info = outcomes[-1]
    assert env.state.battery == 0
    assert (terminated, truncated) == (True, False)
    assert reward == scenario.battery_penalty
    assert info["outcome"] == Outcome.BATTERY_DEPLETED.value


def test_step_limit_truncates_rather_than_terminates() -> None:
    scenario = make_scenario(battery_capacity=200, max_steps=4)
    env = build_env(scenario)
    for _ in range(3):
        _obs, _reward, terminated, truncated, _info = env.step(Action.EAST)
        assert not (terminated or truncated)
    _obs, reward, terminated, truncated, info = env.step(Action.EAST)
    assert (terminated, truncated) == (False, True)
    assert reward == scenario.step_limit_penalty
    assert info["outcome"] == Outcome.STEP_LIMIT.value


def test_delivery_wins_over_simultaneous_battery_depletion() -> None:
    scenario = make_scenario(battery_capacity=5, max_steps=50)
    env = build_env(scenario)
    env.step(Action.EAST)
    env.step(Action.EAST)
    env.step(Action.COLLECT)
    env.step(Action.WEST)
    _obs, reward, terminated, truncated, info = env.step(Action.WEST)
    assert env.state.battery == 0
    assert (terminated, truncated) == (True, False)
    assert info["outcome"] == Outcome.SUCCESS.value
    assert reward == 40.0


def test_stepping_a_finished_episode_raises(tiny_env: MarsRoverEnv) -> None:
    for action in (Action.EAST, Action.EAST, Action.COLLECT, Action.WEST, Action.WEST):
        tiny_env.step(action)
    with pytest.raises(RuntimeError, match="finished episode"):
        tiny_env.step(Action.EAST)


def test_reset_is_reproducible_from_a_seed() -> None:
    scenario = make_scenario(
        terrain={
            ".": {
                "energy_cost": 1,
                "move_probabilities": {
                    "forward": 0.6,
                    "stay": 0.2,
                    "left": 0.1,
                    "right": 0.1,
                },
            }
        }
    )
    model = make_reward_model(RewardMode.SPARSE, scenario, 0.99)

    def rollout(seed: int) -> list[tuple[int, int]]:
        env = MarsRoverEnv(scenario, model)
        env.reset(seed=seed)
        trail = []
        for _ in range(12):
            env.step(Action.EAST)
            trail.append(env.state.position)
            if env.stats.steps >= scenario.max_steps:
                break
        return trail

    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


def test_two_envs_sharing_a_seed_produce_identical_streams() -> None:
    scenario = make_scenario(
        terrain={
            ".": {
                "energy_cost": 1,
                "move_probabilities": {
                    "forward": 0.5,
                    "stay": 0.3,
                    "left": 0.1,
                    "right": 0.1,
                },
            }
        }
    )
    model = make_reward_model(RewardMode.SPARSE, scenario, 0.99)
    first = MarsRoverEnv(scenario, model, rng=np.random.default_rng(3))
    second = MarsRoverEnv(scenario, model, rng=np.random.default_rng(3))
    first.reset()
    second.reset()
    for _ in range(10):
        a = first.step(Action.SOUTH)
        b = second.step(Action.SOUTH)
        assert a[0] == b[0]
        assert a[4]["movement_outcome"] == b[4]["movement_outcome"]


def test_repeated_edge_statistics_expose_a_two_cell_loop(tiny_env: MarsRoverEnv) -> None:
    for _ in range(4):
        tiny_env.step(Action.EAST)
        tiny_env.step(Action.WEST)
    stats = tiny_env.stats
    assert stats.max_directed_edge_repeats == 4
    assert stats.max_undirected_edge_repeats == 8
    assert stats.repeated_edge_fraction == pytest.approx(7 / 8)


def test_a_direct_route_has_no_repeated_edges(tiny_env: MarsRoverEnv) -> None:
    tiny_env.step(Action.EAST)
    tiny_env.step(Action.EAST)
    assert tiny_env.stats.repeated_edge_fraction == 0.0


def test_info_exposes_the_documented_keys(tiny_env: MarsRoverEnv) -> None:
    _obs, _reward, _term, _trunc, info = tiny_env.step(Action.EAST)
    for key in (
        "base_reward",
        "shaping_reward",
        "reward",
        "battery",
        "payload",
        "outcome",
        "slipped",
        "commanded_action",
        "executed_displacement",
        "reward_mode",
        "scenario",
        "episode",
    ):
        assert key in info
    assert set(info["episode"]) >= {"steps", "base_return", "shaped_return", "delivered_value"}


def test_base_and_shaped_returns_are_tracked_separately() -> None:
    scenario = make_scenario()
    model = make_reward_model(RewardMode.NAIVE_DENSE, scenario, 0.99)
    env = MarsRoverEnv(scenario, model, rng=np.random.default_rng(0))
    env.reset()
    env.step(Action.EAST)
    assert env.stats.base_return == 0.0
    assert env.stats.shaped_return != 0.0


def test_observation_matches_the_encoder_for_every_step(tiny_env: MarsRoverEnv) -> None:
    for action in (Action.EAST, Action.EAST, Action.COLLECT, Action.SOUTH):
        observation, *_ = tiny_env.step(action)
        assert observation == tiny_env.encoder.encode(tiny_env.state)
        assert tiny_env.encoder.decode(observation) == tiny_env.state


def test_action_mask_marks_every_action_legal(tiny_env: MarsRoverEnv) -> None:
    mask = tiny_env.action_mask(RoverState(1, 1, 5))
    assert mask.shape == (tiny_env.num_actions,)
    assert mask.all()


def test_render_mode_none_returns_nothing_and_loads_no_pygame(tiny_env: MarsRoverEnv) -> None:
    assert tiny_env.render() is None
    tiny_env.close()


def test_invalid_render_mode_is_rejected(tiny_scenario: Scenario) -> None:
    model = make_reward_model(RewardMode.SPARSE, tiny_scenario, 0.99)
    with pytest.raises(ValueError, match="render_mode"):
        MarsRoverEnv(tiny_scenario, model, render_mode="ansi")


def test_reset_defaults_to_the_canonical_lander_start(tiny_env: MarsRoverEnv) -> None:
    tiny_env.reset()
    assert tiny_env.state == RoverState(1, 1, 20, SampleType.NONE)


def test_reset_accepts_an_explicit_start_state(tiny_env: MarsRoverEnv) -> None:
    """The seam the training curriculum uses; evaluation never touches it."""
    start = RoverState(3, 3, 12, SampleType.BASALT)
    observation, info = tiny_env.reset(start_state=start)
    assert tiny_env.state == start
    assert observation == tiny_env.encoder.encode(start)
    assert info["battery"] == 12
    assert info["payload"] == "BASALT"
    assert info["position"] == (3, 3)


def test_an_explicit_start_state_still_ends_the_episode_normally(
    tiny_env: MarsRoverEnv,
) -> None:
    """Starting one tile out with a sample delivers on the very next step."""
    tiny_env.reset(start_state=RoverState(1, 2, 5, SampleType.BIOSIGNATURE))
    _obs, reward, terminated, truncated, info = tiny_env.step(Action.WEST)
    assert terminated and not truncated
    assert info["outcome"] == "success"
    assert reward == 160.0


@pytest.mark.parametrize(
    "start",
    [
        RoverState(2, 2, 10, SampleType.NONE),  # wall
        RoverState(9, 9, 10, SampleType.NONE),  # off the map
        RoverState(1, 2, 0, SampleType.BASALT),  # already battery-depleted
        RoverState(1, 1, 10, SampleType.BASALT),  # already a completed delivery
        RoverState(1, 2, 999, SampleType.NONE),  # battery above capacity
    ],
)
def test_reset_rejects_an_impossible_start_state(tiny_env: MarsRoverEnv, start: RoverState) -> None:
    with pytest.raises(ValueError):
        tiny_env.reset(start_state=start)
