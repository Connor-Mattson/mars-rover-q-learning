"""Policy GIF export. Pygame runs headless here via the rgb_array render mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from mars_rover_q.animation import caption_for, gif_name, trajectory_frames, write_gif
from mars_rover_q.evaluation import Trajectory, median_trajectory
from mars_rover_q.scenario import Scenario


def _trajectory(scenario: Scenario) -> Trajectory:
    lander = scenario.lander
    return Trajectory(
        scenario=scenario.name,
        reward_mode="sparse",
        seed=1,
        states=[
            [lander[0], lander[1], scenario.battery_capacity, 0],
            [lander[0], lander[1] + 1, scenario.battery_capacity - 1, 0],
            [lander[0], lander[1], scenario.battery_capacity - 2, 1],
        ],
        actions=[1, 3],
        outcome="success",
        base_return=40.0,
        delivered_value=40,
    )


def test_frames_are_rendered_one_per_recorded_state(tiny_scenario: Scenario) -> None:
    trajectory = _trajectory(tiny_scenario)
    frames = trajectory_frames(tiny_scenario, trajectory, caption_lines=("test",))
    assert len(frames) == len(trajectory.states)
    height, width, channels = frames[0].shape
    assert channels == 3
    assert height > 0 and width > 0
    assert frames[0].dtype.name == "uint8"


def test_frames_differ_as_the_rover_moves(tiny_scenario: Scenario) -> None:
    frames = trajectory_frames(tiny_scenario, _trajectory(tiny_scenario))
    assert not (frames[0] == frames[1]).all()


def test_write_gif_produces_a_multi_frame_file(tmp_path: Path, tiny_scenario: Scenario) -> None:
    from PIL import Image

    frames = trajectory_frames(tiny_scenario, _trajectory(tiny_scenario))
    path = write_gif(tmp_path / "policy.gif", frames)
    assert path.exists()
    with Image.open(path) as image:
        assert image.format == "GIF"
        # `n_frames` is declared on GifImageFile, not on the ImageFile base that
        # `Image.open` is typed as returning.
        assert getattr(image, "n_frames") == len(frames)  # noqa: B009
        # The outcome is held rather than repeated: Pillow's optimiser collapses
        # identical consecutive frames, so a repeated final frame vanishes.
        image.seek(len(frames) - 1)
        held = image.info["duration"]
        image.seek(0)
        assert held > image.info["duration"]


def test_write_gif_rejects_an_empty_animation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no frames"):
        write_gif(tmp_path / "empty.gif", [])


def test_write_gif_rejects_a_non_positive_frame_rate(
    tmp_path: Path, tiny_scenario: Scenario
) -> None:
    frames = trajectory_frames(tiny_scenario, _trajectory(tiny_scenario))
    with pytest.raises(ValueError, match="fps must be positive"):
        write_gif(tmp_path / "bad.gif", frames, fps=0)


def test_gif_names_spell_out_the_condition() -> None:
    assert gif_name("safe_corridor", 0.0, 20000, 1).startswith("safe_corridor__no-curriculum__")
    assert "growing-window-curriculum" in gif_name("safe_corridor", 0.5, 20000, 1)
    assert "sliding-window-curriculum" in gif_name("safe_corridor", 0.5, 20000, 1, "sliding")
    # The filename says which episode was chosen, so a best case can never be
    # mistaken for typical behaviour.
    assert gif_name("safe_corridor", 0.5, 20000, 1).endswith("__median-episode.gif")


def test_captions_name_the_condition_not_the_parameter() -> None:
    assert caption_for("safe_corridor", 0.5, 20000, 1)[0] == "Growing Window Curriculum"
    assert caption_for("safe_corridor", 0.0, 20000, 1)[0] == "No Curriculum"
    # A disabled curriculum has no strategy: one control arm, not three.
    assert caption_for("safe_corridor", 0.0, 20000, 1, "sliding")[0] == "No Curriculum"


def test_median_trajectory_picks_a_representative_episode() -> None:
    trajectories = [
        Trajectory(scenario="s", reward_mode="sparse", seed=1, base_return=value)
        for value in (-100.0, 10.0, 40.0, 40.0, 160.0)
    ]
    chosen = median_trajectory(trajectories)
    assert chosen is not None
    assert chosen.base_return == 40.0
    assert median_trajectory([]) is None
