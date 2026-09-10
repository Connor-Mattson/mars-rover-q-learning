"""Animated GIF export of a learned policy performing its mission.

Portfolio evidence, not part of the learning path: nothing here is imported by
:mod:`mars_rover_q.training`, :mod:`mars_rover_q.agent`, or
:mod:`mars_rover_q.environment`, and the experiment grid only reaches it after
every cell has finished.

Frames are drawn from a *recorded* trajectory rather than by re-stepping the
environment, so the animation is the episode that actually happened. Re-running
the policy would re-roll every wheel slip and produce a different mission from the
one whose numbers appear beside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .curriculum import CurriculumStrategy, curriculum_arm_label, curriculum_arm_slug
from .evaluation import Trajectory
from .scenario import Scenario
from .state import SAMPLE_LABELS, RoverState, SampleType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

#: Frames per second in the written GIF. Slow enough to follow a route by eye.
DEFAULT_FPS = 4.0

#: How long the final frame is held, in seconds, so a looping GIF pauses on the
#: outcome instead of snapping back to the lander. Expressed as a duration rather
#: than as repeated frames because Pillow's optimiser collapses identical
#: consecutive frames, which silently undid the repeat-the-frame version.
END_HOLD_SECONDS = 2.0


def _state_from_record(record: list[int]) -> RoverState:
    """Rebuild a :class:`RoverState` from a trajectory's flat state record."""
    row, col, battery, carried = record
    return RoverState(row=row, col=col, battery=battery, carried=SampleType(carried))


def trajectory_frames(
    scenario: Scenario,
    trajectory: Trajectory,
    *,
    caption_lines: Sequence[str] = (),
) -> list[NDArray[np.uint8]]:
    """Render every step of ``trajectory`` to an RGB frame.

    The renderer is constructed in ``rgb_array`` mode, which selects the dummy SDL
    video driver, so this runs headless on a machine with no display.
    """
    from .renderer import MissionRenderer

    renderer = MissionRenderer(scenario, mode="rgb_array")
    frames: list[NDArray[np.uint8]] = []
    try:
        total = len(trajectory.actions)
        for index, record in enumerate(trajectory.states):
            state = _state_from_record(record)
            carried = SAMPLE_LABELS[state.carried]
            lines = [
                *caption_lines,
                "",
                f"step {min(index, total)}/{total}",
                f"battery {state.battery}/{scenario.battery_capacity}",
                f"payload {carried}",
            ]
            if index == len(trajectory.states) - 1:
                lines += [
                    "",
                    f"outcome: {trajectory.outcome}",
                    f"return: {trajectory.base_return:.0f}",
                ]
            frames.append(renderer.draw_state(state, lines))
    finally:
        renderer.close()
    return frames


def write_gif(path: Path, frames: list[NDArray[np.uint8]], *, fps: float = DEFAULT_FPS) -> Path:
    """Write RGB frames to an animated GIF via Pillow.

    Pillow ships with matplotlib, so this adds no dependency. Frames are converted
    to a shared adaptive palette: encoding each frame independently would let the
    palette drift between frames and make the terrain shimmer.
    """
    from PIL import Image

    if not frames:
        raise ValueError("cannot write a GIF with no frames")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    images = [Image.fromarray(np.ascontiguousarray(frame)) for frame in frames]
    palette = images[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    quantised = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    step_ms = int(1000 / fps)
    durations = [step_ms] * len(quantised)
    durations[-1] = step_ms + int(END_HOLD_SECONDS * 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(
        path,
        save_all=True,
        append_images=quantised[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return path


def write_policy_gif(
    path: Path,
    scenario: Scenario,
    trajectory: Trajectory,
    *,
    caption_lines: Sequence[str] = (),
    fps: float = DEFAULT_FPS,
) -> Path:
    """Render one recorded episode to ``path`` as an animated GIF."""
    frames = trajectory_frames(scenario, trajectory, caption_lines=caption_lines)
    return write_gif(path, frames, fps=fps)


def gif_name(
    scenario: str,
    curriculum_fraction: float,
    episodes: int,
    seed: int,
    strategy: CurriculumStrategy | str = CurriculumStrategy.GROWING,
) -> str:
    """Filename for one policy animation.

    The condition is spelled out rather than encoded as ``cf0.5__csgrowing``: these
    files are the ones most likely to be looked at outside the repository, where
    nobody knows what the parameters mean.
    """
    condition = curriculum_arm_slug(curriculum_fraction, strategy)
    return f"{scenario}__{condition}__ep{episodes}__seed{seed}__median-episode.gif"


def caption_for(
    scenario: str,
    curriculum_fraction: float,
    episodes: int,
    seed: int,
    strategy: CurriculumStrategy | str = CurriculumStrategy.GROWING,
) -> tuple[str, ...]:
    """The caption drawn into each frame."""
    condition = curriculum_arm_label(curriculum_fraction, strategy)
    return (f"{condition}", f"{scenario}", f"{episodes} episodes, seed {seed}")


__all__: list[str] = [
    "DEFAULT_FPS",
    "END_HOLD_SECONDS",
    "caption_for",
    "gif_name",
    "trajectory_frames",
    "write_gif",
    "write_policy_gif",
]
