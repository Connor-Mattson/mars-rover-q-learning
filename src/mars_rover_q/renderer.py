"""Pygame mission-control view, manual play, and trajectory replay.

The renderer is portfolio evidence, not a learning input: nothing here ever feeds
the agent. Everything is drawn from primitive shapes and the default Pygame font,
so the repository needs no downloaded art and screenshots are reproducible.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from .actions import ACTION_LABELS, Action
from .rewards import REWARD_MODE_LABELS, RewardMode
from .scenario import Scenario, Terrain
from .state import SAMPLE_LABELS, RoverState, SampleType, StateEncoder

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checking only
    from .environment import MarsRoverEnv
    from .evaluation import Trajectory

Color = tuple[int, int, int]

CELL_SIZE: Final[int] = 44
MARGIN: Final[int] = 16
PANEL_WIDTH: Final[int] = 330
LEGEND_HEIGHT: Final[int] = 138

BACKGROUND: Final[Color] = (18, 16, 22)
PANEL_BG: Final[Color] = (30, 27, 36)
GRID_LINE: Final[Color] = (70, 62, 78)
TEXT: Final[Color] = (232, 228, 235)
MUTED: Final[Color] = (160, 152, 170)
ACCENT: Final[Color] = (255, 176, 74)
GOOD: Final[Color] = (110, 205, 140)
BAD: Final[Color] = (226, 100, 96)

#: Terrain fills paired with a hatch/letter cue so the map never relies on colour alone.
TERRAIN_STYLE: Final[dict[Terrain, tuple[Color, str]]] = {
    Terrain.FLAT: ((172, 112, 84), ""),
    Terrain.ROUGH: ((114, 70, 56), "~"),
    Terrain.SAND: ((206, 168, 108), ":"),
    Terrain.ROCK: ((92, 86, 98), "^"),
    Terrain.WALL: ((44, 40, 50), "#"),
}

SAMPLE_STYLE: Final[dict[SampleType, tuple[Color, str]]] = {
    SampleType.BASALT: ((120, 200, 235), "B"),
    SampleType.HYDRATED_MINERAL: ((150, 230, 170), "H"),
    SampleType.BIOSIGNATURE: ((236, 140, 220), "X"),
}

CONTROLS_HELP: Final[tuple[str, ...]] = (
    "arrows/WASD drive   SPACE collect",
    "P policy overlay    R restart",
    "[ / ] speed         . single-step",
    "TAB pause           Q / ESC quit",
)


@dataclass(slots=True)
class OverlayState:
    """What the optional greedy-policy overlay should draw."""

    policy: NDArray[np.int64] | None = None
    encoder: StateEncoder | None = None
    visible: bool = False

    def action_at(self, cell: tuple[int, int], battery: int, carried: SampleType) -> Action | None:
        """Greedy action for ``cell`` in the current battery/payload slice."""
        if not self.visible or self.policy is None or self.encoder is None:
            return None
        try:
            index = self.encoder.encode(RoverState(cell[0], cell[1], battery, carried))
        except ValueError:
            return None
        if index >= self.policy.size:
            return None
        return Action(int(self.policy[index]))


class MissionRenderer:
    """Draws the mission-control view for one scenario.

    Args:
        scenario: the map being rendered.
        mode: ``"human"`` opens a window; ``"rgb_array"`` draws to an offscreen
            surface and returns pixels. Headless training never constructs one.
        caption: window title in ``human`` mode.
    """

    def __init__(
        self,
        scenario: Scenario,
        mode: str = "human",
        *,
        caption: str = "Mars Sample Return - mission control",
    ) -> None:
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"render mode must be 'human' or 'rgb_array', got {mode!r}")
        import pygame

        self.pygame = pygame
        self.scenario = scenario
        self.mode = mode
        self.overlay = OverlayState()

        self.grid_width = scenario.cols * CELL_SIZE
        self.grid_height = scenario.rows * CELL_SIZE
        self.width = MARGIN * 3 + self.grid_width + PANEL_WIDTH
        self.height = MARGIN * 2 + max(self.grid_height, 420) + LEGEND_HEIGHT

        if mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.font.init()
        if mode == "human":
            self.surface = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(caption)
        else:
            self.surface = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 28)
        self._closed = False

    # -- public API -------------------------------------------------------

    def set_policy(self, policy: NDArray[np.int64] | None, encoder: StateEncoder | None) -> None:
        """Attach a greedy policy for the overlay."""
        self.overlay.policy = policy
        self.overlay.encoder = encoder

    def toggle_policy_overlay(self) -> bool:
        """Flip the overlay on or off; returns the new visibility."""
        self.overlay.visible = not self.overlay.visible
        return self.overlay.visible

    def draw(
        self, env: MarsRoverEnv, extra_lines: tuple[str, ...] = ()
    ) -> NDArray[np.uint8] | None:
        """Render one frame from the environment's current state.

        Returns an RGB array in ``rgb_array`` mode, otherwise ``None`` after
        flipping the display.
        """
        self.surface.fill(BACKGROUND)
        self._draw_grid(env.state)
        self._draw_panel(env, extra_lines)
        self._draw_legend()
        if self.mode == "human":
            self.pygame.display.flip()
            return None
        return self.frame()

    def draw_state(self, state: RoverState, lines: Sequence[str] = ()) -> NDArray[np.uint8]:
        """Render one frame from a recorded state, with no environment involved.

        :meth:`draw` reads the live environment, so replaying a stored episode
        through it means re-stepping the environment and re-rolling every wheel
        slip -- the picture would drift away from the episode that was actually
        recorded. This path draws the recorded state directly, so an exported
        animation is the episode that happened rather than a fresh sample from the
        same policy. ``lines`` is the caption, supplied by the caller because the
        narration has to come from the recording too.
        """
        self.surface.fill(BACKGROUND)
        self._draw_grid(state)
        self._draw_caption(lines)
        # No keyboard hints: an exported animation has nothing to press.
        self._draw_legend(show_controls=False)
        return self.frame()

    def frame(self) -> NDArray[np.uint8]:
        """The current surface as an ``(H, W, 3)`` uint8 RGB array."""
        raw = self.pygame.surfarray.array3d(self.surface)
        return np.asarray(raw, dtype=np.uint8).transpose(1, 0, 2)

    def tick(self, fps: float) -> None:
        """Throttle to ``fps`` frames per second."""
        self.clock.tick(fps)

    def close(self) -> None:
        """Shut down Pygame. Safe to call more than once."""
        if not self._closed:
            self.pygame.quit()
            self._closed = True

    # -- drawing internals ------------------------------------------------

    def _cell_rect(self, row: int, col: int) -> Any:
        return self.pygame.Rect(
            MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE, CELL_SIZE, CELL_SIZE
        )

    def _draw_grid(self, state: RoverState) -> None:
        for row in range(self.scenario.rows):
            for col in range(self.scenario.cols):
                rect = self._cell_rect(row, col)
                terrain = self.scenario.terrain_at((row, col))
                fill, glyph = TERRAIN_STYLE[terrain]
                self.pygame.draw.rect(self.surface, fill, rect)
                self.pygame.draw.rect(self.surface, GRID_LINE, rect, 1)
                if glyph:
                    self._blit_centred(self.font, glyph, rect.center, (26, 20, 18), alpha=190)
                action = self.overlay.action_at((row, col), state.battery, state.carried)
                if action is not None and terrain is not Terrain.WALL:
                    self._blit_centred(
                        self.small_font,
                        ACTION_LABELS[action],
                        (rect.centerx, rect.bottom - 10),
                        ACCENT,
                    )

        self._draw_lander(self._cell_rect(*self.scenario.lander))
        for spec in self.scenario.samples.values():
            self._draw_sample(self._cell_rect(*spec.position), spec.sample_type)
        self._draw_rover(self._cell_rect(state.row, state.col), state.carried)

    def _draw_lander(self, rect: Any) -> None:
        pad = 4
        body = self.pygame.Rect(
            rect.x + pad, rect.y + pad, rect.width - 2 * pad, rect.height - 2 * pad
        )
        self.pygame.draw.rect(self.surface, (246, 242, 236), body, border_radius=4)
        self.pygame.draw.rect(self.surface, (40, 38, 44), body, 2, border_radius=4)
        # Three landing legs, so the pad still reads without relying on colour.
        for corner in (body.topleft, body.topright, body.bottomleft, body.bottomright):
            self.pygame.draw.line(self.surface, (120, 112, 118), body.center, corner, 1)
        self._blit_centred(self.font, "L", body.center, (40, 38, 44))

    def _draw_sample(self, rect: Any, sample_type: SampleType) -> None:
        colour, glyph = SAMPLE_STYLE[sample_type]
        centre = (rect.centerx, rect.centery)
        radius = CELL_SIZE // 2 - 12
        self.pygame.draw.circle(self.surface, colour, centre, radius)
        self.pygame.draw.circle(self.surface, (20, 20, 24), centre, radius, 2)
        self._blit_centred(self.small_font, glyph, centre, (20, 20, 24))

    def _draw_rover(self, rect: Any, carried: SampleType) -> None:
        body = rect.inflate(-14, -14)
        self.pygame.draw.rect(self.surface, (250, 250, 252), body, border_radius=3)
        self.pygame.draw.rect(self.surface, (30, 30, 36), body, 2, border_radius=3)
        wheel = max(3, CELL_SIZE // 12)
        for cx in (body.left + 3, body.right - 3):
            for cy in (body.top + 4, body.bottom - 4):
                self.pygame.draw.circle(self.surface, (30, 30, 36), (cx, cy), wheel)
        if carried is not SampleType.NONE:
            colour, glyph = SAMPLE_STYLE[carried]
            bay = self.pygame.Rect(0, 0, 12, 12)
            bay.center = body.center
            self.pygame.draw.rect(self.surface, colour, bay)
            self.pygame.draw.rect(self.surface, (20, 20, 24), bay, 1)
            self._blit_centred(self.small_font, glyph, bay.center, (20, 20, 24))

    def _draw_caption(self, lines: Sequence[str]) -> None:
        """The side panel for :meth:`draw_state`: a title and caller-supplied text."""
        panel = self.pygame.Rect(
            MARGIN * 2 + self.grid_width, MARGIN, PANEL_WIDTH, self.height - LEGEND_HEIGHT - MARGIN
        )
        self.pygame.draw.rect(self.surface, PANEL_BG, panel, border_radius=6)
        x = panel.x + 14
        y = panel.y + 14
        self.surface.blit(self.title_font.render("MISSION CONTROL", True, TEXT), (x, y))
        y += 34
        for line in lines:
            self.surface.blit(self.font.render(line, True, TEXT if line else MUTED), (x, y))
            y += 24

    def _draw_panel(self, env: MarsRoverEnv, extra_lines: tuple[str, ...]) -> None:
        panel = self.pygame.Rect(
            MARGIN * 2 + self.grid_width, MARGIN, PANEL_WIDTH, self.height - LEGEND_HEIGHT - MARGIN
        )
        self.pygame.draw.rect(self.surface, PANEL_BG, panel, border_radius=6)
        info = env.last_info
        state = env.state
        stats = env.stats
        x = panel.x + 14
        y = panel.y + 14

        self.surface.blit(self.title_font.render("MISSION CONTROL", True, TEXT), (x, y))
        y += 30
        mode = RewardMode(str(info.get("reward_mode", RewardMode.SPARSE.value)))
        for line, colour in (
            (f"scenario: {self.scenario.name}", MUTED),
            (f"reward:   {REWARD_MODE_LABELS[mode]}", MUTED),
        ):
            self.surface.blit(self.small_font.render(line, True, colour), (x, y))
            y += 19
        y += 8

        y = self._draw_battery(x, y, panel.width - 28, state.battery)
        y += 10

        outcome = str(info.get("outcome", "ongoing"))
        outcome_colour = GOOD if outcome == "success" else (BAD if outcome != "ongoing" else TEXT)
        rows: tuple[tuple[str, str, Color], ...] = (
            ("payload", SAMPLE_LABELS[state.carried], TEXT),
            (
                "science value",
                str(self.scenario.samples[state.carried].value)
                if state.carried is not SampleType.NONE
                else "0",
                TEXT,
            ),
            ("step", f"{stats.steps} / {env.max_steps}", TEXT),
            ("base return", f"{stats.base_return:+.1f}", TEXT),
            ("shaped return", f"{stats.shaped_return:+.1f}", TEXT),
            ("last base r", f"{float(info.get('base_reward', 0.0)):+.2f}", MUTED),
            ("last shaping", f"{float(info.get('shaping_reward', 0.0)):+.2f}", MUTED),
            ("commanded", str(info.get("commanded_action") or "-"), TEXT),
            ("executed", self._executed_text(info), ACCENT if info.get("slipped") else TEXT),
            ("outcome", outcome, outcome_colour),
            ("repeat edges", f"{stats.repeated_edge_fraction:0.2f}", TEXT),
        )
        for label, value, colour in rows:
            self.surface.blit(self.small_font.render(label, True, MUTED), (x, y))
            self.surface.blit(self.small_font.render(value, True, colour), (x + 118, y))
            y += 20

        y += 6
        for line in extra_lines:
            self.surface.blit(self.small_font.render(line, True, ACCENT), (x, y))
            y += 19

        y += 6
        overlay_text = "policy overlay: ON" if self.overlay.visible else "policy overlay: off"
        self.surface.blit(self.small_font.render(overlay_text, True, MUTED), (x, y))

    def _executed_text(self, info: dict[str, Any]) -> str:
        movement = str(info.get("movement_outcome", "-"))
        displacement = info.get("executed_displacement", (0, 0))
        if info.get("collided"):
            return f"{movement} (wall)"
        if info.get("invalid_collect"):
            return "collect (invalid)"
        return f"{movement} {tuple(displacement)}"

    def _draw_battery(self, x: int, y: int, width: int, battery: int) -> int:
        capacity = self.scenario.battery_capacity
        fraction = max(0.0, min(1.0, battery / capacity))
        self.surface.blit(
            self.small_font.render(f"battery  {battery} / {capacity}", True, MUTED), (x, y)
        )
        y += 20
        outer = self.pygame.Rect(x, y, width, 16)
        self.pygame.draw.rect(self.surface, (58, 52, 66), outer, border_radius=3)
        inner = self.pygame.Rect(x, y, int(width * fraction), 16)
        colour = GOOD if fraction > 0.4 else (ACCENT if fraction > 0.15 else BAD)
        if inner.width > 0:
            self.pygame.draw.rect(self.surface, colour, inner, border_radius=3)
        self.pygame.draw.rect(self.surface, GRID_LINE, outer, 1, border_radius=3)
        return y + 22

    def _draw_legend(self, *, show_controls: bool = True) -> None:
        top = self.height - LEGEND_HEIGHT
        panel = self.pygame.Rect(MARGIN, top, self.width - 2 * MARGIN, LEGEND_HEIGHT - MARGIN)
        self.pygame.draw.rect(self.surface, PANEL_BG, panel, border_radius=6)
        x = panel.x + 14
        y = panel.y + 10
        self.surface.blit(self.font.render("LEGEND", True, TEXT), (x, y))
        y += 24

        col_x = x
        for terrain, (colour, glyph) in TERRAIN_STYLE.items():
            swatch = self.pygame.Rect(col_x, y, 16, 16)
            self.pygame.draw.rect(self.surface, colour, swatch)
            self.pygame.draw.rect(self.surface, GRID_LINE, swatch, 1)
            spec = self.scenario.terrain[terrain]
            label = (
                f"{glyph or '.'} {terrain.name.lower()}"
                if terrain is Terrain.WALL
                else f"{glyph or '.'} {terrain.name.lower()} (e{spec.energy_cost})"
            )
            self.surface.blit(self.small_font.render(label, True, MUTED), (col_x + 22, y + 1))
            col_x += 150
        y += 24

        col_x = x
        for sample_type, (colour, glyph) in SAMPLE_STYLE.items():
            self.pygame.draw.circle(self.surface, colour, (col_x + 8, y + 8), 8)
            sample_spec = self.scenario.samples[sample_type]
            label = f"{glyph} {SAMPLE_LABELS[sample_type]} ({sample_spec.value})"
            self.surface.blit(self.small_font.render(label, True, MUTED), (col_x + 22, y + 1))
            col_x += 230
        y += 24

        col_x = x
        for line in CONTROLS_HELP if show_controls else ():
            self.surface.blit(self.small_font.render(line, True, MUTED), (col_x, y))
            col_x += 240
            if col_x > panel.right - 200:
                col_x = x
                y += 18

    def _blit_centred(
        self,
        font: Any,
        text: str,
        centre: tuple[int, int],
        colour: Color,
        alpha: int = 255,
    ) -> None:
        surface = font.render(text, True, colour)
        if alpha != 255:
            surface.set_alpha(alpha)
        rect = surface.get_rect(center=centre)
        self.surface.blit(surface, rect)


KEY_ACTIONS: Final[dict[str, Action]] = {
    "up": Action.NORTH,
    "w": Action.NORTH,
    "down": Action.SOUTH,
    "s": Action.SOUTH,
    "right": Action.EAST,
    "d": Action.EAST,
    "left": Action.WEST,
    "a": Action.WEST,
    "space": Action.COLLECT,
}


def run_manual_mission(env: MarsRoverEnv, *, fps: int = 30) -> dict[str, Any]:
    """Drive the rover from the keyboard until the window is closed.

    Returns a small summary of the last episode played, so the CLI can report it.
    """
    import pygame

    renderer = MissionRenderer(env.scenario, mode="human")
    env.reset()
    renderer.draw(env, extra_lines=("manual control",))
    summary: dict[str, Any] = {}
    running = True
    finished = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                name = pygame.key.name(event.key)
                if name in ("q", "escape"):
                    running = False
                elif name == "p":
                    renderer.toggle_policy_overlay()
                elif name == "r":
                    env.reset()
                    finished = False
                elif not finished and name in KEY_ACTIONS:
                    _obs, _reward, terminated, truncated, info = env.step(KEY_ACTIONS[name])
                    finished = terminated or truncated
                    summary = {
                        "outcome": info["outcome"],
                        "base_return": env.stats.base_return,
                        "shaped_return": env.stats.shaped_return,
                        "steps": env.stats.steps,
                    }
        lines = ("manual control",) if not finished else ("episode over - press R",)
        renderer.draw(env, extra_lines=lines)
        renderer.tick(fps)

    renderer.close()
    return summary


def replay_trajectory(
    scenario: Scenario,
    trajectory: Trajectory,
    *,
    policy: NDArray[np.int64] | None = None,
    fps: float = 4.0,
) -> None:
    """Replay a recorded episode with pause, single-step, and speed controls."""
    import pygame

    from .environment import MarsRoverEnv
    from .rewards import make_reward_model

    renderer = MissionRenderer(scenario, mode="human", caption="Mars Sample Return - replay")
    env = MarsRoverEnv(scenario, make_reward_model(trajectory.reward_mode, scenario, 0.99))
    encoder = env.encoder
    renderer.set_policy(policy, encoder)

    env.reset()
    index = 0
    paused = False
    step_once = False
    speed = fps
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                name = pygame.key.name(event.key)
                if name in ("q", "escape"):
                    running = False
                elif name == "tab":
                    paused = not paused
                elif name == "period":
                    step_once = True
                elif name == "p":
                    renderer.toggle_policy_overlay()
                elif name == "r":
                    env.reset()
                    index = 0
                elif name == "]":
                    speed = min(60.0, speed * 1.5)
                elif name == "[":
                    speed = max(0.5, speed / 1.5)

        if index < len(trajectory.actions) and (not paused or step_once):
            env.step(Action(trajectory.actions[index]))
            index += 1
            step_once = False

        status = "paused" if paused else f"{speed:0.1f} steps/s"
        lines = (
            f"replay {index}/{len(trajectory.actions)} - {status}",
            f"recorded outcome: {trajectory.outcome}",
        )
        renderer.draw(env, extra_lines=lines)
        renderer.tick(max(1.0, speed) if not paused else 30.0)

    renderer.close()
    env.close()


__all__ = [
    "CONTROLS_HELP",
    "SAMPLE_STYLE",
    "TERRAIN_STYLE",
    "MissionRenderer",
    "OverlayState",
    "replay_trajectory",
    "run_manual_mission",
]
