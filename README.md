# Mars Sample Return: Reward Design in Tabular Q-Learning

A small, framework-free reinforcement-learning study of one question:

> **How does reward design affect the sample efficiency, final mission performance, and
> unintended behavior of a tabular Q-learning rover choosing among scientifically
> valuable but differently risky sample-return missions?**

An autonomous rover lands on Mars with a finite battery. Three samples are within
reach — a nearby basalt core, a hydrated mineral at medium range, and a distant
biosignature candidate worth four times the basalt. It may carry exactly one, and it
must bring it home. The same tabular Q-learning agent is trained under three reward
definitions and compared on identical maps and seeds.

> **Repository status: teaching handoff.** Five algorithmic functions in
> `src/mars_rover_q/agent.py` are intentionally unimplemented and marked
> `TODO(human):`. Everything around them — environment, rewards, renderer, metrics,
> experiment harness, tests — is complete and green. See
> [Expected repository state](#expected-repository-state) and `.teacher/current.md`.
> **No experimental results exist yet, and none are claimed anywhere in this repo.**

---

## The three reward conditions

| Mode | Flag | What it does |
|---|---|---|
| Sparse mission reward | `--reward sparse` | Reward only on delivery (the sample's scientific value) or failure (`-100`). No step penalty: discounting already pressures the rover to finish sooner. |
| Naive dense shaping | `--reward naive_dense` | Sparse base **plus** an asymmetric progress bonus: `+closer_bonus` for moving one shortest-path step toward the subgoal, `farther_penalty` (smaller magnitude) for moving away. **Intentionally flawed** — the asymmetry makes a back-and-forth cycle profitable. |
| Potential-based shaping | `--reward potential` | Sparse base **plus** `F = γ·Φ(s′) − Φ(s)` with `Φ = 0` in every episode-ending state. Φ is the negative energy-weighted distance still to travel on the heuristic mission plan, computed from static map geometry only. |

`base mission return` (no shaping) and `training reward` (with shaping) are logged as
**separate fields everywhere**. Without that separation the shaped agents cannot be
compared honestly, and every comparison plot uses the base return.

---

## The MDP

**State** — `(row, column, battery_remaining, carried_sample)`.

`carried_sample ∈ {NONE, BASALT, HYDRATED_MINERAL, BIOSIGNATURE}`. Battery is an exact
integer, not binned. The map is fixed within a run, so position already determines
terrain; terrain is deliberately *not* part of the state. `StateEncoder` is a tested
bijection onto `range(num_states)` — every table row is a reachable state, with no
padding.

**Actions** — `NORTH, SOUTH, EAST, WEST, COLLECT`. `COLLECT` succeeds only on a sample
tile with an empty bay, and a valid collection locks that payload for the episode. An
invalid attempt still consumes a step and energy.

**Transitions** — `FLAT`, `ROUGH`, `SAND`, `ROCK` are traversable with per-terrain
integer energy costs; `WALL` is not. Rough and sandy tiles may execute the commanded
move, leave the rover in place, or deflect it 90° to either side, with explicit
probabilities from the scenario file. **Every** attempted action costs energy —
slips, deflections, wall collisions, and invalid collections included. Energy is
charged for the tile the rover *occupies after* the transition.

**Termination and truncation**

| Outcome | `terminated` | `truncated` | Base reward |
|---|---|---|---|
| `success` (payload delivered to lander) | ✅ | — | sample's scientific value |
| `battery_depleted` | ✅ | — | `battery_penalty` (default `-100`) |
| `step_limit` | — | ✅ | `step_limit_penalty` (default `-100`) |
| ordinary transition | — | — | `0` |

Delivery is checked before depletion, so arriving home on the last joule still counts.

> **Bootstrapping convention.** In this educational project, Q-learning bootstraps
> across **neither** terminated nor truncated boundaries. Treating truncation as a
> hard boundary is slightly pessimistic — the rover was not really in an absorbing
> state — but it keeps the update rule simple and the same in both cases. The
> potential-based reward matches the convention: `Φ = 0` in any episode-ending state,
> which makes a finished episode's shaping telescope exactly to `−Φ(s₀)`.

**Randomness** — every stochastic draw comes from an injected
`numpy.random.Generator`. There is no hidden global RNG anywhere in the package.
Environment stochasticity and agent exploration draw from separate streams derived
from one integer seed.

---

## Why tabular, and what does *not* generalize

The Q-table is indexed by `(row, column, battery, payload)` **for one specific map**.
A table trained on `safe_corridor` is meaningless on `shaping_trap`: the row IDs refer
to different cells with different terrain. The experiment therefore trains a **fresh
table for every `(scenario, reward mode, seed)` combination**, and no claim in this
repository asserts generalization to an unseen map. Maps are hand-authored and never
procedurally regenerated between episodes.

Tabular is the right choice here because the question is about *reward design*, not
function approximation. With no network in the loop, a difference between conditions
is attributable to the reward and not to an optimizer, an architecture, or a replay
buffer. Neural networks, DQN, policy gradients, actor-critic methods, and pixel
observations are all out of scope — the renderer is portfolio evidence, never a
learning input.

---

## Scenarios

| Scenario | Size | Battery | Purpose |
|---|---|---|---|
| `safe_corridor` | 10×10 | 60 | Walled corridors on flat ground. Teaches collect-and-return with visibly different route lengths and sample values. |
| `risk_value_tradeoff` | 12×12 | 70 | A sand basin ringed by rough terrain. The biosignature is worth 4× the basalt, but the crossing is slow and slippery, so the best expected choice is not visually trivial. |
| `shaping_trap` | 12×12 | 180 | A wide flat plaza north of a rough/rock barrier. Every adjacent pair in the plaza is a two-cell cycle the naive asymmetric bonus pays for, and the battery is large enough to farm it for longer than the mission is worth. |

Sample values are fixed at 40 / 90 / 160 across all three so cross-scenario plots stay
interpretable. Scenario JSON is validated for rectangularity, exactly one lander,
exactly three uniquely typed samples, reachability of every traversable cell, legal
probabilities, positive battery and step limits, and known terrain symbols.

```
mars-rover-q scenarios     # list the bundled maps and their headline numbers
```

---

## Installation

**With `uv` (locked, reproducible):**

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv lock                      # writes uv.lock
uv sync --extra dev          # installs exactly the locked versions
```

**Plain `venv` fallback:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies are `numpy`, `pygame`, and `matplotlib`. There is **no RL
framework dependency** — no Gymnasium, Stable-Baselines, RLlib, CleanRL, or TorchRL —
so the learning loop stays visible in `src/mars_rover_q/training.py`.

---

## Commands

```bash
# Drive the rover yourself and learn the dynamics before writing any Q-learning
python -m mars_rover_q.cli play --scenario safe_corridor

# Train one table and save a run directory
python -m mars_rover_q.cli train --scenario safe_corridor --reward sparse --seed 1

# Greedy-evaluate a saved run with exploration disabled
python -m mars_rover_q.cli evaluate --run artifacts/runs/safe_corridor__sparse__seed1 --episodes 500

# Run the full reward comparison and write plots, CSVs, and a manifest
python -m mars_rover_q.cli experiment --config configs/experiments/reward_comparison.json

# Replay the best recorded evaluation episode in Pygame
python -m mars_rover_q.cli replay --run artifacts/runs/safe_corridor__sparse__seed1 --episode best
```

The installed console script `mars-rover-q` is equivalent to `python -m mars_rover_q.cli`.

**Manual play controls:** arrows or WASD drive, `SPACE` collects, `R` restarts,
`P` toggles the greedy-policy overlay, `Q`/`ESC` quits. In replay, `TAB` pauses,
`.` single-steps, and `[` / `]` change speed.

---

## Reproducibility protocol

1. Every run is fully determined by one integer seed. `split_rngs(seed)` derives
   independent environment and agent generators from it via `numpy.random.SeedSequence`.
2. Evaluation uses a **separate** seed set (`seed + 1_000_000`) so it never replays the
   training stream, and runs with `epsilon = 0`.
3. Each run directory holds a configuration snapshot, seed and package/version
   metadata (including the git commit), per-episode training metrics, final evaluation
   metrics, the learned Q-table, the greedy policy, the best recorded episode, and a
   machine-readable `manifest.json`.
4. The experiment aggregates **every declared seed**. It does not select a best seed.
   Means are reported with 95% Student-t confidence intervals across seeds, and a
   single-sample interval is reported as `nan` rather than as zero width.
5. `configs/experiments/reward_comparison.json` runs 5 seeds; the identical
   `reward_comparison_10seeds.json` runs 10. `smoke.json` is a fast CI-sized version.

---

## Expected repository state

**Before the human TODOs are complete (now):**

- `play`, `scenarios`, rendering, replay, scenario validation, and the whole experiment
  pipeline run end to end.
- `train` and `experiment` run to completion but print a conspicuous **TEACHING STATE**
  warning and record `"learning_is_meaningful": false` in every manifest. The Q-tables
  they produce are meaningless and must not be reported.
- The default `pytest` run is green. The `human_todo` suite fails, narrowly and
  deterministically, on exactly the placeholder behaviours.

**After the human TODOs are complete:**

- The `human_todo` suite passes with no change to its assertions.
- `train` stops printing the teaching warning, and manifests record
  `"learning_is_meaningful": true`.
- The Results section below can be filled in — from generated output only.

---

## Architecture

```
src/mars_rover_q/
├── actions.py      Action enum, movement deltas, 90° deflections
├── state.py        RoverState and the StateEncoder bijection
├── scenario.py     JSON loading, validation, Dijkstra distances, subgoal heuristic
├── environment.py  the MDP: transitions, termination, info, episode statistics
├── rewards.py      the three reward conditions (outside the agent, by design)
├── agent.py        the five human-owned functions + schedules + the stub probe
├── training.py     the visible Q-learning loop
├── evaluation.py   greedy evaluation and trajectory capture
├── metrics.py      episode records, summaries, CIs, threshold metrics
├── plots.py        matplotlib figures with uncertainty bands
├── experiment.py   the grid, run serialisation, aggregation
├── renderer.py     Pygame mission control, manual play, replay
└── cli.py          play / train / evaluate / experiment / replay
```

The dependency direction is one-way: `scenario → state/actions`, `environment →
scenario + rewards`, `training/evaluation → environment + agent`, `experiment →
training + evaluation + plots`, and nothing in the learning path imports `renderer`.

---

## Tests

```bash
# The green suite. This is the default and must pass.
pytest

# With coverage
pytest --cov

# The teaching suite. EXPECTED TO FAIL until agent.py is implemented.
pytest -m human_todo tests/human_todo

# Skip the slower integration tests
pytest -m "not human_todo and not slow"
```

`pytest` deselects `human_todo` by default (see `addopts` in `pyproject.toml`), so the
two commands above are the whole story: the first is the quality gate, the second is
the assignment. Quality gates:

```bash
ruff format --check .
ruff check .
mypy
```

The environment is tested independently of Pygame; the renderer has its own headless
smoke tests using SDL's dummy video driver.

---

## Results

> **Not yet available.** This section is a deliberate placeholder. It will be filled in
> only after the human-owned functions are implemented and
> `configs/experiments/reward_comparison.json` has actually been run, using numbers
> read from `artifacts/<experiment>/summary_aggregated.csv`. Nothing here is predicted
> in advance, and no reward scheme is claimed to win.

The experiment will report, per `(scenario, reward mode)`, aggregated over all seeds
with 95% confidence intervals:

- episodes and environment steps to reach the declared success threshold;
- evaluation success rate;
- mean delivered scientific value;
- **mean base mission return, excluding shaping bonuses**;
- mean energy remaining on success;
- mission length;
- failure-reason distribution (`battery_depleted` vs `step_limit`);
- shaping-loop behaviour (repeated directed and undirected edge traversals).

| Scenario | Reward mode | Success rate | Mean base return | Episodes to threshold | Repeated-edge fraction |
|---|---|---|---|---|---|
| _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

---

## Limitations

- **Simplified rover physics.** Motion is a four-neighbour grid step with a categorical
  slip model. There is no continuous dynamics, no wheel torque, no terramechanics, no
  sensor noise, and no thermal or communication constraint.
- **Full observability.** The rover knows its exact position, battery, and payload. A
  real mission estimates all three. Localisation error is the first thing that would
  break the tabular formulation.
- **One map per table.** Nothing learned here transfers to a map the agent has not
  trained on. See [Why tabular](#why-tabular-and-what-does-not-generalize).
- **Sampling, not planning.** The MDP is small enough to solve exactly with value
  iteration. Q-learning is used because the question is about learning dynamics under
  different reward signals, not about finding the optimal policy fastest.
- **One hand-chosen subgoal.** Both shaping schemes aim at a single sample picked by a
  fixed heuristic. Potential-based shaping cannot change the optimal policy, but a
  badly chosen subgoal can still slow the discovery of a better sample choice.
- **A hand-authored trap.** `shaping_trap` was built to make the naive scheme's exploit
  observable. That the exploit appears there is by construction; how much it costs, and
  whether it appears on the other two maps, is what the experiment measures.

---

## License

MIT.
