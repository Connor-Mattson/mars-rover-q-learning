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

> **Repository status: agent complete, experiments not yet run.** All five algorithmic
> functions in `src/mars_rover_q/agent.py` are implemented and the agent learns end to
> end; the `human_todo` contract suite passes and `train` records
> `"learning_is_meaningful": true`. See
> [Expected repository state](#expected-repository-state).
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

## Exploration: the start-state curriculum

Epsilon-greedy exploration from the lander alone is not enough on these maps. To
collect the 160-point biosignature the rover has to walk a long route out *and* the
whole route back before the battery dies; under a decaying epsilon the chance of doing
that by accident falls off exponentially with the route length, so the terminal reward
is rarely seen and never propagates. The symptom is visible in the policy overlay: away
from the few corridors the agent actually walked, the greedy action flickers between
arrows, because those rows of the table are still at their initial value and
`select_action` is breaking a five-way tie at random.

`metrics.tied_state_fraction` turns that observation into a number — the share of
states whose action values are all still identical, i.e. that were never meaningfully
visited. `train` prints it, and every manifest records it.

The fix is a curriculum over **start states**, not a change to the reward or the MDP:

- Early training episodes begin somewhere close to a finished mission — carrying a
  sample, a few tiles from home — so the terminal reward is reachable by chance and can
  propagate backwards through the table.
- As training progresses the start distribution widens toward harder states and finally
  collapses onto the canonical lander start, so the tail of training is on-distribution.
- Only **physically reachable** states are ever used: a start must be one the rover
  could have driven itself into from the lander (the battery must account for the
  cheapest route there, plus the detour via the sample cell and the `COLLECT` charge
  when the bay is full) and one it can still finish the mission from. Teleporting the
  rover into a state it could never occupy would train values for a mission that does
  not exist.
- **Evaluation never uses the curriculum.** `evaluate` always resets to the lander with
  a full battery, so a curriculum run's reported success rate is directly comparable
  with a run trained without one.

### Three ways to draw from the pool

Widening the support is only half a schedule; the other half is *how the draw is spread
over it*, and that half is what `--curriculum-strategy` selects. The pool and the
admissibility rules above are identical in all three.

| Strategy | Admits | Draws |
|---|---|---|
| `growing` (default) | the easiest `floor(N x progress)` states | uniformly |
| `sliding` | a band of fixed width whose edges both advance | uniformly |
| `visit_weighted` | the same band as `growing` | weighted against already-updated states |

`growing` is the original schedule, and its support only ever widens: a state admitted
in the first hundred episodes keeps full weight for the rest of the anneal. Dry-running
`risk_value_tradeoff` at `--curriculum-fraction 0.5` shows what that costs — **60% of
the anneal's episodes land in the easiest quarter** of a 14298-state pool, and the
minimum sampled difficulty is still 1.0 in the anneal's final fifth.

`sliding` holds the band width constant and moves both edges, retiring solved states
instead of accumulating them; the same dry run puts 17% of episodes in the easiest
quarter and lifts the last fifth's minimum difficulty to 11.0.

`visit_weighted` leaves admission alone and changes the weights instead, drawing each
admitted state in inverse proportion to the Q-updates it has already received. The
counts come from `TrainResult.visit_counts`, the same per-state counter the coverage
figures are drawn from — so this is the only schedule here that closes a loop on the
table's own experience rather than on a clock, and the only one whose behaviour depends
on how learning is actually going.

The strategies were **not** designed to differ in what they admit: `visit_weighted`
calls the same `growing_window_bounds` that `growing` does, so the comparison between
them isolates one variable. Which of the three is best is an open question — Study 3
is what will answer it, and "no arm won" is one of its three pre-registered outcomes.

The curriculum is **off by default** (`curriculum_fraction = 0.0`), so it is an
experimental condition that has to be asked for rather than a silent change to the
baseline. See [Training with a start-state curriculum](#training-with-a-start-state-curriculum)
for the flags.

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

## Three studies

`docs/experiment-plan.md` pre-registers all three, each before any of its data existed.

- **Study 1 — reward design.** 3 scenarios x 3 reward modes x 5 seeds at a fixed 4000
  episodes. Which reward scheme produces the best *base* mission return, and does naive
  shaping induce the loop the `shaping_trap` map makes available?
- **Study 2 — the start-state curriculum across training budgets.** 3 scenarios x
  {curriculum, no curriculum} x {1k, 3k, 5k, 10k, 20k episodes} x 10 seeds, sparse
  reward throughout. How much of the training budget does the curriculum buy back?
  Every point on the budget axis is its own complete run, so the epsilon and curriculum
  anneals scale with the budget rather than being sampled partway through a longer
  schedule. Because both arms of a seed share their RNG streams, the headline statistic
  is the **within-seed paired difference**, not the two conditions' separate intervals.
- **Study 3 — which curriculum schedule.** The same budget axis and the same seeds,
  crossed with the three sampling strategies: does retiring solved states
  (`sliding`) or down-weighting them (`visit_weighted`) beat the growing window that
  oversamples them? One control arm, three treatment arms, each paired against it.

---

## Using it

Every command below is `python -m mars_rover_q.cli <command>`. The installed console
script `mars-rover-q` is equivalent.

| Command | What it does | Trains? |
|---|---|---|
| `scenarios` | List the bundled maps and their headline numbers | no |
| `play` | Drive the rover yourself in Pygame | no |
| `train` | Train one Q-table and write a run directory | yes |
| `evaluate` | Greedy-evaluate a saved run with exploration disabled | no |
| `replay` | Replay a recorded episode in Pygame | no |
| `figures` | Redraw a saved run's diagnostic figures | no |
| `curriculum` | Dry-run the start-state schedules without training | no |
| `experiment` | Run a grid of cells, then analyse and plot it | yes |
| `analyze` | Re-analyse and re-plot a finished experiment | no |

### Playing

Worth doing before reading any of the learning code — the battery arithmetic that makes
the curriculum necessary is much more obvious from the driver's seat.

```bash
python -m mars_rover_q.cli scenarios          # what maps exist
python -m mars_rover_q.cli play --scenario safe_corridor
```

Arrows or WASD drive, `SPACE` collects, `R` restarts, `P` toggles the greedy-policy
overlay, `Q`/`ESC` quits.

### Training

```bash
# One table, one seed, written to a run directory under artifacts/
python -m mars_rover_q.cli train --scenario safe_corridor --reward sparse --seed 1

# The knobs that matter most
python -m mars_rover_q.cli train --scenario risk_value_tradeoff \
    --reward sparse --seed 1 --episodes 20000 \
    --learning-rate 0.2 --gamma 0.99 --epsilon-start 1.0 --epsilon-end 0.05 \
    --eval-episodes 500
```

`--no-figs` skips the per-run figures; `--no-battery-figs` keeps the two summary
figures and skips only the per-battery value maps. Both are worth passing in a loop —
the battery set is 61 PNGs on `safe_corridor`.

#### Training with a start-state curriculum

The curriculum is **off by default**. `--curriculum-fraction` is what turns it on: it
is the share of the episode budget over which the start distribution anneals from easy
reachable states onto the canonical lander start. `--curriculum-strategy` then chooses
how the draw is spread over the admitted states (the three schedules are explained
under [Three ways to draw from the pool](#three-ways-to-draw-from-the-pool)).

```bash
# growing (the default): uniform over the easiest floor(N x progress) states
python -m mars_rover_q.cli train --scenario risk_value_tradeoff \
    --curriculum-fraction 0.5

# sliding: a fixed-width band whose edges both advance, retiring solved states
python -m mars_rover_q.cli train --scenario risk_value_tradeoff \
    --curriculum-fraction 0.5 \
    --curriculum-strategy sliding --curriculum-window-fraction 0.15

# visit_weighted: growing's window, drawn against the Q-updates each state already has
python -m mars_rover_q.cli train --scenario risk_value_tradeoff \
    --curriculum-fraction 0.5 \
    --curriculum-strategy visit_weighted --curriculum-weight-exponent 1.0
```

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--curriculum-fraction` | `0.0` (off) | all | Share of episodes the anneal runs over. `0.5` means the second half trains entirely from the lander. |
| `--curriculum-strategy` | `growing` | all | `growing`, `sliding`, or `visit_weighted`. |
| `--curriculum-window-fraction` | `0.25` | `sliding` | Band width as a share of the pool. Smaller retires solved states faster. |
| `--curriculum-weight-exponent` | `1.0` | `visit_weighted` | Tilt strength. `1.0` draws a state with nine updates a tenth as often as an untouched one; `0.0` is uniform, i.e. the control. |

A knob belonging to another strategy is accepted and ignored, so sweeps can pass a
uniform flag set across arms.

`train` reports the schedule it actually ran, so a curriculum run confirms itself
rather than being taken on trust. From `--scenario risk_value_tradeoff --episodes 200
--curriculum-fraction 0.5`:

```
curriculum: strategy=growing pool=14298 distinct_starts=101 curriculum_episodes=100
```

Every `manifest.json` records the same run under a `curriculum` object -- `strategy`,
`anneal_fraction`, `anneal_episodes`, `pool_size`, `distinct_start_states`,
`episodes_from_curriculum`, both knobs, the pool's difficulty quantiles, and
`pending_human_functions` (empty unless a schedule is running against a stub) -- next
to the run's `tied_state_fraction`.

Two things to know before comparing a curriculum run with a baseline. **Evaluation
never uses the curriculum** — `evaluate` always resets to the lander with a full
battery, so success rates stay comparable across arms. And because early curriculum
episodes are easier by construction, `EpisodeRecord.from_canonical_start` marks which
episodes began at the lander, and every learning curve and threshold metric is computed
from those alone; mixing them would report a success rate belonging to an easier
mission than the one under comparison.

### Evaluating a saved run

```bash
python -m mars_rover_q.cli evaluate \
    --run artifacts/runs/safe_corridor__sparse__seed1 --episodes 500
```

Exploration is disabled (`epsilon = 0`) and the evaluation seed set is deliberately
disjoint from the training one, so this never replays the training stream. It prints a
JSON summary and writes nothing.

### Replaying an episode

```bash
python -m mars_rover_q.cli replay \
    --run artifacts/runs/safe_corridor__sparse__seed1 --episode best
```

`TAB` pauses, `.` single-steps, `[` / `]` change speed, `R` restarts, `ESC` quits.

### Figures

`train` writes these already; `figures` redraws them from a saved run without
retraining.

```bash
python -m mars_rover_q.cli figures --run artifacts/runs/safe_corridor__sparse__seed1

# Only the two summary figures -- skip the 61 per-battery value maps
python -m mars_rover_q.cli figures --run artifacts/runs/safe_corridor__sparse__seed1 \
    --no-battery-figs
```

#### What the figures show

`train` writes two summary PNGs into `<run>/figs`, plus a directory of per-battery
value maps beneath it. None of them is a performance result; they are diagnostics, and
they are the fastest way to see that a run solved its mission out of one corridor of a
mostly untouched table.

- **`state_coverage.png`** — how many of the run's states hold a value the trainer
  actually wrote, against both denominators that matter: every encodable state, and
  the subset standing on traversable ground. A bullet bar per payload splits that by
  what was in the sample bay, with the reachable ceiling marked on each track.
- **`experience_heatmaps.png`** — one map panel per payload (empty bay, and each of
  the three samples), battery level summed away, painted with the number of Q-updates
  each cell received. All four panels share one log colour scale, because the
  empty-bay panel outweighs the carrying panels by orders of magnitude and four
  independent scales would hide exactly that.
- **`figs/q_by_battery/battery_NN.png`** — one frame per battery reading, sixty-one of
  them on `safe_corridor` (`0` through the capacity). Both figures above sum battery
  away; these do not. Every cell is one fully specified state `(row, col, battery,
  payload)`, painted by the value of its best action, labelled with that value and,
  under it in parentheses, the number of Q-updates behind it. An arrow in the top
  right names the action the value came from — `↑ ↓ ← →` to drive, `O` to collect —
  and is drawn only where that action is *unique*: a cell with no arrow has two or
  more actions tied at the top, so its greedy pick is a coin toss. Every frame shares
  one diverging colour scale pinned to zero, so the set can be flipped through as a
  sequence — a cell that darkens as the battery drains means the value really fell.
  `--no-battery-figs` skips the set on both `train` and `figures`.

"Reached" on the map is a superset of "learned" on the coverage bars: under a sparse
reward most updates carry a zero TD error and write nothing, so a cell can collect
thousands of visits and still leave its states at the initial value. The battery frames
are where that shows most plainly — a cell can be labelled `0 (13)` — thirteen updates
that moved nothing.

### Inspecting the curriculum without training

The dry run enumerates the pool and replays the whole anneal against a synthetic visit
counter, reporting where the episodes actually land. It takes about a second and is the
fastest way to see what a schedule change does.

```bash
# All three strategies side by side
python -m mars_rover_q.cli curriculum --scenario risk_value_tradeoff

# One strategy on its own, with its knob turned
python -m mars_rover_q.cli curriculum --scenario risk_value_tradeoff \
    --strategy visit_weighted --weight-exponent 2.0
```

Because the synthetic counter only increments the state it drew, the `visit_weighted`
row understates the tilt a real run gets from `visit_counts`. Read it as a lower bound.

### Running an experiment

```bash
# Reward comparison (Study 1)
python -m mars_rover_q.cli experiment --config configs/experiments/reward_comparison.json

# Curriculum vs no curriculum across five budgets, 300 cells (Study 2)
python -m mars_rover_q.cli experiment \
    --config configs/experiments/curriculum_budget_sweep.json --jobs 8

# The three strategies against one shared control arm, 600 cells (Study 3)
python -m mars_rover_q.cli experiment \
    --config configs/experiments/curriculum_strategy_sweep.json --jobs 8

# Re-analyse and re-plot a finished experiment without retraining anything
python -m mars_rover_q.cli analyze \
    --summary artifacts/curriculum_budget_sweep/summary.json
```

`--jobs` changes only how fast the grid runs. Cells are independent and every generator
is injected, so the per-seed rows are byte-identical at any worker count.
`configs/experiments/*_smoke.json` are second-scale versions of each grid for checking
the plumbing before committing to the real thing.

### Testing

```bash
# The quality gate. All four must pass.
ruff format --check . && ruff check . && mypy && pytest

# The inner loop: ~12s against the gate's ~55s
pytest -m "not slow and not human_todo"

# The teaching suite, deselected from the default run by addopts
pytest -m human_todo tests/human_todo

# With coverage
pytest --cov
```

`pytest` is 370 passed; `pytest -m human_todo tests/human_todo` is 115 passed. The
`slow` marker covers the two tests that render a matplotlib frame per battery level,
which are 43s of the 55s — keep **both** clauses of `not slow and not human_todo`,
because a command-line `-m` replaces `addopts` rather than combining with it, so
`-m "not slow"` alone silently re-enables the teaching suite.

The environment is tested independently of Pygame; the renderer has its own headless
smoke tests using SDL's dummy video driver.
## Reproducibility protocol

1. Every run is fully determined by one integer seed. `split_rngs(seed, 3)` derives
   independent environment, agent, and start-state generators from it via
   `numpy.random.SeedSequence`. `spawn` is prefix-stable, so adding the third stream
   left the first two byte-identical and no earlier run changed.
2. Evaluation uses a **separate** seed set (`seed + 1_000_000`) so it never replays the
   training stream, and runs with `epsilon = 0`.
3. Each run directory holds a configuration snapshot, seed and package/version
   metadata (including the git commit), per-episode training metrics, final evaluation
   metrics, the learned Q-table, the greedy policy, the per-state visit counts, the best
   recorded episode, a `figs/` folder of diagnostic figures, and a machine-readable
   `manifest.json`.
4. The experiment aggregates **every declared seed**. It does not select a best seed.
   Means are reported with 95% Student-t confidence intervals across seeds, and a
   single-sample interval is reported as `nan` rather than as zero width.
5. `configs/experiments/reward_comparison.json` runs 5 seeds; the identical
   `reward_comparison_10seeds.json` runs 10. `smoke.json` is a fast CI-sized version.
6. `--jobs` only changes how fast the grid runs. Cells are independent and every
   generator is injected, so the per-seed rows are identical at any worker count;
   `test_sweep_results_do_not_depend_on_the_worker_count` asserts it.

---

## Expected repository state

**Now that the learning algorithm is implemented:**

- `play`, `scenarios`, rendering, replay, scenario validation, and the whole experiment
  pipeline run end to end.
- `pytest` is green.
- `train` and `experiment` print no **TEACHING STATE** warning and record
  `"learning_is_meaningful": true` in their manifests, so the Q-tables they produce are
  real results.
- The Results section below can now be filled in — from generated output only. It stays
  a labeled placeholder until the reward-scheme comparison has actually been run.

**The start-state curriculum is implemented too**, and `--curriculum-fraction` is live
on every bundled scenario. The **CURRICULUM NOT ACTIVE** banner and the manifest's
`"curriculum": {"requested": true, "active": false}` remain as guards: a run whose pool
comes back empty still starts every episode at the lander, and says so loudly rather
than being mistaken for a curriculum result.

**All three schedules are complete**, including `--curriculum-strategy visit_weighted`,
the closed-loop one that reads `visit_counts` back out of the table. There is no open
assignment; `pytest -m human_todo tests/human_todo` is 115 passed.

The **TEACHING STATE** machinery has not been removed: `agent.teaching_stub_status()`
and `curriculum.curriculum_stub_status()` still probe the six human-owned functions on
every `train`, so a regression that reduced one of them to placeholder behaviour would
resurface the **TEACHING STATE** or **CURRICULUM STRATEGY STUBBED** banner rather than
quietly producing meaningless tables.

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
├── curriculum.py   physically reachable start states and the annealed schedule
├── training.py     the visible Q-learning loop
├── evaluation.py   greedy evaluation and trajectory capture
├── metrics.py      episode records, summaries, CIs, threshold metrics
├── plots.py        matplotlib figures with uncertainty bands
├── run_figures.py  per-run state coverage and experience heatmaps
├── experiment.py   the grid, run serialisation, aggregation
├── renderer.py     Pygame mission control, manual play, replay
└── cli.py          play / train / evaluate / experiment / replay / figures / curriculum
```

The dependency direction is one-way: `scenario → state/actions`, `curriculum →
scenario + state`, `environment → scenario + rewards`, `training/evaluation →
environment + agent + curriculum`, `experiment → training + evaluation + plots`, and
nothing in the learning path imports `renderer` or `run_figures`.

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
