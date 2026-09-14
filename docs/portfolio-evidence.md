# Portfolio evidence

Filled from verified artifacts only — a file on disk, a number in a generated summary,
a command that was actually run. **Nothing here is predicted or approximated.** Where a
study has not been run its slot says so, and the empty rows are left empty rather than
filled with a plausible number. First person is used only for work Connor actually did;
the boundary between that and the scaffolding is stated explicitly in
[My contribution](#my-contribution) and recorded assignment-by-assignment in
`.teacher/history.md`.

---

## Problem

An autonomous sample-return rover lands on Mars with a finite battery, three reachable
samples of very different scientific value, and room in the bay for exactly one. The
nearby basalt core is worth 40 points and is trivially safe; the biosignature candidate
is worth 160 and sits far enough away that reaching it *and* getting home is a real
energy gamble. The mission is therefore not "find the goal" but "choose which goal is
affordable", and the thing that decides which behaviour a learning agent converges on is
the reward signal it is given. The interesting variable is reward design because two of
the three candidate signals here are defensible on paper — a sparse mission reward, and
a dense progress bonus that any engineer would write first — and one of them is
constructed to be exploitable by an agent that never delivers anything at all.

## Approach

The MDP is hand-written, with no RL framework anywhere in the learning path. State is
`(row, column, battery, carried_sample)`; actions are `NORTH / SOUTH / EAST / WEST /
COLLECT`; rough and sandy tiles may stall the rover or deflect it 90°, every attempted
action costs energy including collisions and invalid collects, and an episode ends on
delivery (`+value`), battery depletion (`-100`), or the step limit (`-100`). Battery is
exact in the environment but binned before it indexes a Q-table row: the default bins at
each sample's lander round trip plus its `COLLECT` charge — the exact charge below which
that sample stops being deliverable — which gives four bins and takes `shaping_trap` from
104,256 rows to 2,304. Three reward conditions sit **outside** the agent, in `rewards.py`:
`sparse` (delivery or failure only), `naive_dense` (sparse plus an asymmetric progress
bonus, deliberately flawed so a back-and-forth cycle is profitable), and `potential`
(sparse plus `F = γ·Φ(s′) − Φ(s)` with `Φ = 0` in every terminal state).

The comparison is controlled by construction: identical agent, identical hand-authored
maps, identical seeds, a fresh Q-table per `(scenario, reward mode, seed)` cell, and
reward as the only factor that moves. `base_return` (no shaping) and `shaped_return`
(what the agent optimised) are separate fields in `info`, in `EpisodeRecord`, in every
CSV and in every plot, so a shaped agent is never credited with its own shaping. Tabular
is the right tool here precisely because the MDP is small enough to solve exactly — the
question is about learning dynamics under different signals, not about finding the
optimum fastest, and having the exact optimum available is what makes "how close did
learning get" answerable at all.

## My contribution

I wrote the learning algorithm and every piece of analysis that turns runs into claims —
twelve functions across five modules, each one handed to me as a specification and a test
contract, implemented without being shown a worked version, and reviewed strictly:

| Module | Functions | What they carry |
|---|---|---|
| `agent.py` | `initialize_q_table`, `calculate_target`, `calculate_td_error`, `update_q_value`, `select_action` | the whole Q-learning core: the Bellman target with no bootstrapping across a `terminated` *or* `truncated` boundary, the signed TD error, the single in-place table update, and ε-greedy selection with random tie-breaking among the greedy-best under an action mask |
| `curriculum.py` | `enumerate_start_states`, `sample_start_state`, `sample_visit_weighted_start_state` | which states the rover could physically have driven itself into *and* still finish from, the annealed draw over them, and the closed-loop draw that down-weights start states the table has already learned |
| `metrics.py`, `sweep.py` | `paired_difference`, `budget_to_reach` | the paired estimator across shared seed streams, and first-crossing budget by interpolation on a `log10` axis — including returning "did not reach" rather than coercing it to a number |
| `tuning.py` | `score_learning_curve`, `pareto_front` | the scalarisation the Optuna study maximises (one normalised area, from which the ceiling, the floor, dominance-monotonicity and both rescaling invariances all fall out) and the two-objective front that scalarisation resolves and therefore hides |

**What was scaffolded and what was not.** This is a learn-by-doing repository, and an AI
assistant built the surrounding machinery: the environment, the renderer, the CLI, the
run/manifest plumbing, the experiment grid, the plotting, the Optuna study loop, and the
test suites. The twelve functions above are mine, and the `.teacher/history.md` record is
only appended after an implementation passes review — never for work the assistant
supplied or substantially repaired. Every one of the four assignments was accepted on a
second pass. In three of the four, the algorithm itself was correct on first submission
and the revisions were engineering defects: a hardcoded action count, `np.zeros(...) +
v` where `np.full` belongs, a guard that raised only incidentally through `max()` on an
empty sequence, a zero-budget case that slipped past a `< 0` check, and a per-episode
weight vector built as a Python comprehension over a 14,298-element band (9.110 ms per
call, vectorised to 0.461 ms). The fourth had a real blocking defect of mine: I clamped
the objective's return axis at the ceiling but not at the floor, and since every trial
starts at `-100`, a large enough failure region cancelled a later recovery and flattened
every score in a study to `0.0` — a fabricated ranking rather than a loud failure. Two
regression tests now pin the floor.

**The direction of the project is also mine, and that is the part I would rather be
asked about.** Four of the design decisions in this repository started as observations I
made from output I was reading rather than from an assignment brief:

- **The exploration failure.** After the Q-core went green I looked at the learned policy
  rather than the success rate and saw the greedy action oscillating at random across
  most of the map. I read that as all-equal action values — `select_action` tossing a
  coin — concluded those states had never been visited, and proposed the fix: anneal a
  curriculum over *physically reachable* start states while continuing to evaluate from
  the canonical lander start, so the numbers stay comparable with every run trained
  without one. It worked: on `risk_value_tradeoff`, seed 1, 4000 episodes, the baseline
  delivered the 160-point biosignature **zero** times and the curriculum run delivered it
  138 times.
- **The oversampled easy end.** I then read the curriculum's own output rather than
  accepting it. `cli curriculum` reported sampled difficulty `min 1.0 / mean 5.4 /
  max 10.0` at `progress = 0.75`: the window had widened but the draw inside it was still
  uniform, so the easiest state was still being drawn as often as anything else. The
  framing that came out of it — a start-state curriculum has two independent knobs,
  which ranks are *admissible* and how the draw is *spread* over them — is what let me
  specify two fixes that isolate one variable each, and keep the visit-weighted arm on
  the growing window's admission rule so the comparison stays controlled.
- **Battery as a clock.** I opened one frame of a Q-value figure set, saw `COLLECT` as
  the greedy action at the lander with an empty bay, and argued it was a representation
  problem rather than a rendering bug: from a fixed start, battery is a deterministic
  function of the path already taken, so the axis was inflating the table without adding
  information. Checking it exactly confirmed the diagnosis and sharpened the answer —
  dropping battery costs nothing from the canonical start (`117.566966` against a true
  `V*` of `117.566967` on `safe_corridor`) but loses badly over the curriculum's start
  distribution, so the resolution was to *coarsen* battery at the affordability
  thresholds rather than drop it. That is the binning described above.
- **The hyper-parameter question, and its two constraints.** Every study before it held
  the hyper-parameters at values nobody had tuned. I asked which set reaches the best
  reward in the fewest episodes, and specified the two things that make that answerable:
  a per-trial episode cap, because without one every eventually-converging setting ties
  and the search silently becomes a search for final performance; and a bounded total
  episode budget, so the cost of the answer is known before it is asked.

I also pushed back three times during the sweep across all three maps, and one of those
overturned a claim that had already been written down: the assertion that the
affordability binning *provably* caps `safe_corridor` at 90 was wrong, and three exact
computations settled it — the best bin-measurable policy is worth `160.000`. Three claims
were retracted and the remaining effect was relabelled a hypothesis.

## Results

**Two studies have been run; two have not.** Study 2 (the start-state curriculum across
training budgets) and Study 4 (the hyper-parameter search) are below, every number read
from a generated `summary_aggregated.csv`, `sweep_analysis.json` or `study.json`.
**Study 1 (the reward-design comparison) and Study 3 (curriculum schedule against
schedule) have not been run**, so no reward scheme is claimed to win anywhere in this
document, and the reward table further down stays empty.

### Study 2 — does the start-state curriculum pay for its episodes?

300 cells: 3 maps × 5 training budgets (1k / 3k / 5k / 10k / 20k) × curriculum fraction
{0.0, 0.5} × 10 seeds, sparse reward throughout, 2.34M training episodes, **7 min 3 s**
wall clock at `--jobs 8`. Each budget trains its own table rather than checkpointing one
long run, because both the ε schedule and the curriculum anneal are defined as fractions
of the budget. Arms are paired on the seed, which shares an RNG stream, so the paired
interval is the right estimator; `n` below is a count of pairs.

Paired difference in **evaluation mean base return** (curriculum − no curriculum),
95% CI across 10 seeds, from `sweep_analysis.json`:

| Scenario | 1,000 | 3,000 | 5,000 | 10,000 | 20,000 |
|---|---|---|---|---|---|
| `shaping_trap` | +0.60 [−0.26, +1.45] | +0.95 [−0.07, +1.97] | **+3.25 [+1.50, +5.00]** | **+138.69 [+122.14, +155.25]** | **+173.81 [+153.86, +193.76]** |
| `safe_corridor` | +38.00 [−4.77, +80.78] | 0.00 [−16.86, +16.86] | +5.00 [−6.31, +16.31] | 0.00 [0.00, 0.00] | +5.00 [−6.31, +16.31] |
| `risk_value_tradeoff` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**The result is `shaping_trap`, and it is large.** At 20,000 episodes the curriculum arm
evaluates at `75.51` base return [55.71, 95.32] against `−98.30` [−99.36, −97.24] without
it; success rate `0.941` against `0.010`; mean delivered value `81.45` against `0.74`.
**10 of 10 seeds** reach the 0.8 success threshold with the curriculum and **0 of 10**
without — `budget_to_reach` puts the crossing at ≈17,300 episodes for the curriculum arm
and returns *did not reach* for the baseline, which is a real answer and is reported as
one rather than as a number. The effect has a threshold: the interval first excludes zero
at 5,000 episodes but only by `+3.25`, and the map flips between 5,000 and 10,000.

**On `safe_corridor` there is no finding**, and the interval says so: every paired
interval straddles zero, and the budgets to reach 90% of target differ by 3.4%
(2,850 vs 2,756 episodes) which is inside the noise. **On `risk_value_tradeoff` both arms
sit at exactly `40.00` at every budget** — under these untuned hyper-parameters neither
arm's greedy policy from the lander ever chooses anything but the basalt within 20,000
episodes, so the paired difference is identically zero. That is a ceiling in the
experiment, not a tie between the arms, and it is one of the reasons Study 4 exists.

A caveat that belongs with these numbers: this grid was run at fixed, untuned
hyper-parameters (lr `0.2`, γ `0.99`, ε `1.0 → 0.05` over 60%, `initial_q 0.0`) and
before the affordability binning existed, so it ran on the dense battery axis —
`shaping_trap` at 104,256 rows. Study 4 later found settings that beat this arm on two
of three maps *with no curriculum at all*.

### Study 4 — which hyper-parameters learn fastest

Four searches, 25 / 32 / 25 / 20 trials, **26.2M training episodes** on one ledger,
sparse reward throughout, seeded TPE sampler (`sampler_seed 0`) with a median pruner, the
battery encoding searched alongside the six agent knobs, and the winner re-trained into a
full run and evaluated over 500 greedy episodes from the canonical lander start with
shaping excluded. `best_affordable_return` is `160.0` on all three maps.

| Map | Cap/trial/seed | Trials | Best score | Mean base return | Success | Of best affordable |
|---|---|---|---|---|---|---|
| `safe_corridor` | 400k | 20 (1 pruned) | 0.7142 | **160.000** | 1.000 | **1.000** |
| `risk_value_tradeoff` | 150k | 32 (10 pruned) | 0.9516 | **158.960** | 0.996 | **0.994** |
| `shaping_trap` | 150k | 25 (3 pruned) | 0.9583 | **160.000** | 1.000 | **1.000** |
| `safe_corridor` | 150k | 25 (1 pruned) | 0.5391 | 90.000 | 1.000 | 0.562 |

The winning configurations:

| Map | lr | γ | `initial_q` | ε | Battery axis | Curriculum |
|---|---|---|---|---|---|---|
| `safe_corridor` (400k) | 0.6114 | 0.9821 | 96.8 | 0.627 → 0.0011 over 64% | dense | sliding, fraction 0.518, window 0.0217 |
| `risk_value_tradeoff` | 0.0763 | 0.9839 | 97.3 | 0.235 → 0.0017 over 49% | affordability | **none** |
| `shaping_trap` | 0.1905 | 0.9143 | 188.9 | 0.617 → 0.0081 over 29% | affordability | **none** |

**Three things that did not go as expected.**

- **Optimistic initialisation does the exploring.** All three winners pair a large
  `initial_q` (97, 97, 189 against a 160-point return scale) with a low, fast-decaying ε.
  None of them wants a long random-exploration phase.
- **No curriculum wins outright on two of three maps.** That runs against most of what
  this repository has spent its effort on, and it is reported as measured.
- **A discount low enough changes the mission rather than the difficulty.** The last row
  of the table is the same map at a shorter cap, kept deliberately. 21 of its 25
  trials chose γ below `0.97` — out of a search range running to `0.9995`, and low enough
  that the 90-point sample genuinely *is* optimal; the winner then converges on it in
  12,500 episodes and scores `0.5391`, beating any slow climb toward a 160 that never
  appears inside the cap. Its confirmation run is a
  perfectly solved MDP — 500 of 500 greedy deliveries in 13 steps — of the wrong sample.
  Raising the cap to 400k dissolved it. The objective is behaving exactly as specified;
  the lesson is about the cap and the discount range as a *pair*.

**Cross-check against exact solutions.** The three winners scored by backward induction
on the true MDP (possible because every traversable tile costs ≥1 energy, so the state
graph is a DAG ordered by battery) give `159.998` / `159.271` / `159.990` against exact
optima of `160.000` / `159.769` / `160.000`, agreeing with the 500-episode Monte-Carlo
evaluation. On `safe_corridor` the winning configuration retrained on seeds 2, 3 and 4 —
none of which the search saw — reached `159.998`, `159.993`, `159.998`. *The exact solver
is a one-off analysis script and is not part of this repository; the Monte-Carlo numbers
above are the ones reproducible from committed code.*

---

## Evidence checklist

| Item | Where it comes from | Status |
|---|---|---|
| Hero screenshot (mission control, mid-episode) | `play` or `replay`, window capture | ☐ not captured |
| Short replay video or GIF | `docs/assets/safe-corridor-tuned-policy.gif` (tuned `safe_corridor`, median episode of a 500-episode greedy batch: 35 steps, `success`, `return 160`, 22/60 battery left); 6 more in `artifacts/curriculum_budget_sweep/gifs/` | ☑ |
| Learning-curve figure | `artifacts/curriculum_budget_sweep/plots/learning_curves.png`; per-study `top_trial_curves.png` under `artifacts/tuning/*/plots/` | ☑ |
| Reward-comparison figure | `artifacts/curriculum_budget_sweep/plots/reward_comparison.png` — **exists but plots the curriculum arms, not reward modes**, because Study 1 has not been run | ☐ blocked on Study 1 |
| Failure-mode figure | `artifacts/curriculum_budget_sweep/plots/failure_modes.png` | ☑ |
| Budget / paired-difference figures | `artifacts/curriculum_budget_sweep/plots/{budget_curves,paired_difference}.png` | ☑ |
| Search figures | `artifacts/tuning/plots/search_overview.png`, `artifacts/tuning/*/plots/{pareto_front,search_progress}.png`, three copied into `docs/assets/` | ☑ |
| Shaping-loop evidence | `eval_repeated_edge_fraction` is present for all 300 Study 2 cells (e.g. `shaping_trap` at 20k: `0.506` with curriculum vs `0.586` without) — but the naive-dense-vs-potential comparison it exists for needs Study 1 | ☐ partial |
| Exact commands run | below, verbatim | ☑ |
| Commit hash | `manifest.json → environment.git_commit`: Study 2 at `63ffea9326cbca041bcdbe7adb671d2949d16bbb`, Study 4 at `4bab6d6bda0b3650706deb134714f9731c478cc2` | ☑ |
| Hardware and runtime | `macOS-26.6.2-arm64-arm-64bit`, Apple Silicon, 10 performance cores. Study 2: 7 min 3 s at `--jobs 8` for 2.34M training episodes, 364 MB of artifacts. Study 4: ~2,900 episodes/s, 26.2M episodes across four searches | ☑ |
| Package versions | `manifest.json → environment`: Python 3.12.7, numpy 2.5.3, matplotlib 3.11.1, pygame 2.6.1, `mars_rover_q` 0.1.0 | ☑ |
| Seeds used | Study 2: `1–10`, all aggregated, none selected. Study 4: seeds `[1, 2]` per trial on `safe_corridor` (150k) and `risk_value_tradeoff`, `[1]` on `shaping_trap` and `safe_corridor` (400k); `sampler_seed 0` throughout | ☑ |
| Quality gates, measured 2026-09-14 | `ruff format --check .` (62 files), `ruff check .`, `mypy` (52 files, clean), `pytest` **533 passed, 166 deselected**, `pytest -m human_todo tests/human_todo` **166 passed** | ☑ |

### Exact commands

```bash
# Study 2 — curriculum vs no curriculum across five budgets, 300 cells
python -m mars_rover_q.cli experiment \
    --config configs/experiments/curriculum_budget_sweep.json --jobs 8
python -m mars_rover_q.cli analyze \
    --summary artifacts/curriculum_budget_sweep/summary.json

# Study 4 — the four hyper-parameter searches
python -m mars_rover_q.cli tune --config configs/tuning/safe_corridor_search_v2.json
python -m mars_rover_q.cli tune --config configs/tuning/risk_value_tradeoff_search_v2.json
python -m mars_rover_q.cli tune --config configs/tuning/shaping_trap_search_v2.json
python -m mars_rover_q.cli tune --config configs/tuning/safe_corridor_search_deep.json

# The gates
ruff format --check . && ruff check . && mypy && pytest
pytest -m human_todo tests/human_todo
```

## Quantitative results table

Fill from `artifacts/<experiment>/summary_aggregated.csv`. Report mean ± 95% CI across
**all** seeds. Never report a single seed.

> **Study 1 has not been run.** Every row below stays empty until
> `configs/experiments/reward_comparison.json` produces a real summary. Single-seed
> `naive_dense` and `potential` runs exist under `artifacts/runs/` and were used to
> exercise the pipeline; they are **not** a comparison and are not quoted here.

| Scenario | Reward mode | Success rate | Mean base return | Mean delivered value | Episodes to 0.8 | Repeated-edge fraction |
|---|---|---|---|---|---|---|
| safe_corridor | sparse | | | | | |
| safe_corridor | naive_dense | | | | | |
| safe_corridor | potential | | | | | |
| risk_value_tradeoff | sparse | | | | | |
| risk_value_tradeoff | naive_dense | | | | | |
| risk_value_tradeoff | potential | | | | | |
| shaping_trap | sparse | | | | | |
| shaping_trap | naive_dense | | | | | |
| shaping_trap | potential | | | | | |

### Study 2, filled (sparse reward, 20,000-episode budget, 10 seeds)

From `artifacts/curriculum_budget_sweep/summary_aggregated.csv`. "Seeds to 0.8" is a
count of seeds reaching the threshold, not a mean over the ones that did.

| Scenario | Arm | Success rate | Mean base return | Mean delivered value | Seeds to 0.8 | Repeated-edge fraction |
|---|---|---|---|---|---|---|
| safe_corridor | no curriculum | 1.000 | 40.00 [40.00, 40.00] | 40.00 | 10/10 | 0.808 |
| safe_corridor | curriculum 0.5 | 1.000 | 45.00 [33.69, 56.31] | 45.00 | 10/10 | 0.516 |
| risk_value_tradeoff | no curriculum | 1.000 | 40.00 [40.00, 40.00] | 40.00 | 10/10 | 0.623 |
| risk_value_tradeoff | curriculum 0.5 | 1.000 | 40.00 [40.00, 40.00] | 40.00 | 10/10 | 0.458 |
| shaping_trap | no curriculum | 0.010 | −98.30 [−99.36, −97.24] | 0.74 | 0/10 | 0.586 |
| shaping_trap | curriculum 0.5 | 0.941 | 75.51 [55.71, 95.32] | 81.45 | 10/10 | 0.506 |

**Reading rules:** compare conditions on **base mission return**, never on shaped return.
Report seeds that never reached the threshold as "did not reach", not as the episode
count. If a difference's confidence intervals overlap, say so — on `safe_corridor` above,
they do, and there is no finding there. State the budget on any single-number claim: "the
curriculum wins" is meaningless, "the curriculum wins on `shaping_trap` at 10,000
episodes" is a result.

---

## STAR interview answer

**Situation.** Autonomous sample return requires balancing scientific value, a finite
energy budget, and uncertain mobility on unfamiliar terrain. A rover that always grabs
the nearest rock is safe and scientifically dull; one that always chases the biosignature
strands itself.

**Task.** Determine how reward design changes learning speed and mission behavior for a
tabular Q-learning rover, and whether a plausible-looking dense reward introduces
behavior nobody asked for.

**Action.** I implemented the Q-learning core from scratch — table initialisation, the
Bellman target with no bootstrapping across terminated *or* truncated boundaries, the TD
error, the in-place update, and ε-greedy selection with random tie-breaking under an
action mask — inside a hand-written MDP with no RL framework. Before running any
comparison I inspected the learned policy rather than the success rate and found the
greedy action oscillating at random over most of the map, which is what all-equal action
values look like: exploration was never discovering the 160-point sample. I proposed and
implemented a curriculum over *physically reachable* start states, keeping evaluation
pinned to the canonical lander start so the numbers stayed comparable with runs trained
without it, and later a closed-loop variant that down-weights start states the table has
already learned. Then I built the analysis to test it honestly rather than by eye: a
paired estimator over arms that share a seed's RNG stream, and a first-crossing budget
that interpolates on a log-episode axis and returns "did not reach" instead of a number.
That ran as a 300-cell grid — three maps, five budgets, two arms, ten seeds, everything
else held constant. Finally I asked which hyper-parameters reach the best reward in the
fewest episodes, pinned it down with a per-trial episode cap and a bounded total budget
so the question had a well-posed objective and a known cost, and wrote the scalarisation
the search maximises as a single normalised area under the learning curve, with the
two-objective Pareto front reported separately so the trade-off the scalar resolves is
still visible.

**Result.** On `shaping_trap` at a 20,000-episode budget the curriculum is worth
**+173.81 base return [+153.86, +193.76]**, paired across all ten seeds; 10 of 10 seeds
reach the 0.8 success threshold with it and 0 of 10 without, so the baseline's budget is
reported as *did not reach* rather than as a number. The effect has a threshold — the
interval first excludes zero at 5,000 episodes, by only `+3.25` — and on `safe_corridor`
every paired interval straddles zero, so I report no effect there. The thing that did not
go as expected came later: a 26.2M-episode hyper-parameter search across all three maps
found that the **no-curriculum** arm wins outright on two of three maps, with the
exploration coming from optimistic initialisation (`initial_q` ≈ 97–189 against a
160-point scale) and a low, fast-decaying ε instead. That runs against most of what the
project had spent its effort on, and it is reported as measured. A second surprise was
diagnostic rather than about performance: at a 150k cap the search on `safe_corridor`
preferred a discount low enough (γ ≈ 0.91) that the 90-point sample is *genuinely*
optimal, and solved that mission perfectly — 500 of 500 greedy deliveries in 13 steps, of
the wrong sample. The objective was right; the cap and the γ range were wrong together.

---

## Talking points to prepare

Questions this project invites. The short answers below come from the repo and the runs;
the long ones should too.

- **Why tabular rather than DQN, and what would break first if the map grew?** The
  question is about learning dynamics under different reward signals, and the MDP is
  small enough to solve exactly, which is what makes "how close did learning get" a
  measurable quantity. What breaks first is the state count, not the memory: `shaping_trap`
  is 104,256 rows on a dense battery axis and 2,304 with affordability binning.
- **What exactly does potential-based shaping guarantee, and what does it *not*
  guarantee?** It preserves the optimal policy; it does not guarantee faster learning,
  and a badly chosen subgoal for `Φ` can still delay discovering a better sample choice.
  Both shaping schemes here aim at one heuristically chosen subgoal.
- **Why are base return and training reward logged separately — what goes wrong without
  it?** Without the split a shaped agent is credited with its own shaping and every
  cross-condition comparison is meaningless. In the single-seed pilot runs the gap is
  visible directly: `naive_dense` on `safe_corridor` evaluates at base `40.0` and shaped
  `81.0`.
- **Why is truncation treated as a non-bootstrapping boundary, and what does it cost?**
  It keeps `calculate_target` a single case distinction and matches `Φ = 0` in every
  episode-ending state, so a finished episode's shaping telescopes exactly to `−Φ(s₀)`.
  It is slightly pessimistic: a truncated rover was not really in an absorbing state.
- **How would you detect a shaping exploit on a map you had not purpose-built for it?**
  `repeated_edge_fraction` and `max_undirected_edge_repeats` are recorded per episode for
  every run, so the loop signature is a logged number rather than something to eyeball —
  though on an unfamiliar map the honest answer is that a threshold on it has to be
  calibrated against a sparse-reward control.
- **Five seeds is few. What would change your mind about a difference you observed?**
  The paired interval crossing zero. Study 2 ran ten seeds paired on the RNG stream for
  exactly this reason, and it is what makes me report no `safe_corridor` effect and a
  large `shaping_trap` one from the same grid.
- **The search says no curriculum wins on two of three maps. Does that not undo the
  curriculum work?** No, and the two results are at different budgets and different
  hyper-parameters: Study 2 says the curriculum rescues `shaping_trap` at 20k episodes
  under untuned settings, Study 4 says that at 150k episodes with tuned optimistic
  initialisation you do not need it. Both are measured; neither is a claim about the
  other's regime.
