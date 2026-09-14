# Experiment plan

> Status: **pre-registration**. Nothing here reports a result. It records what will be
> run and how it will be read *before* any data exists, so the eventual write-up cannot
> be quietly reshaped to fit whatever came out.

## Question

How does reward design affect the sample efficiency, final mission performance, and
unintended behavior of a tabular Q-learning rover choosing among scientifically
valuable but differently risky sample-return missions?

## Design

A fully crossed grid, run by `python -m mars_rover_q.cli experiment`:

| Factor | Levels |
|---|---|
| Scenario | `safe_corridor`, `risk_value_tradeoff`, `shaping_trap` |
| Reward mode | `sparse`, `naive_dense`, `potential` |
| Seed | `1..5` (10 supported via `reward_comparison_10seeds.json`) |

45 cells at 5 seeds. Each cell trains a **fresh** Q-table for a fixed 4000 episodes,
then evaluates for 300 episodes with `epsilon = 0` on a separate evaluation seed
(`seed + 1_000_000`).

Held constant across every cell: learning rate `0.2`, `gamma 0.99`, initial Q `0.0`,
epsilon linear `1.0 → 0.05` over the first 60% of episodes, sample values 40 / 90 / 160,
failure penalties `-100`. Only the reward mode and the map change.

## Measurements

Per evaluation batch, aggregated across seeds as mean with a 95% Student-t interval:

| Metric | Field | Reads as |
|---|---|---|
| Sample efficiency | `episodes_to_threshold`, `env_steps_to_threshold` | First episode whose trailing-100 success rate reaches 0.8. `None` when never reached — reported as "did not reach", never coerced to the episode count. |
| Mission performance | `eval_success_rate` | Fraction of greedy episodes ending in `success`. |
| Science yield | `eval_mean_delivered_value` | Which sample the policy actually chose to return. |
| **Honest comparison** | `eval_mean_base_return` | Mission return with **shaping excluded**. This is the cross-condition comparison. |
| Margin | `eval_mean_energy_remaining_on_success` | Energy left on delivery: how close to the edge the policy runs. |
| Efficiency | `eval_mean_steps` | Mission length. |
| Failure modes | `failure_battery_depleted`, `failure_step_limit` | How missions fail, not just how often. |
| Unintended behavior | `eval_repeated_edge_fraction`, `eval_max_undirected_edge_repeats` | Share of moves re-traversing an already-used map edge, and the most-repeated edge. A back-and-forth shaping loop drives both up; a direct route keeps them near zero. |

`eval_mean_shaped_return` is logged too, but only ever compared *within* a reward mode.
Comparing shaped returns across modes is meaningless, because the modes do not share a
reward scale.

## What each scenario is for

- **`safe_corridor`** — the control. Flat, walled corridors, short routes, a modest
  battery. Every sample is comfortably affordable, so this isolates learning speed from
  risk.
- **`risk_value_tradeoff`** — the biosignature (160) sits in a sand basin ringed by
  rough terrain, where slip is likely and energy costs 3× flat. The basalt core (40) is
  two steps from the lander. The best *expected* choice is not visually obvious, which
  is the point.
- **`shaping_trap`** — a wide flat plaza north of a rough/rock barrier, with a 180-unit
  battery and `closer_bonus = 6` / `farther_penalty = -2`. Any two adjacent plaza cells
  form a cycle worth `+4` undiscounted per round trip under the naive scheme, and the
  battery affords roughly 90 such cycles before depletion. Whether a Q-learner actually
  finds and prefers that loop is an empirical question this experiment answers; the map
  only makes the loop *available* and observable.

## Pre-registered expectations

Stated as hypotheses to be tested, **not** as results:

1. Dense shaping of either kind is expected to reach the success threshold in fewer
   episodes than the sparse condition, most visibly on the longer routes.
2. Under `naive_dense`, `shaping_trap` is expected to show elevated repeated-edge
   statistics and a *worse* base mission return than its shaped return would suggest.
   If it does not, that is a real and reportable negative finding.
3. `potential` shaping cannot change the optimal policy of the base MDP, so its base
   mission return should be no worse than sparse at convergence. It may still learn
   more slowly than `naive_dense` early on, and the fixed subgoal heuristic may bias it
   toward one sample.

Any of these can turn out false. The write-up reports what the runs show.

## Threats to validity

- **One subgoal heuristic.** Both shaping schemes aim at the highest-value sample whose
  round trip fits in 80% of the battery. On all three bundled maps that resolves to the
  biosignature, so shaping never has to pick between samples — a limitation, not a
  finding.
- **Truncation is treated as a hard boundary.** Q-learning does not bootstrap across the
  step limit, which slightly under-values states near it, identically in every
  condition.
- **Fixed hyper-parameters.** No tuning was done per condition. A shaped agent might
  prefer a different learning rate; this design deliberately holds everything but the
  reward constant, at the cost of possibly under-serving some condition.
- **Five seeds.** Small. Intervals use the Student-t critical value rather than a normal
  approximation, and the 10-seed config exists for when a difference looks marginal.

## Runtime

The full 45-cell grid took **3 minutes 49 seconds** on one CPU core of the development
laptop and wrote about 139 MB of artifacts. That measurement was taken on the placeholder
learning path, so it is a lower bound — see the caveat in
`docs/implementation-decisions.md`, and re-measure before quoting it. The smoke config
(`configs/experiments/smoke.json`) is sized for CI and finishes in about a second.

## Reporting rules

- Aggregate every declared seed. No cherry-picking, no "representative run".
- Report `did not reach threshold` explicitly rather than dropping those seeds.
- Never plot shaping reward on the same axes as base mission return.
- Record the git commit and package versions from the run manifest alongside any number
  that appears in the portfolio.

---

# Study 2 — the start-state curriculum across training budgets

> Status: **pre-registration**. Written before the sweep was run. No number below is
> a result; the ones that look like results are costs and capacities measured from
> pilot runs, and they are labelled as such.

## Question

Study 1 asks which reward design works. This one asks a different question about the
same three maps: **how much of the training budget does the start-state curriculum
buy back?**

The curriculum was added because Connor diagnosed an exploration failure — under a
decaying epsilon from the lander, the 160-point biosignature was never once delivered
in 4000 episodes on `risk_value_tradeoff` (see `.teacher/history.md`). A single
matched pair of runs showed the curriculum finding that reward. A single pair is an
anecdote. This study asks whether the effect is real across seeds, whether it holds on
all three maps, and — the part a single pair cannot answer at all — whether it is a
*speedup* that the baseline eventually catches up to, or a difference that persists.

## Design

Run by `python -m mars_rover_q.cli experiment --config
configs/experiments/curriculum_budget_sweep.json`.

| Factor | Levels |
|---|---|
| Scenario | `safe_corridor`, `risk_value_tradeoff`, `shaping_trap` |
| Curriculum | `curriculum_fraction = 0.0` (control) vs `0.5` (treatment) |
| Episode budget | 1000, 3000, 5000, 10000, 20000 |
| Seed | `1..10` |

300 cells. Reward mode is held at `sparse` throughout: the curriculum targets the
sparse-reward exploration failure, and crossing it with the three reward schemes would
triple the grid to answer a different question. That crossing is a follow-up, not this
study.

**Every point on the budget axis is its own complete training run.** A 1000-episode
cell trains a fresh table for exactly 1000 episodes, and *both* schedules scale to
that budget — epsilon anneals `1.0 → 0.05` over its first 60%, and the curriculum
anneals to the canonical start over its first 50%. This is the whole reason the sweep
is not implemented by checkpointing one 20000-episode run: a snapshot taken at
episode 1000 of a 20000-episode schedule is an agent still at `epsilon ≈ 0.7` with a
curriculum still 90% on easy starts, which is not what "a 1000-episode budget" means.
The cost of doing it properly is 39000 training episodes per
(scenario, curriculum, seed) cell instead of 20000.

Held constant: learning rate `0.2`, `gamma 0.99`, initial Q `0.0`, and the same
hand-authored maps as Study 1. Only the curriculum and the budget move.

## Measurements

### Learning curves are measured, not inferred

Every run is paused every `budget / 50` episodes and scored over 40 greedy episodes
from the canonical lander start. This is what makes a learning curve comparable across
the two conditions at all: a curriculum run's own training episodes begin mid-mission,
so their outcomes are not a measure of progress on the mission being evaluated, and the
subset that happens to start at the lander is thin, high-epsilon, and — if indexed by
its position within that subset rather than by true episode number — compressed toward
the origin in a way that makes the curriculum look instantaneous. Checkpoint *k* uses
the same evaluation seed in both arms, so the curves are paired at every point on the
x-axis.

The y-axis is **`eval_mean_base_return`** — mission return with shaping excluded —
from 300 greedy episodes started at the canonical lander position on evaluation seed
`seed + 1_000_000`. Two things make that comparison fair:

- Evaluation never uses the curriculum. Both arms are scored on the same mission from
  the same start, so a curriculum run gets no credit for its easier training episodes.
- The evaluation budget does **not** scale with the training budget. A 1000-episode
  cell and a 20000-episode cell are measured with the same 300-episode ruler.

Three readings, in increasing order of what they commit to:

| Reading | Where | What it answers |
|---|---|---|
| Budget curve | `plots/budget_curves.png` | Mean base return vs budget per condition, with a 95% CI band across the 10 seeds. The picture. |
| Paired difference | `plots/paired_difference.png`, `sweep_analysis.json` | The within-seed `curriculum − baseline` difference at each budget, with a paired 95% interval. The inference. |
| Budget to target | `sweep_analysis.json` | The budget each arm needs to reach a shared target, and the ratio between them. The sample-efficiency claim. |

**Why paired.** `split_rngs` is deterministic, so seed *s* hands both arms the same
environment, exploration, and start-state streams. The arms are therefore not
independent samples, and their separate CI bands can overlap while a consistent
within-seed effect is present. The paired interval is the one that decides;
overlapping bands in the figure do not refute it, and this is stated on the figure so
a reader does not misread the bands.

The secondary metrics from Study 1 — `eval_success_rate`, `eval_mean_delivered_value`,
`episodes_to_threshold`, `train_tied_state_fraction` — are recorded per cell in
`summary_per_seed.csv` and can be re-analysed at any budget with
`cli analyze --metric`. `train_tied_state_fraction` is the coverage diagnostic that
motivated the curriculum in the first place, and it is worth reading alongside return.

## Pre-registered expectations

Hypotheses, **not** results. Any of them can turn out false, and a false one is
reportable.

1. The curriculum's advantage is largest at the smallest budgets and narrows as the
   budget grows. This is what "a sample-efficiency intervention" means, and if the gap
   is instead flat or growing, the curriculum is doing something other than
   accelerating discovery.
2. The effect is largest on `risk_value_tradeoff`, where the high-value sample is the
   one the baseline provably never reached, and smallest on `safe_corridor`, whose
   short routes the baseline already solves.
3. The two arms converge by 20000 episodes on at least `safe_corridor`. If they do
   not, the honest claim is stronger than "faster" — but it must then be checked
   against a larger budget before it is made, because "the baseline never catches up"
   and "the baseline had not caught up by 20000" are different statements.
4. `train_tied_state_fraction` is lower under the curriculum at every budget.

## Threats to validity

- **One curriculum fraction.** `0.5` is carried over from the single matched pair that
  motivated the study; it was not tuned, and no other anneal length was tried. A
  better fraction may exist, and the sweep would not find it.
- **One reward mode.** Everything here is `sparse`. Whether the curriculum and reward
  shaping are substitutes or complements is not addressed.
- **The open thread is still open.** As of the last run, the curriculum made the
  biosignature *discoverable* during training without changing which sample the greedy
  policy returns from the lander. If that is still true, base return may not move even
  where the coverage diagnostic does — which is exactly why both are recorded.
- **Ten seeds.** Larger than Study 1's five, and paired, which is where the real
  power comes from. Still small; intervals use the Student-t critical value.
- **The target is derived from the data.** `budget_to_reach` measures against 90% of
  the best mean observed anywhere on that scenario's curves. That keeps one number
  meaningful across maps whose achievable returns differ by 4x, and it is symmetric
  across the arms, but it is not an absolute standard and moves if either arm improves.

## Cost

Measured on the development laptop (Apple Silicon, 10 performance cores, one core
unless `--jobs` is given), on the real learning path:

**Measured, on the full 300-cell grid** (2.34M training episodes, 90k final-evaluation
episodes, 51 mid-training checkpoints per cell at 40 greedy episodes each):
**7 minutes 3 seconds** wall clock at `--jobs 8` (3201 s of CPU across 8 workers), and
**364 MB** of artifacts under `save_q_tables: "max_budget"` — close to the 350 MB
estimated in advance. `save_q_tables: "none"` removes about 150 MB of that.

The checkpoint evaluation adds roughly 14% to a run and is free of side effects: a run
with checkpoints produces a bit-identical Q-table and identical episode records to the
same run without them, which `test_checkpointing_does_not_perturb_training` asserts in
both arms.

## Reporting rules

Study 1's rules carry over unchanged, plus:

- Report the paired interval, not just the two curves. A gap between curves whose
  paired interval straddles zero is not a finding.
- Report `did not reach` from `budget_to_reach` explicitly. It is a real answer.
- Never compare a curriculum cell's *training* success rate with a baseline cell's.
  Only canonical-start episodes are comparable, which is what
  `canonical_start_records` filters and what the learning-curve figure plots.
- State the budget on any single-number claim. "The curriculum wins" is meaningless
  without it; "the curriculum wins at 3000 episodes" is a result.

---

# Study 3 — which curriculum schedule

> Status: **pre-registration**. Written before the grid was run. No number below is a
> result. The dry-run figures quoted under *Motivation* are properties of the sampler
> alone, produced by `cli curriculum` without training anything, and are labelled as
> such.

## Question

Study 2 asks whether the start-state curriculum is worth its episodes. This one asks a
narrower question that Study 2's design cannot answer: **the curriculum widens its
support as it anneals, but keeps drawing uniformly from everything it has admitted —
does fixing that oversampling buy anything?**

## Motivation, and where it came from

Connor raised it from two artefacts of the Study 2 machinery: the per-run Q-value
heatmaps, and the `curriculum` command's own output. At three-quarters of the way
through the anneal the reported minimum sampled difficulty was still `1.0` — the
easiest state in the pool, one step from a finished mission, still being drawn as often
as the hard states the anneal had just admitted. A growing window with a uniform draw
never retires anything.

Dry-running the schedule quantifies it. On `risk_value_tradeoff` at
`curriculum_fraction = 0.5`, over the anneal's 2000 episodes:

| Strategy | Episodes in the easiest quarter of the pool | Min difficulty, final fifth |
|---|---|---|
| `growing` | 0.60 | 1.0 |
| `sliding` | 0.17 | 11.0 |

Those are sampler properties, not learning results: no agent was trained to produce
them, and they say nothing about whether spending the episodes elsewhere helps.

## Design

Run by `python -m mars_rover_q.cli experiment --config
configs/experiments/curriculum_strategy_sweep.json`.

| Factor | Levels |
|---|---|
| Scenario | `safe_corridor`, `risk_value_tradeoff`, `shaping_trap` |
| Arm | control (`curriculum_fraction = 0.0`), then `growing`, `sliding`, `visit_weighted` at `0.5` |
| Episode budget | 1000, 3000, 5000, 10000, 20000 |
| Seed | `1..10` |

600 cells, 4.68M training episodes. Held constant: `window_fraction = 0.25`,
`weight_exponent = 1.0`, and everything Study 2 held constant. Reward mode stays
`sparse` for Study 2's reason.

**The control arm is run once, not once per strategy.** A disabled curriculum starts
every episode at the lander whatever the sampler says, so the three would be identical
tables under three labels — and splitting the control three ways would cost each paired
comparison two-thirds of its seeds. Each treatment arm is paired against that one
control, seed by seed.

## Measurements

Study 2's measurements carry over unchanged: mid-training checkpoints from the canonical
start, the paired within-seed difference in `eval_mean_delivered_value`, and
`budget_to_reach` on the shared target. Three additions specific to this question:

- `train_tied_state_fraction`, already recorded, is the coverage half of the claim: a
  schedule that spends its episodes better should leave fewer states undecided at the
  same budget.
- The per-run coverage figures, split by payload, are where the biosignature slice is
  visible — the slice that had exactly one learned state before any curriculum existed.
- The dry-run band shares above are the mechanism, reported alongside the outcome so
  that "it spent its episodes differently" and "it learned more" stay two separate
  claims.

## Pre-registered expectations

Written before the grid ran, and recorded so that a wrong one stays on the record:

1. `sliding` and `visit_weighted` both reduce the share of episodes spent on the easiest
   quarter. This is already established above — it is a property of the sampler, not a
   prediction about learning.
2. Whether that converts into base mission return is genuinely open. The mechanism cuts
   both ways: retiring easy states frees episodes for the frontier, but the easy states
   are also where the terminal reward enters the table, and their values are refreshed
   by revisits under a stochastic policy.
3. If any arm wins, the effect should be largest at the *small* budgets, where episodes
   are scarce enough for their allocation to matter, and should shrink as the budget
   grows. An effect that only appears at 20000 episodes would need a different
   explanation.

No arm is predicted to win. The two new strategies were added because the growing
window's sampling defect is real and measurable, not because there is evidence yet that
fixing it helps.

## Threats to validity

Study 2's threats carry over, plus:

- **`visit_weighted` couples the schedule to the agent's own trajectory distribution.**
  Its counts come from wherever the agent actually went, so a change in exploration
  changes the start distribution as a side effect. The arms are still paired by seed,
  but this one is not an open-loop schedule and should not be described as one.
- **The tilt's strength is one fixed exponent.** `weight_exponent = 1.0` is a choice,
  not a tuned value, and a null result for this arm is a null result *for that
  exponent*. Sweeping it is a follow-up, not this study.
- **`sliding` can retire a state before its value has settled.** Nothing re-visits a
  band once it has passed, so a value that was still moving when the window left is
  corrected only by trajectories that happen to pass through. Coverage at the end of
  training is the measurement that would show it.

## Cost

Not yet measured. By proportion to Study 2's measured 7m03s at `--jobs 8` for 2.34M
training episodes, 4.68M episodes projects to roughly 14 minutes and about 700 MB of
artifacts under `save_q_tables: "max_budget"`. Re-measure and replace this projection
with the real figure once it has run, exactly as Study 2's cost section was.

## Reporting rules

Study 2's rules carry over, plus:

- Name the arm on every claim. "The curriculum helps" is now three different claims.
- Never quote a `visit_weighted` number from a run whose manifest carries
  `curriculum.pending_human_functions`. That run started every episode at the lander;
  it is the control wearing a treatment label, and `train` prints a banner saying so.


---

# Study 4 — which hyper-parameters learn fastest

> Status: **pre-registration**. Written before any search was run. Every number below is
> a budget, a bound, or a capacity; none is a result.

**Connor's question, and his framing of it.** Studies 1 to 3 hold the hyper-parameters
fixed at values that were never tuned — learning rate `0.2`, `gamma 0.99`, epsilon
`1.0 → 0.05` over 60% of the budget — because each was asking about something else, and a
factor you are not studying has to be held constant. He asked the question those studies
cannot: *which set of hyper-parameters reaches the best reward in the fewest episodes?*
Both halves of the framing are his, including the two constraints that make it answerable
(see the Discovery log entry for 2026-09-10).

## Question

On one map, with the reward mode held fixed, which hyper-parameters reach a high mission
return in the fewest training episodes — and what does the search pay, in total
episodes, to find out?

## Design

Run by `python -m mars_rover_q.cli tune --config configs/tuning/safe_corridor_search.json`.

| Element | Value |
|---|---|
| Scenario | `safe_corridor` only |
| Reward mode | `sparse`, held fixed |
| Seeds per trial | `1, 2`, averaged into one curve |
| Per-trial episode cap | 6000 per seed, so 12000 per trial |
| Total study budget | 2,000,000 episodes |
| Sampler | Optuna TPE, `sampler_seed = 0`, 10 startup trials |
| Pruner | `MedianPruner`, 5 startup trials, 3 warm-up checkpoints |
| Checkpoints | 12 per trial, 30 greedy episodes each |

166 trials if nothing is ever pruned; more in practice, because a pruned trial returns
its unspent episodes to the ledger. The search space:

| Parameter | Range | Scale |
|---|---|---|
| `learning_rate` | 0.01 – 1.0 | log |
| `gamma` | 0.9 – 0.9995 | linear |
| `initial_q` | 0.0 – 200.0 | linear |
| `epsilon_start` | 0.2 – 1.0 | linear |
| `epsilon_end_ratio` | 0.001 – 0.5 of `epsilon_start` | log |
| `epsilon_decay_fraction` | 0.05 – 0.95 | linear |
| `use_curriculum` | `False`, `True` | categorical |
| `curriculum_fraction` | 0.1 – 0.9 | linear, only when enabled |
| `curriculum_strategy` | `growing`, `sliding`, `visit_weighted` | categorical, only when enabled |
| `curriculum_window_fraction` | 0.02 – 0.5 | log, only under `sliding` |
| `curriculum_weight_exponent` | 0.0 – 2.0 | linear, only under `visit_weighted` |

**Three design constraints, and what each one is for.**

1. **The per-trial cap is deliberately below convergence.** A well-tuned agent reaches
   success 1.0 on `safe_corridor` in roughly 4000 episodes at the repository's fixed
   defaults; the cap is 6000, which is enough for a good setting to converge and not
   enough for a poor one to catch up. Some settings cannot reach the map's best return
   inside it, which is the point: without a cap every eventually-converging setting
   scores alike and the search silently becomes a search for final performance.
2. **The total budget is bounded in advance.** 2M episodes, charged against one ledger,
   spent or not spent; the study ends when what remains cannot fund another full trial.
   The cost of a search is therefore known before it starts, and two searches are
   comparable in total compute rather than in wall clock.
3. **Two objectives, scalarised in exactly one place.** "Best reward in the fewest
   episodes" is a trade-off. `score_learning_curve` is the single function that resolves
   it, and `pareto_front` reports the trade-off it resolved, so the scalar's verdict and
   the shape of the frontier are separate artefacts a reader can disagree with
   separately.

## Measurements

Per trial, written to `study.json` and `trials.csv`:

| Quantity | Reads as |
|---|---|
| `score` | The objective. Comparable across trials because every trial shares the episode cap and the return reference. `null` for a pruned trial — a truncated curve is not the same measurement. |
| `best_return` | Best `mean_base_return` anywhere on the trial's curve: mission return with shaping excluded. |
| `episodes_to_best` | The *earliest* checkpoint attaining that best. With `best_return`, the two Pareto axes. |
| `final_return`, `final_success_rate` | Where the curve ended, for separating "learned fast" from "learned and held". |
| `episodes_run` vs `episodes_granted` | What pruning saved. |
| `curve` | The whole checkpoint curve, so any other objective can be computed after the fact from the same data. |

The reference return is `best_affordable_return` — the most valuable sample whose lander
round trip fits the battery, 160 on all three bundled maps. It is a **planning bound from
shortest paths, not a target**: slip lowers the achievable mean, and a discount below 1.0
can make a nearer sample optimal, so a score of 1.0 is unreachable by construction.

## Pre-registered expectations

1. **No prediction is recorded about which hyper-parameters win.** This is the point of
   running a search rather than arguing from intuition, and a guess written here would
   be a thing to rationalise against afterwards.
2. The search is expected to *find something*: that the best trial's score is meaningfully
   above the median trial's. If it is not, the likely causes are the cap being too
   generous (everything converges) or too tight (nothing does), and the fix is the cap,
   not the objective.
3. Pruning is expected to fund more trials than `fundable_trials`. That is arithmetic
   about the ledger, not a claim about learning.
4. Whether the winning setting uses a curriculum at all is open, and a search that
   selects `use_curriculum = False` is a legitimate result — it would say the curriculum
   does not pay for itself *at this cap, on this map*, which is a narrower claim than
   Study 3's and must not be reported as contradicting it.

## Threats to validity

- **One map, one reward mode.** A winner here is tuned to `safe_corridor` under `sparse`
  and nothing else. Non-negotiable #3 applies with full force: these hyper-parameters
  are not claimed to transfer to `risk_value_tradeoff`, and the honest follow-up is to
  re-run the search per map, not to reuse the winner.
- **The cap is part of the objective.** "Fastest learner" is defined relative to 6000
  episodes. A different cap can reorder the trials, and a reported winner must name the
  cap it won under.
- **Two seeds is a thin average.** It is what the budget buys at this cap. A difference
  between two adjacent trials' scores is well inside seed noise; only the shape of the
  search — where the good region is — should be read from a single study.
- **The best trial is selected on the same data that measured it.** That is what a
  search does, and it makes the winning score optimistically biased. The confirmation
  run exists to re-measure the winner on a full `eval_episodes` evaluation, and that
  number, not the trial's score, is the one to quote.
- **The score hides its trade-off by construction.** Hence the Pareto front, which is
  reported alongside it and not instead of it.
- **Optuna is in the learning path only as a proposer.** It never sees the environment,
  the agent, or a reward; it suggests numbers and reads one score back. Non-negotiable
  #1 is intact — there is still no RL framework here.

## Cost

Not yet measured. The bound is exact by construction: 2M training episodes plus one
confirmation run, whatever the wall clock turns out to be. Pilot timing on a 30k-episode
study measured 42 trials funded against 25 nominal, with pruning returning roughly 60% of
each cut trial's budget; re-measure and record the real figure when the full study runs.

## Reporting rules

- Quote the confirmation run's `eval_mean_base_return`, not the winning trial's score,
  as the performance of the tuned setting.
- Name the cap and the map with every claim about learning speed.
- Report the Pareto front whenever a single winner is named, so that a reader can see
  what the scalar traded away.
- Never quote a number from a study whose `search_is_meaningful` is `false`. That study
  scored every trial identically, so its "best trial" is whichever one was asked first,
  and `tune` prints a banner saying so.
- A study's `budget.spent` belongs in the write-up next to its result. The point of the
  ledger is that the cost of the answer is part of the answer.
