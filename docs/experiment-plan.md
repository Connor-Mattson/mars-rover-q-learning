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
