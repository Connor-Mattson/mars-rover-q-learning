# Portfolio evidence

A collection template. **Every slot below is empty on purpose.** Fill each one only
from a verified artifact — a file on disk, a number in a summary CSV, a command that was
actually run. Do not write a claim here that a generated file cannot back up, and do not
invent collaborators, links, or results. Use first person only for work Connor actually
did.

---

## Problem

<!-- 3-5 sentences. The mission, the constraint, and why reward design is the
interesting variable. Written from the README's research question; no results yet. -->

_TODO_

## Approach

<!-- The MDP in a paragraph, the three reward conditions, and why tabular. State the
controlled comparison: identical agent, identical maps, identical seeds, reward is the
only thing that changes. -->

_TODO_

## My contribution

<!-- First person, and only for work Connor actually did. The five Q-learning functions
in agent.py, plus whatever analysis, tuning, and interpretation he performs. Be explicit
about what was scaffolded versus written, because an interviewer will ask. -->

_TODO_

## Results

> **Blocked until the experiment has been run.** Leave this section empty rather than
> approximate.

_TODO_

---

## Evidence checklist

| Item | Where it comes from | Status |
|---|---|---|
| Hero screenshot (mission control, mid-episode) | `play` or `replay`, window capture | ☐ |
| Short replay video or GIF | `replay --run <dir> --episode best --policy-overlay` | ☐ |
| Learning-curve figure | `artifacts/<experiment>/plots/learning_curves.png` | ☐ |
| Reward-comparison figure | `artifacts/<experiment>/plots/reward_comparison.png` | ☐ |
| Failure-mode figure | `artifacts/<experiment>/plots/failure_modes.png` | ☐ |
| Shaping-loop evidence | `eval_repeated_edge_fraction` in `summary_aggregated.csv` | ☐ |
| Exact commands run | shell history; paste verbatim | ☐ |
| Commit hash | `manifest.json → environment.git_commit` | ☐ |
| Hardware and runtime | `manifest.json → environment.platform`, measured wall clock | ☐ |
| Package versions | `manifest.json → environment` | ☐ |
| Seeds used | `summary.json → config.seeds` | ☐ |

## Quantitative results table

Fill from `artifacts/<experiment>/summary_aggregated.csv`. Report mean ± 95% CI across
**all** seeds. Never report a single seed.

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

**Reading rules:** compare conditions on **base mission return**, never on shaped
return. Report seeds that never reached the threshold as "did not reach", not as the
episode count. If a difference's confidence intervals overlap, say so.

---

## STAR interview answer

**Situation.** Autonomous sample return requires balancing scientific value, a finite
energy budget, and uncertain mobility on unfamiliar terrain. A rover that always grabs
the nearest rock is safe and scientifically dull; one that always chases the biosignature
strands itself.

**Task.** Determine how reward design changes learning speed and mission behavior for a
tabular Q-learning rover, and whether a plausible-looking dense reward introduces
behavior nobody asked for.

**Action.** _Placeholder — fill in after implementing the Q-learning functions and
running the experiment. Cover: what was implemented, what was held constant, how many
seeds, and how the shaping-loop behavior was measured rather than eyeballed._

**Result.** _Placeholder — fill in only from verified metrics. Include at least one
number with its confidence interval, and one thing that did not go as expected._

---

## Talking points to prepare

Questions this project invites. Answers come from the repo and the runs, not from
memory:

- Why tabular rather than DQN, and what would break first if the map grew?
- What exactly does potential-based shaping guarantee, and what does it *not* guarantee?
- Why are base return and training reward logged separately — what goes wrong without it?
- Why is truncation treated as a non-bootstrapping boundary, and what does that cost?
- How would you detect a shaping exploit on a map you had not purpose-built for it?
- Five seeds is few. What would change your mind about a difference you observed?
