# Claude instructions

Identical in substance to `AGENTS.md`; kept as a separate file so tooling that reads
only one of the two still gets the rules.

This is a **learn-by-doing portfolio repository**, not a product. Read this before
changing anything.

## Non-negotiables

1. **No RL framework dependency.** No Gymnasium, Stable-Baselines, RLlib, CleanRL,
   TorchRL, or similar. The Gymnasium-*like* API in `environment.py` is hand-written on
   purpose so the learning loop stays visible.
2. **No pixel observations and no neural networks.** Tabular Q-learning only. PyTorch,
   DQN, policy gradients, and actor-critic methods are out of scope. The Pygame
   renderer is portfolio evidence; nothing in the learning path may import it.
3. **Fixed maps, fresh tables.** Scenarios are hand-authored JSON and are never
   procedurally regenerated between episodes. Every `(scenario, reward mode, seed)`
   cell trains its own Q-table. Never claim a table generalizes to an unseen map.
4. **Base and shaped rewards stay separate.** `base_return` (no shaping) and
   `shaped_return` (what the agent optimized) are distinct fields in `info`,
   `EpisodeRecord`, the CSVs, and every plot. Never substitute one for the other, and
   never mix shaping reward into a base-mission-return comparison.
5. **Injected RNGs only.** Every random draw comes from a `numpy.random.Generator`
   passed in. No `numpy.random` module-level calls, no hidden global state.
6. **Do not implement or bypass active `TODO(human):` functions.** Every function
   carrying a `TODO(human):` marker is Connor's -- the open ones are named in
   `.teacher/current.md`, and the finished ones are recorded in `.teacher/history.md`.
   Do not implement them, do not reimplement them in the trainer, the study, tests, or
   helpers, and do not paste working versions or line-by-line pseudocode into comments,
   docs, examples, or commit messages. If asked to "make the tests pass", say no and
   point at `.teacher/current.md`.
7. **No fabricated experimental claims.** Every number in the README, docs, or a
   portfolio write-up must come from a real generated run. Do not predict which reward
   scheme will win. The Results section stays a labeled placeholder until an experiment
   has actually been run.
8. **Preserve the teaching tests and the `.teacher/` workflow.** Do not skip, weaken,
   `xfail`, or delete anything under `tests/human_todo/`. Do not append to
   `.teacher/history.md` until Connor's implementation passes review.
9. **Keep generated artifacts out of git.** `artifacts/` is gitignored except for a
   small curated result set deliberately added *after* a real experiment.

## Layout

`src/mars_rover_q/` holds the package (see the README's architecture section);
`configs/scenarios/`, `configs/experiments/` and `configs/tuning/` hold JSON inputs;
`tests/{unit,integration,human_todo}/` hold the suites; `docs/` holds the plan,
decisions, and the portfolio template; `.teacher/` holds the teaching state.

## Quality gates

```bash
ruff format --check . && ruff check . && mypy && pytest
```

All four must pass. `pytest` deselects `human_todo` by default; that suite is run
separately with `pytest -m human_todo tests/human_todo` and is **expected to fail**
while an assignment is open.

For the inner loop, `pytest -m "not slow and not human_todo"` is ~12s against the full
gate's ~56s; it drops only the two tests that render a matplotlib frame per battery
level. Keep both clauses — a command-line `-m` replaces `addopts` instead of combining
with it, so `-m "not slow"` alone silently re-enables the teaching suite. Run the
unfiltered `pytest` before calling anything done.

## The "check my work" workflow

When Connor says `check my work`: read `.teacher/current.md`, inspect his diff without
rewriting it, run the focused tests, then the broad gates, and give a strict
`Pass` / `Revise` / `Needs rework` verdict. Only after a pass may you remove the
obsolete `TODO(human):` markers, rerun the checks, append the completion record to
`.teacher/history.md`, and clear the current assignment. If you supplied or
substantially repaired a human-owned function, it is **not** recorded as independently
completed by Connor.

## Recording decisions

Small implementation choices not fixed by the project brief go in
`docs/implementation-decisions.md` with a one-line rationale. Do not stop to ask about
cosmetic decisions.
