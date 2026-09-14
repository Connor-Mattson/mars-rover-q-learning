# Current Assignment

**None.** Every `TODO(human):` in the repository is closed.

The hyper-parameter search objective — `score_learning_curve` and `pareto_front` in
`src/mars_rover_q/tuning.py` — passed review on 2026-09-12 and is recorded in
`.teacher/history.md`. With it, the study ranks: `configs/tuning/smoke.json` produces
distinct scores, a real front, and no **HYPERPARAMETER SEARCH STUBBED** banner, and
`study.json` carries `search_is_meaningful: true` honestly.

## What this unblocks

Study 4 can now be run for real, and it is the first study whose numbers were not
reportable before today:

```bash
python -m mars_rover_q.cli tune --config configs/tuning/safe_corridor_search.json
```

2M episodes, 6000 per trial across two seeds, so 166 trials if nothing is pruned and
more in practice. It is pre-registered in `docs/experiment-plan.md` — including the
deliberate absence of any prediction about which parameters win — and the Results
section stays a labelled placeholder until the run exists.

Worth knowing before starting it: the smoke config's 400-episode cap is far below what
`safe_corridor` needs, so its scores cluster low and several trials legitimately sit at
`0.0`. That is the cap doing its job, not the objective misbehaving. The real cap of
6000 sits just above convergence for a tuned agent and just below it for most of the
search space, which is where the objective has the most to say.

## If you want another assignment

Say what you would like to build. Candidates the repository has left open, in rough
order of how much they would teach:

- **A second map for Study 4.** `risk_value_tradeoff` measures speed *and* risk
  appetite at once; separating them needs a decision about what the reference return
  should be when the best affordable sample is not the best expected one.
- **A multi-objective study.** Optuna supports two objectives directly, which would
  retire the scalarisation rather than report around it — and would make
  `pareto_front` the primary output instead of a footnote.
- **Seed robustness in the objective.** The score currently averages the curves and
  then scores once; scoring per seed and reporting the spread would say which winners
  are real.
