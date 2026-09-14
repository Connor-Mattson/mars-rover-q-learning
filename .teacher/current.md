# Current Assignment

**None.** `sample_visit_weighted_start_state` passed review on 2026-09-10; the record is
in `.teacher/history.md`.

## Where the repository stands

All six human-owned functions are implemented and every teaching test is green:

| Assignment | Functions | Date |
|---|---|---|
| Tabular Q-learning core | `select_action`, `q_update`, `epsilon_at`, `learning_rate_at`, `evaluate_greedy_action` (`agent.py`) | 2026-09-08 |
| Annealed start-state curriculum | `enumerate_start_states`, `sample_start_state` (`curriculum.py`) | 2026-09-09 |
| Budget-sweep analysis | `paired_difference` (`metrics.py`), `budget_to_reach` (`sweep.py`) | 2026-09-09 |
| Visit-weighted curriculum | `sample_visit_weighted_start_state` (`curriculum.py`) | 2026-09-10 |

`pytest` is 370 passed; `pytest -m human_todo tests/human_todo` is 115 passed.
`agent.teaching_stub_status()` and `curriculum.curriculum_stub_status()` both return
empty, so `train` prints no teaching banner.

## What is unblocked by this

**Study 3 — which curriculum schedule** is pre-registered in `docs/experiment-plan.md`
and was waiting on this function: a `visit_weighted` cell trained against the stub is
the control arm wearing a treatment label. It can now be run for real.

```bash
python -m mars_rover_q.cli experiment \
    --config configs/experiments/curriculum_strategy_sweep.json \
    --output artifacts/study3
```

600 cells, 4.68M episodes. Read the pre-registered expectations before looking at the
output, and record what actually happened — including "no arm won", which is a
legitimate result and one of the three registered possibilities.

## Recurring feedback worth carrying forward

The incidental-raise pattern has now cost a revision round in three consecutive
assignments: a guard that covers what its test exercises, with the documented
`Raises:` clause left partly unimplemented and a library exception three frames down
making the test green anyway. The habit that fixes it is to write each guard from the
sentence in the docstring, then to construct an input that misses the exploding line.
