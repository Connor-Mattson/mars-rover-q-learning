# Current assignment: the tabular Q-learning core

Everything around the algorithm is built and green. What is missing is the algorithm.

**File:** `src/mars_rover_q/agent.py`
**Marker:** every unfinished body carries `TODO(human):`
**Scope:** the five functions below, and nothing else. Do not change their signatures,
do not move them to another module, and do not touch the trainer — `training.py` and
`evaluation.py` already call exactly these functions, so implementing them is the whole
job.

Before you start, if you have not already: play the environment for a few minutes.

```bash
python -m mars_rover_q.cli play --scenario safe_corridor
```

You will feel the slip, the energy drain, and the pull of the biosignature long before
you feel it in a reward curve.

---

## 1. `initialize_q_table(num_states, num_actions=5, *, initial_value=0.0, dtype=np.float64)`

**Returns** a fresh `(num_states, num_actions)` array where **every entry equals
`initial_value`**, using `dtype`.

- `initial_value` is not decoration. Optimistic initialisation — starting every entry
  above any achievable return — is a real exploration device, and the tests use nonzero
  and negative values.
- Reject `num_states <= 0` or `num_actions <= 0` with `ValueError`. The current
  placeholder silently clamps them, which is worse than crashing.
- Each call must return an independent array. Two tables must not share memory.

**Edge cases:** zero dimensions, negative dimensions, `dtype=np.float32`, negative
initial values.

## 2. `calculate_target(reward, next_state_values, gamma, terminated, truncated)`

**Returns** the scalar the current estimate should be moved toward.

- Q-learning is **off-policy**: the bootstrap uses the *best* action available in the
  successor state, not the action the behaviour policy will take next. The test with a
  single good action buried at index 2 exists precisely to catch "take the first one".
- **Neither** `terminated` **nor** `truncated` bootstraps. In both cases the target must
  not depend on `next_state_values` at all. This is a stated convention of the project
  (see the README); it makes truncated episodes slightly pessimistic and keeps your
  implementation to one case distinction.
- `gamma` scales the bootstrapped part only.

**Edge cases:** terminated, truncated, both false, `gamma` at 0.1 and 0.99.

## 3. `calculate_td_error(current_estimate, target)`

**Returns** the signed discrepancy, oriented so that a **positive** result means the
outcome was better than the table currently believes and the entry should move **up**.

Sign is the entire content of this function. Get it backwards and the agent learns to
avoid reward, confidently and quietly.

**Edge cases:** target above the estimate (positive), equal (exactly zero), below
(negative).

## 4. `update_q_value(q_table, state_index, action_index, learning_rate, td_error)`

**Mutates** `q_table` in place. Returns `None`.

- Move the selected entry along the TD error, scaled by `learning_rate`. With
  `learning_rate = 1.0` the entry must land exactly on the target.
- **Exactly one entry changes.** Every other entry — including the other four actions of
  the same state — must be bit-for-bit unchanged. A test diffs the whole flattened table
  and asserts a single changed index.
- Repeated calls accumulate; there is no reset.

**Edge cases:** learning rate 1.0, two different learning rates on identical input,
repeated calls on the same entry.

## 5. `select_action(q_table, state_index, epsilon, rng, action_mask=None)`

**Returns** an action index, chosen epsilon-greedily.

- With `epsilon = 0`, purely greedy. With `epsilon = 1`, uniform over the legal actions.
- **Every random draw comes from `rng`.** Never `np.random.something(...)` — the whole
  reproducibility protocol rests on this. Two generators built from the same seed must
  produce the same sequence of actions.
- **Break greedy ties randomly.** A freshly initialised table is *all* ties, so
  "return the first argmax" means the rover drives north into a wall until epsilon
  decays. The test asserts that a table of equal values eventually produces all five
  actions, and that a two-way tie produces exactly those two.
- When `action_mask` is given, restrict **both** exploration and the greedy choice to
  the `True` entries. Raise `ValueError` if nothing is legal.

**Edge cases:** epsilon 0, epsilon 1, intermediate epsilon under a fixed seed, full tie,
partial tie, a mask with one or two legal actions, an all-`False` mask.

---

## Definition of done

1. `pytest -m human_todo tests/human_todo` — all pass, with **no changes** to the test
   file. Adjusting a test to fit an implementation is not a pass.
2. `pytest` — still fully green.
3. `ruff format --check . && ruff check . && mypy` — clean.
4. `python -m mars_rover_q.cli train --scenario safe_corridor --reward sparse --seed 1
   --episodes 4000` — no longer prints the `TEACHING STATE` banner, and the resulting
   `manifest.json` records `"learning_is_meaningful": true`.
5. Every random draw in your code comes from the injected `rng`.
6. No implementation of these five algorithms exists anywhere else in the repository.

Stuck? `.teacher/hints.md` escalates in stages — read one at a time. Test commands and
what to expect before and after are in `.teacher/how-to-test.md`.

**When finished, return to Claude and say: check my work**
