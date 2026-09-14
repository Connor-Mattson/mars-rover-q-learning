"""The project's floating-point error policy.

Silent `nan` is the worst failure mode a Q-table has: it propagates through every
subsequent update, survives into the artefacts, and shows up as a blank plot hours
later. So the entry points make numpy raise instead, and a bad division is a
traceback at the line that caused it.
"""

from __future__ import annotations

import numpy as np

#: Underflow is excluded from the guard on purpose -- see :func:`install_numeric_guard`.
NUMERIC_GUARD = {"divide": "raise", "over": "raise", "invalid": "raise", "under": "ignore"}


def install_numeric_guard() -> None:
    """Make numpy raise on the floating-point conditions that indicate a defect.

    Called from the CLI entry point and from each experiment worker, because
    :func:`numpy.seterr` is per-thread and a process pool's children do not inherit it.

    ``divide``, ``over`` and ``invalid`` are all raised: in this codebase each one means
    a real bug, and a `nan` that reaches a Q-table is unrecoverable.

    **Underflow is deliberately ignored**, which is also numpy's own default. It is not
    an error but an ordinary consequence of the log-sum-exp identity, which deliberately
    evaluates ``exp`` of large negative numbers and relies on them flushing to zero --
    the term it is discarding is the one too small to matter. Optuna's TPE sampler does
    exactly this when it scores candidate points, so raising on underflow crashed a
    search the moment the sampler had enough history to build a model from.
    """
    np.seterr(**NUMERIC_GUARD)  # type: ignore[arg-type]
