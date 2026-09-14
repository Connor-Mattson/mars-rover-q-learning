"""The floating-point guard: what it catches, and the one thing it must not catch."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from mars_rover_q.numerics import install_numeric_guard


@pytest.fixture
def guarded() -> Iterator[None]:
    """Install the guard and put numpy's error state back afterwards.

    ``np.seterr`` is global to the thread, so a test that installs the guard would
    otherwise change how every test after it behaves.
    """
    previous = np.geterr()
    install_numeric_guard()
    yield
    np.seterr(**previous)


@pytest.mark.usefixtures("guarded")
def test_a_nan_producing_operation_raises_instead_of_propagating() -> None:
    """The reason the guard exists: a `nan` in a Q-table is unrecoverable."""
    with pytest.raises(FloatingPointError):
        np.float64(0.0) / np.float64(0.0)


@pytest.mark.usefixtures("guarded")
def test_division_by_zero_and_overflow_raise() -> None:
    with pytest.raises(FloatingPointError):
        np.float64(1.0) / np.float64(0.0)
    with pytest.raises(FloatingPointError):
        np.exp(np.float64(1e6))


@pytest.mark.usefixtures("guarded")
def test_underflow_flushes_to_zero_without_raising() -> None:
    """Underflow is not a defect, and the log-sum-exp identity depends on it.

    ``exp`` of a large negative number is how log-sum-exp discards the terms too
    small to matter. Optuna's TPE sampler scores candidate points this way, so a
    guard that raised here crashed the search as soon as the sampler had enough
    history to model -- on the default ``startup_trials``, trial 10 of 125.
    """
    assert np.exp(np.float64(-10_000.0)) == 0.0
