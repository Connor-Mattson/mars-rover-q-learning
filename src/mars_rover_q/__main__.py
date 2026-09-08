"""Allow ``python -m mars_rover_q ...`` in addition to ``python -m mars_rover_q.cli``."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - module execution path
    raise SystemExit(main())
