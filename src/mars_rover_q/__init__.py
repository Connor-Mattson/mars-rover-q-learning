"""Mars Sample Return: reward design in tabular Q-learning.

A small, dependency-light research repository for studying how reward design
changes the sample efficiency and behaviour of a tabular Q-learning rover.
"""

from __future__ import annotations

import os

# Pygame prints a banner on import. Training and metadata collection import it
# without ever opening a window, so keep stdout clean.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

__version__ = "0.1.0"

__all__ = ["__version__"]
