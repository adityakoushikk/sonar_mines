"""Pytest bootstrap: put the repo root on sys.path so `src` / `pretraining` import.

Tests live at ``pretraining/tests/``; ``parents[2]`` of this file is the repo
root. Inserting it at index 0 lets ``from src...`` and ``from pretraining...``
resolve without installing the project as a package.
"""
from __future__ import annotations

import pathlib
import sys

# parents[0]=tests, parents[1]=pretraining, parents[2]=repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
