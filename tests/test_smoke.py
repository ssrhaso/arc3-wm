"""Import smoke + version sanity."""
from __future__ import annotations

import importlib


def test_arc3_wm_imports():
    importlib.import_module("arc3_wm")
    importlib.import_module("arc3_wm.action_space")
    importlib.import_module("arc3_wm.palette")
    importlib.import_module("arc3_wm.env")


def test_arc_agi_version():
    import importlib.metadata as md
    v = md.version("arc-agi")
    # Pin major.minor; bump explicitly when arc-agi releases something new.
    assert v.startswith("0.9."), f"unexpected arc-agi version: {v}"
