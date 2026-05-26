"""Shared pytest configuration.

Declares the ``requires_jax`` marker and auto-skips tests that carry it
when the JAX/DreamerV3 stack is not importable. The laptop path (Phases
0-1) has no JAX, so a handful of tests must skip there and run on the GPU
boxes. Historically each did its own inline ``pytest.importorskip("jax")``;
the marker gives new JAX-only tests one declared, ``--strict-markers``-safe
way to express that contract.

    @pytest.mark.requires_jax
    def test_something_that_needs_dreamerv3(): ...
"""
from __future__ import annotations

import pytest


def _jax_importable() -> bool:
    try:
        import jax  # noqa: F401
    except Exception:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    if _jax_importable():
        return
    skip_jax = pytest.mark.skip(reason="requires JAX (not installed on this machine)")
    for item in items:
        if "requires_jax" in item.keywords:
            item.add_marker(skip_jax)
