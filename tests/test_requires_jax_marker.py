"""The ``requires_jax`` marker is registered and gates on JAX availability.

On the laptop (no JAX) the marked test below is auto-skipped by
conftest's ``pytest_collection_modifyitems``; on a GPU box it runs and
proves JAX is importable. The unmarked test confirms the marker is known
to the (``--strict-markers``) config, so a typo'd marker fails loudly.
"""
from __future__ import annotations

import pytest


def test_requires_jax_marker_is_registered(pytestconfig):
    registered = "\n".join(pytestconfig.getini("markers"))
    assert "requires_jax" in registered


@pytest.mark.requires_jax
def test_jax_present_when_marker_runs():
    import jax  # noqa: F401  # only reached when conftest did not skip us
