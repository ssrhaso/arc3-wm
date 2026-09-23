"""Module-level import smoke for scripts/launch_pergame.py.

Verifies the launcher is importable on a laptop without JAX, portal, or
the DreamerV3 stack. All heavy imports are lazy (inside helpers); this
test catches a regression where someone adds a top-level
``import dreamerv3`` etc. that breaks laptop ergonomics.

The test also confirms ``third_party/dreamerv3`` is on ``sys.path`` after
import, so ``import embodied`` would resolve in environments where the
deps ARE installed (Vast).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_launcher_module_imports_clean():
    # Force a fresh import to catch lazy-import regressions.
    sys.modules.pop("scripts.launch_pergame", None)
    sys.modules.pop("launch_pergame", None)
    spec = importlib.util.spec_from_file_location(
        "launch_pergame_under_test",
        Path(__file__).resolve().parents[1] / "scripts" / "launch_pergame.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not raise

    # Public surface.
    for name in (
        "build_argparser",
        "parse_args",
        "load_merged_configs",
        "build_config",
        "make_env",
        "make_agent",
        "make_replay",
        "make_stream",
        "make_logger",
        "vast_only_isinstance_check",
        "main",
    ):
        assert hasattr(mod, name), f"launcher missing public symbol {name!r}"


def test_third_party_dreamerv3_on_sys_path_after_import():
    """The launcher prepends third_party/dreamerv3 so 'embodied'/'dreamerv3'
    resolve. We don't import them here (laptop has no portal/JAX), only
    assert the path entry is present."""
    import scripts.launch_pergame  # noqa: F401 - side-effect: sys.path mutation

    expected = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "dreamerv3"
    )
    assert str(expected) in sys.path, (
        f"{expected} not on sys.path after importing launcher"
    )


def test_no_top_level_jax_or_portal_import():
    """Heavy deps must stay lazy. If someone adds ``import jax`` at module
    top, this test fails because importing the launcher imports jax."""
    # Drop any cached state.
    for mod in [m for m in list(sys.modules) if m.startswith(("jax", "portal", "dreamerv3", "embodied"))]:
        sys.modules.pop(mod, None)

    spec = importlib.util.spec_from_file_location(
        "launch_pergame_isolation",
        Path(__file__).resolve().parents[1] / "scripts" / "launch_pergame.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for forbidden in ("jax", "portal", "dreamerv3", "embodied"):
        assert forbidden not in sys.modules, (
            f"top-level import of {forbidden!r} leaked into launcher"
        )
