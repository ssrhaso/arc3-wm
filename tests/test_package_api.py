"""The public import surface of ``arc3_wm``.

A formalised package commits to what it exports. These tests pin the
public API so a refactor that silently drops a name fails loudly:

- headline names are importable straight off the package;
- ``__all__`` is exhaustive and every name in it resolves;
- ``__version__`` is a real (non-zero) PEP 440 string;
- ``ARC3EmbodiedEnv`` is exposed *lazily* (PEP 562) so importing the
  package on a laptop without the JAX-side ``elements`` dep still works;
- unknown attribute access raises ``AttributeError`` (not a bare KeyError
  from the ``__getattr__`` shim).
"""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

# Laptop-importable submodules that declare their own ``__all__``. The
# JAX-only ``embodied_env`` is excluded: importing it needs the
# DreamerV3/``elements`` stack that the laptop path does not install.
_PUBLIC_SUBMODULES = [
    "arc3_wm.action_space",
    "arc3_wm.palette",
    "arc3_wm.rhae",
    "arc3_wm.replay_loader",
    "arc3_wm.registration",
    "arc3_wm.env",
    "arc3_wm.eval_reward_sink",
    "arc3_wm.wm_only_agent",
]


def test_version_is_real():
    import arc3_wm

    v = arc3_wm.__version__
    assert isinstance(v, str)
    assert v != "0.0.0", "Phase-0 scaffold version never bumped"
    parts = v.split(".")
    assert len(parts) >= 2 and all(p.isdigit() for p in parts[:2])


def test_headline_names_importable():
    from arc3_wm import (  # noqa: F401
        ARC3GymEnv,
        N_ACTIONS,
        arc_to_flat,
        build_mask,
        flat_to_arc,
    )

    assert N_ACTIONS == 4102


def test_all_is_exhaustive_and_resolvable():
    import arc3_wm

    assert hasattr(arc3_wm, "__all__")
    for name in arc3_wm.__all__:
        assert hasattr(arc3_wm, name), f"__all__ lists {name!r} but it does not resolve"


@pytest.mark.parametrize("module_name", _PUBLIC_SUBMODULES)
def test_submodule_all_resolves(module_name):
    """Each public submodule declares an ``__all__`` whose names all resolve.

    Guards drift: a renamed/removed public function that is still listed in
    a module's ``__all__`` (or a leading-underscore helper accidentally
    exported) trips here rather than surfacing as a broken ``import *``.
    """
    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__"), f"{module_name} declares no __all__"
    assert module.__all__, f"{module_name}.__all__ is empty"
    for name in module.__all__:
        assert hasattr(module, name), (
            f"{module_name}.__all__ lists {name!r} but it does not resolve"
        )
        assert not name.startswith("_"), (
            f"{module_name}.__all__ exports private name {name!r}"
        )


def test_embodied_env_not_imported_eagerly():
    """``import arc3_wm`` must not pull in ``arc3_wm.embodied_env``.

    Eager import would drag the JAX-side ``elements`` dependency into the
    laptop/Gymnasium-only path. Checked in a *fresh* interpreter - within
    one pytest session an earlier ``hasattr`` over ``__all__`` would have
    already triggered the lazy import, masking a regression.
    """
    code = (
        "import sys, arc3_wm; "
        "sys.exit(1 if 'arc3_wm.embodied_env' in sys.modules else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code])
    assert proc.returncode == 0, "embodied_env was imported eagerly at package import"


def test_embodied_env_lazily_resolvable():
    """``ARC3EmbodiedEnv`` is in ``__all__`` and resolves on demand."""
    import arc3_wm

    assert "ARC3EmbodiedEnv" in arc3_wm.__all__
    obj = arc3_wm.ARC3EmbodiedEnv
    assert obj.__name__ == "ARC3EmbodiedEnv"


def test_unknown_attribute_raises_attributeerror():
    import arc3_wm

    with pytest.raises(AttributeError):
        _ = arc3_wm.NoSuchSymbol


def test_reimport_is_stable():
    import arc3_wm

    importlib.reload(arc3_wm)
    assert arc3_wm.__version__ != "0.0.0"
