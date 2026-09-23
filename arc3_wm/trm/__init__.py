"""Tiny Recursive Model (TRM) components for ARC-AGI-3.

A component library implementing TRM (Jolicoeur-Martineau, 2025,
arXiv:2510.04871) as a set of composable pieces for ARC-AGI-3:

- ``config``     - dataclass configs for every component (no torch import)
- ``core``       - the recursive reasoning core (z/y states, deep supervision,
                   ACT halting, EMA)
- ``tokenizer``  - palette-grid <-> token embedding, action encoding, grid head
- ``world_model``- TRM as a discrete next-frame world model
- ``policy``     - TRM as a factored policy (action type + click cell)
- ``data``       - replay corpus -> tensors, torch datasets
- ``agents``     - agent compositions (BC, WM-planner, hybrid) for the Gym env

Import discipline mirrors the parent package: this module and ``config`` are
importable without torch; everything else imports torch lazily via PEP 562 so
the laptop-only paths (tests, RHAE) stay torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import (
    AgentConfig,
    PolicyConfig,
    TokenizerConfig,
    TRMCoreConfig,
    WorldModelConfig,
)

_LAZY = {
    "TRMCore": ".core",
    "EMAHelper": ".core",
    "GridTokenizer": ".tokenizer",
    "ActionEncoder": ".tokenizer",
    "GridHead": ".tokenizer",
    "TRMWorldModel": ".world_model",
    "TRMPolicy": ".policy",
}

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import EMAHelper, TRMCore
    from .policy import TRMPolicy
    from .tokenizer import ActionEncoder, GridHead, GridTokenizer
    from .world_model import TRMWorldModel

__all__ = [
    "AgentConfig",
    "PolicyConfig",
    "TokenizerConfig",
    "TRMCoreConfig",
    "WorldModelConfig",
    "TRMCore",
    "EMAHelper",
    "GridTokenizer",
    "ActionEncoder",
    "GridHead",
    "TRMWorldModel",
    "TRMPolicy",
]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
