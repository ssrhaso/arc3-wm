"""Access to the official TinyRecursiveModels code (the "Link").

The official repo is NOT vendored into git: ``third_party/`` is gitignored
by design and fetched per machine by ``scripts/fetch_trm_official.sh``
(repo convention, same as ``third_party/dreamerv3``). This shim puts the
clone on ``sys.path`` and re-exports the pieces the TRM adapter uses, so
the rest of ``arc3_wm.trm`` never manipulates ``sys.path`` itself.

Verified against commit c011037 (main HEAD). The official package uses
absolute ``models.*`` imports; the path is APPENDED (not prepended) so an
already-importable ``models``/``utils`` package would shadow it loudly
rather than the other way round.

Nothing in third_party/TinyRecursiveModels may be modified: every
adaptation lives in arc3_wm.trm (see config.py's deviation list).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "TinyRecursiveModels"


def _ensure() -> None:
    if not (_ROOT / "models" / "layers.py").is_file():
        raise ImportError(
            "official TinyRecursiveModels clone missing at "
            f"{_ROOT}; run scripts/fetch_trm_official.sh first"
        )
    if str(_ROOT) not in sys.path:
        sys.path.append(str(_ROOT))


_ensure()

from models.common import trunc_normal_init_  # noqa: E402
from models.ema import EMAHelper as OfficialEMAHelper  # noqa: E402
from models.layers import (  # noqa: E402
    Attention,
    CastedLinear,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)
from models.losses import softmax_cross_entropy, stablemax_cross_entropy  # noqa: E402
from models.recursive_reasoning.trm import (  # noqa: E402
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
)

__all__ = [
    "Attention",
    "CastedLinear",
    "OfficialEMAHelper",
    "RotaryEmbedding",
    "SwiGLU",
    "TinyRecursiveReasoningModel_ACTV1Block",
    "TinyRecursiveReasoningModel_ACTV1Config",
    "TinyRecursiveReasoningModel_ACTV1ReasoningModule",
    "rms_norm",
    "softmax_cross_entropy",
    "stablemax_cross_entropy",
    "trunc_normal_init_",
]
