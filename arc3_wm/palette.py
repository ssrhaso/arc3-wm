"""ARC-AGI-3 16-colour palette.

Frozen RGB lookup table for converting the engine's int8 palette indices
(0..15) to (H, W, 3) uint8 images for DreamerV3.

Source of truth: ``arc_agi.rendering.COLOR_MAP`` (RGBA hex). We mirror the
values verbatim so the wrapper has no import-time dependency on the
rendering module and so a future change in arc_agi is loud (caught by
``tests/test_palette.py``), not silent.
"""
from __future__ import annotations

import numpy as np

# RGB triplets for palette indices 0..15. Values mirror arc_agi.rendering.COLOR_MAP
# at arc-agi==0.9.8 (alpha channel dropped). Pinned by tests/test_palette.py.
PALETTE_RGB: np.ndarray = np.array(
    [
        (0xFF, 0xFF, 0xFF),  # 0  White
        (0xCC, 0xCC, 0xCC),  # 1  Off-white
        (0x99, 0x99, 0x99),  # 2  Neutral light
        (0x66, 0x66, 0x66),  # 3  Neutral
        (0x33, 0x33, 0x33),  # 4  Off black
        (0x00, 0x00, 0x00),  # 5  Black
        (0xE5, 0x3A, 0xA3),  # 6  Magenta
        (0xFF, 0x7B, 0xCC),  # 7  Magenta light
        (0xF9, 0x3C, 0x31),  # 8  Red
        (0x1E, 0x93, 0xFF),  # 9  Blue
        (0x88, 0xD8, 0xF1),  # 10 Blue light
        (0xFF, 0xDC, 0x00),  # 11 Yellow
        (0xFF, 0x85, 0x1B),  # 12 Orange
        (0x92, 0x12, 0x31),  # 13 Maroon
        (0x4F, 0xCC, 0x30),  # 14 Green
        (0xA3, 0x56, 0xD6),  # 15 Purple
    ],
    dtype=np.uint8,
)
PALETTE_RGB.setflags(write=False)

PALETTE_SIZE = 16


def decode_frame(layer: np.ndarray) -> np.ndarray:
    """Map a (H, W) palette-index array to a (H, W, 3) uint8 RGB image.

    Accepts int8 / uint8 / int32; values must be in ``[0, 15]``. Indexing
    is done via ``PALETTE_RGB[layer]`` so the result is contiguous and
    already uint8 - no copy or normalisation needed before DreamerV3's
    image encoder.
    """
    arr = np.asarray(layer)
    if arr.ndim != 2:
        raise ValueError(f"expected (H, W) layer, got shape {arr.shape}")
    # Cast to a small unsigned dtype before indexing to avoid negative
    # int8 wrap-around (values are documented in [0, 15] anyway).
    idx = arr.astype(np.intp, copy=False)
    if idx.min() < 0 or idx.max() >= PALETTE_SIZE:
        raise ValueError(
            f"palette index out of range [0, {PALETTE_SIZE - 1}]: "
            f"min={idx.min()} max={idx.max()}"
        )
    return PALETTE_RGB[idx]
