"""Pin the 16-colour ARC palette.

The wrapper's RGB output is consumed unmodified by DreamerV3's encoder, so
silent palette drift would silently change the training distribution.
These tests pin the LUT to the values shipped in arc_agi.rendering.COLOR_MAP
at arc-agi==0.9.8 and verify decode invariants.
"""
from __future__ import annotations

import numpy as np
import pytest

from arc3_wm.palette import PALETTE_RGB, PALETTE_SIZE, decode_frame


def test_palette_shape_dtype():
    assert PALETTE_RGB.shape == (PALETTE_SIZE, 3) == (16, 3)
    assert PALETTE_RGB.dtype == np.uint8


def test_palette_is_immutable():
    with pytest.raises(ValueError):
        PALETTE_RGB[0, 0] = 0  # writeable=False


def test_palette_matches_arc_agi_color_map():
    """If arc-agi changes the palette, fail loud - don't silently diverge."""
    from arc_agi.rendering import COLOR_MAP, hex_to_rgb

    assert set(COLOR_MAP.keys()) == set(range(16))
    upstream = np.array(
        [hex_to_rgb(COLOR_MAP[i]) for i in range(16)], dtype=np.uint8
    )
    np.testing.assert_array_equal(PALETTE_RGB, upstream)


def test_decode_frame_shape_dtype():
    layer = np.zeros((64, 64), dtype=np.int8)
    rgb = decode_frame(layer)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8


def test_decode_frame_known_values():
    # White (idx 0) and Black (idx 5).
    layer = np.array([[0, 5], [5, 0]], dtype=np.int8)
    rgb = decode_frame(layer)
    assert tuple(rgb[0, 0]) == (0xFF, 0xFF, 0xFF)
    assert tuple(rgb[0, 1]) == (0x00, 0x00, 0x00)
    assert tuple(rgb[1, 0]) == (0x00, 0x00, 0x00)
    assert tuple(rgb[1, 1]) == (0xFF, 0xFF, 0xFF)


def test_decode_frame_full_range():
    """Every palette index maps to its frozen RGB triplet."""
    layer = np.arange(16, dtype=np.int8).reshape(4, 4)
    rgb = decode_frame(layer)
    for i in range(16):
        y, x = divmod(i, 4)
        assert tuple(rgb[y, x]) == tuple(PALETTE_RGB[i])


def test_decode_frame_rejects_out_of_range():
    bad = np.array([[0, 16]], dtype=np.int32)
    with pytest.raises(ValueError, match="out of range"):
        decode_frame(bad)


def test_decode_frame_rejects_wrong_ndim():
    bad = np.zeros((64, 64, 1), dtype=np.int8)
    with pytest.raises(ValueError, match="expected"):
        decode_frame(bad)


def test_decode_frame_empty_layer():
    """A zero-size (H, W) layer is in range and decodes to an empty image,
    rather than crashing on numpy's empty-reduction error."""
    rgb = decode_frame(np.zeros((0, 0), dtype=np.int8))
    assert rgb.shape == (0, 0, 3)
    assert rgb.dtype == np.uint8


def test_decode_frame_handles_int8_negative_safely():
    # int8 with values clipped to [0, 15]; we should not wrap.
    layer = np.array([[0, 15], [15, 0]], dtype=np.int8)
    rgb = decode_frame(layer)
    assert tuple(rgb[0, 1]) == tuple(PALETTE_RGB[15])
