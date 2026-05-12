"""Laptop tests for scripts/validate_5070.py.

The script's actual checks need JAX + GPU; those don't run on the
laptop. These tests verify the *contracts* the script makes:

  - The module imports cleanly without JAX present (heavy imports are
    deferred inside main()). Critical for the pattern to hold;
    otherwise the script would crash on any host missing JAX.
  - The argparse layer parses every documented flag.
  - The WM-shape constants match the documented (B=4, T=16, 64, 64, 3)
    contract from the user's brief.
  - main() exits non-zero with a clear error message on the laptop
    (no JAX), proving the failure path is loud rather than silent.

Why this matters: validate_5070.py is itself a smoke test for the
5070 hardware. It must surface failures clearly. A silent-fail
validate_5070 would be worse than no validator at all.
"""
from __future__ import annotations

import importlib
import io
import sys
from contextlib import redirect_stderr, redirect_stdout

import pytest

import scripts.validate_5070 as V


# ---------------------------------------------------------------------------
# Module-level imports are JAX-free
# ---------------------------------------------------------------------------


def test_module_imports_cleanly_without_jax():
    """validate_5070 module imports without JAX present.

    The import at the top of this test file would have failed if the
    script eagerly imported JAX; this test re-asserts the property
    via importlib.reload() so the contract is pinned even if pytest's
    import cache muddies things."""
    importlib.reload(V)
    assert hasattr(V, "main")
    assert hasattr(V, "build_argparser")


# ---------------------------------------------------------------------------
# Shape constants
# ---------------------------------------------------------------------------


def test_wm_shape_constants_match_user_brief():
    """User brief: '(B=4, T=16, 64, 64, 3)'. Pinned because a refactor
    that changes these silently would let the smoke pass against a
    non-representative shape."""
    assert V.BATCH_SIZE == 4
    assert V.SEQ_LEN == 16
    assert V.IMG_HW == 64
    assert V.IMG_C == 3


def test_encoder_has_four_stride2_stages():
    """DV3 size12m runs 4 stride-2 convs (CLAUDE.md §Architecture →
    'Stock encoder behaviour'). Validator must mirror that."""
    assert len(V.ENC_CHANNELS) == 4
    # All-int channel counts; no None placeholders snuck in.
    assert all(isinstance(c, int) and c > 0 for c in V.ENC_CHANNELS)


# ---------------------------------------------------------------------------
# Argparse layer
# ---------------------------------------------------------------------------


def test_argparse_defaults():
    parser = V.build_argparser()
    args = parser.parse_args([])
    assert args.skip_dreamerv3 is False
    assert args.skip_backward is False


def test_argparse_skip_dreamerv3():
    args = V.build_argparser().parse_args(["--skip-dreamerv3"])
    assert args.skip_dreamerv3 is True


def test_argparse_skip_backward():
    args = V.build_argparser().parse_args(["--skip-backward"])
    assert args.skip_backward is True


def test_argparse_both_flags():
    args = V.build_argparser().parse_args(["--skip-dreamerv3", "--skip-backward"])
    assert args.skip_dreamerv3 is True
    assert args.skip_backward is True


# ---------------------------------------------------------------------------
# Failure path: laptop has no JAX → must exit 1 with a loud message
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is not None,
    reason="JAX is installed; this test only checks the no-JAX failure path",
)
def test_main_fails_loud_when_jax_missing():
    """On a host without JAX, main() must exit non-zero with a clear
    diagnostic on stderr — not silently pass, not crash with a
    cryptic traceback."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        rc = V.main([])
    assert rc == 1, f"expected exit 1 when JAX missing, got {rc}"
    err = stderr_buf.getvalue()
    assert "FAIL: JAX check" in err, (
        f"stderr should announce the JAX check failure, got: {err!r}"
    )
