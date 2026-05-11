"""Tests for the --init-from-ckpt warm-start path in scripts/launch_pergame.py.

Laptop-runnable: no JAX, no dreamerv3 agent construction, no Vast.
``agent.load`` is the real dreamerv3 surface that this code calls;
here it's a ``MagicMock`` whose call args we inspect.

Covers:
- argparse layer (default empty, captured when set)
- ``_is_remote_url`` + ``_resolve_init_ckpt_path`` (local, b2, https)
- ``_check_no_double_load`` (the four combinations)
- ``seed_wm_from_ckpt``: counter reset before agent.load, regex passed
  through, fail-loud on schema drift / zero matches / wrong key count /
  wrong param count
- Integration against the real Phase-3 v1 pkl (skip if not local)
- args.run.from_checkpoint stays empty when --init-from-ckpt is set
  (anti-double-load regression test)
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import scripts.launch_pergame as L  # noqa: E402


# ----------------------------------------------------------------------
# Helpers — synthetic state fabricators
# ----------------------------------------------------------------------


def _wm_key(prefix: str, idx: int = 0, leaf: str = "kernel") -> str:
    return f"{prefix}/mlp/linear{idx}/{leaf}"


def _state_with_matched(keys_and_shapes: dict[str, tuple[int, ...]]) -> dict:
    """Build a minimal valid state dict: 'params' + 'counters'."""
    return {
        "params": {k: np.zeros(s, dtype=np.float32) for k, s in keys_and_shapes.items()},
        "counters": {"updates": 192_000, "batches": 192_001, "actions": 0},
    }


# ----------------------------------------------------------------------
# argparse layer
# ----------------------------------------------------------------------


def test_argparser_init_from_ckpt_defaults_empty():
    args, _ = L.parse_args(
        ["--logdir", "/tmp/r", "--task", "arc3_vc33"]
    )
    assert args.init_from_ckpt == ""


def test_argparser_init_from_ckpt_captured():
    args, _ = L.parse_args(
        [
            "--logdir", "/tmp/r",
            "--task", "arc3_vc33",
            "--init-from-ckpt", "b2://bucket/key.pkl",
        ]
    )
    assert args.init_from_ckpt == "b2://bucket/key.pkl"


def test_init_from_ckpt_does_not_set_run_from_checkpoint():
    """Setting --init-from-ckpt must NOT also set config.run.from_checkpoint.

    Regression guard: the two mechanisms are deliberately separate; a
    refactor that ever conflates them would silently introduce a
    double-load."""
    args, leftover = L.parse_args(
        [
            "--logdir", "/tmp/r",
            "--task", "arc3_vc33",
            "--configs", "size12m", "arc3",
            "--init-from-ckpt", "/tmp/fake.pkl",
        ]
    )
    config = L.build_config(args, leftover)
    assert config.run.from_checkpoint == "", (
        f"--init-from-ckpt leaked into config.run.from_checkpoint: "
        f"{config.run.from_checkpoint!r}"
    )


# ----------------------------------------------------------------------
# _is_remote_url
# ----------------------------------------------------------------------


def test_is_remote_url_b2():
    assert L._is_remote_url("b2://bucket/key.pkl") is True


def test_is_remote_url_https():
    assert L._is_remote_url("https://example.com/x.pkl") is True


def test_is_remote_url_http():
    assert L._is_remote_url("http://example.com/x.pkl") is True


def test_is_remote_url_local_path():
    assert L._is_remote_url("/abs/path.pkl") is False
    assert L._is_remote_url("./relative.pkl") is False
    assert L._is_remote_url("relative.pkl") is False


# ----------------------------------------------------------------------
# _resolve_init_ckpt_path
# ----------------------------------------------------------------------


def test_resolve_init_ckpt_path_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        L._resolve_init_ckpt_path("")


def test_resolve_init_ckpt_path_local_returns_existing(tmp_path):
    f = tmp_path / "latest.pkl"
    f.write_bytes(b"x")
    assert L._resolve_init_ckpt_path(str(f)) == f


def test_resolve_init_ckpt_path_local_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        L._resolve_init_ckpt_path(str(tmp_path / "nope.pkl"))


def test_resolve_init_ckpt_path_b2_invokes_b2_cli(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, check):
        calls.append(list(cmd))
        # Simulate the downloader writing the file.
        target = Path(cmd[-1])
        target.write_bytes(b"fake")
        return mock.Mock(returncode=0)

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    out = L._resolve_init_ckpt_path("b2://bucket/key.pkl", cache_dir=tmp_path)
    assert out == tmp_path / "key.pkl"
    assert calls[0][:3] == ["b2", "file", "download"]
    assert "b2://bucket/key.pkl" in calls[0]


def test_resolve_init_ckpt_path_https_invokes_curl(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, check):
        calls.append(list(cmd))
        target = Path(cmd[-1])
        target.write_bytes(b"fake")
        return mock.Mock(returncode=0)

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    out = L._resolve_init_ckpt_path(
        "https://f003.backblazeb2.com/file/bucket/path.pkl", cache_dir=tmp_path
    )
    assert out == tmp_path / "path.pkl"
    assert calls[0][0] == "curl"


def test_resolve_init_ckpt_path_remote_reuses_cached_file(tmp_path, monkeypatch):
    """Re-running with an already-downloaded file must not re-invoke b2."""
    (tmp_path / "key.pkl").write_bytes(b"already-there")
    called = False

    def fake_run(cmd, check):
        nonlocal called
        called = True
        return mock.Mock(returncode=0)

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    out = L._resolve_init_ckpt_path("b2://bucket/key.pkl", cache_dir=tmp_path)
    assert out == tmp_path / "key.pkl"
    assert called is False, "cache should have short-circuited the download"


# ----------------------------------------------------------------------
# _check_no_double_load
# ----------------------------------------------------------------------


def test_check_no_double_load_both_empty():
    L._check_no_double_load("", "")  # no raise


def test_check_no_double_load_only_init_ok():
    L._check_no_double_load("/path/to/pkl", "")


def test_check_no_double_load_only_from_checkpoint_ok():
    L._check_no_double_load("", "/path/to/ckpt-dir")


def test_check_no_double_load_both_set_raises():
    with pytest.raises(ValueError, match="incompatible"):
        L._check_no_double_load("/init/path", "/run/path")


# ----------------------------------------------------------------------
# seed_wm_from_ckpt — schema validation
# ----------------------------------------------------------------------


def test_seed_wm_from_ckpt_rejects_missing_params(tmp_path):
    f = tmp_path / "bad.pkl"
    pickle.dump({"counters": {"updates": 0, "batches": 0, "actions": 0}}, open(f, "wb"))
    with pytest.raises(ValueError, match="unexpected checkpoint shape"):
        L.seed_wm_from_ckpt(mock.MagicMock(), f)


def test_seed_wm_from_ckpt_rejects_missing_counters(tmp_path):
    f = tmp_path / "bad.pkl"
    pickle.dump({"params": {"dyn/x": np.zeros(1)}}, open(f, "wb"))
    with pytest.raises(ValueError, match="unexpected checkpoint shape"):
        L.seed_wm_from_ckpt(mock.MagicMock(), f)


def test_seed_wm_from_ckpt_rejects_missing_updates_counter(tmp_path):
    f = tmp_path / "bad.pkl"
    pickle.dump(
        {"params": {"dyn/x": np.zeros(1)}, "counters": {"batches": 0, "actions": 0}},
        open(f, "wb"),
    )
    with pytest.raises(ValueError, match="missing counter"):
        L.seed_wm_from_ckpt(mock.MagicMock(), f)


# ----------------------------------------------------------------------
# seed_wm_from_ckpt — fail-loud invariants
# ----------------------------------------------------------------------


def test_seed_wm_from_ckpt_fails_loud_on_zero_regex_matches(tmp_path):
    """No WM-prefixed keys → must abort BEFORE calling agent.load.

    This is the most important fail-loud: a future refactor that
    renames module prefixes or changes the save format would silently
    skip the warm-start without this guard.
    """
    f = tmp_path / "no-wm.pkl"
    state = {
        "params": {"pol/head/kernel": np.zeros(4), "val/head/kernel": np.zeros(4)},
        "counters": {"updates": 1, "batches": 1, "actions": 0},
    }
    pickle.dump(state, open(f, "wb"))
    agent = mock.MagicMock()
    with pytest.raises(ValueError, match="matched zero keys"):
        L.seed_wm_from_ckpt(agent, f)
    agent.load.assert_not_called()


def test_seed_wm_from_ckpt_fails_loud_on_wrong_key_count(tmp_path, monkeypatch):
    """Matched-key count must equal WM_KEY_COUNT exactly."""
    # Shrink the expected count so the synthetic fixture can satisfy it.
    monkeypatch.setattr(L, "WM_KEY_COUNT", 2)
    monkeypatch.setattr(L, "WM_PARAM_COUNT", 8)
    f = tmp_path / "wrong-count.pkl"
    state = _state_with_matched({"dyn/a": (2, 2)})  # 1 key, 4 params
    pickle.dump(state, open(f, "wb"))
    agent = mock.MagicMock()
    with pytest.raises(ValueError, match=r"matched 1 keys; expected exactly 2"):
        L.seed_wm_from_ckpt(agent, f)
    agent.load.assert_not_called()


def test_seed_wm_from_ckpt_fails_loud_on_wrong_param_count(tmp_path, monkeypatch):
    """Matched-key element-sum must equal WM_PARAM_COUNT exactly."""
    monkeypatch.setattr(L, "WM_KEY_COUNT", 2)
    monkeypatch.setattr(L, "WM_PARAM_COUNT", 999)
    f = tmp_path / "wrong-params.pkl"
    state = _state_with_matched({"dyn/a": (2, 2), "enc/b": (2, 2)})  # 8 params, want 999
    pickle.dump(state, open(f, "wb"))
    agent = mock.MagicMock()
    with pytest.raises(ValueError, match=r"params sum to 8"):
        L.seed_wm_from_ckpt(agent, f)
    agent.load.assert_not_called()


# ----------------------------------------------------------------------
# seed_wm_from_ckpt — happy path
# ----------------------------------------------------------------------


def test_seed_wm_from_ckpt_resets_counters_before_load(tmp_path, monkeypatch):
    """state['counters'] must be {0,0,0} at the moment agent.load sees it."""
    monkeypatch.setattr(L, "WM_KEY_COUNT", 2)
    monkeypatch.setattr(L, "WM_PARAM_COUNT", 8)
    f = tmp_path / "good.pkl"
    state = _state_with_matched({"dyn/a": (2, 2), "enc/b": (2, 2)})
    pickle.dump(state, open(f, "wb"))

    captured_counters: dict = {}

    def fake_load(state_arg, regex=None):
        captured_counters.update(state_arg["counters"])

    agent = mock.MagicMock()
    agent.load.side_effect = fake_load

    diag = L.seed_wm_from_ckpt(agent, f)

    assert captured_counters == {"updates": 0, "batches": 0, "actions": 0}
    # Diagnostic dict reports the pre-reset values, for the launcher log line.
    assert diag["counter_values_before_reset"] == {
        "updates": 192_000, "batches": 192_001, "actions": 0,
    }


def test_seed_wm_from_ckpt_passes_wm_regex_to_agent_load(tmp_path, monkeypatch):
    """agent.load must be called with regex=WM_REGEX (defence vs refactor)."""
    monkeypatch.setattr(L, "WM_KEY_COUNT", 1)
    monkeypatch.setattr(L, "WM_PARAM_COUNT", 4)
    f = tmp_path / "good.pkl"
    pickle.dump(_state_with_matched({"dyn/a": (2, 2)}), open(f, "wb"))

    agent = mock.MagicMock()
    L.seed_wm_from_ckpt(agent, f)

    agent.load.assert_called_once()
    _, kwargs = agent.load.call_args
    assert kwargs.get("regex") == L.WM_REGEX


def test_wm_regex_matches_only_wm_prefixes():
    """Direct regex coverage — defends against tweaks to WM_REGEX."""
    good = [
        "dyn/dyngru/kernel", "enc/conv0/kernel", "dec/sp0/scale",
        "rew/head/logits/bias", "con/head/logit/kernel",
    ]
    bad = [
        "opt/state/1/0", "opt/state/1/1/dyn/x",  # opt/-prefixed must NOT match
        "pol/head/kernel", "val/head/kernel",     # future actor/critic
        "agent/dyn/x",                            # nested style — must NOT match flat
    ]
    for k in good:
        assert re.match(L.WM_REGEX, k), f"WM_REGEX should match {k!r}"
    for k in bad:
        assert re.match(L.WM_REGEX, k) is None, f"WM_REGEX should NOT match {k!r}"


# ----------------------------------------------------------------------
# Integration: real Phase-3 v1 pkl if present locally
# ----------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_CKPT = _REPO_ROOT / "checkpoints" / "pretrained-wm" / "v1" / "latest.pkl"


@pytest.mark.skipif(not _REAL_CKPT.exists(), reason="Phase-3 v1 ckpt not local")
def test_seed_wm_from_ckpt_against_real_v1_pkl():
    """Smoke-test the helper end-to-end against the real Phase-3 ckpt.

    Uses a MagicMock agent so no JAX is needed — but the validation
    runs against the actual file, exercising every assertion path
    with the real production-shape state dict.
    """
    agent = mock.MagicMock()
    diag = L.seed_wm_from_ckpt(agent, _REAL_CKPT)

    assert diag["matched_keys"] == L.WM_KEY_COUNT == 68
    assert diag["matched_params"] == L.WM_PARAM_COUNT == 9_898_179
    # Phase-3 v1 counters are well-documented (docs/phase4-warmstart-notes.md).
    assert diag["counter_values_before_reset"] == {
        "updates": 192_000, "batches": 192_001, "actions": 0,
    }
    # agent.load was called exactly once, with the real regex.
    agent.load.assert_called_once()
    _, kwargs = agent.load.call_args
    assert kwargs.get("regex") == L.WM_REGEX
