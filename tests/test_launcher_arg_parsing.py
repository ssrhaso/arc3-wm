"""Argparse + config-resolution tests for scripts/launch_pergame.py.

Exercises the parts of the launcher that don't need JAX:
- argparse defaults and required flags
- merging dreamerv3/configs.yaml + configs/arc3.yaml
- config-block layering on top of defaults
- {timestamp} substitution in --logdir
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the launcher's path mutation runs before we import other things.
import scripts.launch_pergame as L  # noqa: E402


# ----------------------------------------------------------------------
# argparse
# ----------------------------------------------------------------------


def test_argparser_required_flags():
    p = L.build_argparser()
    # --logdir and --task are required; argparse exits non-zero without them.
    with pytest.raises(SystemExit):
        p.parse_args([])
    with pytest.raises(SystemExit):
        p.parse_args(["--logdir", "/tmp/x"])  # missing --task


def test_argparser_happy_path():
    args, leftover = L.parse_args(
        ["--logdir", "/tmp/run1", "--task", "arc3_vc33", "--seed", "7"]
    )
    assert args.logdir == "/tmp/run1"
    assert args.task == "arc3_vc33"
    assert args.seed == 7
    assert args.script == "train"
    assert args.configs == ["defaults"]
    assert leftover == []


def test_argparser_configs_multi():
    args, _ = L.parse_args(
        [
            "--logdir", "/tmp/r",
            "--task", "arc3_vc33",
            "--configs", "size12m", "arc3",
        ]
    )
    assert args.configs == ["size12m", "arc3"]


def test_argparser_passthrough_leftover():
    """elements-style key=value flags must survive parse_known_args."""
    args, leftover = L.parse_args(
        [
            "--logdir", "/tmp/r",
            "--task", "arc3_vc33",
            "--run.steps", "100000",
            "--batch_size", "8",
        ]
    )
    assert "--run.steps" in leftover and "100000" in leftover
    assert "--batch_size" in leftover and "8" in leftover


def test_argparser_script_choice():
    args, _ = L.parse_args(
        ["--logdir", "/tmp/r", "--task", "arc3_vc33", "--script", "eval_only"]
    )
    assert args.script == "eval_only"


# ----------------------------------------------------------------------
# Config merging (dreamerv3 defaults + arc3.yaml)
# ----------------------------------------------------------------------


def test_load_merged_configs_has_defaults_and_arc3_blocks():
    merged = L.load_merged_configs()
    assert "defaults" in merged, "dreamerv3 defaults missing"
    assert "size12m" in merged, "size12m block missing from dreamerv3 configs"
    assert "crafter" in merged, "crafter block missing from dreamerv3 configs"
    assert "arc3" in merged, "arc3 block missing from configs/arc3.yaml"


def test_arc3_block_shape():
    merged = L.load_merged_configs()
    arc3 = merged["arc3"]
    # Named block sets task; per-suite defaults live in defaults.env.arc3.
    assert arc3["task"] == "arc3_vc33"
    defaults_env_arc3 = merged["defaults"]["env"]["arc3"]
    assert defaults_env_arc3["max_steps"] == 1000
    assert defaults_env_arc3["use_seed"] is True


def test_load_merged_configs_collision_guard(tmp_path, monkeypatch):
    """A block name that already exists in dreamerv3 must raise - defends
    against silent overrides of upstream config."""
    bad = tmp_path / "arc3_bad.yaml"
    bad.write_text("size12m:\n  task: bogus\n", encoding="utf-8")
    monkeypatch.setattr(L, "ARC3_CONFIG_PATH", bad)
    with pytest.raises(RuntimeError, match="collision"):
        L.load_merged_configs()


# ----------------------------------------------------------------------
# build_config
# ----------------------------------------------------------------------


def test_build_config_layers_arc3_block():
    args, leftover = L.parse_args(
        [
            "--logdir", "/tmp/r",
            "--task", "arc3_vc33",
            "--configs", "size12m", "arc3",
            "--seed", "3",
        ]
    )
    config = L.build_config(args, leftover)
    # arc3 block sets task to arc3_vc33; --task overrides any block-set task,
    # which we still want at the namespace level.
    assert config.task == "arc3_vc33"
    assert config.seed == 3
    assert config.env.arc3.max_steps == 1000


def test_build_config_unknown_block_raises():
    args, leftover = L.parse_args(
        ["--logdir", "/tmp/r", "--task", "arc3_vc33", "--configs", "no_such_block"]
    )
    with pytest.raises(ValueError, match="unknown config block"):
        L.build_config(args, leftover)


def test_build_config_timestamp_substitution(monkeypatch):
    args, leftover = L.parse_args(
        ["--logdir", "/tmp/runs/{timestamp}", "--task", "arc3_vc33"]
    )
    # Stub elements.timestamp to a known value via monkeypatch.
    import elements as _elements
    monkeypatch.setattr(_elements, "timestamp", lambda: "20260507T120000")
    config = L.build_config(args, leftover)
    assert config.logdir == "/tmp/runs/20260507T120000"


def test_build_config_passthrough_overrides_apply():
    """Leftover --key=value flags propagate via elements.Flags."""
    args, leftover = L.parse_args(
        [
            "--logdir", "/tmp/r",
            "--task", "arc3_vc33",
            "--seed", "0",
            "--batch_size", "4",
        ]
    )
    config = L.build_config(args, leftover)
    assert int(config.batch_size) == 4
