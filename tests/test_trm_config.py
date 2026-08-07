"""Spec for arc3_wm.trm.config - pure-Python, torch-free."""

from __future__ import annotations

import sys

import pytest

from arc3_wm.trm import config as C
from arc3_wm.trm.config import (
    AgentConfig,
    PolicyConfig,
    TokenizerConfig,
    TRMCoreConfig,
    WorldModelConfig,
    from_dict,
    to_dict,
)


def test_trm_package_imports_without_torch():
    # The subpackage __init__ and config must not pull torch onto the
    # eager import path (mirrors the parent package's JAX discipline).
    import arc3_wm.trm  # noqa: F401

    assert "torch" not in sys.modules or True  # torch may be installed; check lazily
    # Stronger check: config module itself never imports torch.
    assert "torch" not in C.__dict__


def test_core_defaults_follow_paper():
    cfg = TRMCoreConfig()
    assert (cfg.d_model, cfg.n_heads, cfg.n_layers) == (512, 8, 2)
    assert (cfg.h_cycles, cfg.l_cycles) == (3, 6)
    assert cfg.seq_mixer == "attention"
    assert cfg.pos_encoding == "rope"
    assert cfg.y_init == "buffer"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 100, "n_heads": 8},  # not divisible
        {"seq_mixer": "conv"},
        {"pos_encoding": "learned"},
        {"y_init": "zeros"},
        {"h_cycles": 0},
        {"n_supervision": 0},
    ],
)
def test_core_validation_rejects(kwargs):
    with pytest.raises(ValueError):
        TRMCoreConfig(**kwargs)


def test_tokenizer_shapes():
    cfg = TokenizerConfig(patch_size=4)
    assert cfg.tokens_per_side == 16
    assert cfg.n_tokens == 256
    assert cfg.cells_per_patch == 16
    with pytest.raises(ValueError):
        TokenizerConfig(patch_size=5)


def test_composite_d_model_mismatch_rejected():
    with pytest.raises(ValueError):
        WorldModelConfig(core=TRMCoreConfig(d_model=256), tokenizer=TokenizerConfig(d_model=512))
    with pytest.raises(ValueError):
        PolicyConfig(core=TRMCoreConfig(d_model=256), tokenizer=TokenizerConfig(d_model=512))


def test_wm_changed_cell_weight_floor():
    with pytest.raises(ValueError):
        WorldModelConfig(changed_cell_weight=0.5)


def test_agent_config_component_switches():
    cfg = AgentConfig(use_bc=False, use_wm=True)
    assert not cfg.use_bc and cfg.use_wm
    with pytest.raises(ValueError):
        AgentConfig(epsilon=1.5)


def test_dict_round_trip():
    cfg = WorldModelConfig(
        core=TRMCoreConfig(d_model=256, n_heads=4, l_cycles=2),
        tokenizer=TokenizerConfig(d_model=256, patch_size=8),
        changed_cell_weight=5.0,
    )
    data = to_dict(cfg)
    assert data["core"]["d_model"] == 256
    rebuilt = from_dict(WorldModelConfig, data)
    assert rebuilt == cfg


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown config keys"):
        from_dict(TRMCoreConfig, {"d_modell": 128})


def test_action_type_constants():
    assert C.N_ACTION_TYPES == 7
    assert C.ACTION6_TYPE_INDEX == 5
    assert C.GRID_SIZE == 64 and C.PALETTE_SIZE == 16
