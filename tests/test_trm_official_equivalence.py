"""Equivalence insurance for the official link: our TRMCore driver must
reproduce the official TinyRecursiveReasoningModel_ACTV1_Inner recursion
exactly - same weights, same init states, same pre-embedded inputs.

The official Inner embeds its inputs internally, so the comparison feeds our
core the official ``_input_embeddings`` output and checks the carried states
(z_H <-> our y, z_L <-> our z) and the halt logit after one supervision
step. Runs on CPU in fp32; skipped when the third_party clone is absent
(run scripts/fetch_trm_official.sh).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
_official = pytest.importorskip("arc3_wm.trm._official")

from arc3_wm.trm.config import TRMCoreConfig  # noqa: E402
from arc3_wm.trm.core import TRMCore  # noqa: E402

SEQ = 16


def _build_pair(mlp: bool):
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1_Inner,
    )

    ocfg = _official.TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=SEQ,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        num_puzzle_identifiers=1,
        vocab_size=16,
        H_cycles=3,
        L_cycles=2,
        H_layers=0,
        L_layers=2,
        hidden_size=32,
        expansion=4.0,
        num_heads=4,
        pos_encodings="none" if mlp else "rope",
        halt_max_steps=4,
        halt_exploration_prob=0.1,
        forward_dtype="float32",
        mlp_t=mlp,
    )
    torch.manual_seed(0)
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(ocfg)
    core = TRMCore(
        TRMCoreConfig(
            d_model=32,
            n_heads=4,
            n_layers=2,
            l_cycles=2,
            h_cycles=3,
            seq_mixer="mlp" if mlp else "attention",
            pos_encoding="none" if mlp else "rope",
        ),
        max_seq_len=SEQ,
    )
    # Same official block classes -> identical submodule names; align weights
    # and init states so only the driver loop can make outputs differ.
    core.net.load_state_dict(inner.L_level.state_dict())
    core.q_head.load_state_dict(inner.q_head.state_dict())
    core.y_init.copy_(inner.H_init)
    core.z_init.copy_(inner.L_init)
    return inner, core


def _compare(inner, core) -> None:
    inputs = torch.randint(0, 16, (2, SEQ))
    batch = {"inputs": inputs, "puzzle_identifiers": torch.zeros(2, dtype=torch.long)}
    with torch.no_grad():
        x = inner._input_embeddings(inputs, batch["puzzle_identifiers"])
        carry = inner.reset_carry(torch.ones(2, dtype=torch.bool), inner.empty_carry(2))
        new_carry, _, (q_halt_ref, _) = inner(carry, batch)
        y, q_halt, (cy, cz) = core(x)
    assert torch.allclose(y, new_carry.z_H, atol=1e-6)
    assert torch.allclose(cy, new_carry.z_H, atol=1e-6)
    assert torch.allclose(cz, new_carry.z_L, atol=1e-6)
    assert torch.allclose(q_halt, q_halt_ref, atol=1e-6)


def test_recursion_matches_official_inner_attention_rope():
    inner, core = _build_pair(mlp=False)
    _compare(inner, core)


def test_recursion_matches_official_inner_mlp():
    inner, core = _build_pair(mlp=True)
    _compare(inner, core)
