"""5070-cluster hardware smoke for Phase 4.

Runs three checks on the host:

  (1) JAX import + device discovery. Asserts at least one GPU device
      is visible and reports its name. Catches a missing CUDA driver,
      a JAX wheel that doesn't see the GPU, or sm_120 codegen failure
      at JAX import time.
  (2) DreamerV3 import. ``from dreamerv3.agent import Agent`` —
      catches the case where DV3's transitive deps (portal, embodied)
      are broken on the host.
  (3) One forward + one backward pass on a WM-shaped dummy batch
      ``(B=4, T=16, 64, 64, 3)`` using a 4-conv stride-2 encoder.
      Exercises JAX's codegen + autograd on sm_120 with the shape the
      DV3 size12m encoder actually runs on.

Why this script exists: the 5070s are Blackwell (sm_120). JAX 0.4.33's
sm_120 support is unvalidated in our stack. Running this once per
new card answers "can I trust this hardware to run Phase 4?" in ~30
seconds vs. paying for a multi-hour Vast run only to fail at start.

Usage::

    python scripts/validate_5070.py

Exits 0 on full pass, 1 on first failure with a clear diagnostic.

Heavy imports are deferred inside ``main()`` so the module is
laptop-importable for tests (no JAX needed for argparse coverage).
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Sequence

# WM-shaped dummy batch — mirrors DV3 size12m's effective shape:
#   B=4 minibatch, T=16 sequence length, 64x64 RGB image.
# The DV3 stock encoder runs 4 stride-2 convs: 64 → 32 → 16 → 8 → 4.
BATCH_SIZE = 4
SEQ_LEN = 16
IMG_HW = 64
IMG_C = 3
ENC_CHANNELS = (32, 64, 128, 256)  # 4 stride-2 stages


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="validate_5070.py",
        description=(
            "Hardware smoke for the 5070 cluster: JAX + GPU + DV3 + "
            "one WM-shaped fwd/bwd."
        ),
    )
    p.add_argument(
        "--skip-dreamerv3",
        action="store_true",
        help="Skip the dreamerv3 import check (use when only validating JAX).",
    )
    p.add_argument(
        "--skip-backward",
        action="store_true",
        help="Skip the backward pass (still runs the forward).",
    )
    return p


def _check_jax():
    """JAX import + GPU device. Returns (jax_module, device_list)."""
    import jax  # noqa: E402

    print(f"jax.__version__ = {jax.__version__}")
    devices = jax.devices()
    print(f"jax.devices() = {devices}")
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError(
            f"no GPU device visible to JAX (platforms: "
            f"{sorted({d.platform for d in devices})}); "
            f"check CUDA driver + jax[cuda12]"
        )
    print(f"first GPU: {gpu_devices[0]} ({gpu_devices[0].device_kind})")
    return jax, gpu_devices


def _check_dreamerv3():
    """DV3 import sanity. Catches portal/embodied install drift."""
    # Make ``import dreamerv3`` resolve our pinned source, same as
    # launch_pergame.py does.
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    dv3_root = repo_root / "third_party" / "dreamerv3"
    if dv3_root.is_dir() and str(dv3_root) not in sys.path:
        sys.path.insert(0, str(dv3_root))
    from dreamerv3.agent import Agent  # noqa: F401
    print("dreamerv3.agent.Agent imported OK")


def _wm_shaped_forward_backward(jax, *, skip_backward: bool) -> None:
    """One forward + (optionally) one backward on a (B, T, 64, 64, 3) batch.

    Builds a tiny 4-conv stride-2 encoder of shape parameters mirroring
    DV3 size12m's encoder (64 → 32 → 16 → 8 → 4 spatial). Runs the
    full pipeline once under jax.jit for forward, then jax.grad for the
    backward, timing each phase.

    Not the real DV3 WM — the goal is to exercise JAX's codegen +
    autograd on the right SHAPE under sm_120, not to test DV3
    correctness. Phase 2's Crafter sanity is the contract test for
    DV3 itself.
    """
    import jax.numpy as jnp  # noqa: E402

    # Flatten (B, T) → leading batch dim of 64 for the conv stack.
    leading_b = BATCH_SIZE * SEQ_LEN
    x_shape = (leading_b, IMG_HW, IMG_HW, IMG_C)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, x_shape, dtype=jnp.float32)

    # Random conv kernels mirroring DV3 size12m's encoder stages.
    kernel_specs = []
    in_c = IMG_C
    for out_c in ENC_CHANNELS:
        kernel_specs.append((4, 4, in_c, out_c))
        in_c = out_c
    keys = jax.random.split(key, len(kernel_specs))
    kernels = [
        jax.random.normal(k, s, dtype=jnp.float32) * 0.05
        for k, s in zip(keys, kernel_specs)
    ]

    def encoder(params, x):
        h = x
        for w in params:
            h = jax.lax.conv_general_dilated(
                h, w,
                window_strides=(2, 2),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            h = jax.nn.relu(h)
        return h

    def loss_fn(params, x):
        h = encoder(params, x)
        return jnp.mean(h * h)

    print(f"forward: input shape {x.shape}")
    t0 = time.time()
    out = jax.jit(encoder)(kernels, x).block_until_ready()
    t_fwd = time.time() - t0
    print(f"forward: output shape {out.shape}, time={t_fwd:.3f}s")

    if skip_backward:
        print("backward: skipped (--skip-backward)")
        return

    grad_fn = jax.jit(jax.grad(loss_fn))
    t0 = time.time()
    grads = grad_fn(kernels, x)
    # block_until_ready on the first grad to force materialization.
    grads[0].block_until_ready()
    t_bwd = time.time() - t0
    print(
        f"backward: {len(grads)} grad tensors, "
        f"first shape {grads[0].shape}, time={t_bwd:.3f}s"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    print("=" * 60)
    print("5070 hardware smoke")
    print("=" * 60)

    try:
        jax, _ = _check_jax()
    except Exception as e:
        print(f"FAIL: JAX check — {e}", file=sys.stderr)
        return 1

    if not args.skip_dreamerv3:
        try:
            _check_dreamerv3()
        except Exception as e:
            print(f"FAIL: dreamerv3 import — {e}", file=sys.stderr)
            return 1
    else:
        print("dreamerv3 import: skipped (--skip-dreamerv3)")

    try:
        _wm_shaped_forward_backward(jax, skip_backward=args.skip_backward)
    except Exception as e:
        print(f"FAIL: WM-shaped fwd/bwd — {e}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("PASS: 5070 hardware smoke green")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
