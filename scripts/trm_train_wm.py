#!/usr/bin/env python
"""Train a TRM world model on preprocessed replay caches.

Per-game (paper-comparable) or cross-game (pass several games). Example:
    python scripts/trm_train_wm.py --data data/trm_cache --games vc33 \
        --out checkpoints/trm_wm/vc33 --epochs 30 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3_wm.trm import config as C  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="npz cache dir")
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue from out/latest.pt if present")
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--unroll-weight", type=float, default=0.0,
                   help="scheduled-sampling 2-step unroll loss weight (0 = off)")
    # Model overrides.
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--h-cycles", type=int, default=3)
    p.add_argument("--l-cycles", type=int, default=6)
    p.add_argument("--n-supervision", type=int, default=16)
    p.add_argument("--halt-max-steps", type=int, default=16)
    p.add_argument("--y-init", choices=["buffer", "input"], default="buffer")
    p.add_argument("--seq-mixer", choices=["attention", "mlp"], default="attention",
                   help="token mixer inside the shared net (mlp = MLP-Mixer ablation)")
    p.add_argument("--halt-bias-init", type=float, default=0.0)
    p.add_argument("--changed-cell-weight", type=float, default=20.0)
    p.add_argument("--loss", choices=["stablemax_ce", "softmax_ce"], default="stablemax_ce")
    return p


def build_model_config(args) -> C.WorldModelConfig:
    core = C.TRMCoreConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        h_cycles=args.h_cycles,
        l_cycles=args.l_cycles,
        n_supervision=args.n_supervision,
        halt_max_steps=args.halt_max_steps,
        y_init=args.y_init,
        halt_bias_init=args.halt_bias_init,
        seq_mixer=args.seq_mixer,
    )
    tok = C.TokenizerConfig(d_model=args.d_model, patch_size=args.patch_size)
    return C.WorldModelConfig(
        core=core,
        tokenizer=tok,
        changed_cell_weight=args.changed_cell_weight,
        loss=args.loss,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from arc3_wm.trm.data import WMTransitionDataset, train_val_split_episodes
    from arc3_wm.trm.training import TrainConfig, evaluate_wm, train_loop
    from arc3_wm.trm.world_model import TRMWorldModel

    paths = []
    for game in args.games:
        path = args.data / f"{game}.npz"
        if not path.exists():
            print(f"missing cache: {path}", file=sys.stderr)
            return 1
        paths.append(path)

    # Episode-grouped split per game, then merged datasets.
    train_specs, val_specs = [], []
    for path in paths:
        tr, va = train_val_split_episodes(path, args.val_fraction, args.seed)
        train_specs.append((path, tr))
        val_specs.append((path, va))

    n_steps = 2 if args.unroll_weight > 0 else 1
    train_ds = WMTransitionDataset(train_specs, dedup=not args.no_dedup, n_steps=n_steps)
    val_ds = WMTransitionDataset(val_specs, dedup=False)
    print(f"train transitions: {len(train_ds)}, val transitions: {len(val_ds)}")

    model_cfg = build_model_config(args)
    torch.manual_seed(args.seed)
    model = TRMWorldModel(model_cfg)

    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        ema_decay=args.ema_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        device=args.device,
        bf16=not args.no_bf16,
        seed=args.seed,
        eval_every=args.eval_every,
        unroll_weight=args.unroll_weight,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(
        json.dumps(
            {
                "games": args.games,
                "model_config": C.to_dict(model_cfg),
                "train_config": vars(args) | {"data": str(args.data), "out": str(args.out)},
                "train_transitions": len(train_ds),
                "val_transitions": len(val_ds),
            },
            indent=2,
            default=str,
        )
    )
    result = train_loop(
        model,
        train_ds,
        train_cfg,
        C.to_dict(model_cfg),
        args.out,
        mode="wm",
        evaluate=lambda m: evaluate_wm(m, val_ds),
        resume=args.resume,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
