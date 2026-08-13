#!/usr/bin/env python
"""Train a TRM behaviour-cloning policy on preprocessed replay caches.

Example:
    python scripts/trm_train_bc.py --data data/trm_cache --games vc33 \
        --out checkpoints/trm_bc/vc33 --epochs 30 --seed 0
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
    p.add_argument("--data", type=Path, required=True)
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
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="stop after this many val evaluations without "
                        "improvement (default: TrainConfig's 5)")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--no-mask", action="store_true",
                   help="train over the full unmasked 4102-way action space "
                        "(reference-paper protocol)")
    p.add_argument("--resume", action="store_true", help="continue from out/latest.pt if present")
    p.add_argument("--warm", action="store_true",
                   help="Regime-B warm start from the shared cross-game pretrain at "
                        "<out>/../../pretrain/plan<K>_s<seed>/best.pt (bc when K=1); "
                        "fresh optimizer/EMA. Ignored when --resume finds out/latest.pt")
    p.add_argument("--init-from", type=Path, default=None,
                   help="explicit warm-start checkpoint path (overrides --warm)")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--h-cycles", type=int, default=3)
    p.add_argument("--l-cycles", type=int, default=6)
    p.add_argument("--n-supervision", type=int, default=16)
    p.add_argument("--halt-max-steps", type=int, default=16)
    p.add_argument("--loss", choices=["stablemax_ce", "softmax_ce"], default="stablemax_ce")
    p.add_argument("--seq-mixer", choices=["attention", "mlp"], default="attention",
                   help="token mixer inside the shared net (mlp = MLP-Mixer ablation)")
    p.add_argument("--plan-step0-weight", type=float, default=0.0,
                   help="fraction of plan loss on the executed slot 0 "
                        "(1.0 reduces exactly to plain BC)")
    p.add_argument("--plan-length", type=int, default=1,
                   help=">1: plan-refinement policy (ARC-AGI-2 usage pattern) - "
                        "y carries the next K human actions, halt = whole plan right")
    return p


def build_model_config(args) -> C.PolicyConfig:
    core = C.TRMCoreConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        h_cycles=args.h_cycles,
        l_cycles=args.l_cycles,
        n_supervision=args.n_supervision,
        halt_max_steps=args.halt_max_steps,
        seq_mixer=args.seq_mixer,
    )
    tok = C.TokenizerConfig(d_model=args.d_model, patch_size=args.patch_size)
    return C.PolicyConfig(core=core, tokenizer=tok, loss=args.loss,
                          plan_length=args.plan_length,
                          plan_step0_weight=args.plan_step0_weight)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from arc3_wm.trm.data import BCDataset, train_val_split_episodes
    from arc3_wm.trm.policy import TRMPolicy
    from arc3_wm.trm.training import (
        TrainConfig,
        evaluate_bc,
        init_from_checkpoint,
        resolve_warm_source,
        train_loop,
    )

    paths = []
    for game in args.games:
        path = args.data / f"{game}.npz"
        if not path.exists():
            print(f"missing cache: {path}", file=sys.stderr)
            return 1
        paths.append(path)

    train_specs, val_specs = [], []
    for path in paths:
        tr, va = train_val_split_episodes(path, args.val_fraction, args.seed)
        train_specs.append((path, tr))
        val_specs.append((path, va))
    train_ds = BCDataset(train_specs, use_mask=not args.no_mask,
                         plan_length=args.plan_length)
    val_ds = BCDataset(val_specs, use_mask=not args.no_mask,
                       plan_length=args.plan_length)
    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    model_cfg = build_model_config(args)
    torch.manual_seed(args.seed)
    model = TRMPolicy(model_cfg)
    if args.warm or args.init_from:
        tag = "bc" if args.plan_length == 1 else f"plan{args.plan_length}"
        warm_src = resolve_warm_source(args.out, args.seed, tag, args.init_from)
        if not (args.resume and (args.out / "latest.pt").exists()):
            if not warm_src.exists():
                raise SystemExit(f"warm start source missing: {warm_src}")
            init_from_checkpoint(model, warm_src)
            print(f"warm start from {warm_src}")

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
    )
    if args.early_stop_patience is not None:
        train_cfg.early_stop_patience = args.early_stop_patience
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(
        json.dumps(
            {
                "games": args.games,
                "model_config": C.to_dict(model_cfg),
                "train_config": vars(args) | {"data": str(args.data), "out": str(args.out)},
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
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
        mode="bc",
        evaluate=lambda m: evaluate_bc(m, val_ds),
        resume=args.resume,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
