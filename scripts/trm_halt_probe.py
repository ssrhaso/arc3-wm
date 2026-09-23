#!/usr/bin/env python
"""ACT halting probe: does the trained halt head know when to stop?

Runs every supervision step up to the halt cap WITHOUT early stopping and
records, per step: the fraction of val samples whose halt logit fires
(q > 0), prediction quality, and the halt/correctness calibration
P(correct | halted) vs P(correct | not halted). Also reports the
per-sample first-halt distribution and the step at which the batch-level
``all halted`` criterion (what predict()/act() actually use) would fire.

Usage:
    python scripts/trm_halt_probe.py --data <cache> --game vc33 \
        --ckpt <wm_or_bc>/best.pt --kind wm|bc [--steps 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--game", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--kind", choices=["wm", "bc"], required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    import torch

    from arc3_wm.trm.data import BCDataset, WMTransitionDataset, train_val_split_episodes
    from arc3_wm.trm.training import load_policy, load_wm, resolve_device

    npz = args.data / f"{args.game}.npz"
    _, va = train_val_split_episodes(npz, args.val_fraction, args.seed)
    device = resolve_device(args.device)
    if args.kind == "wm":
        model = load_wm(args.ckpt, device=device)
        dataset = WMTransitionDataset([(npz, va)], dedup=False)
    else:
        model = load_policy(args.ckpt, device=device)
        dataset = BCDataset([(npz, va)])
    n = min(len(dataset), args.max_samples)
    loader = torch.utils.data.DataLoader(
        [dataset[i] for i in range(n)], batch_size=args.batch_size
    )

    steps = args.steps
    halted_at = []  # per-sample first-halt step (steps+1 = never)
    correct_at_halt = []
    per_step = {k: {"halted": 0, "correct": 0, "halted_and_correct": 0, "n": 0}
                for k in range(1, steps + 1)}
    all_halt_steps = []

    with torch.no_grad():
        for batch in loader:
            b = batch["grid"].shape[0]
            if args.kind == "wm":
                x = model.embed(batch["grid"].to(device), batch["action"].to(device))
            else:
                x = model.embed(batch["grid"].to(device))
            carry = None
            first_halt = np.full(b, steps + 1)
            first_correct = np.zeros(b, dtype=bool)
            batch_all_step = steps + 1
            for k in range(1, steps + 1):
                if args.kind == "wm":
                    out = model(batch["grid"].to(device), batch["action"].to(device),
                                carry=carry, x=x)
                    correct = (
                        out.next_logits.argmax(-1)
                        == batch["next_grid"].to(device).long()
                    ).flatten(1).all(-1).cpu().numpy()
                else:
                    out = model(batch["grid"].to(device), carry=carry,
                                mask=batch["mask"].to(device), x=x)
                    correct = (
                        out.flat_logits.argmax(-1) == batch["action"].to(device)
                    ).cpu().numpy()
                carry = out.carry
                halted = (out.q_halt > 0).cpu().numpy()
                stat = per_step[k]
                stat["halted"] += int(halted.sum())
                stat["correct"] += int(correct.sum())
                stat["halted_and_correct"] += int((halted & correct).sum())
                stat["n"] += b
                newly = halted & (first_halt > steps)
                first_halt[newly] = k
                first_correct[newly] = correct[newly]
                if halted.all() and batch_all_step > steps:
                    batch_all_step = k
            halted_at.extend(first_halt.tolist())
            correct_at_halt.extend(first_correct[first_halt <= steps].tolist())
            all_halt_steps.append(int(batch_all_step))

    halted_at = np.asarray(halted_at)
    report = {
        "game": args.game, "kind": args.kind, "ckpt": str(args.ckpt),
        "samples": int(len(halted_at)),
        "per_step": {
            k: {
                "halt_frac": round(v["halted"] / v["n"], 4),
                "correct_frac": round(v["correct"] / v["n"], 4),
                "p_correct_given_halt": round(
                    v["halted_and_correct"] / v["halted"], 4
                ) if v["halted"] else None,
            }
            for k, v in per_step.items()
        },
        "first_halt_hist": {
            str(k): int((halted_at == k).sum()) for k in range(1, steps + 2)
        },
        "never_halt_frac": round(float((halted_at > steps).mean()), 4),
        "p_correct_at_first_halt": round(float(np.mean(correct_at_halt)), 4)
        if correct_at_halt else None,
        "batch_all_halt_steps": all_halt_steps,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
