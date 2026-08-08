"""Training utilities shared by the WM and BC entry points.

Deep supervision follows the paper semantics: each supervision step is one
optimizer step (forward with carried y/z, loss, backward, update, EMA), and
a batch stops early once every sample's halt head fires (subject to the
ACT exploration minimum). The official repo realises the same thing via a
carry persisting across global batches; the explicit loop here is the
transition-level equivalent (documented divergence, config.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from . import config as C
from .core import AdamATan2, EMAHelper, sample_min_halt_steps, warmup_constant_lr


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    ema_decay: float = 0.999
    batch_size: int = 96
    epochs: int = 30
    max_steps: Optional[int] = None
    val_fraction: float = 0.1
    num_workers: int = 4
    device: str = "auto"
    bf16: bool = True
    seed: int = 0
    log_every: int = 50
    eval_every: int = 1  # run the val evaluation every N epochs (and last)
    # When the batch carries a second transition (dataset n_steps=2), add a
    # scheduled-sampling unroll loss: the model's own (detached, argmax)
    # prediction is fed back with the next action and scored against the
    # true two-step-ahead frame. Trains open-loop robustness.
    unroll_weight: float = 0.0
    loss_weights: dict = field(
        default_factory=lambda: {
            "grid": 1.0, "change": 0.5, "reward": 1.0, "state": 0.5,
            "bc": 1.0, "value": 0.2, "halt": 0.5,
        }
    )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return AdamATan2(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


class MetricsLog:
    """Append-only JSONL metrics writer (one dict per line)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMAHelper,
    optimizer: torch.optim.Optimizer,
    model_config: dict,
    step: int,
    extra: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "extra": extra or {},
        },
        path,
    )


def load_wm(path: Path, device: str = "cpu", use_ema: bool = True):
    """Checkpoint -> TRMWorldModel with EMA weights loaded (eval mode)."""
    from .world_model import TRMWorldModel

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = C.from_dict(C.WorldModelConfig, ckpt["model_config"])
    model = TRMWorldModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    if use_ema and ckpt.get("ema"):
        _load_ema_into(model, ckpt["ema"])
    model.eval()
    return model


def load_policy(path: Path, device: str = "cpu", use_ema: bool = True):
    from .policy import TRMPolicy

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = C.from_dict(C.PolicyConfig, ckpt["model_config"])
    model = TRMPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    if use_ema and ckpt.get("ema"):
        _load_ema_into(model, ckpt["ema"])
    model.eval()
    return model


def _load_ema_into(model: nn.Module, ema_state: dict) -> None:
    shadow = ema_state["shadow"]
    own = model.state_dict()
    cast = {k: shadow[k].to(own[k].dtype) for k in own}
    model.load_state_dict(cast)


def deep_supervision_batch(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    ema: EMAHelper,
    cfg: TrainConfig,
    global_step: int,
    mode: str,
    generator: Optional[torch.Generator] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> tuple[dict, int]:
    """Run one batch through up to ``n_supervision`` optimizer steps.

    ``mode``: "wm" or "bc". Returns (mean metric parts, steps consumed).
    """
    core_cfg = model.cfg.core
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    b = batch["grid"].shape[0]
    min_halt = sample_min_halt_steps(b, core_cfg, generator).to(device)

    if mode == "wm":
        x = model.embed(batch["grid"], batch["action"])
    else:
        x = model.embed(batch["grid"])
    carry = None
    agg: dict[str, float] = {}
    steps_done = 0
    for step_i in range(core_cfg.n_supervision):
        lr_scale = warmup_constant_lr(global_step + steps_done, cfg.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * lr_scale
        ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else _nullcontext()
        )
        with ctx:
            if mode == "wm":
                out = model(batch["grid"], batch["action"], carry=carry, x=x)
                parts = model.loss(
                    out,
                    batch["next_grid"],
                    reward=batch.get("reward"),
                    state=batch.get("state"),
                    prev_grid=batch["grid"],
                )
                if cfg.unroll_weight > 0 and "next_grid_2" in batch:
                    with torch.no_grad():
                        fed_back = out.next_logits.argmax(-1)
                    out2 = model(fed_back, batch["action_2"])
                    parts["unroll"] = cfg.unroll_weight * model.loss(
                        out2, batch["next_grid_2"], prev_grid=fed_back
                    )["grid"]
            else:
                out = model(batch["grid"], carry=carry, mask=batch.get("mask"), x=x)
                parts = model.loss(out, batch["action"])
            loss = sum(
                cfg.loss_weights.get(k, 1.0) * v
                for k, v in parts.items()
                if k not in ("exact_match", "accuracy")
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(model)
        carry = out.carry
        steps_done += 1
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + float(v.detach())
        agg["loss"] = agg.get("loss", 0.0) + float(loss.detach())
        with torch.no_grad():
            halted = (out.q_halt > 0) & ((step_i + 1) >= min_halt)
        if bool(halted.all()):
            break
        # The optimizer just stepped: recompute the input embedding under the
        # fresh weights for the next supervision step.
        if mode == "wm":
            x = model.embed(batch["grid"], batch["action"])
        else:
            x = model.embed(batch["grid"])
    return {k: v / steps_done for k, v in agg.items()}, steps_done


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def train_loop(
    model: nn.Module,
    dataset,
    cfg: TrainConfig,
    model_config: dict,
    out_dir: Path,
    mode: str,
    val_dataset=None,
    evaluate: Optional[Callable[[nn.Module], dict]] = None,
    resume: bool = False,
) -> dict:
    """Generic training loop for WM ("wm") and BC ("bc") modes.

    ``resume=True`` continues from ``out_dir/latest.pt`` when present
    (model, EMA, optimizer, step and epoch all restored), so a requeued
    slurm job picks up where it stopped."""
    device = resolve_device(cfg.device)
    model = model.to(device)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    optimizer = make_optimizer(model, cfg)
    ema = EMAHelper(model, decay=cfg.ema_decay)
    log = MetricsLog(out_dir / "metrics.jsonl")
    start_epoch = 0
    resume_step = 0
    latest = out_dir / "latest.pt"
    if resume and latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        resume_step = int(ckpt["step"])
        start_epoch = int(ckpt.get("extra", {}).get("epoch", -1)) + 1
        log.write({"resumed": True, "step": resume_step, "epoch": start_epoch})
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    autocast_dtype = (
        torch.bfloat16 if (cfg.bf16 and device == "cuda") else None
    )
    global_step = resume_step
    batch_i = 0
    best_metric = -1.0
    start = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        for batch in loader:
            parts, consumed = deep_supervision_batch(
                model, batch, optimizer, ema, cfg, global_step, mode,
                generator=gen, autocast_dtype=autocast_dtype,
            )
            global_step += consumed
            batch_i += 1
            if batch_i % cfg.log_every == 0:
                log.write(
                    {"epoch": epoch, "step": global_step,
                     "elapsed": round(time.time() - start, 1),
                     **{k: round(v, 5) for k, v in parts.items()}}
                )
            if cfg.max_steps and global_step >= cfg.max_steps:
                break
        val_metrics = {}
        is_eval_epoch = (
            (epoch + 1) % max(cfg.eval_every, 1) == 0
            or epoch == cfg.epochs - 1
            or (cfg.max_steps and global_step >= cfg.max_steps)
        )
        if evaluate is not None and is_eval_epoch:
            model.eval()
            with ema.swap(model):
                val_metrics = evaluate(model)
            log.write({"epoch": epoch, "step": global_step, "val": val_metrics})
            key = val_metrics.get("exact_match", val_metrics.get("accuracy", 0.0))
            if key > best_metric:
                best_metric = key
                save_checkpoint(
                    out_dir / "best.pt", model, ema, optimizer, model_config,
                    global_step, extra={"val": val_metrics, "epoch": epoch},
                )
        save_checkpoint(
            out_dir / "latest.pt", model, ema, optimizer, model_config,
            global_step, extra={"val": val_metrics, "epoch": epoch},
        )
        if cfg.max_steps and global_step >= cfg.max_steps:
            break
    return {"steps": global_step, "best": best_metric}


@torch.no_grad()
def evaluate_wm(model, dataset, batch_size: int = 64, max_batches: int = 50, rollout_horizon: int = 8) -> dict:
    """WM validation: exact match, per-cell and changed-cell accuracy vs the
    copy-last-frame baseline, plus an open-loop rollout probe."""
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n = correct_cells = total_cells = 0
    changed_correct = changed_total = 0
    copy_correct = 0
    exact = 0
    reward_hits = reward_total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        grid = batch["grid"].to(device)
        nxt = batch["next_grid"].to(device)
        out = model.predict(grid, batch["action"].to(device))
        pred = out.next_logits.argmax(-1)
        n += grid.shape[0]
        correct_cells += int((pred == nxt).sum())
        total_cells += int(nxt.numel())
        exact += int((pred == nxt).flatten(1).all(-1).sum())
        changed = nxt != grid
        changed_correct += int(((pred == nxt) & changed).sum())
        changed_total += int(changed.sum())
        copy_correct += int((grid == nxt).sum())
        if out.reward_logit is not None:
            r = batch["reward"].to(device) > 0
            if int(r.sum()) > 0:
                reward_hits += int(((out.reward_logit > 0) & r).sum())
                reward_total += int(r.sum())
    return {
        "exact_match": exact / max(n, 1),
        "cell_acc": correct_cells / max(total_cells, 1),
        "copy_cell_acc": copy_correct / max(total_cells, 1),
        "changed_cell_acc": changed_correct / max(changed_total, 1),
        "changed_cell_frac": changed_total / max(total_cells, 1),
        "reward_recall": reward_hits / max(reward_total, 1) if reward_total else None,
        "n": n,
    }


@torch.no_grad()
def evaluate_bc(model, dataset, batch_size: int = 64, max_batches: int = 50) -> dict:
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n = top1 = 0
    type_correct = 0
    from .tokenizer import flat_action_components

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        grid = batch["grid"].to(device)
        action = batch["action"].to(device)
        pred, _ = model.act(grid, mask=batch["mask"].to(device), temperature=0.0)
        n += grid.shape[0]
        top1 += int((pred == action).sum())
        p_type, _, _ = flat_action_components(pred)
        t_type, _, _ = flat_action_components(action)
        type_correct += int((p_type == t_type).sum())
    return {
        "accuracy": top1 / max(n, 1),
        "type_accuracy": type_correct / max(n, 1),
        "n": n,
    }
