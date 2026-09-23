"""PPO baseline on the ARC-AGI-3 Gymnasium substrate.

Model-free control for the DreamerV3 arms: same games, same 4102-way flat
action space (or the factored ``(type, x, y)`` head of the Track-B study),
same native ``r = delta levels_completed`` reward, same post-hoc scoring.
CleanRL-style PPO (Schulman et al. 2017) with the Nature-CNN encoder, an
entropy bonus and the Atari defaults; nothing here is tuned to the games.

Design notes
* Environments are stepped sequentially in one process with manual resets
  (an episode that ends is reset immediately and the reset observation is
  the next observation), which sidesteps gymnasium's vector autoreset
  semantics. The env is CPU-bound at about 2k steps per second, so this is
  as fast as a vector env here.
* ``--head flat`` samples one categorical over 4102 indices. ``--head
  factored`` samples type (7-way), x (64-way) and y (64-way) independently
  and maps them onto the same flat index through ``arc3_wm.action_space``,
  so the executed action space is identical across heads.
* Entropy is logged both in nats and normalized by its uniform ceiling
  (``ln 4102`` for flat, ``ln 7 + 2 ln 64`` for factored), matching the
  ``train/rand/action`` reading used for DreamerV3.
* After training, ``--eval-episodes`` episodes are played with the frozen
  stochastic policy and written to ``{logdir}/eval100/eval_episodes.jsonl``
  in the EvalRewardSink convention (``rewards[0]`` is the reset step), so
  ``scripts/compute_rhae.py`` and the official scorer apply unchanged. The
  evaluation env sets ``full_reset``, so every episode is a complete attempt
  from level 1, matching the DreamerV3 arms.

Torch is imported inside ``main``; the helpers at module level are
laptop-testable without it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("OPERATION_MODE", "offline")

N_FLAT = 4102
N_TYPES = 7
GRID = 64
TYPE_INDEX_OF_FLAT = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 4101: 6}


def factored_to_flat(type_idx: int, x: int, y: int) -> int:
    """(type_idx in 0..6 for ACTION1..7, x, y) -> flat index, via arc3_wm.action_space."""
    from arcengine import GameAction

    from arc3_wm.action_space import arc_to_flat

    action = GameAction[f"ACTION{int(type_idx) + 1}"]  # by name: GameAction(6) is rejected by the enum
    if type_idx == 5:  # ACTION6 carries coordinates
        return int(arc_to_flat(action, int(x), int(y)))
    return int(arc_to_flat(action))


def flat_to_factored(idx: int) -> tuple[int, int, int]:
    """Inverse of :func:`factored_to_flat`; non-click actions get x = y = 0."""
    from arc3_wm.action_space import flat_to_arc

    action, data = flat_to_arc(int(idx))
    t = action.value - 1
    if data is None:
        return t, 0, 0
    return t, int(data["x"]), int(data["y"])


def entropy_ceiling(head: str) -> float:
    if head == "flat":
        return math.log(N_FLAT)
    if head == "factored":
        return math.log(N_TYPES) + 2 * math.log(GRID)
    raise ValueError(head)


def eval_stream_line(rewards: Sequence[float]) -> str:
    """One EvalRewardSink-format line: reset step 0.0 then the per-action rewards."""
    return json.dumps({"rewards": [0.0] + [float(r) for r in rewards]})


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", required=True)
    p.add_argument("--logdir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head", choices=["flat", "factored"], default="flat")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=1000, help="env truncation, matches the DreamerV3 runs")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--log-every-updates", type=int, default=5)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# torch part
# ---------------------------------------------------------------------------

def build_agent(head: str):
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    def layer_init(layer, std=math.sqrt(2), bias=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)
        return layer

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)), nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 4 * 4, 512)), nn.ReLU(),
            )
            self.critic = layer_init(nn.Linear(512, 1), std=1.0)
            if head == "flat":
                self.pi = layer_init(nn.Linear(512, N_FLAT), std=0.01)
            else:
                self.pi_t = layer_init(nn.Linear(512, N_TYPES), std=0.01)
                self.pi_x = layer_init(nn.Linear(512, GRID), std=0.01)
                self.pi_y = layer_init(nn.Linear(512, GRID), std=0.01)

        def features(self, obs_uint8):
            x = obs_uint8.permute(0, 3, 1, 2).float() / 255.0
            return self.encoder(x)

        def value(self, obs_uint8):
            return self.critic(self.features(obs_uint8)).squeeze(-1)

        def act(self, obs_uint8, action=None):
            """Returns (action tensor (B, k), logprob (B,), entropy (B,), value (B,))."""
            h = self.features(obs_uint8)
            v = self.critic(h).squeeze(-1)
            if self.head == "flat":
                dist = Categorical(logits=self.pi(h))
                if action is None:
                    action = dist.sample().unsqueeze(-1)
                a = action[:, 0]
                return action, dist.log_prob(a), dist.entropy(), v
            dt = Categorical(logits=self.pi_t(h))
            dx = Categorical(logits=self.pi_x(h))
            dy = Categorical(logits=self.pi_y(h))
            if action is None:
                action = torch.stack([dt.sample(), dx.sample(), dy.sample()], dim=-1)
            lp = dt.log_prob(action[:, 0]) + dx.log_prob(action[:, 1]) + dy.log_prob(action[:, 2])
            ent = dt.entropy() + dx.entropy() + dy.entropy()
            return action, lp, ent, v

    return Agent()


def to_flat_actions(head: str, action_np: np.ndarray) -> list[int]:
    if head == "flat":
        return [int(a[0]) for a in action_np]
    return [factored_to_flat(int(a[0]), int(a[1]), int(a[2])) for a in action_np]


class EnvPool:
    """Sequential pool of ARC3GymEnv with manual same-step resets and episode stats."""

    def __init__(self, game: str, seeds: Sequence[int], max_steps: int):
        from arc3_wm.env import ARC3GymEnv

        self.envs = [ARC3GymEnv(game_id=game, seed=int(s), max_steps=max_steps) for s in seeds]
        self.obs = np.stack([e.reset(seed=int(s))[0] for e, s in zip(self.envs, seeds)])
        self.ep_return = np.zeros(len(self.envs), dtype=np.float64)
        self.ep_length = np.zeros(len(self.envs), dtype=np.int64)
        self.finished = []  # (return, length)

    def step(self, flat_actions: Sequence[int]):
        rewards = np.zeros(len(self.envs), dtype=np.float32)
        dones = np.zeros(len(self.envs), dtype=np.float32)
        next_obs = np.empty_like(self.obs)
        for i, (env, a) in enumerate(zip(self.envs, flat_actions)):
            obs, r, term, trunc, _ = env.step(int(a))
            rewards[i] = r
            self.ep_return[i] += r
            self.ep_length[i] += 1
            if term or trunc:
                dones[i] = 1.0
                self.finished.append((float(self.ep_return[i]), int(self.ep_length[i])))
                self.ep_return[i] = 0.0
                self.ep_length[i] = 0
                obs, _ = env.reset()
            next_obs[i] = obs
        self.obs = next_obs
        return next_obs, rewards, dones

    def close(self):
        for e in self.envs:
            e.close()


def evaluate(agent, head: str, game: str, seed: int, n_episodes: int, max_steps: int, out_path: Path, device):
    """Frozen stochastic policy, one env, one episode at a time; sink-format JSONL."""
    import torch

    from arc3_wm.env import ARC3GymEnv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # full_reset makes "every episode is a fresh game from level 1" explicit. Passing a
    # new seed per episode below already forces a rebuild and so achieves the same thing,
    # but relying on that side effect hid the protocol: the engine's plain RESET resumes
    # the current level once one has been cleared, which is what made the DreamerV3
    # evaluations understate their clear rates before 2026-09-06.
    env = ARC3GymEnv(game_id=game, seed=seed, max_steps=max_steps, full_reset=True)
    clears = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            rewards = []
            while True:
                with torch.no_grad():
                    action, _, _, _ = agent.act(torch.as_tensor(obs[None], device=device))
                a = to_flat_actions(head, action.cpu().numpy())[0]
                obs, r, term, trunc, _ = env.step(a)
                rewards.append(float(r))
                if term or trunc:
                    break
            clears += int(sum(rewards) > 0)
            f.write(eval_stream_line(rewards) + "\n")
    env.close()
    return clears


def main(argv=None):
    import torch
    import torch.nn as nn

    args = parse_args(argv)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "config.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    print(f"game={args.game} head={args.head} seed={args.seed} device={device} total_steps={args.total_steps}", flush=True)

    agent = build_agent(args.head).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    pool = EnvPool(args.game, [args.seed * 1000 + i for i in range(args.num_envs)], args.max_steps)
    k = 1 if args.head == "flat" else 3
    T, N = args.num_steps, args.num_envs
    obs_buf = torch.zeros((T, N, GRID, GRID, 3), dtype=torch.uint8, device=device)
    act_buf = torch.zeros((T, N, k), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)

    batch_size = T * N
    num_updates = args.total_steps // batch_size
    global_step = 0
    start = time.time()
    metrics = (logdir / "metrics.jsonl").open("a", encoding="utf-8")
    next_obs = torch.as_tensor(pool.obs, device=device)
    next_done = torch.zeros(N, device=device)
    ceiling = entropy_ceiling(args.head)
    train_clears = 0
    episodes_seen = 0

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        for g in opt.param_groups:
            g["lr"] = frac * args.lr
        for t in range(T):
            global_step += N
            obs_buf[t] = next_obs
            done_buf[t] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.act(next_obs)
            act_buf[t] = action
            logp_buf[t] = logp
            val_buf[t] = value
            nobs, rew, done = pool.step(to_flat_actions(args.head, action.cpu().numpy()))
            rew_buf[t] = torch.as_tensor(rew, device=device)
            next_obs = torch.as_tensor(nobs, device=device)
            next_done = torch.as_tensor(done, device=device)
        # GAE
        with torch.no_grad():
            next_value = agent.value(next_obs)
            adv = torch.zeros_like(rew_buf)
            lastgaelam = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nonterminal = 1.0 - (next_done if t == T - 1 else done_buf[t + 1])
                nv = next_value if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * nonterminal - val_buf[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nonterminal * lastgaelam
                adv[t] = lastgaelam
            returns = adv + val_buf
        b_obs = obs_buf.reshape(batch_size, GRID, GRID, 3)
        b_act = act_buf.reshape(batch_size, k)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)
        idx = np.arange(batch_size)
        mb = batch_size // args.minibatches
        stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": []}
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, batch_size, mb):
                m = idx[s:s + mb]
                _, newlogp, ent, newv = agent.act(b_obs[m], b_act[m])
                ratio = (newlogp - b_logp[m]).exp()
                with torch.no_grad():
                    stats["kl"].append(((ratio - 1) - ratio.log()).mean().item())
                    stats["clipfrac"].append(((ratio - 1.0).abs() > args.clip).float().mean().item())
                a_ = b_adv[m]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_clip = b_val[m] + (newv - b_val[m]).clamp(-args.clip, args.clip)
                vf = 0.5 * torch.max((newv - b_ret[m]) ** 2, (v_clip - b_ret[m]) ** 2).mean()
                loss = pg - args.ent_coef * ent.mean() + args.vf_coef * vf
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
                stats["pg"].append(pg.item()); stats["vf"].append(vf.item()); stats["ent"].append(ent.mean().item())
        new_eps = pool.finished[episodes_seen:]
        episodes_seen = len(pool.finished)
        train_clears += sum(1 for r, _ in new_eps if r > 0)
        if update % args.log_every_updates == 0 or update == num_updates:
            ent_mean = float(np.mean(stats["ent"]))
            rec = dict(update=update, step=global_step, fps=global_step / (time.time() - start),
                       entropy=ent_mean, norm_entropy=ent_mean / ceiling,
                       pg_loss=float(np.mean(stats["pg"])), vf_loss=float(np.mean(stats["vf"])),
                       approx_kl=float(np.mean(stats["kl"])), clipfrac=float(np.mean(stats["clipfrac"])),
                       episodes=episodes_seen, train_clears=train_clears,
                       mean_return=float(np.mean([r for r, _ in pool.finished[-50:]])) if pool.finished else 0.0,
                       mean_length=float(np.mean([l for _, l in pool.finished[-50:]])) if pool.finished else 0.0)
            metrics.write(json.dumps(rec) + "\n"); metrics.flush()
            print(f"[{global_step}] fps={rec['fps']:.0f} norm_ent={rec['norm_entropy']:.3f} ent={ent_mean:.2f} "
                  f"pg={rec['pg_loss']:.4f} vf={rec['vf_loss']:.4f} eps={episodes_seen} clears={train_clears}", flush=True)
    metrics.close()
    pool.close()
    torch.save(agent.state_dict(), logdir / "agent_final.pt")
    clears = evaluate(agent, args.head, args.game, args.seed + 1000, args.eval_episodes, args.max_steps,
                      logdir / "eval100" / "eval_episodes.jsonl", device)
    summary = dict(game=args.game, head=args.head, seed=args.seed, total_steps=global_step,
                   train_episodes=episodes_seen, train_clears=train_clears,
                   eval_episodes=args.eval_episodes, eval_clears=clears,
                   wallclock_s=time.time() - start)
    (logdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("PPO-DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
