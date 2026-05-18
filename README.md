# arc3-wm - a world-model RL substrate for ARC-AGI-3

`arc3-wm` is a small, dependency-light Python package that makes
[ARC-AGI-3](https://arcprize.org/tasks) usable as a standard
reinforcement-learning environment, plus the offline-data and metric
plumbing needed to train and evaluate **model-based RL (MBRL) / world
models** on it.

It is the substrate behind a NeurIPS-2026-workshop study. The study's
finding is a *diagnosed negative result* (see
[Contribution](#contribution)); the **wrapper and harness are the
durable artifact** and the reason this repo is public: as of writing,
the [ARC-AGI Living Survey](https://arxiv.org/abs/2603.13372) records
*zero* world-model approaches on ARC-AGI-3, and there was no
Gymnasium-compatible entry point. This provides one.

> **Status:** research code, pinned for reproducibility — not a
> general-purpose library. It does one thing (ARC-AGI-3 → standard RL
> interfaces) and does it cleanly. See [Scope](#scope-and-non-goals).

## What you get

| Component                            | Module                                   | What it is                                                                                                                                                       |
| ------------------------------------ | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Gymnasium env**              | `arc3_wm.env:ARC3GymEnv`               | One ARC-AGI-3 game as a stock `gymnasium.Env`. `Box(0,255,(64,64,3),uint8)` obs, `Discrete(4102)` flat action space, level-up reward. Pure-Python, no JAX. |
| **DreamerV3 `embodied` env** | `arc3_wm.embodied_env:ARC3EmbodiedEnv` | The same game behind DreamerV3's `embodied.Env` duck-typed interface, no fork of `dreamerv3`.                                                                |
| **Flat action space**          | `arc3_wm.action_space`                 | Bijective `idx ↔ (ACTION_TYPE, x, y)` over the 4102-way space, plus per-step boolean masks.                                                                   |
| **Offline replay loader**      | `arc3_wm.replay_loader`                | The 340-replay human-demonstration JSONL dataset → transition tuples for a world-model buffer.                                                                  |
| **RHAE metric**                | `arc3_wm.rhae`                         | Post-hoc Relative Human Action Efficiency: per-game, level-index-weighted, combined across games. The benchmark metric.                                          |

The two interfaces are the contribution: **anything that speaks
Gymnasium or DreamerV3-`embodied` plugs in with no `arc3_wm` changes.**
There is deliberately no custom abstraction layer — the standard
interface *is* the integration point. See
[docs/using-the-wrapper.md](docs/using-the-wrapper.md).

## Install

```bash
pip install -e .                   # the wrapper + Gymnasium path (no JAX)
python scripts/cache_env_files.py  # one-time: cache OFFLINE game files (needs ARC_API_KEY)
```

For the DreamerV3 training path additionally clone the pinned reference
impl and install its deps — see
[docs/vast-quickstart.md](docs/vast-quickstart.md).

## 60-second quickstart (Gymnasium, laptop, no GPU)

```bash
python examples/random_agent.py --game vc33 --episodes 3
```

```python
import arc_agi
from arc3_wm.env import ARC3GymEnv

arcade = arc_agi.Arcade()                        # OFFLINE mode (set in .env)
env = ARC3GymEnv(game_id="vc33", arcade=arcade)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
mask = info["action_mask"]                       # length-4102 bool, apply to your policy logits
```

Or via the registered Gymnasium id — `import arc3_wm` self-registers
`ARC3/<game>-v0` for all 25 public games, so any tooling that resolves
a gym id reaches ARC-AGI-3 with no `arc3_wm` symbol in the loop:

```python
import gymnasium as gym
import arc3_wm                                   # registers ARC3/<game>-v0
env = gym.make("ARC3/vc33-v0")                   # see examples/gym_make.py
```

## Contribution

This repo backs a workshop-paper extension with three pillars, in order
of durability:

1. **The substrate (primary artifact).** The first Gymnasium- and
   DreamerV3-`embodied`-compatible entry point for ARC-AGI-3, with the
   offline human-replay loader and the RHAE harness. Reusable
   independent of any result.
2. **A controlled negative result.** Stock DreamerV3 (`size12m`, the
   config with direct ARC-1 precedent, Lee et al. 2024) on 6 public
   games × 2 seeds × paired {from-scratch, cross-game-pretrained}
   gains ~no traction at a 500k-step budget, and offline world-model
   pretraining from human demos gives no measurable lift (paired Δ
   within seed variance).
3. **A mechanistic diagnosis.** The world model *fits* (image-recon and
   dynamics losses collapse) while RHAE stays ~0 — the model predicts
   the world; the controller cannot exploit it in imagination.
   Latent-probe / FVD / reasoning-axis diagnostics localize where this
   breaks.

The RHAE benchmark reports each game **independently level-weighted**
and then combined (see [`arc3_wm/rhae.py`](arc3_wm/rhae.py) and
[scripts/build_benchmark_table.py](scripts/build_benchmark_table.py)).

## Scope and non-goals

This is pinned research code, not a framework:

- No custom encoder, no DreamerV3 fork, no second world-model backend,
  no intrinsic-motivation/reward-shaping — all are explicit non-goals
  (follow-up work). The wrapper is intentionally *not* generalized for
  speculative future use; it exposes standard interfaces and stops
  there.
- Dependencies are version-pinned for reproducibility of the reported
  numbers, not for breadth.

## Repository map

```
arc3_wm/        the package (env, embodied_env, action_space, replay_loader, rhae)
examples/       runnable, laptop-only demos of the standard interface
scripts/        env-file cache, per-game launcher, RHAE + benchmark builders
docs/           integration guide, contribution skeleton, compute runbook
tests/          property + integration tests (the spec)
```

## Citation

See [docs/contribution.md](docs/contribution.md) for the paper
skeleton and BibTeX (added on submission).

## License

Not yet set — see [docs/contribution.md](docs/contribution.md). A
license must be added before this is published as a citable artifact.
