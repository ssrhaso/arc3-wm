# arc3-wm: a world-model RL substrate for ARC-AGI-3

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-informational)

`arc3-wm` makes [ARC-AGI-3](https://arcprize.org/tasks) usable as a
standard reinforcement-learning environment, and adds the offline-data
and metric plumbing needed to train and evaluate model-based RL (MBRL)
on it.

It is the substrate behind a NeurIPS-2026-workshop study, the first
model-based RL entry on ARC-AGI-3. The study's result is a diagnosed
negative one (see [Contribution](#contribution)): stock DreamerV3 fits
these environments yet cannot act in them under a stock controller at a
realistic budget. The wrapper and harness are the durable artifact and
the reason this repo is public. The
[ARC-AGI Living Survey](https://arxiv.org/abs/2603.13372) finds only 3
of around 80 papers reporting an ARC-AGI-3 result, none world-model
based, and names world-model induction as the next step; there was no
Gymnasium-compatible entry point either. This provides one.

> **Status:** research code, pinned for reproducibility, not a
> general-purpose library. See [Scope](#scope-and-non-goals).

## What you get

| Component                            | Module                                   | What it is                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Gymnasium env**              | `arc3_wm.env:ARC3GymEnv`               | One ARC-AGI-3 game as a stock `gymnasium.Env`. `Box(0,255,(64,64,3),uint8)` obs, `Discrete(4102)` flat action space, level-up reward. Pure-Python, no JAX.                                                                                                                                                                                                                      |
| **DreamerV3 `embodied` env** | `arc3_wm.embodied_env:ARC3EmbodiedEnv` | The same game behind DreamerV3's `embodied.Env` duck-typed interface, no fork of `dreamerv3`.                                                                                                                                                                                                                                                                                     |
| **Flat action space**          | `arc3_wm.action_space`                 | Bijective `idx <-> (ACTION_TYPE, x, y)` over the 4102-way space, plus per-step boolean masks.                                                                                                                                                                                                                                                                                       |
| **Offline replay loader**      | `arc3_wm.replay_loader`                | The 340-replay human-demonstration JSONL dataset -> transition tuples for a world-model buffer.                                                                                                                                                                                                                                                                                       |
| **RHAE metric**                | `arc3_wm.rhae`                         | Post-hoc Relative Human Action Efficiency, the benchmark metric. Per level `s_i = min((human/ai)^2, 1.15)`; per game the level-index-weighted mean over all levels, including uncompleted ones; the total averages games equally. Baselines are the upper-median of first-time-player action counts, dropping levels with fewer than 2 completers (70.5% coverage, 129/183 levels). |

The two interfaces are the contribution: anything that speaks Gymnasium
or DreamerV3-`embodied` plugs in with no `arc3_wm` changes. The standard
interface is the integration point; there is no custom abstraction
layer. See [docs/using-the-wrapper.md](docs/using-the-wrapper.md).

This fork adds `arc3_wm.trm`, a Tiny Recursive Model component library
(discrete world model, behaviour-cloning policy, novelty planner - all
composable) with training/eval entry points under `scripts/trm_*.py`;
see [docs/trm-components.md](docs/trm-components.md). Install with
`pip install -e ".[trm]"`.

## Install

```bash
pip install -e .                   # the wrapper + Gymnasium path (no JAX)
arc3-wm                            # sanity check: prints version + registered ids (no network)
python scripts/cache_env_files.py  # one-time: cache OFFLINE game files (needs ARC_API_KEY)
                                   # caches the Phase-4 set; pass game ids or --all for others
```

The `arc3-wm` command (equivalently `python -m arc3_wm`) needs no game
files or network and confirms the install is wired up correctly.

For the DreamerV3 training path additionally clone the pinned reference
impl and install its deps; see
[docs/vast-quickstart.md](docs/vast-quickstart.md). The entry points are
[scripts/launch_pergame.py](scripts/launch_pergame.py) and the config
blocks in [configs/](configs/).

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
mask = info["action_mask"]                       # length-4102 bool; arc3_wm.logit_bias(mask) -> additive -inf bias for your policy logits
```

Or via the registered Gymnasium id: `import arc3_wm` self-registers
`ARC3/<game>-v0` for all 25 public games, so any gym-id tooling reaches
ARC-AGI-3 with no `arc3_wm` symbol in the loop:

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
   games and 2 seeds, paired from-scratch (Regime A) against warm-start
   from a cross-game world model pretrained on all 340 replays
   (Regime B). The pre-registered gate (RHAE > 0 on at least 2 of
   {vc33, sb26, cd82}) failed 1 of 3 at a 500k-step budget: only vc33 is
   ever non-zero, and there the warm-minus-cold delta disagrees in sign
   across the two seeds, i.e. lies within seed variance. Cross-game
   pretraining yields no measurable benefit.
3. **A mechanistic diagnosis.** The world model fits (image
   reconstruction and dynamics losses collapse to their floors) while
   RHAE stays near zero, and the failure localises to the policy side.
   On 5 of the 6 games the actor never commits: policy entropy stays
   pinned at the uniform maximum `ln(4102) = 8.32` nats and episodic
   return is identically zero for the full 500k-step budget, because the
   sparse reward (change in levels completed) emits no gradient until a
   level is cleared. The dissociation is sharp: cd82's world model fits
   tighter than vc33's (image loss 0.06 vs. 0.16), yet vc33 is the only
   game whose controller ever clears a level.

The RHAE benchmark reports each game independently level-weighted and
then combined (see [`arc3_wm/rhae.py`](arc3_wm/rhae.py) and
[scripts/build_benchmark_table.py](scripts/build_benchmark_table.py)).

## Status and next steps

The substrate (contribution 1) is complete, tested, and reusable. The
negative result and its diagnosis (contributions 2 and 3) are
established across the full 6-game paired sweep, including the two
frozen-model probes that anchor the diagnosis: linear level-identity
decoding from the RSSM latent above raw-pixel and clock controls (real
in cd82 and tn36), and open-loop rollout fidelity versus copy-last-frame
(the model never beats copy). What remains is the controller-side test
that the diagnosis predicts: with the world model held fixed, does
fixing control or exploration close the gap?

### Directions for follow-on work

Five directions for two interns, each one well-defined experiment on the
existing pipeline (not a refactor):

1. **Controller-side intervention, world model frozen (primary).** The
   direct "does fixing control close the gap" test. (a) On ls20, engage
   the per-game mask (already in `info["action_mask"]`) to collapse 4102
   actions to 4 and rerun. (b) On the no-gradient games (sb26, cd82,
   tn36, lf52), swap the stock actor for Plan2Explore, whose exploration
   needs no extrinsic reward. Now viable post the counterfactual fix
   (ad1ee4b: dynamics are weakly action-sensitive, not blind), but gate
   it behind a short action-conditionality diagnostic before the full
   sweep. Higher-variance; feeds the follow-up paper.
2. **Second world-model backend.** The substrate's Gymnasium env is the
   connector; any WM that consumes an image-obs/discrete-action Gym env
   plugs in via a small adapter (the `embodied` path is DreamerV3-only).
   Two flavors, two questions. *Same imagination actor, different WM*
   (is the failure generic, not a Dreamer quirk?): the Atari-lineage
   transformer/diffusion models, IRIS, DIAMOND, STORM, or TWISTER, all
   64x64 image obs with their own actor-critic. *Different controller*
   (does planning escape where the imagination actor stalls?): TD-MPC2
   (MPPI), EfficientZero/MuZero (MCTS), or a JEPA model (DINO-WM, LeWM)
   with an MPC planner. DIAMOND or STORM are the cleanest first cut; the
   4102-way action space is the main porting caveat (these assume small
   Atari action sets). Deliverable: a justified model choice plus the
   same 6-game RHAE table.
3. **Scale the DreamerV3 sweep.** Extend the 6-game sweep to all 25 (or a
   difficulty-stratified subset) at the same budget, to see whether vc33's
   weak non-zero is the ceiling. Pure plumbing on the launcher, no new
   code; embarrassingly parallel, low-variance, good first task.
4. **Prior-matched transfer.** The warm arm pooled all games by data
   volume and bought nothing. Cluster games by shared core-knowledge
   prior (objectness, contact, agentness), pretrain within a cluster, and
   eval zero/few-shot on held-out games inside vs. outside it; a
   volume-matched null cluster separates structured transfer from
   more-data. Higher-variance.
5. **Probe transferred priors on held-out games.** Extend the linear-probe
   protocol (`analysis/`) to a game the WM never trained on: above-control
   decoding of object identity, position, and contact events is the
   representational fingerprint of transfer, isolated from control.
   Reuses the frozen-model harness; compute-light, good first task.

Suggested split: one intern takes the low-variance tasks (3, 5), the
other the higher-variance controller and transfer work (1, 4); 2 is
shared scaffolding either can pick up.

## Scope and non-goals

Pinned research code, not a framework:

- No custom encoder, no DreamerV3 fork, no second world-model backend,
  no intrinsic-motivation or reward-shaping; all are explicit non-goals
  (follow-up work). The wrapper exposes standard interfaces and stops
  there; it is not generalized for speculative future use.
- Dependencies are version-pinned for reproducibility of the reported
  numbers, not for breadth.

## Repository map

```
arc3_wm/        the package (env, embodied_env, action_space, replay_loader, rhae, palette, registration)
examples/       runnable, laptop-only demos of the standard interface
scripts/        env-file cache, per-game launcher, RHAE + benchmark builders
configs/        DreamerV3 config blocks for the training path
data/           the tracked RHAE baseline fixture (replays are gitignored)
docs/           integration guide, contribution skeleton, compute runbook
analysis/       evidence artifacts backing the paper's result tables
figures/        diagnosis figures backing the analysis docs
tests/          property + integration tests (the spec)
```

Most of these directories carry their own `README.md` with a fuller index.

## Development

Install the dev extras and run the suite (the tests are the spec):

```bash
pip install -e ".[dev]"
pytest                                 # full suite
pytest tests/test_action_space.py -q   # a single module
pytest -n auto                         # parallel (pytest-xdist)
```

The common commands above are also available as `make` targets
(`make dev`, `make test`, `make test-fast`, `make check`, `make smoke`,
`make gym-smoke`, `make clean`); run `make help` for the full list.

The pure-Python and Gymnasium tests run on a laptop with no GPU and no
JAX; a few `embodied`/DreamerV3 tests skip automatically when the
JAX-side `elements` dependency is absent. Env tests read cached OFFLINE
game files from `environment_files/`; run
`python scripts/cache_env_files.py` once (needs `ARC_API_KEY`) if they
are missing.

## Citation

If you use this software, cite it via the metadata in
[CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button
reads it). The workshop-paper BibTeX is added on submission; see
[docs/contribution.md](docs/contribution.md).

## License

MIT; see [LICENSE](LICENSE). (c) 2026 Hasaan Ahmad.
