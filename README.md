# arc3-wm: a world-model RL substrate for ARC-AGI-3

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-informational)

`arc3-wm` makes [ARC-AGI-3](https://arcprize.org/tasks) usable as a
standard reinforcement-learning environment, and adds the offline-data
and metric plumbing needed to train and evaluate model-based RL (MBRL)
on it.

It is the substrate behind a study (see [Contributions](#contributions)):
stock DreamerV3 fits these environments yet cannot act in them under a
stock controller at a realistic budget. The wrapper and harness are the
durable artifact and the reason this repository is public. The
[ARC-AGI Living Survey](https://arxiv.org/abs/2603.13372) finds only 3
of around 80 papers reporting an ARC-AGI-3 result, none world-model
based, and names world-model induction as the next step; there was no
Gymnasium-compatible entry point either. This provides one.

> **Status:** research code, pinned for reproducibility, not a
> general-purpose library. The accompanying paper is under review, so
> author and affiliation details are omitted here. See
> [Scope](#scope-and-non-goals).

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

The package also ships `arc3_wm.trm`, a Tiny Recursive Model component
library (discrete world model, behavior-cloning policy, novelty planner,
all composable) with training and evaluation entry points under
`scripts/trm_*.py`; see [docs/trm-components.md](docs/trm-components.md).
Install with `pip install -e ".[trm]"`.

## Install

```bash
pip install -e .                   # the wrapper + Gymnasium path (no JAX)
arc3-wm                            # sanity check: prints version + registered ids (no network)
python scripts/cache_env_files.py  # one-time: cache OFFLINE game files (needs ARC_API_KEY)
                                   # caches the pilot set; pass game ids or --all for others
```

The `arc3-wm` command (equivalently `python -m arc3_wm`) needs no game
files or network and confirms the install is wired up correctly.

For the DreamerV3 training path additionally clone the pinned reference
impl and install its deps. The entry points are
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

## Contributions

The accompanying paper rests on three contributions, given here in order
of durability:

1. **The substrate (primary artifact).** The first Gymnasium- and
   DreamerV3-`embodied`-compatible entry point for ARC-AGI-3, with the
   offline human-replay loader and the RHAE harness. Reusable
   independent of any result.
2. **A controlled negative result.** Stock DreamerV3 (`size12m`, the
   config with direct ARC-AGI-1 precedent, Lee et al. 2024) on 6 public
   games and 2 seeds, trained from scratch (the cold arm) and paired
   against a warm start from a cross-game world model pretrained on all
   340 replays (the warm arm). The pre-registered gate (RHAE > 0 on at
   least 2 of {vc33, sb26, cd82}) failed 1 of 3 at a 500k-step budget:
   only vc33 is ever non-zero, and there the warm-minus-cold delta
   disagrees in sign across the two seeds, so it lies within seed
   variance. Cross-game pretraining yields no measurable benefit.
3. **A mechanistic diagnosis.** The world model fits (image
   reconstruction and dynamics losses collapse to their floors) while
   RHAE stays near zero, and the failure localizes to the policy side.
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

## Status

The substrate (contribution 1) is complete, tested, and reusable. The
negative result and its diagnosis (contributions 2 and 3) are
established across the full 6-game paired sweep, including the two
frozen-model probes that anchor the diagnosis: linear level-identity
decoding from the RSSM latent above raw-pixel and clock controls (real
in cd82 and tn36), and open-loop rollout fidelity versus copy-last-frame
(the model never beats copy). What remains is the controller-side test
that the diagnosis predicts: with the world model held fixed, does
fixing control or exploration close the gap?

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
scripts/        setup, training launchers, baselines, RHAE and table builders, probes, TRM entry points
configs/        DreamerV3 config blocks for the training path
data/           the tracked RHAE baseline fixture (replays are gitignored)
docs/           integration guide, probe protocol, TRM components
analysis/       evidence artifacts backing the paper's result tables
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

The accompanying paper is under review. Its BibTeX entry will be added
here once the review process concludes; until then, please cite the
software artifact by repository name and commit. The manuscript sources
are not kept in this repository, which holds the code artifact only.

## License

MIT; see [LICENSE](LICENSE).
