# Using the wrapper — integrating an RL / world-model method

This guide is for someone who wants to run **their own** agent or
world model on ARC-AGI-3 using `arc3_wm`. The whole point of the
package is that you do **not** modify it: it exposes two standard
interfaces and your method attaches to one of them.

There is intentionally **no `arc3_wm` adapter/plugin abstraction**. A
bespoke "world-model interface" would be a second, non-standard thing
to learn and would rot. Instead the integration seam is an interface
your framework almost certainly already supports:

- **Gymnasium** (`arc3_wm.env:ARC3GymEnv`) — for anything Gym-based.
- **DreamerV3 `embodied`** (`arc3_wm.embodied_env:ARC3EmbodiedEnv`) —
  for the `embodied` ecosystem (Driver / Replay / wrappers).

If your method consumes either, ARC-AGI-3 is already plug-and-play.

## Path A — Gymnasium (most methods)

`ARC3GymEnv` is a stock `gymnasium.Env`. No special casing:

```python
import arc_agi
from arc3_wm.env import ARC3GymEnv

arcade = arc_agi.Arcade()                 # OFFLINE mode; see "Prerequisites"
env = ARC3GymEnv(game_id="vc33", seed=0, max_steps=1000, arcade=arcade)

obs, info = env.reset(seed=0)
done = False
while not done:
    action = policy(obs)                  # your agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

Contract:

| Field | Value | Notes |
|---|---|---|
| `observation_space` | `Box(0, 255, (64, 64, 3), uint8)` | Palette-decoded last frame, HWC. |
| `action_space` | `Discrete(4102)` | Flat layout — see below. |
| `reward` | `Δ levels_completed` | Native level-up signal, unmodified. |
| `terminated` | `state ∈ {WIN, GAME_OVER}` | True task end. |
| `truncated` | `steps == max_steps` | Timeout, not failure. |
| `info["action_mask"]` | length-4102 `bool` ndarray | Valid actions this step. **Exposed, not enforced** — apply it to your policy logits; `arc_agi` silently no-ops invalid actions if you don't. |

Flat action layout (`arc3_wm.action_space`, bijective):

```
0..4     ACTION1..ACTION5    (parameter-less)
5..4100  ACTION6 (x, y) = unravel_index(idx - 5, (64, 64))
4101     ACTION7
```

Use `arc3_wm.action_space.build_mask(available)` /
`decode(idx)` / `encode(...)` if you need to reason about action
semantics; round-trip is property-tested.

## Path B — DreamerV3 `embodied`

`ARC3EmbodiedEnv` duck-types `embodied.Env` (it does **not** subclass
it, so importing it does not drag in JAX/`portal` — laptop-importable).
It exposes `obs_space`, `act_space`, `step(action_dict)`, `close`. The
Gymnasium→embodied translation (e.g. `is_terminal = terminated` only,
not on truncation; `log/`-prefixed non-agent keys) is handled for you.

`scripts/launch_pergame.py` is the reference wiring: it builds the
DreamerV3 agent + replay + driver around `ARC3EmbodiedEnv` **without
forking `dreamerv3`** (the pinned reference impl lives untouched in
`third_party/`). To bring a different `embodied`-compatible world
model, reuse that launcher and swap the agent — the env half is done.

## Swapping in offline data

`arc3_wm.replay_loader` turns the human-demonstration JSONL dataset
into transition tuples. Feed those into your world model's buffer to
reproduce the cross-game offline-pretraining setup, or ignore it for
pure online training. Schema is in
[docs/replay-format.md](replay-format.md); every record field that
matters for training (obs, action, reward, episode boundaries) is
parse-tested across the full dataset.

## Evaluating with RHAE

RHAE (Relative Human Action Efficiency) is the benchmark metric and is
**post-hoc** — it never touches your training loop. Capture per-eval
episode rewards (the `embodied` path does this via
`arc3_wm.eval_reward_sink`) then:

```bash
python scripts/compute_rhae.py \
  --episodes-file <logdir>/eval_episodes.jsonl \
  --game-id vc33 --baselines data/human_baselines.json --step 500000
```

RHAE weights each level by its index *within a game*, then combines
games — so games are **independently weighted**. See
[`arc3_wm/rhae.py`](../arc3_wm/rhae.py) for the exact definition and
`scripts/build_benchmark_table.py` for the multi-game report.

## Prerequisites

- `ARC_API_KEY` (free from <https://three.arcprize.org>) — needed once
  to cache game files.
- `python scripts/cache_env_files.py` — downloads the OFFLINE game
  files into `environment_files/`. After this, training/eval runs
  fully offline (no rate limits). `ARC3GymEnv` *requires* OFFLINE mode
  and raises a clear error otherwise.
- A `.env` with `OPERATION_MODE=offline` and
  `ENVIRONMENTS_DIR=environment_files`.

## What not to expect

This wrapper is project-pinned research code. It will not grow a
multi-backend abstraction, a custom encoder, or config knobs for
methods other than the reported baseline — those are explicit
follow-up-paper non-goals. Fork it if you need that; the standard
interfaces are stable to build on.
