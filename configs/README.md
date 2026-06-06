# configs/

DreamerV3 config blocks for the training path. They are *layered on top
of* the stock `dreamerv3` configs (`size12m` and friends); this repo does
not fork or replace any upstream config.

## `arc3.yaml`

Two named blocks:

- **`arc3`** - per-game online fine-tuning. Selects the `arc3_<game>`
  task. Used by `scripts/launch_pergame.py`:

  ```bash
  python scripts/launch_pergame.py --configs size12m arc3 --task arc3_vc33
  ```

- **`pretrain`** - Phase-3 cross-game world-model-only pretraining on the
  mixed replay buffer. Used by `scripts/pretrain_wm.py`. Sets
  `replay_context: 0` (no actor, so no enc/dyn/dec keys in the buffer)
  and a wall-clock-bounded step budget with a 30-minute checkpoint
  cadence.

The env defaults (`env.arc3.max_steps`, `env.arc3.use_seed`) live in
`scripts/launch_pergame.py` (`DEFAULT_ARC3_ENV`), not here, so they can be
overridden on the CLI (e.g. `--env.arc3.max_steps=500`).

For Crafter sanity runs use the upstream `crafter` block directly and
override `--run.steps` on the CLI; see the note at the bottom of
`arc3.yaml`. Full training walkthrough: [../docs/vast-quickstart.md](../docs/vast-quickstart.md).
