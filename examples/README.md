# examples/

Runnable, laptop-only demonstrations of the `arc3_wm` standard
interface. No GPU, no JAX, no DreamerV3 — these exist to show that
ARC-AGI-3 is a stock RL environment through this package.

| File | What it shows |
|---|---|
| [`random_agent.py`](random_agent.py) | A random policy on one game via the Gymnasium seam (`arc3_wm.env:ARC3GymEnv`), with and without action masking. The minimal "is this really plug-and-play?" check. |

Run:

```bash
python examples/random_agent.py --game vc33 --episodes 3
python examples/random_agent.py --game vc33 --episodes 3 --mask
```

First time only: `pip install -e .`, export `ARC_API_KEY`, run
`python scripts/cache_env_files.py`, and create a `.env`
(`OPERATION_MODE=offline`, `ENVIRONMENTS_DIR=environment_files`). See
[../docs/using-the-wrapper.md](../docs/using-the-wrapper.md).

For the DreamerV3 / world-model training path (not a laptop example),
see `scripts/launch_pergame.py` and
[../docs/vast-quickstart.md](../docs/vast-quickstart.md).
