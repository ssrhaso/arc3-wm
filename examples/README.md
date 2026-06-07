# examples/

Runnable, laptop-only demonstrations of the `arc3_wm` standard
interface. No GPU, no JAX, no DreamerV3 - these exist to show that
ARC-AGI-3 is a stock RL environment through this package.

| File | What it shows |
|---|---|
| [`random_agent.py`](random_agent.py) | A random policy on one game via a direct `arc3_wm.env:ARC3GymEnv` construction, with and without action masking. The minimal "is this really plug-and-play?" check. |
| [`gym_make.py`](gym_make.py) | The same, driven through the **registered** `gym.make("ARC3/<game>-v0")` id - no `arc3_wm` symbol in the agent. The "formalised gym env" path; `--list` prints all 25 ids. |

Run:

```bash
python examples/random_agent.py --game vc33 --episodes 3
python examples/random_agent.py --game vc33 --episodes 3 --mask
python examples/gym_make.py --game vc33 --episodes 3
python examples/gym_make.py --list
```

The first and third of these are also available as `make smoke` and
`make gym-smoke`.

First time only: `pip install -e .`, then run `arc3-wm` (no network, no
game files) to confirm the install is wired up. After that export
`ARC_API_KEY`, run `python scripts/cache_env_files.py`, and create a
`.env` (`OPERATION_MODE=offline`, `ENVIRONMENTS_DIR=environment_files`).
See [../docs/using-the-wrapper.md](../docs/using-the-wrapper.md).

For the DreamerV3 / world-model training path (not a laptop example),
see `scripts/launch_pergame.py` and
[../docs/vast-quickstart.md](../docs/vast-quickstart.md).
