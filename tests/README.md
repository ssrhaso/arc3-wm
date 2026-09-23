# tests/

The test suite is the spec: every public behaviour of `arc3_wm` is
pinned here, including the failure modes (action mapping, reward signs,
episode boundaries, JSONL field names).

## Running

```bash
pip install -e ".[dev]"
pytest                                 # full suite
pytest tests/test_action_space.py -q   # a single module
pytest -n auto                         # parallel (pytest-xdist)
```

The same commands are available as `make test` / `make test-fast`.

## Laptop vs JAX

Most tests are pure-Python and run on a laptop with no GPU and no JAX.
Tests that need the DreamerV3 / JAX stack carry the `requires_jax`
marker and auto-skip when `jax` is unimportable (see
[`conftest.py`](conftest.py)); they run on the GPU boxes.

A few env-level tests read cached OFFLINE game files from
`environment_files/`; run `python scripts/cache_env_files.py` once
(needs `ARC_API_KEY`) if they are missing.

## Layout

- `test_action_space.py`, `test_wrapper_spec.py`, `test_registration.py`,
  `test_package_api.py`, `test_palette.py`, `test_smoke.py`,
  `test_random_agent.py` - the Gymnasium wrapper and its public API.
- `test_replay_loader.py`, `test_extract_human_baselines.py` - the
  offline human-replay dataset path.
- `test_rhae.py`, `test_compute_rhae.py`, `test_eval_random_rhae.py`,
  `test_eval_reward_sink.py` - the RHAE metric and its plumbing.
- `test_embodied_env.py`, `test_pretrain_wm.py`, `test_wm_only_agent.py`,
  `test_launcher_*.py`, `test_phase4_chain.py` - the DreamerV3
  `embodied` / training path (mostly `requires_jax`).
- `test_main_cli.py`, `test_requires_jax_marker.py` - the `arc3_wm` CLI
  and the marker contract itself.
