# Changelog

All notable changes to `arc3-wm` are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This is research code pinned for reproducibility of a NeurIPS-2026
workshop study - the version line tracks the *artifact* (the
Gymnasium / DreamerV3-`embodied` adapter and its harness), not a
general-purpose library API.

## [Unreleased]

### Added

- **`terminal_state` in the eval reward sink.** `EvalRewardSink` now
  appends the inner env's terminal `fd.state` name to each episode
  record: `{"rewards": [...], "terminal_state": "GAME_OVER"}`. It is read
  best-effort from `info["state"]` (written by `ARC3GymEnv._info`,
  re-exposed by `ARC3EmbodiedEnv.info`) at `is_last`, recording the
  terminal cause - `WIN` / `GAME_OVER` for a `terminated` episode,
  `NOT_FINISHED` for a `truncated` one - directly, so future eval runs no
  longer infer it from `reward == 0`. Backward-compatible: the key is
  omitted when no `info["state"]` is exposed, `compute_rhae` keys only on
  `rewards`, and `ai_actions = len(rewards) - 1` is unchanged. Logging
  is best-effort - any read failure degrades to the legacy shape rather
  than breaking the run. Tests in `tests/test_eval_reward_sink.py`.
  (Logging-only; existing run artifacts are not re-generated.)
- **`arc3-wm` console entry point.** `pyproject.toml` registers a
  `console_scripts` entry so the no-network install sanity check
  (`arc3_wm/__main__.py`) is runnable as `arc3-wm`, equivalent to
  `python -m arc3_wm`.
- **`Makefile`.** Convenience targets (`install`, `dev`, `cache`,
  `cache-all`, `check`, `test`, `test-fast`, `smoke`, `gym-smoke`,
  `clean`) wrapping the commands already documented in the README;
  bare `make` prints the target list.
- **`render_fps` env metadata.** `ARC3GymEnv.metadata` now declares
  `render_fps`, which `gymnasium.wrappers.RecordVideo` reads to time
  rendered rollouts; the `rgb_array` render path already existed.
  Playback-only metadata, no effect on stepping, training, or eval.
  Test in `tests/test_wrapper_spec.py`.

### Documentation

- **Per-directory READMEs.** Added `tests/`, `configs/`, and `data/`
  READMEs, rounded out the repository map, and cross-linked the setup
  docs from the docs index.
- **`compute_rhae` docstring.** Replaced a drifted line-number reference
  to the reward signal with the symbol (`ARC3GymEnv.step`) and updated
  the reward-stream note to describe `EvalRewardSink` (the implemented
  per-episode reward sink wired into `launch_pergame.py`) instead of the
  pre-implementation "would need a custom sink".

### Changed

- **Explicit `__all__` on every public submodule.** `action_space`,
  `palette`, `rhae`, `replay_loader`, `registration`, `env`, and
  `eval_reward_sink` each declare an `__all__`, pinning their public
  surface for `import *` and doc tooling. A parametrized drift guard in
  `tests/test_package_api.py` asserts every listed name resolves and no
  private name is exported.
- **`logit_bias` dtype annotation.** The `dtype` parameter is typed as
  `numpy.typing.DTypeLike` (was `type`), matching its documented use with
  dtype objects and string codes; pinned by a float64 test.
- **Cross-platform whitespace policy.** Added `.gitattributes`
  (LF in the repo, native on checkout) and a matching `.editorconfig`,
  forced LF on the `Makefile` so it stays usable on a Windows checkout,
  and extended `.gitignore` to cover editor swap and backup files. Keeps
  a Windows working tree from churning CRLF into every diff.

## [0.2.0] - 2026-06-03

### Added

- **MIT license.** `LICENSE` added (MIT, (c) 2026 Hasaan Ahmad),
  resolving the "license intentionally unset" item deferred in 0.1.0;
  the repo is now safe to make public.
- **`rgb_array` rendering.** `ARC3GymEnv` now accepts `render_mode` and
  implements `render()`, returning the most recent observation as an
  `(H, W, 3)` uint8 array. This completes the Gymnasium render contract,
  so standard utilities such as `gymnasium.wrappers.RecordVideo` work on
  the env with no custom code. The debug-only `arc_agi` terminal renderer
  is deliberately not exposed. Tests in `tests/test_wrapper_spec.py`.
- **`action_space.logit_bias(mask)`.** Canonical helper for the masking
  step CLAUDE.md specifies ("set actor logits to `-inf` on unsupported
  indices"): returns a length-4102 additive bias (`0.0` allowed, `-inf`
  masked) to add to policy logits before sampling. Re-exported from the
  package root. Tests in `tests/test_action_space.py`.
- **`CITATION.cff`.** Citation File Format (1.2.0) metadata for the
  software artifact, so GitHub renders a "Cite this repository" entry. A
  `preferred-citation` block for the workshop paper will be added on
  submission.

### Documentation

- **README.** Added status badges, a Development section (dev install
  and how to run the test suite, including the laptop/JAX split and the
  cached-env-files note), a Citing section pointing at `CITATION.cff`,
  and a quickstart pointer to `logit_bias`.

### Changed

- **`game_id` is validated against `PUBLIC_GAMES`.** `ARC3GymEnv` now
  raises a clear `ValueError` naming the offending id when constructed
  with a game outside the 25 public ARC-AGI-3 games, instead of the
  opaque "make() returned None" `RuntimeError` that a typo previously
  produced downstream.

## [0.1.0] - 2026-05-18

First formalised release of the contribution artifact. The wrapper,
action space, replay loader and RHAE metric were already implemented
and tested through Phases 0-4; this release turns the package into a
citable, installable contribution surface.

### Added

- **Public API.** `arc3_wm/__init__.py` exports the headline names
  (`ARC3GymEnv`, `flat_to_arc`, `arc_to_flat`, `build_mask`,
  `N_ACTIONS`) with an explicit `__all__`. `ARC3EmbodiedEnv` is exposed
  lazily (PEP 562) so the laptop/Gymnasium path does not import the
  JAX-side `elements` dependency.
- **Gymnasium registration.** `arc3_wm.registration` registers
  `ARC3/<game>-v0` for all 25 public games; ids self-register on
  `import arc3_wm`, so `gym.make("ARC3/vc33-v0")` works. Idempotent;
  the wrapper retains ownership of truncation (no double `TimeLimit`).
- **PEP 561 typing marker.** `arc3_wm/py.typed` ships in the wheel and
  the `Typing :: Typed` classifier is set, so downstream type-checkers
  honour the inline annotations.
- Tests: `tests/test_package_api.py`, `tests/test_registration.py`.

### Changed

- `__version__` bumped `0.0.0` -> `0.1.0` (Phase-0 scaffold string
  retired). pyproject now single-sources the version from
  `arc3_wm.__version__` via setuptools dynamic metadata.

### Fixed

- pyproject `dependencies` / `optional-dependencies` had escaped the
  `[project]` table (declared after `[project.urls]`, so scoped as
  `project.urls.dependencies`). A clean `pip install -e .` failed
  validation; static metadata had masked it. Dependency tables moved
  back under `[project]`.

### Notes / deferred

- **License is intentionally unset.** A license must be chosen before
  public release - see [docs/contribution.md](docs/contribution.md).
- BibTeX / citation block is added on paper submission.

[Unreleased]: https://github.com/ssrhaso/ARC_AGI_3/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ssrhaso/ARC_AGI_3/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ssrhaso/ARC_AGI_3/releases/tag/v0.1.0
