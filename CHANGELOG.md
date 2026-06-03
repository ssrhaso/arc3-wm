# Changelog

All notable changes to `arc3-wm` are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This is research code pinned for reproducibility of a NeurIPS-2026
workshop study - the version line tracks the *artifact* (the
Gymnasium / DreamerV3-`embodied` adapter and its harness), not a
general-purpose library API.

## [Unreleased]

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
