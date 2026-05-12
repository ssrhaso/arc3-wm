# Phase checklists (Phase 0 deliverable)

> Explicit pass conditions for every phase's tests, expanded from the table
> in `CLAUDE.md` §"Phases". Each phase is locked behind these assertions —
> a phase is **not** done until every box in its checklist is green AND
> Haso has signed off in-session.

## Phase 0 — Setup

**Where:** laptop. **No training, no GPU.**

| Check | Pass condition | Status |
|---|---|---|
| Repo scaffold | `pyproject.toml`, `.gitignore`, `.env.example`, dirs from CLAUDE.md §"Repo layout" all present | ✅ |
| `arc_agi` import | `python -c "import arc_agi; importlib.metadata.version('arc-agi')"` prints `0.9.8` | ✅ (note: `arc_agi.__version__` does NOT exist — use `importlib.metadata`) |
| `.env` loaded | `OPERATION_MODE=offline` + `ARC_API_KEY=…` in `.env`; `arc_agi.base` auto-loads via `python-dotenv` | ✅ |
| OFFLINE active | `arc_agi.Arcade().operation_mode == OperationMode.OFFLINE` after import | ✅ (verified) |
| Env files cached | `environment_files/{vc33,tu93,cd82}/<version>/{metadata.json,<game>.py}` exist | ✅ (3 games cached for the smoke; full 25 needed before Phase 3) |
| Docs fetched | All pages listed in §6 of the Phase-0 instructions present under `docs/arc-agi-3/` | ✅ |
| `harness-analysis.md` written | Covers obs/action/reward/scorecard/modes/list_actions; answers RHAE question | ✅ |
| `replay-format.md` written | Per-line schema, action/reward extraction, worked example | ✅ |
| `compute-runbook.md` written | Vast.ai launch + resume + data staging | ✅ |
| `phase-checklists.md` written | This file | ✅ |
| References cloned | `third_party/{ARC-AGI-3-Agents,dreamerv3}/` populated | ✅ |
| Replays fetched | `find data/replays -name '*.jsonl' \| wc -l == 342` | ❌ **39/342 — Drive quota blocker.** See `docs/replay-format.md` §"Open issue" |
| Random-agent smoke | `scripts/random_agent_smoke.py` runs 10 episodes on vc33 OFFLINE, no exceptions, scorecard populated | ✅ (921 FPS, score=0 with random play, expected) |
| RHAE source-of-truth | Documented whether `get_scorecard()` returns RHAE | ✅ — yes, on 0–100 scale |

**Phase 0 exit gate:** all rows above ✅ except the replay count, plus
Haso sign-off. The replay count is the open blocker; everything else
is met.

---

## Phase 1 — Wrapper + replay loader

**Where:** laptop. **No GPU.** **All file names are concrete.**

### Test files (write tests first, then implementation)

| File | What it asserts | Touches module |
|---|---|---|
| `tests/test_smoke.py` | `import arc3_wm` and `import arc3_wm.env` succeed; `arc3_wm.__version__` matches `pyproject.toml` | `arc3_wm/__init__.py` |
| `tests/test_wrapper_spec.py` | `arc3_wm.env.ARC3GymEnv` is a `gymnasium.Env`; `observation_space` is `Box(0, 255, (64, 64, 3), uint8)`; `action_space` is `Discrete(4102)`; `reset()` returns `(obs, info)` with correct shape/dtype/range; `step(action)` returns `(obs, reward, terminated, truncated, info)` | `arc3_wm/env.py` |
| `tests/test_action_space.py` | `flat_to_arc(idx)` and `arc_to_flat(action_id, x, y)` round-trip identity for all `idx ∈ [0, 4101]`; `flat_to_arc(0..4) → ACTION1..ACTION5`; `flat_to_arc(5 + 64*y + x) → (ACTION6, x, y)`; `flat_to_arc(4101) → ACTION7`; `build_mask(env)` produces a length-4102 bool tensor whose `True` entries match the union over (action_id, x, y) tuples valid for the current `available_actions` (ACTION6 contributes 4096 indices when 6 ∈ available, otherwise 0) | `arc3_wm/action_space.py` |
| `tests/test_reward.py` | Reward signal matches `Δ levels_completed` between consecutive frames; on terminal `WIN` and `GAME_OVER` the wrapper sets `terminated=True` (not `truncated`); `truncated=True` only on `max_steps` timeout | `arc3_wm/env.py` |
| `tests/test_rhae.py` | `arc3_wm.rhae.compute_rhae(level_baselines, level_actions, levels_completed)` matches the toolkit's `EnvironmentScorecard.score` on a fixture from `methodology.md` (worked example, hand-computed); cap at 1.15 (or 115 on 0–100 scale — pick one); per-level / per-game / total formulas all covered; reproduces toolkit's `max_score` early-exit penalty | `arc3_wm/rhae.py` |
| `tests/test_replay_loader.py` | Parses **all 342 JSONLs** (no sample); accepts `action_input.id` as int OR string; tolerates the trailing session-summary line; emits `(o_t, a_t, r_t, terminated_t, o_{t+1})` tuples with `o.shape == (64, 64, 3)`, `o.dtype == uint8`, `a ∈ [0, 4101]`, `r ∈ {-1, 0, 1, ...}` (just `Δ levels_completed` unless Haso adds shaping); episode-boundary detection inside a single JSONL via `state ∈ {WIN, GAME_OVER}` and trailing summary; total transition count > 0 and reasonable (per CLAUDE.md, ~30k–150k aggregate) | `arc3_wm/replay_loader.py` |
| `tests/test_palette.py` | The 16-entry RGB palette is fixed and matches whatever `arcengine` exposes (or our hardcoded baseline if not exposed); decoding `0..15` produces a `(H, W, 3) uint8` array; idempotent under round-trip if palette is invertible | `arc3_wm/env.py` |

### Pass criteria for Phase 1 (all must hold)

- `pytest -q` exits 0 with all tests above passing.
- 100-episode random-agent smoke on vc33 / tu93 / cd82 — zero exceptions,
  per-game scorecards populated, FPS ≥ 500 on this laptop without
  `render_mode`.
- 1000-episode stress sweep across the three pilot games — no exceptions,
  no NaN, no observation-shape variation across runs.
- Full action-space enumeration: every flat idx 0–4101 produces a
  well-formed `(action_id, data)` pair when fed through `flat_to_arc`,
  and ACTION7 is correctly masked out on games where `7 ∉ available_actions`.
- Replay loader parses all 342 JSONLs (depends on Drive quota
  resolution).

### Decisions Haso must sign off before Phase 1 codes anything

- **Layer-selection policy** for `fd.frame` (multi-layer animation frames).
  Default proposal: `frame[-1]`. Alternative: stack last K, padded.
  See `docs/harness-analysis.md` §"Observations".
- **Reward shaping.** Current default: `r = Δ levels_completed`.
  Alternative: per-frame state-change reward (StochasticGoose pattern). The
  paper sticks with native level-up unless Phase 4 stalls.
- **Whether `arc3_wm/rhae.py` is built.** Default proposal: yes, but as a
  test fixture / per-checkpoint logger, not as the source of truth — the
  toolkit's `get_scorecard()` is authoritative.

---

## Phase 2 — Crafter sanity

**Where:** Vast.ai 1× H100 80GB spot. **~6–12 h.**

| Check | Pass condition |
|---|---|
| DreamerV3 install | `python -c "import dreamerv3"` works on the instance |
| JAX GPU detected | `python -c "import jax; print(jax.devices())"` shows `[CudaDevice(id=0)]` |
| Crafter run launches | `python third_party/dreamerv3/dreamerv3/main.py --logdir … --configs crafter size12m` writes a checkpoint within 10 minutes |
| Within 10% of reference | At 1M env steps, `score / 11.7 ∈ [0.9, 1.1]` (Crafter reference reward; cite DreamerV3 paper Fig 4) |
| Resume-from-preemption | Manually `kill -9` the process, re-launch with same `--logdir`, observe "Loading checkpoint…" log, training continues without losing > 30 min of progress |
| Logdir on persistent volume | `df` shows `/workspace` is the persistent mount; `~/logdir` is a symlink to or under it |
| 30-min sync working | `rsync` cron lands a snapshot in `/workspace/logdir-snapshot/` |

**Open question:** is the 10% reference window scored on raw reward,
norm-reward, or "achievements unlocked"? Confirm against DreamerV3 paper
Table 4 / Fig 4 before claiming pass.

---

## Phase 3 — Cross-game WM pretrain

**Where:** Vast.ai 1× H100 80GB spot. **~6 h.**

| Check | Pass condition |
|---|---|
| Replay buffer | All 342 replays load into a DreamerV3 `embodied.replay.Uniform` buffer; total transitions > 25k; per-game distribution roughly even |
| WM-only updates | Verified by code inspection that `actor.update(...)` and `critic.update(...)` are NOT called during pretrain (Phase 1 design) |
| Loss decreases monotonically | All four WM losses (recon, dynamics, reward, continue) trend down across 1 epoch; smoothed loss at end < 0.9 × smoothed loss at start (per loss) |
| No NaN | Throughout the run, no `NaN` in any logged metric |
| Checkpoint cadence | At least 12 ckpts written (every 30 min for ~6 h); `checkpoints/pretrained_wm/latest/` exists |
| Resume verified | One forced preemption mid-run; re-launch picks up within 30 min of last ckpt |
| Reward / continue heads sensible | On a held-out replay, predicted level-up probability spikes near actual level-up boundaries (qualitative; show one figure) |

**Anti-pattern guard:** if any WM loss is suspiciously flat at zero
("collapsed"), stop — likely a buffer-format mismatch with `arc3_wm/replay_loader.py`.

---

## Phase 4 — 3-game pilot (vc33, sb26, cd82)

> Pilot was originally `(vc33, tu93, cd82)`. tu93 swapped out 2026-05-12:
> under the n≥2 coverage threshold the human-baseline fixture only covers
> 3/9 tu93 levels, invalidating tu93's "9 levels = highest RHAE granularity"
> selection rationale. sb26 chosen for 8/8 coverage and a distinct
> mid-difficulty action-count range vs vc33/cd82.

**Where:** local 5070 cluster. **~5 h wall-clock.** **Load-bearing gate.**

| Check | Pass condition |
|---|---|
| Warm start works | Phase 3 WM checkpoint loads on each fresh per-game run; first imagination rollouts are non-trivial (qualitatively) |
| Fresh actor/critic | Actor + critic are NOT loaded from any prior run; verified by checkpoint-key inspection |
| Per-game replay pre-population | Each game's ~10–15 human replays land in the buffer before any online step |
| Three games × two seeds | All 6 runs launch and reach 500k env steps without crashes |
| `RHAE > 0` on ≥ 2 of 3 games | Toolkit `get_scorecard().score > 0` for at least one seed of at least 2 of {vc33, sb26, cd82} within 500k env steps |
| Reasonable FPS | Each run sustains ≥ 200 FPS env-step (single 5070 + size12m WM is the budget) |
| No silent fallbacks | If a run reaches the 500k step budget with `RHAE = 0`, flag and stop the sweep — do NOT silently scale to 1M and hope |

**Phase 4 outcome decides Phase 5 launch.** If RHAE is 0 on all three
games, debug in this order before scaling: (1) action-mapping correctness,
(2) reward-signal correctness, (3) exploration. Do **not** skip to Plan2Explore
without Haso's sign-off.

---

## Phase 5 — Full sweep

**Where:** local 5070 cluster. **~3.5 days wall-clock.**

| Check | Pass condition |
|---|---|
| All 25 × 3 seeds × 1M steps complete | 75 runs all reach the step budget; failed-run replacements documented |
| IQM + 95% bootstrap CIs | Per-metric IQM with `rliable` (Agarwal et al. 2021); the IQM CI half-width ≤ 0.05 on the per-game RHAE distribution |
| Held-out transfer eval | Each game's WM evaluated on a held-out subset of its own replays; per-game NLL or MSE reported |
| No GPU drop-out unaccounted | If a 5070 dies mid-sweep, the affected runs are re-launched on a different 5070 within 4 h; no quietly-incomplete entries in the final table |
| Per-run W&B / scope log | Every run has a logdir + saved YAML config + final scorecard JSON |

---

## Phase 6 — Diagnostics

**Where:** local 5070s or laptop, 1 GPU sufficient. **~1 day.**

| Check | Pass condition |
|---|---|
| Linear probes on RSSM latents | For each of {object position, last action, level index}, a logistic / linear probe on the trained-RSSM stochastic latent achieves accuracy ≥ random + 5σ |
| FVD on imagination rollouts | FVD computed on 1k imagined trajectories vs 1k real; finite, sane (not adversarial-quality, just sane); reported per game |
| Reasoning-axis stratification | Per-game RHAE binned by ARC-3's reasoning-axis taxonomy; chart present in the paper |
| No new model trained | All diagnostics use Phase 5 checkpoints. No re-training. |

---

## Phase 7 — Writing + submission

**Where:** laptop. **~2 days.**

| Check | Pass condition |
|---|---|
| Paper compiles | `pdflatex` / `xelatex` produces a clean PDF; bibtex resolves; no missing refs |
| All figures present | Phase 5 IQM table, Phase 4 pilot curves, Phase 6 probe / FVD figs |
| All claims sourced | Every numeric claim traceable to a logdir + a config + a scorecard JSON |
| RHAE scale consistent | Either everywhere 0–1 (with cap 1.15) or everywhere 0–100 (with cap 115). Not mixed. |
| Workshop submission | Paper uploaded to NeurIPS 2026 Workshop site by deadline |

---

## Cross-phase invariants (always true)

- No `git push` ever this sprint without Haso's explicit per-action approval.
- No `rm -rf` outside `__pycache__/`, `.pytest_cache/`, build artifacts, or
  files Claude created in the current session.
- DreamerV3 source is unmodified — env registered via `embodied/`, no fork.
- `OPERATION_MODE=offline` for all training/eval. ONLINE is reserved for
  the one-off official scorecard submission, post-Phase 5.
- Every long-running training launch uses `--logdir <persistent>` so that
  re-running the same command resumes.
- Tests are written **before** the implementation they cover. Phase rules
  do not bend on this.
