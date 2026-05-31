# Design decisions

> Locked-in decisions for the ARC-AGI-3 sprint. CLAUDE.md remains the
> high-level spec; this file captures finer-grained choices and any
> rescoping. Do not modify a decision without explicit approval - add an
> entry under "Revision log" instead.

Last updated: 2026-05-08 (Phase 0 -> Phase 1 boundary).

## D1 - Replay download path

**Decision:** OAuth-authenticated `gdown` (`gdown --use-cookies` after
browser login). If still quota-blocked, copy the source Drive folder once
into Haso's own Drive, then `gdown --folder` against that copy.

**Why:** Anonymous `gdown --folder` hit Google's per-IP quota at 39 files of
342 in Phase 0. The bundled `arc_agi_3_public_demo_human_testing.zip` is
also rate-limited from the same IP. Authenticated requests use the user's
own quota.

**Status:** to do (Haso). Phase 0 left 39/342 files staged; downstream
scope (replay loader, WM pretraining) blocked until 342 land.

## D2 - Observation: layer-selection policy

**Decision:** `frame[-1]`, palette-decoded to `(64, 64, 3) uint8` via the
canonical 16-colour ARC LUT. Stock DreamerV3 normalisation downstream.

**Why:** `fd.frame` is `list[ndarray((64,64), int8)]` of intra-step
animation ticks (1-15 per step in pilot games). The last frame is the
post-action settled world state - what the agent observes before its next
decision. Single-frame obs aligns with stock DreamerV3 + size12m. Frame
stacks are out-of-scope.

**Implementation:**
- LUT lives in `arc3_wm/palette.py` as a frozen module constant.
- Sourced from `arcengine` if it exposes one; otherwise hardcoded to the
  ARC-1 palette and pinned with a unit test.
- `tests/test_palette.py` asserts decode is deterministic and shape/dtype
  is `(64, 64, 3) uint8`; round-trip a frame from each pilot game.

## D3 - Action space

**Decision:** Flat `Discrete(4102)` with per-step masking from
`fd.available_actions`. CLAUDE.md spec, unchanged.

- 0-4 -> `ACTION1`-`ACTION5`
- 5..4100 -> `ACTION6` with `(x, y) = unravel_index(idx - 5, (64, 64))`
- 4101 -> `ACTION7`

**Why:** Stock DreamerV3 expects a single discrete action. 4102 is
modest. Per-step masking (not per-episode) handles the dynamic
`available_actions` set correctly.

**Implementation:** `arc3_wm/action_space.py` exposes
`flat_to_arc(idx) -> (GameAction, dict|None)`,
`arc_to_flat(action_id, x, y) -> int`,
`build_mask(available_actions: Iterable[int]) -> np.ndarray[bool]` of
length 4102.

## D4 - Reward signal

**Decision:** `r = levels_completed[t+1] - levels_completed[t]`. Sparse,
integer, non-negative. No terminal bonus. No step penalty. No RHAE in the
training signal.

**Why:** `FrameDataRaw` has no `reward` field - the wrapper must derive
one. Methodology says "native env level-up rewards" -> `delta levels_completed`
is the closest match. Bonuses and penalties change the optimal-policy
magnitude and are out-of-scope until Phase 4 demonstrates they're needed.
RHAE is a post-hoc evaluation metric, not a training signal.

**Implementation:** `arc3_wm/env.py` keeps `_prev_levels_completed`,
emits `r = levels_completed - self._prev_levels_completed` in `step`.

## D5 - `action_input.id` parsing

**Decision:** Replay loader (when built) accepts both serialisations:
- Live wrapper writes the enum NAME (`"ACTION1"`).
- Public Drive replays use the integer VALUE (`1`).

Normalise to `int` internally. Raise on unrecognised values **with file
path + line number** in the message.

**Why:** Two serialisations exist in the wild (verified Phase 0). Silent
coercion masks data corruption.

**Status:** deferred (replay loader is post-vc33-sanity per the rescoping
below).

## D6 - `arc3_wm/rhae.py`

**Decision:** Keep, ~50 lines, reference implementation only.

**Use:**
- Per-checkpoint logging without re-instantiating `Arcade`.
- Sanity-test fixture asserting toolkit `get_scorecard().score` matches the
  reference on a hand-worked example from `methodology.md`.

**Why:** `get_scorecard()` is the source of truth. A reference
implementation guards against silent toolkit changes and gives us a
training-time logger.

**Status:** deferred (not on the wrapper-only critical path).

## D7 - `CLAUDE.md` git status

**Decision:** Keep gitignored. Haso manages it locally.

**Why:** Pre-existing `.GITIGNORE` listed `CLAUDE.md`. Haso confirmed the
choice. Reading the file directly is the way to know it changed; git log
will not help.

## D8 - Cross-game replay mixing

**Decision:** No mixing during per-game fine-tune. Default from CLAUDE.md
(Decisions Haso owns #9), confirmed.

## D9 - Per-episode max steps

**Decision:** Defer the default to 1000 steps unless Phase 4 says
otherwise. Wrapper exposes `max_steps` as a constructor kwarg.

## D10 - `GameAction` enum-construction quirk

**Decision:** Always use `GameAction.from_id(int)` to convert an integer
action ID to a `GameAction`. Never call `GameAction(int)`.

**Why:** `arcengine.GameAction` is an `Enum` with **tuple** values
internally (`(action_id, action_type)`). It overrides `__init__` to set
`_value_ = action_id`, but `Enum.__call__(value)` still looks up by the
*original* tuple. As a result `GameAction(1)` raises
`ValueError: 1 is not a valid GameAction` even though
`GameAction.ACTION1.value == 1`. Caught during Milestone 1.1 by
`tests/test_action_space.py::test_round_trip_all_indices`.

**How to apply:**
- `arc3_wm/action_space.py::flat_to_arc` uses `GameAction.from_id`. Pinned
  with an inline comment.
- The replay loader (deferred) will see `action_input.id` as an integer in
  the public Drive replays (per D5); use `GameAction.from_id` directly,
  not the constructor. Same applies anywhere a third party hands us an
  int and we want a `GameAction`.
- Live wrapper recordings serialise as the enum *name* (string) - use
  `GameAction[name]` (which Enum supports) or `GameAction.from_name(name)`
  for that branch.

## D11 - Action masking through DreamerV3's actor

**Decision:** **No masking** for milestone (3) (vc33). Actor samples
freely from `Discrete(4102)`; `arc_agi` silently no-ops dead actions.
Revisit when low-availability games (tu93 has 4/4102 live initially)
actually stall in Phase 4.

**Why:**
- Stock `danijar/dreamerv3` exposes no per-step action-mask hook on the
  actor. It outputs logits over the full discrete space and samples.
- Adding masking requires either patching the actor head (touches
  DreamerV3 internals -> CLAUDE.md anti-goal "Do not refactor DreamerV3
  internals"; also Decisions-Haso-Owns #5) or shaping the reward (also
  Decisions-Haso-Owns #5).
- vc33 has 4096/4102 ~ 99.85% live indices - the actor will discover
  the live region with negligible learning cost.
- `arc_agi` already no-ops unsupported actions; "no masking" is
  functionally safe, it just wastes some actor capacity.

**How to apply:**
- `arc3_wm/embodied_env.py` exposes `action_mask` (length-4102 bool) in
  the obs dict so a future masking patch is one wire away, but the
  agent does **not** consume it.
- A test asserts `action_mask` is computed correctly per step (so we
  can flip the masking on later without changing the env).
- For Phase 4: if any game shows `RHAE = 0` after 500k steps and the
  debug ladder (action mapping -> reward -> exploration) points to the
  actor wasting capacity on dead actions, *then* we revisit.

## D12 - How to wire `ARC3GymEnv` into DreamerV3

**Decision:** Bypass `dreamerv3/main.py`. Write our own
`scripts/launch_pergame.py` that builds the agent + env + replay buffer
+ driver itself. `arc3_wm/embodied_env.py` subclasses
`embodied.core.base.Env` directly and is the bridge.

**Why:**
- `dreamerv3/main.py:212` has a hardcoded suite-dispatch dict; no
  registration mechanism. Editing it counts as forking -> CLAUDE.md
  anti-goal.
- The existing `embodied.envs.from_gym.FromGym` adapter uses the
  **old** `gym` package (4-tuple `done`), not `gymnasium` (5-tuple
  `terminated, truncated`). Bridging `ARC3GymEnv` through it would
  require an extra shim and risks losing the `truncated!=terminated`
  distinction.
- Subclassing `embodied.core.base.Env` directly is short (~80 lines),
  matches the CLAUDE.md repo-layout listing of `scripts/launch_pergame.py`,
  and keeps the gymnasium 5-tuple semantics intact end-to-end.

**How to apply:**
- `arc3_wm/embodied_env.py` constructs `ARC3GymEnv` internally and
  translates `(obs, reward, terminated, truncated, info)` to
  `{image, reward, is_first, is_last, is_terminal, action_mask, ...}`.
- `is_terminal=True` <-> `terminated=True` (NOT on `truncated`). Pinned
  by a unit test.
- `is_last=True` on `terminated OR truncated` (the embodied "episode
  is done now" signal).
- `obs_space` includes only `image` for the model's image modality;
  `action_mask` is exposed but flagged with key prefix `log/` so
  embodied's wrappers don't try to feed it to the model. Per
  `embodied/core/base.py`: "By convention, keys starting with 'log/'
  are not consumed by the agent."
- `act_space = {"action": Discrete(4102), "reset": bool}` - matches
  what `FromGym` produces.

## D13 - `elements.Config.update()` is strict about new keys

**Decision:** Whenever we want a new env-suite (or any other) config key
to be overridable from the command line, **inject the key into
dreamerv3's `defaults` block** at config-merge time. Do **not** try to
introduce it via a named override block.

**Why:** `elements.Config.update()` raises `KeyError: Unknown key or
pattern <key>.` on any key that isn't already present in the config tree
it's updating. Layering a named block (e.g. `arc3:`) that introduces
brand-new nested keys (`env.arc3.max_steps`) fails - the update path
walks the existing tree and refuses to graft new branches. Caught during
Milestone 1.6 by `tests/test_launcher_arg_parsing.py::test_build_config_layers_arc3_block`
when the named block first tried to define `env.arc3`.

**How to apply:**
- `scripts/launch_pergame.py::DEFAULT_ARC3_ENV` is a module-level dict
  of the per-suite defaults. `load_merged_configs()` calls
  `defaults['env'].setdefault('arc3', {}).update(DEFAULT_ARC3_ENV)`
  *before* building the `elements.Config`. The named `arc3:` block in
  `configs/arc3.yaml` then only overrides keys that already exist.
- **Next time we add a per-suite or per-feature config key:** add it to
  the dict that's injected into defaults, *not* to a named block. Same
  rule applies for any future arc3-only knobs (e.g. action-mask
  enforcement when D11 is revisited at Phase 4).
- The collision guard in `load_merged_configs` (raises `RuntimeError`
  when `configs/arc3.yaml` defines a name dreamerv3 already uses) only
  protects *block names* - it cannot catch the deeper "I added a new key
  not in defaults" mistake. The cross-cutting safeguard is to always run
  `pytest tests/test_launcher_arg_parsing.py` after touching any config
  layout.

## D14 - `replay_context: 0` for offline WM-only pretraining

**Decision:** The `pretrain` block in `configs/arc3.yaml` sets
`replay_context: 0`. Phase 4-5 per-game online runs keep the DreamerV3
default of `1`.

**Why:** With `replay_context > 0`, `Agent.ext_space` declares
`enc/*, dyn/*, dec/*` carry-entry keys
([dreamerv3/agent.py:90-99](../third_party/dreamerv3/dreamerv3/agent.py#L90-L99))
that the agent's `assert sorted(data.keys()) == sorted(self.spaces.keys())`
check at [embodied/jax/agent.py:266](../third_party/dreamerv3/embodied/jax/agent.py#L266)
expects in every training batch. Those keys are written into the replay
buffer only by the actor's `policy()` call
([dreamerv3/agent.py:132-134](../third_party/dreamerv3/dreamerv3/agent.py#L132-L134)).
Offline pretrain on human replays has no actor, so the keys never get
written - every `agent.train` call would fail the assertion.

The DreamerV3 default of `1` implicitly assumes "an actor exists and is
writing carry entries into the buffer." That precondition isn't met in
offline WM-only pretraining, so `replay_context: 0` is the only coherent
value. This is not a deviation from the DreamerV3 recipe - it's the
correct setting for the regime DreamerV3's defaults don't cover.

`replay_context` is also strictly a compute optimization: it warm-starts
the RSSM carry from prior-batch entries to skip re-encoding a `prefix`-
length window from `is_first` on every sample. In online RL where
trajectories are sampled thousands of times, that saves real compute. In
offline pretrain where every chunk starts cold by design, there's
nothing to amortize. Fabricating zero entries to fake the keys would
pass the assertion but corrupt `_apply_replay_context`'s `truncate`
calls - the "tests pass, training silently broken" failure mode CLAUDE.md
flags as worst-case.

**Implementation:**
- `configs/arc3.yaml` `pretrain` block sets `replay_context: 0` with an
  inline comment pointing here.
- `scripts/pretrain_wm.py::pretrain_wm_loop` wraps `replay.sample` with
  `embodied.streams.Consec(prefix=args.replay_context, ...)`. With
  `replay_context: 0`, `Consec` is a clean no-op on the prefix axis
  (line 132 `stop = start + length + 0`); it still injects the required
  `consec` key.
- `make_replay` sizes buffer chunks via `consec*batch_length +
  replay_context`, so `replay_context: 0` yields 64-step chunks for
  size12m, consumed cleanly by `Consec(length=64, consec=1, prefix=0)`
  in strict mode.

**Second-order effects (monitor, not blockers):**

1. With `replay_context: 0`, every chunk starts with `is_first=True` and
   posterior-from-prior=zeros. Within-chunk dynamics still get full TBPTT
   via `Consec`'s `length`. **Cross-chunk** dynamics (carry threading
   between adjacent samples of the same trajectory) is untrained. For
   offline WM pretrain this is the right semantics - each chunk is a
   self-contained encode->predict episode - but it means the pretrained
   checkpoint's dynamics head is conditioned on cold-start initialisation.
2. When Phase 4 online runs resume from the pretrained checkpoint with
   `replay_context: 1`, the WM needs to adapt to also handle warm-start
   carry. Same dynamics function; only the prior at chunk boundaries
   shifts. Should converge fast - but **monitor `loss/dyn` for a
   transient bump in the first ~50k online steps**. If it spikes hard,
   the fix is a short `replay_context: 0` warm-up window before flipping
   to 1. Not paper-claim-affecting; just a thing to watch.
3. Paper note: anchor this in the method section as "offline pretrain on
   actor-less data does not satisfy the precondition for `replay_context`
   warm-starting." Accurate, defensible, no apology needed.

**Status:** landed at the Phase-3 smoke gate (Vast smoke surfaced the
agent.train assertion failure that this resolves).

## Phase-1 rescoping (supersedes the table-row in CLAUDE.md)

CLAUDE.md Section "Phases" row 1 says Phase 1 = "Wrapper + replay loader" with
the full test matrix (smoke / wrapper / action / reward / RHAE / replay
loader / 1000-episode stress). **Rescoped** for risk-management reasons:
unblock the DreamerV3 integration first, then come back for the rest.

### New sequencing

1. **Milestone (1) - Minimal vc33 wrapper.** Laptop. Single game.
   `arc3_wm/env.py` + `arc3_wm/action_space.py` + `arc3_wm/palette.py`
   only. Tests: smoke, wrapper-spec, action-space round-trip,
   palette decode, 100-episode random agent on vc33. **No** replay
   loader, **no** RHAE module, **no** multi-game test sweep.
2. **Milestone (2) - DreamerV3 sanity on Crafter.** Vast.ai. ~1-2 h.
   `--configs crafter size12m` on a single 5070-equivalent. Confirms
   install, JAX, GPU, logging, checkpointing all work end-to-end. Don't
   need to hit reference reward - need the loop to run clean and losses
   to descend.
3. **Milestone (3) - DreamerV3 on vc33, single seed, short.** Vast.ai.
   ~100k-200k env steps. Wires (1) into DreamerV3, no pretraining.
   Confirms the wrapper plays nice with Dreamer's buffer, action masking
   flows through, no shape/dtype errors. Live mask-coverage logging for
   the first 1k steps. RHAE > 0 ideal but not required.
4. **Phase 1 full scope (revisit).** After (3) gives green: replay
   loader, RHAE module, multi-game tests, 1000-episode stress. Phase 3
   cross-game pretraining unblocks at this point.

### Gates

- (1) -> (2): random-agent test green; scorecard populated; mask-coverage
  log per step.
- (2) -> (3): Crafter loop clean for at least 30 minutes; all four WM
  losses descending.
- (3) -> full Phase 1: vc33 run completes without crashing; action masking
  verified live (1k-step coverage log); no shape/dtype errors.

### What stays out of scope until after (3)

- Replay loader (`arc3_wm/replay_loader.py`).
- RHAE module (`arc3_wm/rhae.py`) beyond a unit-test stub.
- Multi-game support in the wrapper (constructor takes a single
  `game_id`).
- 1000-episode stress sweep across pilots.
- Full Phase 1 test matrix.

### Anti-pattern reminders for milestones (2) and (3)

- Vast.ai only. Laptop has no JAX-CUDA12 / no GPU; do not attempt.
- Spot/preemptible only.
- Always launch with `--logdir <persistent>` so the same command resumes.
- 30-min checkpoint cadence.
- Don't watch loss curves on the clock - backgrounded, monitor asynchronously.

## Revision log

- **2026-05-08** - initial entries D1-D9 + Phase-1 rescoping (Haso, kickoff).
- **2026-05-08** - D10 added (GameAction.from_id quirk), uncovered during Milestone 1.1.
- **2026-05-08** - D11/D12 added at the (1) -> (1.5) boundary: no actor masking on vc33; bypass dreamerv3/main.py with our own launcher.
- **2026-05-08** - D13 added during Milestone 1.6 (elements.Config.update strictness; inject per-suite defaults rather than introduce them in a named block).
