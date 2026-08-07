# TRM component library (`arc3_wm.trm`)

Tiny Recursive Model (Jolicoeur-Martineau 2025, arXiv:2510.04871) as a set
of composable components for ARC-AGI-3, targeting the diagnosis of the
DreamerV3 study: the world model fits but the controller never commits, so
the interventions here act on the control/exploration side while replacing
the WM with a discrete one that can be measured against copy-last-frame
honestly.

## Components

| Component | Module | Role |
| --- | --- | --- |
| `TRMCoreConfig` etc. | `trm.config` | dataclass configs, no torch import |
| `TRMCore` | `trm.core` | the recursion engine: shared 2-layer net, y/z states, ACT halt head, EMA helper, AdamATan2, stablemax CE |
| `GridTokenizer` / `ActionEncoder` / `GridHead` / `CellHead` | `trm.tokenizer` | 64x64 palette grid <-> patch tokens; flat-action token; per-cell decoders |
| `TRMWorldModel` | `trm.world_model` | (grid, action) -> next-grid logits + reward/state/change heads |
| `TRMPolicy` | `trm.policy` | grid -> factored action logits (7 types + 4096 click cells), mask enforced |
| datasets | `trm.data` | replay JSONL -> npz caches -> WM/BC torch datasets |
| `TRMAgent` | `trm.agents` | composition: BC prior and/or WM planner (novelty + reward + change) |
| trainer | `trm.training` | deep-supervision loop, EMA, checkpoints, resume, eval metrics |

Compositions are pure config: `AgentConfig(use_bc=..., use_wm=...)` yields
the BC agent, the model-based novelty planner, the hybrid, or the masked
random baseline; every experiment below is the same code with different
switches.

## Faithfulness and documented deviations

Kept from the paper/official repo: single shared tiny network (no H/L pair),
post-norm RMSNorm + SwiGLU, non-causal attention (RoPE) or the MLP mixer
variant, fixed y/z init buffers, h_cycles blocks per supervision step with
only the last back-propagated (full backprop through its l_cycles+1 calls,
no 1-step gradient), binary halt head with BCE-to-correctness target and
exploration minimum, EMA 0.999 for eval, AdamATan2, warmup-then-constant
LR, stablemax cross-entropy, bf16 forward.

Deviations, each with a reason:

1. **No puzzle-ID embedding.** arXiv:2512.11847 shows TRM's accuracy
   collapses to zero with a wrong/blank puzzle ID - it is per-task memory,
   unusable for an interactive agent. Per-game training replaces it.
2. **Explicit deep-supervision loop** (`n_supervision`) instead of the
   official carry-across-batches trick (their issue #26); each supervision
   step is one optimizer step, matching the official semantics
   synchronously.
3. **Augmentation off by default.** ARC-AGI-3 dynamics are not colour- or
   dihedral-equivariant (game code branches on specific colours and
   directions), unlike static ARC tasks.
4. **Changed-cell weighting + change head** in the WM loss. ARC-AGI-3
   boards are mostly static, so unweighted reconstruction is dominated by
   the copy solution - the measured failure of the RSSM baseline (its
   rollouts never beat copy-last-frame). `evaluate_wm` reports
   `changed_cell_acc` against the copy baseline explicitly.
5. **Factored policy head** (7 types + 64x64 click) instead of a monolithic
   4102-way softmax, with the per-game mask always applied - this removes
   the ls20 dilution confound (4 valid / 4102) by construction.

Parameter budget: default d_model 512, 2 layers -> ~7M per model (the
paper's TRM-Att scale), comfortably inside the DreamerV3 `size12m` budget
used by the baseline study.

## Experiment matrix (per game, vc33/sb26/cd82/tn36/ls20/lf52)

1. `trm_preprocess_replays.py` - replay corpus -> npz caches.
2. `trm_train_wm.py` - offline WM on human replays; val metrics include
   exact-match, cell/changed-cell accuracy vs copy, reward recall.
3. `trm_train_bc.py` - behaviour cloning on human replays (the policy
   prior the baseline study never used).
4. `trm_eval_agent.py` - online eval of the four compositions
   (random / bc / wm / bc+wm); writes `eval_episodes.jsonl` consumed by
   the existing `compute_rhae.py` unchanged.
