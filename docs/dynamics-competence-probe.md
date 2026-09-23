# Dynamics-competence probe (Phase 6)

Load-bearing evidence for the paper's **competence-vs-performance** claim: does the
frozen world model actually *represent the dynamics* of these games (competence),
independent of the dead reward head and the collapsed actor (performance)? The
probe answers this with two measurements on frozen per-game checkpoints:

* **Probe B - multi-step rollout fidelity.** Encode a fixed context, imagine `H`
  steps forward under the *real* action sequence (open-loop, no learned policy,
  no peeking at future frames), decode, and compare per-horizon cell accuracy vs.
  a **copy-last-frame** baseline. Beating copy across the actor's planning horizon
  is the direct refutation of "low reconstruction loss is trivial copying on a
  near-static board" (the footnote concession in the draft).
* **Probe A - counterfactual action-sensitivity** *(follow-on; see Stage 2)*. From
  one observed state, decode the one-step prediction under each candidate action.
  Sensitivity > 0 means the dynamics head responds to the action (not action-blind
  copying); the *taken* action's prediction matching the true next frame best (and
  beating copy) means the model holds the *actionable* knowledge the controller
  never exploits.

The result is a **competence × performance plane**: competence (these probes) on
one axis, RHAE on the other. The thesis is that they are decorrelated - e.g. cd82
fits tightest yet scores RHAE 0.

## Three-stage pipeline (only Stage 2 needs a GPU)

```
Stage 1  collect   CPU    scripts/probe_collect_holdout.py   -> holdout/*.npz
Stage 2  predict   GPU    scripts/probe_predict.py           -> pred/*_rollout.npz
Stage 3  score     CPU    scripts/probe_score.py             -> competence_table.csv + horizon_curve.csv
```

Stages 1 and 3 are JAX-free and unit-tested (`tests/test_dynamics_probe.py`,
`tests/test_probe_data.py`). The metrics live in `arc3_wm/dynamics_probe.py`; the
shared episode→window/spec segmentation in `arc3_wm/probe_data.py`. The predicted-
frame npz **contract** between Stage 2 and Stage 3 is defined by
`build_rollout_prediction_npz` and validated locally with
`scripts/probe_predict_synthetic.py` (a truth-derived stand-in for the GPU stage).

### Stage 1 - collect held-out ground truth (laptop)

Two sources per game (run both):

* `human` - the replay corpus (`data/replays/<game>/`). Richer, board-changing,
  goal-directed dynamics: the strong test. In-distribution (seeded the buffer);
  the copy baseline is what keeps it diagnostic. **No env-files needed.**
* `random` - fresh masked-uniform rollouts in the OFFLINE env: genuinely held-out
  samples from the training distribution. Needs `environment_files/<game>/`
  cached (`scripts/cache_env_files.py <game>`). Note: random clicks move very few
  *cells* per step (~0.25%), so the copy baseline is strong here - human is the
  higher-signal source.

```bash
python scripts/probe_collect_holdout.py --game cd82 --source both \
    --n-episodes 40 --outdir results/dynamics_probe/holdout
```

### Stage 2 - frozen-WM forward pass (GH200 / Vast)

Needs the dreamerv3 JAX stack (`pip install -U -r third_party/dreamerv3/requirements.txt`
plus `jax[cuda]`). No env-files required (explicit obs/act spaces).

```bash
# 0. pull a per-game checkpoint from B2 and extract it
b2 file download b2://$B2_BUCKET/phase4-proper/p4-cd82-s0-warm-98de390/ckpt-final.tar.gz ckpt.tar.gz
mkdir -p ckpt_cd82 && tar xzf ckpt.tar.gz -C ckpt_cd82
# extracts directly to: ckpt_cd82/latest  +  ckpt_cd82/<TS>/{agent,step,replay_*}.pkl

# 1. shape/JIT shakeout (no ckpt, random batch) - do this first
python scripts/probe_predict.py --game cd82 --self-test --context-len 4 --horizon 8

# 2. real predictions
python scripts/probe_predict.py --game cd82 --source human \
    --ckpt ckpt_cd82 --holdout results/dynamics_probe/holdout/cd82_human.npz \
    --context-len 4 --horizon 8 --outdir results/dynamics_probe/pred
```

`--ckpt` points at the directory that directly contains the `latest` pointer
file and the `<TS>/` folder (i.e. the tar's extraction root - there is **no**
nested `ckpt/` subdir; see [[project-dv3-ckpt-format]]). Stage 2 loads only the
`agent` attr (`agent.pkl`); the `replay_*`/`step` pkls are ignored.

**Known-risky spots to verify on first GH200 run** (this stage is untestable on
the laptop): the `report` `data`-key assertion (must equal obs∪act∪ext = `image,
reward, is_first, is_last, is_terminal, action, consec, stepid` + `seed`); the
`prevact` prepend alignment; `dec...pred()*255` scaling; and whether
`init_report`/`_seeds` need the `train_mesh` context. Iterate against tracebacks.

### Stage 3 - score (laptop)

```bash
python scripts/probe_score.py --pred-dir results/dynamics_probe/pred \
    --outdir results/dynamics_probe/scored
# -> competence_table.csv (per game/source), horizon_curve.csv (paper figure), probe_summary.json
```

## Local dry-run (validates Stages 1+3 without a GPU)

```bash
python scripts/probe_collect_holdout.py --game vc33 --source both --n-episodes 8
python scripts/probe_predict_synthetic.py --quality perfect --glob "vc33_*.npz" \
    --outdir results/dynamics_probe/pred_perfect
python scripts/probe_score.py --pred-dir results/dynamics_probe/pred_perfect \
    --outdir results/dynamics_probe/scored_perfect
```

Sanity targets: `perfect` → `model_acc≈1.0`, beats copy, `sensitivity>0`;
`copy` → `model_acc==copy_acc`, `sensitivity=0` (the trivial-copying null the
real WM must beat to demonstrate competence).
```
