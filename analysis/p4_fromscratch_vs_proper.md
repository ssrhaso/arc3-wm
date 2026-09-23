# Phase 4: from-scratch (cold) vs pretrained (warm) - paired comparison

Single A100 SXM4 40GB, 3 games x 2 seeds x 500k env-steps, stock DV3 `--configs size12m arc3 --script train_eval`.
Cold-arm sha `a06c02f`. Warm-arm sha `98de390`.
All configs identical between arms; only `--init-from-ckpt b2://.../pretrained-wm/v1/latest.pkl` removed for cold.
Eval-clear counts and eval scores are derived uniformly from the `eval_episodes.jsonl` rewards stream (sum(rewards) per episode = total levels cleared).

## Paired table

| game | seed | warm RHAE | cold RHAE | delta RHAE | warm levels | cold levels | warm eval clears/n | cold eval clears/n | warm mean eval ep-len | cold mean eval ep-len |
|------|------|-----------|-----------|--------|-------------|-------------|--------------------|---------------------|------------------------|------------------------|
| vc33 | 0 | 0.0548 | 0.0411 | -0.0137 | 1 | 1 | 4/23 | 2/21 | 50.5 | 51.4 |
| vc33 | 1 | 0.0166 | 0.0182 | +0.0016 | 1 | 1 | 2/18 | 3/19 | 52.4 | 53.3 |
| sb26 | 0 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 | 0/24 | 0/24 | 1000.0 | 1000.0 |
| sb26 | 1 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 | 0/24 | 0/24 | 1000.0 | 1000.0 |
| cd82 | 0 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 | 0/24 | 0/24 | 100.0 | 100.0 |
| cd82 | 1 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 | 0/24 | 0/24 | 100.0 | 100.0 |

## Train-time clears (count of episodes that cleared >=1 level during train)

| game | seed | warm train clears/n | cold train clears/n |
|------|------|---------------------|---------------------|
| vc33 | 0 | 17/9751 | 33/8451 |
| vc33 | 1 | 17/9746 | 17/9782 |
| sb26 | 0 | 0/502 | 0/502 |
| sb26 | 1 | 1/502 | 1/502 |
| cd82 | 0 | 0/4950 | 0/4950 |
| cd82 | 1 | 0/4950 | 0/4950 |

## Deltas (cold - warm)

| game-seed | delta RHAE | delta eval clears | delta eval ep-count |
|-----------|--------|---------------|------------------|
| vc33-s0 | -0.0137 | -2 | -2 |
| vc33-s1 | +0.0016 | +1 | +1 |
| sb26-s0 | +0.0000 | +0 | +0 |
| sb26-s1 | +0.0000 | +0 | +0 |
| cd82-s0 | +0.0000 | +0 | +0 |
| cd82-s1 | +0.0000 | +0 | +0 |

## Artifacts

- `_p4_analysis/p4_fromscratch_aggregation.json` (cold-only) + B2 `phase4-fromscratch/aggregation.json`
- `_p4_analysis/p4_aggregation_combined.json` (cold + warm + deltas) + B2 `phase4-fromscratch/aggregation_combined.json`
- `figures/p4_fromscratch_vs_proper.{png,svg}` + B2 `phase4-fromscratch/p4_fromscratch_vs_proper.{png,svg}`
