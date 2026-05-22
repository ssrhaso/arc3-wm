# Contribution skeleton — workshop-paper extension

Working scaffold for the NeurIPS-2026-workshop extension. Spine chosen
by Haso 2026-05-17: **env wrapper as the primary artifact**, plus a
**diagnosed negative result**, plus a **short 6-game RHAE baseline of a
world model** with each game independently level-weighted. This file is
the claims→evidence contract; numbers fill in as the runs land.

> Framing decisions here are Haso's (CLAUDE.md §"Decisions Haso owns"
> #3). This document records the agreed spine; it does not lock prose.

## Working title

*A Gymnasium Substrate and World-Model Baseline for ARC-AGI-3:
DreamerV3 Fits the World but Cannot Yet Act in It.*

## Abstract (draft)

ARC-AGI-3 is a new benchmark of interactive abstract-reasoning games
with a human-relative efficiency metric (RHAE). No world-model
approach had been applied to it and there was no standard RL entry
point. We contribute (1) the first Gymnasium- and
DreamerV3-`embodied`-compatible wrapper for ARC-AGI-3, with an
offline human-replay loader and a post-hoc RHAE harness; (2) a
controlled baseline: stock DreamerV3 on 6 public games × 2 seeds,
paired from-scratch vs cross-game offline world-model pretraining,
which gains ~no traction at a 500k-step budget and shows no measurable
benefit from pretraining; and (3) a diagnosis: the world model's
reconstruction and dynamics losses collapse while RHAE stays ≈0 — the
model learns to predict the environment but the controller cannot
exploit it within budget. We release the substrate so future
world-model work has a standard, reproducible starting point.

## Contributions (in order of durability)

1. **Substrate (primary).** First Gymnasium + DreamerV3-`embodied`
   entry point for ARC-AGI-3; 4102-way flat action space with masking;
   340-replay human-demo offline loader; post-hoc RHAE
   (per-game level-index-weighted, combined across games). Reusable
   independent of any result; no DreamerV3 fork.
2. **Controlled negative result.** Stock DreamerV3 `size12m` (the
   ARC-1-precedented config, Lee et al. 2024), 6 games × 2 seeds ×
   paired {cold, warm}. Cross-game offline WM pretraining gives Δ
   within seed variance.
3. **Mechanistic diagnosis.** WM fits (image-recon, dynamics losses
   collapse) but policy does not solve in imagination (RHAE ≈ 0);
   latent-probe / FVD / reasoning-axis localization.

## Claims → evidence map

| # | Claim | Evidence artifact | Status |
|---|---|---|---|
| C1 | First Gym/`embodied` wrapper for ARC-AGI-3 | `arc3_wm/{env,embodied_env}.py`; Living Survey "zero WM on ARC-3" | ✅ shipped |
| C2 | Faithful action/obs/reward mapping | `tests/test_action_space.py` (round-trip 0–4101), `test_wrapper_spec.py`, `test_reward.py` | ✅ green |
| C3 | Offline replay loader parses the full human dataset | `tests/test_replay_loader.py` over all 340 JSONLs | ✅ green |
| C4 | RHAE implemented to methodology (D-A/D-B) | `arc3_wm/rhae.py`; `tests/test_rhae.py` (50); coverage 70.49% | ✅ green |
| C5 | Cross-game WM pretraining converges | Phase-3 ckpt; WM losses ↓ monotone | ✅ done |
| C6 | Stock DreamerV3 ≈0 RHAE at 500k on most games | `scripts/build_benchmark_table.py` → `analysis/benchmark_table.{md,json}` | ◐ pilot in; expansion running |
| C7 | Cross-game pretraining gives no measurable lift (paired) | warm−cold Δ vs seed variance; `analysis/p4_fromscratch_vs_proper.md` | ◐ pilot in; expansion running |
| C8 | WM fits while controller fails (the diagnosis) | train losses (image 53→0.14, dyn ↓) vs RHAE≈0; Phase-6 probes/FVD/axis | ☐ Phase 6 |

## The 6-game benchmark

Pilot {vc33, sb26, cd82} + expansion {tn36, ls20, lf52}. Expansion
chosen for RHAE coverage + a human-hardness spread (fun/hard ratings ×
D-B coverage; see session log 2026-05-17). RHAE weights levels by
1-indexed position within each game, then averages games equally —
each game is **independently weighted**. Table:
`scripts/build_benchmark_table.py` (paired cold/warm, per-seed +
mean + combined `total_score`, graceful pre-results).

## Related-work positioning

- DreamerV3 (Hafner et al. 2025) — the model, used stock, unforked.
- Lee et al. 2024 (arXiv:2408.14855) — DreamerV3 on ARC-1; our config
  choice and the "should-have-precedent" framing of the negative
  result.
- ARC-AGI-3 paper (2603.24621) — RHAE definition.
- ARC-AGI Living Survey (2603.13372) — "zero WM approaches on ARC-3";
  the gap C1 fills.

## Limitations to state plainly (pre-empt reviewers)

- 500k-step single-arm budget by design (compute-bounded); not a claim
  that MBRL *cannot* do ARC-3 — scoped to *stock DreamerV3 at this
  budget ± this pretraining*.
- N = 6 games × 2 seeds — modest; the diagnosis (C8) carries the
  scientific weight, not the headline number.
- Substrate A100 timing/throughput is provenance, not a contribution.

## Open before submission (owner: Haso)

- [x] **License** — MIT (`LICENSE`; `pyproject` `license` field +
  classifier set; README updated). © 2026 Hasaan Ahmad.
- [ ] Authorship / acknowledgements / BibTeX.
- [ ] Venue + format confirmation; whether C8 ships in this paper or a
  follow-up (affects Phase-6 effort allocation).
- [ ] Notion change-log update (pilot swap + dry-run + expansion).

## BibTeX

```bibtex
@misc{arc3wm2026,
  title  = {A Gymnasium Substrate and World-Model Baseline for ARC-AGI-3},
  author = {Ahmad, Hasaan},
  year   = {2026},
  note   = {NeurIPS 2026 Workshop (under submission)}
}
```
