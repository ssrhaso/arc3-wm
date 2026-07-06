# HANDOFF: World Model Selection for Sparse-Reward ARC-AGI-3

**Owner:** Haso
**Status:** decision pending (see "Decision gate")
**Scope note:** TWISTER and reward shaping are both out-of-scope items in CLAUDE.md and require explicit sign-off before implementation. This document is a decision aid, not an approved workstream.

## Purpose

Decide which mechanism, if any, addresses the sparse-reward failure that keeps DreamerV3 at RHAE=0 on most Phase-4 games, and whether TWISTER (Burchi, AC-CPC plus Transformer SSM) is the right tool for it.

## TL;DR

1. The Phase-4 failure is reward cold-start (no positive examples in the buffer), not weak world-model representation. cd82 has the tightest world model in the sweep and still scores 0.
2. TWISTER improves representation quality, which is the axis we have already shown to be orthogonal to escape. It is therefore likely aimed one axis away from the actual wall.
3. Run one cheap probe (reward linear-decodability) before committing to any port. It decides whether TWISTER's lever is even relevant.
4. If the probe confirms scarcity, the on-mechanism fix is state-change reward shaping, applied env-side with no DreamerV3 fork.

## Status at a glance

| Item | State |
|------|-------|
| Reachability (real env) | Probed. Random policy gets 0/300 clears on vc33. |
| Reachability (imagination) | Probed. Imagined return identically 0 on 5 dead games. |
| Reward linear-decodability | Not yet probed. This is the decision gate. |
| State-change shaping | Designed, not implemented. |
| TWISTER port | Not started. Gated on the probe. |
