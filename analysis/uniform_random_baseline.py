"""Uniform-random success-rate baseline for vc33 level 1, vs the observed
12/92 burst rate. Tests whether the H2 'spike' could be the natural rate
of clicking with a uniform-random policy.

Inputs:
  - scratch/p4-vc33-dryrun/data/replays/vc33/*.recording.jsonl  (10 replays)
  - scratch/p4-vc33-dryrun/metrics.jsonl (for entropy methodology check)

Two baseline definitions:
  (A) TERMINAL-CELL: the winning click is one of the k unique cells humans
      clicked at the L0 -> L1 transition row. k may be 1 if all humans found
      the same cell, larger if the env accepts multiple winning cells.
  (B) SOLUTION-PATH: each level-1 win requires c specific cells (in some
      order) to be clicked. c = unique cells in the modal human solution
      path during levels_completed=0. Two sub-cases:
        order-agnostic: all c cells must appear, order doesn't matter
        order-required: the c cells must appear as an ordered subsequence

Reported uniformly with action-space = 4102 (matches observed entropy
log(4102) = 8.3192, NOT log(4096)). Also reports the 4096 variant for
comparison with the user's brief.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

REPLAYS = Path("scratch/p4-vc33-dryrun/data/replays/vc33")
METRICS = Path("scratch/p4-vc33-dryrun/metrics.jsonl")
N_STEPS = 51
ACT_SPACE_FULL = 4102        # 5 (ACTION1-5) + 4096 (ACTION6 cells) + 1 (ACTION7)
ACT_SPACE_MASKED = 4096      # if mask applied at logit level
OBS_SUCCESSES = 12
OBS_EPISODES = 92


def load_replay(path: Path):
    """Return list of (action_id, x, y, levels_completed, state) tuples."""
    rows = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)["data"]
            ai = d.get("action_input", {}) or {}
            ad = ai.get("data") or {}
            rows.append((
                ai.get("id"),
                ad.get("x"),
                ad.get("y"),
                d.get("levels_completed"),
                d.get("state"),
            ))
    return rows


def find_level1_paths(rows):
    """Return list of (winning_cell, path_cells, path_len) per L0->L1 transition.

    A 'path' is the sequence of (x,y) clicks while levels_completed==0
    leading up to the transition row whose action triggered L0->L1.
    """
    results = []
    current_path: list[tuple[int, int]] = []

    for i, (aid, x, y, lvl, state) in enumerate(rows):
        if aid == 6 and x is not None and y is not None:
            # The action at row i is what produced state at row i.
            # If row i-1 had levels_completed=0 and row i has =1, then
            # the click (x,y) at row i is the winning click.
            prev_lvl = rows[i - 1][3] if i > 0 else None
            if prev_lvl == 0 and lvl == 1:
                results.append((
                    (x, y),
                    list(current_path) + [(x, y)],
                    len(current_path) + 1,
                ))
                current_path = []
                continue
            if lvl == 0:
                current_path.append((x, y))
            else:
                # Past level 0; ignore (we only care about L0 paths).
                pass
        elif aid == 0:
            # Reset row — restart path tracking.
            current_path = []
    return results


def wilson_ci(k, n, z=1.96):
    """Wilson 95% binomial CI."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def p_at_least_one_in_51(p_per_click, n=N_STEPS):
    return 1.0 - (1.0 - p_per_click) ** n


def p_all_c_in_51_unordered(c, A, n=N_STEPS):
    """Probability that c specific cells ALL appear at least once in n
    uniform draws from A actions. Inclusion-exclusion exact."""
    if c == 0:
        return 1.0
    if c * 1.0 / A > 0.5:
        return 0.0  # not meaningful here
    total = 0.0
    for i in range(c + 1):
        sign = (-1) ** i
        term = math.comb(c, i) * ((A - i) / A) ** n
        total += sign * term
    return total


def p_ordered_subsequence(c, A, n=N_STEPS):
    """Probability that c specific cells appear in a specific order as a
    subsequence in n uniform draws from A actions. Exact for c <= n.

    P = C(n, c) * (1/A)^c * ((A - 1) / A)^(n - c) ... but this counts
    "the c chosen positions match in order; others are anything except
    constrained-cell positions". For simplicity we use the cleaner
    approximation: C(n, c) * (1/A)^c (ignores re-occurrences, which are
    rare for A >> n).
    """
    if c == 0:
        return 1.0
    if c > n:
        return 0.0
    return math.comb(n, c) * (1.0 / A) ** c


def main():
    # ---------- replay parsing
    replay_paths = sorted(REPLAYS.glob("*.recording.jsonl"))
    print(f"Replays found: {len(replay_paths)}")

    all_wins = []
    all_paths = []
    cleared_count = 0
    not_cleared = 0
    for p in replay_paths:
        rows = load_replay(p)
        wins = find_level1_paths(rows)
        if wins:
            cleared_count += 1
            # take the FIRST L0->L1 transition per replay (humans replay only once)
            wcell, path, length = wins[0]
            all_wins.append(wcell)
            all_paths.append(path)
        else:
            not_cleared += 1
        if len(wins) > 1:
            print(f"  NOTE {p.name}: {len(wins)} L0->L1 transitions (using first)")

    print(f"  Replays that cleared L0: {cleared_count}")
    print(f"  Replays that did NOT clear L0: {not_cleared}")

    # ---------- (A) terminal cells
    unique_wins = sorted(set(all_wins))
    k = len(unique_wins)
    win_counter = Counter(all_wins)
    print()
    print("=" * 70)
    print("(A) TERMINAL-CELL DEFINITION")
    print("=" * 70)
    print(f"  Winning cells (per-replay terminal clicks): {all_wins}")
    print(f"  k = {k} unique winning cells")
    print(f"  Distribution: {dict(win_counter)}")
    for A_label, A in [("4096 (mask applied)", ACT_SPACE_MASKED),
                       ("4102 (no mask, observed entropy)", ACT_SPACE_FULL)]:
        p_click = k / A
        p_ep = p_at_least_one_in_51(p_click)
        print(f"  A={A_label}: p_per_click={p_click:.6f}, p_per_episode={p_ep:.4f} = {p_ep*100:.2f}%")

    # ---------- (B) solution path
    path_lengths = [len(p) for p in all_paths]
    m_median = median(path_lengths) if path_lengths else 0
    m_min = min(path_lengths) if path_lengths else 0
    m_max = max(path_lengths) if path_lengths else 0
    unique_cells_per_path = [len(set(p)) for p in all_paths]
    c_median = median(unique_cells_per_path) if unique_cells_per_path else 0
    c_min = min(unique_cells_per_path) if unique_cells_per_path else 0
    c_max = max(unique_cells_per_path) if unique_cells_per_path else 0

    print()
    print("=" * 70)
    print("(B) SOLUTION-PATH DEFINITION")
    print("=" * 70)
    print(f"  Path lengths (clicks while levels_completed=0): {path_lengths}")
    print(f"  median m = {m_median}, range [{m_min}, {m_max}]")
    print(f"  Unique cells per path: {unique_cells_per_path}")
    print(f"  median c = {c_median}, range [{c_min}, {c_max}]")

    for c_label, c in [("c = min across humans", c_min),
                       ("c = median across humans", c_median),
                       ("c = 1 (single winning cell)", 1)]:
        print(f"  -- {c_label}: c={c} --")
        for A_label, A in [("A=4096", ACT_SPACE_MASKED),
                           ("A=4102", ACT_SPACE_FULL)]:
            p_unordered = p_all_c_in_51_unordered(c, A)
            p_ordered = p_ordered_subsequence(c, A)
            print(f"    {A_label}: P(unordered all-c-appear)={p_unordered:.6f}={p_unordered*100:.3f}%,  P(ordered subseq)={p_ordered:.2e}")

    # ---------- (C) compare to observed 12/92
    obs_p = OBS_SUCCESSES / OBS_EPISODES
    ci_lo, ci_hi = wilson_ci(OBS_SUCCESSES, OBS_EPISODES)
    print()
    print("=" * 70)
    print("(C) COMPARE TO OBSERVED BURST RATE")
    print("=" * 70)
    print(f"  Observed: {OBS_SUCCESSES}/{OBS_EPISODES} = {obs_p*100:.2f}% (Wilson 95% CI [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%])")
    print()

    candidates = []
    for A_label, A in [("4096", ACT_SPACE_MASKED), ("4102", ACT_SPACE_FULL)]:
        candidates.append(("(A) terminal k={}, A={}".format(k, A_label),
                          p_at_least_one_in_51(k / A)))
        for c_label, c in [("min", c_min), ("median", c_median)]:
            candidates.append(("(B-unord) c={} ({}), A={}".format(c, c_label, A_label),
                              p_all_c_in_51_unordered(c, A)))
            candidates.append(("(B-ord) c={} ({}), A={}".format(c, c_label, A_label),
                              p_ordered_subsequence(c, A)))

    for label, p in candidates:
        in_ci = ci_lo <= p <= ci_hi
        note = "  CONSISTENT with observed CI" if in_ci else ""
        print(f"  {label:50s} -> {p*100:.4f}% {note}")

    # ---------- (D) full-resolution entropy methodology check
    print()
    print("=" * 70)
    print("(D) ENTROPY METHODOLOGY CHECK")
    print("=" * 70)
    ent_rows = []
    with METRICS.open() as f:
        for line in f:
            d = json.loads(line)
            v = d.get("train/ent/action")
            if v is not None:
                ent_rows.append((int(d["step"]), float(v)))
    ent_rows.sort()
    pre = [v for s, v in ent_rows if s < 232_000]
    print(f"  Pre-burst samples: n={len(pre)}")
    print(f"  Pre-burst entropy: min={min(pre):.6f}, max={max(pre):.6f}, mean={sum(pre)/len(pre):.6f}")
    print(f"  log(4096) = {math.log(4096):.6f}")
    print(f"  log(4102) = {math.log(4102):.6f}")
    pre_max = max(pre)
    eps_from_log4096 = pre_max - math.log(4096)
    eps_from_log4102 = pre_max - math.log(4102)
    print(f"  pre-burst-max entropy vs log(4096): {eps_from_log4096:+.6f} nats")
    print(f"  pre-burst-max entropy vs log(4102): {eps_from_log4102:+.6f} nats")
    if abs(eps_from_log4102) < 1e-3:
        print("  -> entropy LITERALLY at log(4102): policy samples uniformly over ALL 4102 actions.")
    elif abs(eps_from_log4096) < 1e-3:
        print("  -> entropy LITERALLY at log(4096): mask applied at logit level.")
    else:
        print(f"  -> entropy NOT exactly at either uniform — effective support {math.exp(pre_max):.1f} actions")


if __name__ == "__main__":
    main()
