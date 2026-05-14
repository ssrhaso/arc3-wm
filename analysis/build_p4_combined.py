#!/usr/bin/env python3
"""Build aggregation_combined.json + paired markdown + 3-panel figure for
Phase 4 from-scratch (cold) vs Phase 4 proper (warm).

Reads from-scratch logdirs locally, pulls phase4-proper artifacts from B2,
computes RHAE for all 12 runs via ``scripts/compute_rhae.py``, builds the
combined aggregation + the figure + the paired markdown, and mirrors all
five artifacts to B2 under ``phase4-fromscratch/``.

Defaults are tuned for the Vast A100 convention (``/workspace/logdir`` for
training outputs, ``/tmp/proper_pull`` for the B2 cache). Override via
``--logdir`` and ``--proper-pull`` to run from a laptop or alternate box.

Run as::

    python analysis/build_p4_combined.py            # use defaults
    python analysis/build_p4_combined.py --skip-upload  # local only

Inputs:
  {logdir}/p4-fromscratch-{game}-s{seed}-{fs-sha}/{scores,eval_episodes}.jsonl
  b2://{bucket}/phase4-proper/aggregation.json
  b2://{bucket}/phase4-proper/{run}/eval_episodes.jsonl
  data/human_baselines.json, scripts/compute_rhae.py

Outputs (local; B2-mirrored under phase4-fromscratch/ unless --skip-upload):
  _p4_analysis/p4_fromscratch_aggregation.json
  _p4_analysis/p4_aggregation_combined.json
  analysis/p4_fromscratch_vs_proper.md
  figures/p4_fromscratch_vs_proper.{png,svg}
"""
from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_BUCKET = "arc-agi-3-replays-hasaan"
GAMES = ("vc33", "sb26", "cd82")
SEEDS = (0, 1)
TOTAL_STEPS = 500_000
BIN = 10_000
DEFAULT_FS_SHA = "a06c02f"
DEFAULT_PROPER_SHA = "98de390"
DEFAULT_LOGDIR = Path("/workspace/logdir")
DEFAULT_PROPER_PULL = Path("/tmp/proper_pull")

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "_p4_analysis"
FIGURES_DIR = REPO_ROOT / "figures"
COMPARE_MD = REPO_ROOT / "analysis" / "p4_fromscratch_vs_proper.md"
BASELINES = REPO_ROOT / "data" / "human_baselines.json"
COMPUTE_RHAE = REPO_ROOT / "scripts" / "compute_rhae.py"


def fs_run(g, s, sha):
    return f"p4-fromscratch-{g}-s{s}-{sha}"


def warm_run(g, s, sha):
    return f"p4-{g}-s{s}-warm-{sha}"


def b2_download(bucket, remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["b2", "file", "download", f"b2://{bucket}/{remote}", str(local)],
        check=True, capture_output=True,
    )


def b2_upload(bucket, local, remote):
    subprocess.run(
        ["b2", "file", "upload", bucket, str(local), remote],
        check=True, capture_output=True,
    )


def compute_rhae(eval_path, game_id):
    if not eval_path.exists():
        return (0.0, 0)
    r = subprocess.run(
        ["python", str(COMPUTE_RHAE),
         "--episodes-file", str(eval_path),
         "--game-id", game_id,
         "--baselines", str(BASELINES),
         "--step", str(TOTAL_STEPS),
         "--print-metrics"],
        capture_output=True, text=True, check=True,
    )
    rhae = 0.0
    levels = 0
    for line in r.stdout.splitlines():
        if line.startswith(f"eval/rhae/per_game/{game_id}:"):
            rhae = float(line.split(":", 1)[1].strip())
        elif line.startswith(f"eval/rhae/levels_completed/{game_id}:"):
            levels = int(line.split(":", 1)[1].strip())
    return (rhae, levels)


def episode_lengths(eval_path):
    lengths = []
    if not eval_path.exists():
        return lengths
    with eval_path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            d = json.loads(raw)
            lengths.append(max(0, len(d.get("rewards", [])) - 1))
    return lengths


def eval_fields_from_rewards(eval_path):
    """Derive (eval_scores, eval_episode_lengths) from eval_episodes.jsonl.

    For each episode, ``score = sum(rewards) = total levels cleared in that
    episode`` (the rewards stream is +1 per level cleared per arc3_wm/env.py).
    This is the canonical source — ``scores.jsonl`` ``eval/episode/score`` events
    were observed to be unreliable across the from-scratch (cold) runs on
    2026-05-14 (yielded 0 across all eval episodes while the rewards-derived
    RHAE was non-zero). Using rewards uniformly keeps both arms comparable.
    """
    scores, lengths = [], []
    if not eval_path.exists():
        return scores, lengths
    with eval_path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            d = json.loads(raw)
            rewards = d.get("rewards", [])
            scores.append(float(sum(rewards)))
            lengths.append(max(0, len(rewards) - 1))
    return scores, lengths


def aggregate_run(scores_path, eval_path):
    train = []
    if scores_path.exists():
        with scores_path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                d = json.loads(raw)
                step = int(d.get("step", 0))
                if "episode/score" in d:
                    train.append((step, float(d["episode/score"])))
    eval_scores, ep_lens = eval_fields_from_rewards(eval_path)
    bins = {str(s): {"n_episodes": 0, "n_clear_lvl1": 0, "n_clear_lvl2": 0, "max_score": 0.0}
            for s in range(0, TOTAL_STEPS, BIN)}
    for step, score in train:
        bk = str((step // BIN) * BIN)
        if bk not in bins:
            continue
        bins[bk]["n_episodes"] += 1
        if score >= 1:
            bins[bk]["n_clear_lvl1"] += 1
        if score >= 2:
            bins[bk]["n_clear_lvl2"] += 1
        bins[bk]["max_score"] = max(bins[bk]["max_score"], score)
    return {
        "eval_n": len(eval_scores),
        "eval_max": max(eval_scores) if eval_scores else 0.0,
        "eval_clears_ge1": sum(1 for s in eval_scores if s >= 1),
        "eval_clears_ge2": sum(1 for s in eval_scores if s >= 2),
        "eval_mean_score": (sum(eval_scores) / len(eval_scores)) if eval_scores else 0.0,
        "eval_scores": eval_scores,
        "eval_episode_lengths": ep_lens,
        "train_n_episodes": len(train),
        "train_max_score": max((s for _, s in train), default=0.0),
        "train_clears_ge1": sum(1 for _, s in train if s >= 1),
        "bins_10k": bins,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help="B2 bucket name.")
    p.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR,
                   help="Directory containing the cold-arm run subdirs.")
    p.add_argument("--proper-pull", type=Path, default=DEFAULT_PROPER_PULL,
                   help="Scratch dir for B2 downloads of warm-arm artifacts.")
    p.add_argument("--fs-sha", default=DEFAULT_FS_SHA,
                   help="7-char SHA suffix of the cold-arm run names.")
    p.add_argument("--proper-sha", default=DEFAULT_PROPER_SHA,
                   help="7-char SHA suffix of the warm-arm run names.")
    p.add_argument("--skip-upload", action="store_true",
                   help="Generate artifacts locally but do not upload to B2.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bucket = args.bucket
    logdir = args.logdir
    proper_pull = args.proper_pull
    fs_sha = args.fs_sha
    proper_sha = args.proper_sha

    print("=== 1. Pull proper aggregation + eval_episodes from B2 ===")
    proper_pull.mkdir(parents=True, exist_ok=True)
    proper_agg_path = proper_pull / "aggregation.json"
    b2_download(bucket, "phase4-proper/aggregation.json", proper_agg_path)
    for g in GAMES:
        for s in SEEDS:
            rn = warm_run(g, s, proper_sha)
            b2_download(bucket, f"phase4-proper/{rn}/eval_episodes.jsonl",
                        proper_pull / rn / "eval_episodes.jsonl")
    proper = json.loads(proper_agg_path.read_text())

    print("=== 2. Re-derive eval fields + RHAE for proper runs (uniform source: rewards stream) ===")
    for g in GAMES:
        for s in SEEDS:
            rn = warm_run(g, s, proper_sha)
            ev = proper_pull / rn / "eval_episodes.jsonl"
            rhae, lvls = compute_rhae(ev, g)
            eval_scores, ep_lens = eval_fields_from_rewards(ev)
            run = proper["runs"][rn]
            run["rhae"] = rhae
            run["rhae_levels_completed"] = lvls
            run["eval_n"] = len(eval_scores)
            run["eval_max"] = max(eval_scores) if eval_scores else 0.0
            run["eval_clears_ge1"] = sum(1 for v in eval_scores if v >= 1)
            run["eval_clears_ge2"] = sum(1 for v in eval_scores if v >= 2)
            run["eval_mean_score"] = (sum(eval_scores)/len(eval_scores)) if eval_scores else 0.0
            run["eval_scores"] = eval_scores
            run["eval_episode_lengths"] = ep_lens
            print(f"  WARM {rn}: rhae={rhae:.4f} levels={lvls} eval_n={run['eval_n']} clears_ge1={run['eval_clears_ge1']}")

    print("=== 3. Aggregate from-scratch runs ===")
    fs = {"runs": {}}
    for g in GAMES:
        for s in SEEDS:
            rn = fs_run(g, s, fs_sha)
            d = logdir / rn
            agg = aggregate_run(d / "scores.jsonl", d / "eval_episodes.jsonl")
            rhae, lvls = compute_rhae(d / "eval_episodes.jsonl", g)
            agg["rhae"] = rhae
            agg["rhae_levels_completed"] = lvls
            fs["runs"][rn] = agg
            print(f"  COLD {rn}: rhae={rhae:.4f} levels={lvls} "
                  f"eval_n={agg['eval_n']} train_n={agg['train_n_episodes']} "
                  f"train_clears={agg['train_clears_ge1']}")

    print("=== 4. Deltas + combined JSON ===")
    deltas = {}
    for g in GAMES:
        for s in SEEDS:
            w = proper["runs"][warm_run(g, s, proper_sha)]
            c = fs["runs"][fs_run(g, s, fs_sha)]
            deltas[f"{g}-s{s}"] = {
                "rhae_delta": float(c["rhae"]) - float(w["rhae"]),
                "clears_delta": int(c["eval_clears_ge1"]) - int(w["eval_clears_ge1"]),
                "ep_count_delta": int(c["eval_n"]) - int(w["eval_n"]),
            }
    combined = {"pretrained": proper, "fromscratch": fs, "deltas": deltas}
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    fs_path = ANALYSIS_DIR / "p4_fromscratch_aggregation.json"
    fs_path.write_text(json.dumps(fs, indent=2))
    comb_path = ANALYSIS_DIR / "p4_aggregation_combined.json"
    comb_path.write_text(json.dumps(combined, indent=2))
    print(f"  wrote {fs_path}")
    print(f"  wrote {comb_path}")

    print("=== 5. 3-panel figure ===")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    seed_ls = {0: "-", 1: "--"}
    arm_color = {"warm": "#1f77b4", "cold": "#d62728"}
    for ax, game in zip(axes, GAMES):
        for arm_name, agg, run_fn, sha, color in [
            ("warm", proper, warm_run, proper_sha, arm_color["warm"]),
            ("cold", fs, fs_run, fs_sha, arm_color["cold"]),
        ]:
            for seed in SEEDS:
                rn = run_fn(game, seed, sha)
                r = agg["runs"].get(rn)
                if not r:
                    continue
                bins = r.get("bins_10k", {})
                xs, ys = [], []
                for step in range(0, TOTAL_STEPS, BIN):
                    v = bins.get(str(step)) or {}
                    n = v.get("n_episodes", 0)
                    cl = v.get("n_clear_lvl1", 0)
                    xs.append(step / 1000)
                    ys.append((cl / n) if n else 0.0)
                lbl = (f"{arm_name} s{seed} (RHAE={r['rhae']:.3f}, "
                       f"eval {r['eval_clears_ge1']}/{r['eval_n']})")
                ax.plot(xs, ys, color=color, linestyle=seed_ls[seed],
                        linewidth=1.2, alpha=0.85, label=lbl)
                nz = [(x, y) for x, y in zip(xs, ys) if y > 0]
                if nz:
                    ax.scatter([p[0] for p in nz], [p[1] for p in nz],
                               color=color, s=18, alpha=0.7, zorder=3)
        ax.set_title(game)
        ax.set_xlabel("env-step (×1000)")
        ax.set_xlim(0, TOTAL_STEPS / 1000)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    axes[0].set_ylabel("train-time level-1 clear rate\n(10k-step bin)")
    fig.suptitle(
        f"Phase 4: from-scratch vs pretrained, 3 games x 2 seeds x 500k steps\n"
        f"Cold (red) = from-scratch sha {fs_sha}. "
        f"Warm (blue) = Phase 4 proper sha {proper_sha}.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    png = FIGURES_DIR / "p4_fromscratch_vs_proper.png"
    svg = FIGURES_DIR / "p4_fromscratch_vs_proper.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(f"  wrote {png}")
    print(f"  wrote {svg}")

    print("=== 6. Paired markdown ===")
    COMPARE_MD.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# Phase 4: from-scratch (cold) vs pretrained (warm) — paired comparison")
    L.append("")
    L.append("Single A100 SXM4 40GB, 3 games x 2 seeds x 500k env-steps, "
             "stock DV3 `--configs size12m arc3 --script train_eval`.")
    L.append(f"Cold-arm sha `{fs_sha}`. Warm-arm sha `{proper_sha}`.")
    L.append("All configs identical between arms; only `--init-from-ckpt b2://.../pretrained-wm/v1/latest.pkl` removed for cold.")
    L.append("")
    L.append("## Paired table")
    L.append("")
    L.append("| game | seed | warm RHAE | cold RHAE | Δ RHAE | warm levels | cold levels | "
             "warm eval clears/n | cold eval clears/n | warm mean eval ep-len | cold mean eval ep-len |")
    L.append("|------|------|-----------|-----------|--------|-------------|-------------|"
             "--------------------|---------------------|------------------------|------------------------|")
    for g in GAMES:
        for s in SEEDS:
            w = proper["runs"][warm_run(g, s, proper_sha)]
            c = fs["runs"][fs_run(g, s, fs_sha)]
            wl = statistics.mean(w.get("eval_episode_lengths", [0])) if w.get("eval_episode_lengths") else 0.0
            cl = statistics.mean(c.get("eval_episode_lengths", [0])) if c.get("eval_episode_lengths") else 0.0
            L.append(
                f"| {g} | {s} | {w['rhae']:.4f} | {c['rhae']:.4f} | "
                f"{c['rhae']-w['rhae']:+.4f} | {w['rhae_levels_completed']} | "
                f"{c['rhae_levels_completed']} | {w['eval_clears_ge1']}/{w['eval_n']} | "
                f"{c['eval_clears_ge1']}/{c['eval_n']} | {wl:.1f} | {cl:.1f} |"
            )
    L.append("")
    L.append("## Train-time clears (count of episodes that cleared ≥1 level during train)")
    L.append("")
    L.append("| game | seed | warm train clears/n | cold train clears/n |")
    L.append("|------|------|---------------------|---------------------|")
    for g in GAMES:
        for s in SEEDS:
            w = proper["runs"][warm_run(g, s, proper_sha)]
            c = fs["runs"][fs_run(g, s, fs_sha)]
            L.append(f"| {g} | {s} | {w['train_clears_ge1']}/{w['train_n_episodes']} | "
                     f"{c['train_clears_ge1']}/{c['train_n_episodes']} |")
    L.append("")
    L.append("## Deltas (cold - warm)")
    L.append("")
    L.append("| game-seed | Δ RHAE | Δ eval clears | Δ eval ep-count |")
    L.append("|-----------|--------|---------------|------------------|")
    for k, v in deltas.items():
        L.append(f"| {k} | {v['rhae_delta']:+.4f} | {v['clears_delta']:+d} | {v['ep_count_delta']:+d} |")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("- `_p4_analysis/p4_fromscratch_aggregation.json` (cold-only) + "
             "B2 `phase4-fromscratch/aggregation.json`")
    L.append("- `_p4_analysis/p4_aggregation_combined.json` (cold + warm + deltas) + "
             "B2 `phase4-fromscratch/aggregation_combined.json`")
    L.append("- `figures/p4_fromscratch_vs_proper.{png,svg}` + "
             "B2 `phase4-fromscratch/p4_fromscratch_vs_proper.{png,svg}`")
    L.append("")
    COMPARE_MD.write_text("\n".join(L))
    print(f"  wrote {COMPARE_MD}")

    if args.skip_upload:
        print("=== 7. Skipping B2 upload (--skip-upload) ===")
    else:
        print("=== 7. Upload to B2 ===")
        b2_upload(bucket, fs_path, "phase4-fromscratch/aggregation.json")
        b2_upload(bucket, comb_path, "phase4-fromscratch/aggregation_combined.json")
        b2_upload(bucket, png, "phase4-fromscratch/p4_fromscratch_vs_proper.png")
        b2_upload(bucket, svg, "phase4-fromscratch/p4_fromscratch_vs_proper.svg")
        b2_upload(bucket, COMPARE_MD, "phase4-fromscratch/p4_fromscratch_vs_proper.md")
        print("  uploaded 5 artifacts to phase4-fromscratch/")

    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
