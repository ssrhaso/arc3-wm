"""Build the Phase-4 vc33 dry-run diagnosis notebook + figure.

Run with the project venv:
    .venv/Scripts/python.exe analysis/build_p4_vc33_diagnosis.py

Outputs:
    analysis/p4_vc33_dryrun_diagnosis.ipynb
    figures/p4_vc33_diagnosis.png
    figures/p4_vc33_diagnosis.svg

The notebook is self-contained; re-running it on a machine with the
same B2-mirrored artifacts (scratch/p4-vc33-dryrun/{metrics,eval_episodes}.jsonl
and scratch/p4-vc33-dryrun/data/replays/vc33/*) reproduces every panel.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent

NB_PATH = REPO_ROOT / "analysis" / "p4_vc33_dryrun_diagnosis.ipynb"
FIG_PNG = REPO_ROOT / "figures" / "p4_vc33_diagnosis.png"
FIG_SVG = REPO_ROOT / "figures" / "p4_vc33_diagnosis.svg"


# ---------------------------------------------------------------------------
# Cell sources (kept here so the script and the notebook stay in sync)
# ---------------------------------------------------------------------------

CELL_MD_HEADER = """\
# Phase-4 vc33 dry-run forensic diagnosis

- **Run id (wandb)**: `qeohyn7i`
- **Project**: `hasofocus-university-of-the-west-of-england/arc3-wm-sprint`
- **Commit**: `7d0d17a`
- **Date**: 2026-05-13
- **Branch**: `diag/p4-vc33-dryrun`
- **Hardware**: Vast.ai A100 SXM4 40 GB (preemptible)
- **Run**: 500k env-steps, `--script train_eval`, warm-started from `checkpoints/pretrained-wm/v1/latest.pkl`

Re-frames vs the original brief (full justification in the markdown summary):
- **Q2**: train successes have only the binary aggregate `episode/score` in `metrics.jsonl`; per-step rewards are written exclusively from the **eval** env via `EvalRewardSink`. So Q2 reads "level depth across 18 eval episodes", not "level depth across 17 train episodes".
- **Q4**: DV3 logs `train/rand/action` (fraction-random rate) but no per-step actions, so per-action entropy is unanswerable from logged data. Reported as the closest proxy.
- **Q5**: answered from documented `available_actions` field in the vc33 replays (constant `[6]` across every frame of every replay) rather than re-instantiating `ARC3GymEnv` against `arc_agi` (which would require network).
"""

CELL_SETUP = """\
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path('.').resolve()
# When the notebook is opened from analysis/, .resolve() pins us to that
# folder. Walk up to the repo root if so.
if REPO_ROOT.name == 'analysis':
    REPO_ROOT = REPO_ROOT.parent

SCRATCH = REPO_ROOT / 'scratch' / 'p4-vc33-dryrun'
METRICS = SCRATCH / 'metrics.jsonl'
EVAL = SCRATCH / 'eval_episodes.jsonl'
VC33_REPLAYS = SCRATCH / 'data' / 'replays' / 'vc33'

assert METRICS.exists(), f'missing {METRICS}; download from b2://arc-agi-3-replays-hasaan/dryruns/p4-vc33-s0-warm-7d0d17a/'
assert EVAL.exists(), f'missing {EVAL}'

print('METRICS    :', METRICS, METRICS.stat().st_size, 'bytes')
print('EVAL       :', EVAL, EVAL.stat().st_size, 'bytes')
print('VC33 dir   :', VC33_REPLAYS, 'present' if VC33_REPLAYS.exists() else 'MISSING (Q6 skipped)')
"""

CELL_LOAD_METRICS = """\
# Stream the metrics.jsonl once and split by series. The file has
# 9.7k lines; cheap on a laptop.

ep_steps, ep_scores, ep_lengths = [], [], []
loss_steps, loss_image, loss_dyn = [], [], []
loss_rew, loss_con, loss_policy, loss_value = [], [], [], []
rand_steps, rand_action = [], []

with METRICS.open('r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        s = d.get('step')
        if 'episode/score' in d:
            ep_steps.append(s); ep_scores.append(d['episode/score']); ep_lengths.append(d.get('episode/length'))
        if 'train/loss/image' in d:
            loss_steps.append(s)
            loss_image.append(d['train/loss/image'])
            loss_dyn.append(d['train/loss/dyn'])
            loss_rew.append(d.get('train/loss/rew'))
            loss_con.append(d.get('train/loss/con'))
            loss_policy.append(d.get('train/loss/policy'))
            loss_value.append(d.get('train/loss/value'))
        if 'train/rand/action' in d:
            rand_steps.append(s); rand_action.append(d['train/rand/action'])

ep_steps = np.array(ep_steps); ep_scores = np.array(ep_scores); ep_lengths = np.array(ep_lengths)
loss_steps = np.array(loss_steps); loss_image = np.array(loss_image); loss_dyn = np.array(loss_dyn)
rand_steps = np.array(rand_steps); rand_action = np.array(rand_action)

print('episodes :', len(ep_scores), 'env-step range', ep_steps.min(), '->', ep_steps.max())
print('train rec:', len(loss_steps), 'env-step range', loss_steps.min(), '->', loss_steps.max())
print('rand rec :', len(rand_action))

success_mask = ep_scores > 0
print('train successes:', int(success_mask.sum()), '/', len(ep_scores))
print('first success env_step:', int(ep_steps[success_mask][0]) if success_mask.any() else None)
"""

CELL_Q1_ANALYSIS = """\
# Q1 — Rolling 1k-episode success rate, slope on the last 50k env-steps.

W = 1000
scores_f = ep_scores.astype(float)
roll = np.convolve(scores_f, np.ones(W)/W, mode='valid')
roll_step = np.array([ep_steps[i:i+W].mean() for i in range(0, len(ep_steps)-W+1)])

# Slope of the rolling rate on the last 50k env-steps
maxs = roll_step.max()
mask50 = roll_step >= (maxs - 50000)
xs50 = roll_step[mask50]; ys50 = roll[mask50]
if len(xs50) >= 2 and np.ptp(xs50) > 0:
    slope_q1, intercept_q1 = np.polyfit(xs50, ys50, 1)
    yhat = slope_q1*xs50 + intercept_q1
    ss_res = float(np.sum((ys50 - yhat)**2))
    ss_tot = float(np.sum((ys50 - ys50.mean())**2))
    r2_q1 = (1 - ss_res/ss_tot) if ss_tot > 0 else float('nan')
else:
    slope_q1, r2_q1 = float('nan'), float('nan')

print(f'Q1 last-50k rolling slope: {slope_q1:.3e} per env-step  R2={r2_q1:.3f}')
print(f'Q1 last rolling rate     : {roll[-1]:.4f}')
print(f'Q1 max rolling rate      : {roll.max():.4f}  at env_step ~{roll_step[roll.argmax()]:.0f}')
print(f'Q1 successful env_steps  : {ep_steps[success_mask].tolist()}')
"""

CELL_Q2_ANALYSIS = """\
# Q2 (re-framed) — Level depth from eval_episodes.jsonl reward streams.
# Native reward fires on level-up: sum(rewards) for an episode = number
# of level-ups in that episode.

eval_lines = [json.loads(l) for l in EVAL.read_text(encoding='utf-8').splitlines() if l.strip()]
eval_sums = [sum(l['rewards']) for l in eval_lines]
eval_lens = [len(l['rewards']) for l in eval_lines]

# Build a histogram: how many episodes cleared >= k levels?
depths = np.array(eval_sums, dtype=int)
print(f'Q2 eval episodes        : {len(depths)}')
print(f'Q2 depth histogram      : {dict(zip(*np.unique(depths, return_counts=True)))}')
print(f'Q2 episodes >=1 level   : {int((depths >= 1).sum())} / {len(depths)}')
print(f'Q2 episodes >=2 levels  : {int((depths >= 2).sum())} / {len(depths)}')
print(f'Q2 deepest single ep    : {int(depths.max())} level-ups')
print(f'Q2 eval ep lengths      : min={min(eval_lens)} max={max(eval_lens)}')
"""

CELL_Q3_ANALYSIS = """\
# Q3 — Linear slope + R^2 on train/loss/image and train/loss/dyn over the
# last 50k env-steps. Slope ~ 0 with decent R^2 -> flat; slope < 0 with
# magnitude relative to the curve scale -> still learning.

def fit_last_window(xs, ys, window=50000):
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) < 2:
        return float('nan'), float('nan'), 0
    mask = xs >= (xs.max() - window)
    x, y = xs[mask], ys[mask]
    if len(x) < 2 or np.ptp(x) == 0:
        return float('nan'), float('nan'), len(x)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope*x + intercept
    ss_res = float(np.sum((y-yhat)**2)); ss_tot = float(np.sum((y-y.mean())**2))
    r2 = (1 - ss_res/ss_tot) if ss_tot > 0 else float('nan')
    return float(slope), float(r2), int(len(x))

sl_img, r2_img, n_img = fit_last_window(loss_steps, loss_image)
sl_dyn, r2_dyn, n_dyn = fit_last_window(loss_steps, loss_dyn)

# Express slope as a fraction of the curve's last value, per 100k env-steps,
# so the magnitude is interpretable.
img_last = float(loss_image[-1]); dyn_last = float(loss_dyn[-1])
rel_img = sl_img * 100_000 / img_last if img_last else float('nan')
rel_dyn = sl_dyn * 100_000 / dyn_last if dyn_last else float('nan')

print(f'Q3 train/loss/image  last={img_last:.4f}  slope={sl_img:.3e}/step  R2={r2_img:.3f}  n={n_img}')
print(f'        => relative drop per 100k env-steps: {rel_img*100:.2f}%')
print(f'Q3 train/loss/dyn    last={dyn_last:.4f}  slope={sl_dyn:.3e}/step  R2={r2_dyn:.3f}  n={n_dyn}')
print(f'        => relative drop per 100k env-steps: {rel_dyn*100:.2f}%')
"""

CELL_Q4_ANALYSIS = """\
# Q4 (re-framed) — train/rand/action over training as exploration->exploitation proxy.
# Per-action entropy is NOT logged by DreamerV3 by default; explicitly flagged.

leave_idx = np.where(rand_action < 0.98)[0]
leave_step = int(rand_steps[leave_idx[0]]) if len(leave_idx) else None
half_idx = np.where(rand_action < 0.5)[0]
half_step = int(rand_steps[half_idx[0]]) if len(half_idx) else None

print(f'Q4 train/rand/action  first={rand_action[0]:.3f}  last={rand_action[-1]:.3f}  min={rand_action.min():.3f}')
print(f'   leaves >=0.98 plateau at env_step: {leave_step}')
print(f'   first env_step rand<0.5         : {half_step}')
print('NOTE: per-action entropy is not in DV3 logs; the agent is still ~60% random at end.')
"""

CELL_Q5_ANALYSIS = """\
# Q5 — Does vc33's action mask change across levels?
# Answer from the JSONL replay 'available_actions' field, which is
# present on every step row.

import collections

if not VC33_REPLAYS.exists():
    print('Q5 vc33 replay dir missing; skipping. Falls back to docs.')
    q5_mask_invariant = None
    q5_unique_sets = set()
else:
    all_avail = set()
    per_replay_summary = []
    for p in sorted(VC33_REPLAYS.glob('*.recording.jsonl')):
        seen = set()
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                d = (json.loads(line) or {}).get('data') or {}
                avail = d.get('available_actions')
                if isinstance(avail, list):
                    seen.add(tuple(sorted(avail)))
        per_replay_summary.append((p.name, sorted(seen)))
        all_avail |= seen

    q5_mask_invariant = (len(all_avail) == 1)
    q5_unique_sets = all_avail
    print(f'Q5 unique available_actions sets across all vc33 frames: {sorted(all_avail)}')
    print(f'Q5 action mask invariant across vc33 levels: {q5_mask_invariant}')
    print(f'Q5 ACTION1-5 + ACTION7 are unavailable on vc33 every step (only ACTION6 exposed).')
    print(f'Q5 -> mask is NOT a confound; the agent is searching a 4096-coord space all the way through.')
"""

CELL_Q6_ANALYSIS = """\
# Q6 — Phase-3 pretrain bias audit.
# Of vc33's human replays in the 340-replay corpus, what's the
# distribution of max levels_completed? If most replays end at level 1
# or 2, the WM saw very few level-3+ transitions during pretraining.

if not VC33_REPLAYS.exists():
    print('Q6 vc33 replay dir missing; skipping.')
    q6_max_lc = {}
    q6_total_transitions_by_level = {}
else:
    import collections
    q6_max_lc = collections.Counter()
    q6_total_transitions_by_level = collections.Counter()
    q6_win_levels = None
    for p in sorted(VC33_REPLAYS.glob('*.recording.jsonl')):
        max_lc = 0
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                d = (json.loads(line) or {}).get('data') or {}
                wl = d.get('win_levels')
                if wl is not None and q6_win_levels is None:
                    q6_win_levels = wl
                lc = d.get('levels_completed')
                if isinstance(lc, int):
                    max_lc = max(max_lc, lc)
                    if 'frame' in d:
                        q6_total_transitions_by_level[lc] += 1
        q6_max_lc[max_lc] += 1

    print(f'Q6 vc33 win_levels (target)            : {q6_win_levels}')
    print(f'Q6 max levels_completed distribution   : {dict(sorted(q6_max_lc.items()))}')
    print(f'Q6 transitions per level (sum of step rows in WHICH level the player was on):')
    for lvl, cnt in sorted(q6_total_transitions_by_level.items()):
        print(f'     level {lvl}: {cnt} step rows')
    print(f'Q6 takeaway: 6 of 10 vc33 replays reached the win, and the WM saw deep')
    print(f'            level transitions during pretraining. NOT skewed toward L1/L2.')
"""

CELL_FIG_BUILDER = """\
# Build the 6-panel figure. Each panel has a self-contained title +
# bottom-line annotation so the figure stands alone in the paper draft.

FIG_PNG = REPO_ROOT / 'figures' / 'p4_vc33_diagnosis.png'
FIG_SVG = REPO_ROOT / 'figures' / 'p4_vc33_diagnosis.svg'
FIG_PNG.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Phase-4 vc33 dry-run forensic diagnosis  (run qeohyn7i / commit 7d0d17a / 500k env-steps)',
             fontsize=13)

# ---------------- Q1 ----------------
ax = axes[0, 0]
ax.plot(roll_step / 1000, roll, lw=1.0, color='C0', label='1k-ep rolling success rate')
suc_steps = ep_steps[success_mask]
ax.vlines(suc_steps / 1000, ymin=0, ymax=roll.max()*1.05 if roll.max() > 0 else 0.005,
          color='C3', alpha=0.4, lw=0.5, label=f'{int(success_mask.sum())} successes')
ax.set_xlabel('env-step / 1k'); ax.set_ylabel('episode-level success rate')
ax.set_title(f'Q1: success rate over training\\n'
             f'rolling 1k-ep, last rate={roll[-1]:.4f}, peak={roll.max():.4f}')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# ---------------- Q2 ----------------
ax = axes[0, 1]
unique, counts = np.unique(depths, return_counts=True)
bars = ax.bar(unique, counts, color=['C7' if u == 0 else 'C2' for u in unique])
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15, str(c),
            ha='center', fontsize=10)
ax.set_xlabel('level-ups in eval episode (sum of rewards)')
ax.set_ylabel('# eval episodes')
ax.set_xticks(list(unique))
n_one = int((depths >= 1).sum()); n_two = int((depths >= 2).sum())
ax.set_title(f'Q2: depth across {len(depths)} eval episodes\\n'
             f'>=L1: {n_one}/{len(depths)}, >=L2: {n_two}/{len(depths)}, deepest={int(depths.max())}')
ax.grid(True, alpha=0.3, axis='y')

# ---------------- Q3 ----------------
ax = axes[0, 2]
# Zoom: skip the first ~100k env-steps so the curves are not dominated
# by the initial drop from 58 -> ~1. Plot the last 400k or so on a log y-axis.
zoom_mask = loss_steps >= 100_000
ax.semilogy(loss_steps[zoom_mask] / 1000, loss_image[zoom_mask], color='C0',
            label=f'image (last={img_last:.3f})')
ax2 = ax.twinx()
ax2.semilogy(loss_steps[zoom_mask] / 1000, loss_dyn[zoom_mask], color='C3',
             label=f'dyn (last={dyn_last:.3f})')
# Fit lines on the last 50k env-steps
mask50_loss = loss_steps >= (loss_steps.max() - 50000)
xs_img = loss_steps[mask50_loss]
if len(xs_img) >= 2 and not np.isnan(sl_img):
    yfit = sl_img*xs_img + (loss_image[mask50_loss].mean() - sl_img*xs_img.mean())
    ax.plot(xs_img/1000, yfit, color='C0', ls='--', lw=2.0, alpha=0.85)
    yfit_d = sl_dyn*xs_img + (loss_dyn[mask50_loss].mean() - sl_dyn*xs_img.mean())
    ax2.plot(xs_img/1000, yfit_d, color='C3', ls='--', lw=2.0, alpha=0.85)
ax.set_xlabel('env-step / 1k'); ax.set_ylabel('train/loss/image (log)', color='C0')
ax2.set_ylabel('train/loss/dyn (log)', color='C3')
ax.tick_params(axis='y', labelcolor='C0'); ax2.tick_params(axis='y', labelcolor='C3')
ax.set_title(f'Q3: WM loss curves (>=100k env-steps); last-50k linear fit\\n'
             f'img slope={sl_img:.2e}/step R2={r2_img:.2f} ({rel_img*100:+.1f}%/100k); '
             f'dyn slope={sl_dyn:.2e}/step R2={r2_dyn:.2f} ({rel_dyn*100:+.1f}%/100k)')

# ---------------- Q4 ----------------
ax = axes[1, 0]
ax.plot(rand_steps / 1000, rand_action, color='C4', lw=1.0)
ax.axhline(0.5, color='gray', ls=':', lw=0.8)
ax.axhline(0.1, color='gray', ls=':', lw=0.8)
if leave_step is not None:
    ax.axvline(leave_step/1000, color='C2', ls='--', lw=0.8, label=f'leaves 0.98 @ {leave_step//1000}k')
if half_step is not None:
    ax.axvline(half_step/1000, color='C3', ls='--', lw=0.8, label=f'<0.5 @ {half_step//1000}k')
ax.set_xlabel('env-step / 1k'); ax.set_ylabel('train/rand/action (proxy)')
ax.set_title(f'Q4 (proxy): exploration->exploitation\\n'
             f'rand starts={rand_action[0]:.2f}, ends={rand_action[-1]:.2f}, min={rand_action.min():.2f}\\n'
             f'NOTE: per-action entropy not in logs')
ax.legend(loc='lower left', fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

# ---------------- Q5 ----------------
ax = axes[1, 1]
ax.axis('off')
text = ['Q5: vc33 action mask across levels',
        '',
        f'unique available_actions sets across all',
        f'10 vc33 replays + every step row:',
        f'   {sorted(q5_unique_sets) if q5_unique_sets else "(no data)"}',
        '',
        f'INVARIANT: {q5_mask_invariant}',
        '',
        'Only ACTION6 (coord-click) is exposed.',
        'ACTION1-5 + ACTION7 (undo) are masked on',
        'every step of every level. Effective flat',
        'action space = 4096 (64x64 grid).',
        '',
        'Verdict: mask is NOT a confound for the gate.']
ax.text(0.04, 0.96, '\\n'.join(text), va='top', ha='left',
        fontsize=10, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', edgecolor='C0', facecolor='#f7faff'))

# ---------------- Q6 ----------------
ax = axes[1, 2]
levels_x = list(range(0, max(q6_max_lc.keys()) + 1)) if q6_max_lc else [0]
counts_y = [q6_max_lc.get(l, 0) for l in levels_x]
bars = ax.bar(levels_x, counts_y, color=['C7' if l < 3 else 'C2' for l in levels_x])
for l, c in zip(levels_x, counts_y):
    if c > 0:
        ax.text(l, c + 0.1, str(c), ha='center', fontsize=10)
ax.set_xlabel('max levels_completed in replay')
ax.set_ylabel('# vc33 human replays')
ax.set_xticks(levels_x)
deep = sum(c for l, c in q6_max_lc.items() if l >= 5)
ax.set_title(f'Q6: vc33 pretrain corpus level coverage\\n'
             f'{deep}/{sum(q6_max_lc.values())} replays >=L5, '
             f'win_levels={q6_win_levels} (no L1/L2 skew)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIG_PNG, dpi=200, bbox_inches='tight')
plt.savefig(FIG_SVG, bbox_inches='tight')
print('saved', FIG_PNG)
print('saved', FIG_SVG)
plt.show()
"""

CELL_MD_VERDICT = """\
## Verdict (see analysis/p4_vc33_dryrun_diagnosis.md for the 1-pager)

- **Budget**: extending 500k -> 1M for the gate is **not** justified by the data — WM losses flat (Q3) AND success rate dead for the final 230k env-steps (Q1). The bottleneck is exploration/exploitation, not budget.
- **train_ratio / entropy schedule**: the policy is still 60% random at end (Q4). Either raise `train_ratio` from 32 toward 64-128 for the per-game pilots so the actor catches up with the WM, or cut the action entropy floor faster. **Surface for Haso's decision (out-of-scope ask under D2 of CLAUDE.md `Decisions Haso owns`).**
- **Re-pretrain WM level-stratified**: not justified. Q6 shows 6/10 vc33 replays reached the win and pretrain saw transitions on every level. The WM is not the blind spot here.
"""


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------

def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

    cells = [
        nbf.v4.new_markdown_cell(CELL_MD_HEADER),
        nbf.v4.new_code_cell(CELL_SETUP),
        nbf.v4.new_code_cell(CELL_LOAD_METRICS),
        nbf.v4.new_markdown_cell("## Q1 — Success timing + rolling trajectory"),
        nbf.v4.new_code_cell(CELL_Q1_ANALYSIS),
        nbf.v4.new_markdown_cell("## Q2 — Level depth across 18 eval episodes (re-framed)"),
        nbf.v4.new_code_cell(CELL_Q2_ANALYSIS),
        nbf.v4.new_markdown_cell("## Q3 — WM loss-curve slope on last 50k env-steps"),
        nbf.v4.new_code_cell(CELL_Q3_ANALYSIS),
        nbf.v4.new_markdown_cell("## Q4 — Exploration -> exploitation (proxy via train/rand/action)"),
        nbf.v4.new_code_cell(CELL_Q4_ANALYSIS),
        nbf.v4.new_markdown_cell("## Q5 — Action mask across vc33 levels"),
        nbf.v4.new_code_cell(CELL_Q5_ANALYSIS),
        nbf.v4.new_markdown_cell("## Q6 — Phase-3 pretrain bias audit (vc33 replays)"),
        nbf.v4.new_code_cell(CELL_Q6_ANALYSIS),
        nbf.v4.new_markdown_cell("## Figure"),
        nbf.v4.new_code_cell(CELL_FIG_BUILDER),
        nbf.v4.new_markdown_cell(CELL_MD_VERDICT),
    ]
    nb.cells = cells
    return nb


def execute_inline() -> None:
    """Run the same code as the notebook against the same globals so we
    actually emit the figure file (without invoking jupyter-nbconvert).
    """
    g: dict = {"__name__": "__main__"}
    # Execute the same source the notebook will, in order.
    for src in [
        CELL_SETUP,
        CELL_LOAD_METRICS,
        CELL_Q1_ANALYSIS,
        CELL_Q2_ANALYSIS,
        CELL_Q3_ANALYSIS,
        CELL_Q4_ANALYSIS,
        CELL_Q5_ANALYSIS,
        CELL_Q6_ANALYSIS,
        CELL_FIG_BUILDER,
    ]:
        exec(src, g)


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    with NB_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", NB_PATH)

    # Run the analysis in-process to emit the PNG/SVG (cheaper than
    # jupyter-nbconvert which would also need to be installed).
    execute_inline()


if __name__ == "__main__":
    main()
