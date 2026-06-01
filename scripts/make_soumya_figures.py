"""Build the figures requested by Soumya (2026-05-24).

Outputs (docs/figures/soumya/):
  fig1_what_is_a_world_model.png  - schematic of the DreamerV3 world model
  fig2_wm_reconstruction.png      - real open-loop reconstruction panel from W&B
  fig3_task_vc33.png              - a generic ARC-AGI-3 task: start -> goal (human replay)
  fig4_solved_vc33.png            - SOLVED by our model: level-1 start -> just-cleared
  fig5_notsolved_lf52.png         - NOT solved by our model: start -> end (unchanged)

Frame sources:
  * Human task pair: data/replays/vc33/<...>.recording.jsonl via arc3_wm.replay_loader.
  * Model rollouts + reconstruction panel: GIFs pulled from W&B into scratch/wandb_pull/.

Pure asset generation; no training, no W&B calls (puller already ran).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PULL = ROOT / "scratch" / "wandb_pull"
OUT = ROOT / "docs" / "figures" / "soumya"
OUT.mkdir(parents=True, exist_ok=True)

VC33_POLICY = PULL / "p4-vc33-s0-warm-98de390/media/videos/epstats/policy_image_493062_12fb3800cf3baea3c1af.gif"
LF52_POLICY = PULL / "p4-lf52-s0-warm-98de390/media/videos/epstats/policy_image_495079_9785f2849034f1c4b309.gif"
VC33_OPENL = PULL / "p4-vc33-s0-warm-98de390/media/videos/report/openloop/image_446772_6108a3c360771222885d.gif"

BG = (245, 245, 247)
INK = (24, 24, 28)
SUB = (90, 90, 96)


def _font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
    except OSError:
        return ImageFont.load_default()


def gif_frames(path: Path):
    im = Image.open(path)
    out = []
    try:
        i = 0
        while True:
            im.seek(i)
            out.append(np.asarray(im.convert("RGB")))
            i += 1
    except EOFError:
        pass
    return out


def upscale(arr: np.ndarray, factor: int) -> Image.Image:
    img = Image.fromarray(arr.astype(np.uint8))
    return img.resize((img.width * factor, img.height * factor), Image.NEAREST)


def _text_centered(draw, cx, y, text, font, fill):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (r - l) / 2, y), text, font=font, fill=fill)


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def before_after(left: Image.Image, right: Image.Image, title, lcap, rcap, caption, name):
    """Two equal panels side by side with an arrow, titled, with a caption."""
    pw, ph = left.size
    gap, margin = 90, 36
    title_h, sub_h = 56, 34
    f_title, f_sub, f_cap = _font(34, True), _font(26), _font(22)

    # caption wrapping to know total height
    cv = Image.new("RGB", (10, 10))
    cd = ImageDraw.Draw(cv)
    inner_w = pw * 2 + gap
    cap_lines = _wrap(cd, caption, f_cap, inner_w)
    cap_h = 14 + len(cap_lines) * 28 + 8

    W = margin * 2 + inner_w
    H = margin + title_h + sub_h + ph + sub_h + cap_h + margin
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    _text_centered(d, W / 2, margin, title, f_title, INK)
    y0 = margin + title_h
    lx, rx = margin, margin + pw + gap
    _text_centered(d, lx + pw / 2, y0, lcap, f_sub, SUB)
    _text_centered(d, rx + pw / 2, y0, rcap, f_sub, SUB)
    iy = y0 + sub_h
    canvas.paste(left, (lx, iy))
    canvas.paste(right, (rx, iy))
    for p in (left, right):
        pass
    d.rectangle([lx, iy, lx + pw, iy + ph], outline=(200, 200, 205), width=2)
    d.rectangle([rx, iy, rx + pw, iy + ph], outline=(200, 200, 205), width=2)
    # arrow
    ay = iy + ph / 2
    d.line([(lx + pw + 16, ay), (rx - 16, ay)], fill=INK, width=6)
    d.polygon([(rx - 16, ay), (rx - 34, ay - 12), (rx - 34, ay + 12)], fill=INK)

    cy = iy + ph + sub_h - 6
    for ln in cap_lines:
        _text_centered(d, W / 2, cy, ln, f_cap, SUB)
        cy += 28
    canvas.save(OUT / name)
    print("wrote", name, canvas.size)


def level_ladder(panels, title, caption, name):
    """N image panels in a row with arrows between, titled, with a caption.

    panels: list of (PIL image, sublabel).
    """
    pw, ph = panels[0][0].size
    gap, margin = 80, 36
    title_h, sub_h = 56, 34
    f_title, f_sub, f_cap = _font(34, True), _font(25), _font(22)
    inner_w = pw * len(panels) + gap * (len(panels) - 1)

    cv = Image.new("RGB", (10, 10)); cd = ImageDraw.Draw(cv)
    cap_lines = _wrap(cd, caption, f_cap, inner_w)
    cap_h = 14 + len(cap_lines) * 28 + 8

    W = margin * 2 + inner_w
    H = margin + title_h + sub_h + ph + sub_h + cap_h + margin
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    _text_centered(d, W / 2, margin, title, f_title, INK)
    y0 = margin + title_h
    iy = y0 + sub_h
    x = margin
    for i, (img, lab) in enumerate(panels):
        _text_centered(d, x + pw / 2, y0, lab, f_sub, SUB)
        canvas.paste(img, (x, iy))
        d.rectangle([x, iy, x + pw, iy + ph], outline=(200, 200, 205), width=2)
        if i < len(panels) - 1:
            ax0, ax1, ay = x + pw + 14, x + pw + gap - 14, iy + ph / 2
            d.line([(ax0, ay), (ax1, ay)], fill=INK, width=6)
            d.polygon([(ax1, ay), (ax1 - 18, ay - 12), (ax1 - 18, ay + 12)], fill=INK)
        x += pw + gap
    cy = iy + ph + sub_h - 6
    for ln in cap_lines:
        _text_centered(d, W / 2, cy, ln, f_cap, SUB)
        cy += 28
    canvas.save(OUT / name)
    print("wrote", name, canvas.size)


# Open-loop panel geometry (detected): 6 columns of 64px content separated by
# red/green border strips; 3 stacked rows (truth / prediction / error).
_OL_COL_X = [2, 70, 138, 206, 274, 342]   # left edge of each 64px column
_OL_ROW_TRUTH = (2, 66)
_OL_ROW_PRED = (66, 130)


def _clean_openloop_grid(frame, n_cols=6, cell_up=6, gap=2):
    """Extract truth + prediction cells (no coloured borders, no error row) and
    re-tile into a clean 2 x n_cols grid on a white background."""
    cells_truth, cells_pred = [], []
    for x in _OL_COL_X[:n_cols]:
        cells_truth.append(frame[_OL_ROW_TRUTH[0]:_OL_ROW_TRUTH[1], x:x + 64])
        cells_pred.append(frame[_OL_ROW_PRED[0]:_OL_ROW_PRED[1], x:x + 64])
    c = 64
    grid = np.full((2 * c + gap, n_cols * c + (n_cols - 1) * gap, 3), 255, np.uint8)
    for j in range(n_cols):
        x0 = j * (c + gap)
        grid[0:c, x0:x0 + c] = cells_truth[j]
        grid[c + gap:2 * c + gap, x0:x0 + c] = cells_pred[j]
    img = Image.fromarray(grid)
    return img.resize((img.width * cell_up, img.height * cell_up), Image.NEAREST)


def reconstruction_panel():
    frames = gif_frames(VC33_OPENL)
    # Use an open-loop ("imagined") frame: the model is NOT shown these real
    # frames yet its prediction still matches (verified diff ~0.3/255). Build a
    # clean truth-vs-prediction grid with the coloured borders/error row removed.
    panel = _clean_openloop_grid(frames[28])
    pw, ph = panel.size
    mid = ph // 2
    factor = 1  # already upscaled in _clean_openloop_grid
    pw, ph = panel.size
    lab_w, margin, title_h = 250, 36, 124
    f_title, f_sub, f_row, f_cap = _font(34, True), _font(22), _font(24, True), _font(21)

    cv = Image.new("RGB", (10, 10)); cd = ImageDraw.Draw(cv)
    caption = ("Top row: real game frames. Bottom row: the world model's predictions of them, made "
               "open-loop - the model is not shown the real frames, it generates them from its own "
               "internal state. They match, which is evidence the model has learned vc33's dynamics.")
    cap_lines = _wrap(cd, caption, f_cap, lab_w + pw)
    cap_h = len(cap_lines) * 27 + 16

    CW = margin * 2 + lab_w + pw
    CH = margin + title_h + ph + cap_h + margin
    canvas = Image.new("RGB", (CW, CH), BG)
    d = ImageDraw.Draw(canvas)
    title = 'What the world model "sees": open-loop prediction on vc33'
    for i, ln in enumerate(_wrap(d, title, f_title, CW - 2 * margin)):
        _text_centered(d, CW / 2, margin + i * 40, ln, f_title, INK)

    px, py = margin + lab_w, margin + title_h
    canvas.paste(panel, (px, py))
    d.rectangle([px, py, px + pw, py + ph], outline=(200, 200, 205), width=2)

    # row labels
    d.text((margin, py + mid * factor / 2 - 30), "Real game", font=f_row, fill=INK)
    d.text((margin, py + mid * factor / 2 - 4), "(ground truth)", font=f_sub, fill=SUB)
    d.text((margin, py + mid * factor + mid * factor / 2 - 30), "World model's", font=f_row, fill=INK)
    d.text((margin, py + mid * factor + mid * factor / 2 - 4), "prediction", font=f_row, fill=INK)
    # divider line between rows
    d.line([(px, py + mid * factor), (px + pw, py + mid * factor)], fill=(255, 255, 255), width=2)

    cy = py + ph + 12
    for ln in cap_lines:
        d.text((margin, cy), ln, font=f_cap, fill=SUB)
        cy += 27
    canvas.save(OUT / "fig2_wm_reconstruction.png")
    print("wrote fig2_wm_reconstruction.png", canvas.size)


def schematic():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, tc="#18181c", fs=12, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1, text=None, color="#18181c", rad=0.0, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                                     lw=2, color=color, connectionstyle=f"arc3,rad={rad}",
                                     linestyle=ls))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.22, text, ha="center", va="bottom",
                    fontsize=10.5, color=color)

    blue, green, amber, gray = "#1E93FF", "#4FCC30", "#FF851B", "#666666"
    ax.text(6, 6.05, "The world model (DreamerV3) used in this project", ha="center",
            fontsize=15, fontweight="bold", color="#18181c")

    box(0.3, 3.7, 1.9, 1.2, "Game frame\n(64x64 grid)", "#ffffff", gray, fs=11)
    box(2.8, 3.7, 1.7, 1.2, "Encoder\n(CNN)", "#eaf4ff", blue, fs=11)
    box(5.1, 3.4, 2.3, 1.8, "Latent state\n$z_t$\n(compressed\nmemory)", "#eaffe6", green, fs=12)
    # prediction heads
    box(8.2, 5.0, 3.4, 0.95, "Decoder -> predicted next frame", "#fff7e8", amber, fs=11)
    box(8.2, 3.85, 3.4, 0.95, "Reward head -> will I score?", "#fff7e8", amber, fs=11)
    box(8.2, 2.7, 3.4, 0.95, "Continue head -> game over?", "#fff7e8", amber, fs=11)

    arrow(2.2, 4.3, 2.8, 4.3)
    arrow(4.5, 4.3, 5.1, 4.3)
    arrow(7.4, 4.5, 8.2, 5.45, rad=-0.15)
    arrow(7.4, 4.3, 8.2, 4.32)
    arrow(7.4, 4.1, 8.2, 3.15, rad=0.15)

    # recurrence: action feeds the latent forward
    arrow(6.25, 3.4, 6.25, 2.2, color=blue, rad=0.0)
    box(5.0, 1.2, 2.5, 0.95, "+ action $a_t$", "#eaf4ff", blue, fs=11)
    arrow(5.0, 1.65, 3.0, 1.65, color=blue)
    arrow(2.8, 1.9, 4.9, 3.45, color=blue, rad=0.15,
          text="predict next latent  $z_{t+1}$")

    # imagination box
    box(0.3, 0.15, 4.3, 0.8, "Actor + critic train inside imagined rollouts (no real game needed)",
        "#f3ecff", "#A356D6", fs=10.5)
    ax.text(9.9, 1.9, "These predictions let the\nagent plan by imagining\nfuture moves before acting.",
            ha="center", va="center", fontsize=10.5, color=SUB if False else "#5a5a60")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_what_is_a_world_model.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig1_what_is_a_world_model.png")


def _human_rows():
    """Raw step rows (with frames) of the 7-level human win replay."""
    import json
    rp = ROOT / "data/replays/vc33/837812ad-2c5c-4943-8632-b6f8cd4b5b4d.recording.jsonl"
    rows = []
    for line in open(rp):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line).get("data", {})
        if "frame" in d:
            rows.append(d)
    return rows


def _hframe(rows, i):
    from arc3_wm.palette import decode_frame
    return decode_frame(np.asarray(rows[i]["frame"][-1]))


def task_and_rollouts():
    rows = _human_rows()
    # ---- SOLVED by our model: a 3-step progression ----
    # Panel 1 is the genuine first observation of vc33 level 1 (verified pixel-
    # identical to env.reset() and to the model's own frame 0). Panel 2 is the
    # level-1 solved/goal state (the level-up happens on the next replay row).
    # Panel 3 is a real frame from the trained model's play - level 2, i.e. it
    # cleared level 1 (verified by matching the human level dictionary).
    vw = gif_frames(VC33_POLICY)        # trained policy -> level 2 (cleared level 1)
    level_ladder(
        [
            (upscale(_hframe(rows, 0), 7), "Level 1 - start"),
            (upscale(_hframe(rows, 6), 7), "Level 1 - solved"),
            (upscale(vw[469], 7), "Level 2 - reached by our model"),
        ],
        "Our model clears level 1 of vc33",
        "Level 1 starts (left) and is solved by reaching its goal (centre); that advances the player "
        "to level 2 (right). Our trained model does this - it clears level 1 and reaches level 2 "
        "(the right panel is a real frame from its own play). Confirmed in evaluation: it cleared "
        "level 1 in 4 of 23 episodes (RHAE > 0). It then plateaus at level 2.",
        "fig4_solved_vc33.png",
    )

    # ---- NOT solved by our model: sb26 (a colour-matching puzzle) ----
    sb_gif = next((PULL / "p4-sb26-s0-warm-98de390/media/videos/epstats").glob("*.gif"))
    sb = gif_frames(sb_gif)
    before_after(
        upscale(sb[0], 8), upscale(sb[-1], 8),
        "A task our model does NOT solve (sb26)",
        "Start", "After the agent acts",
        "sb26 asks the player to fill the slots to match the target tiles (top row). Our agent does "
        "act - it places a couple of tiles (right) - but never completes the puzzle correctly: it "
        "scored 0 in all 24 evaluation episodes. This is the typical failure: plausible moves, no "
        "solution.",
        "fig5_notsolved_sb26.png",
    )


def activity_timeline():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def diffs(path):
        fs = gif_frames(path)
        a = np.stack(fs).astype(np.int16)
        return np.abs(a[1:] - a[:-1]).reshape(len(fs) - 1, -1).mean(axis=1)

    dv, dl = diffs(VC33_POLICY), diffs(LF52_POLICY)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), dpi=150, sharex=False)
    a1.plot(dv, color="#4FCC30", lw=1.3)
    a1.set_title("vc33  -  our model engages with the game", fontsize=13, fontweight="bold", loc="left")
    a1.set_ylabel("grid change\nper step")
    spikes = [i for i in range(len(dv)) if dv[i] > 4]
    a1.scatter(spikes, dv[spikes], color="#F93C31", zorder=5, s=28)
    a1.annotate("grid fully redraws\n(deaths / retries at its ceiling level)",
                xy=(spikes[0], dv[spikes[0]]), xytext=(spikes[0] + 50, dv.max() * 0.78),
                fontsize=10, color="#F93C31",
                arrowprops=dict(arrowstyle="->", color="#F93C31"))
    a2.plot(dl, color="#1E93FF", lw=1.3)
    a2.set_title("lf52  -  our model is inert", fontsize=13, fontweight="bold", loc="left")
    a2.set_ylabel("grid change\nper step")
    a2.set_xlabel("step within one rollout")
    a2.annotate("the grid barely changes - the agent never gets anywhere",
                xy=(len(dl) * 0.5, dl.max()), xytext=(len(dl) * 0.22, dv.max() * 0.55),
                fontsize=10, color="#1E93FF",
                arrowprops=dict(arrowstyle="->", color="#1E93FF"))
    for ax in (a1, a2):
        ax.set_ylim(0, dv.max() * 1.1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("How much the game changes as our trained agent plays",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_activity_timeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig6_activity_timeline.png")


def export_raw():
    """Write the clean underlying frames (no titles/arrows/captions) to RAW/.

    Upscaled 8x nearest-neighbour (lossless pixel scaling) so they stay crisp
    in the paper. Names map to the composed figures.
    """
    raw = OUT / "RAW"
    raw.mkdir(parents=True, exist_ok=True)
    rows = _human_rows()
    vw = gif_frames(VC33_POLICY)
    sb = gif_frames(next((PULL / "p4-sb26-s0-warm-98de390/media/videos/epstats").glob("*.gif")))

    items = {
        # fig4 - 3-step progression: L1 start -> L1 solved -> L2 (model)
        "fig4_left_vc33_level1_start.png": upscale(_hframe(rows, 0), 8),
        "fig4_mid_vc33_level1_solved.png": upscale(_hframe(rows, 6), 8),
        "fig4_right_vc33_trained_level2.png": upscale(vw[469], 8),
        # fig5 - not solved by our model (sb26)
        "fig5_left_sb26_start.png": upscale(sb[0], 8),
        "fig5_right_sb26_after.png": upscale(sb[-1], 8),
    }
    for name, img in items.items():
        img.save(raw / name)
        print("wrote RAW/", name, img.size)

    # fig2 - clean truth-vs-prediction grid (borders + error row removed), no labels.
    _clean_openloop_grid(gif_frames(VC33_OPENL)[28]).save(
        raw / "fig2_wm_reconstruction_panel.png")
    print("wrote RAW/ fig2_wm_reconstruction_panel.png")


if __name__ == "__main__":
    schematic()
    reconstruction_panel()
    task_and_rollouts()
    export_raw()
    # activity_timeline() retired: the not-solved example (sb26) is itself active,
    # so the "engagement vs inertness" contrast no longer applies.
    print("\nAll figures in", OUT)
