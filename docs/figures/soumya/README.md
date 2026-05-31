# Deliverables for Soumya (2026-05-24)

Everything Soumya asked for, ready to drop into the paper / Overleaf or paste
into an email reply. All figures are in this folder; regenerate any of them with
`python scripts/make_soumya_figures.py`.

Soumya's request was:

1. Two paragraphs on *what is a world model*.
2. A picture showing a world model.
3. Initial + end state of an ARC-AGI task.
4. Initial + end state of a task that **is solved** by our model.
5. Initial + end state of a task that is **not solved** by our model.

---

## 1. What is a world model? (two paragraphs)

A **world model** is a learned, internal simulator of an environment. Rather than
learning only *what to do*, an agent with a world model learns *how its world
works*: given the current situation and an action, it predicts what will happen
next - the next observation, whether a reward is received, and whether the
episode ends. Once trained, the world model lets the agent **"imagine"** the
consequences of actions entirely in its head, without touching the real
environment. Planning and policy learning can then happen inside this cheap,
fast mental simulation, which is far more sample-efficient than learning purely
by trial and error in the real world. This is the central idea behind the
Dreamer family of agents (DreamerV3, Hafner et al., *Nature* 2025), which we use
in this work.

Concretely, our world model compresses each 64x64 game frame into a small latent
"state" vector with a convolutional encoder, and a recurrent network predicts how
that latent state evolves when the agent takes an action. Three lightweight heads
read the consequences off the latent state: a **decoder** that reconstructs the
predicted next frame, a **reward head**, and a **continue head** that predicts
whether the game is still running. The agent's policy (actor) and value estimate
(critic) are then trained entirely on trajectories *imagined* by rolling the
latent dynamics forward - the real game is used only to collect experience and
keep the model honest. We exploit a key property of this design: because the
world model is trained to *predict* rather than to maximise reward, we can
pretrain it on a dataset of human replays spanning all 25 ARC-AGI-3 games and
then fine-tune per game. Figure 2 shows that on the game **vc33** the model
learned the dynamics well enough to predict several steps into the future on its
own; Figures 4-6 show where that does and does not translate into solving the
game.

---

## 2. A picture showing a world model

Two figures (you asked for a picture; these are complementary - a diagram for
intuition and a real result as evidence):

- **`fig1_what_is_a_world_model.png`** - schematic of the world model: game frame
  -> encoder -> latent state -> decoder / reward / continue heads, with the action
  feeding the latent forward and the actor+critic training inside imagined
  rollouts.
- **`fig2_wm_reconstruction.png`** - a *real* result from our vc33 run. Top row =
  the true game; bottom row = the world model's **open-loop prediction** (it is
  not shown the real frames, it predicts them from its own latent state). They
  match - evidence the model actually learned vc33's dynamics. Pulled from W&B
  (`report/openloop/image`).

---

## 4-5. Solved / not-solved image sets

| Figure | Shows | File |
|---|---|---|
| 4 | A task our model **SOLVES** (vc33), as a 3-step progression: Level 1 start -> Level 1 solved -> Level 2 (reached by our model). Left = the verified first frame; right = a real trained-model frame. | `fig4_solved_vc33.png` |
| 5 | A task our model does **NOT solve** (sb26, a colour-matching puzzle): start -> after the agent acts. It places a couple of tiles but never completes the puzzle. | `fig5_notsolved_sb26.png` |

(The standalone "generic ARC task" figure was dropped - its solved frame now
lives as the centre panel of fig 4, so fig 4 itself shows the task and its
solution. fig 4 and fig 5 are concrete tasks, so "what a task looks like" is
already covered.)

### `RAW/` - annotation-free images for the paper

`RAW/` holds the underlying frames with **no titles, arrows, borders, or
captions** (8x nearest-neighbour upscaled, 512x512), to drop into the paper and
caption yourself. Names map to the composed figures:

- `fig4_left_vc33_level1_start.png`, `fig4_mid_vc33_level1_solved.png`, `fig4_right_vc33_trained_level2.png`
- `fig5_left_sb26_start.png`, `fig5_right_sb26_after.png`
- `fig2_wm_reconstruction_panel.png` (clean 2x6 grid: top row = real game frames,
  bottom row = the WM's open-loop predictions of them; DreamerV3's coloured phase
  borders and the error row have been stripped out)

`fig1` (the schematic) has no annotation-free version - it is a labelled diagram.

### How fig 4 was verified (and a caveat)

- **Left ("Level 1 - start")** is the genuine first observation of vc33 level 1 -
  verified **pixel-identical** to `env.reset()`, to the human replay's row 0, and
  to the trained model's own frame 0.
- **Centre ("Level 1 - solved")** is the level-1 goal configuration (human replay
  row 6; the level-up to level 2 happens on the next row).
- **Right ("Level 2 - reached by our model")** is a real frame from the trained
  policy's rollout. We confirmed it is level 2 (not level 1) by matching every
  rollout frame to the human level dictionary: vc33 rollouts are 100% level 1 up
  to ~step 226k and 100% level 2 from ~step 271k - i.e. the model learns to clear
  level 1. No single rollout captured the level-1 -> solved transition, which is
  why the centre comes from the replay rather than the model.

**Visual caveat:** in vc33 the level-1 start and solved states look nearly
identical (solving is a small, precise change); the obvious visual jump is to
level 2.

Caveats worth knowing before this goes in the paper:

- **The model clears level 1, and only level 1.** Evidence: rollouts sit at
  level 2 (above), and **eval reward caps at exactly 1 level cleared in every vc33
  run** (warm s0 4/23, from-scratch s0 2/21, from-scratch s1 3/19). One
  from-scratch *training* rollout visually resembled level 3 (two levels cleared),
  but since no eval episode ever cleared two levels we deliberately **do not**
  claim level 3 in the figure.
- **vc33 is our only solved game.** Phase-4 proper + expansion: vc33 is the sole
  RHAE > 0 result. sb26/lf52/cd82/tn36/ls20 scored **0** across every eval episode.
  sb26 was chosen for the not-solved figure because it is a clear colour-matching
  puzzle where the agent visibly acts (places tiles) but never solves it - the
  typical failure. (lf52/ls20 agents are nearly inert, which reads as "did
  nothing"; cd82/tn36 are sparser. Swap on request.)

---

## Email-ready reply (draft)

> Hi Soumya,
>
> Thanks! Attached are the pieces you asked for.
>
> **What is a world model** - two paragraphs below / in the attached doc.
>
> **A picture of a world model** - `fig1` is a schematic; `fig2` is a real result
> from our run showing the model predicting the game's future on its own.
>
> **Initial -> end states** - `fig4` (a game our model solves: level 1 start ->
> level 1 solved -> level 2, reached by our model), `fig5` (a game it does not,
> sb26 - the agent places a couple of tiles but never completes the puzzle).
>
> [paste the two paragraphs from Section 1]
>
> Happy to adjust any of these.
>
> Best,
> Haso

---

## Provenance / reproducibility

- Figures generated by [`scripts/make_soumya_figures.py`](../../../scripts/make_soumya_figures.py).
- Model rollouts + reconstruction panel pulled from W&B project
  `hasofocus-university-of-the-west-of-england/arc3-wm-sprint`, runs
  `p4-vc33-s0-warm-98de390` (solved) and `p4-sb26-s0-warm-98de390` (not solved),
  via `scratch/wandb_pull.py` (read-only).
- Human-task pair rendered locally from
  `data/replays/vc33/837812ad-...recording.jsonl` via `arc3_wm.replay_loader`.
- Eval-solve counts from each run's local `eval_episodes.jsonl`.
- The W&B media live only in the cloud; this folder is the local, paper-ready copy.
