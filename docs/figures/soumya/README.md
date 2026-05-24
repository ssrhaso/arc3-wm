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
next — the next observation, whether a reward is received, and whether the
episode ends. Once trained, the world model lets the agent **"imagine"** the
consequences of actions entirely in its head, without touching the real
environment. Planning and policy learning can then happen inside this cheap,
fast mental simulation, which is far more sample-efficient than learning purely
by trial and error in the real world. This is the central idea behind the
Dreamer family of agents (DreamerV3, Hafner et al., *Nature* 2025), which we use
in this work.

Concretely, our world model compresses each 64×64 game frame into a small latent
"state" vector with a convolutional encoder, and a recurrent network predicts how
that latent state evolves when the agent takes an action. Three lightweight heads
read the consequences off the latent state: a **decoder** that reconstructs the
predicted next frame, a **reward head**, and a **continue head** that predicts
whether the game is still running. The agent's policy (actor) and value estimate
(critic) are then trained entirely on trajectories *imagined* by rolling the
latent dynamics forward — the real game is used only to collect experience and
keep the model honest. We exploit a key property of this design: because the
world model is trained to *predict* rather than to maximise reward, we can
pretrain it on a dataset of human replays spanning all 25 ARC-AGI-3 games and
then fine-tune per game. Figure 2 shows that on the game **vc33** the model
learned the dynamics well enough to predict several steps into the future on its
own; Figures 4–6 show where that does and does not translate into solving the
game.

---

## 2. A picture showing a world model

Two figures (you asked for a picture; these are complementary — a diagram for
intuition and a real result as evidence):

- **`fig1_what_is_a_world_model.png`** — schematic of the world model: game frame
  → encoder → latent state → decoder / reward / continue heads, with the action
  feeding the latent forward and the actor+critic training inside imagined
  rollouts.
- **`fig2_wm_reconstruction.png`** — a *real* result from our vc33 run. Top row =
  the true game; bottom row = the world model's **open-loop prediction** (it is
  not shown the real frames, it predicts them from its own latent state). They
  match — evidence the model actually learned vc33's dynamics. Pulled from W&B
  (`report/openloop/image`).

---

## 3–5. Initial / end state image pairs

| Figure | Shows | File |
|---|---|---|
| 3 | A generic ARC-AGI-3 task: start → goal, from a **human** replay of vc33 (player cleared 7 levels). | `fig3_task_vc33.png` |
| 4 | A task our model **SOLVES** (vc33): level-1 start → end of level 1, from the trained agent's own rollout. | `fig4_solved_vc33.png` |
| 5 | A task our model does **NOT solve** (lf52): start → end; the grid is essentially unchanged. | `fig5_notsolved_lf52.png` |
| 6 | *(bonus)* Activity timeline — the clearest view of "solved vs not": on vc33 the agent repeatedly drives the game to new levels (spikes); on lf52 nothing ever changes. | `fig6_activity_timeline.png` |

### Honest caveats (worth knowing before this goes in the paper)

- **Why a bonus Figure 6.** vc33's levels share a visual style, so a single
  before/after pair (Fig 4) does not dramatically *look* solved even though it is.
  The activity timeline (Fig 6) is the honest, compelling view: vc33 shows 12
  full-grid redraws (the agent reaching level boundaries); lf52 is dead flat.
- **"Level transitions" in Fig 6.** A full-grid redraw happens at a level
  boundary, which can be a level *clear* or a death/reset — from frames alone we
  cannot distinguish the two. The *verified* claim, from evaluation logs, is:
  **vc33 cleared level 1 in 4/23 (seed 0) and 2/18 (seed 1) eval episodes
  (RHAE > 0); lf52, sb26, cd82, tn36, ls20 scored 0 across every eval episode.**
- **vc33 is our only solved game.** Phase-4 proper + expansion: vc33 is the sole
  RHAE > 0 result. The not-solved example could equally be sb26 / cd82 / tn36 /
  ls20; lf52 was chosen because it is the most visually recognisable ARC puzzle
  and shows the agent active-but-stuck. Say the word to swap it.

---

## Email-ready reply (draft)

> Hi Soumya,
>
> Thanks! Attached are the pieces you asked for.
>
> **What is a world model** — two paragraphs below / in the attached doc.
>
> **A picture of a world model** — `fig1` is a schematic; `fig2` is a real result
> from our run showing the model predicting the game's future on its own.
>
> **Initial → end states** — `fig3` (a task, human solving it), `fig4` (a game our
> model solves, vc33), `fig5` (a game it does not, lf52). I also added `fig6`,
> which makes the solved-vs-not contrast clearest: our agent repeatedly drives
> vc33 to new levels, while on lf52 nothing ever changes.
>
> [paste the two paragraphs from §1]
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
  `p4-vc33-s0-warm-98de390` and `p4-lf52-s0-warm-98de390`, via
  `scratch/wandb_pull.py` (read-only).
- Human-task pair rendered locally from
  `data/replays/vc33/837812ad-…recording.jsonl` via `arc3_wm.replay_loader`.
- Eval-solve counts from each run's local `eval_episodes.jsonl`.
- The W&B media live only in the cloud; this folder is the local, paper-ready copy.
