# docs/

Documentation for the `arc3-wm` substrate and the world-model study.

## Using the substrate
- `using-the-wrapper.md` - integrate an RL / world-model method against the Gymnasium and DreamerV3-`embodied` interfaces.
- `replay-format.md` - schema of the human-replay JSONL dataset consumed by the offline loader.
- `compute-runbook.md` - running training (Phases 2-5), including the Vast.ai path.
- `vast-quickstart.md` - minimal Vast.ai DreamerV3 sanity + pretraining quickstart.

## Design & rationale
- `design-decisions.md` - committed design choices (action space, encoder, RHAE semantics) and why.
- `harness-analysis.md` - analysis of the `arc_agi` harness.
- `phase-checklists.md` - per-phase exit criteria.
- `phase4-warmstart-notes.md` - warm-start checkpoint layout expected by `scripts/launch_pergame.py`.

## Paper
- `contribution.md` - workshop-paper claims->evidence skeleton.
- `figures/soumya/` - composed paper figures and their provenance notes.
- `arc-agi-3/` - fetched upstream ARC-AGI-3 reference docs.

## See also
Setup and data are documented next to the files they describe:
[`../configs/README.md`](../configs/README.md) (training config blocks),
[`../data/README.md`](../data/README.md) (the RHAE baseline fixture), and
[`../tests/README.md`](../tests/README.md) (the test suite). The root
[`../Makefile`](../Makefile) (`make help`) wraps the common laptop
commands as task-runner targets.
