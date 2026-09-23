# docs/

Documentation for the `arc3-wm` substrate and the world-model study.

## Using the substrate
- `using-the-wrapper.md` - integrate an RL / world-model method against the Gymnasium and DreamerV3-`embodied` interfaces.

## The diagnosis
- `dynamics-competence-probe.md` - the frozen-model probe protocol behind the mechanistic diagnosis.
- `trm-components.md` - the Tiny Recursive Model components used as the supervised control arms.

## See also
Setup and data are documented next to the files they describe:
[`../configs/README.md`](../configs/README.md) (training config blocks),
[`../data/README.md`](../data/README.md) (the RHAE baseline fixture),
[`../scripts/README.md`](../scripts/README.md) (the toolchain), and
[`../tests/README.md`](../tests/README.md) (the test suite). The root
[`../Makefile`](../Makefile) (`make help`) wraps the common laptop
commands as task-runner targets.
