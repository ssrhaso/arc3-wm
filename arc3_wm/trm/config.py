"""Dataclass configs for the TRM component library.

Pure-Python (no torch) so configs can be built, validated, serialised, and
tested on any machine. Every component takes exactly one config object;
composite components (world model, policy) nest the configs of the pieces
they are built from. ``to_dict``/``from_dict`` round-trip through plain JSON
types for CLI flags and run manifests.

Defaults follow the TRM paper (arXiv:2510.04871) and its official repo
(SamsungSAILMontreal/TinyRecursiveModels) unless a deviation is documented
inline. Deviations, with reasons:

- no puzzle-ID embedding: follow-up analysis (arXiv:2512.11847) shows it is
  a brittle per-task memory (wrong ID -> 0 percent); an interactive agent
  must not depend on it. Optional per-game embedding replaces it for
  cross-game training only.
- deep supervision is an explicit inner loop (``n_supervision``), not the
  official repo's carry-across-batches trick (their issue #26 documents the
  paper/code divergence); the explicit form suits transition-level training.
- data augmentation defaults off: ARC-AGI-3 dynamics are not colour- or
  rotation-equivariant (game code branches on specific colours/directions),
  unlike the static ARC puzzles the paper augments.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping

GRID_SIZE = 64
PALETTE_SIZE = 16
# Action-type vocabulary for encoders/heads: RESET is excluded (it is not in
# the flat action space either); ACTION1..ACTION7 -> indices 0..6.
N_ACTION_TYPES = 7
ACTION6_TYPE_INDEX = 5  # ACTION6 (click) position within the 7 types


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass
class TRMCoreConfig:
    """The recursive reasoning core (single shared network, z/y states)."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    expansion: float = 4.0
    seq_mixer: str = "attention"  # "attention" | "mlp"
    pos_encoding: str = "rope"  # "rope" | "none"
    norm_eps: float = 1e-5
    # Recursion schedule: H_cycles blocks of (L_cycles z-updates + 1 y-update);
    # all but the last block run under no_grad, the last backprops fully.
    h_cycles: int = 3
    l_cycles: int = 6
    # Deep supervision / ACT.
    n_supervision: int = 6
    halt_max_steps: int = 6
    halt_exploration_prob: float = 0.1
    # "buffer" = fixed random init states (paper). "input" initialises y from
    # the input embedding, biasing the world model toward copy-then-refine.
    y_init: str = "buffer"
    # Halt-head bias init. The official TRM zero-init (default) lets the
    # head learn both directions; the original sweep-1 runs used -5, which
    # trapped the head in a never-halt regime (measured: 0 percent halt
    # rate at every step). Recorded per-run in run.json.
    halt_bias_init: float = 0.0

    def __post_init__(self) -> None:
        _require(self.d_model % self.n_heads == 0, "d_model must divide n_heads")
        _require(self.seq_mixer in ("attention", "mlp"), f"bad seq_mixer {self.seq_mixer!r}")
        _require(self.pos_encoding in ("rope", "none"), f"bad pos_encoding {self.pos_encoding!r}")
        _require(self.y_init in ("buffer", "input"), f"bad y_init {self.y_init!r}")
        _require(self.h_cycles >= 1 and self.l_cycles >= 1, "cycles must be >= 1")
        _require(self.n_supervision >= 1, "n_supervision must be >= 1")
        _require(1 <= self.halt_max_steps, "halt_max_steps must be >= 1")


@dataclass
class TokenizerConfig:
    """Palette grid -> patch tokens and back."""

    d_model: int = 512
    patch_size: int = 4  # 64/4 -> 16x16 = 256 tokens
    cell_embed_dim: int = 16
    learned_pos: bool = True  # learned 2D positions added at embed time

    def __post_init__(self) -> None:
        _require(GRID_SIZE % self.patch_size == 0, "patch_size must divide 64")

    @property
    def tokens_per_side(self) -> int:
        return GRID_SIZE // self.patch_size

    @property
    def n_tokens(self) -> int:
        return self.tokens_per_side**2

    @property
    def cells_per_patch(self) -> int:
        return self.patch_size**2


@dataclass
class WorldModelConfig:
    """TRM as a discrete next-frame world model.

    The answer state y decodes to per-cell colour logits for the next frame;
    auxiliary heads predict reward (level clear), terminal state, and a
    per-cell change mask. ``changed_cell_weight`` up-weights cells that
    differ between frames in the cross-entropy, so the loss is not dominated
    by the static background (the copy-last-frame degenerate optimum).
    """

    core: TRMCoreConfig = field(default_factory=TRMCoreConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    changed_cell_weight: float = 20.0
    reward_head: bool = True
    state_head: bool = True  # NOT_FINISHED / WIN / GAME_OVER
    change_head: bool = True
    loss: str = "stablemax_ce"  # "stablemax_ce" | "softmax_ce"

    def __post_init__(self) -> None:
        _require(self.core.d_model == self.tokenizer.d_model, "d_model mismatch core/tokenizer")
        _require(self.loss in ("stablemax_ce", "softmax_ce"), f"bad loss {self.loss!r}")
        _require(self.changed_cell_weight >= 1.0, "changed_cell_weight must be >= 1")


@dataclass
class PolicyConfig:
    """TRM as a factored policy over the flat 4102-way action space.

    Factorisation: 7 action-type logits + 4096 click-cell logits; the flat
    logit for ACTION6 at (x, y) is type_logit[ACTION6] + cell_logit[y, x].
    This keeps the head tiny and makes the per-game action mask trivial to
    apply, unlike a monolithic 4102-way softmax.
    """

    core: TRMCoreConfig = field(default_factory=TRMCoreConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    value_head: bool = True
    loss: str = "stablemax_ce"

    def __post_init__(self) -> None:
        _require(self.core.d_model == self.tokenizer.d_model, "d_model mismatch core/tokenizer")
        _require(self.loss in ("stablemax_ce", "softmax_ce"), f"bad loss {self.loss!r}")


@dataclass
class AgentConfig:
    """Composition switches for the online agent.

    Components combine additively in the action score:
      score(a) = w_bc * log pi_BC(a|s)          (if use_bc)
               + w_novelty * novelty(WM(s, a))  (if use_wm)
               + w_reward * P_reward(WM(s, a))  (if use_wm)
               + w_change * P_change(WM(s, a))  (if use_wm)
    With use_wm=False this degrades to a pure BC agent; with use_bc=False to
    a pure model-based novelty planner; with both False to masked random.
    """

    use_bc: bool = True
    use_wm: bool = True
    w_bc: float = 1.0
    w_novelty: float = 1.0
    w_reward: float = 10.0
    w_change: float = 0.5
    epsilon: float = 0.05  # residual masked-uniform exploration
    # >0 switches the BC-only agent to full-distribution sampling at this
    # temperature (the DV3-actor analogue): every step draws from the
    # policy softmax over all actions; epsilon and candidate pruning are
    # bypassed. 0 keeps the default argmax+epsilon path.
    bc_temperature: float = 0.0
    max_click_candidates: int = 64  # salience-pruned ACTION6 candidates per step
    novelty_count_power: float = 0.5  # novelty = 1 / count^power
    # Epistemic bonus when an ensemble of world models is supplied: fraction
    # of cells the members disagree on (Plan2Explore-style, decision-time).
    w_disagree: float = 0.0
    # Supervision steps per WM prediction at decision time. Follow-up
    # analysis (arXiv:2512.11847) finds most accuracy arrives at the first
    # recursion step; 2 trades a little fidelity for ~3x planner speed.
    wm_predict_steps: int = 2
    # BC component scale: "logp" adds w_bc * log pi(a|s) (unbounded below,
    # tends to dominate); "prob" adds w_bc * pi(a|s), commensurate with the
    # [0, 1] novelty/reward terms.
    bc_score: str = "logp"
    # Planning depth: 1 = one-step lookahead; 2 = beam over the top
    # ``beam_width`` first actions, each scored by its best discounted
    # second-step outcome.
    plan_depth: int = 1
    beam_width: int = 8
    plan_discount: float = 0.5
    second_step_candidates: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        _require(0.0 <= self.epsilon <= 1.0, "epsilon must be in [0, 1]")
        _require(self.max_click_candidates >= 1, "max_click_candidates must be >= 1")
        _require(self.wm_predict_steps >= 1, "wm_predict_steps must be >= 1")
        _require(self.bc_score in ("logp", "prob"), f"bad bc_score {self.bc_score!r}")
        _require(self.plan_depth in (1, 2), "plan_depth must be 1 or 2")
        _require(self.beam_width >= 1, "beam_width must be >= 1")
        _require(self.second_step_candidates >= 1, "second_step_candidates must be >= 1")


@dataclass
class GraphAgentConfig:
    """Scientist-loop agent: state-graph exploration with BFS navigation,
    explore/execute phases, object-centric click candidates, optional WM
    no-op pruning. Requires no training; the graph persists across
    episodes (min-over-episodes RHAE rewards explore-then-execute)."""

    max_click_objects: int = 24
    # With a world model attached: defer untested edges whose predicted
    # next state equals the current state with change prob below this
    # threshold (saves real actions on predicted no-ops). 0 disables.
    wm_noop_prune: float = 0.0
    wm_predict_steps: int = 1
    # With a BC policy attached: add w_bc * pi_BC(a|s) to the probe score,
    # steering first-pass exploration along the demonstrated behaviour
    # (decisive on resume-type games where the first clear sets the score).
    w_bc: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        _require(self.max_click_objects >= 1, "max_click_objects must be >= 1")
        _require(0.0 <= self.wm_noop_prune < 1.0, "wm_noop_prune in [0, 1)")


def to_dict(cfg: Any) -> dict:
    """Dataclass config -> plain-JSON dict (recursive)."""
    _require(dataclasses.is_dataclass(cfg), f"not a dataclass: {type(cfg)!r}")
    return dataclasses.asdict(cfg)


_NESTED_FIELDS: Mapping[str, Any] = {"core": TRMCoreConfig, "tokenizer": TokenizerConfig}


def from_dict(cls: type, data: Mapping[str, Any]) -> Any:
    """Plain dict -> config dataclass, rebuilding nested configs.

    Unknown keys raise (typo tripwire); missing keys take defaults.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    _require(not unknown, f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        nested = _NESTED_FIELDS.get(key)
        if nested is not None and isinstance(value, Mapping):
            kwargs[key] = from_dict(nested, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)
