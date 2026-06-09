"""
Co-activation-penalised SAE — extends SignedBatchTopKSAE with a geometry penalty
that pushes co-firing decoder atoms toward orthogonality.
"""
from dataclasses import dataclass

import torch
from typing_extensions import override

from sae_lens.saes.sae import TrainStepInput, TrainStepOutput

from .signed_batchtopk_sae import SignedBatchTopKSAE, SignedBatchTopKSAEConfig


@dataclass
class CoActPenalisedSAEConfig(SignedBatchTopKSAEConfig):
    # Penalty coefficient (after warmup).
    beta: float = 0.01
    # Steps over which beta is linearly ramped from 0; early-training C_ema is
    # noise and applying the penalty immediately destabilises training.
    geo_warmup_steps: int = 1000
    # EMA decay for the co-activation matrix; single-batch C is too noisy.
    c_ema_decay: float = 0.99

    @override
    @classmethod
    def architecture(cls) -> str:
        return "coact_penalised_batchtopk"


class CoActPenalisedSAE(SignedBatchTopKSAE):
    """Signed BatchTopK SAE with a co-activation-coherence geometry penalty.

    Adds to the reconstruction + aux loss:

        beta(t) * mean_{i≠j} C_ij * (W_dec_i · W_dec_j)²

    where C_ij is the EMA cosine similarity of per-feature magnitude-activation
    profiles across batches. This pushes decoder atoms that tend to co-fire toward
    orthogonality, nudging the SAE toward compact subspace capture (a small group
    of atoms spanning each manifold) rather than tiling/shattering.

    Key design choices (see implementation_guide.md §4–5 for rationale):
    - C built from |z|, not signed z: signed correlation is ~0 within a manifold.
    - C is detached + EMA'd: not a thing to optimise; we shape geometry Γ, not C.
    - Gram from W_dec (rows = atoms): penalty is on decoder geometry.
    - Diagonal excluded: self-coherence is always 1, not informative.
    - beta warmup: C_ema is uninformative for the first ~warmup_steps batches.

    Reuses the active mask from SignedBatchTopK via output.feature_acts (no extra
    forward pass needed).
    """

    C_ema: torch.Tensor
    cfg: CoActPenalisedSAEConfig

    def __init__(self, cfg: CoActPenalisedSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.register_buffer(
            "C_ema",
            torch.zeros(cfg.d_sae, cfg.d_sae, dtype=torch.float32, device=self.device),
        )

    @torch.no_grad()
    def _update_C_ema(self, feature_acts: torch.Tensor) -> None:
        M = feature_acts.detach().abs()  # (B, G) — magnitudes, not signed values
        Mn = M / M.norm(dim=0, keepdim=True).clamp(min=1e-8)  # column-normalise over batch
        C_batch = Mn.T @ Mn  # (G, G) cosine similarity of magnitude profiles
        decay = self.cfg.c_ema_decay
        self.C_ema = decay * self.C_ema + (1 - decay) * C_batch

    def _compute_geo_penalty(self) -> torch.Tensor:
        # Gram of decoder atoms: off-diagonal entries are cosine similarities when
        # rows are unit-norm (enforced by normalise_decoder() each step).
        G_gram = self.W_dec @ self.W_dec.T  # (G, G)
        off_diag = ~torch.eye(self.cfg.d_sae, dtype=torch.bool, device=self.W_dec.device)
        return (self.C_ema * G_gram.pow(2) * off_diag).sum() / off_diag.sum()

    def _beta(self, step: int) -> float:
        if step < self.cfg.geo_warmup_steps:
            return self.cfg.beta * step / self.cfg.geo_warmup_steps
        return self.cfg.beta

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)

        # Reuse feature_acts from the already-completed forward pass — no extra pass needed.
        self._update_C_ema(output.feature_acts)

        beta_t = self._beta(step_input.n_training_steps)
        L_geo = self._compute_geo_penalty()
        output.losses["geo_loss"] = L_geo
        output.metrics["beta"] = beta_t
        # Add to output.loss so the penalty propagates whether the caller uses
        # output.loss.backward() directly or constructs its own total loss.
        output.loss = output.loss + beta_t * L_geo

        return output
