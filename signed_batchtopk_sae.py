from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.topk_sae import act_times_W_dec


class SignedBatchTopK(nn.Module):
    """BatchTopK activation that selects by absolute value and preserves sign.

    Replaces sae_lens's BatchTopK (which applies ReLU first) so that negative
    pre-activations (e.g. the cos θ coordinate of a circle) can be selected and
    passed through intact.
    """

    def __init__(self, k: float):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_samples = x.shape[:-1].numel()
        n_keep = int(self.k * num_samples)
        flat_x = x.flatten()
        topk_result = torch.topk(flat_x.abs(), n_keep, dim=-1)
        return (
            torch.zeros_like(flat_x)
            .scatter(-1, topk_result.indices, flat_x[topk_result.indices])
            .reshape(x.shape)
        )


@dataclass
class SignedBatchTopKSAEConfig(BatchTopKTrainingSAEConfig):
    # Override parent defaults for this architecture.
    # d_sae is auto-computed as d_in * expansion_factor when left at 0.
    # k should be one of K_SWEEP; default 3 is the paper's smallest sweep value.
    d_sae: int = 0           # 0 → set to d_in * expansion_factor in __post_init__
    k: float = 3.0           # override BatchTopKTrainingSAEConfig default of 100
    expansion_factor: int = 4  # c/d = 4 throughout (Appendix E)

    def __post_init__(self) -> None:
        if self.d_sae == 0:
            self.d_sae = self.d_in * self.expansion_factor
        super().__post_init__()  # SAEConfig validates normalize_activations

    @override
    @classmethod
    def architecture(cls) -> str:
        return "signed_batchtopk"


class SignedBatchTopKSAE(BatchTopKTrainingSAE):
    """BatchTopK SAE with signed (absolute-value) feature selection.

    The single most important change from a standard SAE: top-k selection is done
    on |pre_activation| and the sign is preserved. This allows compact representation
    of manifold coordinates that must go negative (cos θ, sin θ, etc.).

    Everything else (AuxK dead-latent loss, EMA threshold for batch-independent eval)
    is inherited from BatchTopKTrainingSAE, with the signed-specific overrides below.
    """

    cfg: SignedBatchTopKSAEConfig

    def __init__(self, cfg: SignedBatchTopKSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return SignedBatchTopK(self.cfg.k)

    @override
    @torch.no_grad()
    def update_topk_threshold(self, feature_acts: torch.Tensor) -> None:
        # Track minimum magnitude of active features (not minimum positive), since
        # active features can be negative.
        active_mask = feature_acts != 0
        lr = self.cfg.topk_threshold_lr
        with torch.autocast(self.topk_threshold.device.type, enabled=False):
            if active_mask.any():
                min_active_mag = (
                    feature_acts[active_mask].abs().min().to(self.topk_threshold.dtype)
                )
                self.topk_threshold = (1 - lr) * self.topk_threshold + lr * min_active_mag

    @override
    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """AuxK loss for signed activations: select dead latents by magnitude."""
        if dead_neuron_mask is None or (num_dead := int(dead_neuron_mask.sum())) == 0:
            return sae_out.new_tensor(0.0)

        residual = (sae_in - sae_out).detach()
        k_aux = min(sae_in.shape[-1] // 2, num_dead)
        scale = min(num_dead / (sae_in.shape[-1] // 2), 1.0)

        # Zero live latents so abs-topk only picks from dead ones (not -inf,
        # which would dominate abs-value selection).
        dead_pre = torch.where(
            dead_neuron_mask[None], hidden_pre, torch.zeros_like(hidden_pre)
        )
        auxk_topk = dead_pre.abs().topk(k_aux, dim=-1, sorted=False)
        auxk_acts = torch.zeros_like(hidden_pre)
        auxk_acts.scatter_(-1, auxk_topk.indices, dead_pre.gather(-1, auxk_topk.indices))

        # b_dec is already in the residual, so reconstruction is W_dec only.
        recons = act_times_W_dec(auxk_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm)
        recons = self.reshape_fn_out(recons, self.d_head)
        return (
            self.cfg.aux_loss_coefficient
            * scale
            * (recons - residual).pow(2).sum(dim=-1).mean()
        )

