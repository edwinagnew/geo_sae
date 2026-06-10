"""
Training loop for the geometric SAE experiments (arXiv 2604.28119, Appendix E).

Matches Appendix E: ℓ1 reconstruction + AuxK reanimation, Adam lr=3e-3, batch=1024,
10 epochs over 2M samples (~20k steps).

Four variants, selected via make_trainer(..., variant=...):
  "baseline"       — TopKTrainingSAE      (per-sample ReLU-TopK; paper's algorithm)
  "baseline_batch" — BatchTopKTrainingSAE (ReLU-BatchTopK; intermediate baseline)
  "signed"         — SignedBatchTopKSAE   (BatchTopK by |value|; sign preserved)
  "penalised"      — CoActPenalisedSAE    (signed + co-activation coherence penalty)

Sweep k over K_SWEEP; use PAPER_CONFIG (d=128, c=512, L0=4, 48 instances) for full runs.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig, TopKTrainingSAE
from sae_lens.saes.sae import TrainStepInput

from .data import EASY_CONFIG, EvalBatch, ManifoldZoo, ZooConfig, build_zoo
from .penalised_sae import CoActPenalisedSAE, CoActPenalisedSAEConfig
from .signed_batchtopk_sae import SignedBatchTopKSAE, SignedBatchTopKSAEConfig

# Sparsity values from Appendix E: "chosen to span all three theoretical regimes"
K_SWEEP: tuple[int, ...] = (3, 4, 6, 8, 10, 14, 16, 20, 25)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    lr: float = 3e-3           # Adam learning rate (Appendix E)
    grad_clip: float = 1.0
    n_epochs: int = 10         # Appendix E: "10 epochs"
    n_train: int = 2_000_000   # Appendix E: "N = 2,000,000 training samples" (fixed dataset)
    batch_size: int = 1024     # Appendix E
    n_eval: int = 1_000_000    # Appendix E: "evaluation set consists of 1,000,000 samples"
    train_seed: int = 0        # seed for training data generation and epoch shuffling
    eval_seed: int = 1         # separate seed (paper: "separate random seed")
    log_every: int = 200
    eval_every: int = 1_000
    device: str = "cpu"
    use_l1_loss: bool = True   # Appendix E: "ℓ1 reconstruction error + dead-neuron reanimation"
    r2_method: str = "centred"  # "centred" | "uncentred" | "both" — passed to process_snapshot


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """Train any BatchTopKTrainingSAE variant on a ManifoldZoo.

    Use make_trainer() rather than constructing directly.
    Call train(return_snapshot=True) to train and return a metrics snapshot.
    """

    def __init__(
        self,
        sae: BatchTopKTrainingSAE,
        zoo: ManifoldZoo,
        cfg: TrainConfig | None = None,
        device: str = "cpu",
        train_data: np.ndarray | None = None,
        eval_data: EvalBatch | None = None,
    ):
        cfg = cfg or TrainConfig(device=device)
        self.sae = sae.to(cfg.device)
        self.zoo = zoo
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

        # Per-batch dead detection (Appendix E): atom is dead if it didn't fire on any
        # sample in the previous batch. Simpler and more faithful than cumulative tracking.
        self.dead_mask = torch.zeros(sae.cfg.d_sae, dtype=torch.bool)

        self.logs: list[dict[str, Any]] = []
        self.norm_scale: float = 1.0

        if train_data is not None:
            self.train_x: np.ndarray = train_data
        else:
            train_rng = np.random.default_rng(cfg.train_seed)
            print(f"Generating {cfg.n_train:,} training samples...", end=" ", flush=True)
            self.train_x = zoo.generate(cfg.n_train, train_rng).x
            print("done.")

        if eval_data is not None:
            self.eval_data: EvalBatch = eval_data
        else:
            eval_rng = np.random.default_rng(cfg.eval_seed)
            self.eval_data = zoo.generate(cfg.n_eval, eval_rng, return_ground_truth=True)

        # scale = √E[‖x‖²] so that E[‖x/scale‖²] = 1 (unit mean-sq norm, Appendix E)
        x_sample = torch.from_numpy(self.train_x[:8192]).float()
        self.norm_scale = x_sample.pow(2).sum(-1).mean().sqrt().item()

    # ── Data and encoding ──────────────────────────────────────────────────────

    def _normalise_input(self, x_np: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x_np).to(self.device) / self.norm_scale

    @torch.no_grad()
    def _encode_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a tensor (N, d) → (N, d_sae) using per-sample TopK.

        Uses absolute-value TopK for signed SAEs and ReLU-TopK for the baseline.
        This is batch-independent (unlike BatchTopK training) and gives exact L0=k.
        """
        sae = self.sae
        k = max(1, int(sae.cfg.k))
        signed = isinstance(sae, SignedBatchTopKSAE)

        chunks = []
        for i in range(0, x.shape[0], 4096):
            xc = x[i : i + 4096]
            x_cent = xc - sae.b_dec if sae.cfg.apply_b_dec_to_input else xc
            pre = x_cent @ sae.W_enc + sae.b_enc
            if sae.cfg.rescale_acts_by_decoder_norm:
                pre = pre * sae.W_dec.norm(dim=-1)

            if signed:
                idx = pre.abs().topk(k, dim=-1).indices
                vals = pre.gather(-1, idx)                 # sign preserved
            else:
                pre_relu = pre.clamp(min=0)
                idx = pre_relu.topk(k, dim=-1).indices
                vals = pre_relu.gather(-1, idx)            # always ≥ 0

            chunks.append(torch.zeros_like(pre).scatter_(-1, idx, vals))
        return torch.cat(chunks)

    # ── Training step ──────────────────────────────────────────────────────────

    def step(self, x_np: np.ndarray, global_step: int) -> dict[str, Any]:
        self.sae.train()
        x = self._normalise_input(x_np)

        step_input = TrainStepInput(
            sae_in=x,
            coefficients={},
            dead_neuron_mask=self.dead_mask.to(self.device),
            n_training_steps=global_step,
            is_logging_step=(global_step % self.cfg.log_every == 0),
        )

        t0 = time.perf_counter()
        output = self.sae.training_forward_pass(step_input)

        l1_loss_val: float | None = None
        if self.cfg.use_l1_loss:
            # ℓ1 reconstruction loss (Appendix E) — decode again from feature_acts so
            # gradients flow through W_enc, W_dec, b_enc, b_dec via the ℓ1 path.
            # output.loss (MSE) is not backpropped; aux loss is reused as-is.
            # Any extra penalty terms (e.g. geo penalty) stored in output.losses are
            # included here so they aren't silently dropped.
            sae_out = self.sae.decode(output.feature_acts)
            l1_recon = (x - sae_out).abs().sum(-1).mean()
            aux = output.losses.get("auxiliary_reconstruction_loss", x.new_tensor(0.0))
            total = l1_recon + aux
            if "geo_loss" in output.losses:
                beta_t = output.metrics.get("beta", 0.0)
                total = total + beta_t * output.losses["geo_loss"]
            total.backward()
            l1_loss_val = l1_recon.item()
        else:
            output.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        with torch.no_grad():
            self.sae.W_dec.data /= self.sae.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.optimizer.zero_grad()
        elapsed = time.perf_counter() - t0

        with torch.no_grad():
            fired = (output.feature_acts.detach() != 0).any(dim=0).cpu()
            self.dead_mask = ~fired

        log = self._collect_metrics(output, x, global_step, elapsed)
        if l1_loss_val is not None:
            log["l1_loss"] = l1_loss_val
        return log

    def _collect_metrics(self, output: Any, x: torch.Tensor, step: int, elapsed: float) -> dict[str, Any]:
        with torch.no_grad():
            fa = output.feature_acts.detach()
            mse = output.losses.get("mse_loss", output.loss).detach().item()
            log: dict[str, Any] = {
                "step": step,
                "loss": output.loss.item(),
                "mse_loss": mse,
                "aux_loss": output.losses.get("auxiliary_reconstruction_loss", torch.tensor(0.0)).item(),
                "mean_l0": (fa != 0).float().sum(-1).mean().item(),
                "fvu": mse / (x.pow(2).sum(-1).mean().item() + 1e-8),
                "n_dead": int(self.dead_mask.sum()),
                "elapsed_ms": elapsed * 1000,
            }
        if "geo_loss" in output.losses:
            log["geo_loss"] = output.losses["geo_loss"].item()
            log["beta"] = float(output.metrics.get("beta", 0.0))
        return log

    # ── Eval ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval(self) -> dict[str, float]:
        """FVU, L0, dead count and neg-activation fraction on the held-out set."""
        self.sae.eval()
        x_eval = self._normalise_input(self.eval_data.x)
        z_all = self._encode_eval(x_eval)
        x_hat = z_all @ self.sae.W_dec + self.sae.b_dec

        mse = (x_hat - x_eval).pow(2).sum(-1).mean().item()
        n_act = (z_all != 0).sum().item()
        return {
            "eval/fvu": mse / (x_eval.pow(2).sum(-1).mean().item() + 1e-8),
            "eval/mse": mse,
            "eval/mean_l0": (z_all != 0).float().sum(-1).mean().item(),
            "eval/n_dead": int((z_all != 0).any(dim=0).logical_not().sum()),
            "eval/neg_frac": float((z_all < 0).sum()) / max(n_act, 1),
        }

    # ── Snapshot export ───────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_snapshot(self) -> dict:
        """Extract all arrays needed by results.py. Call after train()."""
        self.sae.eval()
        codes = self._encode_eval(self._normalise_input(self.eval_data.x)).cpu().numpy()
        active_mask = self.eval_data.active_mask          # (N, m)
        # contributions[j] is already (n_j, d) for active samples only (sparse format)
        contribs = [c / self.norm_scale for c in self.eval_data.contributions]
        return {
            "k":               float(self.sae.cfg.k),
            "eval_codes":      codes,
            "W_dec":           self.sae.W_dec.detach().cpu().numpy(),
            "active_mask":     active_mask,
            "instance_contribs": contribs,
            # gamma = contrib @ V.T: recovers normalised intrinsic coords since
            # contrib = gamma @ V and V has orthonormal rows (V @ V.T = I_{k_i}).
            "instance_coords": [
                c @ self.zoo.instances[j].V.T
                for j, c in enumerate(contribs)
            ],
            "instance_names":  [i.name for i in self.zoo.instances],
            "instance_types":  [i.type for i in self.zoo.instances],
            "instance_ki":     [i.k_i  for i in self.zoo.instances],
            "logs":            [{k: v for k, v in s.items() if k != "elapsed_ms"} for s in self.logs],
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def train(self, return_snapshot: bool = False, include_vis_data: bool = False) -> dict | None:
        """Run n_epochs over the pre-generated training dataset.

        Args:
            return_snapshot: If True, compute and return a compact snapshot dict
                containing all metrics and compressed training logs. Raw eval arrays
                are discarded after metric computation so memory stays bounded.
                Defaults to False — useful for quick training runs where only the
                final loss matters.
            include_vis_data: If True, attach a subsampled visualisation payload
                (2000 samples per manifold) to the snapshot. Requires return_snapshot=True.
                Only meaningful for one k per variant — ~240MB per k in the stored file.
        """
        if include_vis_data and not return_snapshot:
            raise ValueError("include_vis_data=True requires return_snapshot=True")
        cfg = self.cfg
        sae = self.sae
        n_batches = cfg.n_train // cfg.batch_size  # steps per epoch
        n_total_steps = cfg.n_epochs * n_batches
        loss_name = "ℓ1" if cfg.use_l1_loss else "mse"
        print(
            f"\n{'─'*56}\n"
            f"  {type(sae).__name__}  "
            f"d_in={sae.cfg.d_in}  d_sae={sae.cfg.d_sae}  k={sae.cfg.k}\n"
            f"  zoo: {self.zoo.n_instances} instances  L0={self.zoo.L0}\n"
            f"  epochs={cfg.n_epochs}  steps/epoch={n_batches}  total={n_total_steps}\n"
            f"  batch={cfg.batch_size}  lr={cfg.lr}  loss={loss_name}\n"
            f"{'─'*56}\n"
        )

        t_start = time.perf_counter()
        shuffle_rng = np.random.default_rng(cfg.train_seed + 1)
        global_step = 0

        for epoch in range(cfg.n_epochs):
            perm = shuffle_rng.permutation(cfg.n_train)
            t_epoch = time.perf_counter()

            for b in range(n_batches):
                batch_x = self.train_x[perm[b * cfg.batch_size : (b + 1) * cfg.batch_size]]
                log = self.step(batch_x, global_step)
                log["epoch"] = epoch
                self.logs.append(log)

                if global_step % cfg.log_every == 0:
                    loss_str = f"l1={log['l1_loss']:.4f}" if "l1_loss" in log else f"loss={log['loss']:.4f}"
                    parts = [f"ep {epoch+1:>2}/{cfg.n_epochs}  step {global_step:>6}",
                             loss_str,
                             f"L0={log['mean_l0']:.1f}",
                             f"fvu={log['fvu']:.3f}",
                             f"dead={log['n_dead']}"]
                    if "geo_loss" in log:
                        parts += [f"geo={log['geo_loss']:.2e}", f"β={log['beta']:.2e}"]
                    parts.append(f"{log['elapsed_ms']:.1f}ms")
                    print("  ".join(parts))

                if global_step > 0 and global_step % cfg.eval_every == 0:
                    em = self.eval()
                    neg = f"  neg={100*em['eval/neg_frac']:.0f}%" if isinstance(sae, SignedBatchTopKSAE) else ""
                    print(f"  [eval step {global_step}] fvu={em['eval/fvu']:.4f}  "
                          f"L0={em['eval/mean_l0']:.1f}  dead={em['eval/n_dead']}{neg}")
                    self.logs[-1].update(em)

                global_step += 1

            ep_time = time.perf_counter() - t_epoch
            print(f"  [epoch {epoch+1}/{cfg.n_epochs} done]  {ep_time:.1f}s")

        total = time.perf_counter() - t_start
        print(f"\nDone: {total:.1f}s  ({1000*total/n_total_steps:.1f}ms/step)")
        em = self.eval()
        neg = f"  neg={100*em['eval/neg_frac']:.0f}%" if isinstance(sae, SignedBatchTopKSAE) else ""
        self.logs.append({"step": n_total_steps, **em})

        if not return_snapshot:
            print(f"Final: fvu={em['eval/fvu']:.4f}  L0={em['eval/mean_l0']:.1f}  dead={em['eval/n_dead']}{neg}")
            return None

        from .results.results import process_snapshot, build_vis_data
        import math as _math
        snap = self.extract_snapshot()
        k = int(snap["k"])
        result = process_snapshot(k, snap, r2_method=self.cfg.r2_method)
        r2_c = result["r2"]["aggregate_r2"].get(k, float("nan"))
        r2_u = result["r2"].get("uncentred_aggregate_r2", {}).get(k, float("nan"))
        if math.isfinite(r2_u):
            r2_str = f"R²(c)={r2_c:.3f}  R²(u)={r2_u:.3f}"
        else:
            r2_str = f"R²={r2_c:.3f}"
        print(f"Final: fvu={em['eval/fvu']:.4f}  L0={em['eval/mean_l0']:.1f}  "
              f"dead={em['eval/n_dead']}{neg}  {r2_str}")
        if include_vis_data:
            # Scale isolated contribs up to training distribution norm before encoding,
            # then scale codes back so reconstruction is at the original contribution scale.
            #
            # Norm caveat: contribs[j] = Z_j V_j / norm_scale has norm ~1/√L0 relative to
            # a full L0-mixture sample (which was the SAE training distribution). Without
            # scaling, pre-activations are weaker and may miss atoms that only fire above
            # a bias threshold. The √L0 correction restores the typical input magnitude.
            #
            # Centering caveat: _encode_eval subtracts b_dec when apply_b_dec_to_input=True.
            # Here b_dec ≈ 0 because all manifolds are zero-centered by construction
            # ((γ−µ)/σ @ V has zero mean), so this is negligible for this dataset.
            _scale = _math.sqrt(self.zoo.L0)
            device = self.device
            def _encode_isolated(contribs: np.ndarray) -> np.ndarray:
                x = torch.from_numpy((contribs * _scale).astype(np.float32)).to(device)
                z = self._encode_eval(x).cpu().numpy()
                return (z / _scale).astype(np.float32)
            result["vis_data"] = build_vis_data(snap, encode_fn=_encode_isolated)
        return result


# ── Factory ───────────────────────────────────────────────────────────────────

def generate_shared_data(
    zoo_cfg: ZooConfig = EASY_CONFIG,
    train_cfg: TrainConfig | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, EvalBatch]:
    """Generate training and eval data once for reuse across all k values.

    Example::

        train_x, eval_data = generate_shared_data(PAPER_CONFIG, device=device)
        for k in K_SWEEP:
            trainer = make_trainer(PAPER_CONFIG, k=k, train_x=train_x, eval_data=eval_data)
            ...
    """
    zoo = build_zoo(zoo_cfg)
    cfg = train_cfg or TrainConfig(device=device)
    train_rng = np.random.default_rng(cfg.train_seed)
    print(f"Generating {cfg.n_train:,} training samples...", end=" ", flush=True)
    train_x: np.ndarray = zoo.generate(cfg.n_train, train_rng).x
    print("done.")
    eval_rng = np.random.default_rng(cfg.eval_seed)
    print(f"Generating {cfg.n_eval:,} eval samples...", end=" ", flush=True)
    eval_data: EvalBatch = zoo.generate(cfg.n_eval, eval_rng, return_ground_truth=True)
    print("done.")
    return train_x, eval_data


def make_trainer(
    zoo_cfg: ZooConfig = EASY_CONFIG,
    k: float | int | None = None,
    variant: str = "signed",
    train_cfg: TrainConfig | None = None,
    device: str = "cpu",
    train_data: np.ndarray | None = None,
    eval_data: EvalBatch | None = None,
    **sae_kwargs: Any,
) -> Trainer:
    """Build a ready-to-run Trainer.

    variant:
      "baseline"       — TopKTrainingSAE      (per-sample TopK; paper's algorithm)
      "baseline_batch" — BatchTopKTrainingSAE (ReLU-BatchTopK; intermediate baseline)
      "signed"         — SignedBatchTopKSAE   (BatchTopK by |value|; sign preserved)
      "penalised"      — CoActPenalisedSAE    (signed + co-activation coherence penalty)

    k: sparsity budget; use K_SWEEP values for paper experiments.
       Defaults to L0 × mean(k_i) — one atom per manifold dimension.

    d_sae is always 4 × d_in (expansion_factor=4, Appendix E).

    Examples:
        make_trainer(PAPER_CONFIG, k=4)
        make_trainer(PAPER_CONFIG, k=4, variant="penalised")
        make_trainer(PAPER_CONFIG, k=4, variant="baseline")
    """
    if variant not in ("signed", "penalised", "baseline", "baseline_batch"):
        raise ValueError(
            f"variant must be one of 'signed', 'penalised', 'baseline', 'baseline_batch'; got {variant!r}"
        )

    zoo = build_zoo(zoo_cfg)
    train_cfg = train_cfg or TrainConfig(device=device)

    if k is None:
        mean_ki = sum(inst.k_i for inst in zoo.instances) / zoo.n_instances
        k = float(max(2, round(zoo.L0 * mean_ki)))
    else:
        k = float(k)

    d = zoo_cfg.d
    base_kwargs = dict(d_in=d, k=k, dtype="float32", device=train_cfg.device)

    if variant == "baseline":
        sae_cfg = TopKTrainingSAEConfig(
            d_in=d, d_sae=d * 4, k=int(k), device=train_cfg.device,
            rescale_acts_by_decoder_norm=False,
        )
        sae = TopKTrainingSAE(sae_cfg)
    elif variant == "baseline_batch":
        sae_cfg = BatchTopKTrainingSAEConfig(
            d_in=d, d_sae=d * 4, k=k, dtype="float32", device=train_cfg.device,
            rescale_acts_by_decoder_norm=False,
        )
        sae: BatchTopKTrainingSAE = BatchTopKTrainingSAE(sae_cfg)
    elif variant == "penalised":
        sae = CoActPenalisedSAE(CoActPenalisedSAEConfig(**base_kwargs, **sae_kwargs))
    else:
        sae = SignedBatchTopKSAE(SignedBatchTopKSAEConfig(**base_kwargs))

    return Trainer(sae, zoo, train_cfg, train_data=train_data, eval_data=eval_data)
