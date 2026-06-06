"""
Training loop for the geometric SAE experiments (arXiv 2604.28119, Appendix E).

Matches Appendix E: ℓ1 reconstruction + AuxK reanimation, Adam lr=3e-3, batch=1024,
10 epochs over 2M samples (~20k steps). The signed variant uses BatchTopK (average k
per batch) while the paper uses per-sample TopK; the baseline uses per-sample TopK.

Three variants, selected via make_trainer(..., variant=...):
  "signed"    — SignedBatchTopKSAE     (paper's architecture: TopK by |value|, sign kept)
  "penalized" — CoActPenalizedSAE     (signed + co-activation coherence penalty)
  "baseline"  — TopKTrainingSAE       (per-sample ReLU-TopK; direct paper replication)

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
from .penalized_sae import CoActPenalizedSAE, CoActPenalizedSAEConfig
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


class SanityError(AssertionError):
    """Raised when a pre- or mid-training sanity check fails."""


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """Train any BatchTopKTrainingSAE variant on a ManifoldZoo.

    Use make_trainer() rather than constructing directly.
    After training, call eval() and compute_r2() for metrics.
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
            self.train_data = zoo.generate(cfg.n_train, train_rng).x
            print("done.")

        if eval_data is not None:
            self.eval_data: EvalBatch = eval_data
        else:
            eval_rng = np.random.default_rng(cfg.eval_seed)
            self.eval_data = zoo.generate(cfg.n_eval, eval_rng, return_ground_truth=True)

        self._check_zoo()
        self._check_sae_init()
        self._estimate_norm_scale()

    # ── Sanity checks ─────────────────────────────────────────────────────────

    def _check_zoo(self) -> None:
        zoo, sae = self.zoo, self.sae
        if zoo.d != sae.cfg.d_in:
            raise SanityError(f"Zoo d={zoo.d} ≠ SAE d_in={sae.cfg.d_in}.")
        rng = np.random.default_rng(99999)
        for inst in zoo.instances:
            if inst.sigma <= 0 or not math.isfinite(inst.sigma):
                raise SanityError(f"{inst.name}: degenerate sigma={inst.sigma:.2e}.")
            err = float(np.abs(inst.V @ inst.V.T - np.eye(inst.k_i, dtype=np.float32)).max())
            if err > 1e-4:
                raise SanityError(f"{inst.name}: V not orthonormal (max err {err:.2e}).")
            rms = float(np.sqrt(np.mean(np.sum(inst.sample_normalized(2_000, rng) ** 2, axis=1))))
            if abs(rms - 1.0) > 0.08:
                raise SanityError(f"{inst.name}: normalized RMS={rms:.3f}, expected ≈1.")
        print(f"[✓] Zoo: {zoo.n_instances} instances, d={zoo.d}, L0={zoo.L0}, σ_ε={zoo.noise_std}")

    def _check_sae_init(self) -> None:
        sae = self.sae
        with torch.no_grad():
            row_norms = sae.W_dec.norm(dim=-1)
            b_dec_max = sae.b_dec.abs().max().item()
        expected = sae.cfg.decoder_init_norm
        if expected is not None:
            mean_n = row_norms.mean().item()
            if abs(mean_n - expected) > max(0.02 * abs(expected), 0.005):
                raise SanityError(f"W_dec mean norm={mean_n:.4f}, expected {expected}.")
        if b_dec_max > 1e-6:
            raise SanityError(f"b_dec non-zero at init: max|b_dec|={b_dec_max:.2e}.")
        print(f"[✓] SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, k={sae.cfg.k}, "
              f"W_dec norm={row_norms.mean():.4f}, type={type(sae).__name__}")

    def _estimate_norm_scale(self) -> None:
        """scale = √E[‖x‖²] so that E[‖x/scale‖²] = 1 (unit mean-sq norm)."""
        x = torch.from_numpy(self.train_x[:8192]).float()
        self.norm_scale = x.pow(2).sum(-1).mean().sqrt().item()
        if self.norm_scale < 1e-6:
            raise SanityError(f"norm_scale={self.norm_scale:.2e}; data is zero?")
        expected = math.sqrt(self.zoo.L0)
        ratio = self.norm_scale / expected
        ok = "✓" if 0.5 <= ratio <= 2.0 else "warn"
        print(f"[{ok}] norm_scale={self.norm_scale:.4f} (expected ≈√L0={expected:.4f})")

    def _check_first_step(self, output: Any, x: torch.Tensor) -> None:
        if not torch.isfinite(output.loss):
            raise SanityError(f"Loss={output.loss.item()} at step 0.")
        with torch.no_grad():
            fa = output.feature_acts.detach()
            mean_l0 = (fa != 0).float().sum(-1).mean().item()
            k = self.sae.cfg.k
            if abs(mean_l0 - k) > max(k * 0.3, 2.0):
                print(f"[warn step 0] mean L0={mean_l0:.1f}, expected ≈{k}.")
            # Signed SAEs must produce negative activations at init (~50%)
            if isinstance(self.sae, SignedBatchTopKSAE):
                n_act = (fa != 0).sum().item()
                n_neg = (fa < 0).sum().item()
                if n_act > 0 and n_neg == 0:
                    raise SanityError(
                        "No negative activations at step 0 — ReLU may still be active. "
                        "Check that SignedBatchTopK.get_activation_fn() is being called."
                    )
                if n_act > 0 and n_neg / n_act < 0.1:
                    print(f"[warn step 0] only {100*n_neg/n_act:.0f}% negative activations (expected ≈50%).")
            mse = output.losses.get("mse_loss", output.loss).item()
            fvu = mse / (x.pow(2).sum(-1).mean().item() + 1e-8)
            if fvu > 2.0:
                print(f"[warn step 0] FVU={fvu:.2f}; expected ≈1 at random init.")

    # ── Data and encoding ──────────────────────────────────────────────────────

    def _normalize_input(self, x_np: np.ndarray) -> torch.Tensor:
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
        x = self._normalize_input(x_np)

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

        # Update per-batch dead mask: atoms that didn't fire on any sample this batch
        with torch.no_grad():
            fired = (output.feature_acts.detach() != 0).any(dim=0).cpu()
            self.dead_mask = ~fired

        if global_step == 0:
            self._check_first_step(output, x)

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
        x_eval = self._normalize_input(self.eval_data.x)
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
        codes = self._encode_eval(self._normalize_input(self.eval_data.x)).cpu().numpy()
        active_mask = self.eval_data.active_mask          # (N, m)
        # contributions[j] is already (n_j, d) for active samples only (sparse format)
        contribs = [c / self.norm_scale for c in self.eval_data.contributions]
        return {
            "k":               float(self.sae.cfg.k),
            "eval_codes":      codes,
            "W_dec":           self.sae.W_dec.detach().cpu().numpy(),
            "active_mask":     active_mask,
            "instance_contribs": contribs,
            # gamma = contrib @ V.T: recovers normalized intrinsic coords since
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

    # ── Restricted R² ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_r2(self) -> dict[str, float]:
        """Restricted R² per manifold instance (Eq. 14, arXiv 2604.28119).

        Matches paper's compute_r2 exactly:
          - Geometric OMP on decoder directions with SVD deflation (not mean |z| ranking)
          - Actual SAE codes for reconstruction
          - M_hat centred after reconstruction to remove encoder bias and cross-manifold contamination
          - R² = 1 - ||M_c - M_hat_c||² / ||M_c||²  where M_c = M - M.mean(0)
        GPU path: scoring and all matmuls on self.device; SVD stays on CPU (tiny matrix).
        """
        self.sae.eval()
        x_eval = self._normalize_input(self.eval_data.x)
        z_all = self._encode_eval(x_eval)                  # (N, d_sae) on self.device
        W_dec_np = self.sae.W_dec.detach().cpu().numpy()   # (d_sae, d) CPU — for SVD only
        W_t = self.sae.W_dec.detach()                      # (d_sae, d) on self.device
        active = self.eval_data.active_mask                # (N, m) numpy bool

        r2_dict: dict[str, float] = {}
        for j, inst in enumerate(self.zoo.instances):
            act_j = active[:, j]
            if act_j.sum() < 10:
                r2_dict[inst.name] = float("nan")
                continue

            act_j_t = torch.from_numpy(act_j).to(self.device)
            z_j = z_all[act_j_t]                           # (n_j, d_sae) on self.device
            m_j = torch.from_numpy(
                self.eval_data.contributions[j].astype(np.float32) / self.norm_scale
            ).to(self.device)                              # (n_j, d)

            M_c_t = m_j - m_j.mean(0)
            total_var = float(M_c_t.pow(2).sum().item())
            if total_var < 1e-10:
                r2_dict[inst.name] = 1.0
                continue

            selected: list[int] = []
            residual_t = M_c_t.clone()
            r2 = float("nan")

            for step in range(inst.k_i + 3):
                var_exp = (residual_t @ W_t.T).pow(2).sum(0).cpu().numpy()  # (d_sae,)
                var_exp[selected] = -1.0
                best = int(var_exp.argmax())
                if var_exp[best] <= 0:
                    break
                selected.append(best)

                _, s, Vt = np.linalg.svd(W_dec_np[selected], full_matrices=False)
                basis_t = torch.from_numpy(Vt[s > 1e-8]).to(self.device)   # (rank, d)
                residual_t = M_c_t - (M_c_t @ basis_t.T) @ basis_t

                M_hat_t = z_j[:, selected] @ W_t[selected]                 # (n_j, d)
                M_hat_c_t = M_hat_t - M_hat_t.mean(0)

                if step == inst.k_i - 1:
                    ss_res = float(((M_c_t - M_hat_c_t) ** 2).sum().item())
                    r2 = 1.0 - ss_res / (total_var + 1e-12)

            r2_dict[inst.name] = r2

        valid = [v for v in r2_dict.values() if math.isfinite(v)]
        return {"r2/mean": float(np.mean(valid)) if valid else float("nan"),
                **{f"r2/{n}": v for n, v in r2_dict.items()}}

    # ── Main loop ─────────────────────────────────────────────────────────────

    def train(self) -> list[dict[str, Any]]:
        """Run n_epochs over the pre-generated training dataset; return per-step metric dicts."""
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
                    parts = [f"ep {epoch+1:>2}/{cfg.n_epochs}  step {global_step:>6}",
                             f"loss={log['loss']:.4f}",
                             f"L0={log['mean_l0']:.1f}",
                             f"fvu={log['fvu']:.3f}",
                             f"dead={log['n_dead']}"]
                    if "l1_loss" in log:
                        parts.insert(2, f"l1={log['l1_loss']:.4f}")
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
        r2 = self.compute_r2()
        neg = f"  neg={100*em['eval/neg_frac']:.0f}%" if isinstance(sae, SignedBatchTopKSAE) else ""
        print(f"Final: fvu={em['eval/fvu']:.4f}  L0={em['eval/mean_l0']:.1f}  "
              f"dead={em['eval/n_dead']}{neg}  R²={r2['r2/mean']:.3f}")
        self.logs.append({"step": n_total_steps, **em, **r2})
        return self.logs


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
      "signed"    — SignedBatchTopKSAE (paper's architecture)
      "penalized" — CoActPenalizedSAE  (signed + co-activation coherence penalty)
      "baseline_batch"  — BatchTopKTrainingSAE (standard ReLU-TopK; rough lower bound)
      "baseline"

    k: sparsity budget; use K_SWEEP values for paper experiments.
       Defaults to L0 × mean(k_i) — one atom per manifold dimension.

    d_sae is always 4 × d_in (expansion_factor=4, Appendix E).

    Examples:
        make_trainer(PAPER_CONFIG, k=4)
        make_trainer(PAPER_CONFIG, k=4, variant="penalized")
        make_trainer(PAPER_CONFIG, k=4, variant="baseline")
    """
    if variant not in ("signed", "penalized", "baseline", "baseline_batch"):
        raise ValueError(f"variant must be 'signed', 'penalized', or 'baseline'; got {variant!r}")

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
    elif variant == "penalized":
        sae = CoActPenalizedSAE(CoActPenalizedSAEConfig(**base_kwargs, **sae_kwargs))
    else:
        sae = SignedBatchTopKSAE(SignedBatchTopKSAEConfig(**base_kwargs))

    return Trainer(sae, zoo, train_cfg, train_data=train_data, eval_data=eval_data)
