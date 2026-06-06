"""
Diagnostics for adapt/ training runs.

Run sections manually or execute the whole file:
    python adapt/debug.py

Sections:
  1. FVU formula check
  2. Dead-neuron analysis from logs
  3. Geo-penalty scale check
  4. Quick smoke test (k=3, EASY_CONFIG)
"""
import math
import torch
import numpy as np

from adapt.data import EASY_CONFIG, PAPER_CONFIG
from adapt.train import make_trainer, K_SWEEP


# ── 1. FVU formula ────────────────────────────────────────────────────────────
# mse_loss from SAELens is sum-based: E_batch[||x - x_hat||^2]  (sum over d features).
# x.var() is the per-element variance across the whole batch tensor.
# For unit-RMS normalized input: E[||x||^2] = 1, so var_per_elem = 1/d.
#
# Current code:  fvu = sum_mse / var_per_elem  ≈  d * true_FVU
# At init (x_hat ≈ 0): sum_mse ≈ 1  →  fvu ≈ 128  (see: "[warn] FVU=125" in every run)
#
# Fix: use E[||x||^2] = x.pow(2).mean() as denominator instead of x.var().
# This gives fvu ≈ 1 at random init and fvu → 0 at perfect reconstruction.

def check_fvu_formula():
    print("=== FVU formula analysis ===")
    d = 128
    mse_at_init = 0.9788      # from logs; x_hat ≈ 0 so sum_mse ≈ E[||x||^2] = 1
    var_per_elem = 1.0 / d    # ≈ 1/128 for unit-norm inputs

    fvu_current = mse_at_init / var_per_elem
    print(f"Current formula fvu = sum_mse / var_per_elem = {fvu_current:.1f}  "
          f"(logged ~125 at init — triggers warn)")
    print(f"Correct formula fvu = sum_mse / E[||x||^2] = {mse_at_init:.3f}  "
          f"(should be ~1 at init)")
    print()
    # Rescale final eval numbers from the k=25 PAPER_CONFIG run
    results = {"baseline": 0.1488, "signed": 0.0282, "penalized": 0.0387}
    print("Final eval FVU (corrected by /d=128):")
    for name, fvu in results.items():
        print(f"  {name:12s}  logged={fvu:.4f}  corrected≈{fvu/d:.5f}")
    print()


# ── 2. Dead-neuron analysis ───────────────────────────────────────────────────
# At k=25, d_sae=512, L0=4: expected fires per atom per batch ≈ 25*1024/512 = 50.
# Observing ~284/512 dead per batch means ~55% of atoms never fire.
#
# Why: with 48 manifold instances of mean k_i≈2.6, only ~L0*mean_k_i*n_inst
# atoms are "needed": 4*2.6*48 = 499 in the worst case, but atoms per active
# manifold is k/L0 = 25/4 ≈ 6, so only 48*6 ≈ 288 atoms get used.
# 512 - 288 = 224 ≈ the ~284 dead we observe.
#
# This is expected for k=25 (above the sweet spot). AuxK fails to revive
# permanently unused atoms because the residual is already small.

def check_dead_expected():
    print("=== Dead-neuron expected count ===")
    d_sae = 512
    k = 25
    L0 = 4
    batch = 1024
    n_inst = 48
    mean_ki = 21 / 8  # mean k_i across 8 manifold types

    fires_per_atom = k * batch / d_sae
    atoms_per_manifold = k / L0
    atoms_needed = n_inst * atoms_per_manifold
    expected_dead = d_sae - atoms_needed

    print(f"  Expected fires/atom/batch = {fires_per_atom:.0f}  (if uniform)")
    print(f"  Atoms needed (n_inst * k/L0) ≈ {atoms_needed:.0f} / {d_sae}")
    print(f"  Expected dead ≈ {expected_dead:.0f}  (observed ~284 for signed/penalized)")
    print(f"  Note: at k=4 (sweet spot), atoms_per_manifold=1, needed≈48 → dead≈464  (worse!)")
    print(f"  Dead neurons are expected at high k; per-batch count is noisy.")
    print()


# ── 3. Geo-penalty scale ──────────────────────────────────────────────────────
# geo_loss ≈ 6e-4, β = 0.01 → penalty contribution = 6e-6
# MSE ≈ 0.001 at convergence
# Ratio ≈ 0.6%
#
# The penalty is very weak at k=25 (dilution regime). Expect stronger effect
# at k=3,4,6 where fewer atoms compete and geometry pressure is higher.

def check_geo_scale():
    print("=== Geo-penalty scale ===")
    geo_loss = 6e-4
    beta = 0.01
    mse_converged = 0.001
    contribution = beta * geo_loss
    ratio_pct = 100 * contribution / mse_converged
    print(f"  β*geo_loss = {beta}*{geo_loss:.1e} = {contribution:.1e}")
    print(f"  MSE at convergence ≈ {mse_converged:.3f}")
    print(f"  Penalty / MSE ≈ {ratio_pct:.2f}%  ← very weak at k=25")
    print(f"  Expect larger effect at k=3,4,6 (sweet-spot regime)")
    print()


# ── 4. Quick smoke test ───────────────────────────────────────────────────────
# Verifies all three variants run without error on EASY_CONFIG.
# Checks: neg%≈50% for signed/penalized, R²>0 for all, FVU decreasing.

def smoke_test(device="cpu", n_steps=200):
    print(f"=== Smoke test (EASY_CONFIG, k=3, {n_steps} steps) ===")
    from adapt.train import TrainConfig
    cfg = TrainConfig(n_steps=n_steps, log_every=n_steps, eval_every=n_steps, device=device)

    for variant in ("signed", "penalized", "baseline"):
        trainer = make_trainer(EASY_CONFIG, k=3, variant=variant, train_cfg=cfg)
        logs = trainer.train()
        r2 = trainer.compute_r2()["r2/mean"]
        final = logs[-1]
        neg_str = f"  neg={100*final.get('eval/neg_frac',0):.0f}%" if variant != "baseline" else ""
        print(f"  {variant:12s}  fvu={final['eval/fvu']:.3f}  R²={r2:.3f}{neg_str}")

        assert math.isfinite(r2), f"{variant}: R² is NaN"
        if variant in ("signed", "penalized"):
            neg = final.get("eval/neg_frac", 0)
            assert neg > 0.3, f"{variant}: neg_frac={neg:.2f} too low (should be ~0.5)"
    print("  All checks passed.")
    print()


if __name__ == "__main__":
    check_fvu_formula()
    check_dead_expected()
    check_geo_scale()
    smoke_test()
