"""
Computations for Figure 4 replication (arXiv 2604.28119, Figure 4).

Three metric groups stored per k sweep:
  Restricted R²  (r2 in output)       — aggregate R² and per-manifold sweep over n_atoms
  Coverage       (coverage in output)  — mean support size and RF diameter
  Phi-coherence  (phi in output)       — pairwise phi-coefficient matrix + atom-manifold scores
  Visualisation  (vis_data in output)  — per-instance codes/coords/contribs for plotting

TODO: Replace phi coefficient with the Ising inverse coupling matrix.

Usage:

    from geo_sae.results.results import collate_snapshots
    import pickle

    per_k = {}
    for k in K_SWEEP:
        trainer = make_trainer(...)
        per_k[k] = trainer.train(return_snapshot=True, include_vis_data=(k == 10))

    results = collate_snapshots(per_k)
    pickle.dump(results, open("baseline_snapshots.b", "wb"))
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _TORCH = True
except ImportError:
    _TORCH = False

_RF_RNG = np.random.default_rng(42)


def _mean_pairwise_dist(X: np.ndarray, max_pts: int = 2000, n_pairs: int = 5000) -> float:
    """Estimate mean pairwise Euclidean distance in ambient space by sampling pairs.

    Subsample to max_pts before pair sampling so cost is O(n_pairs * d) regardless
    of how many points fired.
    """
    n = X.shape[0]
    if n < 2:
        return 0.0
    if n > max_pts:
        idx = _RF_RNG.choice(n, max_pts, replace=False)
        X = X[idx]
        n = max_pts
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    if n_pairs == 0:
        return 0.0
    a = _RF_RNG.integers(0, n, n_pairs)
    b = _RF_RNG.integers(0, n, n_pairs)
    same = a == b
    b[same] = (b[same] + 1) % n
    return float(np.sqrt(((X[a] - X[b]) ** 2).sum(-1)).mean())


# ── Geometric greedy OMP + SVD deflation (paper's R² algorithm, Appendix E) ──────
#
# Selection:    score = ||residual @ d̂_j||² per atom, where d̂_j = d_j / ||d_j||
#               (paper uses unit-norm decoder columns; we normalise explicitly)
# Deflation:    exact SVD projection onto span(W_dec[selected]) after each step;
#               this is OMP-style orthogonalizing deflation (Appendix D frames
#               recovery via OMP, which resolves step 3's ambiguity toward SVD)
# Reconstruction: actual SAE codes — M_hat^(n) = Z_i^(n) @ W_dec^T  (steps 4+5)
# R²:           1 - Σ‖m − m̂‖² / Σ‖m − m̄‖²  (paper Eq. 14)
#               Greedy selection runs on centered M_c; residual uses raw M_i.
#               Denominator Σ‖m − m̄‖² = ‖M_c‖²_F.

def _geometric_greedy_gpu_t(
    m_c_t: "torch.Tensor",
    W_t: "torch.Tensor",
    W_dec_np: np.ndarray,
    n: int,
) -> list[int]:
    """Geometric OMP with SVD deflation. GPU for scoring, CPU for SVD (tiny matrix).

    Basis (rank × d) is the only CPU→GPU transfer per step — negligible cost.
    """
    d_sae = W_t.shape[0]
    residual_t = m_c_t.clone()
    alive_t = torch.ones(d_sae, dtype=torch.bool, device=m_c_t.device)
    selected: list[int] = []
    d_norms_sq_t = W_t.pow(2).sum(1).clamp(min=1e-10)         # (d_sae,) — precompute once

    for _ in range(min(n, d_sae)):
        var_exp = (residual_t @ W_t.T).pow(2).sum(0) / d_norms_sq_t  # normalise by ||d_j||²
        var_exp.masked_fill_(~alive_t, -1.0)
        best = int(var_exp.argmax().item())
        if float(var_exp[best].item()) <= 0:
            break
        selected.append(best)
        alive_t[best] = False

        # SVD of selected decoder rows (n_sel × d) — always tiny, fast on CPU
        _, s, Vt = np.linalg.svd(W_dec_np[selected], full_matrices=False)
        basis_t = torch.from_numpy(Vt[s > 1e-8]).to(m_c_t.device)  # (rank, d)
        P_t = (m_c_t @ basis_t.T) @ basis_t                    # (n_j, d)
        residual_t = m_c_t - P_t

    return selected


def _geometric_greedy_np(m_c: np.ndarray, W_dec: np.ndarray, n: int) -> list[int]:
    """Numpy fallback — geometric OMP with SVD deflation."""
    residual = m_c.copy()
    alive = np.ones(W_dec.shape[0], dtype=bool)
    selected: list[int] = []
    d_norms_sq = (W_dec ** 2).sum(1).clip(1e-10)               # (d_sae,) — precompute once

    for _ in range(min(n, W_dec.shape[0])):
        scores = ((residual @ W_dec.T) ** 2).sum(0) / d_norms_sq   # normalise by ||d_j||²
        scores[~alive] = -1.0
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            break
        selected.append(best)
        alive[best] = False
        _, s, Vt = np.linalg.svd(W_dec[selected], full_matrices=False)
        basis = Vt[s > 1e-8]
        residual = m_c - (m_c @ basis.T) @ basis

    return selected


def _code_greedy_np(
    Z_i: np.ndarray,
    m_c: np.ndarray,
    W_dec: np.ndarray,
    n: int,
) -> list[int]:
    """Code-based greedy OMP: atoms selected by code-weighted alignment with residual.

    Geometric OMP (paper, Appendix E step 3) scores each atom by decoder direction
    variance alone, ignoring actual activation magnitudes:
        geo_score_a = Σ_j (r_j · d̂_a)²

    This alternative weights by actual codes on the manifold instance:
        code_score_a = (Σ_j z_{j,a} · r_j · W_dec[a])² / (Σ_j z_{j,a}² · ‖W_dec[a]‖²)

    By Cauchy–Schwarz, code_score ≤ geo_score always. Atoms that rarely fire on this
    manifold instance are penalised even if their decoder direction is well-aligned.
    SVD deflation on selected decoder rows is identical to geometric OMP.
    """
    residual = m_c.copy()
    alive = np.ones(W_dec.shape[0], dtype=bool)
    selected: list[int] = []
    d_norms_sq  = (W_dec ** 2).sum(1).clip(1e-10)   # (d_sae,) ‖W_dec[a]‖²
    code_energy = (Z_i  ** 2).sum(0) + 1e-12         # (d_sae,) Σ_j z_{j,a}²

    for _ in range(min(n, W_dec.shape[0])):
        geo_proj  = residual @ W_dec.T                # (n_j, d_sae): r_j · W_dec[a]
        code_corr = (Z_i * geo_proj).sum(0)           # (d_sae,): Σ_j z_{j,a} * r_j · W_dec[a]
        scores = code_corr ** 2 / (code_energy * d_norms_sq)
        scores[~alive] = -1.0
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            break
        selected.append(best)
        alive[best] = False
        _, s, Vt = np.linalg.svd(W_dec[selected], full_matrices=False)
        basis = Vt[s > 1e-8]
        residual = m_c - (m_c @ basis.T) @ basis

    return selected


def greedy_select_atoms(m_j: np.ndarray, W_dec: np.ndarray, n: int) -> list[int]:
    """Geometric OMP to select the n atoms that best explain manifold instance j.

    Returns atom indices in selection order (most → least important).
    """
    return _geometric_greedy_np(m_j - m_j.mean(0), W_dec, n)


# ── Restricted R² helpers ────────────────────────────────────────────────────

def _r2_sweep_for_instance(
    Z_i: np.ndarray,
    M_i: np.ndarray,
    W_dec: np.ndarray,
    n_atoms_range: list[int],
    centre_mhat: bool = False,
) -> list[float]:
    """R² sweep — numpy fallback. Implements paper Appendix E Eq. 14.

    Greedy selection (step 3) operates on centered M_c to isolate variance.
    Residual (step 6 numerator) uses raw M_i per Eq. 14: Σ‖m − m̂‖².
    Denominator: Σ‖m − m̄‖² = ‖M_c‖²_F.
    centre_mhat is not in the paper; subtracts M̂ mean to diagnose mean-shift.
    """
    if len(Z_i) < 10:
        return [float("nan")] * len(n_atoms_range)
    M_c = M_i - M_i.mean(0)
    ss_tot = float((M_c ** 2).sum())                        # Σ‖m − m̄‖²
    if ss_tot < 1e-12:
        return [1.0] * len(n_atoms_range)
    selected = _geometric_greedy_np(M_c, W_dec, max(n_atoms_range))   # step 3
    n_sel = len(selected)
    results = []
    for n in n_atoms_range:
        n_use = min(n, n_sel)
        M_hat = Z_i[:, selected[:n_use]] @ W_dec[selected[:n_use]]     # steps 4+5
        if centre_mhat:
            M_hat = M_hat - M_hat.mean(0)
        ss_res = float(((M_i - M_hat) ** 2).sum())                     # Eq. 14 numerator
        results.append(float(1.0 - ss_res / (ss_tot + 1e-12)))
    return results


def _r2_sweep_code_greedy(
    Z_i: np.ndarray,
    M_i: np.ndarray,
    W_dec: np.ndarray,
    n_atoms_range: list[int],
) -> list[float]:
    """R² sweep using code-based greedy OMP (numpy only). Pair with _r2_sweep_for_instance."""
    if len(Z_i) < 10:
        return [float("nan")] * len(n_atoms_range)
    M_c = M_i - M_i.mean(0)
    ss_tot = float((M_c ** 2).sum())
    if ss_tot < 1e-12:
        return [1.0] * len(n_atoms_range)
    selected = _code_greedy_np(Z_i, M_c, W_dec, max(n_atoms_range))
    n_sel = len(selected)
    results = []
    for n in n_atoms_range:
        n_use = min(n, n_sel)
        M_hat = Z_i[:, selected[:n_use]] @ W_dec[selected[:n_use]] if n_use > 0 else np.zeros_like(M_i)
        ss_res = float(((M_i - M_hat) ** 2).sum())
        results.append(float(1.0 - ss_res / (ss_tot + 1e-12)))
    return results


# ── Coverage helpers ─────────────────────────────────────────────────────────

def _support_stats(
    z_j: np.ndarray,
    min_fires: int,
    percentile: float,
    contribs_j: np.ndarray | None = None,
    manifold_diam: float | None = None,
) -> tuple[int, list[float]]:
    """(n_support, rf_vals) for one manifold instance.

    Paper (Appendix E): atom is in support iff
      - fires on ≥ 10% of the n_j manifold-active eval points  (relative threshold)
      - fires on ≥ min_fires points  (absolute floor for tiny manifolds)
    "Firing" = |z| above the atom's 10th-percentile nonzero activation magnitude.

    rf_vals is mean pairwise Euclidean distance in ambient space normalised by
    manifold diameter (paper's definition) when contribs_j and manifold_diam are
    provided, or firing fraction as a fallback for old snapshots.

    GPU path vectorises the per-atom percentile loop via sort+gather over the
    full (n_j, d_sae) matrix — eliminates 512 serial np.percentile calls.
    """
    n_j = z_j.shape[0]
    min_fires_eff = max(min_fires, int(0.10 * n_j))   # paper's 10% relative condition
    use_spatial = contribs_j is not None and manifold_diam is not None and manifold_diam > 1e-8

    if _TORCH:
        with torch.no_grad():
            z_t = torch.from_numpy(z_j).to(_DEVICE)          # (n_j, d_sae)
            fire = (z_t != 0)
            n_fires = fire.sum(0)                             # (d_sae,)
            candidates = n_fires >= min_fires
            if not candidates.any():
                return 0, []

            # Threshold on |z| so signed SAEs (with negative activations) don't get
            # a negative threshold that inflates support counts.
            z_abs = z_t.abs()
            z_abs_sort = z_abs.clone()
            z_abs_sort[~fire] = float("inf")
            sorted_abs, _ = z_abs_sort.sort(dim=0)
            pct_idx = (n_fires.float() * (percentile / 100.0)).long().clamp(0, z_t.shape[0] - 1)
            thresholds = sorted_abs.gather(0, pct_idx.unsqueeze(0)).squeeze(0)  # (d_sae,)

            above = (z_abs >= thresholds.unsqueeze(0)).sum(0)
            in_support_t = candidates & (above >= min_fires_eff)
            n_support = int(in_support_t.sum().item())
            if n_support == 0:
                return 0, []

            if not use_spatial:
                coverages = fire[:, in_support_t].float().mean(0).tolist()
                return n_support, coverages

            in_support_np = in_support_t.cpu().numpy()
            thresholds_np = thresholds.cpu().numpy()

    else:
        fire_np = (z_j != 0)
        n_fires_np = fire_np.sum(0)
        d_sae = z_j.shape[1]
        in_support_np = np.zeros(d_sae, dtype=bool)
        thresholds_np = np.zeros(d_sae, dtype=np.float32)
        for a in np.where(n_fires_np >= min_fires)[0]:
            nz = np.abs(z_j[fire_np[:, a], a])
            t = float(np.percentile(nz, percentile))
            thresholds_np[a] = t
            if (np.abs(z_j[:, a]) >= t).sum() >= min_fires_eff:
                in_support_np[a] = True
        n_support = int(in_support_np.sum())
        if n_support == 0:
            return 0, []
        if not use_spatial:
            coverages = fire_np[:, in_support_np].mean(0).tolist()
            return n_support, coverages

    # Correct paper metric: mean pairwise Euclidean distance in ambient space,
    # normalised by manifold's own mean pairwise distance.
    z_abs_np = np.abs(z_j)
    rf_vals: list[float] = []
    for a in np.where(in_support_np)[0]:
        fired_mask = z_abs_np[:, a] >= thresholds_np[a]
        pts = contribs_j[fired_mask]  # type: ignore[index]
        if len(pts) < 2:
            rf_vals.append(0.0)
            continue
        rf_vals.append(min(_mean_pairwise_dist(pts) / manifold_diam, 1.0))  # type: ignore[operator]
    return n_support, rf_vals


# ── Phi-coherence helpers ────────────────────────────────────────────────────

def _rho_G_from_global_phi(
    phi: np.ndarray,
    scores: np.ndarray,
    inst_names: list,
    inst_ki: list,
) -> dict:
    """ρ(G) proxy from the global marginal phi matrix (Definition 6, proxy for Ising J).

    Uses all N evaluation samples (no manifold-active conditioning). Conditioning
    on manifold-active samples breaks for signed SAEs: those atoms always fire when
    the manifold is active (|z|>0), so p≈1 → Var→0 → phi undefined.

    Binarisation in phi: s = 1[z≠0] over all N samples. For signed SAEs this
    correctly marks both positive and negative activations as "active".
    """
    rho_G: dict = {}
    for j, (name, k_i) in enumerate(zip(inst_names, inst_ki)):
        if k_i < 2:
            continue
        top_atoms = np.argsort(scores[:, j])[-k_i:]
        sub = phi[np.ix_(top_atoms, top_atoms)]
        mask = np.triu(np.ones((k_i, k_i), dtype=bool), k=1)
        vals = sub[mask]
        if len(vals) > 0:
            rho_G[name] = float(np.mean(np.sign(vals)))
    return rho_G


def _phi_stats(
    eval_codes: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Global marginal phi matrix + atom-manifold scores.

    Used for the block-diagonal visualisation (phi_matrix plot) and ρ(G) computation.
    Note: marginal phi conflates true co-activation with joint silence; this inflates
    off-diagonal values, so ρ(G) is an optimistic estimate.
    """
    if _TORCH:
        with torch.no_grad():
            codes_t = torch.from_numpy(eval_codes).to(_DEVICE)        # (N, d_sae)
            N = codes_t.shape[0]

            S = (codes_t != 0).float()                                # (N, d_sae)
            p = S.mean(0)
            joint = (S.T @ S) / N
            var = p * (1.0 - p)
            denom = torch.outer(var, var).sqrt_().clamp_(min=1e-8)
            phi = (joint - torch.outer(p, p)) / denom
            phi.fill_diagonal_(1.0)
            phi_np = phi.cpu().numpy().astype(np.float32)

            mask_t = torch.from_numpy(active_mask).to(_DEVICE, dtype=torch.float32)
            counts = mask_t.sum(0).clamp(min=1.0)
            scores = (codes_t.abs().T @ mask_t) / counts.unsqueeze(0)
            scores_np = scores.cpu().numpy().astype(np.float32)

            return phi_np, scores_np

    S = (eval_codes != 0).astype(np.float32)
    N = S.shape[0]
    p = S.mean(0)
    joint = (S.T @ S) / N
    denom = np.sqrt(np.outer(p * (1.0 - p), p * (1.0 - p))).clip(1e-8)
    phi_np = ((joint - np.outer(p, p)) / denom).astype(np.float32)
    np.fill_diagonal(phi_np, 1.0)

    d_sae, m = eval_codes.shape[1], active_mask.shape[1]
    scores_np = np.zeros((d_sae, m), dtype=np.float32)
    for j in range(m):
        mask_j = active_mask[:, j]
        if mask_j.sum() > 0:
            scores_np[:, j] = np.abs(eval_codes[mask_j]).mean(0)

    return phi_np, scores_np


# ── Per-metric compute functions (private) ────────────────────────────────────

def _compute_r2(
    k: int,
    snap: dict,
    max_atoms: int,
    centre_mhat: bool = True,
) -> dict:
    """Compute restricted R² for one snapshot using the paper's geometric greedy OMP (Appendix E).

    centre_mhat=True  — centres M̂ before computing residual (inflates R²; ~0.68).
    centre_mhat=False — paper Eq. 14 exactly (R²~0.50).
    """

    Z = snap["eval_codes"]         # (N, c) full-eval SAE codes
    W_dec = snap["W_dec"]          # (c, d) decoder matrix
    active_mask = snap["active_mask"]
    inst_names = snap["instance_names"]
    inst_ki    = snap["instance_ki"]
    inst_types = snap["instance_types"]

    n_atoms_range = list(range(1, max_atoms + 1))
    W_t = torch.from_numpy(W_dec).to(_DEVICE) if _TORCH else None

    r2_sweep: dict[str, list[float]] = {}
    r2_at_ki: list[float] = []

    for j, (name, k_i) in enumerate(zip(inst_names, inst_ki)):
        active_j = active_mask[:, j]
        n_i = int(active_j.sum())
        if n_i < 10:                                         # practical sample-floor, not from paper
            r2_sweep[name] = [float("nan")] * len(n_atoms_range)
            continue

        Z_i = Z[active_j]                                   # (n_i, c)  — paper step 2
        M_i = snap["instance_contribs"][j]                  # (n_i, d)  — paper step 2
        M_c = M_i - M_i.mean(0)                             # centered contributions (for greedy)

        ss_tot = float((M_c ** 2).sum())                    # Σ‖m − m̄‖²  (Eq. 14 denominator)
        if ss_tot < 1e-12:
            r2_sweep[name] = [1.0] * len(n_atoms_range)
            if k_i <= max_atoms:
                r2_at_ki.append(1.0)
            continue

        if _TORCH:
            with torch.no_grad():
                M_c_t = torch.from_numpy(M_c).to(_DEVICE)
                M_i_t = torch.from_numpy(M_i).to(_DEVICE)
                Z_i_t = torch.from_numpy(Z_i).to(_DEVICE)

                # Step 3: greedy-select atoms by decoder directions explaining M_c
                selected = _geometric_greedy_gpu_t(M_c_t, W_t, W_dec, max(n_atoms_range))
                if not selected:
                    r2_sweep[name] = [float("nan")] * len(n_atoms_range)
                    continue

                n_sel = len(selected)
                sel_t = torch.tensor(selected, device=_DEVICE, dtype=torch.long)

                # Steps 4+5: M_hat^(n) = Z_i^(n) @ W_dec^T  (cumulative by selection order)
                atom_contribs = Z_i_t[:, sel_t].unsqueeze(-1) * W_t[sel_t].unsqueeze(0)
                M_hat = atom_contribs.cumsum(1)              # (n_i, n_sel, d)
                if centre_mhat:
                    M_hat = M_hat - M_hat.mean(0, keepdim=True)

                # Step 6: Eq. 14 numerator — Σ‖m^(j) − m̂^(j,n)‖² for each n
                ss_res = ((M_i_t.unsqueeze(1) - M_hat) ** 2).sum(dim=(0, 2))

                sweep = []
                for n in n_atoms_range:
                    idx = min(n, n_sel) - 1
                    sweep.append(float(1.0 - ss_res[idx].item() / (ss_tot + 1e-12)))
        else:
            sweep = _r2_sweep_for_instance(Z_i, M_i, W_dec, n_atoms_range, centre_mhat)

        r2_sweep[name] = sweep
        if k_i <= max_atoms and math.isfinite(sweep[k_i - 1]):
            r2_at_ki.append(sweep[k_i - 1])

    valid = [v for v in r2_at_ki if math.isfinite(v)]
    return {
        "k_values":         [k],
        "n_atoms_range":    n_atoms_range,
        "instance_types":   dict(zip(inst_names, inst_types)),
        "aggregate_r2":     {k: float(np.mean(valid)) if valid else float("nan")},
        "aggregate_r2_std": {k: float(np.std(valid))  if len(valid) > 1 else 0.0},
        "r2_sweep":         {name: {k: sweep} for name, sweep in r2_sweep.items()},
    }


def _compute_coverage(
    k: int,
    snap: dict,
    min_fires: int,
    percentile: float,
) -> dict:
    codes = snap["eval_codes"]
    active_mask = snap["active_mask"]
    instance_contribs: list | None = snap.get("instance_contribs")

    support_sizes: list[int] = []
    all_rf_vals: list[float] = []

    for j in range(active_mask.shape[1]):
        act_j = active_mask[:, j]
        n_j = int(act_j.sum())
        if n_j < min_fires:
            continue

        z_j = codes[act_j]

        contribs_j: np.ndarray | None = None
        manifold_diam: float | None = None
        if instance_contribs is not None:
            contribs_j = np.asarray(instance_contribs[j], dtype=np.float32)
            manifold_diam = _mean_pairwise_dist(contribs_j)

        n_sup, rf_vals = _support_stats(z_j, min_fires, percentile, contribs_j, manifold_diam)
        if n_sup > 0:
            support_sizes.append(n_sup)
            all_rf_vals.extend(rf_vals)

    return {
        "k_values":          [k],
        "mean_support_size": {k: float(np.mean(support_sizes)) if support_sizes else 0.0},
        "mean_rf_diameter":  {k: float(np.median(all_rf_vals)) if all_rf_vals else 0.0},
    }


def _compute_phi(k: int, snap: dict) -> dict:
    phi, scores = _phi_stats(snap["eval_codes"], snap["active_mask"])
    inst_names = snap["instance_names"]
    inst_ki    = snap["instance_ki"]
    return {
        "k_values":             [k],
        "instance_names":       inst_names,
        "instance_ki":          inst_ki,
        "phi_matrices":         {k: phi},
        "atom_manifold_scores": {k: scores},
        "rho_G":                {k: _rho_G_from_global_phi(phi, scores, inst_names, inst_ki)},
    }


def _compress_logs(raw_logs: dict, max_pts: int = 1000) -> dict:
    """Downsample per-k log entry lists to compact numpy arrays.

    Reduces ~38MB of raw Python dicts to ~150KB of float32 arrays with no
    meaningful loss of curve shape (training runs are 20k+ steps).
    """
    out = {}
    for k, entries in raw_logs.items():
        if not entries:
            continue
        stride = max(1, len(entries) // max_pts)
        sampled = entries[::stride]
        out[k] = {
            "steps":   np.array([e["step"] for e in sampled], dtype=np.int32),
            "l1_loss": np.array([e.get("l1_loss", e.get("loss", math.nan)) for e in sampled], dtype=np.float32),
            "fvu":     np.array([e.get("fvu",     math.nan) for e in sampled], dtype=np.float32),
            "n_dead":  np.array([e.get("n_dead",  math.nan) for e in sampled], dtype=np.float32),
        }
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def process_snapshot(
    k: int,
    snap: dict,
    max_atoms: int = 25,
    support_min_fires: int = 30,
    support_percentile: float = 10.0,
    r2_method: str = "centred",
) -> dict:
    """Process one raw snapshot into a compact per-k metrics dict.

    Computes R², coverage, phi, and compressed logs — then discards raw arrays
    (eval_codes, W_dec, active_mask, instance_contribs, instance_coords).
    The returned dict is ~10MB vs ~2GB for the raw snapshot.

    r2_method controls which R² formula to use:
      "centred"   (default) — centres M̂ before residual; ~0.68.
      "uncentred"           — paper Eq. 14 exactly; ~0.50.
      "both"                — centred (primary keys) + uncentred stored under
                              uncentred_aggregate_r2 / uncentred_r2_sweep.

    Pass to collate_snapshots() once all k values are done.
    """
    print(f"  k={k}: computing metrics", flush=True)

    if r2_method == "both":
        r2_c = _compute_r2(k, snap, max_atoms, centre_mhat=True)
        r2_u = _compute_r2(k, snap, max_atoms, centre_mhat=False)
        r2_result = {
            **r2_c,
            "uncentred_aggregate_r2":     r2_u["aggregate_r2"],
            "uncentred_aggregate_r2_std": r2_u["aggregate_r2_std"],
            "uncentred_r2_sweep":         r2_u["r2_sweep"],
        }
    else:
        r2_result = _compute_r2(k, snap, max_atoms, centre_mhat=(r2_method == "centred"))

    return {
        "k":        snap["k"],
        "r2":       r2_result,
        "coverage": _compute_coverage(k, snap, support_min_fires, support_percentile),
        "phi":      _compute_phi(k, snap),
        "logs":     _compress_logs({k: snap.get("logs", [])}),
        "vis_data": None,
    }


def collate_snapshots(per_k_snaps: dict[int, dict]) -> dict:
    """Merge {k: compact_snap} from process_snapshot into the multi-k finalised format.

    The output is the same format generate_plots.py expects — ready to pickle.

    Example::

        per_k = {}
        for k in K_SWEEP:
            trainer = make_trainer(...)
            per_k[k] = trainer.train(return_snapshot=True, include_vis_data=(k == 10))
        results = collate_snapshots(per_k)
        pickle.dump(results, open("baseline_snapshots.b", "wb"))
    """
    k_values = sorted(per_k_snaps.keys())
    if not k_values:
        raise ValueError("per_k_snaps is empty")

    first = per_k_snaps[k_values[0]]
    r0 = first["r2"]
    phi0 = first["phi"]

    # R² — merge aggregate and per-instance sweep across all k values
    r2_merged: dict = {
        "k_values":         k_values,
        "n_atoms_range":    r0["n_atoms_range"],
        "instance_types":   r0["instance_types"],
        "aggregate_r2":     {},
        "aggregate_r2_std": {},
        "r2_sweep":         {name: {} for name in r0["r2_sweep"]},
    }
    has_uncentred = "uncentred_aggregate_r2" in r0
    if has_uncentred:
        r2_merged.update({
            "uncentred_aggregate_r2":     {},
            "uncentred_aggregate_r2_std": {},
            "uncentred_r2_sweep":         {name: {} for name in r0["uncentred_r2_sweep"]},
        })

    for k in k_values:
        r = per_k_snaps[k]["r2"]
        r2_merged["aggregate_r2"][k]     = r["aggregate_r2"][k]
        r2_merged["aggregate_r2_std"][k] = r["aggregate_r2_std"][k]
        for name in r["r2_sweep"]:
            r2_merged["r2_sweep"][name][k] = r["r2_sweep"][name][k]
        if has_uncentred and "uncentred_aggregate_r2" in r:
            r2_merged["uncentred_aggregate_r2"][k]     = r["uncentred_aggregate_r2"][k]
            r2_merged["uncentred_aggregate_r2_std"][k] = r["uncentred_aggregate_r2_std"][k]
            for name in r["uncentred_r2_sweep"]:
                r2_merged["uncentred_r2_sweep"][name][k] = r["uncentred_r2_sweep"][name][k]

    # Coverage
    cov_merged: dict = {
        "k_values":          k_values,
        "mean_support_size": {k: per_k_snaps[k]["coverage"]["mean_support_size"][k] for k in k_values},
        "mean_rf_diameter":  {k: per_k_snaps[k]["coverage"]["mean_rf_diameter"][k]  for k in k_values},
    }

    # Phi
    phi_merged: dict = {
        "k_values":             k_values,
        "instance_names":       phi0["instance_names"],
        "instance_ki":          phi0["instance_ki"],
        "phi_matrices":         {k: per_k_snaps[k]["phi"]["phi_matrices"][k]         for k in k_values},
        "atom_manifold_scores": {k: per_k_snaps[k]["phi"]["atom_manifold_scores"][k] for k in k_values},
        "rho_G":                {k: per_k_snaps[k]["phi"]["rho_G"][k]                for k in k_values},
    }

    # Logs — merge all per-k compressed log dicts
    logs_merged: dict = {}
    for k in k_values:
        logs_merged.update(per_k_snaps[k].get("logs", {}))

    out = {
        "r2":      r2_merged,
        "coverage": cov_merged,
        "phi":     phi_merged,
        "logs":    logs_merged,
    }

    # vis_data — take the first non-None entry (usually k=10)
    for k in k_values:
        vd = per_k_snaps[k].get("vis_data")
        if vd is not None:
            out["vis_data"] = vd
            break

    return out


# ── R² from vis_data (isolated single-manifold codes) ────────────────────────

def compute_r2_from_vis_data(
    vis_data: dict,
    max_atoms: int = 25,
    centre_mhat: bool = False,
) -> dict:
    """Compute R² sweep from a vis_data payload — no re-training needed.

    Works on finalised snapshots at the single k stored in vis_data (k=10 by default).
    Uses isolated single-manifold codes (iso_indices/iso_values), so results reflect
    per-manifold reconstruction quality on single-manifold inputs rather than the
    mixed-L0 snapshot R² (which the paper uses; see _compute_r2 for that).

    centre_mhat=False (default) matches paper Eq. 14. Pass True to diagnose
    mean-shift effects (inflates R² by removing the mean of M̂).

    Returns dict with keys: k, n_atoms_range, r2_sweep, aggregate_r2, centre_mhat.
    """
    W_dec = vis_data["W_dec"]
    d_sae = W_dec.shape[0]
    n_atoms_range = list(range(1, max_atoms + 1))
    r2_sweep: dict[str, list[float]] = {}
    ki_at_r2: list[float] = []

    for inst in vis_data["instances"]:
        name = inst["name"]
        k_i  = inst["k_i"]
        if "iso_indices" not in inst:
            continue
        codes = expand_iso_codes(inst, d_sae)
        sweep = _r2_sweep_for_instance(
            codes, inst["contribs"], W_dec, n_atoms_range, centre_mhat,
        )
        r2_sweep[name] = sweep
        if k_i <= max_atoms and math.isfinite(sweep[k_i - 1]):
            ki_at_r2.append(sweep[k_i - 1])

    valid = [v for v in ki_at_r2 if math.isfinite(v)]
    return {
        "k":            vis_data["k"],
        "n_atoms_range": n_atoms_range,
        "r2_sweep":     r2_sweep,
        "aggregate_r2": float(np.mean(valid)) if valid else float("nan"),
        "centre_mhat":  centre_mhat,
    }


# ── Geometry diagnostics ──────────────────────────────────────────────────────

def compute_tuning_curves(snap: dict) -> dict:
    """SAE feature activations vs. true intrinsic manifold coordinates.

    For each instance, selects k_i atoms by greedy OMP and returns their
    activations alongside the ground-truth normalised intrinsic coordinates.

    Returns dict[inst_name -> {
        "coords":        (n_j, k_i)  normalised intrinsic coordinates,
        "activations":   (n_j, k_i)  SAE activations for selected atoms,
        "atom_ids":      list[int],
        "instance_type": str,
    }]
    """
    codes = snap["eval_codes"]
    W_dec = snap["W_dec"]
    active_mask = snap["active_mask"]
    instance_coords = snap["instance_coords"]

    out: dict = {}
    for j, name in enumerate(snap["instance_names"]):
        act_j = active_mask[:, j]
        if act_j.sum() < 10:
            continue
        z_j = codes[act_j]
        m_j = snap["instance_contribs"][j]
        k_i = snap["instance_ki"][j]
        atoms = greedy_select_atoms(m_j, W_dec, k_i)
        out[name] = {
            "coords":        instance_coords[j],
            "activations":   z_j[:, atoms].astype(np.float32),
            "atom_ids":      atoms,
            "instance_type": snap["instance_types"][j],
        }
    return out


# ── Visualisation payload ─────────────────────────────────────────────────────

def _to_sparse(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense (n, d_sae) float32 → sparse (indices int16, values float32).

    Entries within each row are sorted by |value| descending so that top-n
    reconstruction is a simple slice of the first n columns.
    Unused slots are marked with index -1 and value 0.0 (padding).
    """
    n = z.shape[0]
    nz_counts = (z != 0).sum(1)
    k_max = int(nz_counts.max()) if n > 0 and nz_counts.max() > 0 else 0
    indices = np.full((n, k_max), -1, dtype=np.int16)
    values  = np.zeros((n, k_max), dtype=np.float32)
    for i in range(n):
        nz = np.where(z[i] != 0)[0]
        if len(nz) == 0:
            continue
        order = np.argsort(-np.abs(z[i, nz]))
        nz_s = nz[order]
        indices[i, :len(nz_s)] = nz_s.astype(np.int16)
        values[i,  :len(nz_s)] = z[i, nz_s]
    return indices, values


def expand_iso_codes(inst: dict, d_sae: int) -> np.ndarray:
    """Reconstruct dense (n_j, d_sae) codes from sparse iso_indices/iso_values."""
    idx = inst["iso_indices"]   # (n_j, k_max) int16, -1 = padding
    val = inst["iso_values"]    # (n_j, k_max) float32
    n_j = idx.shape[0]
    codes = np.zeros((n_j, d_sae), dtype=np.float32)
    valid = idx >= 0
    codes[np.where(valid)[0], idx[valid].astype(np.int32)] = val[valid]
    return codes


def build_vis_data(snap: dict, encode_fn=None, max_samples: int = 2000) -> dict:
    """Subsample a full snapshot to a compact visualisation payload.

    If encode_fn is provided, runs isolated single-manifold forward passes on
    instance contributions and stores activations in sparse format (~6 MB total
    vs ~192 MB for dense mixed codes). This gives clean reconstruction plots
    since every non-zero activation can be attributed to the manifold of interest.

    encode_fn: (n, d) float32 → (n, d_sae) float32. Should absorb the norm
        rescaling needed for distributional alignment with training — see train.py.

    Returns:
        k         — float, sparsity budget
        W_dec     — (d_sae, d) float32 decoder weight matrix
        instances — list of per-manifold dicts:
            name, type, k_i,
            contribs    — (min(n_j, max_samples), d)    float32
            coords      — (min(n_j, max_samples), k_i)  float32
            iso_indices — (min(n_j, max_samples), k)    int16  (if encode_fn given)
            iso_values  — (min(n_j, max_samples), k)    float32 (if encode_fn given)
    """
    instances = []
    for j, name in enumerate(snap["instance_names"]):
        contribs_j = snap["instance_contribs"][j][:max_samples].astype(np.float32)
        coords_j   = snap["instance_coords"][j][:max_samples].astype(np.float32)
        inst: dict = {
            "name":     name,
            "type":     snap["instance_types"][j],
            "k_i":      snap["instance_ki"][j],
            "contribs": contribs_j,
            "coords":   coords_j,
        }
        if encode_fn is not None:
            iso_z = encode_fn(contribs_j)                         # (n_j, d_sae)
            inst["iso_indices"], inst["iso_values"] = _to_sparse(iso_z)
        instances.append(inst)

    return {
        "k":        snap["k"],
        "W_dec":    snap["W_dec"].astype(np.float32),
        "instances": instances,
    }
