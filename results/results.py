"""
Computations for Figure 4 replication (arXiv 2604.28119, Figure 4).

Two usage modes:

  Incremental (memory-efficient — one snapshot in RAM at a time):

      state = {}
      for k in K_SWEEP:
          trainer.train()
          snap = trainer.extract_snapshot()
          update_results_state(k, snap, state)
          del snap, trainer
      fig4 = finalize_results_state(state)

  Batch (backward-compatible, requires all snapshots in memory):

      fig4 = compute_figure4(snapshots_by_k)

Panels:
  A) Restricted R²  — aggregate sweep over k, plus per-manifold sweep over n_atoms
  B) Phase diagram  — mean support size and RF diameter per k
  C) CCA affinity   — pairwise phi-coefficient matrix + block-diagonal atom ordering

Geometry diagnostics:
  compute_tuning_curves  — SAE feature activations vs. true intrinsic coordinates

TODO (Panel C): Replace phi coefficient with the Ising inverse coupling matrix.

Bug fixes applied:
  Bug-1: contributions divided by norm_scale in extract_snapshot
  Bug-2: support uses per-atom 10th-percentile threshold + 30-fire floor
  Bug-3: RF diameter uses median across support atoms
  Bug-4: R² atom selection uses greedy OMP on decoder directions (Appendix E)
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


# ── Geometric greedy OMP + SVD deflation (paper's compute_r2 algorithm) ─────────
#
# Selection:    score = ||residual @ W.T||² per atom (geometric — decoder directions)
# Deflation:    exact SVD projection onto span(W_dec[selected]) after each step
# Reconstruction: actual SAE codes — M_hat = Z[:, selected] @ W_dec[selected]
# Centering:    M_hat_c = M_hat - M_hat.mean(0)  ← removes encoder bias and
#               L0>1 cross-manifold contamination from uncentered reconstruction
# R²:           1 - ||M_c - M_hat_c||² / ||M_c||²  where M_c = M - M.mean(0)

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

    for _ in range(min(n, d_sae)):
        var_exp = (residual_t @ W_t.T).pow(2).sum(0)          # (d_sae,)
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

    for _ in range(min(n, W_dec.shape[0])):
        scores = ((residual @ W_dec.T) ** 2).sum(0)            # (d_sae,) variance explained per atom
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


def _greedy_select(m_j: np.ndarray, W_dec: np.ndarray, n: int) -> list[int]:
    """Geometric OMP for compute_tuning_curves."""
    return _geometric_greedy_np(m_j - m_j.mean(0), W_dec, n)


# ── Panel A helpers ───────────────────────────────────────────────────────────

def _r2_sweep_for_instance(
    z_j: np.ndarray,
    m_j: np.ndarray,
    W_dec: np.ndarray,
    n_atoms_range: list[int],
) -> list[float]:
    """R² sweep — numpy fallback. Matches paper's compute_r2 exactly."""
    if len(z_j) < 10:
        return [float("nan")] * len(n_atoms_range)
    m_c = m_j - m_j.mean(0)
    ss_tot = float((m_c ** 2).sum())
    if ss_tot < 1e-12:
        return [1.0] * len(n_atoms_range)
    selected = _geometric_greedy_np(m_c, W_dec, max(n_atoms_range))
    n_sel = len(selected)
    results = []
    for n in n_atoms_range:
        idx = min(n, n_sel)
        m_hat = z_j[:, selected[:idx]] @ W_dec[selected[:idx]]
        m_hat_c = m_hat - m_hat.mean(0)       # centre M_hat — removes bias
        ss_res = float(((m_c - m_hat_c) ** 2).sum())
        results.append(float(1.0 - ss_res / (ss_tot + 1e-12)))
    return results


# ── Panel B helpers ───────────────────────────────────────────────────────────

def _support_stats(
    z_j: np.ndarray,
    min_fires: int,
    percentile: float,
) -> tuple[int, list[float]]:
    """(n_support, coverages) for one manifold instance.

    Paper (Appendix E): atom is in support iff
      - fires on ≥ 10% of the n_j manifold-active eval points  (relative threshold)
      - fires on ≥ min_fires points  (absolute floor for tiny manifolds)
    "Firing" = |z| above the atom's 10th-percentile nonzero activation magnitude.

    GPU path vectorises the per-atom percentile loop via sort+gather over the
    full (n_j, d_sae) matrix — eliminates 512 serial np.percentile calls.
    """
    n_j = z_j.shape[0]
    min_fires_eff = max(min_fires, int(0.10 * n_j))   # paper's 10% relative condition

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
            in_support = candidates & (above >= min_fires_eff)
            n_support = int(in_support.sum().item())
            if n_support == 0:
                return 0, []

            coverages = fire[:, in_support].float().mean(0).tolist()
            return n_support, coverages

    fire = (z_j != 0)
    n_fires = fire.sum(0)
    candidates = np.where(n_fires >= min_fires)[0]
    d_sae = z_j.shape[1]
    in_support = np.zeros(d_sae, dtype=bool)
    for a in candidates:
        nz = np.abs(z_j[fire[:, a], a])
        threshold = float(np.percentile(nz, percentile))
        if (np.abs(z_j[:, a]) >= threshold).sum() >= min_fires_eff:
            in_support[a] = True
    n_support = int(in_support.sum())
    if n_support == 0:
        return 0, []
    coverages = fire[:, in_support].mean(0).tolist()
    return n_support, coverages


# ── Panel C helpers ───────────────────────────────────────────────────────────

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


def _panel_c_stats(
    eval_codes: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Global marginal phi matrix + atom-manifold scores.

    Marginal phi is kept for the block-diagonal visualisation (phi_matrix plot).
    For ρ(G) computation, use _conditional_rho_G instead — it removes the
    joint-silence confounder that inflates marginal phi.
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


# ── Incremental state (private) ───────────────────────────────────────────────

def _init_panel_a(snap: dict, max_atoms: int) -> dict:
    return {
        "max_atoms": max_atoms,
        "n_atoms_range": list(range(1, max_atoms + 1)),
        "instance_names": snap["instance_names"],
        "instance_ki": snap["instance_ki"],
        "instance_types": dict(zip(snap["instance_names"], snap["instance_types"])),
        "r2_sweep": {name: {} for name in snap["instance_names"]},
        "aggregate_r2": {},
        "aggregate_r2_std": {},
        "k_values": [],
    }


def _update_panel_a(k: int, snap: dict, st: dict) -> None:
    print(f"  [Panel A] k={k}", flush=True)
    codes = snap["eval_codes"]    # (N, d_sae)
    W_dec = snap["W_dec"]         # (d_sae, d)
    active_mask = snap["active_mask"]
    n_atoms_range = st["n_atoms_range"]
    st["k_values"].append(k)

    W_t = torch.from_numpy(W_dec).to(_DEVICE) if _TORCH else None
    max_n = max(n_atoms_range)

    r2_at_ki: list[float] = []
    for j, (name, k_i) in enumerate(zip(st["instance_names"], st["instance_ki"])):
        act_j = active_mask[:, j]
        if act_j.sum() < 10:
            st["r2_sweep"][name][k] = [float("nan")] * len(n_atoms_range)
            continue

        z_j = codes[act_j]                  # (n_j, d_sae) actual codes
        m_j = snap["instance_contribs"][j]  # (n_j, d)

        m_c = m_j - m_j.mean(0)            # centre contribution (Eq. 14 denominator)
        ss_tot = float((m_c ** 2).sum())
        if ss_tot < 1e-12:
            st["r2_sweep"][name][k] = [1.0] * len(n_atoms_range)
            continue

        if _TORCH:
            with torch.no_grad():
                m_c_t = torch.from_numpy(m_c).to(_DEVICE)
                z_t   = torch.from_numpy(z_j).to(_DEVICE)

                # Geometric OMP with SVD deflation — matching paper's compute_r2
                selected = _geometric_greedy_gpu_t(m_c_t, W_t, W_dec, max_n)

                if not selected:
                    st["r2_sweep"][name][k] = [float("nan")] * len(n_atoms_range)
                    continue

                n_sel  = len(selected)
                sel_t  = torch.tensor(selected, device=_DEVICE, dtype=torch.long)

                # Reconstruction with actual codes, then centre M_hat
                # atom_c[sample, i, dim] = z[sample, atom_i] * W[atom_i, dim]
                atom_c    = z_t[:, sel_t].unsqueeze(-1) * W_t[sel_t].unsqueeze(0)  # (n_j, n_sel, d)
                recon_cum = atom_c.cumsum(1)                                         # (n_j, n_sel, d)
                # Centre each cumulative reconstruction over samples — removes mean offset
                recon_cum_c = recon_cum - recon_cum.mean(0, keepdim=True)           # (n_j, n_sel, d)

                ss_res = ((m_c_t.unsqueeze(1) - recon_cum_c) ** 2).sum(dim=(0, 2)) # (n_sel,)

                sweep = []
                for n in n_atoms_range:
                    idx = min(n, n_sel) - 1
                    sweep.append(float(1.0 - ss_res[idx].item() / (ss_tot + 1e-12)))
        else:
            sweep = _r2_sweep_for_instance(z_j, m_j, W_dec, n_atoms_range)

        st["r2_sweep"][name][k] = sweep
        if k_i <= st["max_atoms"] and math.isfinite(sweep[k_i - 1]):
            r2_at_ki.append(sweep[k_i - 1])

    valid = [v for v in r2_at_ki if math.isfinite(v)]
    st["aggregate_r2"][k] = float(np.mean(valid)) if valid else float("nan")
    st["aggregate_r2_std"][k] = float(np.std(valid)) if len(valid) > 1 else 0.0


def _finalize_panel_a(st: dict) -> dict:
    return {
        "k_values": sorted(st["k_values"]),
        "n_atoms_range": st["n_atoms_range"],
        "aggregate_r2": st["aggregate_r2"],
        "aggregate_r2_std": st["aggregate_r2_std"],
        "r2_sweep": st["r2_sweep"],
        "instance_types": st["instance_types"],
    }


def _init_panel_b() -> dict:
    return {"k_values": [], "mean_support_size": {}, "mean_rf_diameter": {}}


def _update_panel_b(
    k: int, snap: dict, st: dict, min_fires: int, percentile: float,
) -> None:
    print(f"  [Panel B] k={k}", flush=True)
    codes = snap["eval_codes"]
    active_mask = snap["active_mask"]
    st["k_values"].append(k)

    support_sizes: list[int] = []
    all_coverages: list[float] = []
    for j in range(active_mask.shape[1]):
        act_j = active_mask[:, j]
        n_j = int(act_j.sum())
        if n_j < min_fires:
            continue

        # Paper (Appendix E): count ALL atoms that fire on ≥ 10% of manifold j's eval
        # points. The 10% threshold in _support_stats naturally excludes atoms from the
        # L0-1 co-active manifolds (their conditional fire rate ≈ (L0-1)/(M-1) ≈ 6.4%
        # for our zoo with M=48, L0=4 — below the 10% cutoff).
        z_j = codes[act_j]   # (n_j, d_sae) — all atoms
        n_sup, covs = _support_stats(z_j, min_fires, percentile)
        if n_sup > 0:
            support_sizes.append(n_sup)
            all_coverages.extend(covs)

    st["mean_support_size"][k] = float(np.mean(support_sizes)) if support_sizes else 0.0
    st["mean_rf_diameter"][k] = float(np.median(all_coverages)) if all_coverages else 0.0


def _finalize_panel_b(st: dict) -> dict:
    return {
        "k_values": sorted(st["k_values"]),
        "mean_support_size": st["mean_support_size"],
        "mean_rf_diameter": st["mean_rf_diameter"],
    }


def _init_panel_c(snap: dict) -> dict:
    return {
        "k_values": [],
        "phi_matrices": {},
        "atom_order": {},
        "atom_dominant": {},
        "atom_scores": {},
        "cond_rho_G": {},
        "instance_names": snap["instance_names"],
        "instance_ki": snap["instance_ki"],
    }


def _update_panel_c(
    k: int, snap: dict, st: dict,
    phi: np.ndarray, scores: np.ndarray, dominant: np.ndarray,
) -> None:
    print(f"  [Panel C] k={k}", flush=True)
    order = np.argsort(dominant, kind="stable")
    st["k_values"].append(k)
    st["phi_matrices"][k] = phi
    st["atom_order"][k] = order
    st["atom_dominant"][k] = dominant
    st["atom_scores"][k] = scores
    st["cond_rho_G"][k] = _rho_G_from_global_phi(
        phi, scores, st["instance_names"], st["instance_ki"],
    )


def _finalize_panel_c(st: dict) -> dict:
    return {
        "k_values": sorted(st["k_values"]),
        "phi_matrices": st["phi_matrices"],
        "atom_order": st["atom_order"],
        "atom_dominant_manifold": st["atom_dominant"],
        "atom_manifold_scores": st["atom_scores"],
        "cond_rho_G": st["cond_rho_G"],
        "instance_names": st["instance_names"],
        "instance_ki": st["instance_ki"],
    }


# ── Public incremental API ────────────────────────────────────────────────────

def update_results_state(
    k: int,
    snap: dict,
    state: dict,
    max_atoms: int = 25,
    support_min_fires: int = 30,
    support_percentile: float = 10.0,
    fig4d_k: int = 10,
) -> None:
    """Process one snapshot into state. Call once per k, then finalize_results_state().

    Automatically captures:
      - training logs from snap["logs"] (every k)
      - Fig 4D payload via build_fig4d_payload (at k == fig4d_k, default 10)

    Example::

        state = {}
        for k in K_SWEEP:
            trainer.train()
            snap = trainer.extract_snapshot()
            update_results_state(k, snap, state)
            del snap, trainer
        fig4 = finalize_results_state(state)
        pickle.dump(fig4, open("variant_snapshots.b", "wb"))
        # fig4 now contains panel_a/b/c, logs, and fig4d
    """
    if not state:
        state["panel_a"] = _init_panel_a(snap, max_atoms)
        state["panel_b"] = _init_panel_b()
        state["panel_c"] = _init_panel_c(snap)
        state["logs"] = {}
        state["fig4d"] = None

    # Compute phi + scores once per snapshot; Panel B no longer needs dominant.
    phi, scores = _panel_c_stats(snap["eval_codes"], snap["active_mask"])
    dominant = scores.argmax(0)   # (d_sae,) — dominant manifold per atom, used by Panel C only

    _update_panel_a(k, snap, state["panel_a"])
    _update_panel_b(k, snap, state["panel_b"], support_min_fires, support_percentile)
    _update_panel_c(k, snap, state["panel_c"], phi, scores, dominant)

    if "logs" in snap:
        state["logs"][k] = snap["logs"]

    if k == fig4d_k:
        state["fig4d"] = build_fig4d_payload(snap)


def finalize_results_state(state: dict) -> dict:
    """Return completed Figure 4 data as a single dict ready to pickle.

    Keys always present: panel_a, panel_b, panel_c
    Keys present when available: logs, fig4d
    """
    out = {
        "panel_a": _finalize_panel_a(state["panel_a"]),
        "panel_b": _finalize_panel_b(state["panel_b"]),
        "panel_c": _finalize_panel_c(state["panel_c"]),
    }
    if state.get("logs"):
        out["logs"] = state["logs"]
    if state.get("fig4d") is not None:
        out["fig4d"] = state["fig4d"]
    return out


# ── Batch API (backward-compatible) ──────────────────────────────────────────

def compute_panel_a(snapshots_by_k: dict[int, dict], max_atoms: int = 25) -> dict:
    st = _init_panel_a(next(iter(snapshots_by_k.values())), max_atoms)
    for k, snap in sorted(snapshots_by_k.items()):
        _update_panel_a(k, snap, st)
    return _finalize_panel_a(st)


def compute_panel_b(
    snapshots_by_k: dict[int, dict],
    min_fires: int = 30,
    percentile: float = 10.0,
) -> dict:
    st = _init_panel_b()
    for k, snap in sorted(snapshots_by_k.items()):
        _update_panel_b(k, snap, st, min_fires, percentile)
    return _finalize_panel_b(st)


def compute_panel_c(snapshots_by_k: dict[int, dict]) -> dict:
    st = _init_panel_c(next(iter(snapshots_by_k.values())))
    for k, snap in sorted(snapshots_by_k.items()):
        phi, scores = _panel_c_stats(snap["eval_codes"], snap["active_mask"])
        _update_panel_c(k, snap, st, phi, scores, scores.argmax(0))
    return _finalize_panel_c(st)


def compute_figure4(
    snapshots_by_k: dict[int, dict],
    max_atoms: int = 25,
    support_min_fires: int = 30,
    support_percentile: float = 10.0,
) -> dict:
    """Compute all panels. Prefer update_results_state when memory is constrained."""
    state: dict = {}
    for k, snap in sorted(snapshots_by_k.items()):
        update_results_state(k, snap, state, max_atoms, support_min_fires, support_percentile)
    return finalize_results_state(state)


# ── Geometry diagnostics ──────────────────────────────────────────────────────

def compute_tuning_curves(snap: dict) -> dict:
    """SAE feature activations vs. true intrinsic manifold coordinates.

    For each instance, selects k_i atoms by greedy OMP and returns their
    activations alongside the ground-truth normalized intrinsic coordinates.

    Returns dict[inst_name -> {
        "coords":        (n_j, k_i)  normalized intrinsic coordinates,
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
        atoms = _greedy_select(m_j, W_dec, k_i)
        out[name] = {
            "coords":        instance_coords[j],
            "activations":   z_j[:, atoms].astype(np.float32),
            "atom_ids":      atoms,
            "instance_type": snap["instance_types"][j],
        }
    return out


# ── Fig 4D payload ────────────────────────────────────────────────────────────

def build_fig4d_payload(snap: dict, max_samples: int = 2000) -> dict:
    """Trim a full snapshot down to the arrays needed for Fig 4D.

    Designed to be called at k=10 before the snapshot is discarded:
        if k == 10:
            pickle.dump(build_fig4d_payload(snap), open("signed_fig4d_k10.b", "wb"))

    Returns:
        k        — float, sparsity budget
        W_dec    — (d_sae, d) float32 decoder weight matrix
        instances — list of per-manifold dicts:
            name, type, k_i,
            codes    — (min(n_j, max_samples), d_sae) float32
            coords   — (min(n_j, max_samples), k_i)  float32
            contribs — (min(n_j, max_samples), d)    float32
    """
    active_mask = snap["active_mask"]
    eval_codes = snap["eval_codes"]

    instances = []
    for j, name in enumerate(snap["instance_names"]):
        act_j = active_mask[:, j]
        row_ids = np.where(act_j)[0][:max_samples]
        instances.append({
            "name":    name,
            "type":    snap["instance_types"][j],
            "k_i":     snap["instance_ki"][j],
            "codes":   eval_codes[row_ids].astype(np.float32),
            "coords":  snap["instance_coords"][j][:max_samples].astype(np.float32),
            "contribs": snap["instance_contribs"][j][:max_samples].astype(np.float32),
        })

    return {
        "k":         snap["k"],
        "W_dec":     snap["W_dec"].astype(np.float32),
        "instances": instances,
    }
