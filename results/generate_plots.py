"""
Generate Figure 4 plots from snapshot files.

Usage:
    python results/generate_plots.py <snapshot_files...> [--figures FIGURE ...]

Examples:
    python results/generate_plots.py results/snapshots/*3.b
    python results/generate_plots.py results/snapshots/*4.b --figures aggregate_r2 phase_diagram

Snapshot filenames must follow the pattern {variant}_snapshots{suffix}
(e.g. baseline_snapshots3.b, signed_snapshots3.b). The suffix determines
the output tag used in figure filenames (e.g. suffix 3.b → tag 3b →
aggregate_r2_3b.png).
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results as _results  # noqa: E402  (results.py lives alongside this script)

FIGS_DIR = Path("results/figs/")

ALL_FIGURES = [
    "aggregate_r2",
    "phase_diagram",
    "r2_by_type",
    "r2_at_ki_bar",
    "phi_capture",
    "phi_matrix",
    "signed_cohesion",
    "loss_curves",
    "coordinate_encoding",
]
EXTRA_FIGURES = ["reconstruction_grid"]  # not in ALL_FIGURES; requires --shape

COLORS = {"baseline": "steelblue", "baseline_batch": "orange", "signed": "green", "penal": "red"}
LABELS = {"baseline": "TopK (baseline)", "baseline_batch": "BatchTopK",
          "signed": "Signed", "penal": "Penalised"}
ATOM_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb",
]

_KI = {"circle": 2, "sphere": 3, "torus": 4, "mobius": 3,
       "swiss_roll": 3, "helix": 3, "flat_disk": 2, "segment": 1}


def _inst_type(name: str) -> str:
    for t in _KI:
        if name.startswith(t):
            return t
    return name.split("_")[0]


def parse_snapshot_filename(path: str) -> tuple[str, str] | None:
    """Extract (variant, tag) from a snapshot filename.

    Pattern: {variant}_snapshots{suffix}  e.g. baseline_snapshots3.b → ("baseline", "3b")
    Returns None if the filename doesn't match.
    """
    basename = os.path.basename(path)
    m = re.match(r"^(.+?)_snapshots(.+)$", basename)
    if not m:
        return None
    variant = m.group(1)
    tag = m.group(2).replace(".", "")
    return variant, tag


# ── Load ──────────────────────────────────────────────────────────────────────

def load_snapshot(path: str) -> dict:
    print(f"Loading {os.path.basename(path)} ...", end=" ", flush=True)
    with open(path, "rb") as f:
        d = pickle.load(f)
    print("done.")
    return d


def extract_loss_curves(logs: dict) -> dict[int, dict[str, list]]:
    """Convert compressed logs to plottable lists.

    logs: {k: {steps: int32_arr, l1_loss: float32_arr, fvu: float32_arr, n_dead: float32_arr}}
    """
    return {
        k: {m: entry[m].tolist() for m in ("steps", "l1_loss", "fvu", "n_dead") if m in entry}
        for k, entry in sorted(logs.items())
        if isinstance(entry, dict) and "steps" in entry
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_loss_curves(loss_data: dict[str, dict[int, dict]], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants_with_loss = {v: lc for v, lc in loss_data.items() if lc}
    if not variants_with_loss:
        print("  No loss data found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    metrics = ["l1_loss", "fvu", "n_dead"]
    titles  = ["ℓ1 reconstruction loss", "FVU (fraction variance unexplained)", "Dead neurons"]

    for mi, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[mi]
        cmap = plt.get_cmap("viridis")
        for v, lc in variants_with_loss.items():
            k_vals = sorted(lc.keys())
            for i, k in enumerate(k_vals):
                curve = lc[k]
                color = cmap(i / max(len(k_vals) - 1, 1))
                label = f"{LABELS.get(v, v)} k={k}" if mi == 0 else None
                ax.plot(curve["steps"], curve[metric], lw=0.8, alpha=0.7, color=color, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("training step")
        ax.set_ylabel(metric)
        if mi == 0:
            ax.legend(fontsize=5, ncol=2, loc="upper right")

    ax4 = axes[3]
    for v, lc in variants_with_loss.items():
        k_vals = sorted(lc.keys())
        final_l1 = []
        for k in k_vals:
            vals = [x for x in lc[k]["l1_loss"] if np.isfinite(x)]
            final_l1.append(vals[-1] if vals else float("nan"))
        ax4.plot(k_vals, final_l1, "o-", color=COLORS.get(v, "gray"), label=LABELS.get(v, v))
    ax4.set_xlabel("k")
    ax4.set_ylabel("Final ℓ1 loss")
    ax4.set_title("Final training loss vs k")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_aggregate_r2(data: dict[str, dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for v, d in data.items():
        pa = d["r2"]
        ks   = pa["k_values"]
        r2s  = [pa["aggregate_r2"][k] for k in ks]
        stds = [pa["aggregate_r2_std"][k] for k in ks]
        ax.plot(ks, r2s, "o-", color=COLORS.get(v, "gray"), label=LABELS.get(v, v))
        ax.fill_between(ks,
                        [r - s for r, s in zip(r2s, stds)],
                        [r + s for r, s in zip(r2s, stds)],
                        alpha=0.15, color=COLORS.get(v, "gray"))
    ax.set_xlabel("Training sparsity k")
    ax.set_ylabel("Aggregate restricted R²")
    ax.set_title("Aggregate restricted R² vs training sparsity k")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_phase_diagram(data: dict[str, dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_sup, ax_rf) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Phase diagram: support size and RF diameter vs k", fontsize=10)
    for v, d in data.items():
        pb = d["coverage"]
        ks  = pb["k_values"]
        sup = [pb["mean_support_size"][k] for k in ks]
        rf  = [pb["mean_rf_diameter"][k] for k in ks]
        ax_sup.plot(ks, sup, "o-", color=COLORS.get(v, "gray"), label=LABELS.get(v, v))
        ax_rf.plot(ks, rf,  "o-", color=COLORS.get(v, "gray"), label=LABELS.get(v, v))
    ax_sup.set_xlabel("k"); ax_sup.set_ylabel("Mean support size")
    ax_sup.set_title("Support size vs k"); ax_sup.legend(fontsize=8)
    ax_rf.set_xlabel("k"); ax_rf.set_ylabel("Mean RF diameter (coverage)")
    ax_rf.set_title("RF diameter vs k"); ax_rf.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_r2_by_type(data: dict[str, dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = sorted(_KI.keys())

    ref_k = None
    if "baseline" in data:
        pa = data["baseline"]["r2"]
        ref_k = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))
    if ref_k is None:
        pa = next(iter(data.values()))["r2"]
        ref_k = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
    axes = axes.flatten()
    fig.suptitle(f"R²(n_atoms) at k={ref_k} by manifold type", fontsize=10)

    for ti, mtype in enumerate(types):
        ax = axes[ti]
        ki = _KI[mtype]
        for v, d in data.items():
            pa = d["r2"]
            sweeps = []
            for inst_name, by_k in pa["r2_sweep"].items():
                if _inst_type(inst_name) == mtype and ref_k in by_k:
                    sweeps.append(by_k[ref_k])
            if not sweeps:
                continue
            mean_sweep = np.mean(sweeps, axis=0)
            n_atoms = list(range(1, len(mean_sweep) + 1))
            ax.plot(n_atoms, mean_sweep, color=COLORS.get(v, "gray"),
                    label=LABELS.get(v, v), lw=1.5)
        ax.axvline(ki, color="black", ls="--", lw=0.8, alpha=0.6, label=f"k_i={ki}")
        ax.axhline(1.0, color="gray", ls=":", lw=0.6)
        ax.set_title(f"{mtype}  (k_i={ki})", fontsize=9)
        ax.set_xlabel("n_atoms", fontsize=8)
        ax.set_ylabel("R²", fontsize=8)
        ax.set_xlim(1, 12)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=6, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_r2_at_ki_bar(data: dict[str, dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = sorted(_KI.keys())
    k_vals = sorted(next(iter(data.values()))["r2"]["k_values"])
    x = np.arange(len(types))
    width = 0.8 / len(data)
    offsets = (np.arange(len(data)) - (len(data) - 1) / 2) * width

    fig, axes = plt.subplots(1, len(k_vals), figsize=(2.2 * len(k_vals), 4), sharey=True)
    if len(k_vals) == 1:
        axes = [axes]
    fig.suptitle("R² at n_atoms = k_i", fontsize=10)

    for ki_ax, k in zip(axes, k_vals):
        for i, (v, d) in enumerate(data.items()):
            pa = d["r2"]
            scores = []
            for mtype in types:
                ki = _KI[mtype]
                type_r2s = []
                for inst_name, by_k in pa["r2_sweep"].items():
                    if _inst_type(inst_name) == mtype and k in by_k:
                        sweep = by_k[k]
                        if ki - 1 < len(sweep):
                            type_r2s.append(sweep[ki - 1])
                scores.append(np.mean(type_r2s) if type_r2s else float("nan"))
            ki_ax.bar(x + offsets[i], scores, width * 0.9,
                      color=COLORS.get(v, "gray"), alpha=0.8,
                      label=LABELS.get(v, v) if k == k_vals[0] else "")
        ki_ax.set_title(f"k={k}", fontsize=8)
        ki_ax.set_xticks(x)
        ki_ax.set_xticklabels([t[:5] for t in types], rotation=35, ha="right", fontsize=7)
        ki_ax.axhline(0, color="black", lw=0.4)
        ki_ax.set_ylim(-0.1, 1.05)

    axes[0].set_ylabel("R² at n_atoms=k_i")
    axes[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_phi_capture(data: dict[str, dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_vals = sorted(next(iter(data.values()))["r2"]["k_values"])
    sweet_k = {}
    for v, d in data.items():
        pa = d["r2"]
        sweet_k[v] = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))
    ref_k = sweet_k.get("baseline", k_vals[len(k_vals) // 2])

    fig, (ax_mean, ax_hist) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Within-manifold phi coherence  |  positive = capture, negative = tiling",
                 fontsize=10)

    for v, d in data.items():
        pc = d["phi"]
        mean_phis = []
        for k in k_vals:
            if k not in pc["phi_matrices"]:
                mean_phis.append(float("nan"))
                continue
            phi       = pc["phi_matrices"][k]
            scores    = pc["atom_manifold_scores"][k]
            inst_names = pc["instance_names"]
            within_vals = []
            for j, inst_name in enumerate(inst_names):
                ki = _KI.get(_inst_type(inst_name), 2)
                top_atoms = np.argsort(scores[:, j])[-ki:]
                if len(top_atoms) < 2:
                    continue
                sub  = phi[np.ix_(top_atoms, top_atoms)]
                mask = ~np.eye(len(top_atoms), dtype=bool)
                within_vals.extend(sub[mask].tolist())
            mean_phis.append(float(np.mean(within_vals)) if within_vals else float("nan"))
        ax_mean.plot(k_vals, mean_phis, "o-", color=COLORS.get(v, "gray"), label=LABELS.get(v, v))

    ax_mean.set_xlabel("Training sparsity k")
    ax_mean.set_ylabel("Mean within-manifold phi")
    ax_mean.set_title("Phi coherence vs k  (capture ↑, tiling ↓)")
    ax_mean.legend(fontsize=8)

    for v, d in data.items():
        pc = d["phi"]
        if ref_k not in pc["phi_matrices"]:
            continue
        phi       = pc["phi_matrices"][ref_k]
        scores    = pc["atom_manifold_scores"][ref_k]
        inst_names = pc["instance_names"]
        within_vals = []
        for j, inst_name in enumerate(inst_names):
            ki = _KI.get(_inst_type(inst_name), 2)
            top_atoms = np.argsort(scores[:, j])[-ki:]
            if len(top_atoms) < 2:
                continue
            sub  = phi[np.ix_(top_atoms, top_atoms)]
            mask = ~np.eye(len(top_atoms), dtype=bool)
            within_vals.extend(sub[mask].tolist())
        ax_hist.hist(within_vals, bins=40, alpha=0.5, color=COLORS.get(v, "gray"),
                     label=LABELS.get(v, v), density=True)

    ax_hist.set_xlabel("Within-manifold phi (pairwise)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Distribution at k={ref_k}")
    ax_hist.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_phi_matrix(data: dict[str, dict], out_path: str, k: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_variants = {v: d for v, d in data.items() if v != "baseline_batch"}
    if not plot_variants:
        plot_variants = data

    fig, axes = plt.subplots(1, len(plot_variants), figsize=(5.5 * len(plot_variants), 5.5))
    if len(plot_variants) == 1:
        axes = [axes]
    fig.suptitle(f"Phi matrix — live atoms only, sorted by manifold  (k={k})", fontsize=10)

    im = None
    for ax, (v, d) in zip(axes, plot_variants.items()):
        pc = d["phi"]
        if k not in pc["phi_matrices"]:
            ax.set_title(f"{LABELS.get(v, v)}\n(k={k} not found)")
            continue

        phi    = pc["phi_matrices"][k]
        scores = pc["atom_manifold_scores"][k]

        max_score = scores.max(axis=1)
        threshold = max_score.max() * 0.05
        live      = max_score > threshold
        live_ids  = np.where(live)[0]

        dominant_live = scores[live_ids].argmax(axis=1)
        inner_order   = np.argsort(dominant_live, kind="stable")
        order         = live_ids[inner_order]

        phi_ord = phi[np.ix_(order, order)]
        n_live  = len(order)

        im = ax.imshow(phi_ord, vmin=-1, vmax=1, cmap="RdBu_r",
                       interpolation="nearest", aspect="equal")
        ax.set_title(f"{LABELS.get(v, v)}\n({n_live} live atoms)", fontsize=9)
        ax.set_xlabel("atom (sorted by manifold)")
        ax.set_ylabel("atom")

        boundaries = np.where(np.diff(dominant_live[inner_order]))[0] + 1
        for b in boundaries:
            ax.axhline(b - 0.5, color="black", lw=0.5, alpha=0.6)
            ax.axvline(b - 0.5, color="black", lw=0.5, alpha=0.6)

    if im is not None:
        fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="phi")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_signed_cohesion(data: dict[str, dict], out_path: str, k: int = 10, L0: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc0        = next(iter(data.values()))["phi"]
    inst_names = pc0["instance_names"]
    k_vals     = sorted(pc0["k_values"])
    inst_ki    = pc0["instance_ki"]
    mean_ki    = sum(inst_ki) / len(inst_ki)
    tau1_k     = L0 * mean_ki

    fig, (ax_box, ax_vs_k) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Signed cohesion ρ(G) per manifold  (proxy: global marginal phi)", fontsize=10)

    def _get_rhos(pc, kk):
        if kk not in pc["phi_matrices"]:
            return []
        phi    = pc["phi_matrices"][kk]
        scores = pc["atom_manifold_scores"][kk]
        rhos   = []
        for j, name in enumerate(inst_names):
            ki = _KI.get(_inst_type(name), 2)
            top_atoms = np.argsort(scores[:, j])[-ki:]
            if len(top_atoms) < 2:
                continue
            sub  = phi[np.ix_(top_atoms, top_atoms)]
            mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
            rhos.append(float(np.mean(np.sign(sub[mask]))))
        return rhos

    for vi, (v, d) in enumerate(data.items()):
        pc   = d["phi"]
        rhos = _get_rhos(pc, k)
        if not rhos:
            continue
        ax_box.boxplot(rhos, positions=[vi], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=COLORS.get(v, "gray"), alpha=0.6),
                       medianprops=dict(color="black", lw=2))

    ax_box.set_xticks(range(len(data)))
    ax_box.set_xticklabels([LABELS.get(v, v) for v in data], rotation=10, ha="right", fontsize=8)
    ax_box.set_ylabel("ρ(G)")
    ax_box.set_title(f"Distribution at k={k}")

    for v, d in data.items():
        pc = d["phi"]
        mean_rhos = []
        for kk in k_vals:
            rh = _get_rhos(pc, kk)
            mean_rhos.append(float(np.mean(rh)) if rh else float("nan"))
        ax_vs_k.plot(k_vals, mean_rhos, "o-", color=COLORS.get(v, "gray"), label=LABELS.get(v, v))

    ax_vs_k.axvline(tau1_k, color="orange", lw=1.2, ls="--", label=f"τ=1  (k={tau1_k:.1f})")
    ax_vs_k.text(tau1_k * 0.35, -0.85, "shattering", color="gray", fontsize=7, ha="center")
    ax_vs_k.text(tau1_k * 1.7,  -0.85, "dilution",   color="gray", fontsize=7, ha="center")
    ax_vs_k.set_xlabel("Training sparsity k")
    ax_vs_k.set_ylabel("Mean ρ(G)")
    ax_vs_k.set_title("Signed cohesion vs k")
    ax_vs_k.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Reconstruction grid — Fig 11 style ───────────────────────────────────────

_RECON_N_ATOMS = (1, 2, 4, 8, 16)


def _find_instance(vis_data: dict, shape_name: str) -> dict | None:
    """Return the first instance whose name matches or contains shape_name."""
    for inst in vis_data["instances"]:
        if inst["name"] == shape_name or shape_name in inst["name"]:
            return inst
    return None


def _coord_colors(coords: np.ndarray) -> np.ndarray:
    """Map intrinsic coordinates to a scalar in [0, 1] for colormapping.

    Uses atan2 of the first two coord axes for k_i≥2 (captures circular structure).
    Falls back to normalised first axis for k_i=1.
    """
    if coords.shape[1] >= 2:
        angle = np.arctan2(coords[:, 1], coords[:, 0])
        return (angle - angle.min()) / max(angle.max() - angle.min(), 1e-8)
    c = coords[:, 0]
    return (c - c.min()) / max(c.max() - c.min(), 1e-8)


def _prepare_panels(
    inst: dict,
    W_dec: np.ndarray,
    n_pca: int,
    n_atoms_list: tuple[int, ...] = _RECON_N_ATOMS,
) -> dict | None:
    """Fig 11 style: per-sample top-n reconstruction, colored by intrinsic coordinate.

    For each column n in n_atoms_list, every sample uses its n most-active atoms
    from the isolated single-manifold forward pass. The color is fixed to the
    intrinsic manifold coordinate, making piecewise-linear structure visible as a
    continuous color gradient.
    """
    from sklearn.decomposition import PCA

    if "iso_indices" not in inst:
        return None

    contribs = inst["contribs"]     # (n_j, d)
    coords   = inst["coords"]       # (n_j, k_i)
    iso_idx  = inst["iso_indices"]  # (n_j, k_max) int16, sorted by |val| desc, -1=padding
    iso_val  = inst["iso_values"]   # (n_j, k_max) float32
    n_j      = len(contribs)

    if n_j == 0:
        return None

    pca        = PCA(n_components=n_pca)
    coords_pca = pca.fit_transform(contribs)
    colors     = _coord_colors(coords)

    valid_mask = iso_idx >= 0  # (n_j, k_max)
    safe_idx   = np.where(valid_mask, iso_idx, 0).astype(np.int32)  # safe for W_dec indexing

    recons = []
    for n in n_atoms_list:
        n = min(n, iso_idx.shape[1])
        top_val = np.where(valid_mask[:, :n], iso_val[:, :n], 0.0)  # (n_j, n)
        top_W   = W_dec[safe_idx[:, :n]]                             # (n_j, n, d)
        recon_n = np.einsum("ij,ijd->id", top_val, top_W)            # (n_j, d)
        recons.append(pca.transform(recon_n))

    return {
        "coords_pca":   coords_pca,
        "colors":       colors,
        "recons":       recons,
        "n_atoms_list": n_atoms_list,
    }


def plot_reconstruction_grid(
    shape_name: str,
    snapshots_by_variant: dict[str, dict],
    out_path: str | None = None,
    interactive: bool = False,
    n_atoms_list: tuple[int, ...] = _RECON_N_ATOMS,
) -> None:
    """Fig 11-style reconstruction grid: rows=variants, cols=n_atoms, color=manifold coord.

    Args:
        shape_name: substring to match against instance names (e.g. "helix").
        snapshots_by_variant: {variant_name: finalised snapshot dict}
        out_path: PNG save path; defaults to results/figs/reconstruction_grid_{shape}.png
        interactive: if True return a Plotly figure (for notebooks); else save PNG.
        n_atoms_list: atom counts shown as columns (default 1,2,4,8,16).
    """
    n_pca = 3 if interactive else 2
    variant_panels: dict[str, dict] = {}

    for variant, snap in snapshots_by_variant.items():
        vis_data = snap.get("vis_data")
        if vis_data is None:
            print(f"  [{variant}] no vis_data in snapshot — skipping")
            continue
        inst = _find_instance(vis_data, shape_name)
        if inst is None:
            print(f"  [{variant}] no instance matching '{shape_name}' — skipping")
            continue
        panels = _prepare_panels(inst, vis_data["W_dec"], n_pca=n_pca, n_atoms_list=n_atoms_list)
        if panels is None:
            print(f"  [{variant}] no active samples — skipping")
            continue
        variant_panels[variant] = panels

    if not variant_panels:
        raise ValueError(f"No variants produced data for '{shape_name}'")

    if interactive:
        return _plotly_grid(shape_name, variant_panels)
    _matplotlib_grid(shape_name, variant_panels, out_path)


def _plotly_grid(shape_name: str, variant_panels: dict):
    """3D rotatable Plotly figure — call from a notebook, not the CLI."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    variants     = list(variant_panels.keys())
    n_variants   = len(variants)
    n_atoms_list = next(iter(variant_panels.values()))["n_atoms_list"]
    n_cols       = len(n_atoms_list) + 1  # PCA + reconstructions
    n_rows       = n_variants

    specs = [[{"type": "scene"}] * n_cols for _ in range(n_rows)]
    fig = make_subplots(rows=n_rows, cols=n_cols, specs=specs,
                        vertical_spacing=0.04, horizontal_spacing=0.02)

    for vi, (variant, p) in enumerate(variant_panels.items()):
        row = vi + 1
        colors = p["colors"]

        fig.add_trace(go.Scatter3d(
            x=p["coords_pca"][:, 0], y=p["coords_pca"][:, 1], z=p["coords_pca"][:, 2],
            mode="markers",
            marker=dict(size=2, color=colors, colorscale="Twilight", opacity=0.8),
            showlegend=False,
        ), row=row, col=1)

        for ci, recon in enumerate(p["recons"]):
            fig.add_trace(go.Scatter3d(
                x=recon[:, 0], y=recon[:, 1], z=recon[:, 2],
                mode="markers",
                marker=dict(size=2, color=colors, colorscale="Twilight", opacity=0.8),
                showlegend=False,
            ), row=row, col=ci + 2)

    axis_kw = dict(showticklabels=False, showgrid=False, zeroline=False, title="")
    fig.update_scenes(xaxis=axis_kw, yaxis=axis_kw, zaxis=axis_kw, aspectmode="cube")

    row_h = 1.0 / n_rows
    annots = []
    for vi, variant in enumerate(variants):
        annots.append(dict(
            text=f"<b>{LABELS.get(variant, variant)}</b>",
            x=-0.01, y=1.0 - (vi + 0.5) * row_h,
            xref="paper", yref="paper",
            showarrow=False, font=dict(size=9), xanchor="right",
        ))
    col_labels = ["PCA (true)"] + [f"k={n}" for n in n_atoms_list]
    for ci, label in enumerate(col_labels):
        annots.append(dict(
            text=label, x=(ci + 0.5) / n_cols, y=1.01,
            xref="paper", yref="paper",
            showarrow=False, font=dict(size=8, color="#666"),
        ))

    fig.update_layout(
        title=dict(text=f"Reconstruction grid — {shape_name}  (k_SAE=10)", font=dict(size=13)),
        height=300 * n_rows, margin=dict(l=80, r=10, t=60, b=10), annotations=annots,
    )
    return fig


def _matplotlib_grid(shape_name: str, variant_panels: dict, out_path: str | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    variants     = list(variant_panels.keys())
    n_variants   = len(variants)
    n_atoms_list = next(iter(variant_panels.values()))["n_atoms_list"]
    n_cols       = len(n_atoms_list) + 1

    fig = plt.figure(figsize=(2.8 * n_cols, 2.8 * n_variants))
    gs  = GridSpec(n_variants, n_cols, figure=fig, hspace=0.15, wspace=0.05)

    for vi, (variant, p) in enumerate(variant_panels.items()):
        colors  = p["colors"]
        scatter_kw = dict(c=colors, cmap="twilight", s=3, alpha=0.8, linewidths=0, vmin=0, vmax=1)

        ax = fig.add_subplot(gs[vi, 0])
        ax.scatter(p["coords_pca"][:, 0], p["coords_pca"][:, 1], **scatter_kw)
        if vi == 0:
            ax.set_title("PCA (true)", fontsize=8, pad=3)
        # set_ylabel is hidden by axis("off"); use a rotated text annotation instead
        ax.text(-0.08, 0.5, LABELS.get(variant, variant), transform=ax.transAxes,
                fontsize=8, rotation=90, va="center", ha="right")
        ax.set_aspect("equal"); ax.axis("off")

        for ci, (n, recon) in enumerate(zip(n_atoms_list, p["recons"])):
            ax = fig.add_subplot(gs[vi, ci + 1])
            ax.scatter(recon[:, 0], recon[:, 1], **scatter_kw)
            if vi == 0:
                ax.set_title(f"k={n}", fontsize=8, pad=3)
            ax.set_aspect("equal"); ax.axis("off")

    plt.suptitle(f"Reconstruction grid — {shape_name}  (k_SAE=10)", fontsize=10, y=1.01)

    if out_path:
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()


# ── Coordinate encoding diagnostic ───────────────────────────────────────────

def plot_coordinate_encoding(
    data: dict[str, dict],
    out_path: str,
    show_types: list[str] | None = None,
) -> None:
    """Diagnostic: do SAE atoms encode manifold coordinates globally?

    For each selected manifold type and variant, shows:
      Left panels: activation vs intrinsic coordinate for the k_i atoms with
        highest mean |z| on isolated single-manifold inputs. Signed variant →
        closed loops / smooth gradients. Baseline → fragmented positive bumps.
      Right panel: globality — fraction of manifold samples where each atom fires.

    Uses vis_data (stored at k=10) from each snapshot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    if show_types is None:
        show_types = ["segment", "circle", "sphere"]

    # variant → type → {"z_atoms": (n_j, k_i), "colors": (n_j,), "globality": (k_i,)}
    panels: dict[str, dict[str, dict]] = {}

    for variant, snap in data.items():
        vis_data = snap.get("vis_data")
        if vis_data is None:
            continue
        W_dec = vis_data["W_dec"]
        variant_panels: dict[str, dict] = {}

        d_sae = W_dec.shape[0]
        for inst in vis_data["instances"]:
            itype = inst["type"]
            if itype not in show_types or itype in variant_panels:
                continue
            if "iso_indices" not in inst:
                continue

            coords = inst["coords"]   # (n_j, k_i)
            k_i    = inst["k_i"]

            # Expand sparse iso codes and select the k_i atoms with highest mean |z|.
            # With isolated single-manifold inputs, these are directly the atoms that
            # respond to this manifold — no OMP needed.
            codes = _results.expand_iso_codes(inst, d_sae)    # (n_j, d_sae)
            top_atoms = np.argsort(-np.abs(codes).mean(0))[:k_i]

            z_atoms   = codes[:, top_atoms]                                    # (n_j, k_i)
            globality = (z_atoms != 0).mean(axis=0).astype(np.float32)        # (k_i,)

            variant_panels[itype] = {
                "z_atoms":   z_atoms.astype(np.float32),
                "colors":    _coord_colors(coords),
                "globality": globality,
                "k_i":       k_i,
            }

        panels[variant] = variant_panels

    variants_with_data = [v for v in data if v in panels and panels[v]]
    if not variants_with_data:
        print("  No vis_data found for coordinate encoding plot.")
        return

    n_variants = len(variants_with_data)
    n_types    = len(show_types)
    n_cols     = n_types + 1  # scatter cols + globality panel

    fig = plt.figure(figsize=(3.0 * n_types + 3.5, 2.8 * n_variants))
    gs  = GridSpec(n_variants, n_cols, figure=fig,
                   hspace=0.25, wspace=0.35,
                   width_ratios=[3.0] * n_types + [3.5])

    globality_by_variant: dict[str, dict[str, np.ndarray]] = {}

    for vi, variant in enumerate(variants_with_data):
        vp = panels[variant]
        globality_by_variant[variant] = {}

        for ti, mtype in enumerate(show_types):
            ax = fig.add_subplot(gs[vi, ti])
            if mtype not in vp:
                ax.axis("off")
                continue

            p      = vp[mtype]
            z      = p["z_atoms"]    # (n_j, k_i)
            k_i    = p["k_i"]
            globality_by_variant[variant][mtype] = p["globality"]

            # X-axis: normalised intrinsic coordinate (atan2-based for k_i≥2, linear for k_i=1)
            coord_axis = p["colors"]  # [0,1] scalar derived from intrinsic coords

            # Activation profiles: each atom's activation vs intrinsic coordinate.
            # For k_i=1: single curve; for k_i>1: k_i overlaid curves.
            # Signed atoms: smooth/global (span [−1,1] with a curve).
            # Baseline atoms: positive bumps (ReLU) covering different regions.
            sort_idx = np.argsort(coord_axis)
            x_sorted = coord_axis[sort_idx]
            z_sorted = z[sort_idx]

            for ai in range(k_i):
                ax.scatter(x_sorted, z_sorted[:, ai],
                           c=ATOM_COLORS[ai % len(ATOM_COLORS)],
                           s=1, alpha=0.5, linewidths=0,
                           label=f"atom {ai}")

            ax.axhline(0, color="gray", lw=0.6, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=6)
            if vi == 0:
                ax.set_title(f"{mtype}  (k_i={k_i})", fontsize=9)
            if vi == n_variants - 1:
                ax.set_xlabel("intrinsic coord (norm.)", fontsize=7)
            if ti == 0:
                ax.set_ylabel(f"{LABELS.get(variant, variant)}\nactivation", fontsize=7)
            else:
                ax.set_ylabel("activation", fontsize=7)

        # Globality panel — collected after loop
        ax_glob = fig.add_subplot(gs[vi, n_types])
        x_pos = 0
        for ti2, mtype2 in enumerate(show_types):
            if mtype2 not in vp:
                continue
            glob = globality_by_variant[variant][mtype2]  # (k_i,)
            k_i2 = vp[mtype2]["k_i"]
            color = COLORS.get(variant, "gray")
            for ai, g in enumerate(glob):
                ax_glob.bar(x_pos, g, color=color, alpha=0.7 - 0.15 * ai, edgecolor="none")
                ax_glob.text(x_pos, g + 0.02, f"{g:.2f}", ha="center", va="bottom",
                             fontsize=5.5)
                x_pos += 1
            x_pos += 0.4  # gap between types

        ax_glob.set_ylim(0, 1.15)
        ax_glob.axhline(1.0, color="gray", lw=0.6, ls="--", alpha=0.7)
        ax_glob.set_xticks([])
        ax_glob.set_ylabel("support fraction", fontsize=7)
        ax_glob.tick_params(labelsize=6)
        if vi == 0:
            ax_glob.set_title("Globality of selected atoms", fontsize=9)
        if vi == n_variants - 1:
            # Add type labels at bottom
            x_pos2 = 0
            for mtype2 in show_types:
                if mtype2 not in vp:
                    continue
                k_i3 = vp[mtype2]["k_i"]
                mid = x_pos2 + (k_i3 - 1) / 2
                ax_glob.text(mid, -0.13, mtype2[:5], ha="center", va="top",
                             fontsize=6.5, transform=ax_glob.transData)
                x_pos2 += k_i3 + 0.4

    plt.suptitle("Coordinate encoding: activation vs intrinsic coordinate (left) and globality (right)",
                 fontsize=10, y=1.02)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(data: dict[str, dict]) -> None:
    print("\n=== Summary: aggregate R² at sweet-spot k ===")
    for v, d in data.items():
        pa = d["r2"]
        sk = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))
        r2  = pa["aggregate_r2"][sk]
        std = pa["aggregate_r2_std"][sk]
        print(f"  {LABELS.get(v, v):22s}  sweet k={sk:>2}  R²={r2:.4f} ± {std:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 4 plots from snapshot files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Default figures: {', '.join(ALL_FIGURES)}\n"
            f"Extra figures (not in default): {', '.join(EXTRA_FIGURES)}"
        ),
    )
    parser.add_argument(
        "files", nargs="+",
        help="Snapshot files. Filenames must match {variant}_snapshots{suffix} "
             "(e.g. results/snapshots/*3.b).",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=ALL_FIGURES + EXTRA_FIGURES, metavar="FIGURE",
        help="Figures to generate (default: all in ALL_FIGURES).",
    )
    parser.add_argument(
        "--shape", default=None,
        help="Instance name or substring for reconstruction_grid figure (e.g. 'helix').",
    )
    args = parser.parse_args()

    figures = set(args.figures) if args.figures else set(ALL_FIGURES)

    variant_paths: dict[str, str] = {}
    tag: str | None = None

    for path in args.files:
        parsed = parse_snapshot_filename(path)
        if parsed is None:
            print(f"Warning: skipping unrecognized filename: {os.path.basename(path)}")
            continue
        variant, file_tag = parsed
        if tag is None:
            tag = file_tag
        elif file_tag != tag:
            print(f"Warning: mixed suffix tags ({tag} vs {file_tag}), using {tag!r} for output names")
        variant_paths[variant] = path

    if not variant_paths:
        print("No valid snapshot files found.")
        sys.exit(1)

    os.makedirs(FIGS_DIR, exist_ok=True)

    finalised: dict[str, dict] = {}
    loss_data: dict[str, dict[int, dict]] = {}

    for v, path in variant_paths.items():
        if not os.path.exists(path):
            print(f"  {v}: {path} not found, skipping.")
            continue
        fin = load_snapshot(path)
        finalised[v] = fin
        loss_data[v] = extract_loss_curves(fin.get("logs", {}))

    if not finalised:
        print("No data loaded — nothing to plot.")
        sys.exit(1)

    print_summary(finalised)

    if "aggregate_r2"  in figures: plot_aggregate_r2(finalised,  FIGS_DIR / "aggregate_r2.png")
    if "phase_diagram" in figures: plot_phase_diagram(finalised, FIGS_DIR / "phase_diagram.png")
    if "r2_by_type"         in figures: plot_r2_by_type(finalised,           FIGS_DIR / "r2_by_type.png")
    if "r2_at_ki_bar"       in figures: plot_r2_at_ki_bar(finalised,         FIGS_DIR / "r2_at_ki_bar.png")
    if "phi_capture"        in figures: plot_phi_capture(finalised,          FIGS_DIR / "phi_capture.png")
    if "phi_matrix"         in figures: plot_phi_matrix(finalised,           FIGS_DIR / "phi_matrix.png", k=10)
    if "signed_cohesion"    in figures: plot_signed_cohesion(finalised,      FIGS_DIR / "signed_cohesion.png", k=10)
    if "loss_curves"        in figures: plot_loss_curves(loss_data,          FIGS_DIR / "loss_curves.png")
    if "coordinate_encoding" in figures: plot_coordinate_encoding(finalised, FIGS_DIR / "coordinate_encoding.png")

    if "reconstruction_grid" in figures:
        if args.shape is None:
            print("Warning: --figures reconstruction_grid requires --shape, skipping.")
        else:
            plot_reconstruction_grid(
                args.shape, finalised,
                FIGS_DIR / f"reconstruction_grid_{args.shape}.png",
            )


if __name__ == "__main__":
    main()
