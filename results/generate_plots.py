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

# ── Import compute module ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results as _results  # noqa: E402

FIGS_DIR = Path("results/figs/") #os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

ALL_FIGURES = [
    "aggregate_r2",
    "phase_diagram",
    "reconstruction_regimes",
    "r2_by_type",
    "r2_at_ki_bar",
    "phi_capture",
    "phi_matrix",
    "signed_cohesion",
    "loss_curves",
    "tuning_curves",
]
EXTRA_FIGURES = ["reconstruction_grid"]  # not in ALL_FIGURES; requires --shape

COLORS = {"baseline": "steelblue", "baseline_batch": "orange", "signed": "green", "penal": "red"}
LABELS = {"baseline": "TopK (baseline)", "baseline_batch": "BatchTopK",
          "signed": "Signed", "penal": "Penalized"}
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

def load_raw(path: str) -> dict:
    print(f"Loading {os.path.basename(path)} ...", end=" ", flush=True)
    with open(path, "rb") as f:
        d = pickle.load(f)
    print("done.")
    return d


def detect_format(d: dict) -> str:
    keys = list(d.keys())
    if not keys:
        return "unknown"
    if "panel_a" in keys:
        return "finalized"
    first_val = next(iter(d.values()))
    if isinstance(first_val, dict) and ("eval_codes" in first_val or "W_dec" in first_val):
        return "raw"
    return "unknown"


def ensure_finalized(v: str, raw: dict) -> tuple[dict, dict | None]:
    fmt = detect_format(raw)
    print(f"  {v}: format={fmt}, top-level keys={sorted(raw.keys())[:6]}")
    if fmt == "finalized":
        return raw, None
    if fmt == "raw":
        print(f"  {v}: running compute_figure4 ...", flush=True)
        finalized = _results.compute_figure4(raw)
        return finalized, raw
    raise ValueError(f"Unknown format for {v}: keys={sorted(raw.keys())[:8]}")


# ── Loss extraction ────────────────────────────────────────────────────────────

def extract_loss_curves(raw_by_k: dict) -> dict[int, dict[str, list]]:
    curves: dict[int, dict[str, list]] = {}
    for k, snap in sorted(raw_by_k.items()):
        # Compressed format (finalized logs): snap is already {metric: np.ndarray}
        if isinstance(snap, dict) and "steps" in snap and isinstance(snap["steps"], np.ndarray):
            curves[k] = {m: snap[m].tolist() for m in ("steps", "l1_loss", "fvu", "n_dead") if m in snap}
            continue
        # Raw format: snap is a snapshot dict or a list of log entries directly
        logs = snap if isinstance(snap, list) else snap.get("logs", [])
        if not logs:
            continue
        steps, l1s, fvus, dead = [], [], [], []
        for entry in logs:
            if "step" not in entry:
                continue
            steps.append(entry["step"])
            l1s.append(entry.get("l1_loss", entry.get("loss", float("nan"))))
            fvus.append(entry.get("fvu", float("nan")))
            dead.append(entry.get("n_dead", float("nan")))
        curves[k] = {"steps": steps, "l1_loss": l1s, "fvu": fvus, "n_dead": dead}
    return curves


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
        pa = d["panel_a"]
        ks   = pa["k_values"]
        r2s  = [pa["aggregate_r2"][k] for k in ks]
        stds = [pa["aggregate_r2_std"][k] for k in ks]
        ax.plot(ks, r2s, "o-", color=COLORS.get(v, "gray"), label=LABELS.get(v, v))
        ax.fill_between(ks,
                        [r - s for r, s in zip(r2s, stds)],
                        [r + s for r, s in zip(r2s, stds)],
                        alpha=0.15, color=COLORS.get(v, "gray"))
    ax.axvline(4, color="gray", ls=":", lw=0.8, label="paper sweet-spot (k=4)")
    ax.set_xlabel("Training sparsity k")
    ax.set_ylabel("Aggregate restricted R²")
    ax.set_title("Figure 4A — Restricted R² vs sparsity")
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
    fig.suptitle("Figure 4B — Phase diagram", fontsize=10)
    for v, d in data.items():
        pb = d["panel_b"]
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


def plot_reconstruction_regimes(data: dict[str, dict], out_path: str) -> None:
    """Fig 4B replica: phase diagram of RF spread vs support size.

    Each (support_size, rf_diameter) pair is one k value; connecting them
    traces the trajectory through Shattering → Capture → Dilution as k grows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ANNOTATE_K = {3, 8, 14, 25}  # k values to label on the baseline curve

    fig, ax = plt.subplots(figsize=(6, 5))

    for v, d in data.items():
        pb  = d["panel_b"]
        ks  = pb["k_values"]
        sup = [pb["mean_support_size"][k] for k in ks]
        rf  = [pb["mean_rf_diameter"][k]  for k in ks]
        color = COLORS.get(v, "gray")
        label = LABELS.get(v, v)
        ax.plot(sup, rf, "o-", color=color, label=label, lw=1.8, ms=5, zorder=3)
        if v == "baseline":
            for k, sx, rx in zip(ks, sup, rf):
                if k in _ANNOTATE_K:
                    ax.annotate(
                        f"k={k}", (sx, rx),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=color,
                    )

    # Regime labels: Shattering=top-left (low k), Dilution=bottom-right (high k),
    # Capture=mid-left inflection (intermediate k)
    ax.text(0.05, 0.97, "Shattering", transform=ax.transAxes,
            fontsize=8, va="top", style="italic", color="#555555")
    ax.text(0.97, 0.08, "Dilution", transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right", style="italic", color="#555555")
    ax.text(0.18, 0.62, "Capture", transform=ax.transAxes,
            fontsize=8, va="bottom", style="italic", color="#555555")

    ax.set_xlabel("Support size (atoms / manifold)", fontsize=10)
    ax.set_ylabel("RF spread (coverage fraction)", fontsize=10)
    ax.set_title("Regimes of manifold reconstruction", fontsize=11)
    ax.legend(fontsize=8, loc="upper center")
    ax.set_ylim(bottom=0)
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
        pa = data["baseline"]["panel_a"]
        ref_k = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))
    if ref_k is None:
        pa = next(iter(data.values()))["panel_a"]
        ref_k = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
    axes = axes.flatten()
    fig.suptitle(f"R²(n_atoms) at k={ref_k} by manifold type", fontsize=10)

    for ti, mtype in enumerate(types):
        ax = axes[ti]
        ki = _KI[mtype]
        for v, d in data.items():
            pa = d["panel_a"]
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
    k_vals = sorted(next(iter(data.values()))["panel_a"]["k_values"])
    x = np.arange(len(types))
    width = 0.8 / len(data)
    offsets = (np.arange(len(data)) - (len(data) - 1) / 2) * width

    fig, axes = plt.subplots(1, len(k_vals), figsize=(2.2 * len(k_vals), 4), sharey=True)
    if len(k_vals) == 1:
        axes = [axes]
    fig.suptitle("R² at n_atoms = k_i", fontsize=10)

    for ki_ax, k in zip(axes, k_vals):
        for i, (v, d) in enumerate(data.items()):
            pa = d["panel_a"]
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

    k_vals = sorted(next(iter(data.values()))["panel_a"]["k_values"])
    sweet_k = {}
    for v, d in data.items():
        pa = d["panel_a"]
        sweet_k[v] = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))
    ref_k = sweet_k.get("baseline", k_vals[len(k_vals) // 2])

    fig, (ax_mean, ax_hist) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Within-manifold phi coherence  |  positive = capture, negative = tiling",
                 fontsize=10)

    for v, d in data.items():
        pc = d["panel_c"]
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

    #ax_mean.axhline(0, color="black", lw=0.8, ls="--")
    ax_mean.set_xlabel("Training sparsity k")
    ax_mean.set_ylabel("Mean within-manifold phi")
    ax_mean.set_title("Phi coherence vs k  (capture ↑, tiling ↓)")
    ax_mean.legend(fontsize=8)

    for v, d in data.items():
        pc = d["panel_c"]
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

    #ax_hist.axvline(0, color="black", lw=0.8, ls="--")
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
        pc = d["panel_c"]
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


def plot_signed_cohesion(data: dict[str, dict], out_path: str, k: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inst_names = next(iter(data.values()))["panel_c"]["instance_names"]
    k_vals = sorted(next(iter(data.values()))["panel_c"]["k_values"])
    L0 = 4
    mean_ki = sum(_KI.values()) / len(_KI)
    tau1_k  = L0 * mean_ki

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
        pc   = d["panel_c"]
        rhos = _get_rhos(pc, k)
        if not rhos:
            continue
        ax_box.boxplot(rhos, positions=[vi], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=COLORS.get(v, "gray"), alpha=0.6),
                       medianprops=dict(color="black", lw=2))
        print(f"  {LABELS.get(v, v):22s}  mean ρ={np.mean(rhos):.3f}  median ρ={np.median(rhos):.3f}")

    ax_box.set_xticks(range(len(data)))
    ax_box.set_xticklabels([LABELS.get(v, v) for v in data], rotation=10, ha="right", fontsize=8)
    ax_box.axhline(0,  color="black", lw=0.8, ls="--")
    #ax_box.axhline(+1, color="green", lw=0.6, ls=":", label="capture (+1)")
    #ax_box.axhline(-1, color="red",   lw=0.6, ls=":", label="tiling (−1)")
    ax_box.set_ylabel("ρ(G)")
    ax_box.set_title(f"Distribution at k={k}")
    ax_box.legend(fontsize=8)

    for v, d in data.items():
        pc = d["panel_c"]
        mean_rhos = [float(np.mean(_get_rhos(pc, kk))) if _get_rhos(pc, kk) else float("nan")
                     for kk in k_vals]
        ax_vs_k.plot(k_vals, mean_rhos, "o-", color=COLORS.get(v, "gray"), label=LABELS.get(v, v))

    ax_vs_k.axhline(0, color="black", lw=0.8, ls="--")
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


def _intrinsic_param(coords: np.ndarray) -> np.ndarray:
    """1D summary of k_i-dimensional intrinsic coordinates."""
    if coords.shape[1] == 1:
        return coords[:, 0]
    return np.arctan2(coords[:, 1], coords[:, 0])


def plot_tuning_curves(data: dict[str, dict], out_path: str) -> None:
    """Tuning curves: SAE feature activation vs. intrinsic manifold coordinate.

    Uses fig4d payload (fixed k_SAE=10). One row per manifold type (first instance
    of each type), one column per variant.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _TYPES = ["circle", "sphere", "torus", "mobius", "swiss_roll", "helix", "flat_disk", "segment"]
    variants   = [v for v in data if data[v].get("fig4d") is not None]
    n_variants = len(variants)

    if n_variants == 0:
        print("  No variants have fig4d data — skipping tuning_curves.")
        return

    fig, axes = plt.subplots(
        len(_TYPES), n_variants,
        figsize=(3.5 * n_variants, 2.8 * len(_TYPES)),
        squeeze=False,
    )
    fig.suptitle("Tuning curves — activation vs. intrinsic coordinate  (k_SAE=10)",
                 fontsize=11, y=1.01)

    for ci, variant in enumerate(variants):
        fig4d = data[variant]["fig4d"]
        W_dec = fig4d["W_dec"]

        inst_by_type: dict[str, dict] = {}
        for inst in fig4d["instances"]:
            t = inst.get("type") or _inst_type(inst["name"])
            if t not in inst_by_type:
                inst_by_type[t] = inst

        for ri, mtype in enumerate(_TYPES):
            ax   = axes[ri, ci]
            inst = inst_by_type.get(mtype)
            if inst is None:
                ax.axis("off")
                continue

            contribs = inst["contribs"]
            codes    = inst["codes"]
            coords   = inst["coords"]
            k_i      = inst["k_i"]

            atoms = _results._greedy_select(contribs, W_dec, k_i)
            if not atoms:
                ax.axis("off")
                continue

            param = _intrinsic_param(coords)
            order = np.argsort(param)

            for ai, atom in enumerate(atoms):
                act   = codes[:, atom]
                color = ATOM_COLORS[ai % len(ATOM_COLORS)]
                ax.scatter(param[order], act[order],
                           s=1.5, alpha=0.4, color=color, rasterized=True, linewidths=0)

            ax.axhline(0, color="black", lw=0.5, alpha=0.5)
            ax.tick_params(labelsize=6)
            ax.set_xlim(float(param.min()), float(param.max()))
            if ri == 0:
                ax.set_title(LABELS.get(variant, variant), fontsize=9)
            if ci == 0:
                ax.set_ylabel(mtype, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_reconstruction_grid_png(shape_name: str, data: dict[str, dict], out_path: str) -> None:
    """Thin wrapper around explore.plot_reconstruction_grid for static PNG output.

    Not in ALL_FIGURES — invoke with --figures reconstruction_grid --shape <name>.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "explore",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.plot_reconstruction_grid(shape_name, data, interactive=False, out_path=out_path)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(data: dict[str, dict]) -> None:
    print("\n=== Summary: aggregate R² at sweet-spot k ===")
    for v, d in data.items():
        pa = d["panel_a"]
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

    finalized: dict[str, dict] = {}
    loss_data: dict[str, dict[int, dict]] = {}

    for v, path in variant_paths.items():
        if not os.path.exists(path):
            print(f"  {v}: {path} not found, skipping.")
            continue
        raw = load_raw(path)
        fin, raw_for_logs = ensure_finalized(v, raw)
        finalized[v] = fin

        if raw_for_logs is not None:
            loss_data[v] = extract_loss_curves(raw_for_logs)
        elif "logs" in fin:
            loss_data[v] = extract_loss_curves(fin["logs"])
        else:
            loss_data[v] = {}

    if not finalized:
        print("No data loaded — nothing to plot.")
        sys.exit(1)

    print_summary(finalized)

    def fig_path(name: str) -> str:
        return os.path.join(FIGS_DIR, f"{name}.png")

    if "aggregate_r2"  in figures: plot_aggregate_r2(finalized,              fig_path("aggregate_r2"))
    if "phase_diagram" in figures: plot_phase_diagram(finalized,              fig_path("phase_diagram"))
    if "reconstruction_regimes" in figures: plot_reconstruction_regimes(finalized, fig_path("reconstruction_regimes"))
    if "r2_by_type"    in figures: plot_r2_by_type(finalized,           fig_path("r2_by_type"))
    if "r2_at_ki_bar"  in figures: plot_r2_at_ki_bar(finalized,         fig_path("r2_at_ki_bar"))
    if "phi_capture"   in figures: plot_phi_capture(finalized,           fig_path("phi_capture"))
    if "phi_matrix"    in figures: plot_phi_matrix(finalized,            fig_path("phi_matrix"), k=10)
    if "signed_cohesion" in figures: plot_signed_cohesion(finalized,    fig_path("signed_cohesion"), k=10)
    if "loss_curves"     in figures: plot_loss_curves(loss_data,         fig_path("loss_curves"))
    if "tuning_curves"   in figures: plot_tuning_curves(finalized,       fig_path("tuning_curves"))

    if "reconstruction_grid" in figures:
        if args.shape is None:
            print("Warning: --figures reconstruction_grid requires --shape, skipping.")
        else:
            plot_reconstruction_grid_png(
                args.shape, finalized,
                fig_path(f"reconstruction_grid_{args.shape}"),
            )


if __name__ == "__main__":
    main()
