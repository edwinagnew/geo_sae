"""
Run results on *3.b snapshot files and plot Figure 4 panels + loss over time.

Usage:
    python adapt/results/run3b.py
"""
from __future__ import annotations

import importlib.util
import os
import pickle
import sys

import numpy as np


def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RESULTS_PY = os.path.join(os.path.dirname(__file__), "results.py")
results_mod = _load_mod("results", RESULTS_PY)

SNAP_DIR = os.path.dirname(__file__)
FIGS_DIR = os.path.join(SNAP_DIR, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

VARIANTS_3B = {
    "baseline":       "baseline_snapshots3.b",
    "baseline_batch": "baseline_batch_snapshots3.b",
    "signed":         "signed_snapshots3.b",
    #"penal":          "penal_snapshots3.b",
    "penal":          "penal_beta01.b",
}
COLORS = {"baseline": "steelblue", "baseline_batch": "orange", "signed": "green", "penal": "red"}
LABELS = {"baseline": "TopK (baseline)", "baseline_batch": "BatchTopK",
          "signed": "Signed", "penal": "Penalized"}

_KI = {"circle": 2, "sphere": 3, "torus": 4, "mobius": 3,
       "swiss_roll": 3, "helix": 3, "flat_disk": 2, "segment": 1}


def _inst_type(name: str) -> str:
    for t in _KI:
        if name.startswith(t):
            return t
    return name.split("_")[0]


# ── Load ──────────────────────────────────────────────────────────────────────

def load_raw(path: str) -> dict:
    print(f"Loading {os.path.basename(path)} ...", end=" ", flush=True)
    with open(path, "rb") as f:
        d = pickle.load(f)
    print("done.")
    return d


def detect_format(d: dict) -> str:
    """Return 'raw' ({k: snap}), 'finalized' ({panel_a/b/c}), or 'unknown'."""
    keys = list(d.keys())
    if not keys:
        return "unknown"
    if "panel_a" in keys:
        return "finalized"
    first_val = next(iter(d.values()))
    if isinstance(first_val, dict) and ("eval_codes" in first_val or "W_dec" in first_val):
        return "raw"
    return "unknown"


# ── Finalize or pass through ───────────────────────────────────────────────────

def ensure_finalized(v: str, raw: dict) -> tuple[dict, dict | None]:
    """
    Returns (finalized_dict, raw_dict_if_available).
    raw_dict_if_available is needed to extract logs.
    """
    fmt = detect_format(raw)
    print(f"  {v}: format={fmt}, top-level keys={sorted(raw.keys())[:6]}")
    if fmt == "finalized":
        return raw, None
    if fmt == "raw":
        print(f"  {v}: running compute_figure4 ...", flush=True)
        finalized = results_mod.compute_figure4(raw)
        return finalized, raw
    raise ValueError(f"Unknown format for {v}: keys={sorted(raw.keys())[:8]}")


# ── Loss extraction ────────────────────────────────────────────────────────────

def extract_loss_curves(raw_by_k: dict) -> dict[int, dict[str, list]]:
    """Extract loss curves from raw {k: snap} snapshots."""
    curves: dict[int, dict[str, list]] = {}
    for k, snap in sorted(raw_by_k.items()):
        logs = snap.get("logs", [])
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

    # Filter to variants that actually have loss data
    variants_with_loss = {v: lc for v, lc in loss_data.items() if lc}
    if not variants_with_loss:
        print("  No loss data found in any 3.b file.")
        return

    k_vals_per_v = {v: sorted(lc.keys()) for v, lc in variants_with_loss.items()}
    all_k = sorted({k for ks in k_vals_per_v.values() for k in ks})

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
                steps = curve["steps"]
                vals  = curve[metric]
                color = cmap(i / max(len(k_vals) - 1, 1))
                label = f"{LABELS.get(v, v)} k={k}" if mi == 0 else None
                ax.plot(steps, vals, lw=0.8, alpha=0.7, color=color, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("training step")
        ax.set_ylabel(metric)
        if mi == 0:
            ax.legend(fontsize=5, ncol=2, loc="upper right")

    # 4th panel: final l1 loss per variant, k on x-axis
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

    fig.suptitle("Training curves — *3.b batch", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig4a(data: dict[str, dict], out_path: str) -> None:
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
    ax.set_title("Figure 4A (batch 3) — Restricted R² vs sparsity")
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
    fig.suptitle("Figure 4B (batch 3) — Phase diagram", fontsize=10)
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


def plot_r2_by_type(data: dict[str, dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = sorted(_KI.keys())

    # Pick a common reference k: highest aggregate R² for baseline if available
    ref_k = None
    if "baseline" in data:
        pa = data["baseline"]["panel_a"]
        ref_k = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))
    if ref_k is None:
        pa = next(iter(data.values()))["panel_a"]
        ref_k = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
    axes = axes.flatten()
    fig.suptitle(f"R²(n_atoms) at k={ref_k} by manifold type  (batch 3)", fontsize=10)

    for ti, mtype in enumerate(types):
        ax = axes[ti]
        ki = _KI[mtype]
        for v, d in data.items():
            pa = d["panel_a"]
            if ref_k not in pa.get("r2_sweep", {pa["k_values"][0]: {}}):
                # fallback: use first available k
                pass
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
    fig.suptitle("R² at n_atoms = k_i  (batch 3)", fontsize=10)

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
    fig.suptitle("Within-manifold phi coherence (batch 3)  |  positive = capture, negative = tiling",
                 fontsize=10)

    for v, d in data.items():
        pc = d["panel_c"]
        mean_phis = []
        for k in k_vals:
            if k not in pc["phi_matrices"]:
                mean_phis.append(float("nan"))
                continue
            phi    = pc["phi_matrices"][k]
            scores = pc["atom_manifold_scores"][k]
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

    ax_mean.axhline(0, color="black", lw=0.8, ls="--")
    ax_mean.set_xlabel("Training sparsity k")
    ax_mean.set_ylabel("Mean within-manifold phi")
    ax_mean.set_title("Phi coherence vs k  (capture ↑, tiling ↓)")
    ax_mean.legend(fontsize=8)

    for v, d in data.items():
        pc = d["panel_c"]
        if ref_k not in pc["phi_matrices"]:
            continue
        phi    = pc["phi_matrices"][ref_k]
        scores = pc["atom_manifold_scores"][ref_k]
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

    ax_hist.axvline(0, color="black", lw=0.8, ls="--")
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
    fig.suptitle(f"Phi matrix — live atoms only, sorted by manifold  (batch 3, k={k})", fontsize=10)

    im = None
    for ax, (v, d) in zip(axes, plot_variants.items()):
        pc = d["panel_c"]
        if k not in pc["phi_matrices"]:
            ax.set_title(f"{LABELS.get(v, v)}\n(k={k} not found)"); continue

        phi    = pc["phi_matrices"][k]
        scores = pc["atom_manifold_scores"][k]     # (d_sae, n_manifolds)

        # Dead atoms (zero or near-zero score on every manifold) create a large
        # uninformative block — exclude them.  Threshold: top 5% of max per-atom score.
        max_score = scores.max(axis=1)             # (d_sae,)
        threshold = max_score.max() * 0.05
        live      = max_score > threshold          # (d_sae,) bool mask
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
    fig.suptitle("Signed cohesion ρ(G) per manifold  (batch 3; proxy: global marginal phi)", fontsize=10)

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
    ax_box.axhline(+1, color="green", lw=0.6, ls=":", label="capture (+1)")
    ax_box.axhline(-1, color="red",   lw=0.6, ls=":", label="tiling (−1)")
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


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(data: dict[str, dict]) -> None:
    print("\n=== Summary: aggregate R² at sweet-spot k (batch 3) ===")
    for v, d in data.items():
        pa = d["panel_a"]
        sk = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))
        r2  = pa["aggregate_r2"][sk]
        std = pa["aggregate_r2_std"][sk]
        print(f"  {LABELS.get(v, v):22s}  sweet k={sk:>2}  R²={r2:.4f} ± {std:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    finalized: dict[str, dict] = {}
    loss_data: dict[str, dict[int, dict]] = {}

    for v, fname in VARIANTS_3B.items():
        path = os.path.join(SNAP_DIR, fname)
        if not os.path.exists(path):
            print(f"  {v}: {fname} not found, skipping.")
            continue
        raw = load_raw(path)
        fin, raw_for_logs = ensure_finalized(v, raw)
        finalized[v] = fin

        src = raw_for_logs if raw_for_logs is not None else {}
        if src:
            loss_data[v] = extract_loss_curves(src)
        elif "logs" in fin:
            loss_data[v] = extract_loss_curves(fin["logs"])
        else:
            print(f"  {v}: no logs available (pre-fix batch).")
            loss_data[v] = {}

    if not finalized:
        print("No data loaded — nothing to plot.")
        sys.exit(1)

    print_summary(finalized)

    plot_fig4a(finalized,          os.path.join(FIGS_DIR, "fig4a_3b.png"))
    plot_phase_diagram(finalized,  os.path.join(FIGS_DIR, "phase_diagram_3b.png"))
    plot_r2_by_type(finalized,     os.path.join(FIGS_DIR, "r2_by_type_3b.png"))
    plot_r2_at_ki_bar(finalized,   os.path.join(FIGS_DIR, "r2_at_ki_bar_3b.png"))
    plot_phi_capture(finalized,    os.path.join(FIGS_DIR, "phi_capture_3b.png"))
    plot_phi_matrix(finalized,     os.path.join(FIGS_DIR, "phi_matrix_3b.png"), k=10)
    plot_signed_cohesion(finalized, os.path.join(FIGS_DIR, "signed_cohesion_3b.png"), k=10)
    plot_loss_curves(loss_data,    os.path.join(FIGS_DIR, "loss_curves_3b.png"))


if __name__ == "__main__":
    main()
