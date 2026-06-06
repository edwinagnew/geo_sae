"""
Exploration and diagnostic script for synthetic SAE experiment results.
Add new investigation code here; don't create separate files.
"""
import pickle
import importlib.util
import os
import sys
import argparse
import numpy as np

# ── Lazy imports ───────────────────────────────────────────────────────────────

def _load_results_mod():
    spec = importlib.util.spec_from_file_location(
        "results", os.path.join(os.path.dirname(__file__), "results.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_snap(name: str, suffix: str = "") -> dict:
    path = f"adapt/results/{name}_snapshots{suffix}.b"
    print(f"Loading {path} ...", end=" ", flush=True)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("done.")
    return data


# ── Format inspection ──────────────────────────────────────────────────────────

def cmd_format(args):
    """Print format summary for all snapshot files."""
    for name in ["baseline_batch", "signed", "penal"]:
        try:
            d = _load_snap(name)
        except FileNotFoundError:
            print(f"{name}: file not found")
            continue
        keys = list(d.keys())
        first_val = next(iter(d.values()))
        print(f"{name}: top-level keys={keys[:8]}")
        if isinstance(first_val, dict):
            print(f"  first entry keys={list(first_val.keys())[:8]}")
            if "eval_codes" in first_val:
                print(f"  eval_codes.shape={first_val['eval_codes'].shape}")
            if "instance_names" in first_val:
                print(f"  n_instances={len(first_val['instance_names'])}")


# ── R² comparison ─────────────────────────────────────────────────────────────

def cmd_compare_r2(args):
    """Compare aggregate R² between old baseline and recomputed baseline_batch."""
    mod = _load_results_mod()

    with open("adapt/results/baseline_snapshots.b", "rb") as f:
        old = pickle.load(f)
    pa = old["panel_a"]
    print("=== baseline (old, TopK per-sample) ===")
    for k in pa["k_values"]:
        print(f"  k={k:>2}  R²={pa['aggregate_r2'][k]:.4f} ± {pa['aggregate_r2_std'][k]:.4f}")

    raw = _load_snap("baseline_batch")
    # Check if raw is already finalized or is {k: snap}
    first_key = next(iter(raw.keys()))
    if isinstance(first_key, int):
        print("\n=== baseline_batch (BatchTopK, greedy OMP) ===")
        pa2 = mod.compute_panel_a(raw)
        for k in pa2["k_values"]:
            print(f"  k={k:>2}  R²={pa2['aggregate_r2'][k]:.4f} ± {pa2['aggregate_r2_std'][k]:.4f}")
    else:
        print(f"\nbaseline_batch already finalized. Keys: {list(raw.keys())}")
        pa2 = raw.get("panel_a", raw)
        if "aggregate_r2" in pa2:
            for k in pa2["k_values"]:
                print(f"  k={k:>2}  R²={pa2['aggregate_r2'][k]:.4f}")


# ── Coordinate alignment ───────────────────────────────────────────────────────

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den = np.sqrt(((x - mx) ** 2).sum() * ((y - my) ** 2).sum())
    return float(num / (den + 1e-12))


def compute_alignment(snap: dict, greedy_select, min_active: int = 50) -> dict:
    """
    For each manifold instance, select k_i features by greedy OMP, then compute
    Pearson r between each feature activation and each ground-truth intrinsic coord.

    Returns dict[name] -> {atom_ids, r_matrix (k_i x k_i), alignment (k_i,),
                           mean_alignment, instance_type, k_i, n_active}
    """
    codes        = snap["eval_codes"]
    W_dec        = snap["W_dec"]
    active_mask  = snap["active_mask"]
    contribs     = snap["instance_contribs"]
    coords_list  = snap["instance_coords"]

    out = {}
    for j, name in enumerate(snap["instance_names"]):
        act_j = active_mask[:, j]
        n_j = int(act_j.sum())
        if n_j < min_active:
            continue
        m_j    = contribs[j]
        z_j    = codes[act_j]
        coords = coords_list[j]
        k_i    = snap["instance_ki"][j]

        atoms = greedy_select(m_j, W_dec, k_i)
        k_eff = min(k_i, len(atoms), coords.shape[1])

        r_mat = np.zeros((k_eff, k_eff), dtype=np.float32)
        for ai, atom in enumerate(atoms[:k_eff]):
            for ci in range(k_eff):
                r_mat[ai, ci] = _pearson_r(z_j[:, atom], coords[:, ci])

        alignment = np.abs(r_mat).max(axis=1)
        out[name] = {
            "atom_ids":       atoms[:k_eff],
            "r_matrix":       r_mat,
            "alignment":      alignment,
            "mean_alignment": float(alignment.mean()),
            "instance_type":  snap["instance_types"][j],
            "k_i":            k_i,
            "n_active":       n_j,
            "z_j":            z_j,
            "coords":         coords,
        }
    return out


def cmd_coord_interp(args):
    """Coordinate alignment + tuning curve plots for one or more variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mod = _load_results_mod()
    greedy = mod._greedy_select_np

    results_by_variant = {}
    for name in args.file:
        raw = _load_snap(name)
        first_key = next(iter(raw.keys()))
        if not isinstance(first_key, int):
            print(f"  {name}: not raw snapshot format (keys={list(raw.keys())[:4]})")
            continue
        if args.k not in raw:
            print(f"  {name}: k={args.k} not found. Available: {sorted(raw.keys())}")
            continue
        snap = raw[args.k]
        print(f"\nAlignment analysis: {name}, k={args.k}")
        alignment = compute_alignment(snap, greedy)
        results_by_variant[name] = (snap, alignment)

        # Print by type
        by_type: dict[str, list] = {}
        for res in alignment.values():
            by_type.setdefault(res["instance_type"], []).append(res["mean_alignment"])
        print("  Type              |r|")
        for t, vals in sorted(by_type.items()):
            print(f"  {t:17s} {np.mean(vals):.3f}")
        overall = np.mean([r["mean_alignment"] for r in alignment.values()])
        print(f"  Overall:          {overall:.3f}")

        # Tuning curves plot
        _plot_tuning_curves(snap, alignment, name, args.k)

    if len(results_by_variant) > 1:
        _plot_alignment_summary(results_by_variant)


def _plot_tuning_curves(snap, alignment, variant_label, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One instance per manifold type
    seen: set = set()
    to_show = []
    for name, res in alignment.items():
        t = res["instance_type"]
        if t not in seen:
            to_show.append(name)
            seen.add(t)
        if len(to_show) >= 6:
            break

    max_ki = max(alignment[n]["k_i"] for n in to_show)
    fig, axes = plt.subplots(len(to_show), max_ki,
                             figsize=(2.5 * max_ki, 2.5 * len(to_show)),
                             squeeze=False)
    fig.suptitle(f"{variant_label}  k={k}  —  Feature activation vs ground-truth coordinate",
                 fontsize=10, y=1.01)

    for row, name in enumerate(to_show):
        res    = alignment[name]
        z_j    = res["z_j"]
        coords = res["coords"]
        atoms  = res["atom_ids"]
        r_mat  = res["r_matrix"]
        k_i    = res["k_i"]

        for col in range(max_ki):
            ax = axes[row, col]
            if col >= k_i:
                ax.axis("off")
                continue
            atom   = atoms[col]
            best_c = int(np.abs(r_mat[col]).argmax())
            feat   = z_j[:, atom]
            coord  = coords[:, best_c]
            r_val  = r_mat[col, best_c]

            idx = np.random.default_rng(42).choice(len(feat), min(2000, len(feat)), replace=False)
            ax.scatter(coord[idx], feat[idx], s=2, alpha=0.3, rasterized=True)
            ax.set_title(f"r={r_val:+.2f}", fontsize=8)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"{res['instance_type']}\nactivation", fontsize=7)
            ax.set_xlabel(f"coord {best_c}", fontsize=7)

    plt.tight_layout()
    out = f"adapt/results/tuning_curves_{variant_label}_k{k}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _plot_alignment_summary(results_by_variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(results_by_variant.keys())
    all_types = sorted({r["instance_type"]
                        for _, alignment in results_by_variant.values()
                        for r in alignment.values()})
    x = np.arange(len(all_types))
    width = 0.8 / len(variants)
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (variant, (_, alignment)) in enumerate(results_by_variant.items()):
        by_type = {t: [] for t in all_types}
        for res in alignment.values():
            by_type[res["instance_type"]].append(res["mean_alignment"])
        scores = [np.mean(by_type[t]) if by_type[t] else float("nan") for t in all_types]
        ax.bar(x + offsets[i], scores, width * 0.9, label=variant, alpha=0.85)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="ReLU half-rectified ceiling (~0.5)")
    ax.axhline(1.0, color="black", ls=":", lw=0.8, label="Linear ideal (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=20, ha="right")
    ax.set_ylabel("Mean |r| with best-matching coordinate")
    ax.set_title("Coordinate alignment by manifold type and SAE variant")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = "adapt/results/alignment_summary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 4A replication plot ─────────────────────────────────────────────────

def cmd_fig4a(args):
    """Plot aggregate R² vs k for baseline (old TopK) and overlay available variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))

    # Old baseline
    with open("adapt/results/baseline_snapshots.b", "rb") as f:
        old = pickle.load(f)
    pa = old["panel_a"]
    ks = pa["k_values"]
    r2s = [pa["aggregate_r2"][k] for k in ks]
    stds = [pa["aggregate_r2_std"][k] for k in ks]
    ax.plot(ks, r2s, "o-", label="baseline (TopK)", color="steelblue")
    ax.fill_between(ks,
                    [r - s for r, s in zip(r2s, stds)],
                    [r + s for r, s in zip(r2s, stds)],
                    alpha=0.2, color="steelblue")

    mod = _load_results_mod()
    colors = {"baseline_batch": "orange", "signed": "green", "penal": "red"}
    for name in args.file:
        try:
            raw = _load_snap(name)
        except FileNotFoundError:
            continue
        first_key = next(iter(raw.keys()))
        if isinstance(first_key, int):
            pa_v = mod.compute_panel_a(raw)
        elif "panel_a" in raw:
            pa_v = raw["panel_a"]
        else:
            print(f"  {name}: unrecognized format")
            continue
        ks_v = pa_v["k_values"]
        r2s_v = [pa_v["aggregate_r2"][k] for k in ks_v]
        stds_v = [pa_v["aggregate_r2_std"][k] for k in ks_v]
        ax.plot(ks_v, r2s_v, "o-", label=name, color=colors.get(name, "gray"))
        ax.fill_between(ks_v,
                        [r - s for r, s in zip(r2s_v, stds_v)],
                        [r + s for r, s in zip(r2s_v, stds_v)],
                        alpha=0.2, color=colors.get(name, "gray"))

    ax.set_xlabel("Training sparsity k")
    ax.set_ylabel("Aggregate restricted R²")
    ax.set_title("Figure 4A — Restricted R² vs sparsity (synthetic manifold zoo)")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    out = "adapt/results/figs/fig4a.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Type helpers ──────────────────────────────────────────────────────────────

_KI = {"circle": 2, "sphere": 3, "torus": 4, "mobius": 3,
       "swiss_roll": 3, "helix": 3, "flat_disk": 2, "segment": 1}

def _inst_type(name: str) -> str:
    for t in _KI:
        if name.startswith(t):
            return t
    return name.split("_")[0]

def _inst_ki(name: str) -> int:
    return _KI.get(_inst_type(name), 2)


# ── Full analysis (all plots) ──────────────────────────────────────────────────

def cmd_analysis(args):
    """
    Produce 4 plots from finalized .2.b state:
      1. fig4a_all.png         — Aggregate R² vs k, all variants
      2. r2_sweep_by_type.png  — R²(n_atoms) at sweet-spot k, per manifold type
      3. r2_at_ki_bar.png      — R² at exactly k_i atoms, by type & variant
      4. phi_capture.png       — Within-manifold phi coherence vs k (capture regime detector)
      5. phase_diagram.png     — Panel B: support size & RF diameter vs k
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SUFFIX = "2"
    VARIANTS = ["baseline", "baseline_batch", "signed", "penal"]
    COLORS   = {"baseline": "steelblue", "baseline_batch": "orange",
                "signed": "green", "penal": "red"}
    LABELS   = {"baseline": "TopK (baseline)", "baseline_batch": "BatchTopK",
                "signed": "Signed", "penal": "Penalized"}

    # Load all
    data = {}
    for v in VARIANTS:
        try:
            data[v] = _load_snap(v, suffix=SUFFIX)
        except FileNotFoundError:
            print(f"  {v}2 not found, skipping")

    k_vals = sorted(next(iter(data.values()))["panel_a"]["k_values"])

    # ── Find sweet-spot k for each variant ────────────────────────────────────
    sweet_k = {}
    for v, d in data.items():
        pa = d["panel_a"]
        sweet_k[v] = max(pa["k_values"], key=lambda k: pa["aggregate_r2"].get(k, -9))
    print("Sweet-spot k per variant:", sweet_k)

    # ── 1. Figure 4A: Aggregate R² vs k ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for v, d in data.items():
        pa = d["panel_a"]
        ks   = pa["k_values"]
        r2s  = [pa["aggregate_r2"][k] for k in ks]
        stds = [pa["aggregate_r2_std"][k] for k in ks]
        ax.plot(ks, r2s, "o-", color=COLORS[v], label=LABELS[v])
        ax.fill_between(ks,
                        [r - s for r, s in zip(r2s, stds)],
                        [r + s for r, s in zip(r2s, stds)],
                        alpha=0.15, color=COLORS[v])
    ax.axvline(4, color="gray", ls=":", lw=0.8, label="paper sweet-spot (k=4)")
    ax.set_xlabel("Training sparsity k")
    ax.set_ylabel("Aggregate restricted R²")
    ax.set_title("Figure 4A — Restricted R² vs sparsity")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.4)
    plt.tight_layout()
    plt.savefig("adapt/results/figs/fig4a_all.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: adapt/results/fig4a_all.png")

    # ── 2. R² sweep by manifold type at sweet-spot k ─────────────────────────
    # Use baseline's sweet-spot as the common reference k
    ref_k = sweet_k.get("baseline", 8)
    types = sorted(_KI.keys())
    n_types = len(types)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
    axes = axes.flatten()
    fig.suptitle(f"R²(n_atoms) at k={ref_k} by manifold type  —  key: does variant reach ~1 at n_atoms=k_i?",
                 fontsize=10)

    for ti, mtype in enumerate(types):
        ax = axes[ti]
        ki = _KI[mtype]
        for v, d in data.items():
            pa = d["panel_a"]
            if ref_k not in pa["aggregate_r2"]:
                continue
            # Average r2_sweep across all instances of this type
            sweeps = []
            for inst_name, by_k in pa["r2_sweep"].items():
                if _inst_type(inst_name) == mtype and ref_k in by_k:
                    sweeps.append(by_k[ref_k])
            if not sweeps:
                continue
            mean_sweep = np.mean(sweeps, axis=0)   # (25,)
            n_atoms = list(range(1, len(mean_sweep) + 1))
            ax.plot(n_atoms, mean_sweep, color=COLORS[v], label=LABELS[v], lw=1.5)

        ax.axvline(ki, color="black", ls="--", lw=0.8, alpha=0.6, label=f"k_i={ki}")
        ax.axhline(1.0, color="gray", ls=":", lw=0.6)
        ax.set_title(f"{mtype}  (k_i={ki})", fontsize=9)
        ax.set_xlabel("n_atoms", fontsize=8)
        ax.set_ylabel("R²", fontsize=8)
        ax.set_xlim(1, 12)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=6, loc="lower right")
    plt.tight_layout()
    plt.savefig("adapt/results/figs/r2_sweep_by_type.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: adapt/results/r2_sweep_by_type.png")

    # ── 3. R² at exactly k_i atoms, bar chart ────────────────────────────────
    fig, axes = plt.subplots(1, len(k_vals), figsize=(2.2 * len(k_vals), 4), sharey=True)
    fig.suptitle("R² at n_atoms = k_i  (coordinate efficiency: higher = more coordinate-like)",
                 fontsize=10)
    x = np.arange(len(types))
    width = 0.8 / len(data)
    offsets = (np.arange(len(data)) - (len(data) - 1) / 2) * width

    for ki_ax, k in zip(axes, k_vals):
        for i, (v, d) in enumerate(data.items()):
            pa = d["panel_a"]
            if k not in pa["aggregate_r2"]:
                ki_ax.bar(x + offsets[i], [float("nan")] * len(types),
                          width * 0.9, color=COLORS[v], alpha=0.8)
                continue
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
                      color=COLORS[v], alpha=0.8, label=LABELS[v] if k == k_vals[0] else "")
        ki_ax.set_title(f"k={k}", fontsize=8)
        ki_ax.set_xticks(x)
        ki_ax.set_xticklabels([t[:5] for t in types], rotation=35, ha="right", fontsize=7)
        ki_ax.axhline(0, color="black", lw=0.4)
        ki_ax.set_ylim(-0.1, 1.05)

    axes[0].set_ylabel("R² at n_atoms=k_i")
    axes[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("adapt/results/figs/r2_at_ki_bar.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: adapt/results/r2_at_ki_bar.png")

    # ── 4. Within-manifold phi coherence ─────────────────────────────────────
    # For each variant & k: assign top k_i atoms per manifold, compute mean within-manifold phi
    fig, (ax_mean, ax_hist) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Within-manifold phi coherence  |  positive = capture/coordinate, negative = tiling",
                 fontsize=10)

    ref_k_phi = sweet_k.get("baseline", 8)
    for v, d in data.items():
        pc = d["panel_c"]
        mean_phis = []
        for k in k_vals:
            if k not in pc["phi_matrices"]:
                mean_phis.append(float("nan"))
                continue
            phi = pc["phi_matrices"][k]            # (512, 512)
            scores = pc["atom_manifold_scores"][k] # (512, 48)
            inst_names = pc["instance_names"]

            within_vals = []
            for j, inst_name in enumerate(inst_names):
                ki = _inst_ki(inst_name)
                top_atoms = np.argsort(scores[:, j])[-ki:]
                if len(top_atoms) < 2:
                    continue
                sub = phi[np.ix_(top_atoms, top_atoms)]
                # Off-diagonal elements only
                mask = ~np.eye(len(top_atoms), dtype=bool)
                within_vals.extend(sub[mask].tolist())

            mean_phis.append(float(np.mean(within_vals)) if within_vals else float("nan"))

        ax_mean.plot(k_vals, mean_phis, "o-", color=COLORS[v], label=LABELS[v])

    ax_mean.axhline(0, color="black", lw=0.8, ls="--")
    ax_mean.set_xlabel("Training sparsity k")
    ax_mean.set_ylabel("Mean within-manifold phi")
    ax_mean.set_title("Phi coherence vs k  (capture ↑, tiling ↓)")
    ax_mean.legend(fontsize=8)

    # Histogram at reference k
    for v, d in data.items():
        pc = d["panel_c"]
        if ref_k_phi not in pc["phi_matrices"]:
            continue
        phi = pc["phi_matrices"][ref_k_phi]
        scores = pc["atom_manifold_scores"][ref_k_phi]
        inst_names = pc["instance_names"]
        within_vals = []
        for j, inst_name in enumerate(inst_names):
            ki = _inst_ki(inst_name)
            top_atoms = np.argsort(scores[:, j])[-ki:]
            if len(top_atoms) < 2:
                continue
            sub = phi[np.ix_(top_atoms, top_atoms)]
            mask = ~np.eye(len(top_atoms), dtype=bool)
            within_vals.extend(sub[mask].tolist())
        ax_hist.hist(within_vals, bins=40, alpha=0.5, color=COLORS[v],
                     label=LABELS[v], density=True)

    ax_hist.axvline(0, color="black", lw=0.8, ls="--")
    ax_hist.set_xlabel("Within-manifold phi (pairwise)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Distribution of within-manifold phi at k={ref_k_phi}")
    ax_hist.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("adapt/results/figs/phi_capture.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: adapt/results/phi_capture.png")

    # ── 5. Phase diagram ─────────────────────────────────────────────────────
    fig, (ax_sup, ax_rf) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Figure 4B — Phase diagram", fontsize=10)
    for v, d in data.items():
        pb = d["panel_b"]
        ks = pb["k_values"]
        sup  = [pb["mean_support_size"][k] for k in ks]
        rf   = [pb["mean_rf_diameter"][k] for k in ks]
        ax_sup.plot(ks, sup, "o-", color=COLORS[v], label=LABELS[v])
        ax_rf.plot(ks, rf,  "o-", color=COLORS[v], label=LABELS[v])
    ax_sup.set_xlabel("k"); ax_sup.set_ylabel("Mean support size")
    ax_sup.set_title("Support size vs k"); ax_sup.legend(fontsize=8)
    ax_rf.set_xlabel("k"); ax_rf.set_ylabel("Mean RF diameter (coverage)")
    ax_rf.set_title("RF diameter vs k"); ax_rf.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("adapt/results/figs/phase_diagram.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: adapt/results/phase_diagram.png")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n=== Summary: aggregate R² at sweet-spot k ===")
    for v, d in data.items():
        pa = d["panel_a"]
        sk = sweet_k[v]
        r2 = pa["aggregate_r2"][sk]
        std = pa["aggregate_r2_std"][sk]
        print(f"  {LABELS[v]:20s}  sweet k={sk}  R²={r2:.4f} ± {std:.4f}")


# ── Ordered phi matrix (Figure 4C style) ──────────────────────────────────────

def cmd_phi_matrix(args):
    """
    Plot the 512×512 phi matrix reordered by dominant manifold per atom.
    Capture regime → clean block-diagonal (all atoms for same manifold co-fire).
    Tiling regime  → anti-diagonal blocks (atoms tile non-overlapping patches).
    Dilution regime → messy mixed structure.

    Also plots signed cohesion ρ(G) (Definition 6, Appendix G) per manifold
    per variant, and its distribution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SUFFIX = "2"
    VARIANTS = ["baseline", "signed", "penal"]
    COLORS   = {"baseline": "steelblue", "signed": "green", "penal": "red"}
    LABELS   = {"baseline": "TopK (baseline)", "signed": "Signed", "penal": "Penalized"}

    data = {}
    for v in VARIANTS:
        try:
            data[v] = _load_snap(v, suffix=SUFFIX)
        except FileNotFoundError:
            print(f"  {v}2 not found"); continue

    # Use sweet-spot k=10
    k = args.k
    inst_names = next(iter(data.values()))["panel_c"]["instance_names"]
    n_inst = len(inst_names)

    # ── 1. Ordered phi matrix side-by-side ────────────────────────────────────
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 5))
    if len(data) == 1:
        axes = [axes]
    fig.suptitle(f"Phi matrix ordered by dominant manifold  (k={k})\n"
                 "Capture → block-diagonal  |  Tiling → anti-blocks  |  Dilution → messy",
                 fontsize=10)

    for ax, (v, d) in zip(axes, data.items()):
        pc = d["panel_c"]
        if k not in pc["phi_matrices"]:
            ax.set_title(f"{LABELS[v]}\n(k={k} not found)"); continue

        phi    = pc["phi_matrices"][k]          # (512, 512)
        scores = pc["atom_manifold_scores"][k]  # (512, 48)

        # Correct dominant manifold per atom: argmax over manifolds (axis=1)
        dominant = scores.argmax(axis=1)        # (512,) — manifold index for each atom
        order    = np.argsort(dominant, kind="stable")

        phi_ord = phi[np.ix_(order, order)]

        im = ax.imshow(phi_ord, vmin=-1, vmax=1, cmap="RdBu_r",
                       interpolation="nearest", aspect="auto")
        ax.set_title(f"{LABELS[v]}", fontsize=9)
        ax.set_xlabel("atom (sorted by manifold)"); ax.set_ylabel("atom")

        # Draw manifold boundary lines
        boundaries = np.where(np.diff(dominant[order]))[0] + 1
        for b in boundaries:
            ax.axhline(b - 0.5, color="black", lw=0.3, alpha=0.4)
            ax.axvline(b - 0.5, color="black", lw=0.3, alpha=0.4)

    fig.colorbar(im, ax=axes[-1], fraction=0.03, label="phi")
    plt.tight_layout()
    out = f"adapt/results/figs/phi_matrix_k{k}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── 2. Signed cohesion ρ(G) per manifold per variant ─────────────────────
    # Definition 6: ρ(G) = mean sign(J_ab) for a,b in G, a<b
    # Proxy: global marginal phi (all N samples, 1[z≠0] binarisation).
    # Conditioning on manifold-active samples breaks signed SAEs (p≈1 → phi undefined).
    fig, (ax_box, ax_vs_k) = plt.subplots(1, 2, figsize=(12, 4))

    def _get_rhos_at_k(pc, kk):
        if kk not in pc["phi_matrices"]:
            return []
        phi    = pc["phi_matrices"][kk]
        scores = pc["atom_manifold_scores"][kk]
        rhos   = []
        for name in inst_names:
            ki = _inst_ki(name)
            top_atoms = np.argsort(scores[:, inst_names.index(name)])[-ki:]
            if len(top_atoms) < 2:
                continue
            sub  = phi[np.ix_(top_atoms, top_atoms)]
            mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
            rhos.append(float(np.mean(np.sign(sub[mask]))))
        return rhos

    fig.suptitle("Signed cohesion ρ(G) per manifold  (Defn 6; proxy: global marginal phi)",
                 fontsize=10)

    for v, d in data.items():
        pc   = d["panel_c"]
        rhos = _get_rhos_at_k(pc, k)
        if not rhos:
            continue
        ax_box.boxplot(rhos, positions=[list(data.keys()).index(v)],
                       widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=COLORS[v], alpha=0.6),
                       medianprops=dict(color="black", lw=2))
        print(f"  {LABELS[v]:20s}  mean ρ={np.mean(rhos):.3f}  median ρ={np.median(rhos):.3f}")

    ax_box.set_xticks(range(len(data)))
    ax_box.set_xticklabels([LABELS[v] for v in data])
    ax_box.axhline(0,  color="black", lw=0.8, ls="--")
    ax_box.axhline(+1, color="green", lw=0.6, ls=":", label="capture (+1)")
    ax_box.axhline(-1, color="red",   lw=0.6, ls=":", label="tiling (−1)")
    ax_box.set_ylabel("ρ(G)")
    ax_box.set_title(f"Distribution of ρ(G) at k={k}")
    ax_box.legend(fontsize=8)

    # ρ vs k
    k_vals = sorted(next(iter(data.values()))["panel_c"]["k_values"])
    for v, d in data.items():
        pc = d["panel_c"]
        mean_rhos_k = []
        for kk in k_vals:
            rhos_k = _get_rhos_at_k(pc, kk)
            mean_rhos_k.append(float(np.mean(rhos_k)) if rhos_k else float("nan"))
        ax_vs_k.plot(k_vals, mean_rhos_k, "o-", color=COLORS[v], label=LABELS[v])

    # τ = k / (L0 × mean_k_i) = k / 10.5; paper: shattering τ<1, capture τ≈1, dilution τ>1
    L0 = 4
    mean_ki = sum(_KI.values()) / len(_KI)   # 2.625
    tau1_k  = L0 * mean_ki                    # ≈ 10.5

    ax_vs_k.axhline(0, color="black", lw=0.8, ls="--")
    ax_vs_k.axvline(tau1_k, color="orange", lw=1.2, ls="--",
                    label=f"τ=1  (k={tau1_k:.1f})")
    ax_vs_k.text(tau1_k + 0.3, -0.85, "τ=1", color="orange", fontsize=8)
    ax_vs_k.text(tau1_k * 0.35, -0.85, "shattering", color="gray", fontsize=7, ha="center")
    ax_vs_k.text(tau1_k * 1.7, -0.85, "dilution", color="gray", fontsize=7, ha="center")
    ax_vs_k.set_xlabel("Training sparsity k")
    ax_vs_k.set_ylabel("Mean ρ(G) across manifolds")
    ax_vs_k.set_title("Signed cohesion vs k  (τ = k / (L0·mean k_i))")
    ax_vs_k.legend(fontsize=8)

    plt.tight_layout()
    out2 = f"adapt/results/figs/signed_cohesion_k{k}.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


# ── Inspect new .2.b files ─────────────────────────────────────────────────────

def cmd_inspect2(args):
    """Print keys and shapes for all *2.b snapshot files."""
    for name in ["baseline", "baseline_batch", "signed", "penal"]:
        try:
            d = _load_snap(name, suffix="2")
        except FileNotFoundError:
            print(f"  {name}2: not found"); continue
        print(f"\n{name}2 — top-level keys: {sorted(d.keys())}")
        for top_key, val in d.items():
            if isinstance(val, dict):
                print(f"  [{top_key}] keys: {sorted(val.keys())}")
                for k2, v2 in val.items():
                    if isinstance(v2, np.ndarray):
                        print(f"    {k2}: ndarray {v2.shape} {v2.dtype}")
                    elif isinstance(v2, dict):
                        # e.g. r2_sweep or aggregate_r2 keyed by int
                        inner_keys = sorted(v2.keys())[:4]
                        first_val2 = next(iter(v2.values()))
                        print(f"    {k2}: dict keys={inner_keys}... first_val type={type(first_val2).__name__}")
                        if isinstance(first_val2, np.ndarray):
                            print(f"      -> shape {first_val2.shape}")
                    elif isinstance(v2, list):
                        print(f"    {k2}: list len={len(v2)}")
                    else:
                        print(f"    {k2}: {type(v2).__name__} = {str(v2)[:60]}")
            else:
                print(f"  {top_key}: {type(val).__name__}")


# ── Dispatch ───────────────────────────────────────────────────────────────────

COMMANDS = {
    "format":       cmd_format,
    "compare_r2":   cmd_compare_r2,
    "coord_interp": cmd_coord_interp,
    "fig4a":        cmd_fig4a,
    "inspect2":     cmd_inspect2,
    "analysis":     cmd_analysis,
    "phi_matrix":   cmd_phi_matrix,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=COMMANDS)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--file", nargs="+", default=["baseline_batch"])
    args = p.parse_args()
    COMMANDS[args.cmd](args)
