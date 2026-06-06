"""
Interactive and diagnostic visualizations for geo_sae experiments.

Usage (reconstruction grid):
    python results/explore.py <shape_name> <snapshot_files...> [--interactive] [--out PATH]

Examples:
    python results/explore.py helix results/snapshots/*4.b
    python results/explore.py circle results/snapshots/baseline_snapshots4.b --interactive
    python results/explore.py torus results/snapshots/*4.b --out results/figs/torus_grid.png
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ATOM_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb",
]
LABELS = {
    "baseline":       "TopK",
    "baseline_batch": "BatchTopK",
    "signed":         "Signed",
    "penal":          "Penalized",
}


def _load_all_shapes() -> list[str]:
    """Generate all 48 PAPER_CONFIG instance names from data.py."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from data import _MANIFOLD_TYPES
    names = []
    for type_name, type_spec in _MANIFOLD_TYPES.items():
        for params in type_spec["paper"]:
            param_str = "_".join(f"{k}{v:.3g}" for k, v in sorted(params.items()))
            names.append(f"{type_name}_{param_str}")
    return names


ALL_SHAPES: list[str] = _load_all_shapes()


def _find_instance(fig4d: dict, shape_name: str) -> dict | None:
    """Return the first instance whose name matches or contains shape_name."""
    for inst in fig4d["instances"]:
        if inst["name"] == shape_name or shape_name in inst["name"]:
            return inst
    return None


def _prepare_panels(inst: dict, W_dec: np.ndarray, n_pca: int) -> dict | None:
    """Per-sample top-n reconstruction (Fig 5 style: piecewise-linear tiling).

    For each column n, every sample keeps its n most-active atoms. The union
    of patches progressively covers the full manifold — matching the paper's
    visualization approach, not OMP-selected fixed atoms.
    """
    contribs = inst["contribs"]   # (n_j, d)
    codes    = inst["codes"]      # (n_j, d_sae)
    k_i      = inst["k_i"]
    n_j      = len(codes)

    if n_j == 0:
        return None

    pca         = PCA(n_components=n_pca)
    coords_true = pca.fit_transform(contribs)

    # Sort atoms by activation magnitude per sample (descending)
    order = np.argsort(-np.abs(codes), axis=1)  # (n_j, d_sae)

    # Color each point by its most-active atom, consistently across all panels
    dominant_atoms = order[:, 0]                                          # (n_j,)
    unique, counts = np.unique(dominant_atoms, return_counts=True)
    top_atoms      = unique[np.argsort(-counts)[:len(ATOM_COLORS) - 1]]
    color_map      = {int(a): i for i, a in enumerate(top_atoms)}
    dominant_full  = np.array([color_map.get(int(a), len(ATOM_COLORS) - 1)
                               for a in dominant_atoms])

    # Per-sample top-n reconstructions for n = 1 .. k_i
    recons = []
    for n in range(1, k_i + 1):
        top_n_idx   = order[:, :n]                                   # (n_j, n)
        top_n_codes = np.take_along_axis(codes, top_n_idx, axis=1)  # (n_j, n)
        top_n_W     = W_dec[top_n_idx]                               # (n_j, n, d)
        recon_n     = np.einsum("ij,ijd->id", top_n_codes, top_n_W) # (n_j, d)
        recons.append({"coords": pca.transform(recon_n), "dominant": dominant_full})

    return {
        "coords_true":   coords_true,
        "dominant_full": dominant_full,
        "recons":        recons,
        "k_i":           k_i,
    }


def plot_reconstruction_grid(
    shape_name: str,
    snapshots_by_variant: dict[str, dict],
    interactive: bool = False,
    out_path: str | None = None,
) -> object | None:
    """Reconstruction grid for a manifold instance across variants.

    Layout:
      Row 1:  True PCA shape (wide panel spanning all columns)
      Row 2+: One per variant — n=1..k_i cumulative OMP atoms

    Args:
        shape_name: substring to match against instance names (e.g. "helix")
        snapshots_by_variant: {variant_name: finalized_snapshot_dict}
        interactive: True → Plotly 3D rotatable figure (returned);
                     False → matplotlib 2D PNG saved to out_path
        out_path: save path for PNG (required when interactive=False)

    Returns:
        plotly.graph_objects.Figure when interactive=True, else None.
    """
    n_pca = 3 if interactive else 2

    variant_panels: dict[str, dict] = {}
    k_i: int | None = None

    for variant, snap in snapshots_by_variant.items():
        fig4d = snap.get("fig4d")
        if fig4d is None:
            print(f"  [{variant}] no fig4d in snapshot — skipping")
            continue
        inst = _find_instance(fig4d, shape_name)
        if inst is None:
            print(f"  [{variant}] no instance matching '{shape_name}' — skipping")
            continue
        panels = _prepare_panels(inst, fig4d["W_dec"], n_pca)
        if panels is None:
            print(f"  [{variant}] OMP returned no atoms — skipping")
            continue
        variant_panels[variant] = panels
        k_i = panels["k_i"]

    if not variant_panels:
        raise ValueError(f"No variants produced data for '{shape_name}'")

    if interactive:
        return _plotly_grid(shape_name, variant_panels, k_i)
    else:
        _matplotlib_grid(shape_name, variant_panels, k_i, out_path)
        return None


def _plotly_grid(shape_name: str, variant_panels: dict, k_i: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    variants   = list(variant_panels.keys())
    n_variants = len(variants)
    n_rows     = n_variants + 1

    # Row 1 spans all k_i columns (true PCA); rows 2+ have k_i scenes each
    specs = (
        [[{"type": "scene", "colspan": k_i}] + [None] * (k_i - 1)]
        + [[{"type": "scene"}] * k_i for _ in range(n_variants)]
    )

    fig = make_subplots(
        rows=n_rows,
        cols=k_i,
        specs=specs,
        vertical_spacing=0.04,
        horizontal_spacing=0.02,
    )

    # Row 1: true PCA (first variant's data — all share the same ground truth)
    p0  = next(iter(variant_panels.values()))
    xyz = p0["coords_true"]
    for ai in range(p0["k_i"]):
        mask  = p0["dominant_full"] == ai
        color = ATOM_COLORS[ai % len(ATOM_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
            mode="markers",
            marker=dict(size=2, color=color, opacity=0.65),
            showlegend=False,
        ), row=1, col=1)

    # Rows 2+: cumulative reconstructions
    for vi, (variant, p) in enumerate(variant_panels.items()):
        row = vi + 2
        for n, rec in enumerate(p["recons"], start=1):
            xyz = rec["coords"]
            for ai in range(n):
                mask  = rec["dominant"] == ai
                color = ATOM_COLORS[ai % len(ATOM_COLORS)]
                fig.add_trace(go.Scatter3d(
                    x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
                    mode="markers",
                    marker=dict(size=2, color=color, opacity=0.65),
                    showlegend=False,
                ), row=row, col=n)

    axis_kw = dict(showticklabels=False, showgrid=False, zeroline=False, title="")
    fig.update_scenes(xaxis=axis_kw, yaxis=axis_kw, zaxis=axis_kw, aspectmode="cube")

    # Row/column label annotations
    row_h  = 1.0 / n_rows
    annots = []
    for vi, variant in enumerate(variants):
        annots.append(dict(
            text=f"<b>{LABELS.get(variant, variant)}</b>",
            x=-0.01, y=1.0 - (vi + 1.5) * row_h,
            xref="paper", yref="paper",
            showarrow=False, font=dict(size=9), xanchor="right",
        ))
    for n in range(1, k_i + 1):
        x_pos = (n - 0.5) / k_i
        annots.append(dict(
            text=f"{n} atom{'s' if n > 1 else ''}",
            x=x_pos, y=1.0 - row_h + 0.005,
            xref="paper", yref="paper",
            showarrow=False, font=dict(size=8, color="#666"),
        ))

    fig.update_layout(
        title=dict(text=f"Reconstruction grid — {shape_name}  (k_SAE=10)", font=dict(size=13)),
        height=300 * n_rows,
        margin=dict(l=70, r=10, t=80, b=10),
        annotations=annots,
    )
    return fig


def _matplotlib_grid(shape_name: str, variant_panels: dict, k_i: int, out_path: str | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    variants   = list(variant_panels.keys())
    n_variants = len(variants)
    n_rows     = n_variants + 1

    fig = plt.figure(figsize=(3.5 * k_i, 3.0 * n_rows))
    gs  = GridSpec(n_rows, k_i, figure=fig, hspace=0.35, wspace=0.08)

    # Row 0: true PCA spanning all columns
    ax_true = fig.add_subplot(gs[0, :])
    p0 = next(iter(variant_panels.values()))
    for ai in range(p0["k_i"]):
        mask  = p0["dominant_full"] == ai
        color = ATOM_COLORS[ai % len(ATOM_COLORS)]
        ax_true.scatter(
            p0["coords_true"][mask, 0], p0["coords_true"][mask, 1],
            c=color, s=4, alpha=0.65, linewidths=0,
        )
    ax_true.set_title(f"True PCA — {shape_name}", fontsize=10, pad=4)
    ax_true.set_aspect("equal")
    ax_true.axis("off")

    # Rows 1+: one per variant, k_i columns
    for vi, (variant, p) in enumerate(variant_panels.items()):
        row = vi + 1
        for n, rec in enumerate(p["recons"], start=1):
            ax = fig.add_subplot(gs[row, n - 1])
            for ai in range(n):
                mask  = rec["dominant"] == ai
                color = ATOM_COLORS[ai % len(ATOM_COLORS)]
                ax.scatter(
                    rec["coords"][mask, 0], rec["coords"][mask, 1],
                    c=color, s=4, alpha=0.65, linewidths=0,
                )
            if vi == 0:
                ax.set_title(f"{n} atom{'s' if n > 1 else ''}", fontsize=8, pad=3)
            if n == 1:
                ax.set_ylabel(LABELS.get(variant, variant), fontsize=8, labelpad=4)
            ax.set_aspect("equal")
            ax.axis("off")

        # Empty panels for unused columns (k_i < max across variants)
        for col in range(len(p["recons"]), k_i):
            fig.add_subplot(gs[row, col]).axis("off")

    plt.suptitle(f"Reconstruction grid — {shape_name}  (k_SAE=10)", fontsize=11, y=1.01)

    if out_path:
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_variant(path: str) -> str | None:
    m = re.match(r"^(.+?)_snapshots", os.path.basename(path))
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruction grid: true PCA vs. cumulative SAE atom reconstructions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python results/explore.py helix results/snapshots/*4.b
  python results/explore.py circle results/snapshots/baseline_snapshots4.b --interactive
  python results/explore.py torus results/snapshots/*4.b --out results/figs/torus_grid.png
""",
    )
    parser.add_argument("shape_name", help="Instance name or substring (e.g. 'helix').")
    parser.add_argument("files", nargs="+", help="Finalized snapshot files.")
    parser.add_argument("--interactive", action="store_true",
                        help="Open a Plotly 3D rotatable figure in the browser.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: results/figs/reconstruction_grid_<shape>.png).")
    args = parser.parse_args()

    snapshots_by_variant: dict[str, dict] = {}
    for path in args.files:
        variant = _parse_variant(path)
        if variant is None:
            print(f"Warning: skipping unrecognized filename: {os.path.basename(path)}")
            continue
        print(f"Loading {os.path.basename(path)} ...", end=" ", flush=True)
        with open(path, "rb") as f:
            snapshots_by_variant[variant] = pickle.load(f)
        print("done.")

    if not snapshots_by_variant:
        print("No valid snapshot files found.")
        sys.exit(1)

    out_path = args.out
    if not args.interactive and out_path is None:
        figs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
        os.makedirs(figs_dir, exist_ok=True)
        out_path = os.path.join(figs_dir, f"reconstruction_grid_{args.shape_name}.png")

    fig = plot_reconstruction_grid(
        args.shape_name,
        snapshots_by_variant,
        interactive=args.interactive,
        out_path=out_path,
    )

    if args.interactive and fig is not None:
        fig.show()


if __name__ == "__main__":
    main()
