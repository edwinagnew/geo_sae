"""
Synthetic manifold zoo (Appendix E, arXiv 2604.28119).

Implements the eight manifold types from Table 4, the RMS-normalization
calibration (Eq. 12), the random orthonormal ambient embeddings, and the
additive mixture sampler (Eq. 13).

Output contract: x and contributions are in raw mixture space (no global
ℓ2 normalization). The training loop normalizes x by its mean ℓ2 norm;
eval code must apply the same factor to contributions, or restricted-R²
will be capped below 1 (Bug 1 in synthetic_experiment_notes.md).

Usage:
    zoo = build_zoo(EASY_CONFIG)
    rng = np.random.default_rng(0)

    # Training: generate batches on-the-fly (no ground truth stored)
    batch = zoo.generate(1024, rng)
    x = torch.from_numpy(batch.x)

    # Eval: fixed set with per-manifold contributions
    eval_data = zoo.generate(50_000, rng, return_ground_truth=True)
    # eval_data.contributions[j] = (n_j, d) float32 — active-sample contributions for
    #                               instance j, in the same row order as active_mask[:, j]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


# ── Raw samplers ──────────────────────────────────────────────────────────────
# Each returns float64 (n, k_i) — calibration needs the precision.
# Normalization (Eq. 12) converts to float32.

def _sample_circle(r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    th = rng.uniform(0.0, 2 * math.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)


def _sample_sphere(r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    # Area-uniform: cos φ ∼ Uniform(−1, 1)
    th = rng.uniform(0.0, 2 * math.pi, n)
    cos_ph = rng.uniform(-1.0, 1.0, n)
    sin_ph = np.sqrt(1.0 - cos_ph ** 2)
    return r * np.stack([sin_ph * np.cos(th), sin_ph * np.sin(th), cos_ph], axis=1)


def _sample_torus(R: float, r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    # Clifford 4D embedding confirmed in synthetic_experiment_notes.md:
    # ((R+r cosφ)cosθ, (R+r cosφ)sinθ, r cosφ, r sinφ) — k_i = 4
    th = rng.uniform(0.0, 2 * math.pi, n)
    ph = rng.uniform(0.0, 2 * math.pi, n)
    return np.stack([
        (R + r * np.cos(ph)) * np.cos(th),
        (R + r * np.cos(ph)) * np.sin(th),
        r * np.cos(ph),
        r * np.sin(ph),
    ], axis=1)


def _sample_mobius(w: float, n: int, rng: np.random.Generator) -> np.ndarray:
    th = rng.uniform(0.0, 2 * math.pi, n)
    t = rng.uniform(-w / 2.0, w / 2.0, n)
    return np.stack([
        (1.0 + t * np.cos(th / 2.0)) * np.cos(th),
        (1.0 + t * np.cos(th / 2.0)) * np.sin(th),
        t * np.sin(th / 2.0),
    ], axis=1)


def _sample_swiss_roll(theta_max: float, h_max: float, n: int, rng: np.random.Generator) -> np.ndarray:
    th = rng.uniform(0.0, theta_max, n)
    h = rng.uniform(0.0, h_max, n)
    return np.stack([th * np.cos(th), h, th * np.sin(th)], axis=1)


def _sample_helix(alpha: float, n: int, rng: np.random.Generator,
                  n_turns: int = 3, r: float = 1.0) -> np.ndarray:
    th = rng.uniform(0.0, 2 * math.pi * n_turns, n)
    return np.stack([r * np.cos(th), r * np.sin(th), alpha * th], axis=1)


def _sample_flat_disk(r_max: float, n: int, rng: np.random.Generator) -> np.ndarray:
    # r ∼ r_max · √U(0,1) ensures uniform area coverage (inverse CDF of disk)
    rho = r_max * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2 * math.pi, n)
    return np.stack([rho * np.cos(th), rho * np.sin(th)], axis=1)


def _sample_segment(length: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, length, (n, 1))


# ── Manifold type registry ────────────────────────────────────────────────────
# Each entry: d_i (intrinsic dim), k_i (embedding dim), sampler, easy params,
# and all 6 paper variants (Table 4 of arXiv 2604.28119).

_MANIFOLD_TYPES: dict[str, dict] = {
    "circle": {
        "d_i": 1, "k_i": 2, "sampler": _sample_circle,
        "easy": {"r": 1.0},
        "paper": [{"r": r} for r in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)],
    },
    "sphere": {
        "d_i": 2, "k_i": 3, "sampler": _sample_sphere,
        "easy": {"r": 1.0},
        "paper": [{"r": r} for r in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)],
    },
    "torus": {
        "d_i": 2, "k_i": 4, "sampler": _sample_torus,
        "easy": {"R": 2.0, "r": 1.0},
        "paper": [
            {"R": R, "r": r}
            for R, r in ((2, 0.5), (2, 1), (3, 1), (3, 1.5), (4, 1.5), (4, 2))
        ],
    },
    "mobius": {
        "d_i": 2, "k_i": 3, "sampler": _sample_mobius,
        "easy": {"w": 0.5},
        "paper": [{"w": w} for w in (0.2, 0.3, 0.5, 0.7, 1.0, 1.5)],
    },
    "swiss_roll": {
        "d_i": 2, "k_i": 3, "sampler": _sample_swiss_roll,
        "easy": {"theta_max": 3.0 * math.pi, "h_max": 3.0},
        "paper": [
            {"theta_max": t, "h_max": h}
            for t, h in (
                (2.0 * math.pi, 1.5),
                (2.5 * math.pi, 2.0),
                (3.0 * math.pi, 3.0),
                (3.5 * math.pi, 4.0),
                (4.0 * math.pi, 5.0),
                (4.5 * math.pi, 6.0),
            )
        ],
    },
    "helix": {
        "d_i": 1, "k_i": 3, "sampler": _sample_helix,
        "easy": {"alpha": 0.3},
        "paper": [{"alpha": a} for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)],
    },
    "flat_disk": {
        "d_i": 2, "k_i": 2, "sampler": _sample_flat_disk,
        "easy": {"r_max": 1.0},
        "paper": [{"r_max": r} for r in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)],
    },
    "segment": {
        "d_i": 1, "k_i": 1, "sampler": _sample_segment,
        "easy": {"length": 1.0},
        "paper": [{"length": l} for l in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)],
    },
}

ALL_MANIFOLD_TYPES: tuple[str, ...] = tuple(_MANIFOLD_TYPES)


# ── Manifold instance ─────────────────────────────────────────────────────────

@dataclass
class ManifoldInstance:
    """A single calibrated, embedded manifold instance.

    V has orthonormal rows, so the ambient embedding is norm-preserving:
    ‖γ̃ V‖₂ = ‖γ̃‖₂ for any local-coordinate vector γ̃.
    """
    name: str
    type: str
    d_i: int         # intrinsic dimension
    k_i: int         # embedding dimension (= number of atoms needed for subspace capture)
    params: dict
    V: np.ndarray    # (k_i, d) float32 — orthonormal rows
    mu: np.ndarray   # (k_i,) float32   — calibration centroid
    sigma: float     # scalar           — calibration RMS norm

    def sample_normalized(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n, k_i) float32 — uniform samples in normalized local coordinates."""
        raw = _MANIFOLD_TYPES[self.type]["sampler"](n=n, rng=rng, **self.params)
        return ((raw - self.mu) / self.sigma).astype(np.float32)

    def embed(self, gamma: np.ndarray) -> np.ndarray:
        """(n, k_i) normalized coords → (n, d) ambient-space contribution."""
        return gamma @ self.V


# ── Batch output types ────────────────────────────────────────────────────────

class TrainBatch(NamedTuple):
    x: np.ndarray  # (n, d) float32 — mixture samples


class EvalBatch(NamedTuple):
    x: np.ndarray                   # (n, d) float32 — mixture samples
    active_mask: np.ndarray         # (n, m) bool    — True where instance j was active
    contributions: list[np.ndarray] # list[m] of (n_j, d) float32 — per-instance active contribs,
                                    # rows in same order as active_mask[:, j].nonzero()


# ── Zoo configuration ─────────────────────────────────────────────────────────

@dataclass
class ZooConfig:
    d: int = 64
    n_variants: int = 1             # 1 = easy (one central param set per type),
                                    # 6 = full paper zoo (all Table 4 variants)
    L0: int = 1                     # active instances per sample
    types: list[str] | None = None  # None → all 8 types in canonical order
    n_calib: int = 50_000           # calibration points for Eq. 12
    noise_std: float = 1e-5         # ε ~ N(0, noise_std² I)
    seed: int = 42


# Pre-defined configs matching the two regimes in the guide.
EASY_CONFIG = ZooConfig(d=64, n_variants=1, L0=1, seed=42)
PAPER_CONFIG = ZooConfig(d=128, n_variants=6, L0=4, seed=42)


# ── Instance construction ─────────────────────────────────────────────────────

def _build_instance(
    type_name: str,
    params: dict,
    d: int,
    n_calib: int,
    rng: np.random.Generator,
) -> ManifoldInstance:
    spec = _MANIFOLD_TYPES[type_name]
    k_i, d_i = spec["k_i"], spec["d_i"]

    # Calibration (Eq. 12): μ_i = mean, σ_i = √E[‖γ(θ) − μ_i‖²]
    calib = spec["sampler"](n=n_calib, rng=rng, **params).astype(np.float64)
    mu = calib.mean(axis=0)
    centered = calib - mu
    sigma = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))

    # Orthonormal frame V ∈ ℝ^{k_i × d} via QR of a Gaussian matrix.
    # Q-factor of (d × k_i) Gaussian has orthonormal columns → V = Q.T has orthonormal rows.
    G = rng.standard_normal((d, k_i))
    Q, _ = np.linalg.qr(G)
    V = Q.T.astype(np.float32)  # (k_i, d)

    param_str = "_".join(f"{k}{v:.3g}" for k, v in sorted(params.items()))
    return ManifoldInstance(
        name=f"{type_name}_{param_str}",
        type=type_name,
        d_i=d_i,
        k_i=k_i,
        params=params,
        V=V,
        mu=mu.astype(np.float32),
        sigma=sigma,
    )


# ── Manifold zoo ──────────────────────────────────────────────────────────────

class ManifoldZoo:
    """Collection of manifold instances with an additive-mixture generator.

    Build via ``build_zoo(config)``; do not construct directly.
    """

    def __init__(
        self,
        instances: list[ManifoldInstance],
        L0: int,
        noise_std: float = 1e-5,
    ):
        self.instances = instances
        self.L0 = L0
        self.noise_std = noise_std

    @property
    def d(self) -> int:
        return self.instances[0].V.shape[1]

    @property
    def n_instances(self) -> int:
        return len(self.instances)

    def instance_index(self, name: str) -> int:
        for i, inst in enumerate(self.instances):
            if inst.name == name:
                return i
        raise KeyError(name)

    def generate(
        self,
        n: int,
        rng: np.random.Generator,
        return_ground_truth: bool = False,
    ) -> TrainBatch | EvalBatch:
        """Sample n points from the additive mixture model (Eq. 13).

        For training: ``return_ground_truth=False`` (default) returns only x.
        For eval: ``return_ground_truth=True`` also returns active_mask and
        contributions. Contributions are stored sparsely (per-instance active rows
        only): n × L0 × d × 4 bytes total, regardless of m.
        """
        m = self.n_instances
        d = self.d
        L0 = self.L0

        # Uniform random subset of L0 instances per sample, without replacement.
        # Sorting per-row uniform scores is a standard trick for this.
        active_ids = np.argsort(rng.random((n, m)), axis=1)[:, :L0]  # (n, L0)

        x = np.zeros((n, d), dtype=np.float32)
        if return_ground_truth:
            active_mask = np.zeros((n, m), dtype=bool)
            contrib_chunks: list[list[np.ndarray]] = [[] for _ in range(m)]
            index_chunks: list[list[np.ndarray]] = [[] for _ in range(m)]

        # For each slot l ∈ [L0], scatter each instance's contribution to the
        # samples where it was drawn for that slot.
        for l in range(L0):
            slot_ids = active_ids[:, l]  # (n,) — instance index at slot l per sample
            for j, inst in enumerate(self.instances):
                mask = slot_ids == j    # (n,) bool
                n_j = int(mask.sum())
                if n_j == 0:
                    continue
                gamma = inst.sample_normalized(n_j, rng)  # (n_j, k_i)
                contrib = inst.embed(gamma)               # (n_j, d)
                x[mask] += contrib
                if return_ground_truth:
                    active_mask[mask, j] = True
                    contrib_chunks[j].append(contrib)
                    index_chunks[j].append(np.where(mask)[0])

        if self.noise_std > 0:
            x += (rng.standard_normal((n, d)) * self.noise_std).astype(np.float32)

        if return_ground_truth:
            # Sort each instance's contributions by sample index so rows align with
            # active_mask[:, j].nonzero() — needed for correct R² computation.
            contributions: list[np.ndarray] = []
            for j in range(m):
                if not contrib_chunks[j]:
                    contributions.append(np.zeros((0, d), dtype=np.float32))
                else:
                    idx = np.concatenate(index_chunks[j])
                    con = np.concatenate(contrib_chunks[j])
                    order = np.argsort(idx)
                    contributions.append(con[order])
            return EvalBatch(x=x, active_mask=active_mask, contributions=contributions)
        return TrainBatch(x=x)


def build_zoo(config: ZooConfig = EASY_CONFIG) -> ManifoldZoo:
    """Construct a ManifoldZoo from a ZooConfig.

    The same seed governs both calibration and the random orthonormal frames,
    so the zoo is fully reproducible.
    """
    rng = np.random.default_rng(config.seed)
    types = config.types or list(_MANIFOLD_TYPES)
    instances: list[ManifoldInstance] = []
    for type_name in types:
        spec = _MANIFOLD_TYPES[type_name]
        variants = [spec["easy"]] if config.n_variants == 1 else spec["paper"][: config.n_variants]
        for params in variants:
            instances.append(
                _build_instance(type_name, params, config.d, config.n_calib, rng)
            )
    return ManifoldZoo(instances=instances, L0=config.L0, noise_std=config.noise_std)
