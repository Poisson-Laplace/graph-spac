# author: ferat
# date: 2026
# vol3 — GraphMetrics
# Graph-theoretic and spectral metrics for a seismic sensor array.
# Implements:
#   • Graph Laplacian  L = D − A  (weights = inter-sensor distances)
#   • Algebraic connectivity  λ₂  (Fiedler value)
#   • Array Response Function  W(k)  (ARF, 2D k-space)
#   • Side-Lobe Level  (SLL, dB)
#   • Bessel J₀ and Hankel H₀⁽¹⁾ kernel sampling error

import numpy as np
from scipy.special import j0, y0   # Bessel / Weber functions of order 0
import networkx as nx


# ── Hankel H₀⁽¹⁾ = J₀ + i·Y₀ ─────────────────────────────────────────────────

def hankel0(x):
    """
    Hankel function of the first kind, order 0.
    H₀⁽¹⁾(x) = J₀(x) + i·Y₀(x)
    Safe for x > 0  (Y₀ diverges as x → 0).
    """
    return j0(x) + 1j * y0(x)


class GraphMetrics:
    """
    Wraps an N-sensor array  coords : (N, 2)  and provides:
      - Graph Laplacian construction
      - λ₂ (algebraic connectivity)
      - ARF computation
      - SLL extraction
      - Bessel / Hankel kernel sampling error (Φ)
      - Azimuthal isotropy score
    """

    def __init__(self, coords):
        """
        Parameters
        ----------
        coords : array-like, shape (N, 2)  — physical [m] sensor positions.
        """
        self.coords = np.asarray(coords, dtype=float)
        assert self.coords.ndim == 2 and self.coords.shape[1] == 2
        self.N = self.coords.shape[0]

        # pre-compute all pairwise distances and direction vectors
        self._pairs = None       # list of (i, j, r_ij, dx, dy)
        self._distances = None
        self._build_pairs()

    # ── pairwise geometry ─────────────────────────────────────────────────────

    def _build_pairs(self):
        N = self.N
        n = N * (N - 1) // 2
        pairs   = []
        dists   = np.empty(n, dtype=float)

        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                dx = self.coords[j, 0] - self.coords[i, 0]
                dy = self.coords[j, 1] - self.coords[i, 1]
                r  = np.sqrt(dx * dx + dy * dy)
                pairs.append((i, j, r, dx, dy))
                dists[k] = r
                k += 1

        self._pairs     = pairs
        self._distances = dists

    @property
    def distances(self):
        return self._distances

    @property
    def r_min(self):
        return float(np.min(self._distances)) if len(self._distances) else 0.0

    @property
    def r_max(self):
        return float(np.max(self._distances)) if len(self._distances) else 0.0

    # ── Graph Laplacian ───────────────────────────────────────────────────────

    def laplacian(self):
        """
        Weighted graph Laplacian  L = D − A.
        Edge weight  w_{ij} = r_{ij}  [m]  (Euclidean distance).

        Returns
        -------
        L : (N, N) ndarray
        """
        N = self.N
        W = np.zeros((N, N), dtype=float)
        for i, j, r, _, _ in self._pairs:
            W[i, j] = r
            W[j, i] = r

        D = np.diag(W.sum(axis=1))
        return D - W

    def algebraic_connectivity(self):
        """
        λ₂ = second smallest eigenvalue of the Laplacian.
        Also called Fiedler value.  Higher → better connected / more robust.

        Returns
        -------
        lambda2 : float  ≥ 0
        """
        L  = self.laplacian()
        ev = np.linalg.eigvalsh(L)
        ev_sorted = np.sort(ev)
        # ev_sorted[0] ≈ 0 always;  λ₂ = ev_sorted[1]
        return float(ev_sorted[1])

    # ── Array Response Function (ARF) ─────────────────────────────────────────

    def arf(self, n_k=150, k_max=None):
        """
        2D Array Response Function (fully vectorised)
            W(k) = |Σ_n exp(i k · x_n)|² / N²

        Parameters
        ----------
        n_k   : int   — grid size in each k direction
        k_max : float — maximum wavenumber [rad/m].
                        Defaults to π / r_min (Nyquist), capped at 0.8.

        Returns
        -------
        kx_grid, ky_grid : (n_k, n_k) ndarrays
        arf_grid         : (n_k, n_k) ndarray  ∈ [0, 1]
        k_max_used       : float
        """
        if k_max is None:
            nyq = np.pi / self.r_min if self.r_min > 0 else 0.5
            main_lobe_width = 2 * np.pi / self.r_max if self.r_max > 0 else 0.1
            k_max = min(10 * main_lobe_width, nyq, 0.8)
            k_max = max(k_max, 0.05)

        k_vec = np.linspace(-k_max, k_max, n_k)
        kx, ky = np.meshgrid(k_vec, k_vec, indexing='xy')

        # Fully vectorised: phase = coords @ [kx_flat; ky_flat]
        # coords : (N, 2)
        # kx_flat: (n_k*n_k,)   → stacked → (2, n_k*n_k)
        kx_flat = kx.ravel()   # (n_k²,)
        ky_flat = ky.ravel()   # (n_k²,)
        # phase: (N, n_k²)
        phase = self.coords[:, 0:1] * kx_flat[None, :] + \
                self.coords[:, 1:2] * ky_flat[None, :]
        # sum over N sensors
        csum = np.sum(np.exp(1j * phase), axis=0)  # (n_k²,)
        w = (np.abs(csum / self.N) ** 2).reshape(n_k, n_k)
        return kx, ky, w, k_max

    # ── Side-Lobe Level ───────────────────────────────────────────────────────

    def sll_db(self, arf_grid=None, mask_fraction=0.12):
        """
        Side-Lobe Level in dB.

        SLL = 20 log₁₀( max(side-lobes) / peak )

        Parameters
        ----------
        arf_grid      : pre-computed ARF; if None, computed internally.
        mask_fraction : fraction of grid radius to mask as main-lobe.

        Returns
        -------
        sll : float  (dB, non-positive; closer to 0 → worse)
        """
        if arf_grid is None:
            _, _, arf_grid, _ = self.arf()

        n = arf_grid.shape[0]
        cx, cy = n // 2, n // 2
        peak = arf_grid[cy, cx]

        # circular mask around centre
        r_mask = int(n * mask_fraction)
        y_idx, x_idx = np.ogrid[:n, :n]
        mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= r_mask ** 2

        side = arf_grid.copy()
        side[mask] = 0.0
        max_side = np.max(side)

        if peak <= 0:
            return 0.0
        return float(-20 * np.log10(max_side / peak + 1e-12))

    # ── Bessel Sampling Error Φ_Bessel ────────────────────────────────────────

    def phi_bessel(self, k_eval=None):
        """
        Bessel kernel sampling error:
            Φ_J0(X) = Σ_{i<j} ( J₀(k·r_{ij}) − 1/N_pairs )²

        We evaluate at the mid-band wavenumber k = 2π / (geometric mean of r).

        The second term (1/N_pairs) is a normalisation placeholder that pushes the
        optimiser to spread lags; in practice the absolute value matters less than
        the relative comparison between individuals.

        Returns
        -------
        phi : float  (non-negative; lower → better Bessel coverage)
        """
        if k_eval is None:
            r_geom = np.exp(np.mean(np.log(self._distances + 1e-9)))
            k_eval = 2 * np.pi / r_geom if r_geom > 0 else 0.1

        vals = j0(k_eval * self._distances)
        target = 1.0 / len(vals)   # uniform target
        return float(np.sum((vals - target) ** 2))

    # ── Hankel Sampling Error Φ_Hankel ────────────────────────────────────────

    def phi_hankel(self, k_eval=None):
        """
        Hankel H₀⁽¹⁾ kernel sampling error (magnitude):
            Φ_H(X) = Σ_{i<j} ( |H₀⁽¹⁾(k·r_{ij})| − |H₀⁽¹⁾(k·r_geom)| )²

        Penalises lags that deviate from the Hankel manifold.
        We use |H₀⁽¹⁾| to keep the objective real-valued.

        The d_min constraint in the NSGA problem prevents r→0 singularity.
        """
        if k_eval is None:
            r_geom = np.exp(np.mean(np.log(self._distances + 1e-9)))
            k_eval = 2 * np.pi / r_geom if r_geom > 0 else 0.1

        h_vals = np.abs(hankel0(k_eval * self._distances))
        target = np.mean(h_vals)
        return float(np.sum((h_vals - target) ** 2))

    # ── Azimuthal isotropy score ──────────────────────────────────────────────

    def isotropy_score(self, n_bins=18):
        """
        Isotropy ∈ [0, 1].  1 = perfectly uniform azimuthal distribution.
        Based on CV of the full 360° direction histogram.
        """
        # both directions (i→j and j→i)
        azimuths = []
        for _, _, _, dx, dy in self._pairs:
            a = np.arctan2(dy, dx)
            azimuths += [a, a + np.pi]
        azimuths = np.arctan2(np.sin(azimuths), np.cos(azimuths))

        hist, _ = np.histogram(azimuths, bins=n_bins, range=(-np.pi, np.pi))
        mu = np.mean(hist)
        if mu == 0:
            return 0.0
        cv = np.std(hist) / mu
        return float(max(0.0, 1.0 - cv))

    # ── networkx graph (for NetworkX-based analyses) ──────────────────────────

    def to_networkx(self):
        G = nx.Graph()
        for i in range(self.N):
            G.add_node(i, pos=tuple(self.coords[i]))
        for i, j, r, _, _ in self._pairs:
            G.add_edge(i, j, weight=r)
        return G

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        _, _, arf_grid, _ = self.arf()
        sll = self.sll_db(arf_grid)
        lam2 = self.algebraic_connectivity()
        iso  = self.isotropy_score()
        return {
            "N":            self.N,
            "n_pairs":      len(self._pairs),
            "r_min_m":      self.r_min,
            "r_max_m":      self.r_max,
            "lambda2":      lam2,
            "sll_db":       sll,
            "isotropy":     iso,
            "phi_bessel":   self.phi_bessel(),
            "phi_hankel":   self.phi_hankel(),
        }


# ── standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(1249718046570)
    coords = rng.uniform(0, 100, size=(10, 2))
    gm = GraphMetrics(coords)
    s = gm.summary()
    for k, v in s.items():
        print(f"  {k:15s}: {v}")
