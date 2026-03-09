# author: ferat
# date: 2026
# vol3 — CoArray
# Computes the co-array (difference set Δ) and the
# Lag Sampling Density (LSD) entropy for an N-sensor deployment.

import numpy as np


class CoArray:
    """
    For N sensor coordinates X = {x_1, …, x_N} ⊂ R²
    the co-array is

        Δ = { r_ij = x_i − x_j   for all i < j }

    The scalar lag r = ||r_ij||₂  serves as argument for J₀(kr) or H₀⁽¹⁾(kr).
    """

    def __init__(self, coords):
        """
        Parameters
        ----------
        coords : array-like, shape (N, 2)
            Physical sensor positions in metres.
        """
        self.coords = np.asarray(coords, dtype=float)
        assert self.coords.ndim == 2 and self.coords.shape[1] == 2, \
            "coords must be (N, 2)"
        self.N = self.coords.shape[0]
        self._compute()

    # ── internals ─────────────────────────────────────────────────────────────

    def _compute(self):
        """Compute all pairwise difference vectors and scalar lags."""
        N = self.N
        n_pairs = N * (N - 1) // 2

        diff_vecs = np.empty((n_pairs, 2), dtype=float)  # r_ij = x_i − x_j
        lags      = np.empty(n_pairs, dtype=float)        # ||r_ij||

        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                dv = self.coords[i] - self.coords[j]
                diff_vecs[k] = dv
                lags[k] = np.linalg.norm(dv)
                k += 1

        self.diff_vecs = diff_vecs   # (n_pairs, 2)
        self.lags      = lags        # (n_pairs,)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def r_min(self):
        return float(np.min(self.lags)) if len(self.lags) else 0.0

    @property
    def r_max(self):
        return float(np.max(self.lags)) if len(self.lags) else 0.0

    @property
    def n_pairs(self):
        return len(self.lags)

    # ── Lag Sampling Density (LSD) entropy ────────────────────────────────────

    def lsd_entropy(self, n_bins=30):
        """
        Logarithmic-scale Lag Sampling Density entropy.

        A perfectly log-uniform distribution of lags has maximum entropy.
        We compute Shannon entropy H = -Σ p_k log(p_k) in log₁₀ space.

        Returns
        -------
        entropy : float  ∈ [0, log(n_bins)]
            Higher is better (more diverse lag coverage).
        """
        if self.r_min <= 0:
            return 0.0

        log_lags = np.log10(self.lags)
        lo, hi = np.log10(self.r_min * 0.9), np.log10(self.r_max * 1.1)
        hist, _ = np.histogram(log_lags, bins=n_bins, range=(lo, hi))

        p = hist / hist.sum() if hist.sum() > 0 else hist.astype(float)
        # avoid log(0)
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        return float(entropy)

    def log_uniformity(self, n_bins=20):
        """
        Coefficient of variation of the log-space histogram.
        0 = perfectly uniform (ideal), larger = more clustered.
        """
        if self.r_min <= 0:
            return 1.0
        log_lags = np.log10(self.lags)
        lo, hi = np.log10(self.r_min * 0.9), np.log10(self.r_max * 1.1)
        hist, _ = np.histogram(log_lags, bins=n_bins, range=(lo, hi))
        mu = np.mean(hist)
        if mu == 0:
            return 1.0
        return float(np.std(hist) / mu)

    # ── convenience ──────────────────────────────────────────────────────────

    def azimuths_rad(self):
        """
        Azimuth of each lag vector in geographic/math convention.
        NOTE: grid y increases southward, so we NEGATE dy so that
        positive dy_geo = northward (correct for angle computations).
        Returns array of length 2 * n_pairs (both directions).
        """
        dy_geo = -self.diff_vecs[:, 1]   # negate: grid-south -> geographic-north
        dx_geo =  self.diff_vecs[:, 0]
        az = np.arctan2(dy_geo, dx_geo)
        return np.concatenate([az, az + np.pi])

    def azimuths_geo_folded(self):
        """
        Geographic azimuth (North=0, CW), folded to [0, π).
        Grid y increases southward, so negate dy before computing.
        """
        dy_geo = -self.diff_vecs[:, 1]   # flip: screen-south → geographic-north
        dx_geo =  self.diff_vecs[:, 0]
        az_math = np.arctan2(dy_geo, dx_geo)          # math (East=0, CCW)
        geo     = (np.pi / 2 - az_math) % (2 * np.pi) # convert to geographic
        geo_opp = (geo + np.pi) % (2 * np.pi)
        all_az  = np.concatenate([geo, geo_opp])
        return all_az % np.pi   # fold to [0, π)

    def isotropy_score(self, n_bins=18):
        """
        Isotropy ∈ [0, 1].  1 = perfectly uniform azimuthal distribution.
        """
        dy_geo = -self.diff_vecs[:, 1]   # correct for screen-y convention
        dx_geo =  self.diff_vecs[:, 0]
        az_fwd = np.arctan2(dy_geo, dx_geo)
        azimuths = np.concatenate([az_fwd, az_fwd + np.pi])
        azimuths = np.arctan2(np.sin(azimuths), np.cos(azimuths))
        hist, _ = np.histogram(azimuths, bins=n_bins, range=(-np.pi, np.pi))
        if np.mean(hist) == 0:
            return 0.0
        cv = np.std(hist) / np.mean(hist)
        return float(max(0.0, 1.0 - cv))

    def bessel_coverage_score(self, n_bins=10):
        """
        Reward arrays whose log-distance distribution spans a wide range
        and is uniform (maximises J₀ kernel sampling).
        Returns score ∈ [0, 1].
        """
        if self.r_min <= 0 or len(self.lags) < 2:
            return 0.0

        log_lags = np.log10(self.lags)
        log_range = log_lags.max() - log_lags.min()

        hist, _ = np.histogram(log_lags, bins=n_bins)
        mu = np.mean(hist)
        uniformity = max(0.0, 1.0 - (np.std(hist) / (mu + 1e-10)))

        # two decades of range → perfect
        range_score = min(log_range / 2.0, 1.0)
        return float(0.5 * range_score + 0.5 * uniformity)


# ── standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 100, size=(10, 2))
    ca = CoArray(coords)
    print(f"N pairs : {ca.n_pairs}")
    print(f"r_min   : {ca.r_min:.2f} m")
    print(f"r_max   : {ca.r_max:.2f} m")
    print(f"LSD H   : {ca.lsd_entropy():.4f}")
    print(f"Isotropy: {ca.isotropy_score():.4f}")
    print(f"Bessel  : {ca.bessel_coverage_score():.4f}")
