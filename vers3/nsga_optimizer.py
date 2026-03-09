# author: ferat
# date: 2026
# vol3 — Extended NSGA-II Optimizer
# Supports: noiseazimuth, multi-azimuth, Golomb ruler, LoS penalty, near-field Hankel

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize as pymoo_minimize

from coarray import CoArray
from graph_metrics import GraphMetrics


# ── Directional score (noiseazimuth) ─────────────────────────────────────────

def directional_score(coords, noise_azimuths_deg):
    """
    Reward sensor pairs oriented PERPENDICULAR to each noise source azimuth.
    A pair perpendicular to the wave propagation direction maximally samples
    the wavefield coherence.

    Returns a score ∈ [0, 1]. Higher = better directional alignment.
    noise_azimuths_deg : list of floats (geographic degrees, 0=North, CW)
    """
    if not noise_azimuths_deg:
        return 1.0   # isotropic — neutral

    coords = np.asarray(coords)
    N = len(coords)
    scores = []
    for az_deg in noise_azimuths_deg:
        # geographic az → math angle of noise propagation direction
        az_math = np.pi / 2 - np.radians(az_deg)
        # perpendicular to propagation = az_math ± π/2
        perp = az_math + np.pi / 2

        pair_scores = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = coords[j, 0] - coords[i, 0]
                dy = coords[j, 1] - coords[i, 1]
                pair_az = np.arctan2(dy, dx)
                # |cos(pair_az - perp)| → max when pair is perpendicular to perp
                pair_scores.append(abs(np.cos(pair_az - az_math)))
        scores.append(np.mean(pair_scores))

    return float(np.mean(scores))


# ── Line-of-Sight (LoS) penalty ──────────────────────────────────────────────

def los_penalty(coords, gm, n_samples=10):
    """
    For each pair (i, j), sample n_samples points along the segment.
    If any sample falls on an obstacle cell, that edge is penalised.
    Returns the fraction of edges that cross obstacles ∈ [0, 1].
    This acts as a 4th NSGA-II objective to minimise.
    """
    has_obstacles = (gm.grid == 0).any()
    if not has_obstacles:
        return 0.0

    coords = np.asarray(coords)
    N = len(coords)
    bad_edges = 0
    total_edges = 0

    for i in range(N):
        for j in range(i + 1, N):
            total_edges += 1
            for t in np.linspace(0, 1, n_samples):
                pt = coords[i] + t * (coords[j] - coords[i])
                c = int(round(pt[0] / gm.grid_spacing))
                r = int(round(pt[1] / gm.grid_spacing))
                c = np.clip(c, 0, gm.cols - 1)
                r = np.clip(r, 0, gm.rows - 1)
                if gm.grid[r, c] == 0:
                    bad_edges += 1
                    break   # one obstruction = edge is blocked

    if total_edges == 0:
        return 0.0
    return bad_edges / total_edges


# ── Golomb ruler penalty ──────────────────────────────────────────────────────

def golomb_redundancy(coords, n_bins=100, r_max_hint=None):
    """
    Penalise repeated lag distances in the co-array.
    For a perfect 2D Golomb ruler, each histogram bin has ≤ 1 entry.
    Returns the fraction of bins with count > 1 (redundancy score ∈ [0, 1]).
    Lower is better.
    """
    ca = CoArray(coords)
    lags = ca.lags
    if len(lags) == 0:
        return 1.0

    r_max = r_max_hint or lags.max()
    if r_max <= 0:
        return 1.0

    hist, _ = np.histogram(lags, bins=n_bins, range=(0, r_max * 1.05))
    redundant = np.sum(hist > 1)
    return float(redundant / n_bins)


# ── Near-field Hankel objective ───────────────────────────────────────────────

def near_field_phi(coords, source_pos, k_eval=None):
    """
    Near-field Hankel sampling error.
    For each pair (i, j), the expected coherence involves the relative
    phase to a point source at source_pos:
        γ_ij = H₀⁽¹⁾(k·r_i) / H₀⁽¹⁾(k·r_j)   (amplitude)
    We penalise large variance in the sampled Hankel phase across pairs.
    """
    from scipy.special import j0, y0

    coords = np.asarray(coords)
    src = np.asarray(source_pos)
    dists_to_src = np.linalg.norm(coords - src, axis=1)

    if k_eval is None:
        r_mean = np.mean(dists_to_src)
        k_eval = 2 * np.pi / r_mean if r_mean > 0 else 0.1

    h_vals = np.abs(j0(k_eval * dists_to_src) + 1j * y0(k_eval * dists_to_src + 1e-9))
    # reward an array where the Hankel magnitude varies maximally
    # (samples different phase-front curvatures)
    if np.mean(h_vals) == 0:
        return 1.0
    cv = np.std(h_vals) / np.mean(h_vals)
    return float(1.0 - cv)   # lower = less variation = penalised


# ── NSGA-II Problem ───────────────────────────────────────────────────────────

class GraphSPACProblem(ElementwiseProblem):
    """
    Multi-objective problem. Objectives (all minimised):
      f1 = −LSD entropy
      f2 = −SLL (dB)
      f3 = −λ₂
      f4 = −directional_score  (if noise_azimuths given)
      f5 = LoS penalty         (if los_penalty=True)
      f6 = Golomb redundancy   (if golomb=True)
      f7 = near_field_phi      (if near_field=True)

    Constraints (ieq ≤ 0):
      d_min pair constraints + domain feasibility constraints
    """

    def __init__(self, grid_manager, n_sensors, d_min,
                 kernel="bessel",
                 noise_azimuths=None,
                 los_penalty_flag=False,
                 golomb_flag=False,
                 near_field=False,
                 near_field_source=None,
                 focus=30.0):

        self.gm              = grid_manager
        self.N               = n_sensors
        self.d_min           = float(d_min)
        self.kernel          = kernel
        self.noise_azimuths  = noise_azimuths or []
        self.los_flag        = los_penalty_flag
        self.golomb_flag     = golomb_flag
        self.near_field      = near_field
        self.near_field_src  = near_field_source
        self.focus           = float(focus)

        x_max = grid_manager.x_max
        y_max = grid_manager.y_max

        xl = np.zeros(2 * n_sensors)
        xu = np.empty(2 * n_sensors)
        for i in range(n_sensors):
            xu[2 * i]     = x_max
            xu[2 * i + 1] = y_max

        n_pairs = n_sensors * (n_sensors - 1) // 2
        has_obstacles = (grid_manager.grid == 0).any()
        n_feas = n_sensors if has_obstacles else 0

        # count objectives
        n_obj = 3   # LSD, SLL, lambda2 always
        if self.noise_azimuths:
            n_obj += 1
        if self.los_flag:
            n_obj += 1
        if self.golomb_flag:
            n_obj += 1
        if self.near_field:
            n_obj += 1

        super().__init__(
            n_var=2 * n_sensors,
            n_obj=n_obj,
            n_ieq_constr=n_pairs + n_feas,
            xl=xl,
            xu=xu,
        )
        print(f"[NSGA-II] Objectives: {n_obj}  |  "
              f"kernel={kernel}  noiseaz={self.noise_azimuths}  "
              f"golomb={golomb_flag}  los={los_penalty_flag}  nf={near_field}")

    def _is_in_domain(self, x, y):
        idx = self.gm.physical_to_index(x, y)
        return idx is not None

    def _evaluate(self, x, out, *args, **kwargs):
        coords = x.reshape(self.N, 2)

        # ── core objectives ──
        ca   = CoArray(coords)
        gmet = GraphMetrics(coords)

        lsd_h = ca.lsd_entropy(n_bins=30)
        lam2  = gmet.algebraic_connectivity()
        _, _, arf_grid, _ = gmet.arf()
        sll   = gmet.sll_db(arf_grid)

        # Baseline objective: negative LSD entropy
        f1 = -lsd_h

        # ── Deep Focus Fixes (Aperture Bias & Ellipticity) ──
        # 1. Dynamic Aperture Penalty (Regularization)
        # Penalize rmax stretching > 1.5 * focus.
        r_max = ca.r_max
        if r_max > 1.5 * self.focus:
            f1 += 5.0 * (r_max - 1.5 * self.focus) / self.focus

        # 2. Isotropy Weight Rebalancing
        # Enhance the importance of circular uniformity.
        iso = gmet.isotropy_score()
        f1 += 2.0 * (1.0 - iso)

        # 3. r_min Density Check (Spectral Hole Filler)
        # Ensure at least 20% of station pairs are within r < 1.0 * focus.
        n_pairs = len(ca.lags)
        if n_pairs > 0:
            near_frac = np.sum(ca.lags < 1.0 * self.focus) / n_pairs
            if near_frac < 0.20:
                f1 += 3.0 * (0.20 - near_frac)

        # 4. Kernel Fit Error
        # Ensure the GA respects the specified wavefield propagation model.
        if self.kernel == "hankel":
            phi = gmet.phi_hankel()
        else:
            phi = gmet.phi_bessel()
        f1 += phi

        F = [f1, -sll, -lam2]

        # ── optional objectives ──
        if self.noise_azimuths:
            ds = directional_score(coords, self.noise_azimuths)
            F.append(-ds)    # maximise directional alignment

        if self.los_flag:
            lp = los_penalty(coords, self.gm, n_samples=8)
            F.append(lp)     # minimise LoS crossings

        if self.golomb_flag:
            gr = golomb_redundancy(coords)
            F.append(gr)     # minimise redundancy

        if self.near_field:
            src = self.near_field_src or (
                self.gm.x_max / 2, -20.0)   # default: 20m south of centre
            nf = near_field_phi(coords, src)
            F.append(nf)     # minimise (1 - cv) i.e. maximise sampling variation

        out["F"] = F

        # ── constraints ──
        g = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                r  = np.sqrt(dx * dx + dy * dy)
                g.append(self.d_min - r)

        has_obstacles = (self.gm.grid == 0).any()
        if has_obstacles:
            for i in range(self.N):
                xi, yi = coords[i]
                g.append(0.0 if self._is_in_domain(xi, yi) else 1.0)

        out["G"] = g


# ── Pareto knee-point ─────────────────────────────────────────────────────────

def knee_point(F):
    """Minimum-norm point in normalised objective space (utopia = origin)."""
    F_min  = F.min(axis=0)
    F_max  = F.max(axis=0)
    denom  = F_max - F_min
    denom[denom == 0] = 1.0
    F_norm = (F - F_min) / denom
    return int(np.argmin(np.linalg.norm(F_norm, axis=1)))


# ── Main optimizer interface ──────────────────────────────────────────────────

class NSGAOptimizer:
    SEED = 1249718046570

    def __init__(self, grid_manager, n_sensors=10, d_min=5.0,
                 kernel="bessel",
                 noise_azimuths=None,
                 los_penalty_flag=False,
                 golomb_flag=False,
                 near_field=False,
                 near_field_source=None,
                 focus=30.0,
                 pop_size=100, n_gen=200, seed=None):

        self.gm               = grid_manager
        self.N                = n_sensors
        self.d_min            = d_min
        self.kernel           = kernel
        self.noise_azimuths   = noise_azimuths or []
        self.los_flag         = los_penalty_flag
        self.golomb_flag      = golomb_flag
        self.near_field       = near_field
        self.near_field_src   = near_field_source
        self.focus            = focus
        self.pop_size         = pop_size
        self.n_gen            = n_gen
        self.seed             = seed if seed is not None else self.SEED
        np.random.seed(self.seed % (2**31 - 1))

    def run(self, verbose=True, callback=None):
        problem = GraphSPACProblem(
            self.gm, self.N, self.d_min,
            kernel=self.kernel,
            noise_azimuths=self.noise_azimuths,
            los_penalty_flag=self.los_flag,
            golomb_flag=self.golomb_flag,
            near_field=self.near_field,
            near_field_source=self.near_field_src,
            focus=self.focus
        )

        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        kwargs = {
            "seed": self.seed % (2**31 - 1),
            "verbose": verbose,
            "save_history": False,
        }
        if callback is not None:
            kwargs["callback"] = callback

        res = pymoo_minimize(
            problem,
            algorithm,
            get_termination("n_gen", self.n_gen),
            **kwargs
        )

        pareto_X = res.X
        pareto_F = res.F

        if pareto_X is None or len(pareto_X) == 0:
            # Fallback: pick the least-infeasible individual from final population
            pop = res.pop
            if pop is not None and len(pop) > 0:
                import warnings
                warnings.warn(
                    "[NSGA-II] No feasible Pareto front found. "
                    "Returning least-infeasible individual. "
                    "Try increasing --pop or --gens, or relaxing --dmin.",
                    RuntimeWarning
                )
                cv_vals = np.array([ind.CV[0] for ind in pop])
                best_idx = int(np.argmin(cv_vals))
                best_coords = pop[best_idx].X.reshape(self.N, 2)
                # build synthetic single-point Pareto for downstream code
                pareto_X = pop[best_idx].X[None, :]
                pareto_F = pop[best_idx].F[None, :]
            else:
                raise RuntimeError(
                    "NSGA-II returned empty Pareto front and no population. "
                    "Try increasing --pop or relaxing --dmin."
                )
        else:
            idx = knee_point(pareto_F)
            best_coords = pareto_X[idx].reshape(self.N, 2)


        if verbose:
            obj_str = "  ".join(f"f{i+1}={pareto_F[0,i]:.4f}"
                                 for i in range(pareto_F.shape[1]))
            print(f"\n[NSGA-II] Pareto front: {len(pareto_F)} solutions")
            print(f"[NSGA-II] Knee-point: {obj_str}")


        return best_coords, pareto_F, pareto_X, res


if __name__ == "__main__":
    from grid_manager import GridManager
    gm = GridManager(default_grid_size=20, grid_spacing=5.0)
    opt = NSGAOptimizer(gm, n_sensors=6, d_min=5.0, kernel="bessel",
                        pop_size=30, n_gen=15)
    coords, pF, pX, _ = opt.run(verbose=True)
    print(coords)
