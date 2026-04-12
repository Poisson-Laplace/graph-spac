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
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.termination import get_termination
from pymoo.optimize import minimize as pymoo_minimize

from coarray import CoArray
from graph_metrics import GraphMetrics
from geometry_transforms import apply_transforms


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

def near_field_phi(coords, source_pos, depth=0.0, k_eval=None):
    """
    Near-field Hankel phase matching objective.
    ONLY computes phase variance — centroid anchoring is now a hard G constraint.
    """
    from scipy.special import j0, y0

    coords = np.asarray(coords)
    sx, sy = source_pos[0], source_pos[1]

    # True 3D distance from sensors (Z=0) to source at depth
    dx = coords[:, 0] - sx
    dy = coords[:, 1] - sy
    dz = abs(depth)
    dists_to_src = np.sqrt(dx**2 + dy**2 + dz**2)

    if k_eval is None:
        r_mean = np.mean(dists_to_src)
        k_eval = 2 * np.pi / r_mean if r_mean > 0 else 0.1

    h_vals = np.abs(j0(k_eval * dists_to_src) + 1j * y0(k_eval * dists_to_src + 1e-9))

    cv = 0.0
    h_mean = np.mean(h_vals)
    if h_mean > 0:
        cv = np.std(h_vals) / h_mean

    return float(cv)


# ── Topological Awareness / Spatial Seeding ───────────────────────────────────

class SpatialSeedingSampling(Sampling):
    """
    Topologically Aware Initialization Motor (Spatial Seeding Engine).
    Instead of blind Euclidean randomization, this engine extracts the entire 
    feasible manifold (Ω_free) and seeds the initial population exactly on 
    the accessible grid nodes. This prevents non-convex "Topological Locking" 
    by ensuring that every isolated branch or labyrinth path has an initial 
    density of sensors (anchor candidates).
    """
    def __init__(self, manifold_manager, N):
        super().__init__()
        self.gm = manifold_manager
        self.N = N
        
        # Extract physical coordinates of all accessible nodes
        self.valid_pts = np.zeros((len(self.gm.valid_indices), 2))
        for i, (r, c) in enumerate(self.gm.valid_indices):
            self.valid_pts[i, 0] = c * self.gm.grid_spacing
            self.valid_pts[i, 1] = r * self.gm.grid_spacing

    def _do(self, problem, n_samples, **kwargs):
        X = np.empty((n_samples, problem.n_var))
        for i in range(n_samples):
            # Select N unique sensors fully randomly distributed across the valid manifold
            indices = np.random.choice(len(self.valid_pts), self.N, replace=False)
            pts = self.valid_pts[indices].copy()
            
            # Add spatial jitter (micro-mutations)
            noise = np.random.uniform(-self.gm.grid_spacing/3, self.gm.grid_spacing/3, (self.N, 2))
            pts += noise
            
            # Constrain to strict boundary limits
            pts[:, 0] = np.clip(pts[:, 0], 0, self.gm.x_max)
            pts[:, 1] = np.clip(pts[:, 1], 0, self.gm.y_max)
            
            # Ensure even the jittered points are magnetic to the valid cells
            pts = self.gm.snap_to_valid_coords(pts)
            
            X[i, :] = pts.flatten()
        return X

class TopologicalRepair(Repair):
    """
    Magnetic Mutation & Mating Repair Operator.
    If crossover or mutation throws a sensor into the Red Zone (obstacles),
    this instantly snaps it to the nearest Euclidean Valid Point before evaluation.
    This ensures 100% of the population evaluates on the actual street network.
    """
    def __init__(self, manifold_manager, N):
        super().__init__()
        self.gm = manifold_manager
        self.N = N

    def _do(self, problem, Z, **kwargs):
        for i in range(len(Z)):
            coords = Z[i].reshape(self.N, 2)
            coords[:, 0] = np.clip(coords[:, 0], 0, self.gm.x_max)
            coords[:, 1] = np.clip(coords[:, 1], 0, self.gm.y_max)
            # Snap out of buildings
            snapped = self.gm.snap_to_valid_coords(coords)
            Z[i, :] = snapped.flatten()
        return Z

# ── NSGA-II Problem ───────────────────────────────────────────────────────────

class GraphSPACProblem(ElementwiseProblem):
    """
    Multi-objective problem. Objectives (all minimised):
      f1 = −LSD entropy
      f2 = −SLL (dB)
      f3 = −λ₂
      f4 = −Graph Entropy (maximize robustness)
      f5 = Spatial Entropy / Voronoi CV (minimize spacing discrepancy)
      f6..f9 = auxiliary constraints

    Constraints (ieq ≤ 0):
      d_min pair constraints + domain feasibility constraints
    """

    def __init__(self, manifold_manager, n_sensors, d_min,
                 kernel="bessel",
                 noise_azimuths=None,
                 los_penalty_flag=False,
                 golomb_flag=False,
                 near_field=False,
                 near_field_source=None,
                 focus=30.0,
                 vs=500.0,
                 use_poincare=False,
                 r_max=None,
                 weights=None):  # NEW: scenario-specific weights

        self.gm              = manifold_manager
        self.N               = n_sensors
        self.d_min           = float(d_min)
        self.kernel          = kernel
        self.noise_azimuths  = noise_azimuths or []
        self.los_flag        = los_penalty_flag
        self.golomb_flag     = golomb_flag
        self.near_field      = near_field
        self.near_field_src  = near_field_source
        self.focus           = float(focus)
        self.vs              = float(vs)
        self.use_poincare    = use_poincare
        self.r_max           = float(r_max) if r_max is not None else None
        
        # NEW: Scenario-based objective weights (default: precision mode)
        self.weights = weights or {
            "lsd": 1.0, "sll": 5.0, "lam2": 0.5,
            "graph_ent": 0.5, "dr": 0.5, "eta": 0.5
        }

        self.x_max = manifold_manager.x_max
        self.y_max = manifold_manager.y_max

        if self.use_poincare:
            xl = np.ones(2 * n_sensors) * -0.999
            xu = np.ones(2 * n_sensors) * 0.999
        else:
            xl = np.zeros(2 * n_sensors)
            xu = np.empty(2 * n_sensors)
            for i in range(n_sensors):
                xu[2 * i]     = self.x_max
                xu[2 * i + 1] = self.y_max

        n_pairs = n_sensors * (n_sensors - 1) // 2
        
        has_obstacles = (manifold_manager.grid == 0).any()
        n_feas = n_sensors if has_obstacles else 0
        
        # constraints: n_pairs for d_min, n_feas for domain, 1 for hard aperture limit
        n_ieq = n_pairs + n_feas + 1
        if self.r_max is not None:
            n_ieq += n_pairs
        if self.near_field and self.near_field_src is not None:
            n_ieq += 1  # hard centroid anchor: centroid must be within d_min of source XY

        # count objectives
        n_obj = 2   # Multi-Objective Pareto Optimization

        super().__init__(
            n_var=2 * n_sensors,
            n_obj=n_obj,
            n_ieq_constr=n_ieq,
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
        coords_raw = x.reshape(self.N, 2)
        if self.use_poincare:
            coords = apply_transforms(coords_raw, self.x_max, self.y_max, use_poincare=True)
        else:
            coords = coords_raw
        
        has_obstacles = (self.gm.grid == 0).any()
        if has_obstacles:
            dist_matrix = self.gm.get_geodesic_distance_matrix(coords)
        else:
            dist_matrix = None

        # ── core objectives ──
        ca   = CoArray(coords, dist_matrix=dist_matrix)
        gmet = GraphMetrics(coords, dist_matrix=dist_matrix)

        lsd_h = ca.lsd_entropy(n_bins=30)
        lam2  = gmet.algebraic_connectivity()
        _, _, arf_grid, _ = gmet.arf()
        sll   = gmet.sll_db(arf_grid)
        graph_ent = gmet.graph_entropy()

        # Calculate target wavelength
        f_target = self.vs / (4 * self.focus)  # [Hz]
        lambda_target = self.vs / f_target     # [m]
        
        # Normalized λ₂ objective (maximize → minimize negative)
        lam2_norm = gmet.lambda2_normalized(lambda_target)
        
        # Dynamic Range
        dr_db = gmet.dynamic_range(arf_grid)
        
        # Resolution efficiency
        eta_res = gmet.resolution_efficiency(arf_grid)

        # ── Baseline objective ──
        f1_spatial = -lsd_h
        
        # ── Deep Focus Fixes ──
        r_max = ca.r_max
        r_min = ca.r_min

        # 1. Soft aperture penalty (if r_max > 2.5 * focus)
        aperture_limit = 2.5 * self.focus
        if r_max > aperture_limit:
            excess = (r_max - aperture_limit) / self.focus
            f1_spatial += 2.0 * np.log1p(excess)

        # 2. Isotropy penalty
        iso = gmet.isotropy_score()
        f1_spatial += 1.5 * (1.0 - iso)
        
        # 3. Near-pair density check
        n_pairs_active = len(ca.lags)
        if n_pairs_active > 0:
            near_frac = np.sum(ca.lags < 1.0 * self.focus) / n_pairs_active
            if near_frac < 0.20:
                f1_spatial += 2.0 * (0.20 - near_frac)

        # 4. Kernel fit
        # Ensure the GA respects the specified wavefield propagation model.
        # ADAPTIVE KERNEL: If directional noise is present, default to Hankel
        # because the wavefield is no longer perfectly isotropic.
        active_kernel = self.kernel
        if self.noise_azimuths:
            active_kernel = "hankel"
            
        if active_kernel == "hankel":
            phi = gmet.phi_hankel()
        else:
            phi = gmet.phi_bessel()
        f1_spatial += phi

        # Scale f3 down logarithmically to prevent objective dominance over SLL
        lam2_norm_scaled = np.log10(max(1e-9, lam2_norm + 1))
        
        # ── Multi-Objective Setup ──
        # Objective 1: Spatial/Geometric Quality
        # Objective 2: Spectral Quality
        w = self.weights
        
        obj_1 = (
            w.get('lsd', 1.0) * f1_spatial +
            w.get('lam2', 1.0) * (-lam2_norm_scaled) +
            w.get('graph_ent', 1.0) * (-graph_ent)
        )
        
        obj_2 = (
            w.get('sll', 1.0) * sll +
            w.get('dr', 1.0) * (-dr_db) +
            w.get('eta', 1.0) * (-eta_res)
        )
        
        # Save individual raw metrics for visualizer tracking (1D convergence plots)
        out["f1_lsd"]   = float(f1_spatial)
        out["f2_sll"]   = float(sll)
        out["f3_lam2"]  = float(lam2_norm)
        out["f4_graph"] = float(graph_ent)
        out["f5_dr"]    = float(dr_db)
        out["f6_eta"]   = float(eta_res)
        out["f7_ds"]    = 0.0

        # ── optional objectives (add them to obj_1 or obj_2 as appropriate) ──
        if self.noise_azimuths:
            ds = directional_score(coords, self.noise_azimuths)
            # Increase weight specifically for this scenario to enforce directional alignment
            obj_1 += 10.0 * (-ds)    # maximise directional alignment with a stronger weight
            out["f7_ds"] = float(ds)

        if self.los_flag:
            lp = los_penalty(coords, self.gm, n_samples=8)
            obj_1 += lp     # minimise LoS crossings

        if self.golomb_flag:
            gr = golomb_redundancy(coords)
            obj_1 += gr     # minimise redundancy

        if self.near_field:
            src = self.near_field_src if self.near_field_src is not None else (self.x_max / 2, 20.0)
            nf = near_field_phi(coords, src, depth=self.focus)
            # 50.0 çok büyüktü, 10.0 yapıyoruz ki diğer objective'lerle yarışabilsin
            obj_1 += 10.0 * nf

        out["F"] = [obj_1, obj_2]

        # ── constraints ──
        g = []
        # d_min and r_max constraints
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = dist_matrix[i, j] if dist_matrix is not None else np.linalg.norm(coords[j] - coords[i])
                g.append(self.d_min - r)
                if self.r_max is not None:
                    g.append(r - self.r_max)

        # NEW: Hard aperture constraint (r_max < 2.5 * focus)
        g.append(r_max - aperture_limit)

        has_obstacles = (self.gm.grid == 0).any()
        if has_obstacles:
            for i in range(self.N):
                xi, yi = coords[i]
                g.append(0.0 if self._is_in_domain(xi, yi) else 1.0)

        # ── near-field hard centroid anchor ──
        if self.near_field and self.near_field_src is not None:
            cx = np.mean(coords[:, 0])
            cy = np.mean(coords[:, 1])
            sx, sy = self.near_field_src[0], self.near_field_src[1]
            centroid_dist = np.sqrt((cx - sx)**2 + (cy - sy)**2)
            # G <= 0 means feasible, so: centroid_dist - tolerance <= 0
            # allow centroid within d_min radius of source XY
            g.append(centroid_dist - self.d_min * 2.0)

        out["G"] = g


# ── Pareto knee-point ─────────────────────────────────────────────────────────

def knee_point(F, weights=None):
    """Minimum-norm point in normalised objective space (utopia = origin)."""
    F_min  = F.min(axis=0)
    F_max  = F.max(axis=0)
    denom  = F_max - F_min
    denom[denom == 0] = 1.0
    F_norm = (F - F_min) / denom
    if weights is not None:
        F_norm = F_norm * np.asarray(weights)
    return int(np.argmin(np.linalg.norm(F_norm, axis=1)))


# ── Main optimizer interface ──────────────────────────────────────────────────

class NSGAOptimizer:
    SEED = 1249718046570

    def __init__(self, manifold_manager, n_sensors=10, d_min=5.0,
                 kernel="bessel",
                 noise_azimuths=None,
                 los_penalty_flag=False,
                 golomb_flag=False,
                 near_field=False,
                 near_field_source=None,
                 focus=30.0,
                 vs=500.0,
                 use_poincare=False,
                 use_seeding=True,
                 r_max=None,
                 pop_size=100, n_gen=200, seed=None, weights=None):

        self.gm               = manifold_manager
        self.N                = n_sensors
        self.d_min            = d_min
        self.kernel           = kernel
        self.noise_azimuths   = noise_azimuths or []
        self.los_flag         = los_penalty_flag
        self.golomb_flag      = golomb_flag
        self.near_field       = near_field
        self.near_field_src   = near_field_source
        self.focus            = focus
        self.vs               = vs
        self.use_poincare     = use_poincare
        self.pop_size         = pop_size
        self.use_seeding      = use_seeding
        self.r_max            = r_max
        print(f"[DEBUG NSGA-II] r_max constraint boundary set to: {self.r_max}")
        self.n_gen            = n_gen
        self.seed             = seed if seed is not None else self.SEED
        self.weights          = weights
        np.random.seed(self.seed % (2**31 - 1))

    def get_weight_array(self):
        """Constructs an ordered weight array exactly matching the F vector objectives."""
        return np.array([1.0, 1.0])

    def run(self, verbose=True, callback=None):
        problem = GraphSPACProblem(
            self.gm, self.N, self.d_min,
            kernel=self.kernel,
            noise_azimuths=self.noise_azimuths,
            los_penalty_flag=self.los_flag,
            golomb_flag=self.golomb_flag,
            near_field=self.near_field,
            near_field_source=self.near_field_src,
            focus=self.focus,
            vs=self.vs,
            use_poincare=self.use_poincare,
            r_max=self.r_max,
            weights=self.weights
        )

        if self.use_seeding:
            sampling = SpatialSeedingSampling(self.gm, self.N)
            print("[NSGA-II] Initialization: Topological Awareness (Spatial Seeding)")
        else:
            sampling = FloatRandomSampling()
            print("[NSGA-II] Initialization: Blind Euclidean (FloatRandomSampling)")

        # Link the Magnetic Repair
        repair_op = TopologicalRepair(self.gm, self.N) if self.use_seeding else None

        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=sampling,
            crossover=SBX(prob=0.9, eta=15, repair=repair_op),
            mutation=PM(eta=20, repair=repair_op),
            eliminate_duplicates=True,
        )

        kwargs = {
            "seed": self.seed % (2**31 - 1),
            "verbose": verbose,
            "save_history": True,
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
                    "[NSGA-II] No feasible solutions found. "
                    "Returning least-infeasible individual. "
                    "Try increasing --pop or --gens, or relaxing --dmin.",
                    RuntimeWarning
                )
                cv_vals = np.array([ind.CV[0] for ind in pop])
                best_idx = int(np.argmin(cv_vals))
                best_coords_raw = pop[best_idx].X.reshape(self.N, 2)
                pareto_X = np.array([pop[best_idx].X])
                pareto_F = np.array([pop[best_idx].F])
                
                # Assign best_coords to fix UnboundLocalError
                if self.use_poincare:
                    from geometry_transforms import apply_transforms
                    best_coords = apply_transforms(best_coords_raw, self.gm.x_max, self.gm.y_max, use_poincare=True)
                else:
                    best_coords = best_coords_raw
            else:
                raise RuntimeError(
                    "NSGA-II returned empty population. "
                    "Try increasing --pop or relaxing --dmin."
                )
        else:
            # Single objective res.X is 1D vector of optimal layout
            if getattr(pareto_X, "ndim", 1) == 1:
                best_coords_raw = pareto_X.reshape(self.N, 2)
                pareto_X = np.array([pareto_X])
                pareto_F = np.array([pareto_F])
            else:
                w_arr = self.get_weight_array()
                idx = knee_point(pareto_F, weights=w_arr)
                best_coords_raw = pareto_X[idx].reshape(self.N, 2)
            if self.use_poincare:
                from geometry_transforms import apply_transforms
                best_coords = apply_transforms(best_coords_raw, self.gm.x_max, self.gm.y_max, use_poincare=True)
            else:
                best_coords = best_coords_raw


        if verbose:
            if pareto_F.shape[1] == 1:
                print(f"\n[NSGA-II] Single-Objective Fitness: {pareto_F[0][0]:.4f}")
            else:
                print(f"\n[NSGA-II] Multi-Objective F1: {pareto_F[0][0]:.4f}, F2: {pareto_F[0][1]:.4f}")


        return best_coords, pareto_F, pareto_X, res


if __name__ == "__main__":
    from manifold_manager import ManifoldManager
    gm = ManifoldManager(default_grid_size=20, grid_spacing=5.0)
    opt = NSGAOptimizer(gm, n_sensors=6, d_min=5.0, kernel="bessel",
                        pop_size=30, n_gen=15)
    coords, pF, pX, _ = opt.run(verbose=True)
    print(coords)
