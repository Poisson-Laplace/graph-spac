import numpy as np
import argparse
import os
import sys

from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grid_generators import make_grid
from scenarios import SCENARIO_MAP
from nsga_optimizer import GraphSPACProblem, knee_point
from main import compute_all_metrics, save_metrics_txt
from visualizer import plot_scenario_result


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
            
            # Add spatial jitter (micro-mutations) so they don't sit perfectly on grid centers
            noise = np.random.uniform(-self.gm.grid_spacing/3, self.gm.grid_spacing/3, (self.N, 2))
            pts += noise
            
            # Constrain to strict boundary limits
            pts[:, 0] = np.clip(pts[:, 0], 0, self.gm.x_max)
            pts[:, 1] = np.clip(pts[:, 1], 0, self.gm.y_max)
            
            X[i, :] = pts.flatten()
        return X


class GenerationCallback(Callback):
    def __init__(self, out_dir, gm, sc):
        super().__init__()
        self.out_dir = out_dir
        self.gm = gm
        self.sc = sc
        self.weights_array = np.array([1.0, 1.0])
        self.saved_gen0 = False
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[Frames] Saving generation snapshots to: {self.out_dir}")

    def notify(self, algorithm):
        pop = algorithm.pop
        if pop is None or len(pop) == 0:
            return
            
        gen = algorithm.n_gen
        
        # Capture Generation 0 (initial population before selection) if we just started
        if gen == 1 and not self.saved_gen0:
            if hasattr(algorithm, 'initialization') and hasattr(algorithm.initialization, 'pop'):
                self._save_frame(algorithm.initialization.pop, 0)
            self.saved_gen0 = True
            
        self._save_frame(pop, gen)

    def _save_frame(self, pop, gen):
        if pop is None or len(pop) == 0:
            return
            
        F = np.array([ind.F for ind in pop])
        X = np.array([ind.X for ind in pop])
        
        if hasattr(pop[0], 'CV') and pop[0].CV is not None:
            CV = np.array([ind.CV[0] for ind in pop])
            feas_idx = np.where(CV <= 0)[0]
        else:
            feas_idx = np.arange(len(pop))
            CV = np.zeros(len(pop))
            
        if len(feas_idx) > 0:
            idx = feas_idx[knee_point(F[feas_idx], weights=self.weights_array)]
        else:
            idx = int(np.argmin(CV))
            
        best_coords = X[idx].reshape(self.sc['N'], 2)
        
        frame_sc = self.sc.copy()
        frame_sc["name"] = f"{self.sc['name']} // Gen: {gen:04d}"
        
        metrics, arf_grid, k_max, ca = compute_all_metrics(best_coords, self.gm, frame_sc)
        filename = os.path.join(self.out_dir, f"gen_{gen:04d}.png")
        plot_scenario_result(best_coords, self.gm, metrics, frame_sc, filename=filename)
        
        # If this is the final generation, also save layout_final.png and metrics_final.txt
        if gen == self.sc["generations"] or gen == 0:
            if gen == self.sc["generations"]:
                plot_scenario_result(best_coords, self.gm, metrics, self.sc, filename=os.path.join(self.out_dir, f"layout_final.png"))
                save_metrics_txt(metrics, self.sc, os.path.join(self.out_dir, f"metrics_final.txt"), coords=best_coords)


def test_scenario(sc_id, m_name, use_seeding, args, base_out):
    print(f"\n==========================================")
    print(f" Running {m_name} on Scenario {sc_id}")
    print(f"==========================================")
    
    sc = SCENARIO_MAP[sc_id].copy()
    if sc_id == 6:
        sc["noise_azimuths"] = [90.0]
        
    sc["generations"] = args.gens
    gm = make_grid(sc)
    
    sc_m = sc.copy()
    sc_m["name"] = f"Sc{sc_id} | {m_name} (Pop: {args.pop}, Gen: {args.gens})"
    out_dir = os.path.join(base_out, f"sc{sc_id}_{m_name.replace(' ', '_').lower()}")
    
    problem = GraphSPACProblem(
        gm, sc_m["N"], sc_m["d_min"],
        kernel=sc_m["kernel"], noise_azimuths=sc_m["noise_azimuths"],
        focus=sc_m["focus"], vs=sc_m["vs"], weights=sc_m["weights"]
    )
    
    if use_seeding:
        # Topology-aware initialization
        sampling = SpatialSeedingSampling(gm, sc_m["N"])
    else:
        # Blind Euclidean initialization
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        sampling = FloatRandomSampling()
        
    algorithm = NSGA2(
        pop_size=args.pop,
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    callback = GenerationCallback(out_dir, gm, sc_m)
    
    print("Optimizing...")
    res = minimize(
        problem, algorithm,
        ("n_gen", args.gens),
        seed=1249718046570,
        callback=callback,
        verbose=False
    )
    print(f"[{m_name}] Finished. Outputs saved strictly to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test Spatial Seeding in Topologically Porous Domains")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    args = parser.parse_args()

    base_out = os.path.dirname(os.path.abspath(__file__))
    
    # ── Test on Scenario 26: The Sponge (Sünger Gibi Saha) ──
    # Compare Blind Euclidean (Baseline) vs Topological Awareness (Spatial Seeding)
    test_scenario(26, "Baseline", False, args, base_out)
    test_scenario(26, "Topological Awareness", True, args, base_out)

if __name__ == '__main__':
    main()
