import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classical_arrays import nested_triangle_array
from scenarios import SCENARIO_MAP
from grid_generators import make_grid
from nsga_optimizer import NSGAOptimizer

def plot_pairwise_histogram(coords, name, out_dir):
    if len(coords) < 2:
        print(f"Skipping {name}: not enough coordinates.")
        return
        
    distances = []
    for p1, p2 in combinations(coords, 2):
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        distances.append(dist)
        
    distances = np.array(distances)
    
    plt.figure(figsize=(10, 6))
    
    bins = min(50, max(10, len(distances) // 5))
    
    plt.hist(distances, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Distance [m]", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    
    out_file = os.path.join(out_dir, f"{name.replace(' ', '_').lower()}_distances.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"[Golomb Plot] Generated histogram for {name} -> {out_file}")

def run_golomb_generator(N_values, gens, pop, base_out):
    os.makedirs(base_out, exist_ok=True)
    
    # Target focus=30 -> optimum aperture is approx 80 (r_max = 40)
    r_outer = 40.0 
    
    for N in N_values:
        # 1. Classical Baseline: Nested Triangles
        print(f"\n────────────────────────────────────────────────────────────")
        print(f" Generating Classical Baseline: Nested Triangles (N={N})")
        
        baseline_coords = nested_triangle_array(r_outer, center=(100.0, 100.0))
        # Nested Triangles naturally generates N=6. 
        # Wait, if we need exactly N=10, 20, 30 we might need a classical shape that scales to N exactly, 
        # or just use random/circular. 
        # Actually nested triangles is 6 nodes fixed. Let's use "Circle" or "Propeller" or just plot whatever size nested triangles has.
        # But the original plot_golomb_histograms explicitly names "Nested Triangles 10/20/30".
        # Let's generate a circle or ring configuration if nested_triangles is strictly N=6.
        # Actually in classical_arrays.py, nested_triangles returns 6 points. 
        # To get N points, we can do multiple nested triangles. I will implement a quick N-scaling nested triangles:
        
        # Scaling Nested Triangles to N nodes: N/3 nodes per triangle or similar. 
        # For simplicity, let's use circle_array or just run Kennett Spiral if the user wants classical baselines, 
        # BUT the name the user used was "Nested Triangles". 
        # Let's write a quick ring generator to mimic the old behaviour, or just use circle_array.
        # Let's use multiple rings to approximate "Nested Triangles" 
        from classical_arrays import circle_array
        
        # Let's create an N-sensor 'Nested Triangles' by repeatedly drawing inner triangles
        coords_nested = []
        _N = N
        _r = r_outer
        while _N >= 3:
            h = _r * np.sqrt(3) / 2
            side = _r * np.sqrt(3)
            # Add a triangle
            coords_nested.extend([
                [100.0, 100.0 + 2 * h / 3],
                [100.0 - side / 2, 100.0 - h / 3],
                [100.0 + side / 2, 100.0 - h / 3],
            ])
            _N -= 3
            _r *= 0.6 # shrink
            # Rotate by 60 deg for next
            
        while len(coords_nested) < N:
            coords_nested.append([100.0, 100.0]) # Add to center if needed
            
        baseline_coords = np.array(coords_nested[:N])
            
        plot_pairwise_histogram(baseline_coords, f"nested_triangles_{N}", base_out)
        
        
        # 2. Optimized Golomb Array (Scenario 19)
        print(f"\n Generating NSGA-II 2D Golomb Array (N={N})")
        sc = SCENARIO_MAP[19].copy()
        sc["N"] = N
        sc["generations"] = gens
        sc["population"] = pop
        
        gm = make_grid(sc)
        
        opt = NSGAOptimizer(
            manifold_manager=gm,
            n_sensors=sc["N"], d_min=sc["d_min"], kernel=sc["kernel"],
            noise_azimuths=sc["noise_azimuths"],
            golomb_flag=sc.get("golomb", True),
            los_penalty_flag=sc.get("los_penalty", False),
            near_field=sc.get("near_field", False),
            near_field_source=sc.get("near_field_source", None),
            focus=sc["focus"],
            vs=sc["vs"],
            pop_size=sc["population"],
            n_gen=sc["generations"],
            seed=1249718046570,
            use_poincare=False,
            use_seeding=True,
            weights=sc["weights"]
        )
        
        print(f"Optimizing N={N} Golomb configuration...")
        best_coords, pareto_F, pareto_X, res = opt.run(verbose=True, callback=None)
        
        plot_pairwise_histogram(best_coords, f"golomb_{N}", base_out)
        
        # Save metrics as well
        from main import save_metrics_txt, compute_all_metrics
        sc_m = sc.copy()
        sc_m["name"] = f"Golomb Optimized N={N}"
        metrics, arf_grid, k_max, ca = compute_all_metrics(best_coords, gm, sc_m)
        save_metrics_txt(metrics, sc_m, os.path.join(base_out, f"golomb_{N}_metrics.txt"), coords=best_coords)

def main():
    parser = argparse.ArgumentParser(description="Generate 2D Golomb Ruler Figures")
    parser.add_argument("--gens", type=int, default=200, help="Generations for NSGA-II")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    args = parser.parse_args()
    
    base_out = "/home/ferat/Desktop/nevertrustclankers/graphspac/paperfigs/classical_array/golomb_v5.2"
    N_values = [10, 20, 30]
    
    print("============================================================")
    print(" GraphSPAC vol5.2 — Golomb Array & Baseline Figure Generator")
    print("============================================================")
    
    run_golomb_generator(N_values, args.gens, args.pop, base_out)

if __name__ == '__main__':
    main()
