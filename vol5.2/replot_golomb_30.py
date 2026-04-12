import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_golomb_figures import plot_pairwise_histogram
from classical_arrays import nested_triangle_array

def replot():
    base_out = "/home/ferat/Desktop/nevertrustclankers/graphspac/paperfigs/classical_array/golomb_v5.2"
    N = 30
    r_outer = 40.0
    
    # 1. Regenerate Nested Triangles N=30
    coords_nested = []
    _N = N
    n_rings = _N // 3
    d_min_val = 5.0
    
    if n_rings > 1:
        delta_r = (r_outer - d_min_val) / (n_rings - 1)
    else:
        delta_r = 0.0

    _r = r_outer
    angle_offset = 0.0
    
    while _N >= 3:
        for k in range(3):
            angle = angle_offset + k * (2 * np.pi / 3) + np.pi / 6
            x = 100.0 + _r * np.cos(angle)
            y = 100.0 + _r * np.sin(angle)
            coords_nested.append([x, y])
            
        _N -= 3
        _r -= delta_r
        angle_offset += np.pi / 6  # rotate 30 degrees for next layer
        
    if _N == 1:
        coords_nested.append([100.0, 100.0])
    elif _N == 2:
        coords_nested.append([100.0 - d_min_val, 100.0])
        coords_nested.append([100.0 + d_min_val, 100.0])
        
    baseline_coords = np.array(coords_nested)
    plot_pairwise_histogram(baseline_coords, f"nested_triangles_{N}", base_out)
    
    from main import compute_all_metrics
    from scenarios import SCENARIO_MAP
    from grid_generators import make_grid
    
    sc = SCENARIO_MAP[19].copy()
    sc["N"] = N
    sc["name"] = f"Nested Triangles classical N={N}"
    gm = make_grid(sc)
    
    metrics, arf_grid, k_max, ca = compute_all_metrics(baseline_coords, gm, sc)
    print(f"[REPLOT SLL CHECK] SLL dB for Nested Triangles N={N} is: {metrics['sll_db']}")
    print(f"[REPLOT LSD CHECK] LSD Entropy for Nested Triangles N={N} is: {metrics['lsd_H']}")
    
    # 2. Extract coords from golomb_30_metrics.txt
    metrics_path = os.path.join(base_out, "golomb_30_metrics.txt")
    coords = []
    import re
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            in_sensors = False
            for line in f:
                if "SENSOR POSITIONS" in line:
                    in_sensors = True
                    continue
                if in_sensors and "COMPUTED METRICS" in line:
                    break
                if in_sensors:
                    m = re.search(r'x\s*=\s*([\d\.\-]+)\s+y\s*=\s*([\d\.\-]+)', line)
                    if m:
                        coords.append((float(m.group(1)), float(m.group(2))))
                        
        if len(coords) > 0:
            plot_pairwise_histogram(np.array(coords), f"golomb_{N}", base_out)
            
if __name__ == "__main__":
    replot()
