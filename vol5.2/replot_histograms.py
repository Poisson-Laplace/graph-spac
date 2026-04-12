import os
import sys
import numpy as np
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_golomb_figures import plot_pairwise_histogram

def get_nested_triangles_exact(N, r_outer=40.0):
    cx, cy = 100.0, 100.0
    nodes = []
    
    def add_triangle(r, angle_offset, points_per_edge):
        corners = [np.array([cx + r*np.cos(angle_offset + i*2*np.pi/3), 
                             cy + r*np.sin(angle_offset + i*2*np.pi/3)]) for i in range(3)]
        nodes.extend(corners)
        for i in range(3):
            p1 = corners[i]
            p2 = corners[(i+1)%3]
            for j in range(1, points_per_edge + 1):
                t = j / (points_per_edge + 1)
                nodes.append(p1 * (1 - t) + p2 * t)
    
    if N == 10:
        nodes.append(np.array([cx, cy]))
        add_triangle(r_outer, 0.0, 0)
        add_triangle(r_outer * 0.50, np.pi/3, 0)
        add_triangle(r_outer * 0.15, np.pi/6, 0)
        
    elif N == 30:
        add_triangle(r_outer, np.pi/2, 3) 
        add_triangle(r_outer * 0.5, -np.pi/2, 2)
        add_triangle(r_outer * 0.25, np.pi/2, 2)
        
    else: # N == 20
        add_triangle(r_outer, np.pi/2, 2) 
        add_triangle(r_outer * 0.5, -np.pi/2, 1) 
        add_triangle(r_outer * 0.25, np.pi/2, 0) 
        nodes.append(np.array([cx - 2.5, cy]))
        nodes.append(np.array([cx + 2.5, cy]))
        
    return np.array(nodes[:N])

def replot_histograms():
    base_out = "/home/ferat/Desktop/nevertrustclankers/graphspac/paperfigs/classical_array/golomb_v5.2"
    N_levels = [10, 20, 30]
    
    for N in N_levels:
        # 1. Classical Array
        try:
            baseline_coords = get_nested_triangles_exact(N, 40.0)
            plot_pairwise_histogram(baseline_coords, f"nested_triangles_{N}", base_out)
        except Exception as e:
            print(f"Error plotting nested {N}: {e}")
            
        # 2. Golomb Array
        metrics_path = os.path.join(base_out, f"golomb_{N}_metrics.txt")
        coords = []
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
            else:
                print(f"No coords found in {metrics_path}")
        else:
            print(f"File not found: {metrics_path}")

if __name__ == "__main__":
    replot_histograms()
    print("Replot completed!")
