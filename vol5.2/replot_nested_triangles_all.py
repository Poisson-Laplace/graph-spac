import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import save_metrics_txt, compute_all_metrics
from visualizer import plot_scenario_result
from scenarios import SCENARIO_MAP
from grid_generators import make_grid

def get_nested_triangles(N, r_outer=40.0):
    cx, cy = 75.0, 75.0
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
        # Sensör 1: merkez
        nodes.append(np.array([cx, cy]))
        
        # Sensör 2, 3, 4: dış üçgen köşeleri (yukarı bakan, np.pi/2 = kuzey)
        outer_corners = [np.array([cx + r_outer*np.cos(np.pi/2 + i*2*np.pi/3),
                                   cy + r_outer*np.sin(np.pi/2 + i*2*np.pi/3)]) for i in range(3)]
        nodes.extend(outer_corners)  # s2, s3, s4
        
        # Sensör 5, 6, 7: orta ters üçgen köşeleri (aşağı bakan, -np.pi/2)
        mid_r = r_outer * 0.5
        inner_corners = [np.array([cx + mid_r*np.cos(-np.pi/2 + i*2*np.pi/3),
                                   cy + mid_r*np.sin(-np.pi/2 + i*2*np.pi/3)]) for i in range(3)]
        nodes.extend(inner_corners)  # s5, s6, s7
        
        # Sensör 9: 5→6 arası orta nokta (s5=nodes[4], s6=nodes[5])
        nodes.append((inner_corners[0] + inner_corners[1]) / 2.0)
        # Sensör 10: 6→7 arası orta nokta (s6=nodes[5], s7=nodes[6])
        nodes.append((inner_corners[1] + inner_corners[2]) / 2.0)
        # Sensör 8: 7→5 arası orta nokta (s7=nodes[6], s5=nodes[4])
        nodes.append((inner_corners[2] + inner_corners[0]) / 2.0)
        
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

def regenerate_all():
    base_dir = "/home/ferat/Desktop/nevertrustclankers/graphspac/paperfigs/classical_array/Nested Triangles"
    levels = [10, 20, 30]
    
    sc = SCENARIO_MAP[19].copy()
    sc["golomb"] = False
    
    for N in levels:
        print(f"\\n--- Generating Classical Nested Triangles N={N} ---")
        out_folder = os.path.join(base_dir, f"{N}_sensors")
        os.makedirs(out_folder, exist_ok=True)
        
        sc_n = sc.copy()
        sc_n["N"] = N
        sc_n["name"] = f"Classical Nested Triangles N={N}"
        
        gm = make_grid(sc_n)
        coords = get_nested_triangles(N, r_outer=40.0)
        
        print("Computing metrics and ARF...")
        metrics, arf_grid, k_max, ca = compute_all_metrics(coords, gm, sc_n)
        
        metrics_file = os.path.join(out_folder, f"nested_triangles_metrics_{N}.txt")
        save_metrics_txt(metrics, sc_n, metrics_file, coords=coords)
        
        print(f"LSD_H: {metrics['lsd_H']:.3f} | SLL: {metrics['sll_db']:.2f} dB")
        
        print(f"Saving 5 PDF visuals to {out_folder} ...")
        # Generate exact a, b, c, d, e figures as GraphSPAC does
        plot_scenario_result(coords, gm, metrics, sc_n, filename=os.path.join(out_folder, f"nested_triangles_N{N}.png"))

if __name__ == "__main__":
    regenerate_all()
