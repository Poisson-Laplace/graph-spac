import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import save_metrics_txt, compute_all_metrics
from visualizer import plot_scenario_result
from scenarios import SCENARIO_MAP
from grid_generators import make_grid

def get_concentric(N=31, cx=75.0, cy=75.0):
    nodes = [[cx, cy]]
    # 10 rings of 3 nodes, R shrunk by 3
    radii = np.linspace(5.0, 50.0, 10)
    for r in radii:
        for angle in [0, 120, 240]:
            rad = np.deg2rad(angle)
            nodes.append([cx + r * np.cos(rad), cy + r * np.sin(rad)])
    return np.array(nodes)

def get_concentric_10(N=10, cx=75.0, cy=75.0):
    nodes = [[cx, cy]]
    # 3 rings of 3 nodes, max R=50
    radii = np.linspace(50.0/3, 50.0, 3)
    for r in radii:
        for angle in [0, 120, 240]:
            rad = np.deg2rad(angle)
            nodes.append([cx + r * np.cos(rad), cy + r * np.sin(rad)])
    return np.array(nodes)

def get_dominant_fk(N=31, cx=75.0, cy=75.0):
    nodes = [[cx, cy]]
    
    def add_ring(r, num_sensors, offset_deg):
        offset = np.deg2rad(offset_deg)
        for i in range(num_sensors):
            angle = offset + i * 2 * np.pi / num_sensors
            nodes.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
            
    # Inner Ring: R=5m (15/3), 6 sensors, 0 deg
    add_ring(5.0, 6, 0.0)
    # Middle Ring: R=13.333m (40/3), 6 sensors, 30 deg
    add_ring(40.0 / 3.0, 6, 30.0)
    # Large Ring: R=26.666m (80/3), 9 sensors, 15 deg
    add_ring(80.0 / 3.0, 9, 15.0)
    # Outer Perimeter: R=50m (150/3), 9 sensors, 10 deg
    add_ring(50.0, 9, 10.0)
    
    return np.array(nodes)

def get_star_shaped(N=31, cx=75.0, cy=75.0):
    nodes = [[cx, cy]]
    R0 = 75.0 / np.sqrt(3)  # ~43.30127 m
    
    for k in range(10):  # 10 layers, each providing 3 corners = 30 + 1 center = 31 total sensors
        R_k = R0 * (0.5 ** k)
        phi_k = np.pi/2 if k % 2 == 0 else -np.pi/2
        
        for i in range(3):
            angle = phi_k + i * 2 * np.pi / 3
            nodes.append([cx + R_k * np.cos(angle), cy + R_k * np.sin(angle)])
            
    return np.array(nodes)


def generate_all():
    base_dir = "/home/ferat/Desktop/nevertrustclankers/graphspac/paperfigs/classical_array/Nested Triangles"
    
    configs = [
        {"name": "Configuration_1_Concentric", "N": 31, "func": get_concentric},
        {"name": "Configuration_1_Concentric_10", "N": 10, "func": get_concentric_10},
        {"name": "Configuration_2_Dominant", "N": 31, "func": get_dominant_fk},
        {"name": "Configuration_3_Star", "N": 31, "func": get_star_shaped},
    ]
    
    sc = SCENARIO_MAP[19].copy()
    sc["golomb"] = False
    # Revert grid to standard 150m x 150m size since Rmax is now 50m.
    sc["gridsize"] = 30  # 30 * 5.0m = 150m
    
    for cfg in configs:
        name = cfg["name"]
        n_sensors = cfg.get("N", 31)
        print(f"\\n--- Generating {name} with N={n_sensors} ---")
        out_folder = os.path.join(base_dir, name)
        os.makedirs(out_folder, exist_ok=True)
        
        sc_n = sc.copy()
        sc_n["name"] = name
        sc_n["N"] = n_sensors
        
        gm = make_grid(sc_n)
        # Bounding box is 0 to 150. Center is 75.0.
        coords = cfg["func"](N=n_sensors, cx=75.0, cy=75.0)
        
        print("Computing metrics and ARF...")
        metrics, arf_grid, k_max, ca = compute_all_metrics(coords, gm, sc_n)
        
        metrics_file = os.path.join(out_folder, f"{name}_metrics.txt")
        save_metrics_txt(metrics, sc_n, metrics_file, coords=coords)
        
        print(f"LSD_H: {metrics['lsd_H']:.3f} | SLL: {metrics['sll_db']:.2f} dB")
        
        print(f"Saving 5 PDF visuals to {out_folder} ...")
        plot_scenario_result(coords, gm, metrics, sc_n, filename=os.path.join(out_folder, f"{name}.png"))

if __name__ == "__main__":
    generate_all()
