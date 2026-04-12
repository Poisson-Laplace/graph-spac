import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import argparse
from skimage import measure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios import SCENARIO_MAP
from grid_generators import make_grid

def compute_banana_doughnut_kernel(x, y, z, p1, p2, freq=2.0, vs=500.0, focus=30.0):
    R_earth = float(np.linalg.norm(p1 - p2))
    
    lam = vs / freq
    
    d1 = np.sqrt((x - p1[0])**2 + (y - p1[1])**2 + z**2)
    d2 = np.sqrt((x - p2[0])**2 + (y - p2[1])**2 + z**2)
    
    delta_L = (d1 + d2) - R_earth
    
    # Dynamic expansion and depth constraint based on scenario focus
    sigma = 0.5 * lam * (abs(z) / focus + 1.0)
    sensitivity = np.exp(-(delta_L**2) / (2 * sigma**2))
    
    doughnut_hole = 1.0 - np.exp(-(z**2) / (0.1 * lam**2))
    depth_focus = np.exp(-((z + focus)**2) / (2 * (focus / 2.0)**2)) 
    
    return sensitivity * doughnut_hole * depth_focus


def plot_3d_isosurface(coords, sc, gm, out_dir):
    sc_id = sc["id"]
    is_lens_effect = sc.get("los_penalty", False)
    is_near_field  = sc.get("near_field", False)
    
    print(f"[3D Map] Extracting precise Iso-surface mesh for Sc {sc_id}...")
    
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    # 3D Grid definition
    X, Y, Z = np.mgrid[0:gm.x_max:40j, 0:gm.y_max:40j, 0:-60:40j]
    V = np.zeros_like(X)
    
    # Compute superimposed FIM kernels dynamically
    for p1, p2 in combinations(coords, 2):
        if np.linalg.norm(p1 - p2) > 5.0:
            V += compute_banana_doughnut_kernel(X, Y, Z, p1, p2, vs=sc["vs"], focus=sc["focus"])
            
    # Normalize Volume to [0.0, 1.0]
    V_max = np.max(V)
    if V_max > 0:
        V = V / V_max
    
    # Adaptive threshold depending on scenario type
    threshold = 0.90 if is_near_field else 0.55
    mesh_color = 'dodgerblue' if is_near_field else 'crimson'
    
    # Extract Iso-surface via Marching Cubes
    while threshold > 0.1:
        try:
            verts, faces, normals, values = measure.marching_cubes(V, level=threshold, spacing=(gm.x_max/40, gm.y_max/40, 60/40))
            verts[:, 2] = -verts[:, 2] 
            
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                            color=mesh_color, lw=0, alpha=0.35, antialiased=True)
            print(f"[3D Map] Generated Iso-surface at intensity threshold: {threshold*100:.0f}%")
            break
        except ValueError:
            threshold -= 0.05
    
    # Draw Obstacle IF Scenario 18 (Lens Effect / LoS)
    if is_lens_effect:
        cx, cy = gm.x_max / 2.0, gm.y_max / 2.0
        r_obs = gm.x_max * 0.15
        
        theta = np.linspace(0, 2 * np.pi, 50)
        z_cyl = np.linspace(0, -60, 2)
        Theta, Zc = np.meshgrid(theta, z_cyl)
        Xc = cx + r_obs * np.cos(Theta)
        Yc = cy + r_obs * np.sin(Theta)
        ax.plot_surface(Xc, Yc, Zc, color='gray', alpha=0.15, linewidth=0, antialiased=True)
    
    # Target Point Source IF Scenario 20 (Near Field)
    if is_near_field and "near_field_source" in sc and sc["near_field_source"] is not None:
        sx, sy = sc["near_field_source"]
        sz = -sc["focus"]
        print(f"[3D Map] Setting Deep Focal Noise Source (Red Sphere) absolutely positioned at X={sx:.1f}, Y={sy:.1f}, Z={sz:.1f}")
        
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        r_sphere = 2.0
        Xs = sx + r_sphere * np.outer(np.cos(u), np.sin(v))
        Ys = sy + r_sphere * np.outer(np.sin(u), np.sin(v))
        Zs = sz + r_sphere * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(Xs, Ys, Zs, color='red', alpha=0.9, linewidth=0, antialiased=True)
        
        # Ray-paths from sensors to the red sphere
        for p in coords:
            ax.plot([p[0], sx], [p[1], sy], [0, sz], color='crimson', alpha=0.3, linestyle='-', linewidth=1.5)
            
    # Surface connecting geodesics IF Scenario 18
    if is_lens_effect:
        for p1, p2 in combinations(coords, 2):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0], color='black', alpha=0.15, linestyle=':')

    # Plot array sensors (z=0)
    ax.scatter(coords[:, 0], coords[:, 1], 0, color='forestgreen', marker='^', s=120, edgecolors='black', depthshade=False, label="Array Sensors")
    
    ax.set_xlabel("East-West Coordinate [m]", color='black', labelpad=10)
    ax.set_ylabel("North-South Coordinate [m]", color='black', labelpad=10)
    ax.set_zlabel("Depth [m]", color='black', labelpad=10)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis='z', colors='black')
    ax.grid(True, color='gray', linestyle=':', alpha=0.4)
    
    # Academic view angles
    view_elev = 20 if is_near_field else 15
    view_azim = -45 if is_near_field else -60
    ax.view_init(elev=view_elev, azim=view_azim)
    
    mode_name = "nearfield" if is_near_field else "lens_effect"
    out_file = os.path.join(out_dir, f"sc{sc_id}_3D_{mode_name}_coupling.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=400, facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"[3D Map] Successfully rendered true Iso-surface to -> {out_file}")


def extract_coords_from_metrics(filepath):
    coords = []
    if not os.path.exists(filepath):
        print(f"Error: Required coordinates file not found at '{filepath}'.")
        print("Please run this scenario in GraphSPAC main.py first to generate the metrics.")
        sys.exit(1)
        
    with open(filepath, 'r') as f:
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
    
    coords = np.array(coords)
    if len(coords) < 3:
        print("Error: Could not extract valid sensors from the metrics file. Regex parsing failed.")
        sys.exit(1)
        
    return coords


def generate_lens_effect(sc_id):
    base_out = "/home/ferat/Desktop/nevertrustclankers/graphspac/vol5.2/3dmap"
    os.makedirs(base_out, exist_ok=True)
    
    if sc_id not in SCENARIO_MAP:
        print(f"Error: Scenario {sc_id} does not exist.")
        sys.exit(1)
        
    results_dir = "/home/ferat/Desktop/nevertrustclankers/graphspac/vol5.2/results"
    
    possible_paths = [
        os.path.join(results_dir, f"sc{sc_id}_metrics.txt"),
    ]
    
    if os.path.isdir(os.path.join(results_dir, str(sc_id))):
        possible_paths.append(os.path.join(results_dir, str(sc_id), f"sc{sc_id}_metrics.txt"))
        
    target_file = None
    for p in possible_paths:
        if os.path.exists(p):
            target_file = p
            break
            
    if not target_file:
        for root, dirs, files in os.walk(results_dir):
            if f"sc{sc_id}_metrics.txt" in files:
                target_file = os.path.join(root, f"sc{sc_id}_metrics.txt")
                break
                
    if not target_file:
        target_file = possible_paths[0] 
        
    print("============================================================")
    print(f" GraphSPAC vol5.2 — 3D Iso-Surface Rendering Engine ")
    print(f" Reading static coords from Scenario {sc_id}...")
    print("============================================================")
    
    coords = extract_coords_from_metrics(target_file)
    print(f"[Parser] Extracted {len(coords)} coordinates.")
    
    sc = SCENARIO_MAP[sc_id].copy()
    gm = make_grid(sc)
    
    plot_3d_isosurface(coords, sc, gm, base_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 3D Isosurface of FIM from existing coordinates")
    parser.add_argument("--scenario", type=int, default=18, help="Scenario ID (e.g. 18 for Lens, 20 for Near-Field)")
    args = parser.parse_args()
    
    generate_lens_effect(sc_id=args.scenario)
