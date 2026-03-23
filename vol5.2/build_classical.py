import os
import numpy as np
import sys

# Ensure we can import vol5 modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from manifold_manager import ManifoldManager
from graph_metrics import GraphMetrics
from visualizer import plot_scenario_result

base_dir = "/home/ferat/Desktop/nevertrustclankers/graphspac/classical_array"
os.makedirs(base_dir, exist_ok=True)

grid_size = 60
spacing = 5.0
center = (grid_size * spacing / 2.0, grid_size * spacing / 2.0)
radius = 100.0

def circle_points(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)])

def triangle_perimeter_points(side, n_pts_total, center_pt, angle_offset=0):
    if n_pts_total == 0:
        return np.empty((0, 2))
    h = side * np.sqrt(3) / 2
    v = np.array([
        [0, 2 * h / 3],
        [-side / 2, -h / 3],
        [side / 2, -h / 3],
    ])
    c, s = np.cos(angle_offset), np.sin(angle_offset)
    R = np.array([[c, -s], [s, c]])
    v = v @ R.T
    
    pts = []
    pts_per_edge = n_pts_total // 3
    for i in range(3):
        p1 = v[i]
        p2 = v[(i+1)%3]
        for j in range(pts_per_edge):
            t = j / pts_per_edge
            pts.append(p1 * (1-t) + p2 * t)
    
    rem = n_pts_total % 3
    rem_pts = []
    for i in range(rem):
        p1 = v[i]
        p2 = v[(i+1)%3]
        rem_pts.append((p1 + p2)/2)
    
    all_pts = pts + rem_pts
    return np.array(all_pts) + np.array(center_pt)

def nested_triangles(n):
    n_out = min(n, int(np.ceil(n * 2 / 3)))
    n_in = n - n_out
    out_pts = triangle_perimeter_points(radius * np.sqrt(3), n_out, center, 0)
    if n_in > 0:
        in_pts = triangle_perimeter_points((radius * 0.5) * np.sqrt(3), n_in, center, np.pi/3)
        return np.vstack([out_pts, in_pts])
    return out_pts

def l_shape(n):
    if n == 0: return np.empty((0,2))
    n_east = (n - 1) // 2 + (n - 1) % 2
    n_north = (n - 1) // 2
    pts = [list(center)]
    for i in range(1, n_east + 1):
        r = radius * i / n_east
        pts.append([center[0] + r, center[1]])
    for i in range(1, n_north + 1):
        r = radius * i / n_north
        pts.append([center[0], center[1] + r])
    return np.array(pts)

def cross_shape(n):
    if n == 0: return np.empty((0,2))
    arms = 4
    rem = (n - 1) % arms
    base = (n - 1) // arms
    
    n_per_arm = [base] * 4
    for i in range(rem):
        n_per_arm[i] += 1
        
    pts = [list(center)]
    directions = [0, np.pi/2, np.pi, 3*np.pi/2]
    for n_arm, direction in zip(n_per_arm, directions):
        if n_arm == 0: continue
        for i in range(1, n_arm + 1):
            r = radius * i / n_arm
            pts.append([center[0] + r * np.cos(direction), center[1] + r * np.sin(direction)])
    return np.array(pts)

def matrix_shape(n):
    if n == 0: return np.empty((0,2))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    r_cols = radius if cols > 1 else 0
    r_rows = radius if rows > 1 else 0
    
    pts = []
    x_steps = np.linspace(-r_cols, r_cols, cols) if cols > 1 else [0]
    y_steps = np.linspace(-r_rows, r_rows, rows) if rows > 1 else [0]
    
    for y in y_steps:
        for x in x_steps:
            pts.append([center[0] + x, center[1] + y])
            if len(pts) == n:
                return np.array(pts)
    return np.array(pts)

shapes = {
    "Circular Array": circle_points,
    "Nested Triangles": nested_triangles,
    "L-Shape": l_shape,
    "Cross_Cruciform": cross_shape
}

from main import save_metrics_txt, compute_all_metrics
import warnings
warnings.filterwarnings("ignore")

for shape_name, func in shapes.items():
    for n_sensors in [10, 20, 30]:
        friendly_name = shape_name
        out_dir = os.path.join(base_dir, friendly_name, f"{n_sensors}_sensors")
        os.makedirs(out_dir, exist_ok=True)
        
        coords = func(n_sensors)
        
        sc = {
            "id": 0,
            "name": f"{friendly_name} {n_sensors}",
            "domain": "open",
            "gridsize": grid_size,
            "spacing": spacing,
            "weights": {"f1_lsd": -1, "f2_sll": -1},
            "focus": 30.0,
            "kernel": "bessel",
            "N": n_sensors,
            "d_min": 5.0,
            "vs": 300.0
        }
        gm = ManifoldManager(default_grid_size=sc["gridsize"], grid_spacing=sc["spacing"])

        met, arf_grid, k_max, ca = compute_all_metrics(coords, gm, sc)
        
        filepath = os.path.join(out_dir, "result.png")
        print(f"Generating for {friendly_name} with {n_sensors} sensors...")
        plot_scenario_result(coords, gm, met, sc, filename=filepath)
        
        txtpath = os.path.join(out_dir, "metrics.txt")
        save_metrics_txt(met, sc, txtpath, coords=coords)

print("Done generating all classical array figures and metrics!")
