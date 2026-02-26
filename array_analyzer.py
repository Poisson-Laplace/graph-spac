# author: ferat
# date: 2026
import numpy as np
import matplotlib.pyplot as plt
from spac_graph import SPACGraph, SPACResponseAnalyzer
import argparse

def create_circle_array(radius, n_sensors, center=(0, 0)):
    
    angles = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
    coords = [(center[0] + radius * np.cos(a), 
               center[1] + radius * np.sin(a)) for a in angles]
    return coords

def create_cross_array(arm_length, n_per_arm=3, center=(0, 0)):
    
    coords = [center]  # center sensor
    
    for direction in [0, np.pi/2, np.pi, 3*np.pi/2]:  # 4 directions
        for i in range(1, n_per_arm + 1):
            r = arm_length * i / n_per_arm
            x = center[0] + r * np.cos(direction)
            y = center[1] + r * np.sin(direction)
            coords.append((x, y))
    
    return coords

def create_x_array(arm_length, n_per_arm=3, center=(0, 0)):
    
    coords = [center]  # center sensor
    
    for direction in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:  # diagonals
        for i in range(1, n_per_arm + 1):
            r = arm_length * i / n_per_arm
            x = center[0] + r * np.cos(direction)
            y = center[1] + r * np.sin(direction)
            coords.append((x, y))
    
    return coords

def create_triangle_array(side_length, center=(0, 0)):
    
    # vertices of equilateral triangle centered at origin
    h = side_length * np.sqrt(3) / 2  # height
    coords = [
        (center[0], center[1] + h * 2/3),           # top
        (center[0] - side_length/2, center[1] - h/3),  # bottom-left
        (center[0] + side_length/2, center[1] - h/3),  # bottom-right
    ]
    return coords

def create_triangle_center_array(side_length, center=(0, 0)):
    
    coords = create_triangle_array(side_length, center)
    coords.append(center)  # add center point
    return coords

def create_nested_triangle_array(outer_side, inner_side, center=(0, 0)):
    
    outer = create_triangle_array(outer_side, center)
    inner = create_triangle_array(inner_side, center)
    return outer + inner

def create_l_array(arm_length, n_per_arm=4, center=(0, 0)):
    
    coords = [center]
    
    # horizontal arm (East)
    for i in range(1, n_per_arm + 1):
        x = center[0] + arm_length * i / n_per_arm
        coords.append((x, center[1]))
    
    # vertical arm (North)
    for i in range(1, n_per_arm + 1):
        y = center[1] + arm_length * i / n_per_arm
        coords.append((center[0], y))
    
    return coords

def create_fan_array(radius, n_sensors, arc_degrees=180, center=(0, 0)):
    
    arc_rad = np.radians(arc_degrees)
    start_angle = -arc_rad / 2
    
    if n_sensors > 1:
        angles = np.linspace(start_angle, start_angle + arc_rad, n_sensors)
    else:
        angles = [0]
    
    coords = [(center[0] + radius * np.cos(a), 
               center[1] + radius * np.sin(a)) for a in angles]
    return coords

def create_propeller_array(n_blades=3, sensors_per_blade=3, blade_length=50, center=(0, 0)):
    
    coords = [center]
    
    blade_angles = np.linspace(0, 2*np.pi, n_blades, endpoint=False)
    
    for angle in blade_angles:
        for i in range(1, sensors_per_blade + 1):
            r = blade_length * i / sensors_per_blade
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            coords.append((x, y))
    
    return coords

def create_spiral_arm_array(aperture, n_arms=3, n_rings=4, span_deg=120,
                             log_spacing=False, center=(0, 0)):
    
    coords = [center]  # central station
    span_rad = np.radians(span_deg)
    
    for k in range(n_arms):
        theta_0k = 2 * np.pi * k / n_arms  # initial angle for arm k
        
        for j in range(1, n_rings + 1):  # ring index 1..n_rings
            # radius
            if log_spacing:
                # logarithmic: r_j = aperture * (exp(j/n_rings) - 1) / (e - 1)
                r_j = aperture * (np.exp(j / n_rings) - 1) / (np.e - 1)
            else:
                r_j = aperture * j / n_rings  # linear
            
            # azimuth — Eq. 7 in Kennett et al. 2015
            theta_jk = theta_0k + span_rad * j / n_rings
            
            x = center[0] + r_j * np.cos(theta_jk)
            y = center[1] + r_j * np.sin(theta_jk)
            coords.append((x, y))
    
    return coords

def create_random_array(n_sensors, max_radius, center=(0, 0), seed=42):
    
    np.random.seed(seed)
    coords = []
    
    for _ in range(n_sensors):
        r = max_radius * np.sqrt(np.random.random())  # uniform in area
        theta = np.random.random() * 2 * np.pi
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        coords.append((x, y))
    
    return coords

def create_asymmetric_fan_array(center=(0, 0)):
    
    coords = []
    
    # phi=0° (East direction) - r from 2.5 to 62.5 with 5m steps
    phi_0_radii = np.arange(2.5, 62.5 + 0.1, 5)  # [2.5, 7.5, 12.5, ..., 62.5]
    for r in phi_0_radii:
        x = center[0] + r * np.cos(np.radians(0))
        y = center[1] + r * np.sin(np.radians(0))
        coords.append((x, y))
    
    # phi=180° (West direction) - same as phi=0
    for r in phi_0_radii:
        x = center[0] + r * np.cos(np.radians(180))
        y = center[1] + r * np.sin(np.radians(180))
        coords.append((x, y))
    
    # phi=45°, 90°, 135° - specific radii
    other_radii = [5, 10, 20, 35, 50, 70, 90, 120]
    
    for phi in [45, 90, 135]:
        for r in other_radii:
            x = center[0] + r * np.cos(np.radians(phi))
            y = center[1] + r * np.sin(np.radians(phi))
            coords.append((x, y))
    
    return coords

def create_symmetric_fan_array(center=(0, 0)):
    
    coords = []
    
    # phi=0° and 180° - denser sampling at shorter distances
    phi_0_radii = np.arange(2.5, 62.5 + 0.1, 5)  # 13 sensors
    
    for r in phi_0_radii:
        coords.append((center[0] + r, center[1]))  # 0°
        coords.append((center[0] - r, center[1]))  # 180°
    
    # other 6 directions - sparser sampling at longer distances  
    other_radii = [5, 10, 20, 35, 50, 70, 90, 120]  # 8 sensors
    
    for phi_deg in [45, 90, 135, 225, 270, 315]:
        phi_rad = np.radians(phi_deg)
        for r in other_radii:
            x = center[0] + r * np.cos(phi_rad)
            y = center[1] + r * np.sin(phi_rad)
            coords.append((x, y))
    
    return coords

def analyze_array(coords, name, target_depth=30, vs_halfspace=500):
    
    graph = SPACGraph(coords)
    analyzer = SPACResponseAnalyzer(graph, target_depth, vs_halfspace)
    
    # basic stats
    n_sensors = graph.n_sensors
    n_edges = len(graph.edges)
    r_min = graph.r_min
    r_max = graph.r_max
    
    # distance distribution
    distances = graph.get_distance_distribution()
    r_mean = np.mean(distances)
    r_std = np.std(distances)
    
    # bessel coverage
    bessel_score, _ = analyzer.compute_bessel_coverage(freq_range=None)
    
    # depth penalty
    depth_penalty = analyzer.compute_depth_penalty()
    
    # azimuthal coverage (isotropy)
    azimuth_score, isotropy_score = analyzer.compute_azimuthal_coverage()
    
    # k-space analysis
    # k_max should capture main lobe (width ~ 2pi/aperture) AND some side lobes
    # main lobe width = 2*pi / r_max
    # nyquist limit = pi / r_min
    # we want k_max to show main lobe clearly plus a few side lobes
    
    main_lobe_width = 2 * np.pi / r_max if r_max > 0 else 0.1
    nyquist_k = np.pi / r_min if r_min > 0 else 1.0
    
    # use 10x main lobe width, but cap at Nyquist
    k_max = min(10 * main_lobe_width, nyquist_k, 0.8)
    k_max = max(k_max, 0.05)  # minimum k_max
    
    k_range = np.linspace(-k_max, k_max, 150)
    kx, ky, arf = analyzer.compute_arf(k_range)
    
    # calculate SLL
    center = arf.shape[0] // 2
    peak = arf[center, center]
    
    # mask center
    mask_radius = arf.shape[0] // 6
    y, x = np.ogrid[:arf.shape[0], :arf.shape[1]]
    center_mask = ((x - center)**2 + (y - center)**2) <= mask_radius**2
    side = arf.copy()
    side[center_mask] = 0
    max_side = np.max(side)
    
    sll_db = -20 * np.log10(max_side / peak + 1e-10) if peak > 0 else 0
    
    return {
        'name': name,
        'n_sensors': n_sensors,
        'n_edges': n_edges,
        'r_min': r_min,
        'r_max': r_max,
        'r_mean': r_mean,
        'r_std': r_std,
        'bessel_score': bessel_score,
        'depth_penalty': depth_penalty,
        'azimuth_score': azimuth_score,
        'isotropy_score': isotropy_score,
        'sll_db': sll_db,
        'coords': coords,
        'arf': arf,
        'k_max': k_max
    }

def visualize_comparison(results, filename='array_comparison.png'):
    
    n = len(results)
    n_cols = 5
    fig = plt.figure(figsize=(5 * n_cols, 4.2 * n))
    
    for i, res in enumerate(results):
        coords = np.array(res['coords'])
        xs, ys = coords[:, 0], coords[:, 1]
        row_base = i * n_cols  # subplot index offset
        
        # ── 1. Array geometry ──────────────────────────────────────────────
        ax1 = fig.add_subplot(n, n_cols, row_base + 1)
        # edge lines between sensors (thicker, more visible)
        n_s_geom = len(coords)
        for j in range(n_s_geom):
            for k in range(j + 1, n_s_geom):
                ax1.plot([xs[j], xs[k]], [ys[j], ys[k]],
                         color='steelblue', alpha=0.25, linewidth=1.2, zorder=1)
        ax1.scatter(xs, ys, s=70, zorder=5,
                    c='steelblue', edgecolors='white', linewidth=1.2)
        ax1.set_aspect('equal')
        ax1.set_title(f"{res['name']}\n"
                      f"N={res['n_sensors']}  r_max={res['r_max']:.0f}m",
                      fontsize=9)
        ax1.set_xlabel('X (m)', fontsize=8)
        ax1.set_ylabel('Y (m)', fontsize=8)
        ax1.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
        ax1.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
        ax1.tick_params(labelsize=7)
        
        # ── 2. Inter-station vectors ───────────────────────────────────────
        # aRF quality directly depends on how well inter-station vectors
        # fill the 2D plane: denser, more uniform fill → lower side lobes.
        ax2 = fig.add_subplot(n, n_cols, row_base + 2)
        n_s = len(coords)
        ivx, ivy = [], []
        for j in range(n_s):
            for k in range(j + 1, n_s):
                dx = coords[k, 0] - coords[j, 0]
                dy = coords[k, 1] - coords[j, 1]
                ivx += [dx, -dx]
                ivy += [dy, -dy]
        ivx, ivy = np.array(ivx), np.array(ivy)
        
        # axis limit = actual max inter-station distance (not r_max!)
        iv_lengths = np.sqrt(ivx**2 + ivy**2)
        lim = iv_lengths.max() * 1.08 if len(iv_lengths) > 0 else res['r_max'] * 2
        iv_min = iv_lengths.min() if len(iv_lengths) > 0 else res['r_min']
        
        ms = max(1, min(6, 200 // n_s))   # marker size: bigger for fewer sensors
        ax2.scatter(ivx, ivy, s=ms, alpha=0.5, c='darkorange', lw=0)
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_aspect('equal')
        ax2.set_title('Inter-station\nVectors', fontsize=9)
        ax2.set_xlabel('Δx (m)', fontsize=8)
        ax2.set_ylabel('Δy (m)', fontsize=8)
        # reference circles at shortest and longest inter-sensor distance
        for r, col, lab in [(iv_min, 'red', f'min={iv_min:.0f}m'),
                             (lim / 1.08, 'green', f'max={lim/1.08:.0f}m')]:
            circ = plt.Circle((0, 0), r, fill=False,
                              color=col, lw=0.8, ls='--', alpha=0.7)
            ax2.add_patch(circ)
        ax2.tick_params(labelsize=7)
        
        # ── 3. ARF linear scale ────────────────────────────────────────────
        ax3 = fig.add_subplot(n, n_cols, row_base + 3)
        k_max = res['k_max']
        arf = res['arf']
        im3 = ax3.imshow(arf.T, extent=[-k_max, k_max, -k_max, k_max],
                         origin='lower', cmap='hot', vmin=0, vmax=1)
        ax3.set_title(f'ARF (linear)\nSLL={res["sll_db"]:.1f} dB', fontsize=9)
        ax3.set_xlabel('kx (rad/m)', fontsize=8)
        ax3.set_ylabel('ky (rad/m)', fontsize=8)
        plt.colorbar(im3, ax=ax3, shrink=0.85, pad=0.02)
        ax3.tick_params(labelsize=7)
        
        # ── 4. ARF dB scale ────────────────────────────────────────────────
        ax4 = fig.add_subplot(n, n_cols, row_base + 4)
        arf_db = 20 * np.log10(arf + 1e-6)
        arf_db -= arf_db.max()          # normalise peak to 0 dB
        im4 = ax4.imshow(arf_db.T, extent=[-k_max, k_max, -k_max, k_max],
                         origin='lower', cmap='seismic_r', vmin=-40, vmax=0)
        ax4.set_title('ARF (dB)', fontsize=9)
        ax4.set_xlabel('kx (rad/m)', fontsize=8)
        ax4.set_ylabel('ky (rad/m)', fontsize=8)
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.85, pad=0.02)
        cbar4.set_label('dB', fontsize=7)
        ax4.tick_params(labelsize=7)
        
        # ── 5. Rose diagram ────────────────────────────────────────────────
        from spac_graph import SPACGraph
        graph_tmp = SPACGraph(res['coords'])
        angles_folded = graph_tmp.get_angle_distribution_geo()   # [0, pi)
        
        ax5 = fig.add_subplot(n, n_cols, row_base + 5, projection='polar')
        n_bins = 18
        hist, bin_edges = np.histogram(angles_folded,
                                       bins=n_bins, range=(0, np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = np.pi / n_bins
        ax5.bar(bin_centers, hist, width=width,
                alpha=0.75, color='steelblue', edgecolor='white', lw=0.5)
        ax5.bar(bin_centers + np.pi, hist, width=width,
                alpha=0.75, color='steelblue', edgecolor='white', lw=0.5)
        ax5.set_title(f'Rose Diagram\nIsotropy={res["isotropy_score"]:.3f}',
                      fontsize=9)
        ax5.set_theta_zero_location('N')
        ax5.set_theta_direction(-1)
        ax5.tick_params(labelsize=6)
    
    fig.suptitle('SPAC Array Comparison  —  ARF · SLL · Azimuthal Coverage',
                 fontsize=12, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved comparison to {filename}')
    return fig

def print_comparison_table(results):
    
    print("\n" + "="*100)
    print("ARRAY COMPARISON TABLE")
    print("="*100)
    print(f"{'Array':<20} {'Sensors':<8} {'Edges':<7} {'r_min':<8} {'r_max':<8} "
          f"{'Bessel':<8} {'Isotropy':<9} {'SLL(dB)':<8}")
    print("-"*100)
    
    for res in results:
        print(f"{res['name']:<20} {res['n_sensors']:<8} {res['n_edges']:<7} "
              f"{res['r_min']:<8.1f} {res['r_max']:<8.1f} "
              f"{res['bessel_score']:<8.3f} {res['isotropy_score']:<9.3f} "
              f"{res['sll_db']:<8.1f}")
    
    print("="*100)
    
    # find best
    best_sll = max(results, key=lambda x: x['sll_db'])
    best_bessel = max(results, key=lambda x: x['bessel_score'])
    best_isotropy = max(results, key=lambda x: x['isotropy_score'])
    
    print(f"\nBest SLL: {best_sll['name']} ({best_sll['sll_db']:.1f} dB)")
    print(f"Best Bessel Coverage: {best_bessel['name']} ({best_bessel['bessel_score']:.3f})")
    print(f"Best Isotropy: {best_isotropy['name']} ({best_isotropy['isotropy_score']:.3f})")

def run_standard_comparison(target_depth=30, vs_halfspace=500):
    
    
    print("="*60)
    print("SPAC ARRAY ANALYZER")
    print("Kennett et al. (2015) Methodology")
    print("="*60)
    print(f"Target Depth: {target_depth}m, Vs: {vs_halfspace}m/s")
    print(f"Required aperture (4*depth): {4*target_depth}m")
    
    aperture = 4 * target_depth  # λmax = 4*depth
    r50 = aperture / 2           # half-aperture for smaller arrays
    
    arrays = [
        # classic shapes
        ("Circle (8)",           create_circle_array(aperture/2, 8)),
        ("Circle (12)",          create_circle_array(aperture/2, 12)),
        ("Cross (13)",           create_cross_array(aperture/2, n_per_arm=3)),
        ("Propeller 3×3",        create_propeller_array(3, 3, aperture/2)),
        # kennett et al. 2015 spiral configurations
        ("Spiral 3a×4r (lin)",   create_spiral_arm_array(aperture/2, n_arms=3, n_rings=4, span_deg=120)),
        ("Spiral 3a×4r (log)",   create_spiral_arm_array(aperture/2, n_arms=3, n_rings=4, span_deg=120,
                                                          log_spacing=True)),
    ]
    
    results = []
    for name, coords in arrays:
        print(f"  Analyzing {name} ({len(coords)} sensors)...")
        res = analyze_array(coords, name, target_depth, vs_halfspace)
        results.append(res)
    
    print_comparison_table(results)
    visualize_comparison(results, 'standard_arrays_comparison.png')
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="SPAC Array Analyzer - Compare different array configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
    )
    
    parser.add_argument('--shape', type=str, default=None,
                        choices=['circle', 'cross', 'x', 'triangle', 'triangle_center', 
                                'nested_triangle', 'l', 'fan', 'propeller', 'random',
                                'fan_asym', 'fan_sym'],
                        help='Shape to analyze (default: compare all)')
    parser.add_argument('--radius', type=float, default=50,
                        help='Radius/size in meters (default: 50)')
    parser.add_argument('--sensors', type=int, default=8,
                        help='Number of sensors (default: 8)')
    parser.add_argument('--depth', type=float, default=30,
                        help='Target depth in meters (default: 30)')
    parser.add_argument('--vs', type=float, default=400,
                        help='Vs halfspace in m/s (default: 400)')
    parser.add_argument('--custom', type=str, default=None,
                        help='Path to custom coordinates file (x,y per line)')
    
    args = parser.parse_args()
    
    # if no specific shape or custom file, run full comparison
    if args.shape is None and args.custom is None:
        print("No shape specified - running full comparison...")
        run_standard_comparison(args.depth, args.vs)
        return
    
    # single shape analysis
    if args.custom:
        coords = []
        with open(args.custom, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    x, y = map(float, parts[:2])
                    coords.append((x, y))
        name = f"Custom ({len(coords)} sensors)"
    else:
        shape_funcs = {
            'circle': lambda: create_circle_array(args.radius, args.sensors),
            'cross': lambda: create_cross_array(args.radius, n_per_arm=3),
            'x': lambda: create_x_array(args.radius, n_per_arm=3),
            'triangle': lambda: create_triangle_array(args.radius),
            'triangle_center': lambda: create_triangle_center_array(args.radius),
            'nested_triangle': lambda: create_nested_triangle_array(args.radius, args.radius/2),
            'l': lambda: create_l_array(args.radius, n_per_arm=4),
            'fan': lambda: create_fan_array(args.radius, args.sensors),
            'propeller': lambda: create_propeller_array(3, 3, args.radius),
            'random': lambda: create_random_array(args.sensors, args.radius),
            'fan_asym': lambda: create_asymmetric_fan_array(),
            'fan_sym': lambda: create_symmetric_fan_array(),
        }
        coords = shape_funcs[args.shape]()
        name = args.shape.replace('_', ' ').title()
    
    res = analyze_array(coords, name, args.depth, args.vs)
    
    print(f"\n{'='*50}")
    print(f"{name} Analysis")
    print(f"{'='*50}")
    print(f"  Sensors: {res['n_sensors']}")
    print(f"  Edges: {res['n_edges']}")
    print(f"  r_min: {res['r_min']:.1f} m")
    print(f"  r_max: {res['r_max']:.1f} m")
    print(f"  Bessel Score: {res['bessel_score']:.3f}")
    print(f"  Isotropy: {res['isotropy_score']:.3f}")
    print(f"  SLL: {res['sll_db']:.1f} dB")
    
    output_file = f"{args.shape or 'custom'}_analysis.png"
    visualize_comparison([res], output_file)

if __name__ == "__main__":
    main()
