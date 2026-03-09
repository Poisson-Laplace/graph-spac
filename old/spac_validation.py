# author: ferat
# date: 2026
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j0 as J0
from scipy.optimize import brentq
import argparse

from spac_graph import SPACGraph, SPACResponseAnalyzer
from array_analyzer import (
    create_circle_array, create_cross_array, create_triangle_center_array,
    create_propeller_array, analyze_array
)

# 1. Layered Earth Models

def make_two_layer_model(vs1=200.0, h1=15.0, vs2=500.0):
    
    return {
        'layers': [
            {'vs': vs1, 'vp': vs1 * 1.8, 'rho': 1800, 'h': h1},
        ],
        'halfspace': {'vs': vs2, 'vp': vs2 * 1.7, 'rho': 2000},
        'description': f'2-layer: Vs1={vs1}m/s (h={h1}m) over Vs2={vs2}m/s'
    }

def make_three_layer_model(vs1=150.0, h1=8.0, vs2=300.0, h2=15.0, vs3=600.0):
    
    return {
        'layers': [
            {'vs': vs1, 'vp': vs1 * 2.0, 'rho': 1700, 'h': h1},
            {'vs': vs2, 'vp': vs2 * 1.8, 'rho': 1900, 'h': h2},
        ],
        'halfspace': {'vs': vs3, 'vp': vs3 * 1.7, 'rho': 2100},
        'description': f'3-layer: Vs1={vs1}, Vs2={vs2}, Vs3={vs3} m/s'
    }

def theoretical_rayleigh_dispersion(model, freqs):
    
    layers = model['layers']
    halfspace = model['halfspace']
    
    # build cumulative depth profile
    depths = []
    vs_vals = []
    z = 0
    for layer in layers:
        depths.append(z)
        depths.append(z + layer['h'])
        vs_vals.append(layer['vs'])
        vs_vals.append(layer['vs'])
        z += layer['h']
    depths.append(z)
    vs_vals.append(halfspace['vs'])
    
    depths = np.array(depths)
    vs_vals = np.array(vs_vals)
    
    c_rayleigh = np.zeros(len(freqs))
    
    for i, f in enumerate(freqs):
        if f <= 0:
            c_rayleigh[i] = halfspace['vs'] * 0.92
            continue
        
        # penetration depth ≈ lambda / 3  (standard approximation)
        # start with halfspace Vs as initial guess
        c_guess = halfspace['vs'] * 0.92
        
        # iterate: c_R → lambda → depth → Vs_eff → c_R
        for _ in range(8):
            lam = c_guess / f
            z_pen = lam / 3.0
            
            # depth-weighted average Vs down to z_pen
            vs_eff = _depth_weighted_vs(depths, vs_vals, z_pen)
            c_new = 0.92 * vs_eff
            
            if abs(c_new - c_guess) < 0.1:
                break
            c_guess = c_new
        
        c_rayleigh[i] = c_guess
    
    return c_rayleigh

def _depth_weighted_vs(depths, vs_vals, z_max):
    
    if z_max <= 0:
        return vs_vals[0]
    
    vs_integral = 0.0
    z_prev = 0.0
    
    for k in range(0, len(depths) - 1, 2):
        z_top = depths[k]
        z_bot = depths[k + 1]
        vs = vs_vals[k]
        
        dz_in = min(z_bot, z_max) - max(z_top, 0)
        if dz_in <= 0:
            continue
        
        vs_integral += vs * dz_in
        z_prev = min(z_bot, z_max)
        
        if z_prev >= z_max:
            break
    
    # halfspace contribution
    if z_prev < z_max:
        vs_integral += vs_vals[-1] * (z_max - z_prev)
    
    return vs_integral / z_max

# 2. Synthetic SPAC Correlation

def synthetic_spac_correlation(coords, freqs, c_rayleigh):
    
    coords = np.array(coords)
    n = len(coords)
    
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]
            r = np.sqrt(dx**2 + dy**2)
            pairs.append(r)
    
    pair_distances = np.array(pairs)
    n_pairs = len(pairs)
    
    rho_matrix = np.zeros((n_pairs, len(freqs)))
    for fi, (f, c) in enumerate(zip(freqs, c_rayleigh)):
        k = 2 * np.pi * f / c
        rho_matrix[:, fi] = J0(k * pair_distances)
    
    return rho_matrix, pair_distances

def recover_dispersion(rho_matrix, pair_distances, freqs,
                       c_min=100.0, c_max=1200.0, n_c=200):
    
    c_test = np.linspace(c_min, c_max, n_c)
    c_recovered = np.zeros(len(freqs))
    misfit = np.zeros(len(freqs))
    
    for fi, f in enumerate(freqs):
        rho_obs = rho_matrix[:, fi]
        best_misfit = np.inf
        best_c = c_test[0]
        
        for c in c_test:
            k = 2 * np.pi * f / c
            rho_pred = J0(k * pair_distances)
            m = np.sqrt(np.mean((rho_obs - rho_pred) ** 2))
            if m < best_misfit:
                best_misfit = m
                best_c = c
        
        c_recovered[fi] = best_c
        misfit[fi] = best_misfit
    
    return c_recovered, misfit

# 3. Array Comparison

def compare_arrays_synthetic(arrays, model, target_depth=30, vs_halfspace=500,
                              freq_range=(0.5, 20), n_freqs=40,
                              output='validation_comparison.png'):
    
    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    c_true = theoretical_rayleigh_dispersion(model, freqs)
    
    print(f"\nSynthetic Validation")
    print(f"Model: {model['description']}")
    print(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
    print("=" * 70)
    
    results = []
    for name, coords in arrays:
        rho, dists = synthetic_spac_correlation(coords, freqs, c_true)
        c_rec, misfit = recover_dispersion(rho, dists, freqs,
                                           c_min=vs_halfspace * 0.2,
                                           c_max=vs_halfspace * 1.5)
        
        # metrics
        rms_error = np.sqrt(np.mean((c_rec - c_true) ** 2))
        mean_misfit = np.mean(misfit)
        
        # geometric metrics
        geo = analyze_array(coords, name, target_depth, vs_halfspace)
        
        results.append({
            'name': name,
            'coords': coords,
            'c_recovered': c_rec,
            'c_true': c_true,
            'misfit': misfit,
            'rms_error': rms_error,
            'mean_misfit': mean_misfit,
            'bessel_score': geo['bessel_score'],
            'isotropy_score': geo['isotropy_score'],
            'r_min': geo['r_min'],
            'r_max': geo['r_max'],
        })
        
        print(f"  {name:<25}: RMS c error = {rms_error:6.1f} m/s  |  "
              f"Bessel = {geo['bessel_score']:.3f}  |  "
              f"r_min={geo['r_min']:.1f}m  r_max={geo['r_max']:.1f}m")
    
    # plot
    _plot_validation(results, freqs, c_true, model, output)
    
    return results

def _plot_validation(results, freqs, c_true, model, filename):
    
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    
    if n == 1:
        axes = axes.reshape(2, 1)
    
    for i, res in enumerate(results):
        # top: dispersion recovery
        ax = axes[0, i]
        ax.plot(freqs, c_true, 'k-', lw=2.5, label='True c(f)')
        ax.plot(freqs, res['c_recovered'], 'r--', lw=1.5, label='Recovered')
        ax.fill_between(freqs,
                        res['c_recovered'] - res['misfit'] * 50,
                        res['c_recovered'] + res['misfit'] * 50,
                        alpha=0.2, color='red', label='Uncertainty')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase Velocity (m/s)')
        ax.set_title(f"{res['name']}\nRMS error = {res['rms_error']:.1f} m/s")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # bottom: array layout
        ax2 = axes[1, i]
        xs = [c[0] for c in res['coords']]
        ys = [c[1] for c in res['coords']]
        ax2.scatter(xs, ys, s=80, c='steelblue', edgecolors='white', zorder=5)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f"r_min={res['r_min']:.0f}m  r_max={res['r_max']:.0f}m\n"
                      f"Bessel={res['bessel_score']:.3f}  "
                      f"Isotropy={res['isotropy_score']:.3f}")
        ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f"Synthetic SPAC Validation\n{model['description']}", fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved validation plot to {filename}")

# 4. Main

def main():
    parser = argparse.ArgumentParser(
        description='Graphspac Synthetic SPAC Validation',
        epilog='Validates array designs against synthetic dispersion data.'
    )
    parser.add_argument('--depth', type=float, default=30,
                        help='Target depth [m] (default: 30)')
    parser.add_argument('--vs', type=float, default=500,
                        help='Halfspace Vs [m/s] (default: 500)')
    parser.add_argument('--sensors', type=int, default=10,
                        help='Number of sensors for optimized array (default: 10)')
    parser.add_argument('--output', type=str, default='validation_comparison.png',
                        help='Output filename (default: validation_comparison.png)')
    args = parser.parse_args()
    
    # earth model
    model = make_two_layer_model(
        vs1=args.vs * 0.35,
        h1=args.depth * 0.5,
        vs2=args.vs
    )
    print(f"Earth model: {model['description']}")
    
    # arrays to compare
    r = args.depth * 4 / 3  # required aperture
    arrays = [
        ('Circle (8)',      create_circle_array(r, 8)),
        ('Cross',           create_cross_array(r, n_per_arm=3)),
        ('Triangle+Center', create_triangle_center_array(r * 1.2)),
        ('Propeller (3)',   create_propeller_array(3, 3, r)),
    ]
    
    print(f"\nComparing {len(arrays)} standard arrays against synthetic data...")
    print(f"Target depth: {args.depth} m | Required aperture: {r:.1f} m")
    
    results = compare_arrays_synthetic(
        arrays, model,
        target_depth=args.depth,
        vs_halfspace=args.vs,
        freq_range=(0.5, min(20, args.vs / (2 * args.depth))),
        output=args.output
    )
    
    # summary table
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Array':<25} {'RMS c error':>12} {'Bessel':>8} {'Isotropy':>10}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['rms_error']):
        print(f"{r['name']:<25} {r['rms_error']:>12.1f} {r['bessel_score']:>8.3f} "
              f"{r['isotropy_score']:>10.3f}")
    print("=" * 70)
    best = min(results, key=lambda x: x['rms_error'])
    print(f"\nBest dispersion recovery: {best['name']} (RMS = {best['rms_error']:.1f} m/s)")

if __name__ == '__main__':
    main()
