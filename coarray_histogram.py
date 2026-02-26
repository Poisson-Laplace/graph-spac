# author: ferat
# date: 2026
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from grid_manager import GridManager
from spac_optimizer import SPACOptimizer
from spac_graph import SPACGraph, SPACResponseAnalyzer
from array_analyzer import (
    analyze_array,
    create_circle_array,
    create_nested_triangle_array,
    create_propeller_array,
    create_spiral_arm_array,
)

SCENARIOS = [
    {"id": 1, "name": "Shallow (10m)",    "focus": 10, "vs": 300, "sensors": 6,  "gridsize": 15},
    {"id": 2, "name": "Medium (30m)",     "focus": 30, "vs": 500, "sensors": 10, "gridsize": 30},
    {"id": 3, "name": "Deep (60m)",       "focus": 60, "vs": 700, "sensors": 13, "gridsize": 50},
    {"id": 4, "name": "Focal Target",     "focus": 30, "vs": 500, "sensors": 10, "gridsize": 30},
    {"id": 5, "name": "N-Source Adapted", "focus": 30, "vs": 500, "sensors": 10, "gridsize": 30},
]

def run_ga(sc, gens=50, pop=60, seed=42):
    
    gm = GridManager(None, grid_spacing=5.0, default_grid_size=sc["gridsize"])
    opt = SPACOptimizer(
        gm,
        n_sensors=sc["sensors"],
        target_depth=sc["focus"],
        vs_halfspace=sc["vs"],
        focus_point=(75.0, 75.0) if sc["id"] == 4 else None,
        focus_direction=0.0 if sc["id"] == 5 else None,
        population_size=pop,
        generations=gens,
        random_seed=seed,
    )
    best_indices, best_fitness, _ = opt.run(verbose=True)
    return gm.get_physical_coords(best_indices, jitter=False)

def get_classic_arrays(n_sensors, aperture):
    
    classics = [
        ("Circle",            create_circle_array(aperture / 2, n_sensors)),
        ("Propeller 3×3",     create_propeller_array(3, 3, aperture / 2)),
        ("Nested Triangles",  create_nested_triangle_array(aperture / 2, aperture / 4)),
        ("Spiral (Kennett)",  create_spiral_arm_array(aperture / 2, n_arms=3, n_rings=4)),
    ]
    return classics

def compute_coarray(coords):
    
    coords = np.array(coords)
    n = len(coords)
    dists, vecs_x, vecs_y = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[j, 0] - coords[i, 0]
                dy = coords[j, 1] - coords[i, 1]
                dists.append(np.sqrt(dx**2 + dy**2))
                vecs_x.append(dx)
                vecs_y.append(dy)
    return np.array(dists), np.array(vecs_x), np.array(vecs_y)

def redundancy_ratio(dists, n_bins=30):
    
    hist, _ = np.histogram(dists, bins=n_bins)
    mean_count = np.mean(hist[hist > 0])
    redundant = np.sum(hist > 1.5 * mean_count)
    return redundant / n_bins

def spectral_gap_ratio(dists, r_max, n_bins=30):
    
    hist, _ = np.histogram(dists, bins=n_bins, range=(0, r_max))
    return np.sum(hist == 0) / n_bins

def plot_coarray(graphspac_coords, classics, sc, filename="coarray_comparison.png"):
    
    all_arrays = [("Graphspac (GA)", graphspac_coords)] + classics
    n = len(all_arrays)

    fig = plt.figure(figsize=(4.0 * n, 9))
    fig.suptitle(
        f"Co-array Analysis  —  Scenario: {sc['name']}\n"
        "Graphspac vs Classical Arrays  |  Inter-station Vectors & Distance Histogram",
        fontsize=11, fontweight='bold', y=1.01,
    )

    top_axes, bot_axes = [], []
    for col, (name, coords) in enumerate(all_arrays):
        ax_top = fig.add_subplot(2, n, col + 1)
        ax_bot = fig.add_subplot(2, n, n + col + 1)
        top_axes.append(ax_top)
        bot_axes.append(ax_bot)

        coords_arr = np.array(coords)
        dists, vx, vy = compute_coarray(coords_arr)
        r_max = dists.max()

        # ── Row A: 2D Vector cloud ──────────────────────────────────────────
        color = '#2563EB' if col == 0 else '#9CA3AF'
        edge_c = 'white' if col == 0 else '#6B7280'
        ms = max(2, min(8, 300 // len(coords_arr)))
        ax_top.scatter(vx, vy, s=ms, alpha=0.4, c=color, lw=0)
        lim = r_max * 1.05
        ax_top.set_xlim(-lim, lim)
        ax_top.set_ylim(-lim, lim)
        ax_top.set_aspect('equal')
        ax_top.set_title(name, fontsize=9, fontweight='bold' if col == 0 else 'normal')
        ax_top.set_xlabel('Δx (m)', fontsize=7)
        if col == 0:
            ax_top.set_ylabel('Δy (m)', fontsize=7)
        ax_top.tick_params(labelsize=6)
        # reference circles
        for r_ref, lc in [(dists.min(), 'red'), (r_max, 'green')]:
            circ = plt.Circle((0, 0), r_ref, fill=False, color=lc, lw=0.8, ls='--', alpha=0.7)
            ax_top.add_patch(circ)

        # ── Row B: 1D Distance histogram ────────────────────────────────────
        half_dists = dists[dists > 0][::2]   # unique pairs only (each distance appears twice)
        n_bins = 25
        counts, edges = np.histogram(half_dists, bins=n_bins, range=(0, r_max))
        centers = (edges[:-1] + edges[1:]) / 2
        width = (edges[1] - edges[0]) * 0.85

        bar_color = '#2563EB' if col == 0 else '#d1D5DB'
        bar_edge  = '#1D4ED8' if col == 0 else '#9CA3AF'
        ax_bot.bar(centers, counts, width=width, color=bar_color,
                   edgecolor=bar_edge, lw=0.6)

        # mark empty bins (spectral gaps) in red on classical arrays
        if col > 0:
            gap_mask = counts == 0
            for gc, gw in zip(centers[gap_mask], np.full(gap_mask.sum(), width)):
                ax_bot.axvspan(gc - gw / 2, gc + gw / 2, color='red', alpha=0.25)

        gap_frac = spectral_gap_ratio(half_dists, r_max, n_bins)
        red_frac  = redundancy_ratio(half_dists, n_bins)
        ax_bot.set_title(f'Gap={gap_frac:.0%}  Redund.={red_frac:.0%}',
                         fontsize=7.5, color='#dC2626' if col > 0 else '#16A34A')
        ax_bot.set_xlabel('Inter-station Distance (m)', fontsize=7)
        if col == 0:
            ax_bot.set_ylabel('Pair Count', fontsize=7)
        ax_bot.tick_params(labelsize=6)
        ax_bot.set_xlim(0, r_max)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved co-array comparison → {filename}")
    return fig

def print_summary(graphspac_coords, classics, sc):
    
    aperture = 4 * sc["focus"]
    all_arrays = [("Graphspac (GA)", graphspac_coords)] + classics
    print("\n" + "=" * 75)
    print(f"CO-ARRAY ANALYSIS - {sc['name']}")
    print("=" * 75)
    print(f"{'Array':<22} {'N pairs':>8} {'Spectral Gap':>14} {'Redundancy':>12} {'r_min':>7} {'r_max':>7}")
    print("-" * 75)
    for name, coords in all_arrays:
        dists, _, _ = compute_coarray(coords)
        half = dists[dists > 0][::2]
        r_max = half.max()
        n_pairs = len(half)
        gap = spectral_gap_ratio(half, r_max)
        red = redundancy_ratio(half)
        print(f"  {name:<20} {n_pairs:>8} {gap:>13.1%} {red:>12.1%} "
              f"{half.min():>7.1f} {r_max:>7.1f}")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="Co-array Histogram Visualizer")
    parser.add_argument('--scenario', type=int, default=2, help='Scenario ID (1-5)')
    parser.add_argument('--gens',     type=int, default=50)
    parser.add_argument('--pop',      type=int, default=60)
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    sc = next(s for s in SCENARIOS if s["id"] == args.scenario)
    print(f"\n  Co-array Analysis - Scenario {sc['id']}: {sc['name']}")

    print("\n  Running GA optimizer...")
    graphspac_coords = run_ga(sc, gens=args.gens, pop=args.pop, seed=args.seed)

    aperture = 4 * sc["focus"]
    classics = get_classic_arrays(sc["sensors"], aperture)
    print_summary(graphspac_coords, classics, sc)

    tag = f"sc{sc['id']}"
    plot_coarray(graphspac_coords, classics, sc, f"coarray_comparison_{tag}.png")
    print("\nDone.")

if __name__ == "__main__":
    main()
