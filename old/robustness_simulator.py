# author: ferat
# date: 2026
import argparse
import numpy as np
import matplotlib.pyplot as plt

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

# perturbation sigmas in meters (±0 = ideal, up to ±5 m)
SIGMAS = [0.0, 0.10, 0.50, 1.00, 2.00, 5.00]
N_TRIALS = 30

def run_ga(sc, gens=50, pop=60, seed=42):
    gm = GridManager(None, grid_spacing=5.0, default_grid_size=sc["gridsize"])
    opt = SPACOptimizer(
        gm,
        n_sensors=sc["sensors"],
        target_depth=sc["focus"],
        vs_halfspace=sc["vs"],
        focus_point=(75.0, 75.0) if sc["id"] == 4 else None,
        focus_direction=0.0    if sc["id"] == 5 else None,
        population_size=pop,
        generations=gens,
        random_seed=seed,
    )
    best_indices, _, _ = opt.run(verbose=True)
    return gm.get_physical_coords(best_indices, jitter=False)

def compute_metrics(coords, target_depth, vs_halfspace):
    
    graph = SPACGraph(coords)
    if graph.r_max < 1e-6 or graph.r_min < 1e-6:
        return np.nan, np.nan, np.nan

    analyzer = SPACResponseAnalyzer(graph, target_depth, vs_halfspace)

    k_max = min(2 * np.pi / graph.r_min, 0.8)
    k_range = np.linspace(-k_max, k_max, 100)
    kx, ky, arf = analyzer.compute_arf(k_range)

    center = arf.shape[0] // 2
    peak = arf[center, center]
    mask_r = arf.shape[0] // 6
    y, x = np.ogrid[:arf.shape[0], :arf.shape[1]]
    mask = ((x - center) ** 2 + (y - center) ** 2) <= mask_r ** 2
    side = arf.copy()
    side[mask] = 0
    max_side = np.max(side)
    sll_db = -20 * np.log10(max_side / peak + 1e-10) if peak > 0 else 0

    bessel_score, _ = analyzer.compute_bessel_coverage(freq_range=None)
    _, isotropy_score = analyzer.compute_azimuthal_coverage(focus_direction=None)

    return sll_db, bessel_score, isotropy_score

def perturb_coords(coords, sigma, rng):
    
    coords_arr = np.array(coords, dtype=float)
    noise = rng.normal(0, sigma, size=coords_arr.shape)
    return (coords_arr + noise).tolist()

def simulate_robustness(coords, target_depth, vs_halfspace, sigmas=SIGMAS, n_trials=N_TRIALS, seed=0):
    
    rng = np.random.default_rng(seed)
    results = {s: {"sll": [], "bessel": [], "iso": []} for s in sigmas}

    for sigma in sigmas:
        for _ in range(n_trials if sigma > 0 else 1):
            perturbed = perturb_coords(coords, sigma, rng)
            sll, bessel, iso = compute_metrics(perturbed, target_depth, vs_halfspace)
            results[sigma]["sll"].append(sll)
            results[sigma]["bessel"].append(bessel)
            results[sigma]["iso"].append(iso)

    return results

def extract_stats(results, sigmas, key):
    means = [np.nanmean(results[s][key]) for s in sigmas]
    stds  = [np.nanstd(results[s][key])  for s in sigmas]
    return np.array(means), np.array(stds)

def plot_robustness(all_results, sc, filename="robustness_results.png"):
    
    sigmas_cm = [s * 100 for s in SIGMAS]  # display in cm
    metrics = [
        ("sll",    "SLLdB (higher = better)",         True),
        ("bessel", "Bessel Coverage Score",            False),
        ("iso",    "Isotropy Score",                  False),
    ]

    colors = {
        "Graphspac (GA)":    "#2563EB",
        "Circle":           "#dC2626",
        "Nested Triangles": "#16A34A",
        "Propeller":        "#d97706",
        "Spiral (Kennett)": "#7C3AED",
    }
    markers = {
        "Graphspac (GA)":    "o",
        "Circle":           "s",
        "Nested Triangles": "^",
        "Propeller":        "D",
        "Spiral (Kennett)": "P",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Topological Robustness — Scenario: {sc['name']}\n"
                 f"SLLdB, Bessel, Isotropy vs Gaussian Deployment Error σ",
                 fontsize=11, fontweight='bold')

    for ax, (key, ylabel, invert) in zip(axes, metrics):
        for name, results in all_results.items():
            means, stds = extract_stats(results, SIGMAS, key)
            lw = 2.5 if name == "Graphspac (GA)" else 1.4
            ls = '-'   if name == "Graphspac (GA)" else '--'
            color   = colors.get(name, "gray")
            marker  = markers.get(name, "o")

            ax.plot(sigmas_cm, means, color=color, lw=lw, ls=ls,
                    marker=marker, ms=6, label=name, zorder=3 if name == "Graphspac (GA)" else 2)
            ax.fill_between(sigmas_cm,
                            means - stds, means + stds,
                            alpha=0.12 if name == "Graphspac (GA)" else 0.06,
                            color=color)

        ax.set_xlabel('Deployment Error σ (cm)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # shade reference error zones
        ax.axvspan(0, 10, alpha=0.06, color='green',  label='RTK-GPS range')
        ax.axvspan(10, 50, alpha=0.06, color='orange', label='Tape measure')
        ax.axvspan(50, 100 * SIGMAS[-1], alpha=0.06, color='red', label='Rough placement')

        if invert:
            ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved robustness figure → {filename}")
    return fig

def print_robustness_table(all_results, sc):
    print("\n" + "=" * 90)
    print(f"ROBUSTNESS SIMULATION - {sc['name']}")
    header = f"{'Array':<22}" + "".join(f"  σ={s*100:.0f}cm" for s in SIGMAS)
    print(header + "   (SLLdB mean)")
    print("-" * 90)
    for name, results in all_results.items():
        means, _ = extract_stats(results, SIGMAS, "sll")
        row = f"  {name:<20}" + "".join(f"  {m:>7.2f}" for m in means)
        print(row)
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description="Graphspac Robustness Simulator")
    parser.add_argument('--scenario', type=int, default=2, help='Scenario ID (1-5, default: 2)')
    parser.add_argument('--gens',   type=int, default=50)
    parser.add_argument('--pop',    type=int, default=60)
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--trials', type=int, default=N_TRIALS,
                        help=f'Number of Monte Carlo trials per sigma (default: {N_TRIALS})')
    args = parser.parse_args()

    sc = next(s for s in SCENARIOS if s["id"] == args.scenario)
    print(f"\n{'='*60}")
    print(f"  Robustness Simulator - Scenario {sc['id']}: {sc['name']}")
    print(f"  Perturbations: {[f'{s*100:.0f}cm' for s in SIGMAS]}")
    print(f"  Trials per σ: {args.trials}")
    print(f"{'='*60}")

    # ── Run GA ──
    print("\n  [1/5] Running GA optimizer...")
    graphspac_coords = run_ga(sc, gens=args.gens, pop=args.pop, seed=args.seed)

    # ── Classic arrays ──
    aperture = 4 * sc["focus"]
    classic_list = [
        ("Circle",           create_circle_array(aperture / 2, sc["sensors"])),
        ("Nested Triangles", create_nested_triangle_array(aperture / 2, aperture / 4)),
        ("Propeller",        create_propeller_array(3, 3, aperture / 2)),
        ("Spiral (Kennett)", create_spiral_arm_array(aperture / 2, n_arms=3, n_rings=4)),
    ]

    # ── Simulate robustness ──
    all_results = {}

    print(f"\n  [2/5] Simulating Graphspac robustness ({args.trials} trials/σ)...")
    all_results["Graphspac (GA)"] = simulate_robustness(
        graphspac_coords, sc["focus"], sc["vs"],
        sigmas=SIGMAS, n_trials=args.trials, seed=args.seed
    )

    for i, (name, coords) in enumerate(classic_list, 3):
        print(f"  [{i + 1}/5] Simulating {name} robustness...")
        all_results[name] = simulate_robustness(
            coords, sc["focus"], sc["vs"],
            sigmas=SIGMAS, n_trials=args.trials, seed=args.seed
        )

    print_robustness_table(all_results, sc)

    tag = f"sc{sc['id']}"
    plot_robustness(all_results, sc, f"robustness_results_{tag}.png")
    print("\nDone.")

if __name__ == "__main__":
    main()
