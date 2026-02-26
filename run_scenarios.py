# author: ferat
# date: 2026
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from grid_manager import GridManager
from spac_optimizer import SPACOptimizer
from spac_graph import SPACGraph, SPACResponseAnalyzer
from array_analyzer import analyze_array, visualize_comparison

# ─── Scenario definitions ────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": 1,
        "name": "Shallow (10m)",
        "description": "Urban noise survey — shallow targets (e.g. utility pipes, fill layer)",
        "focus": 10.0,
        "vs": 300.0,
        "sensors": 6,
        "gridsize": 15,    # 15×15 × 5m = 75m side
        "spacing": 5.0,
        "focus_point": None,
        "noise_azimuth": None,
    },
    {
        "id": 2,
        "name": "Medium (30m)",
        "description": "Standard site characterisation — engineering bedrock depth",
        "focus": 30.0,
        "vs": 500.0,
        "sensors": 10,
        "gridsize": 30,    # 30×30 × 5m = 150m side
        "spacing": 5.0,
        "focus_point": None,
        "noise_azimuth": None,
    },
    {
        "id": 3,
        "name": "Deep (60m)",
        "description": "Deep basin study — sediment thickness / bedrock below 60m",
        "focus": 60.0,
        "vs": 700.0,
        "sensors": 13,
        "gridsize": 50,    # 50×50 × 5m = 250m side
        "spacing": 5.0,
        "focus_point": None,
        "noise_azimuth": None,
    },
    {
        "id": 4,
        "name": "Focal target (30m)",
        "description": "Array optimised toward a specific subsurface target (building foundation)",
        "focus": 30.0,
        "vs": 500.0,
        "sensors": 10,
        "gridsize": 30,
        "spacing": 5.0,
        "focus_point": (75.0, 75.0),   # centre of the 150m grid
        "noise_azimuth": None,
    },
    {
        "id": 5,
        "name": "N-Source Adapted",
        "description": "Array adapted for dominant noise from North — sensors biased East-West",
        "focus": 30.0,
        "vs": 500.0,
        "sensors": 10,
        "gridsize": 30,
        "spacing": 5.0,
        "focus_point": None,
        "noise_azimuth": 0.0,   # 0° = North
    },
]

# ─── Helper: run GA for one scenario ─────────────────────────────────────────

def run_scenario(sc, gens=50, pop=60, seed=42, verbose=True):
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Scenario {sc['id']}: {sc['name']}")
        print(f"  {sc['description']}")
        print(f"  depth={sc['focus']}m  Vs={sc['vs']}m/s  "
              f"N={sc['sensors']}  grid={sc['gridsize']}×{sc['gridsize']}")
        print(f"{'='*60}")

    gm = GridManager(None, grid_spacing=sc['spacing'],
                     default_grid_size=sc['gridsize'])

    optimizer = SPACOptimizer(
        gm,
        n_sensors=sc['sensors'],
        target_depth=sc['focus'],
        vs_halfspace=sc['vs'],
        focus_point=sc['focus_point'],
        focus_direction=sc['noise_azimuth'],
        population_size=pop,
        generations=gens,
        d_min=None,
        random_seed=seed,
    )

    best_indices, best_fitness, _ = optimizer.run(verbose=verbose)
    phys_coords = gm.get_physical_coords(best_indices, jitter=False)

    # metrics
    graph = SPACGraph(phys_coords)
    analyzer = SPACResponseAnalyzer(graph, sc['focus'], sc['vs'])
    az_score, iso_score = analyzer.compute_azimuthal_coverage(sc['noise_azimuth'])
    bessel_score, _ = analyzer.compute_bessel_coverage(freq_range=None)

    if verbose:
        print(f"\n  → Fitness={best_fitness:.4f}  "
              f"r_min={graph.r_min:.1f}m  r_max={graph.r_max:.1f}m  "
              f"Isotropy={iso_score:.3f}  Bessel={bessel_score:.3f}")

    return phys_coords, {
        "scenario": sc,
        "coords": phys_coords,
        "fitness": best_fitness,
        "r_min": graph.r_min,
        "r_max": graph.r_max,
        "isotropy": iso_score,
        "bessel": bessel_score,
    }

# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary(all_results):
    print("\n" + "="*80)
    print("SCENARIO COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Scenario':<25} {'N':>4} {'r_min':>7} {'r_max':>7} "
          f"{'Bessel':>7} {'Isotropy':>9} {'Fitness':>8}")
    print("-"*80)
    for r in all_results:
        sc = r["scenario"]
        print(f"  {sc['name']:<23} {sc['sensors']:>4} "
              f"{r['r_min']:>7.1f} {r['r_max']:>7.1f} "
              f"{r['bessel']:>7.3f} {r['isotropy']:>9.3f} "
              f"{r['fitness']:>8.4f}")
    print("="*80)

# ─── Scenario geometry panel (standalone grid + sensors) ─────────────────────

def plot_scenario_layouts(all_results, filename='scenarios_layout.png'):
    
    from array_analyzer import analyze_array, visualize_comparison as vc
    analyzed = []
    for r in all_results:
        sc = r['scenario']
        print(f"  Computing ARF for scenario {sc['id']}: {sc['name']}...")
        res = analyze_array(r['coords'], sc['name'], sc['focus'], sc['vs'])
        analyzed.append(res)
    vc(analyzed, filename)
    print(f'Saved full scenario layout to {filename}')

# ─── Full ARF/SLL comparison for one scenario ────────────────────────────────

def analyze_and_compare(r, filename=None):
    
    sc = r['scenario']
    if filename is None:
        filename = f"scenario{sc['id']}_comparison.png"

    res = analyze_array(r['coords'], sc['name'], sc['focus'], sc['vs'])
    visualize_comparison([res], filename)

    print(f"  ARF analysis: Bessel={res['bessel_score']:.3f}  "
          f"Isotropy={res['isotropy_score']:.3f}  SLL={res['sll_db']:.1f}dB")
    return res

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Graphspac Scenario Runner — optimise & compare SPAC arrays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--scenario', type=int, default=None,
                        help='Run only this scenario (1-5). Default: all.')
    parser.add_argument('--gens', type=int, default=50,
                        help='GA generations (default: 50)')
    parser.add_argument('--pop', type=int, default=60,
                        help='GA population size (default: 60)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--arf', action='store_true',
                        help='Also produce per-scenario ARF/SLL comparison panels')
    args = parser.parse_args()

    target_scenarios = SCENARIOS
    if args.scenario is not None:
        target_scenarios = [s for s in SCENARIOS if s['id'] == args.scenario]
        if not target_scenarios:
            print(f"Error: scenario {args.scenario} not found (choose 1-5).")
            sys.exit(1)

    print("\n" + "="*60)
    print("GRAPHSPAC - Scenario Runner")
    print("="*60)
    print(f"Scenarios : {[s['id'] for s in target_scenarios]}")
    print(f"GA params : gens={args.gens}  pop={args.pop}  seed={args.seed}")

    all_results = []
    for sc in target_scenarios:
        _, r = run_scenario(sc, gens=args.gens, pop=args.pop, seed=args.seed)
        all_results.append(r)

    print_summary(all_results)
    plot_scenario_layouts(all_results, 'scenarios_layout.png')

    if args.arf:
        print("\nGenerating ARF/SLL comparison panels...")
        for r in all_results:
            sc = r['scenario']
            analyze_and_compare(r, f"scenario{sc['id']}_comparison.png")

    print("\nDone.")

if __name__ == '__main__':
    main()
