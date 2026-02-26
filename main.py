# author: ferat
# date: 2026
import argparse
import os
import glob
from grid_manager import GridManager
from spac_optimizer import SPACOptimizer
from spac_visualizer import SPACVisualizer
from spac_graph import SPACGraph

def get_next_filename(base_name, extension='.png'):
    
    # remove extension if present
    if base_name.endswith(extension):
        base_name = base_name[:-len(extension)]
    
    # find existing files
    pattern = f"{base_name}*{extension}"
    existing = glob.glob(pattern)
    
    if not existing:
        # first file
        return f"{base_name}1{extension}"
    
    # find highest number
    max_num = 0
    for f in existing:
        # extract number from filename
        name = os.path.basename(f)[:-len(extension)]
        suffix = name.replace(base_name, '')
        if suffix.isdigit():
            max_num = max(max_num, int(suffix))
    
    return f"{base_name}{max_num + 1}{extension}"

def main():
    parser = argparse.ArgumentParser(
        description="Graphspac: Graph-Based SPAC Array Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
    )
    
    # grid parameters
    parser.add_argument('--matrix', type=str, default='matrix.xlsx',
                        help='Path to matrix.xlsx (1=valid, 0=obstacle)')
    parser.add_argument('--spacing', type=float, default=5.0,
                        help='Grid cell size in meters (default: 5.0)')
    parser.add_argument('--yeryok', type=int, default=0, choices=[0, 1],
                        help='0=ignore matrix.xlsx, 1=use matrix.xlsx for constraints')
    
    # sPAC parameters
    parser.add_argument('--focus', type=float, default=30.0,
                        help='Target investigation depth in meters (default: 30)')
    parser.add_argument('--vs', type=float, default=500.0,
                        help='Estimated Vs of bedrock in m/s (default: 500)')
    parser.add_argument('--sensors', type=int, default=10,
                        help='Number of sensors (default: 10)')
    
    # focus parameters
    parser.add_argument('--focusl', type=str, default=None,
                        help='Focal point X,Y in meters (e.g., "50,50")')
    parser.add_argument('--noiseazimuth', type=float, default=None,
                        help='Azimuth of dominant noise source in degrees (0=North). '
                             'If not set, optimizes for isotropic (circular) coverage.')
    
    # gA parameters
    parser.add_argument('--gens', type=int, default=80,
                        help='Number of generations (default: 80)')
    parser.add_argument('--pop', type=int, default=60,
                        help='Population size (default: 60)')
    parser.add_argument('--dmin', type=float, default=None,
                        help='Minimum inter-sensor distance in meters (default: grid_spacing). '
                             'Enforces practical deployment constraint.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42). '
                             'Use --seed -1 for non-deterministic runs.')
    
    # grid size parameter
    parser.add_argument('--gridsize', type=int, default=None,
                        help='Grid size (X=Y dimension). If specified, ignores matrix.xlsx. Default: 100 when not using matrix')
    
    # output
    parser.add_argument('--output', type=str, default='spac_result.png',
                        help='Output filename (default: spac_result.png)')
    parser.add_argument('--kennett', action='store_true',
                        help='After GA, run Kennett et al. (2015) ARF+SLL analysis '
                             'and compare optimized array vs standard shapes.')
    
    args = parser.parse_args()
    
    # parse focal point
    focus_point = None
    if args.focusl is not None:
        try:
            x, y = map(float, args.focusl.split(','))
            focus_point = (x, y)
        except:
            print(f"Warning: Could not parse --focusl '{args.focusl}'. Expected format: X,Y")
    
    # initialize grid
    print("="*60)
    print("GRAPHSPAC - Graph-Based SPAC Array Optimizer")
    print("="*60)
    
    # handle grid constraints
    # priority: --gridsize (if specified) > --yeryok > default
    if args.gridsize is not None:
        # user explicitly specified grid size - ignore matrix.xlsx completely
        print(f"\nUsing custom grid size: {args.gridsize}x{args.gridsize} (--gridsize overrides matrix.xlsx)")
        gm = GridManager(None, grid_spacing=args.spacing, default_grid_size=args.gridsize)
    elif args.yeryok == 1:
        if os.path.exists(args.matrix):
            print(f"\nUsing grid constraints from {args.matrix} (spacing: {args.spacing}m)...")
            gm = GridManager(args.matrix, grid_spacing=args.spacing)
        else:
            print(f"\nWarning: {args.matrix} not found. Creating default 100x100 grid...")
            gm = GridManager(None, grid_spacing=args.spacing, default_grid_size=100)
    else:
        print(f"\nNo grid constraints (--yeryok 0). Creating open 100x100 grid...")
        gm = GridManager(None, grid_spacing=args.spacing, default_grid_size=100)
    
    # calculate derived parameters
    lambda_max = 4 * args.focus
    r_required = lambda_max / 3
    
    print(f"\nSPAC Parameters:")
    print(f"  - Target depth (--focus): {args.focus} m")
    print(f"  - Vs (bedrock): {args.vs} m/s")
    print(f"  - λ_max (max wavelength): {lambda_max:.1f} m")
    print(f"  - Required aperture (r_max ≥ λ_max/3): {r_required:.1f} m")
    print(f"  - Number of sensors: {args.sensors}")
    print(f"  - Grid constraints (--yeryok): {'Yes' if args.yeryok == 1 else 'No'}")
    
    if focus_point:
        print(f"  - Focal point (--focusl): {focus_point}")
    
    if args.noiseazimuth is not None:
        print(f"  - Noise direction (--noiseazimuth): {args.noiseazimuth}° from North")
    else:
        print(f"  - Azimuthal mode: ISOTROPIC (optimizing for circular rose diagram)")
    
    # check if grid is large enough
    grid_diagonal = args.spacing * ((gm.rows**2 + gm.cols**2)**0.5)
    if grid_diagonal < r_required:
        print(f"\n⚠️  WARNING: Grid diagonal ({grid_diagonal:.1f}m) is smaller than "
              f"required aperture ({r_required:.1f}m)!")
        print("   Consider reducing target depth or using a larger grid.")
    
    # initialize optimizer
    print(f"\nStarting optimization ({args.gens} generations, pop={args.pop})...\n")
    
    seed = None if args.seed == -1 else args.seed
    dmin = args.dmin
    
    if dmin is not None:
        print(f"  - Min sensor distance (--dmin): {dmin} m")
    else:
        print(f"  - Min sensor distance (--dmin): {args.spacing} m (= grid_spacing, default)")
    print(f"  - Random seed (--seed): {seed if seed is not None else 'None (non-deterministic)'}")
    
    optimizer = SPACOptimizer(
        gm,
        n_sensors=args.sensors,
        target_depth=args.focus,
        vs_halfspace=args.vs,
        focus_point=focus_point,
        focus_direction=args.noiseazimuth,
        population_size=args.pop,
        generations=args.gens,
        d_min=dmin,
        random_seed=seed
    )
    
    # run optimization
    best_indices, best_fitness, history = optimizer.run(verbose=True)
    
    # results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    # analyze best array
    phys_coords = gm.get_physical_coords(best_indices, jitter=False)
    graph = SPACGraph(phys_coords)
    
    from spac_graph import SPACResponseAnalyzer
    analyzer = SPACResponseAnalyzer(graph, args.focus, args.vs)
    azimuth_score, isotropy_score = analyzer.compute_azimuthal_coverage(args.noiseazimuth)
    
    print(f"\nBest Fitness: {best_fitness:.4f}")
    print(f"\nArray Statistics:")
    print(f"  - Number of sensors: {graph.n_sensors}")
    print(f"  - r_min (shortest edge): {graph.r_min:.2f} m")
    print(f"  - r_max (aperture): {graph.r_max:.2f} m")
    print(f"  - Number of edges: {len(graph.edges)}")
    print(f"  - Isotropy score: {isotropy_score:.3f} (1.0 = perfect circle)")
    
    # check depth capability
    if graph.r_max >= r_required:
        print(f"\n✓ Aperture is sufficient for {args.focus}m depth")
    else:
        deficit = r_required - graph.r_max
        print(f"\n⚠ Aperture is {deficit:.1f}m short for {args.focus}m depth")
    
    # print sensor positions
    print("\nSensor Grid Positions:")
    for i, (r, c) in enumerate(best_indices):
        x, y = phys_coords[i]
        print(f"  {i+1}: Row={r}, Col={c} → ({x:.1f}m, {y:.1f}m)")
    
    # generate visualization with auto-increment filename
    output_file = get_next_filename(args.output)
    print(f"\nGenerating visualization...")
    visualizer = SPACVisualizer(gm)
    visualizer.plot_comprehensive(best_indices, args.focus, args.vs, output_file)
    print(f"\n✓ Results saved to {output_file}")

    # ── Kennett et al. 2015 ARF / SLL analysis ────────────────────────────
    if args.kennett:
        print("\n" + "="*60)
        print("KENNETT et al. (2015) METHODOLOGY ANALYSIS")
        print("ARF, SLL, Inter-station Vectors, Rose Diagram")
        print("="*60)
        from array_analyzer import (
            analyze_array, visualize_comparison,
            create_circle_array, create_cross_array,
            create_propeller_array, create_spiral_arm_array
        )
        aperture = 4 * args.focus  # λ_max = 4 * depth
        
        # build comparison list: GA optimized + reference shapes
        kennett_arrays = [
            ("GA Optimized", phys_coords),
            ("Circle (12)",  create_circle_array(aperture / 2, 12)),
            ("Propeller 3×3", create_propeller_array(3, 3, aperture / 2)),
            ("Spiral 3a×4r",  create_spiral_arm_array(aperture / 2,
                                                       n_arms=3, n_rings=4,
                                                       span_deg=120)),
        ]
        
        kennett_results = []
        for kname, kcoords in kennett_arrays:
            print(f"  Analyzing {kname} ({len(kcoords)} sensors)...")
            kres = analyze_array(kcoords, kname, args.focus, args.vs)
            kennett_results.append(kres)
        
        # print table
        print("\n" + "-"*80)
        print(f"{'Array':<20} {'N':>4} {'r_min':>7} {'r_max':>7} "
              f"{'Bessel':>8} {'Isotropy':>9} {'SLL(dB)':>9}")
        print("-"*80)
        for kr in kennett_results:
            print(f"{kr['name']:<20} {kr['n_sensors']:>4} "
                  f"{kr['r_min']:>7.1f} {kr['r_max']:>7.1f} "
                  f"{kr['bessel_score']:>8.3f} {kr['isotropy_score']:>9.3f} "
                  f"{kr['sll_db']:>9.1f}")
        print("-"*80)
        
        ken_out = output_file.replace('.png', '_kennett.png')
        visualize_comparison(kennett_results, ken_out)
        print(f"✓ Kennett analysis saved to {ken_out}")

if __name__ == "__main__":
    main()
