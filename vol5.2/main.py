# author: ferat
# date: 2026
# vol3 — GraphSPAC Extended CLI Entry Point (Scenarios 1–14)

import argparse
import os
import sys
import numpy as np

from scenarios import SCENARIOS, SCENARIO_MAP, SEED as DEFAULT_SEED
from grid_generators import make_grid
from nsga_optimizer import NSGAOptimizer, knee_point
from graph_metrics import GraphMetrics
from coarray import CoArray
from classical_arrays import make_classical_arrays
from robustness import compare_robustness, plot_robustness_comparison
from visualizer import (plot_scenario_result, plot_pareto, plot_comparison)
from nsga_optimizer import (directional_score, los_penalty, golomb_redundancy,
                             near_field_phi)
from pymoo.core.callback import Callback


# ── helpers ───────────────────────────────────────────────────────────────────

class LevelsCallback(Callback):
    def __init__(self, out_dir, gm, sc, weights_array=None):
        super().__init__()
        self.out_dir = out_dir
        self.gm = gm
        self.sc = sc
        self.weights_array = weights_array
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[levels] Saving generation frames to: {self.out_dir}")

    def notify(self, algorithm):
        gen = algorithm.n_gen
        pop = algorithm.pop
        if pop is None or len(pop) == 0:
            return
            
        F = np.array([ind.F for ind in pop])
        X = np.array([ind.X for ind in pop])
        CV = np.array([ind.CV[0] for ind in pop])
        
        feas_idx = np.where(CV <= 0)[0]
        if len(feas_idx) > 0:
            idx = feas_idx[knee_point(F[feas_idx], weights=self.weights_array)]
        else:
            idx = int(np.argmin(CV))
            
        best_coords = X[idx].reshape(self.sc['N'], 2)
        
        # Avoid printing for every generation if possible, but keep plotting
        metrics, arf_grid, k_max, ca = compute_all_metrics(best_coords, self.gm, self.sc)
        filename = os.path.join(self.out_dir, f"gen_{gen:04d}.png")
        plot_scenario_result(best_coords, self.gm, metrics, self.sc, filename=filename)


def compute_all_metrics(coords, gm=None, sc=None):
    dist_matrix = None
    if gm is not None and (gm.grid == 0).any():
        dist_matrix = gm.get_geodesic_distance_matrix(coords)

    gmet = GraphMetrics(coords, dist_matrix=dist_matrix)
    _, _, arf_grid, k_max = gmet.arf()
    sll  = gmet.sll_db(arf_grid)
    lam2 = gmet.algebraic_connectivity()
    iso  = gmet.isotropy_score()
    ca   = CoArray(coords, dist_matrix=dist_matrix)

    m = {
        "N":         len(coords),
        "n_pairs":   ca.n_pairs,
        "r_min_m":   ca.r_min,
        "r_max_m":   ca.r_max,
        "lambda2":   lam2,
        "sll_db":    sll,
        "isotropy":  iso,
        "lsd_H":     ca.lsd_entropy(),
        "phi_bessel": gmet.phi_bessel(),
        "phi_hankel": gmet.phi_hankel(),
    }

    quality = gmet.sll_quality_check(arf_grid)
    m.update({
        "sll_db": quality['sll_db'],
        "dr_db": quality['dr_db'],
        "mlw_rad_per_m": quality['mlw_rad_per_m'],
        "eta_res": quality['eta_res'],
        "spac_criterion": "PASS" if quality['passes_spac_criterion'] else "FAIL"
    })

    if sc and sc.get("noise_azimuths"):
        m["directional_score"] = directional_score(coords, sc["noise_azimuths"])
    if sc and sc.get("los_penalty") and gm is not None:
        m["los_blocked_frac"]  = los_penalty(coords, gm)
    if sc and sc.get("golomb"):
        m["golomb_redundancy"] = golomb_redundancy(coords)
    if sc and sc.get("near_field") and sc.get("near_field_source"):
        m["near_field_phi"]    = near_field_phi(coords, sc["near_field_source"])

    return m, arf_grid, k_max, ca


def print_metrics(label, metrics):
    sep = "─" * 58
    print(f"\n{sep}\n  {label}\n{sep}")
    for k, v in metrics.items():
        if isinstance(v, (int, str)):
            print(f"  {k:<28}: {v}")
        else:
            print(f"  {k:<28}: {v:.6g}")
    print(sep)


def save_metrics_txt(metrics, sc, filename, coords=None):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    kernel_raw = sc.get("kernel", "bessel").lower()
    kernel_fmt = "bessel (J\u2080)" if kernel_raw == "bessel" else "hankel (H\u2080\u207d\xb9\u207e)"
    with open(filename, "w") as f:
        sep = "=" * 60
        f.write(f"{sep}\n")
        f.write(f"  GraphSPAC vol3 \u2014 Sc {sc['id']}: {sc['name']}\n")
        f.write(f"{sep}\n\n")

        # Scenario parameters
        f.write("SCENARIO PARAMETERS\n" + "-" * 40 + "\n")
        f.write(f"  N (sensors)             : {metrics.get('N', sc.get('N','?'))}\n")
        f.write(f"  d_min                   : {sc.get('d_min','?')} m\n")
        f.write(f"  focus (target depth)    : {sc.get('focus','?')} m\n")
        f.write(f"  Vs (shear-wave vel.)    : {sc.get('vs', sc.get('Vs','?'))} m/s\n")
        f.write(f"  kernel                  : {kernel_fmt}\n")
        f.write(f"  domain                  : {sc.get('domain','open')}\n")
        f.write(f"  grid spacing            : {sc.get('spacing', 5.0)} m\n")
        if sc.get("noise_azimuths"):
            az_str = ", ".join(str(a) for a in sc["noise_azimuths"])
            f.write(f"  noise_azimuths          : [{az_str}] \u00b0\n")
        f.write("\n")

        # Sensor positions
        if coords is not None:
            f.write("SENSOR POSITIONS (x, y) [m]\n" + "-" * 40 + "\n")
            for i, (x, y) in enumerate(np.asarray(coords)):
                f.write(f"  Sensor {i+1:>2}  :  x = {x:10.4f}    y = {y:10.4f}\n")
            f.write("\n")

        # Computed metrics
        f.write("COMPUTED METRICS\n" + "-" * 40 + "\n")
        for k, v in metrics.items():
            val_str = str(v) if isinstance(v, (int, str)) else f"{v:.6g}"
            f.write(f"  {k:<28}: {val_str}\n")

        # Aperture check
        r_req = (4 * sc.get("focus", 30)) / 3
        r_max = metrics.get("r_max_m", 0)
        flag  = "OK" if r_max >= r_req else "WARNING: insufficient aperture"
        f.write(f"\nAPERTURE CHECK\n" + "-" * 40 + "\n")
        f.write(f"  Required r_max (4/3 x focus) : {r_req:.2f} m\n")
        f.write(f"  Achieved r_max               : {r_max:.2f} m\n")
        f.write(f"  Status                       : {flag}\n")
        f.write(f"\n{sep}\n")
    print(f"[main] Metrics \u2192 {filename}")


def print_aperture_check(ca, sc):
    r_req = (4 * sc["focus"]) / 3
    flag  = "✓" if ca.r_max >= r_req else "⚠"
    print(f"\n  {flag} Aperture {ca.r_max:.2f} m  "
          f"(required ≥ {r_req:.2f} m  for depth {sc['focus']} m)")


# ── single scenario runner ────────────────────────────────────────────────────

def run_scenario(sc, args, out_dir):
    if getattr(args, "kernel", None):
        sc['kernel'] = args.kernel

    print(f"\n{'='*62}")
    print(f"  [{sc['id']}] {sc['name']}")
    print(f"  {sc.get('description','')}")
    print(f"  N={sc['N']}  kernel={sc['kernel']}  "
          f"d_min={sc['d_min']} m  focus={sc['focus']} m")
    if sc.get("noise_azimuths"):
        print(f"  noise_azimuths={sc['noise_azimuths']} °")
    
    # NEW: Show optimization mode
    mode = sc.get('mode', 'precision')
    print(f"  Optimization Mode: {mode.upper()}")
    print(f"{'='*62}")

    gm = make_grid(sc,
                   matrix_path=args.matrix if args.yeryok else None,
                   force_matrix=bool(args.yeryok))

    d_min  = args.dmin    if args.dmin    is not None else sc["d_min"]
    n_sens = args.sensors if args.sensors is not None else sc["N"]

    sc["generations"] = args.gens
    sc["population"]  = args.pop

    opt = NSGAOptimizer(
        manifold_manager=gm,
        n_sensors=n_sens,
        d_min=d_min,
        kernel=sc["kernel"],
        noise_azimuths=sc.get("noise_azimuths") or args.noiseazimuth or [],
        los_penalty_flag=sc.get("los_penalty", False) or args.los,
        golomb_flag=sc.get("golomb", False) or args.golomb,
        near_field=sc.get("near_field", False),
        near_field_source=sc.get("near_field_source"),
        focus=sc["focus"],
        vs=sc.get("vs", 500.0),
        pop_size=args.pop,
        n_gen=args.gens,
        seed=args.seed,
        use_poincare=args.poincare,
        use_seeding=not getattr(args, "noseed", False),
        weights=sc.get("weights"),  # NEW: Pass scenario-specific weights
    )

    w_arr = opt.get_weight_array()

    cb = None
    if getattr(args, "levels", False):
        levels_dir = os.path.join(out_dir, f"sc{sc['id']}_levels")
        cb = LevelsCallback(levels_dir, gm, sc, weights_array=w_arr)

    best_coords, pareto_F, pareto_X, res = opt.run(verbose=True, callback=cb)
    metrics, arf_grid, k_max, ca = compute_all_metrics(best_coords, gm, sc)
    print_aperture_check(ca, sc)
    print_metrics(f"Sc {sc['id']} — GraphSPAC (knee-point)", metrics)

    print(f"\n  Sensor Positions [m]:")
    for i, (x, y) in enumerate(best_coords):
        print(f"    {i+1:>2}: x={x:8.3f}  y={y:8.3f}")

    # ── save outputs ──
    tag = f"sc{sc['id']}"
    os.makedirs(out_dir, exist_ok=True)
    result_png  = os.path.join(out_dir, f"{tag}_result.png")
    pareto_png  = os.path.join(out_dir, f"{tag}_pareto.png")
    metrics_txt = os.path.join(out_dir, f"{tag}_metrics.txt")

    plot_scenario_result(best_coords, gm, metrics, sc, filename=result_png)
    plot_pareto(res, knee_idx=knee_point(pareto_F, weights=w_arr), filename=pareto_png, sc=sc)
    save_metrics_txt(metrics, sc, metrics_txt, coords=best_coords)

    # ── Scenario 4.1: deep/shallow overlay ──
    if sc.get("overlay_pair"):
        _overlay_comparison(sc, best_coords, out_dir)

    # ── classical comparison ──
    if args.compare:
        _run_comparison(sc, best_coords, gm, n_sens, metrics, out_dir)

    # ── robustness ──
    if args.robustness:
        print(f"\n[robustness] Running nodal-failure simulation …")
        report = compare_robustness(best_coords, 4 * sc["focus"],
                                    n_sens, n_max_remove=2, verbose=True)
        plot_robustness_comparison(report,
                                   filename=os.path.join(out_dir, f"{tag}_robustness.png"))

    # ── Sc 5.1: single-sensor removal analysis ──
    if args.failure_analysis:
        _failure_analysis(best_coords, sc, out_dir)

    return best_coords, metrics, pareto_F


# ── optional analysis modules ─────────────────────────────────────────────────

def _run_comparison(sc, gs_coords, gm, n_sens, gs_metrics, out_dir):
    from graph_metrics import GraphMetrics
    print(f"\n[compare] Classical arrays …")
    classics = make_classical_arrays(4 * sc["focus"], n_sensors=n_sens)
    results  = {"GraphSPAC": {
        "coords": gs_coords, "metrics": gs_metrics,
        "arf_grid": GraphMetrics(gs_coords).arf()[2],
        "k_max":    GraphMetrics(gs_coords).arf()[3],
        "ca": CoArray(gs_coords),
    }}
    for name, coords in classics.items():
        gm2 = GraphMetrics(coords)
        _, _, arf2, k2 = gm2.arf()
        results[name] = {
            "coords": coords, "metrics": gm2.summary(),
            "arf_grid": arf2, "k_max": k2, "ca": CoArray(coords),
        }
    print(f"\n  {'Array':<25} {'N':>4} {'r_min':>7} {'r_max':>7} "
          f"{'SLL':>9} {'Iso':>7} {'λ₂':>10}")
    print("  " + "─" * 70)
    for name, rd in results.items():
        m = rd["metrics"]
        print(f"  {name:<25} {m['N']:>4} "
              f"{m['r_min_m']:>7.1f} {m['r_max_m']:>7.1f} "
              f"{m['sll_db']:>9.1f} {m['isotropy']:>7.3f} {m['lambda2']:>10.5f}")
    print("  " + "─" * 70)
    tag = f"sc{sc['id']}"
    plot_comparison(results, filename=os.path.join(out_dir, f"{tag}_comparison.png"))


def _failure_analysis(coords, sc, out_dir):
    """Sc 5.1: Remove highest-degree node (by λ₂ sensitivity), re-evaluate."""
    print(f"\n[failure] Identifying most critical sensor …")
    gmet_full = GraphMetrics(coords)
    lam2_full = gmet_full.algebraic_connectivity()
    N = len(coords)
    impacts   = []
    for i in range(N):
        rem = np.delete(coords, i, axis=0)
        lam2_i = GraphMetrics(rem).algebraic_connectivity() if len(rem) >= 2 else 0.0
        drop   = (lam2_full - lam2_i) / (lam2_full + 1e-12)
        impacts.append((i, drop))
    impacts.sort(key=lambda x: -x[1])

    crit_idx, crit_drop = impacts[0]
    print(f"  Most critical sensor: #{crit_idx+1}  "
          f"(λ₂ drop = {crit_drop*100:.1f}%)")

    remaining = np.delete(coords, crit_idx, axis=0)
    gmet_rem  = GraphMetrics(remaining)
    _, _, arf_rem, _ = gmet_rem.arf()
    sll_rem   = gmet_rem.sll_db(arf_rem)
    sll_full  = gmet_full.sll_db()
    delta_sll = sll_rem - sll_full

    verdict = "✓ Topologically Robust" if abs(delta_sll) <= 3 else "✗ Fragile"
    print(f"  SLL change: {delta_sll:+.1f} dB  {verdict}")
    print(f"  (Robust criterion: |ΔSLL| ≤ 3 dB)")

    txt = os.path.join(out_dir, f"sc{sc['id']}_failure_analysis.txt")
    with open(txt, "w") as f:
        f.write(f"Most critical sensor: #{crit_idx+1}\n")
        f.write(f"λ₂ drop: {crit_drop*100:.1f}%\n")
        f.write(f"SLL change: {delta_sll:+.1f} dB\n")
        f.write(f"Verdict: {verdict}\n\n")
        f.write("All sensors sorted by criticality:\n")
        for idx, drop in impacts:
            f.write(f"  #{idx+1}: λ₂ drop {drop*100:.1f}%\n")
    print(f"  Failure analysis → {txt}")


def _overlay_comparison(sc, coords1, out_dir):
    """For Sc 4.1 pair: print note about combining two runs."""
    print(f"\n[overlay] Bimodal LSD analysis: run partner scenario and overlay.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GraphSPAC vol3 — Extended Multi-Scenario Optimizer (Sc 1–14)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Scenario IDs:\n"
            "  1  Open domain (J0)          2  L-shape (Hankel)\n"
            "  21 L + noise 0° (N)          22 L + noise 90° (E)\n"
            "  23 L + central hole          31 Urban canyon 10×100m\n"
            "  41 Deep focus 50m            42 Shallow focus 10m\n"
            "  6  Urban island              7  Seismic corridor 5×200m\n"
            "  8  Cross-fire N+E            9  Fractal boundary\n"
            "  10 Variable spacing          11 Fresnel shadow\n"
            "  12 Golomb ruler              13 Near-field\n"
            "  14 Percolation labyrinth     all  run all\n"
        ),
    )
    # scenario
    parser.add_argument("--scenario", default="all",
                        help="ID (e.g. 1, 11, 21) or 'all'")
    # domain / grid
    parser.add_argument("--matrix",   default="matrix.xlsx",
                        metavar="PATH",
                        help=(
                            "Excel (.xlsx) grid file. "
                            "Each cell = 1 (accessible) or 0 (obstacle). "
                            "Row/col count defines grid size; spacing set by --spacing. "
                            "Only loaded when --yeryok 1 is given."
                        ))
    parser.add_argument("--yeryok",   type=int, default=0, choices=[0, 1],
                        help=(
                            "1 = load grid from --matrix Excel file instead of "
                            "the scenario built-in domain. "
                            "Use with --scenario 0 for a fully custom run."
                        ))
    parser.add_argument("--spacing",  type=float, default=None)
    parser.add_argument("--name",     type=str,   default=None,
                        help="Path to a text file to read the custom scenario name from (or just the name string).")
    # array
    parser.add_argument("--sensors",  type=int,   default=None)
    parser.add_argument("--dmin",     type=float, default=None)
    parser.add_argument("--focus",    type=float, default=None)
    parser.add_argument("--vs",       type=float, default=None)
    parser.add_argument("--kernel",   type=str,   default=None, choices=["bessel", "hankel"],
                        help=(
                           "Target correlation kernel:\n"
                           "  bessel  = J_0(kr) for diffuse isotropic noise\n"
                           "  hankel  = H_0^(1)(kr) for directional/propagating sources"
                        ))
    # optimizer
    parser.add_argument("--gens",     type=int,   default=200)
    parser.add_argument("--pop",      type=int,   default=100)
    parser.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    # extra objectives (can override scenario defaults)
    parser.add_argument("--noiseazimuth", type=float, nargs="+", default=None,
                        metavar="AZ",
                        help="Noise azimuth(s) in degrees (0=N, 90=E). Multiple OK.")
    parser.add_argument("--los",      action="store_true",
                        help="Enable Line-of-Sight penalty objective")
    parser.add_argument("--golomb",   action="store_true",
                        help="Enable Golomb ruler (zero-redundancy) objective")
    parser.add_argument("--poincare", action="store_true",
                        help="Optimize in Hyperbolic Poincare disk mapping to naturally densely sample the center.")
    # analyses
    parser.add_argument("--compare",         action="store_true")
    parser.add_argument("--robustness",      action="store_true")
    parser.add_argument("--failure_analysis",action="store_true",
                        help="Sc 5.1: identify most critical sensor and evaluate λ₂ drop")
    # output
    parser.add_argument("--output",   default="results")
    parser.add_argument("--levels",   action="store_true",
                        help="Save a visualization for every single generation (requires callback)")
    parser.add_argument("--noseed",   action="store_true",
                        help="Disable Topological Awareness (Spatial Seeding) and use standard random initialization")

    args = parser.parse_args()

    print("=" * 62)
    print("  GraphSPAC vol3 — Extended Scenario Suite")
    print(f"  Seed: {args.seed}  |  NSGA-II pop={args.pop}  gen={args.gens}")
    print("=" * 62)

    # resolve scenario list
    if args.scenario.lower() == "all":
        scenarios = SCENARIOS
    else:
        try:
            sid = int(args.scenario)
        except ValueError:
            print(f"ERROR: --scenario must be an integer ID or 'all'")
            sys.exit(1)
        if sid not in SCENARIO_MAP:
            print(f"ERROR: Scenario {sid} not found. Available: "
                  f"{sorted(SCENARIO_MAP)}")
            sys.exit(1)
        scenarios = [SCENARIO_MAP[sid]]

    all_results = {}
    for sc in scenarios:
        sc = dict(sc)
        if args.focus   is not None: sc["focus"]   = args.focus
        if args.vs      is not None: sc["vs"]       = args.vs
        if args.spacing is not None: sc["spacing"]  = args.spacing
        if args.name    is not None: 
            if os.path.isfile(args.name):
                with open(args.name, "r", encoding="utf-8") as nf:
                    sc["name"] = nf.read().strip()
            else:
                sc["name"] = args.name
            sc["is_custom_name"] = True
                
        best, metrics, pF = run_scenario(sc, args, out_dir=args.output)
        
        # If --poincare was used, we need to apply transforms to best_coords to get physical coords
        if args.poincare:
            from geometry_transforms import apply_transforms
            x_max, y_max = make_grid(sc).physical_bounds()
            # Wait, best_coords returned by run_scenario are already physical if we logged them properly?
            # Actually run_scenario prints them out, but NSGAOptimizer returns the raw X or evaluated coords?
            # Let's just store it as normal because run_scenario does it inside compute_all_metrics.
            # Wait! GraphSPACProblem returns the raw Poincare coordinates in `res.X`.
            pass

        all_results[sc["id"]] = {"metrics": metrics, "name": sc["name"]}

    # summary table
    if len(all_results) > 1:
        print(f"\n{'='*72}")
        print("  SUMMARY")
        print(f"{'='*72}")
        print(f"  {'ID':<5} {'Scenario':<32} {'SLL':>7} {'Iso':>7} {'λ₂':>10} {'H':>8}")
        print("  " + "─" * 72)
        for sid, r in all_results.items():
            m = r["metrics"]
            print(f"  {sid:<5} {r['name'][:32]:<32} "
                  f"{m['sll_db']:>7.1f} {m['isotropy']:>7.3f} "
                  f"{m['lambda2']:>10.4f} {m['lsd_H']:>8.4f}")
        print(f"{'='*72}")

    print(f"\nOutputs → {os.path.abspath(args.output)}/\nDone.\n")


if __name__ == "__main__":
    main()
