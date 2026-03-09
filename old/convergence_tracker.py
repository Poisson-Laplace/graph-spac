# author: ferat
# date: 2026
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from grid_manager import GridManager
from spac_optimizer import SPACOptimizer
from spac_graph import SPACGraph, SPACResponseAnalyzer

# ─── Instrumented optimizer ───────────────────────────────────────────────────

class InstrumentedOptimizer(SPACOptimizer):
    

    def fitness_decomposed(self, individual):
        
        phys_coords = self._indices_to_physical(individual)
        graph = SPACGraph(phys_coords)

        if not self._check_d_min(individual):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if graph.r_max < 1e-6:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        analyzer = SPACResponseAnalyzer(
            graph,
            target_depth=self.target_depth,
            vs_halfspace=self.vs_halfspace
        )

        bessel_score, _ = analyzer.compute_bessel_coverage(freq_range=None)
        depth_penalty = analyzer.compute_depth_penalty()
        depth_score = 1.0 - depth_penalty

        k_min = 2 * np.pi / (graph.r_max * 4) if graph.r_max > 0 else 0.01
        k_max = 2 * np.pi / graph.r_min if graph.r_min > 0 else 1.0
        k_max = min(k_max, 0.5)
        k_range = np.linspace(-k_max, k_max, 50)
        kx, ky, arf = analyzer.compute_arf(k_range)

        center_idx = arf.shape[0] // 2
        peak_val = arf[center_idx, center_idx]
        n = arf.shape[0]
        y, x = np.ogrid[:n, :n]
        center_mask = ((x - n // 2) ** 2 + (y - n // 2) ** 2) <= (n // 8) ** 2
        side_lobes = arf.copy()
        side_lobes[center_mask] = 0
        max_side_lobe = np.max(side_lobes)
        sll_db = -20 * np.log10(max_side_lobe / peak_val + 1e-10) if peak_val > 0 else 0
        kspace_score = min(sll_db / 20.0, 1.0)

        azimuth_score, _ = analyzer.compute_azimuthal_coverage(self.focus_direction)
        focus_score = analyzer.compute_focus_area_density(self.focus_point)

        total = (0.35 * bessel_score +
                 0.25 * depth_score +
                 0.20 * kspace_score +
                 0.15 * azimuth_score +
                 0.05 * focus_score)

        return total, bessel_score, depth_score, kspace_score, azimuth_score, focus_score

    def run_instrumented(self, verbose=True):
        
        import random

        self.initialize_population()

        best_ever = None
        best_fitness_ever = -1

        history = {
            "gen": [],
            "best_fitness": [],
            "mean_fitness": [],
            "best_bessel": [],
            "best_depth": [],
            "best_kspace": [],
            "best_azimuth": [],
            # objective-space clouds: list of arrays (one per sampled generation)
            "cloud_gen": [],
            "cloud_bessel": [],
            "cloud_kspace": [],
        }

        sample_gens = set(range(0, self.generations, max(1, self.generations // 10)))
        sample_gens.add(self.generations - 1)

        for gen in range(self.generations):
            # evaluate
            decomp = [self.fitness_decomposed(ind) for ind in self.population]
            totals = [d[0] for d in decomp]

            max_fit = max(totals)
            best_idx = totals.index(max_fit)
            best_d = decomp[best_idx]

            if max_fit > best_fitness_ever:
                best_fitness_ever = max_fit
                best_ever = self.population[best_idx][:]

            history["gen"].append(gen)
            history["best_fitness"].append(max_fit)
            history["mean_fitness"].append(float(np.mean(totals)))
            history["best_bessel"].append(best_d[1])
            history["best_depth"].append(best_d[2])
            history["best_kspace"].append(best_d[3])
            history["best_azimuth"].append(best_d[4])

            if gen in sample_gens:
                history["cloud_gen"].extend([gen] * len(decomp))
                history["cloud_bessel"].extend([d[1] for d in decomp])
                history["cloud_kspace"].extend([d[3] for d in decomp])

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:3d}: F={max_fit:.4f}  "
                      f"Bessel={best_d[1]:.3f}  Depth={best_d[2]:.3f}  "
                      f"kspace={best_d[3]:.3f}  Az={best_d[4]:.3f}")

            # elitism + next generation (mirrors SPACOptimizer.run logic)
            sorted_pop = sorted(zip(self.population, totals),
                                key=lambda x: x[1], reverse=True)
            elites = [ind[:] for ind, _ in sorted_pop[:self.elite_count]]

            import random as _rnd
            parents = self.select_parents(totals)
            next_gen = elites[:]
            while len(next_gen) < self.pop_size:
                p1, p2 = _rnd.sample(parents, 2)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                next_gen.extend([c1, c2])
            self.population = next_gen[:self.pop_size]

        return best_ever, best_fitness_ever, history

# ─── Figure 1: Convergence curves ────────────────────────────────────────────

def plot_convergence(history, scenario_name, filename="convergence_history.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"GA Convergence — {scenario_name}", fontsize=13, fontweight='bold')

    gens = history["gen"]

    # top panel: overall fitness
    ax = axes[0]
    ax.plot(gens, history["best_fitness"], color='#2563EB', lw=2.0, label='Best Fitness')
    ax.plot(gens, history["mean_fitness"], color='#93C5FD', lw=1.2,
            ls='--', label='Mean Fitness')
    ax.fill_between(gens, history["mean_fitness"], history["best_fitness"],
                    alpha=0.12, color='#2563EB')
    ax.set_ylabel('Total Fitness', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # mark stabilization generation (where improvement < 0.001 for 5 gens)
    best = np.array(history["best_fitness"])
    stable_gen = None
    for i in range(5, len(best)):
        if np.all(np.abs(np.diff(best[i-5:i+1])) < 0.001):
            stable_gen = i - 4
            break
    if stable_gen is not None:
        ax.axvline(stable_gen, color='red', ls=':', lw=1.5, alpha=0.8,
                   label=f'Stabilizes @ gen {stable_gen}')
        ax.legend(fontsize=9)

    # bottom panel: sub-objectives
    ax2 = axes[1]
    colors = {'Bessel Coverage': ('#16A34A', history["best_bessel"]),
              'Depth Score':      ('#eA580C', history["best_depth"]),
              'k-space (SLL)':    ('#7C3AED', history["best_kspace"]),
              'Azimuthal':        ('#dB2777', history["best_azimuth"])}

    for label, (color, vals) in colors.items():
        ax2.plot(gens, vals, color=color, lw=1.6, label=label)

    ax2.set_xlabel('Generation', fontsize=10)
    ax2.set_ylabel('Sub-objective Score', fontsize=10)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved convergence figure → {filename}")
    return fig

# ─── Figure 2: Objective Space (Pareto Cloud) ────────────────────────────────

def plot_objective_space(history, scenario_name, filename="objective_space.png"):
    cloud_gen    = np.array(history["cloud_gen"])
    cloud_bessel = np.array(history["cloud_bessel"])
    cloud_kspace = np.array(history["cloud_kspace"])

    if len(cloud_gen) == 0:
        print("  No cloud data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Objective Space — {scenario_name}\n"
                 f"Bessel Coverage vs k-space Score, coloured by generation",
                 fontsize=12, fontweight='bold')

    gen_norm = (cloud_gen - cloud_gen.min()) / (cloud_gen.max() - cloud_gen.min() + 1e-10)
    scatter = ax.scatter(cloud_bessel, cloud_kspace,
                         c=gen_norm, cmap='plasma', s=18, alpha=0.6, lw=0)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalised Generation (early → late)', fontsize=9)
    gen_vals = np.unique(cloud_gen)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels([f'Gen {int(gen_vals[0])}',
                          f'Gen {int(gen_vals[len(gen_vals)//2])}',
                          f'Gen {int(gen_vals[-1])}'])

    # mark the final generation's best point
    last_gen_mask = cloud_gen == cloud_gen.max()
    best_in_last = np.argmax(cloud_kspace[last_gen_mask])
    bx = cloud_bessel[last_gen_mask][best_in_last]
    by = cloud_kspace[last_gen_mask][best_in_last]
    ax.scatter([bx], [by], s=120, c='red', marker='*', zorder=10,
               label=f'Best (gen {int(cloud_gen.max())})')

    ax.set_xlabel('Bessel Coverage Score $F_{\\mathrm{Bessel}}$', fontsize=11)
    ax.set_ylabel('k-space Quality Score $F_{k-space}$', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    # annotate trade-off direction
    ax.annotate('Population converges\ntoward Pareto Front',
                xy=(0.72, 0.72), xytext=(0.4, 0.55),
                fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved objective-space figure → {filename}")
    return fig

# ─── Main ─────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {"id": 1, "name": "Shallow (10m)",     "focus": 10, "vs": 300, "sensors": 6,  "gridsize": 15},
    {"id": 2, "name": "Medium (30m)",      "focus": 30, "vs": 500, "sensors": 10, "gridsize": 30},
    {"id": 3, "name": "Deep (60m)",        "focus": 60, "vs": 700, "sensors": 13, "gridsize": 50},
    {"id": 4, "name": "Focal Target",      "focus": 30, "vs": 500, "sensors": 10, "gridsize": 30},
    {"id": 5, "name": "N-Source Adapted",  "focus": 30, "vs": 500, "sensors": 10, "gridsize": 30},
]

def main():
    parser = argparse.ArgumentParser(description="Graphspac Convergence Tracker")
    parser.add_argument('--scenario', type=int, default=2,
                        help='Scenario ID to run (1-5, default: 2)')
    parser.add_argument('--gens', type=int, default=80,
                        help='GA generations (default: 80)')
    parser.add_argument('--pop',  type=int, default=60,
                        help='GA population size (default: 60)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    sc = next(s for s in SCENARIOS if s["id"] == args.scenario)
    print(f"\n{'='*60}")
    print(f"  Convergence Tracker - Scenario {sc['id']}: {sc['name']}")
    print(f"  depth={sc['focus']}m  Vs={sc['vs']}m/s  N={sc['sensors']}")
    print(f"  Generations={args.gens}  Population={args.pop}")
    print(f"{'='*60}")

    gm = GridManager(None, grid_spacing=5.0, default_grid_size=sc["gridsize"])

    opt = InstrumentedOptimizer(
        gm,
        n_sensors=sc["sensors"],
        target_depth=sc["focus"],
        vs_halfspace=sc["vs"],
        focus_point=(75.0, 75.0) if sc["id"] == 4 else None,
        focus_direction=0.0 if sc["id"] == 5 else None,
        population_size=args.pop,
        generations=args.gens,
        random_seed=args.seed,
    )

    best, fitness, history = opt.run_instrumented(verbose=True)

    tag = f"sc{sc['id']}"
    plot_convergence(history, sc["name"], f"convergence_{tag}.png")
    plot_objective_space(history, sc["name"], f"objective_space_{tag}.png")

    print(f"\n  Best fitness achieved: {fitness:.4f}")
    print("Done.")

if __name__ == "__main__":
    main()
