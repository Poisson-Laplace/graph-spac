# author: ferat
# date: 2026
# vol3 — Visualizer
# Five-panel result figure, Pareto scatter, and robustness bar chart.

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "figure.dpi":   600,
})


# ══════════════════════════════════════════════════════════════════════════════
#  Five-panel scenario result figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_scenario_result(coords, gm, metrics, sc, pareto_F=None,
                         filename="result.png"):
    """
    Saves five separate figures for the scenario result to be used in LaTeX:
      (a) Array layout
      (b) Co-array scatter
      (c) ARF — linear scale
      (d) ARF — dB scale
      (e) Rose diagram (azimuthal distribution)

    Parameters
    ----------
    coords   : (N, 2) ndarray — sensor positions [m]
    gm       : GridManager
    metrics  : dict from GraphMetrics.summary()
    sc       : scenario dict
    pareto_F : (M, 3) ndarray | None — Pareto front objectives
    filename : str
    """
    from graph_metrics import GraphMetrics
    from coarray import CoArray

    coords = np.asarray(coords)
    gmet = GraphMetrics(coords)
    ca   = CoArray(coords)

    _, _, arf_grid, k_max_used = gmet.arf()
    iso  = gmet.isotropy_score()

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    base, ext = os.path.splitext(filename)

    # ── (a) Array Layout ──
    fig, ax0 = plt.subplots(figsize=(4.5, 4.5))
    _plot_array_layout(ax0, coords, gm)
    ax0.set_title("(a)", loc="left", fontsize=12, fontweight="bold")
    plt.savefig(f"{base}_a_layout{ext}", dpi=600, bbox_inches="tight")
    plt.close()

    # ── (b) Co-array (Δ) ──
    fig, ax1 = plt.subplots(figsize=(4.5, 4.5))
    _plot_coarray(ax1, ca)
    ax1.set_title("(b)", loc="left", fontsize=12, fontweight="bold")
    plt.savefig(f"{base}_b_coarray{ext}", dpi=600, bbox_inches="tight")
    plt.close()

    # ── (c) ARF linear ──
    fig, ax2 = plt.subplots(figsize=(4.5, 4.5))
    im2 = ax2.imshow(arf_grid.T, extent=[-k_max_used, k_max_used,
                                          -k_max_used, k_max_used],
                     origin="lower", cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_title("(c)", loc="left", fontsize=12, fontweight="bold")
    ax2.set_xlabel("kx (rad/m)", fontsize=10)
    ax2.set_ylabel("ky (rad/m)", fontsize=10)
    ax2.tick_params(labelsize=9)
    plt.savefig(f"{base}_c_arf_lin{ext}", dpi=600, bbox_inches="tight")
    plt.close()

    # ── (d) ARF dB ──
    fig, ax3 = plt.subplots(figsize=(4.5, 4.5))
    arf_db = 20 * np.log10(arf_grid + 1e-9)
    arf_db -= arf_db.max()
    im3 = ax3.imshow(arf_db.T, extent=[-k_max_used, k_max_used,
                                         -k_max_used, k_max_used],
                     origin="lower", cmap="seismic_r", vmin=-40, vmax=0)
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label("dB", fontsize=10)
    ax3.set_title("(d)", loc="left", fontsize=12, fontweight="bold")
    ax3.set_xlabel("kx (rad/m)", fontsize=10)
    ax3.set_ylabel("ky (rad/m)", fontsize=10)
    ax3.tick_params(labelsize=9)
    plt.savefig(f"{base}_d_arf_db{ext}", dpi=600, bbox_inches="tight")
    plt.close()

    # ── (e) Rose diagram ──
    fig = plt.figure(figsize=(4.5, 4.5))
    ax4 = fig.add_subplot(111, projection="polar")
    _plot_rose(ax4, ca, iso)
    ax4.set_title("(e)", loc="left", fontsize=12, fontweight="bold")
    plt.savefig(f"{base}_e_rose{ext}", dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[visualizer] Saved 5 separated figures → {base}_[a-e]_...{ext}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _plot_array_layout(ax, coords, gm):
    from matplotlib.colors import ListedColormap
    grid    = gm.grid
    spacing = gm.grid_spacing
    rows, cols = grid.shape

    # white=accessible (1), red=obstacle (0)
    img = grid.astype(float)   # 1=white, 0=obstacle
    cmap_wr = ListedColormap(["#e02c2c", "white"])   # 0→red, 1→white
    extent = [0, cols * spacing, rows * spacing, 0]
    ax.imshow(img, cmap=cmap_wr, origin="upper", extent=extent,
              vmin=0, vmax=1, alpha=0.9, aspect="equal")

    # edges
    N = len(coords)
    for i in range(N):
        for j in range(i + 1, N):
            ax.plot([coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color="#4a90d9", alpha=0.25, lw=0.8, zorder=1)

    ax.scatter(coords[:, 0], coords[:, 1],
               s=80, c="#2563EB", edgecolors="white", lw=1.2, zorder=5)

    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i + 1), (x, y), fontsize=7, ha="center", va="center",
                    color="white", fontweight="bold", zorder=6)

    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def _plot_coarray(ax, ca):
    dvecs = ca.diff_vecs
    # dx = eastward, dy_grid = southward → negate dy so north = up
    all_vx = np.concatenate([ dvecs[:, 0], -dvecs[:, 0]])
    all_vy = np.concatenate([-dvecs[:, 1],  dvecs[:, 1]])  # ← negated: north-up
    lim = ca.r_max * 1.1
    ms = max(1, min(6, 300 // ca.n_pairs))
    ax.scatter(all_vx, all_vy, s=ms, alpha=0.5, c="darkorange", lw=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # reference circles
    for r, col in [(ca.r_min, "red"), (ca.r_max, "green")]:
        circ = plt.Circle((0, 0), r, fill=False, color=col, lw=0.8, ls="--", alpha=0.7)
        ax.add_patch(circ)

    ax.set_xlabel("Δx (East) [m]", fontsize=8)
    ax.set_ylabel("Δy (North) [m]", fontsize=8)
    ax.tick_params(labelsize=7)


def _plot_rose(ax, ca, iso):
    """
    Rose diagram in geographic convention (N=up, CW=positive).
    Bars are displayed with a single solid color.
    """
    geo_folded = ca.azimuths_geo_folded()   # geographic [0, pi), 0=N, pi/2=E

    n_bins    = 18
    half_bin  = np.pi / (2 * n_bins)          # 5 degrees in radians
    
    # Calculate histogram counts
    hist, bin_edges = np.histogram(
        geo_folded, bins=n_bins,
        range=(-half_bin, np.pi - half_bin)
    )
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width   = np.pi / n_bins
    color = "steelblue"

    for i in range(n_bins):
        if hist[i] > 0:
            ax.bar(centers[i], hist[i], width, alpha=0.85, color=color,
                   edgecolor="white", lw=0.5)
            ax.bar(centers[i] + np.pi, hist[i], width, alpha=0.85, color=color,
                   edgecolor="white", lw=0.5)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(labelsize=6)


# ══════════════════════════════════════════════════════════════════════════════
#  Pareto front plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_pareto(res_or_pareto_F, knee_idx=None, filename="pareto.png", sc=None):
    """
    3D scatter of Pareto front objectives from 3 different viewing angles.
    If res_or_pareto_F has a 'history', it will plot all points across
    generations and color them by generation using the 'plasma' colormap.
    
    res_or_pareto_F : (M, 3) ndarray OR pymoo optimization result object
    """
    from mpl_toolkits.mplot3d import Axes3D   # noqa
    import matplotlib.cm as cm

    # Identify if we have history
    has_history = False
    if hasattr(res_or_pareto_F, "history") and res_or_pareto_F.history:
        has_history = True
        history = res_or_pareto_F.history
        n_gens = len(history)
        
        # Extract all individuals from all generations
        all_xs, all_ys, all_zs, all_gens = [], [], [], []
        
        # Storage for 1D population metrics
        all_comps = { "f1_lsd": [], "f2_sll": [], "f3_lam2": [], "f4_graph": [], "f5_dr": [], "f6_eta": [], "f7_ds": [] }
        all_gens_pop, all_xs_pop = [], []

        for gen_idx, algo in enumerate(history):
            if algo.opt is not None and len(algo.opt) > 0:
                F = np.array([ind.F for ind in algo.opt])
                all_xs.extend(F[:, 0])
                if F.shape[1] >= 2:
                    all_ys.extend(F[:, 1])
                else:
                    all_ys.extend(np.zeros_like(F[:, 0]))
                    
                if F.shape[1] >= 3:
                    all_zs.extend(F[:, 2])
                else:
                    all_zs.extend(np.zeros_like(F[:, 0]))
                all_gens.extend([gen_idx + 1] * len(F))
                
            # Extract entire population for dense 1D scatter plotting
            pop = algo.pop
            if pop is not None and len(pop) > 0:
                F_pop = np.array([ind.F for ind in pop])
                all_xs_pop.extend(F_pop[:, 0])
                all_gens_pop.extend([gen_idx + 1] * len(F_pop))
                for ind in pop:
                    for key in all_comps.keys():
                        val = ind.get(key)
                        all_comps[key].append(val if val is not None else 0.0)
                
        # The final generation values (to locate the knee point if provided)
        final_pop = history[-1].opt if history[-1].opt is not None else []
        final_F = np.array([ind.F for ind in final_pop]) if len(final_pop) > 0 else np.array([])
        if len(final_F) > 0:
            if final_F.shape[1] == 1:
                xs_final = final_F[:, 0]
                ys_final, zs_final = [], []
            else:
                xs_final, ys_final = final_F[:, 0], final_F[:, 1]
                zs_final = final_F[:, 2] if final_F.shape[1] >= 3 else np.zeros_like(xs_final)
        else:
            xs_final, ys_final, zs_final = [], [], []

    else:
        # Fallback to simple array
        if hasattr(res_or_pareto_F, "F"):
            pareto_F = res_or_pareto_F.F
        else:
            pareto_F = np.asarray(res_or_pareto_F)
            
        all_xs = pareto_F[:, 0]
        if pareto_F.shape[1] >= 2:
            all_ys = pareto_F[:, 1]
            all_zs = pareto_F[:, 2] if pareto_F.shape[1] >= 3 else np.zeros_like(all_xs)
        else:
            all_ys, all_zs = [], []
        
        # Without history, we just color by a solid color
        all_gens = np.ones(len(all_xs))
        n_gens = 1
        xs_final, ys_final, zs_final = all_xs, all_ys, all_zs

    # Determine dimensionality
    is_1d = False
    is_2d = False
    if has_history:
        for algo in history:
            if algo.opt is not None and len(algo.opt) > 0:
                is_1d = (len(algo.opt[0].F) == 1)
                is_2d = (len(algo.opt[0].F) == 2)
                break
    else:
        is_1d = (pareto_F.shape[1] == 1)
        is_2d = (pareto_F.shape[1] == 2)

    # Pre-compute colors if using history
    if has_history:
        cmap = cm.get_cmap("plasma")
        norm = matplotlib.colors.Normalize(vmin=1, vmax=n_gens)

    if is_1d:
        fig = plt.figure(figsize=(15, 12))
        
        # Title construction
        title_str = "GA Optimization Convergence (Single Objective)"
        if sc and "weights" in sc:
            w_str = " + ".join([f"{v}*{k}" for k, v in sc["weights"].items() if v != 0])
            title_str += f"\n( F = {w_str} )"
        fig.suptitle(title_str, fontsize=14, fontweight="bold")
        
        if has_history and len(all_comps["f1_lsd"]) > 0:
            import matplotlib.cm as cm
            cmap = cm.get_cmap("jet")
            norm = matplotlib.colors.Normalize(vmin=1, vmax=n_gens)

            # Subplot 1: Pareto-like 2D space colored by Generation (F1 vs F2)
            ax1 = plt.subplot(3, 3, 1)
            scat = ax1.scatter(all_comps["f1_lsd"], all_comps["f2_sll"], c=all_gens_pop, cmap=cmap, norm=norm, s=15, alpha=0.6)
            ax1.set_xlabel("f1 (LSD + Penalty)", fontsize=9)
            ax1.set_ylabel("f2 (SLL dB)", fontsize=9)
            ax1.set_title("Population Scatter (2D Pareto Point View)", fontsize=10)
            ax1.grid(True, alpha=0.3)
            cbar = plt.colorbar(scat, ax=ax1)
            cbar.set_label("Generation", fontsize=8)
            
            # Subplot 2: Scalar Objective (Fitness) Convergence
            ax2 = plt.subplot(3, 3, 2)
            gen_mins = []
            unique_gens = sorted(list(set(all_gens_pop)))
            for g in unique_gens:
                vals = [all_xs_pop[j] for j in range(len(all_xs_pop)) if all_gens_pop[j] == g]
                gen_mins.append(np.min(vals))
            ax2.plot(unique_gens, gen_mins, 'b-', label='Best/Min Fitness', linewidth=2)
            ax2.scatter(all_gens_pop, all_xs_pop, c='gray', alpha=0.1, s=5)
            ax2.set_xlabel("Generation", fontsize=9)
            ax2.set_ylabel("Scalar Objective (F)", fontsize=9)
            ax2.set_title("Overall Fitness Convergence", fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)
            
            # Subplots 3-9: convergence plots for f1-f7
            labels = ["f1 (-LSD + Penalty)", "f2 (SLL dB)", "f3 (λ₂_norm)", "f4 (Graph Entropy)", "f5 (Dynamic Range dB)", "f6 (Resolution Eff)", "f7 (Directional/Pen)"]
            keys = ["f1_lsd", "f2_sll", "f3_lam2", "f4_graph", "f5_dr", "f6_eta", "f7_ds"]
            
            # Find the index of the best fitness individual per generation
            best_idx_per_gen = []
            cur_idx = 0
            for g in unique_gens:
                count = all_gens_pop.count(g)
                gen_f = all_xs_pop[cur_idx:cur_idx+count]
                best_local_idx = np.argmin(gen_f)
                best_idx_per_gen.append(cur_idx + best_local_idx)
                cur_idx += count
                
            for i in range(7):
                ax = plt.subplot(3, 3, 3 + i)
                vals = np.array(all_comps[keys[i]])
                ax.scatter(all_gens_pop, vals, c='lightgray', alpha=0.2, s=5)
                best_vals = [vals[idx] for idx in best_idx_per_gen]
                ax.plot(unique_gens, best_vals, 'r-', linewidth=2)
                ax.set_title(labels[i], fontsize=10)
                ax.set_xlabel("Generation", fontsize=9)
                ax.grid(True, alpha=0.3)
                
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            ax = plt.subplot(1, 1, 1)
            ax.hist(all_xs, bins=20, color="blue", alpha=0.7)
            ax.set_xlabel("Fitness values", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.grid(True, alpha=0.3)
    elif is_2d:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle("NSGA-II Pareto Front (2D)", fontsize=14, fontweight="bold")
        
        if has_history:
            sc = ax.scatter(all_xs, all_ys, c=all_gens, cmap=cmap, norm=norm, s=20, alpha=0.7)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Generation", fontsize=10)
        else:
            ax.scatter(all_xs, all_ys, c="#2563EB", s=20, alpha=0.7)

        if knee_idx is not None and len(xs_final) > knee_idx:
            ax.scatter([xs_final[knee_idx]], [ys_final[knee_idx]],
                       c="red", s=80, marker="*", label="Knee point", zorder=5)
            ax.legend(fontsize=9)

        ax.set_xlabel("F₁ = Spatial Quality (-LSD + Penalties)", fontsize=10)
        ax.set_ylabel("F₂ = Spectral Quality (-SLL - DR - ETA)", fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        fig = plt.figure(figsize=(18, 5))
        fig.suptitle("NSGA-II Pareto Front (Multi-Angle View)", fontsize=14, fontweight="bold")
        
        angles = [(20, 45), (30, 135), (15, -60)]
        
        for i, (elev, azim) in enumerate(angles):
            ax = fig.add_subplot(1, 4, i + 1, projection="3d")
            
            if has_history:
                sc = ax.scatter(all_xs, all_ys, all_zs, c=all_gens, cmap=cmap, norm=norm, 
                                s=20, alpha=0.7, depthshade=True)
            else:
                sc = ax.scatter(all_xs, all_ys, all_zs, c="#2563EB", s=20, alpha=0.7, depthshade=True)

            if knee_idx is not None and len(xs_final) > knee_idx:
                ax.scatter([xs_final[knee_idx]], [ys_final[knee_idx]], [zs_final[knee_idx]],
                           c="red", s=80, marker="*", label="Knee point", zorder=5)
                if i == 0:
                    ax.legend(fontsize=9)

            ax.set_xlabel("F₁ = Spatial", fontsize=8)
            ax.set_ylabel("F₂ = Spectral", fontsize=8)
            ax.set_zlabel("F₃ = Custom", fontsize=8)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"View {i+1} (elev={elev}°, azim={azim}°)", fontsize=10)
            ax.tick_params(labelsize=7)

        if has_history:
            ax = fig.add_subplot(1, 4, 4)
            ax.axis("off") # hide the axes
            cbar = plt.colorbar(sc, ax=ax, fraction=0.5, pad=0.04)
            cbar.set_label("Generation", fontsize=10)
            cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    print(f"[visualizer] Saved Pareto → {filename}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Classical comparison figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(results_dict, filename="comparison.png"):
    """
    Multi-row five-panel comparison of GraphSPAC vs classical arrays.

    results_dict : {name: {coords, metrics, arf_grid, k_max, ca}}
    """
    names = list(results_dict.keys())
    n = len(names)

    fig, axes_all = plt.subplots(n, 5, figsize=(22, 4.5 * n))
    if n == 1:
        axes_all = [axes_all]

    for row, name in enumerate(names):
        rd = results_dict[name]
        coords   = np.asarray(rd["coords"])
        arf_grid = rd["arf_grid"]
        k_max    = rd["k_max"]
        ca       = rd["ca"]
        met      = rd["metrics"]
        sll      = met["sll_db"]
        iso      = met["isotropy"]
        lam2     = met["lambda2"]

        ax0, ax1, ax2, ax3, ax4 = axes_all[row]

        # geometry
        N = len(coords)
        for i in range(N):
            for j in range(i + 1, N):
                ax0.plot([coords[i, 0], coords[j, 0]],
                         [coords[i, 1], coords[j, 1]],
                         c="steelblue", alpha=0.2, lw=0.8)
        ax0.scatter(coords[:, 0], coords[:, 1],
                    s=60, c="steelblue", edgecolors="white", lw=1.0)
        ax0.set_aspect("equal")
        ax0.set_title(f"(a) {name}", loc="left", fontsize=10, fontweight="bold")
        ax0.set_xlabel("X (m)", fontsize=8); ax0.set_ylabel("Y (m)", fontsize=8)
        ax0.tick_params(labelsize=7)

        # co-array
        _plot_coarray(ax1, ca)
        ax1.set_title("(b)", loc="left", fontsize=10, fontweight="bold")

        # ARF linear
        im2 = ax2.imshow(arf_grid.T, extent=[-k_max, k_max, -k_max, k_max],
                         origin="lower", cmap="hot", vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        ax2.set_title("(c)", loc="left", fontsize=10, fontweight="bold")
        ax2.tick_params(labelsize=7)

        # ARF dB
        arf_db = 20 * np.log10(arf_grid + 1e-9)
        arf_db -= arf_db.max()
        im3 = ax3.imshow(arf_db.T, extent=[-k_max, k_max, -k_max, k_max],
                         origin="lower", cmap="seismic_r", vmin=-40, vmax=0)
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        ax3.set_title("(d)", loc="left", fontsize=10, fontweight="bold")
        ax3.tick_params(labelsize=7)

        # rose
        ax4_pol = fig.add_subplot(n, 5, row * 5 + 5, projection="polar")
        _plot_rose(ax4_pol, ca, iso)
        ax4_pol.set_title("(e)", loc="left", fontsize=10, fontweight="bold")
        ax4.axis("off")   # hide placeholder ax4

    fig.suptitle("GraphSPAC vs Classical Arrays",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    print(f"[visualizer] Saved comparison → {filename}")
    plt.close()


# ── standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from manifold_manager import ManifoldManager
    from coarray import CoArray
    from graph_metrics import GraphMetrics
    from scenarios import SCENARIOS

    sc = SCENARIOS[0]
    gm = ManifoldManager(default_grid_size=sc["gridsize"], grid_spacing=sc["spacing"])
    rng = np.random.default_rng(1249718046570)
    coords = rng.uniform(0, gm.x_max, size=(10, 2))
    gmet = GraphMetrics(coords)
    met  = gmet.summary()
    plot_scenario_result(coords, gm, met, sc, filename="test_result.png")
    print("Done.")
