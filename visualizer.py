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
    "figure.dpi":   150,
})


# ══════════════════════════════════════════════════════════════════════════════
#  Five-panel scenario result figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_scenario_result(coords, gm, metrics, sc, pareto_F=None,
                         filename="result.png"):
    """
    Five-panel figure:
      [0] Array layout on feasibility grid
      [1] Inter-station vectors (co-array scatter)
      [2] ARF — linear scale
      [3] ARF — dB scale
      [4] Rose diagram (azimuthal distribution)

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
    sll = gmet.sll_db(arf_grid)
    lam2 = gmet.algebraic_connectivity()
    iso  = gmet.isotropy_score()

    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)

    # ── 0: Array Layout ──────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    _plot_array_layout(ax0, coords, gm)
    ax0.set_title(f"Array Geometry\nN={len(coords)}  r_max={metrics['r_max_m']:.1f} m", fontsize=9)

    # ── 1: Co-array (Δ) ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    _plot_coarray(ax1, ca)
    ax1.set_title(f"Co-array Δ\nLSD H={ca.lsd_entropy():.3f}  pairs={ca.n_pairs}", fontsize=9)

    # ── 2: ARF linear ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(arf_grid.T, extent=[-k_max_used, k_max_used,
                                          -k_max_used, k_max_used],
                     origin="lower", cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2, shrink=0.85)
    ax2.set_title(f"ARF (linear)\nSLL={sll:.1f} dB", fontsize=9)
    ax2.set_xlabel("kx (rad/m)", fontsize=8)
    ax2.set_ylabel("ky (rad/m)", fontsize=8)
    ax2.tick_params(labelsize=7)

    # ── 3: ARF dB ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    arf_db = 20 * np.log10(arf_grid + 1e-9)
    arf_db -= arf_db.max()
    im3 = ax3.imshow(arf_db.T, extent=[-k_max_used, k_max_used,
                                         -k_max_used, k_max_used],
                     origin="lower", cmap="seismic_r", vmin=-40, vmax=0)
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.85)
    cbar3.set_label("dB", fontsize=7)
    ax3.set_title("ARF (dB)", fontsize=9)
    ax3.set_xlabel("kx (rad/m)", fontsize=8)
    ax3.set_ylabel("ky (rad/m)", fontsize=8)
    ax3.tick_params(labelsize=7)

    # ── 4: Rose diagram ────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[4], projection="polar")
    _plot_rose(ax4, ca, iso)
    ax4.set_title(f"Rose Diagram\nIsotropy={iso:.3f}", fontsize=9)

    # Main title — everything from real computed data, not the scenario template
    import re as _re
    kernel_raw  = sc.get("kernel", "bessel").lower()
    kernel_fmt  = "BESSEL (J\u2080)" if kernel_raw == "bessel" else "HANKEL (H\u2080\u207d\xb9\u207e)"
    N_actual    = len(coords)
    sc_name_raw = sc.get("name", "Custom")
    # strip embedded N=xx, J₀, H₀, kernel notations from scene name
    sc_name = sc_name_raw
    sc_name = _re.sub(r"[\s,]*N=\d+", "", sc_name)      # remove ', N=10'
    sc_name = _re.sub(r"\(J\u2080[^)]*\)", "", sc_name) # remove '(J₀)'
    sc_name = _re.sub(r"\(H\u2080[^)]*\)", "", sc_name) # remove '(H₀...)'
    sc_name = sc_name.strip().rstrip(",").strip()

    # Optional extra metrics
    extras = []
    if "directional_score" in metrics:
        az_str = ", ".join(str(int(a)) for a in (sc.get("noise_azimuths") or []))
        extras.append(f"Dir={metrics['directional_score']:.3f}@[{az_str}\u00b0]")
    if "golomb_redundancy" in metrics:
        extras.append(f"Golomb={metrics['golomb_redundancy']:.3f}")
    if "los_blocked_frac" in metrics:
        extras.append(f"LoS_blk={metrics['los_blocked_frac']:.3f}")
    
    # ── Display GA Parameter Info ──
    gens = sc.get("generations", "?")
    pop  = sc.get("population", "?")
    ga_str = f" [GA: Pop={pop}, Gens={gens}]"
    
    extra_str = "  " + "  ".join(extras) if extras else ""

    if sc.get("is_custom_name"):
        title_top = f"GraphSPAC \u2014 {sc_name}   [Kernel: {kernel_fmt}]{ga_str}"
    else:
        title_top = f"GraphSPAC \u2014 Sc {sc['id']}: {sc_name}   [Kernel: {kernel_fmt}]{ga_str}"

    fig.suptitle(
        f"{title_top}\n"
        f"N={N_actual}  "
        f"\u03bb\u2082={lam2:.4f}  SLL={sll:.1f} dB  "
        f"Isotropy={iso:.3f}  r_min={metrics['r_min_m']:.2f} m  r_max={metrics['r_max_m']:.1f} m"
        f"{extra_str}",
        fontsize=10, fontweight="bold"
    )


    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"[visualizer] Saved → {filename}")
    plt.close()


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
    18 bins of 10 degrees each.
    Range is shifted by half a bin (5 deg) so that cardinal angles
    (0, 90, 180 ...) land exactly on bin CENTRES, not between bins.
    """
    geo_folded = ca.azimuths_geo_folded()   # geographic [0, pi), 0=N, pi/2=E

    n_bins    = 18
    half_bin  = np.pi / (2 * n_bins)          # 5 degrees in radians
    hist, bin_edges = np.histogram(
        geo_folded, bins=n_bins,
        range=(-half_bin, np.pi - half_bin)   # shift so centres = 0,10,20,...90,...
    )
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width   = np.pi / n_bins

    ax.bar(centers,          hist, width, alpha=0.75, color="steelblue",
           edgecolor="white", lw=0.5)
    ax.bar(centers + np.pi,  hist, width, alpha=0.75, color="steelblue",
           edgecolor="white", lw=0.5)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(labelsize=6)


# ══════════════════════════════════════════════════════════════════════════════
#  Pareto front plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_pareto(pareto_F, knee_idx=None, filename="pareto.png"):
    """
    3D scatter of Pareto front objectives from 3 different viewing angles.
    pareto_F : (M, 3)  [f1=−LSD, f2=−SLL, f3=−λ₂]
    """
    from mpl_toolkits.mplot3d import Axes3D   # noqa

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("NSGA-II Pareto Front (Multi-Angle View)", fontsize=14, fontweight="bold")
    
    xs, ys, zs = pareto_F[:, 0], pareto_F[:, 1], pareto_F[:, 2]
    
    angles = [(20, 45), (30, 135), (15, -60)]
    
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(xs, ys, zs, c="#2563EB", s=20, alpha=0.7, depthshade=True)

        if knee_idx is not None:
            ax.scatter([xs[knee_idx]], [ys[knee_idx]], [zs[knee_idx]],
                       c="red", s=80, marker="*", label="Knee point", zorder=5)
            # Only put legend on the first one
            if i == 0:
                ax.legend(fontsize=9)

        ax.set_xlabel("f₁ = −LSD entropy", fontsize=8)
        ax.set_ylabel("f₂ = −SLL (dB)", fontsize=8)
        ax.set_zlabel("f₃ = −λ₂", fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View {i+1} (elev={elev}°, azim={azim}°)", fontsize=10)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
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
        ax0.set_title(f"{name}\nN={N}  r_max={met['r_max_m']:.0f} m", fontsize=9)
        ax0.set_xlabel("X (m)", fontsize=8); ax0.set_ylabel("Y (m)", fontsize=8)
        ax0.tick_params(labelsize=7)

        # co-array
        _plot_coarray(ax1, ca)
        ax1.set_title("Co-array Δ", fontsize=9)

        # ARF linear
        im2 = ax2.imshow(arf_grid.T, extent=[-k_max, k_max, -k_max, k_max],
                         origin="lower", cmap="hot", vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        ax2.set_title(f"ARF (linear)\nSLL={sll:.1f} dB", fontsize=9)
        ax2.tick_params(labelsize=7)

        # ARF dB
        arf_db = 20 * np.log10(arf_grid + 1e-9)
        arf_db -= arf_db.max()
        im3 = ax3.imshow(arf_db.T, extent=[-k_max, k_max, -k_max, k_max],
                         origin="lower", cmap="seismic_r", vmin=-40, vmax=0)
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        ax3.set_title("ARF (dB)", fontsize=9)
        ax3.tick_params(labelsize=7)

        # rose
        ax4_pol = fig.add_subplot(n, 5, row * 5 + 5, projection="polar")
        _plot_rose(ax4_pol, ca, iso)
        ax4_pol.set_title(f"Rose\nIso={iso:.3f}  λ₂={lam2:.4f}", fontsize=9)
        ax4.axis("off")   # hide placeholder ax4

    fig.suptitle("GraphSPAC vs Classical Arrays — ARF · SLL · Azimuthal Coverage",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"[visualizer] Saved comparison → {filename}")
    plt.close()


# ── standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from grid_manager import GridManager
    from coarray import CoArray
    from graph_metrics import GraphMetrics
    from scenarios import SCENARIOS

    sc = SCENARIOS[0]
    gm = GridManager(default_grid_size=sc["gridsize"], grid_spacing=sc["spacing"])
    rng = np.random.default_rng(1249718046570)
    coords = rng.uniform(0, gm.x_max, size=(10, 2))
    gmet = GraphMetrics(coords)
    met  = gmet.summary()
    plot_scenario_result(coords, gm, met, sc, filename="test_result.png")
    print("Done.")
