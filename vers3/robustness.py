# author: ferat
# date: 2026
# vol3 — Robustness Simulator
# Nodal-failure simulation: monitors λ₂ decay and SLL change
# when sensors are systematically removed from the array.

import numpy as np
from graph_metrics import GraphMetrics
from classical_arrays import make_classical_arrays


# ── Nodal failure simulation ──────────────────────────────────────────────────

def nodal_failure_sim(coords, n_max_remove=3, seed=1249718046570):
    """
    Systematically remove 1 … n_max_remove sensors (all combinations)
    and record the resulting λ₂ and SLL.

    Parameters
    ----------
    coords       : (N, 2) ndarray
    n_max_remove : int  — maximum number of sensors to remove
    seed         : int  — not used (deterministic removal), kept for API consistency

    Returns
    -------
    results : dict  {n_removed: {'lambda2': list, 'sll_db': list}}
              Relative drop compared to baseline is stored separately.
    baseline : dict  — metrics of the full array
    """
    coords = np.asarray(coords, dtype=float)
    N = len(coords)

    gm_base = GraphMetrics(coords)
    _, _, arf0, _ = gm_base.arf()
    base = {
        "lambda2": gm_base.algebraic_connectivity(),
        "sll_db":  gm_base.sll_db(arf0),
        "isotropy": gm_base.isotropy_score(),
    }

    from itertools import combinations

    results = {}
    for k in range(1, min(n_max_remove, N - 2) + 1):
        lam2_vals, sll_vals = [], []
        for removed in combinations(range(N), k):
            remaining = np.delete(coords, list(removed), axis=0)
            if len(remaining) < 2:
                continue
            gm_r = GraphMetrics(remaining)
            _, _, arf_r, _ = gm_r.arf()
            lam2_vals.append(gm_r.algebraic_connectivity())
            sll_vals.append(gm_r.sll_db(arf_r))
        results[k] = {
            "lambda2": lam2_vals,
            "sll_db":  sll_vals,
        }

    return results, base


def lambda2_drop_percent(results, base):
    """
    For each removal level k, compute the mean % drop in λ₂.

    Returns
    -------
    drop_dict : {k: mean_pct_drop}
    """
    drop_dict = {}
    lam2_base = base["lambda2"]
    for k, res in results.items():
        if lam2_base > 0:
            drops = [100 * (lam2_base - v) / lam2_base for v in res["lambda2"]]
        else:
            drops = [0.0] * len(res["lambda2"])
        drop_dict[k] = float(np.mean(drops)) if drops else 0.0
    return drop_dict


# ── Comparative robustness ────────────────────────────────────────────────────

def compare_robustness(graphspac_coords, aperture, n_sensors=10,
                       n_max_remove=2, verbose=True):
    """
    Compare GraphSPAC vs classical arrays in terms of λ₂ robustness.

    Parameters
    ----------
    graphspac_coords : (N, 2) ndarray
    aperture         : float  — design aperture [m]  (= 4 * focus depth)
    n_sensors        : int    — used to scale classical arrays
    n_max_remove     : int

    Returns
    -------
    report : dict  {array_name: {k: mean_pct_drop_lambda2}}
    """
    classical = make_classical_arrays(aperture, n_sensors)
    all_arrays = {"GraphSPAC (GA)": graphspac_coords}
    all_arrays.update(classical)

    report = {}
    for name, coords in all_arrays.items():
        if verbose:
            print(f"  Robustness: {name} (N={len(coords)}) …", end=" ", flush=True)
        res, base = nodal_failure_sim(np.array(coords), n_max_remove=n_max_remove)
        drop = lambda2_drop_percent(res, base)
        report[name] = drop
        if verbose:
            parts = [f"−{k} sensor: {drop[k]:.1f}%" for k in sorted(drop)]
            print("  ".join(parts))

    return report


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_robustness_comparison(report, filename="robustness_comparison.png"):
    """
    Bar chart of mean λ₂ % drop per removal level, per array type.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True,
                                 "grid.alpha": 0.3})

    k_vals   = sorted({k for drops in report.values() for k in drops})
    names    = list(report.keys())
    n_k      = len(k_vals)
    n_arr    = len(names)

    colors = ["#2563EB", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]
    hatches = ["", "//", "xx", "...", "\\\\", "oo"]

    x = np.arange(n_k)
    width = 0.7 / n_arr

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        drops = [report[name].get(k, 0.0) for k in k_vals]
        offset = (i - n_arr / 2 + 0.5) * width
        ax.bar(x + offset, drops, width, label=name,
               color=colors[i % len(colors)],
               hatch=hatches[i % len(hatches)],
               edgecolor="white", linewidth=0.5, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Remove {k}" for k in k_vals])
    ax.set_ylabel("Mean λ₂ Drop (%)", fontsize=11)
    ax.set_title("Topological Robustness: λ₂ Decay vs Nodal Failure\n"
                 "(Lower bar = more robust)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"[robustness] Saved → {filename}")
    plt.close()


# ── standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(1249718046570)
    test_coords = rng.uniform(0, 100, size=(10, 2))
    res, base = nodal_failure_sim(test_coords, n_max_remove=2)
    print("Baseline:", base)
    print("λ₂ drop %:", lambda2_drop_percent(res, base))
