# author: ferat
# date: 2026
# vol5 — Scenario Definitions (sequential IDs 1–26)
# With scenario-specific optimization modes and weight sets

SEED = 1249718046570

# ── Optimization Weight Sets ──────────────────────────────────────────────────
# Each mode emphasizes different objectives based on scenario topology

WEIGHT_SETS = {
    "precision": {
        "lsd": 1.0,
        "sll": 5.0,
        "lam2": 0.5,
        "graph_ent": 0.5,
        "dr": 0.5,
        "eta": 0.5,
    },
    "targeting": {
        "lsd": 0.1,
        "sll": 5.0,
        "lam2": 0.5,
        "graph_ent": 0.5,
        "dr": 0.5,
        "eta": 0.5,
    },
    "discovery": {
        "lsd": 3.0,
        "sll": 1.0,
        "lam2": 1.5,
        "graph_ent": 1.0,
        "dr": 0.5,
        "eta": 0.5,
    },
    "robust": {
        "lsd": 1.0,
        "sll": 1.0,
        "lam2": 4.0,
        "graph_ent": 2.0,
        "dr": 0.5,
        "eta": 0.5,
    },
}

DEFAULT_SCENARIO = {
    "noise_azimuths": None,
    "los_penalty": False,
    "golomb": False,
    "near_field": False,
    "near_field_source": None,
    "overlay_pair": None,
    "generations": 200,
    "population": 100,
    "mode": "precision",
}

def _sc(id, name, desc, domain, N, d_min, focus, vs, kernel,
        noise_az=None, golomb=False, los=False, near_field=False,
        nf_src=None, mode="precision", **kw):
    sc = DEFAULT_SCENARIO.copy()
    sc.update({
        "id": id, "name": name, "description": desc,
        "domain": domain, "gridsize": kw.get("gridsize", 30),
        "spacing": kw.get("spacing", 5.0),
        "N": N, "d_min": d_min, "focus": focus, "vs": vs,
        "kernel": kernel,
        "noise_azimuths": noise_az,
        "golomb": golomb, "los_penalty": los,
        "near_field": near_field, "near_field_source": nf_src,
        "mode": mode,
        "weights": WEIGHT_SETS[mode],
        "r_max": kw.get("r_max", 1.5 * focus if mode == "targeting" else None),
        **{k: v for k, v in kw.items() if k not in ("gridsize", "spacing", "r_max")},
    })
    return sc


SCENARIOS = [
    # ── 0. Custom Configuration (Baseline for CLI overrides) ────────────────
    _sc(0, "Custom CLI Configuration",
        "Generic open domain. Use CLI flags to override N, focus, etc.",
        "open", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="bessel",
        mode="precision"),

    # ========================================================================
    # GROUP 1: PRECISION / IDEAL ENVIRONMENTS
    # Unconstrained or gently constrained open environments demanding high
    # spectral purity and side-lobe suppression.
    # Old Order: 1, 2, 3, 11, 12, 17, 23, 24 (Included 19)
    # ========================================================================

    # 1
    _sc(1, "Open Domain (J₀, N=10)",
        "Idealised open Euclidean plane — theoretical maximum for LSD entropy.",
        "open", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="bessel",
        mode="precision"),


    # 11
    _sc(11, "Deep Focus (focus=50 m, N=10)",
        "Large aperture needed — GA biases r_max. Compare LSD with Sc 12.",
        "open", N=10, d_min=5.0, focus=50.0, vs=600.0, kernel="bessel",
        gridsize=25, spacing=4.0, mode="precision"),

    # 12
    _sc(12, "Shallow Focus (focus=10 m, N=8)",
        "Small aperture needed — GA biases r_min. Bimodal LSD when overlaid with Sc 11.",
        "open", N=8, d_min=2.5, focus=75.0, vs=250.0, kernel="bessel",
        gridsize=15, spacing=2.5, mode="precision"),

    # 19 (Added here because it fits Precision logic and was missing from the user's prompt)
    _sc(19, "2D Golomb Ruler — Zero-Redundancy Co-array",
        "Penalise repeated lag distances: SLL approaches −30 dB aperiodic limit.",
        "open", N=8, d_min=5.0, focus=30.0, vs=500.0, kernel="bessel",
        golomb=True, mode="precision"),

    # 23
    _sc(23, "Open Domain + Noise 0° (North)",
        "Unconstrained domain with noise exactly from the North (0°).",
        "open", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[0.0], gridsize=30, mode="discovery"),

    # 24
    _sc(24, "Open Domain + Noise 90° (East)",
        "Unconstrained domain with noise exactly from the East (90°).",
        "open", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[90.0], gridsize=30, mode="discovery"),


    # ========================================================================
    # GROUP 2: DISCOVERY / DIRECTIONAL & CONSTRAINED MANIFOLDS
    # Scenarios enforcing strong directionality or geometric bottlenecks.
    # Old Order: 4, 5, 6, 8, 9, 14, 15, 18, 25 (Included 20)
    # ========================================================================

    # 4
    _sc(4, "L-Shaped Domain (H₀⁽¹⁾, N=10)",
        "Street-corner: 2 perpendicular corridors, Isotropic (No dominant noise).",
        "L_shape", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        mode="discovery"),

    # 5
    _sc(5, "L-Shape + Noise 0° (Directional Stress, North)",
        "Dominant noise from North: does GA cluster sensors in vertical arm?",
        "L_shape", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[0.0], mode="discovery"),

    # 6
    _sc(6, "L-Shape + Noise 90° (Aliasing Check, East)",
        "Dominant noise from East: does horizontal arm alias?",
        "L_shape", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[90.0], mode="discovery"),

    # 8
    _sc(8, "C-Terminal (U-Shaped Obstacle)",
        "Deep U-shaped building. Sensors at tips are Euclidean-close but Geodesic-far.",
        "u_shape", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        mode="discovery"),

    # 15
    _sc(15, "Cross-Fire: Noise from N + E (90° Sources)",
        "Two dominant sources: North (0°) and East (90°). Dual-azimuth Pareto.",
        "open", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[0.0, 90.0], mode="discovery"),

    # 18
    _sc(18, "Fresnel Shadow — Line-of-Sight Constraint",
        "Central obstacle: edges crossing it are penalised → lens-like placement.",
        "los_obstacle", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        los=True, obstacle_fraction=0.12, mode="discovery"),
    
    # 20
    _sc(20, "Spherical Near-Field (source at 20 m)",
        "Point source nearby — wavefront curvature. Array acts as parabolic dish.",
        "open", N=10, d_min=5.0, focus=20.0, vs=400.0, kernel="hankel",
        near_field=True, nf_src=(50.0, 20.0), gridsize=20, mode="targeting"),

    # 25
    _sc(25, "Divided Highway (Two Corridors)",
        "Two parallel long corridors simulating a dual-carriageway with a central median.",
        "divided_highway", N=10, d_min=5.0, focus=40.0, vs=500.0, kernel="hankel",
        gridsize=50, spacing=4.0, mode="discovery"),


    # ========================================================================
    # GROUP 3: ROBUST / FRAGMENTED TOPOLOGIES
    # Highly non-convex or disconnected areas targeting Fiedler Value (λ₂) checks.
    # Old Order: 7, 10, 13, 16, 21, 26, 22
    # ========================================================================

    # 10
    _sc(10, "The Graveyard (Scattered Blocks)",
        "15 small random obstacles. High fragmentation risk for the graph.",
        "scattered", N=12, d_min=3.0, focus=25.0, vs=400.0, kernel="bessel",
        mode="robust"),

    # 13
    _sc(13, "Urban Island — Two Separated Zones",
        "Two valid areas split by a forbidden road. Long edges=deep, short=shallow.",
        "island", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="bessel",
        gap_fraction=0.35, mode="robust"),

    # 16
    _sc(16, "Fractal Boundary (Organic, p=0.65)",
        "Smoothed percolation grid — slime-mold topology above threshold.",
        "fractal", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        p_open=0.65, mode="robust"),

    # 21
    _sc(21, "Percolation Labyrinth — Critical Threshold (p=0.59)",
        "Square lattice at criticality: fractal labyrinth, nöron-ağı topolojisi.",
        "percolation", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        p_open=0.59, gridsize=35, mode="robust"),

    # 26
    _sc(26, "The Sponge (Topological Porosity)",
        "A highly non-convex, porous structure creating topological fragmentation.",
        "sponge", N=12, d_min=3.0, focus=25.0, vs=400.0, kernel="bessel",
        gridsize=40, spacing=4.0, porosity=0.45, mode="discovery"),
]

SCENARIO_MAP = {sc["id"]: sc for sc in SCENARIOS}
