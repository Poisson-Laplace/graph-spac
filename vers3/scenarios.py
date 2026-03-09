# author: ferat
# date: 2026
# vol3 — Scenario Definitions (sequential IDs 1–22)

SEED = 1249718046570

# Helper shorthand
def _sc(id, name, desc, domain, N, d_min, focus, vs, kernel,
        noise_az=None, golomb=False, los=False, near_field=False,
        nf_src=None, **kw):
    return {
        "id": id, "name": name, "description": desc,
        "domain": domain, "gridsize": kw.get("gridsize", 30),
        "spacing": kw.get("spacing", 5.0),
        "N": N, "d_min": d_min, "focus": focus, "vs": vs,
        "kernel": kernel,
        "noise_azimuths": noise_az,
        "golomb": golomb, "los_penalty": los,
        "near_field": near_field, "near_field_source": nf_src,
        **{k: v for k, v in kw.items() if k not in ("gridsize", "spacing")},
    }


SCENARIOS = [
    # ── 0. Custom Configuration (Baseline for CLI overrides) ────────────────
    _sc(0, "Custom CLI Configuration",
        "Generic open domain. Use CLI flags to override N, focus, etc.",
        "open", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="bessel"),

    # ── 1. Open domain, N=10, Bessel ────────────────────────────────────────
    _sc(1, "Open Domain (J₀, N=10)",
        "Idealised open Euclidean plane — theoretical maximum for LSD entropy.",
        "open", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="bessel"),

    # ── 2. Open domain, N=2 (edge case — single pair) ───────────────────────
    _sc(2, "Open Domain (J₀, N=2) — Single Pair",
        "N=2 edge case: single co-array pair. λ₂ and LSD entropy at minimum.",
        "open", N=2, d_min=5.0, focus=30.0, vs=500.0, kernel="bessel",
        gridsize=20),
        
    # ── 3. L-shape, N=10, Hankel ────────────────────────────────────────────
    _sc(3, "L-Shaped Domain (H₀⁽¹⁾, N=10)",
        "Street-corner: 2 perpendicular corridors, dominant noise from North.",
        "L_shape", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[0.0]),

    # ── 4. Open domain, N=2 (edge case — single pair) ───────────────────────
    _sc(4, "Open Domain (H₀⁽¹⁾, N=2) — Single Pair",
        "N=2 edge case: single co-array pair. λ₂ and LSD entropy at minimum.",
        "open", N=2, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        gridsize=20),

    # ── 5. L-shape + noise 0° (North) — Directional Stress ─────────────────
    _sc(5, "L-Shape + Noise 0° (Directional Stress, North)",
        "Dominant noise from North: does GA cluster sensors in vertical arm?",
        "L_shape", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[0.0]),

    # ── 6. L-shape + noise 90° (East) — Aliasing check ─────────────────────
    _sc(6, "L-Shape + Noise 90° (Aliasing Check, East)",
        "Dominant noise from East: does horizontal arm alias?",
        "L_shape", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[90.0]),

    # ── 7. L-shape + central obstacle (Hole Scenario) ───────────────────────
    _sc(7, "L-Shape + Central Obstacle (Hole Scenario)",
        "Building at inner corner — how does GA bridge λ₂ across the gap?",
        "L_hole", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="bessel",
        hole_radius=3),


    # ── 9. Urban Canyon 10m × 100m (High Aspect Ratio) ──────────────────────
    _sc(9, "Urban Canyon 15m × 105m (High Aspect Ratio)",
        "Extreme aspect ratio — Euclidean Bottleneck. Pseudo-linear advantage.",
        "canyon", N=8, d_min=5.0, focus=20.0, vs=400.0, kernel="bessel",
        canyon_width_cells=3, canyon_length_cells=21, gridsize=21),

    # ── 11. Deep focus 50m ──────────────────────────────────────────────────
    _sc(11, "Deep Focus (focus=50 m, N=10)",
        "Large aperture needed — GA biases r_max. Compare LSD with Sc 12.",
        "open", N=10, d_min=5.0, focus=50.0, vs=600.0, kernel="bessel",
        gridsize=25, spacing=4.0),


    # ── 12. Shallow focus 10m ───────────────────────────────────────────────
    _sc(12, "Shallow Focus (focus=10 m, N=8)",
        "Small aperture needed — GA biases r_min. Bimodal LSD when overlaid with Sc 11.",
        "open", N=8, d_min=2.5, focus=10.0, vs=250.0, kernel="bessel",
        gridsize=15, spacing=2.5),

    # ── 13. Urban Island — two separated zones ───────────────────────────────
    _sc(13, "Urban Island — Two Separated Zones",
        "Two valid areas split by a forbidden road. Long edges=deep, short=shallow.",
        "island", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="bessel",
        gap_fraction=0.35),

    # ── 14. Seismic Corridor 5m × 200m ──────────────────────────────────────
    _sc(14, "Seismic Corridor 15m × 205m (1D Confinement)",
        "Extreme 1D confinement. Hankel pseudo-linear gain, noise from tunnel end.",
        "canyon", N=10, d_min=5.0, focus=40.0, vs=500.0, kernel="hankel",
        noise_az=[0.0], canyon_width_cells=3, canyon_length_cells=41, gridsize=41),

    # ── 15. Multi-Source Cross-Fire (N + E) ──────────────────────────────────
    _sc(15, "Cross-Fire: Noise from N + E (90° Sources)",
        "Two dominant sources: North (0°) and East (90°). Dual-azimuth Pareto.",
        "open", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[0.0, 90.0]),

    # ── 16. Fractal Boundary (Percolation p=0.65) ────────────────────────────
    _sc(16, "Fractal Boundary (Organic, p=0.65)",
        "Smoothed percolation grid — slime-mold topology above threshold.",
        "fractal", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        p_open=0.65),

    # ── 17. Variable Spacing (spacing=10, dmin=2) ────────────────────────────
    _sc(17, "Variable Spacing (spacing=10 m, dmin=2 m) — Grid-Lock Test",
        "Coarse grid but fine dmin: does GA escape Grid-Lock into Hankel manifold?",
        "open", N=10, d_min=2.0, focus=30.0, vs=500.0, kernel="bessel",
        gridsize=20, spacing=10.0),

    # ── 18. Fresnel Shadow (LoS Constraint) ─────────────────────────────────
    _sc(18, "Fresnel Shadow — Line-of-Sight Constraint",
        "Central obstacle: edges crossing it are penalised → lens-like placement.",
        "los_obstacle", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        los=True, obstacle_fraction=0.12),

    # ── 19. 2D Golomb Ruler (Zero-Redundancy Co-array) ──────────────────────
    _sc(19, "2D Golomb Ruler — Zero-Redundancy Co-array",
        "Penalise repeated lag distances: SLL approaches −30 dB aperiodic limit.",
        "open", N=8, d_min=5.0, focus=30.0, vs=500.0, kernel="bessel",
        golomb=True),

    # ── 20. Near-Field Hankel (Focal Source 20m) ─────────────────────────────
    _sc(20, "Spherical Near-Field (source at 20 m)",
        "Point source nearby — wavefront curvature. Array acts as parabolic dish.",
        "open", N=10, d_min=5.0, focus=20.0, vs=400.0, kernel="hankel",
        near_field=True, nf_src=(50.0, 20.0), gridsize=20),

    # ── 21. Percolation Labyrinth (critical p=0.59) ──────────────────────────
    _sc(21, "Percolation Labyrinth — Critical Threshold (p=0.59)",
        "Square lattice at criticality: fractal labyrinth, nöron-ağı topolojisi.",
        "percolation", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        p_open=0.59, gridsize=35),

    # ── 23. Open Domain + Noise 0° (North) ───────────────────────────────────
    _sc(23, "Open Domain + Noise 0° (North)",
        "Unconstrained domain with noise exactly from the North (0°).",
        "open", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[0.0], gridsize=30),

    # ── 24. Open Domain + Noise 90° (East) ───────────────────────────────────
    _sc(24, "Open Domain + Noise 90° (East)",
        "Unconstrained domain with noise exactly from the East (90°).",
        "open", N=10, d_min=5.0, focus=30.0, vs=500.0, kernel="hankel",
        noise_az=[90.0], gridsize=30),
]

SCENARIO_MAP = {sc["id"]: sc for sc in SCENARIOS}
