# author: ferat
# date: 2026
# vol3 — Classical reference array factories
# All functions return (N, 2) numpy arrays centred at origin.

import numpy as np


# ── Circle ────────────────────────────────────────────────────────────────────

def circle_array(radius, n_sensors, center=(0.0, 0.0)):
    """
    N sensors equally spaced on a single circle.
    """
    angles = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])


# ── Nested equilateral triangles ──────────────────────────────────────────────

def _equilateral_triangle(side, center=(0.0, 0.0)):
    h = side * np.sqrt(3) / 2
    return np.array([
        [center[0],              center[1] + 2 * h / 3],
        [center[0] - side / 2,  center[1] -     h / 3],
        [center[0] + side / 2,  center[1] -     h / 3],
    ])


def nested_triangle_array(r_outer, r_inner=None, center=(0.0, 0.0)):
    """
    Two nested equilateral triangles (inner rotated 60°).
    r_outer / r_inner are circumradii.
    """
    if r_inner is None:
        r_inner = r_outer / 2.0

    # outer triangle — circumradius r → side = r * sqrt(3)
    side_out = r_outer * np.sqrt(3)
    side_in  = r_inner * np.sqrt(3)

    outer = _equilateral_triangle(side_out, center)
    inner_raw = _equilateral_triangle(side_in, center)

    # rotate inner by 60° so vertices point between outer vertices
    theta = np.pi / 3
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    inner = (inner_raw - np.array(center)) @ R.T + np.array(center)

    return np.vstack([outer, inner])


# ── Propeller ─────────────────────────────────────────────────────────────────

def propeller_array(n_blades=3, sensors_per_blade=3, radius=50.0, center=(0.0, 0.0)):
    """
    Centre sensor + n_blades arms, each with sensors_per_blade stations
    equally spaced from 0 to radius.
    """
    pts = [list(center)]
    for angle in np.linspace(0, 2 * np.pi, n_blades, endpoint=False):
        for i in range(1, sensors_per_blade + 1):
            r = radius * i / sensors_per_blade
            pts.append([center[0] + r * np.cos(angle),
                        center[1] + r * np.sin(angle)])
    return np.array(pts)


# ── Kennett spiral ────────────────────────────────────────────────────────────

def kennett_spiral(aperture, n_arms=3, n_rings=4, span_deg=120,
                   log_spacing=False, center=(0.0, 0.0)):
    """
    Kennett et al. (2015) spiral array.
    aperture   : maximum radius [m]
    n_arms     : number of spiral arms
    n_rings    : number of stations per arm
    span_deg   : azimuthal span of each arm [°]
    log_spacing: use logarithmic radial spacing if True
    """
    pts = [list(center)]
    span_rad = np.radians(span_deg)

    for k in range(n_arms):
        theta0 = 2 * np.pi * k / n_arms
        for j in range(1, n_rings + 1):
            if log_spacing:
                r = aperture * (np.exp(j / n_rings) - 1) / (np.e - 1)
            else:
                r = aperture * j / n_rings
            theta = theta0 + span_rad * j / n_rings
            pts.append([center[0] + r * np.cos(theta),
                        center[1] + r * np.sin(theta)])

    return np.array(pts)


# ── Cross ─────────────────────────────────────────────────────────────────────

def cross_array(arm_length, n_per_arm=3, center=(0.0, 0.0)):
    """
    Centre + 4 arms (N, S, E, W).
    """
    pts = [list(center)]
    for direction in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        for i in range(1, n_per_arm + 1):
            r = arm_length * i / n_per_arm
            pts.append([center[0] + r * np.cos(direction),
                        center[1] + r * np.sin(direction)])
    return np.array(pts)


# ── L-shape (for reference comparison) ───────────────────────────────────────

def l_array(arm_length, n_per_arm=4, center=(0.0, 0.0)):
    """
    L-shaped: centre + East arm + North arm.
    Mirrors typical street-corner deployment.
    """
    pts = [list(center)]
    for i in range(1, n_per_arm + 1):
        r = arm_length * i / n_per_arm
        pts.append([center[0] + r, center[1]])        # East
    for i in range(1, n_per_arm + 1):
        r = arm_length * i / n_per_arm
        pts.append([center[0], center[1] + r])        # North
    return np.array(pts)


# ── factory ───────────────────────────────────────────────────────────────────

def make_classical_arrays(aperture, n_sensors=10):
    """
    Return a dict of {name: (N,2) coords} for all classical reference shapes
    scaled to the given half-aperture, with approximately n_sensors stations.
    """
    R = aperture / 2.0
    arrays = {
        "Circle":            circle_array(R, n_sensors),
        "NestedTriangles":   nested_triangle_array(R, R / 2),
        "Propeller (3×3)":   propeller_array(3, 3, R),
        "Spiral (Kennett)":  kennett_spiral(R, n_arms=3, n_rings=4, span_deg=120),
    }
    return arrays


# ── standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name, coords in make_classical_arrays(aperture=120.0, n_sensors=10).items():
        print(f"{name:25s}  N={len(coords)}  "
              f"r_max={np.max(np.linalg.norm(coords, axis=1)):.1f} m")
