# author: ferat
# date: 2026
# vol3 — Grid Generators
# Programmatic domain constructors for all extended scenarios.
# All return GridManager instances.

import numpy as np
from manifold_manager import ManifoldManager


# ── Scenario 2.2: L-shape with central hole ───────────────────────────────────

def l_shape_with_hole(grid_size=30, spacing=5.0, hole_radius_cells=3):
    """
    L-shaped domain with an obstacle at the inner corner junction.
    Simulates 'a large building at the street corner'.
    """
    gm = ManifoldManager.l_shape(grid_size, spacing)
    # hole centre = inner corner of L (top-left of horizontal arm + bottom of vertical arm)
    w = max(3, grid_size // 10)
    # corner region: rows near bottom, cols near left
    corner_row = grid_size - w     # first row of horizontal corridor
    corner_col = w                  # last col of vertical corridor
    r = hole_radius_cells
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            rr = corner_row + dr
            cc = corner_col + dc
            if 0 <= rr < grid_size and 0 <= cc < grid_size:
                gm.grid[rr, cc] = 0
    print(f"[GridGen] L-shape + hole at corner ({corner_row},{corner_col}), "
          f"radius={r} cells")
    return gm


# ── Scenario 3.1 / 7: Urban Canyon / Seismic Corridor ────────────────────────

def urban_canyon(width_cells, length_cells, spacing=5.0):
    """
    Narrow rectangular corridor.
    width_cells  : short dimension (e.g. 2 = 10m for spacing=5)
    length_cells : long dimension  (e.g. 20 = 100m)
    """
    gm = ManifoldManager(file_path=None, grid_spacing=spacing,
                     default_grid_size=max(width_cells, length_cells))
    gm.grid = np.ones((width_cells, length_cells), dtype=int)
    gm.rows, gm.cols = width_cells, length_cells
    print(f"[GridGen] Urban canyon: {width_cells * spacing:.0f} m × "
          f"{length_cells * spacing:.0f} m  (cells {width_cells}×{length_cells})")
    return gm


# ── Scenario 6: Urban Island (two separated zones) ────────────────────────────

def urban_island(grid_size=30, spacing=5.0, gap_fraction=0.4):
    """
    Two valid rectangular zones separated by a forbidden band in the centre.
    Simulates 'two parks on either side of a wide road'.
    gap_fraction : fraction of total width occupied by the gap/road.
    """
    gm = ManifoldManager(file_path=None, grid_spacing=spacing,
                     default_grid_size=grid_size)
    gm.grid[:] = 0
    gap_start = int(grid_size * (0.5 - gap_fraction / 2))
    gap_end   = int(grid_size * (0.5 + gap_fraction / 2))
    gm.grid[:, :gap_start] = 1    # left island
    gm.grid[:, gap_end:]   = 1    # right island
    print(f"[GridGen] Urban island: gap cols {gap_start}–{gap_end} "
          f"({gap_fraction*100:.0f}% forbidden), "
          f"left={gap_start} cols, right={grid_size-gap_end} cols")
    return gm


# ── Scenario 9 / 14: Fractal / Percolation grid ─────────────────────────────

def percolation_grid(grid_size=30, spacing=5.0, p_open=0.59, seed=1249718046570):
    """
    Random percolation grid: each cell is open (1) with probability p_open.
    At p_open ≈ 0.59 (site percolation threshold on square lattice) the
    grid is at criticality — a fractal labyrinth.
    """
    rng = np.random.default_rng(seed)
    gm = ManifoldManager(file_path=None, grid_spacing=spacing,
                     default_grid_size=grid_size)
    gm.grid = (rng.random((grid_size, grid_size)) < p_open).astype(int)
    n_open = int(gm.grid.sum())
    print(f"[GridGen] Percolation grid {grid_size}×{grid_size}: "
          f"p={p_open:.2f} → {n_open}/{grid_size**2} open cells "
          f"({100*n_open/grid_size**2:.1f}%)")
    return gm


def fractal_percolation(grid_size=40, spacing=5.0, p_open=0.65, seed=1249718046570):
    """
    Like percolation_grid but with an additional smoothing pass to create
    slightly more connected 'organic' shapes.
    """
    gm = percolation_grid(grid_size, spacing, p_open, seed)
    # one pass: a cell is forced open if ≥3 of its 4 neighbours are open
    new_grid = gm.grid.copy()
    for r in range(1, gm.rows - 1):
        for c in range(1, gm.cols - 1):
            neighbours = (gm.grid[r-1, c] + gm.grid[r+1, c] +
                          gm.grid[r, c-1] + gm.grid[r, c+1])
            if neighbours >= 3:
                new_grid[r, c] = 1
    gm.grid = new_grid
    print(f"[GridGen] Fractal percolation (smoothed): "
          f"{int(gm.grid.sum())} open cells")
    return gm


# ── Scenario 10: Variable-spacing open grid ───────────────────────────────────

def open_grid(grid_size=30, spacing=5.0):
    """Plain open grid — used when only spacing/dmin params change."""
    return ManifoldManager(file_path=None, grid_spacing=spacing,
                       default_grid_size=grid_size)


# ── Scenario 11: Fresnel Shadow (obstacle with LoS masking) ──────────────────
# The LoS masking is applied in the NSGA objective, not in the grid.
# The grid simply has the obstacle; the optimizer computes ray-crossing penalties.

def los_obstacle_grid(grid_size=30, spacing=5.0, obstacle_fraction=0.10):
    """
    Open grid with a solid rectangular obstacle block in the centre.
    The obstacle is used for Line-of-Sight edge pruning during evaluation.
    obstacle_fraction: fraction of each dimension occupied by the block.
    """
    gm = ManifoldManager(file_path=None, grid_spacing=spacing,
                     default_grid_size=grid_size)
    half = int(grid_size * obstacle_fraction / 2)
    mid  = grid_size // 2
    gm.grid[mid - half: mid + half, mid - half: mid + half] = 0
    print(f"[GridGen] LoS obstacle: {2*half}×{2*half} cell block at centre")
    return gm


# ── Scenario 8: C-Terminal (U-shape) ──────────────────────────────────────────

def u_shape_grid(grid_size=30, spacing=5.0, thickness_fraction=0.3):
    """
    U-shaped domain: A square grid with the top-center block removed.
    """
    gm = ManifoldManager(file_path=None, grid_spacing=spacing, default_grid_size=grid_size)
    thickness = int(grid_size * thickness_fraction)
    gm.grid[: (grid_size - thickness), thickness : (grid_size - thickness)] = 0
    print(f"[GridGen] U-shape: {grid_size}×{grid_size}, thickness {thickness} cells")
    return gm


# ── Scenario 10: The Graveyard (Scattered Blocks) ─────────────────────────────

def scattered_blocks(grid_size=30, spacing=5.0, num_blocks=15, block_size=2):
    """
    Random scattered blocks acting as fragmented obstacles.
    """
    gm = ManifoldManager(file_path=None, grid_spacing=spacing, default_grid_size=grid_size)
    rng = np.random.default_rng(1249718046570)
    for _ in range(num_blocks):
        r = rng.integers(0, grid_size - block_size)
        c = rng.integers(0, grid_size - block_size)
        gm.grid[r:r+block_size, c:c+block_size] = 0
    print(f"[GridGen] Scattered: {num_blocks} blocks of size {block_size}×{block_size}")
    return gm


# ── Scenario 22: The Donut ───────────────────────────────────────────────────

def donut_grid(grid_size=30, spacing=5.0, inner_radius_frac=0.3):
    """
    Central circular obstacle.
    """
    gm = ManifoldManager(file_path=None, grid_spacing=spacing, default_grid_size=grid_size)
    center = grid_size / 2.0
    for r in range(grid_size):
        for c in range(grid_size):
            dist = np.sqrt((r - center)**2 + (c - center)**2)
            if dist < grid_size * inner_radius_frac:
                gm.grid[r, c] = 0
    print(f"[GridGen] Donut: inner radius {grid_size * inner_radius_frac:.1f} cells")
    return gm


# ── Scenario 25: Divided Highway ─────────────────────────────────────────────

def divided_highway(grid_size=40, spacing=5.0, length_cells=40, width_cells=20, median_width=2):
    """
    Two parallel corridors separated by an inaccessible central median.
    """
    gm = ManifoldManager(file_path=None, grid_spacing=spacing, default_grid_size=max(length_cells, width_cells))
    gm.grid = np.zeros((width_cells, length_cells), dtype=int)
    gm.rows, gm.cols = width_cells, length_cells
    lane_width = (width_cells - median_width) // 2
    gm.grid[0:lane_width, :] = 1
    gm.grid[-lane_width:, :] = 1
    print(f"[GridGen] Divided Highway: {length_cells}×{width_cells}, {lane_width}-cell lanes, {median_width}-cell median")
    return gm


# ── Scenario 26: The Sponge ──────────────────────────────────────────────────

def sponge_grid(grid_size=40, spacing=4.0, porosity=0.45, seed=1249718046570):
    """
    Sponge-like domain: highly porous but generally connected.
    Created by randomly dropping cells and then culling completely isolated points.
    """
    rng = np.random.default_rng(seed)
    gm = ManifoldManager(file_path=None, grid_spacing=spacing, default_grid_size=grid_size)
    
    gm.grid = (rng.random((grid_size, grid_size)) > porosity).astype(int)
    
    new_grid = gm.grid.copy()
    for r in range(1, gm.rows - 1):
        for c in range(1, gm.cols - 1):
            if gm.grid[r, c] == 1:
                neighbours = (gm.grid[r-1, c] + gm.grid[r+1, c] +
                              gm.grid[r, c-1] + gm.grid[r, c+1])
                if neighbours < 1:  # Clean completely isolated specks
                    new_grid[r, c] = 0
    gm.grid = new_grid
    
    n_open = int(gm.grid.sum())
    print(f"[GridGen] Sponge Grid {grid_size}×{grid_size}: porosity {porosity:.2f} "
          f"→ {n_open} open cells")
    return gm




# ── Excel Matrix Grid (user-supplied) ─────────────────────────────────

def matrix_grid(file_path, spacing=5.0):
    """
    Load a ManifoldManager from an Excel (.xlsx) file.
    Spreadsheet must contain only 0s and 1s:
      1 = accessible (sensor can be placed)
      0 = obstacle / forbidden
    Each cell = one grid point spaced spacing metres apart.
    Example: 20x20 sheet with spacing=5.0 gives 100m x 100m domain.
    """
    return ManifoldManager(file_path=file_path, grid_spacing=spacing)


# ── factory ──────────────────────────────────────────────────────

def make_grid(sc, matrix_path=None, force_matrix=False):
    """
    Build a ManifoldManager from a scenario dict.
    If force_matrix=True or domain=matrix, load from matrix_path instead.
    """
    domain  = sc.get("domain", "open")
    size    = sc.get("gridsize", 30)
    spacing = sc.get("spacing", 5.0)

    if force_matrix or domain == "matrix":
        if matrix_path is None:
            raise ValueError("--matrix <path> is required when using --yeryok 1")
        return matrix_grid(matrix_path, spacing=spacing)

    if domain == "open":
        return open_grid(size, spacing)
    elif domain == "L_shape":
        return ManifoldManager.l_shape(size, spacing)
    elif domain == "L_hole":
        hole_r = sc.get("hole_radius", 3)
        return l_shape_with_hole(size, spacing, hole_radius_cells=hole_r)
    elif domain == "canyon":
        w = sc.get("canyon_width_cells", 3)
        l = sc.get("canyon_length_cells", 20)
        return urban_canyon(w, l, spacing)
    elif domain == "island":
        gap = sc.get("gap_fraction", 0.4)
        return urban_island(size, spacing, gap_fraction=gap)
    elif domain == "percolation":
        p = sc.get("p_open", 0.59)
        seed = sc.get("seed", 1249718046570)
        return percolation_grid(size, spacing, p_open=p, seed=seed)
    elif domain == "fractal":
        p = sc.get("p_open", 0.65)
        seed = sc.get("seed", 1249718046570)
        return fractal_percolation(size, spacing, p_open=p, seed=seed)
    elif domain == "los_obstacle":
        frac = sc.get("obstacle_fraction", 0.10)
        return los_obstacle_grid(size, spacing, obstacle_fraction=frac)
    elif domain == "u_shape":
        return u_shape_grid(size, spacing)
    elif domain == "scattered":
        return scattered_blocks(size, spacing)
    elif domain == "donut":
        return donut_grid(size, spacing)
    elif domain == "divided_highway":
        return divided_highway(size, spacing)
    elif domain == "sponge":
        porosity = sc.get("porosity", 0.45)
        seed = sc.get("seed", 1249718046570)
        return sponge_grid(size, spacing, porosity=porosity, seed=seed)
    else:
        raise ValueError(f"Unknown domain type: '{domain}'")


if __name__ == "__main__":
    for fn, kwargs in [
        ("l_shape_with_hole", {"grid_size": 20}),
        ("urban_canyon", {"width_cells": 3, "length_cells": 20}),
        ("urban_island", {"grid_size": 20}),
        ("percolation_grid", {"grid_size": 20, "p_open": 0.59}),
        ("fractal_percolation", {"grid_size": 20}),
        ("los_obstacle_grid", {"grid_size": 20}),
    ]:
        gm = globals()[fn](**kwargs)
        print(f"  -> {fn}: {gm.n_accessible} accessible cells")
