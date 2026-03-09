# author: ferat
# date: 2026
# vol3 — GridManager
# Manages the feasible deployment domain Ω_free.
# No jitter: returns exact cell-centre physical coords.

import numpy as np
import pandas as pd
import os


class GridManager:
    """
    Represents the feasible deployment space Ω_free as a 2D binary grid.
    grid[r, c] = 1  → accessible (can place sensor)
    grid[r, c] = 0  → obstacle / forbidden zone
    """

    def __init__(self, file_path=None, grid_spacing=5.0, default_grid_size=100):
        self.file_path = file_path
        self.grid_spacing = float(grid_spacing)
        self.default_grid_size = int(default_grid_size)
        self.grid = None
        self.rows = 0
        self.cols = 0
        self._load()

    # ── loading ──────────────────────────────────────────────────────────────

    def _load(self):
        if self.file_path is None:
            self._default_grid()
            return
        if os.path.exists(self.file_path):
            try:
                df = pd.read_excel(self.file_path, header=None)
                self.grid = df.values.astype(int)
                self.rows, self.cols = self.grid.shape
                print(f"[GridManager] Loaded {self.rows}×{self.cols} grid from {self.file_path}")
            except Exception as e:
                print(f"[GridManager] Error loading {self.file_path}: {e}  → using default")
                self._default_grid()
        else:
            print(f"[GridManager] {self.file_path} not found → using default")
            self._default_grid()

    def _default_grid(self, size=None):
        s = size or self.default_grid_size
        self.grid = np.ones((s, s), dtype=int)
        self.rows, self.cols = s, s
        print(f"[GridManager] Created default {s}×{s} open grid  (spacing={self.grid_spacing} m)")

    # ── L-shaped obstacle (Scenario 2) ────────────────────────────────────────

    @classmethod
    def l_shape(cls, grid_size=30, spacing=5.0):
        """
        L-shaped domain: two perpendicular corridors of width w_cells.
        Sensors may only be placed in the bottom row or the left column.
        Returns a GridManager instance with the L-shaped grid.
        """
        gm = cls(file_path=None, grid_spacing=spacing, default_grid_size=grid_size)
        # Start with all zeros (forbidden)
        gm.grid = np.zeros((grid_size, grid_size), dtype=int)
        w = max(3, grid_size // 10)   # corridor width in cells

        # Horizontal corridor: bottom w rows
        gm.grid[-w:, :] = 1
        # Vertical corridor: left w columns
        gm.grid[:, :w] = 1

        print(f"[GridManager] L-shaped domain: {grid_size}×{grid_size}, "
              f"corridor width={w} cells ({w*spacing:.1f} m)")
        return gm

    # ── helpers ──────────────────────────────────────────────────────────────

    def is_valid(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return bool(self.grid[r, c] == 1)
        return False

    def get_accessible_indices(self):
        """Return list of (row, col) tuples where grid==1."""
        rows, cols = np.where(self.grid == 1)
        return list(zip(rows.tolist(), cols.tolist()))

    def get_physical_coords(self, indices):
        """
        Convert grid (row, col) indices → physical (x, y) coordinates.
        x = col * spacing,  y = row * spacing.
        No jitter — deterministic.
        """
        coords = []
        for r, c in indices:
            x = float(c) * self.grid_spacing
            y = float(r) * self.grid_spacing
            coords.append((x, y))
        return coords

    def physical_bounds(self):
        """Return (x_max, y_max) of the grid."""
        return (self.cols - 1) * self.grid_spacing, (self.rows - 1) * self.grid_spacing

    def physical_to_index(self, x, y):
        """Snap continuous (x, y) to nearest valid grid cell. Returns (r, c) or None."""
        c = int(round(x / self.grid_spacing))
        r = int(round(y / self.grid_spacing))
        c = np.clip(c, 0, self.cols - 1)
        r = np.clip(r, 0, self.rows - 1)
        if self.is_valid(r, c):
            return (r, c)
        return None

    def feasible_mask(self):
        """Return boolean mask (rows × cols) of accessible cells."""
        return self.grid == 1

    @property
    def n_accessible(self):
        return int(np.sum(self.grid))

    @property
    def x_max(self):
        return (self.cols - 1) * self.grid_spacing

    @property
    def y_max(self):
        return (self.rows - 1) * self.grid_spacing


# ── standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    gm = GridManager(default_grid_size=10, grid_spacing=5.0)
    print("Accessible:", gm.n_accessible)

    gm_l = GridManager.l_shape(grid_size=20, spacing=5.0)
    print("L-shape accessible:", gm_l.n_accessible)
    print("Physical coords (first 5):", gm_l.get_physical_coords(
        gm_l.get_accessible_indices()[:5]))
