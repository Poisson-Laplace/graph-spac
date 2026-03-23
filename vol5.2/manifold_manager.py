# author: ferat
# date: 2026
# vol4 — ManifoldManager
# Manages the feasible deployment domain Ω_free.
# Computes Geodesic (shortest-path) distances around obstacles using Dijkstra's algorithm.

import numpy as np
import pandas as pd
import os
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

class ManifoldManager:
    """
    Represents the feasible deployment space Ω_free as a 2D binary grid,
    but acts as a manifold where distances between points are measured geodesically 
    (avoiding obstacles) rather than Euclidean.
    """

    def __init__(self, file_path=None, grid_spacing=5.0, default_grid_size=100):
        self.file_path = file_path
        self.grid_spacing = float(grid_spacing)
        self.default_grid_size = int(default_grid_size)
        self.grid = None
        self.rows = 0
        self.cols = 0
        self.valid_indices = [] # list of (r, c)
        self._load()
        self._compute_geodesic_graph()

    def _load(self):
        if self.file_path is None:
            self._default_grid()
        elif os.path.exists(self.file_path):
            try:
                df = pd.read_excel(self.file_path, header=None)
                self.grid = df.values.astype(int)
                self.rows, self.cols = self.grid.shape
                print(f"[ManifoldManager] Loaded {self.rows}×{self.cols} grid from {self.file_path}")
            except Exception as e:
                print(f"[ManifoldManager] Error loading {self.file_path}: {e}  → using default")
                self._default_grid()
        else:
            print(f"[ManifoldManager] {self.file_path} not found → using default")
            self._default_grid()
            
        # extract valid indices list just once
        r, c = np.where(self.grid == 1)
        self.valid_indices = list(zip(r.tolist(), c.tolist()))
        self.n_valid = len(self.valid_indices)

        # map (r, c) to integer ID
        self.coord_to_id = {pos: i for i, pos in enumerate(self.valid_indices)}

    def _default_grid(self, size=None):
        s = size or self.default_grid_size
        self.grid = np.ones((s, s), dtype=int)
        self.rows, self.cols = s, s
        print(f"[ManifoldManager] Created default {s}×{s} open manifold (spacing={self.grid_spacing} m)")

    @classmethod
    def l_shape(cls, grid_size=30, spacing=5.0):
        """
        L-shaped domain: two perpendicular corridors.
        """
        gm = cls(file_path=None, grid_spacing=spacing, default_grid_size=grid_size)
        gm.grid = np.zeros((grid_size, grid_size), dtype=int)
        w = max(3, grid_size // 10)
        gm.grid[-w:, :] = 1
        gm.grid[:, :w] = 1
        
        r, c = np.where(gm.grid == 1)
        gm.valid_indices = list(zip(r.tolist(), c.tolist()))
        gm.n_valid = len(gm.valid_indices)
        gm.coord_to_id = {pos: i for i, pos in enumerate(gm.valid_indices)}
        gm._compute_geodesic_graph()
        
        print(f"[ManifoldManager] L-shaped domain: {grid_size}×{grid_size}, corridor width={w} cells ({w*spacing:.1f} m)")
        return gm
        
    def _compute_geodesic_graph(self):
        """
        Builds a graph of accessible cells and computes all-pairs shortest paths 
        using Dijkstra. Stored internally to instantly return geodesic distances.
        """
        print(f"[ManifoldManager] Building graph for {self.n_valid} accessible nodes...")
        # Adjacency matrix (8-way connectivity)
        adj = dok_matrix((self.n_valid, self.n_valid), dtype=np.float32)
        
        directions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), 
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)
        ]
        
        for i, (r, c) in enumerate(self.valid_indices):
            for dr, dc, weight in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr, nc] == 1:
                        j = self.coord_to_id[(nr, nc)]
                        adj[i, j] = weight * self.grid_spacing
        
        adj_csr = adj.tocsr()
        print("[ManifoldManager] Computing all-pairs geodesic distances (Dijkstra)...")
        # Computes all shortest paths. Returns [n_valid, n_valid] dense array.
        self.dist_matrix = dijkstra(csgraph=adj_csr, directed=False, return_predecessors=False)
        print("[ManifoldManager] Geodesic distances cached.")

    def get_geodesic_distance(self, pt1, pt2):
        """
        Returns the geodesic distance between continuous physical (x,y) coords.
        Snaps each coordinate to the nearest feasible grid cell and does a fast O(1) lookup.
        If a point falls into an obstacle, snaps to the absolute nearest feasible node first.
        """
        id1 = self._snap_to_valid_id(pt1[0], pt1[1])
        id2 = self._snap_to_valid_id(pt2[0], pt2[1])
        return float(self.dist_matrix[id1, id2])

    def get_geodesic_distance_matrix(self, coords):
        """
        Returns an NxN dense symmetric distance matrix for N physical coordinates.
        """
        N = len(coords)
        D = np.zeros((N, N))
        ids = [self._snap_to_valid_id(x, y) for (x, y) in coords]
        for i in range(N):
            for j in range(i+1, N):
                d = float(self.dist_matrix[ids[i], ids[j]])
                D[i, j] = d
                D[j, i] = d
        return D

    def _snap_to_valid_id(self, x, y):
        """
        Finds the nearest VALID grid cell ID for given (x,y).
        If (r,c) is inside an obstacle, searches for the closest valid neighbor via Euclidean dist.
        """
        c = int(round(x / self.grid_spacing))
        r = int(round(y / self.grid_spacing))
        c = np.clip(c, 0, self.cols - 1)
        r = np.clip(r, 0, self.rows - 1)
        
        if self.grid[r, c] == 1:
            return self.coord_to_id[(r, c)]
            
        # If obstructed, do a brute force search against all valid cells (Euclidean nearest)
        valid_arr = np.array(self.valid_indices)
        dists = (valid_arr[:, 0] - r)**2 + (valid_arr[:, 1] - c)**2
        nearest_idx = np.argmin(dists)
        return nearest_idx # The index in valid_indices is the ID!

    # ── Repair Methods ────────────────────────────────────────────────────────

    def snap_to_valid_coords(self, coords):
        """
        Strictly projects coordinates back onto the valid manifold ONLY if they are outside
        the green zone. If a coordinate is already valid, its continuous precision is PRESERVED.
        """
        snapped = np.copy(coords)
        for i, (x, y) in enumerate(coords):
            c_int = int(round(x / self.grid_spacing))
            r_int = int(round(y / self.grid_spacing))
            
            is_valid = False
            if 0 <= r_int < self.rows and 0 <= c_int < self.cols:
                if self.grid[r_int, c_int] == 1:
                    is_valid = True
                    
            if not is_valid:
                valid_idx = self._snap_to_valid_id(x, y)
                r, c = self.valid_indices[valid_idx]
                snapped[i, 0] = c * self.grid_spacing
                snapped[i, 1] = r * self.grid_spacing
        return snapped

    # ── compatibility wrappers ───────────────────────────────────────────────

    def physical_to_index(self, x, y):
        c = int(round(x / self.grid_spacing))
        r = int(round(y / self.grid_spacing))
        c = np.clip(c, 0, self.cols - 1)
        r = np.clip(r, 0, self.rows - 1)
        if self.grid[r, c] == 1:
            return (r, c)
        return None

    def physical_bounds(self):
        return (self.cols - 1) * self.grid_spacing, (self.rows - 1) * self.grid_spacing
        
    @property
    def x_max(self):
        return (self.cols - 1) * self.grid_spacing

    @property
    def y_max(self):
        return (self.rows - 1) * self.grid_spacing

if __name__ == "__main__":
    mm = ManifoldManager.l_shape(grid_size=20, spacing=5.0)
    p1 = (5.0, 90.0)
    p2 = (90.0, 90.0)
    print(f"Geodesic {p1} -> {p2} = {mm.get_geodesic_distance(p1, p2):.2f}")
    # Euclidean would be 85.0, but in L-shape it'll go through the corner node!
