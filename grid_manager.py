import pandas as pd
import numpy as np
import os

class GridManager:
    def __init__(self, file_path='matrix.xlsx', grid_spacing=10.0, default_grid_size=100):
        self.file_path = file_path
        self.grid_spacing = grid_spacing
        self.default_grid_size = default_grid_size
        self.grid = None
        self.rows = 0
        self.cols = 0
        self.load_grid()

    def load_grid(self):
        # if file_path is None, create default grid (no constraints)
        if self.file_path is None:
            self.create_default_grid()
            return
            
        if os.path.exists(self.file_path):
            try:
                # load without header to treat all rows as data
                df = pd.read_excel(self.file_path, header=None)
                self.grid = df.values
                self.rows, self.cols = self.grid.shape
                print(f"Loaded grid from {self.file_path}. Size: {self.rows}x{self.cols}")
            except Exception as e:
                print(f"Error loading {self.file_path}: {e}")
                self.create_default_grid()
        else:
            print(f"{self.file_path} not found. Initializing default grid.")
            self.create_default_grid()

    def create_default_grid(self, size=None):
        
        if size is None:
            size = self.default_grid_size
        self.grid = np.ones((size, size), dtype=int)
        self.rows, self.cols = size, size
        print(f"Created default {size}x{size} grid.")

    def is_valid(self, x, y):
        
        if 0 <= x < self.rows and 0 <= y < self.cols:
            return self.grid[x, y] == 1
        return False

    def get_accessible_indices(self):
        
        return list(zip(*np.where(self.grid == 1)))

    def get_physical_coords(self, indices, jitter=True, jitter_amount=0.4):
        
        coords = []
        for r, c in indices:
            x = c * self.grid_spacing
            y = r * self.grid_spacing
            
            if jitter:
                # deterministic jitter based on coordinate hash
                # this ensures same coordinates always get same jitter
                seed = hash((r, c)) % (2**31)
                rng = np.random.RandomState(seed)
                offset = jitter_amount * self.grid_spacing
                x += rng.uniform(-offset, offset)
                y += rng.uniform(-offset, offset)
            
            coords.append((x, y))
        return coords

if __name__ == "__main__":
    gm = GridManager()
    print("Grid shape:", gm.grid.shape)
    print("Is (0,0) valid?", gm.is_valid(0, 0))
