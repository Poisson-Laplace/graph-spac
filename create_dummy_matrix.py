import pandas as pd
import numpy as np

# create a 20x20 grid
# 1 represents accessible, 0 represents inaccessible
grid_size = 20
grid = np.ones((grid_size, grid_size), dtype=int)

# add some "obstacles" (0s)
grid[5:8, 5:8] = 0
grid[12:15, 12:15] = 0

df = pd.DataFrame(grid)
df.to_excel('matrix.xlsx', index=False, header=False)
print("Dummy matrix.xlsx created.")
