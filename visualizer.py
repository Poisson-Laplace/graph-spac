import matplotlib.pyplot as plt
import numpy as np
from spac_analysis import SPACAnalysis

class Visualizer:
    def __init__(self, grid_manager):
        self.gm = grid_manager

    def plot_results(self, best_array, filename='result.png'):
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # plot 1: Grid and Array
        # draw grid
        rows, cols = self.gm.rows, self.gm.cols
        grid_display = np.zeros((rows, cols))
        
        # mark obstacles
        for r in range(rows):
            for c in range(cols):
                if not self.gm.is_valid(r, c):
                    grid_display[r, c] = -1 # obstacle
                else:
                    grid_display[r, c] = 0 # empty valid
        
        # mark sensors
        for x, y in best_array:
            grid_display[x, y] = 1 # sensor
            
        ax[0].imshow(grid_display, cmap='viridis', origin='upper')
        ax[0].set_title(f"Optimized Array Configuration\n({len(best_array)} sensors)")
        ax[0].set_xlabel("Column (Y)")
        ax[0].set_ylabel("Row (X)")
        
        # plot 2: k-space ARF
        # use NO jittering for final visualization to show base design
        phys_coords = self.gm.get_physical_coords(best_array, jitter=False)
        spac = SPACAnalysis(phys_coords)
        
        # calculate k-range based on array geometry
        coords_array = np.array(phys_coords)
        
        # calculate all pairwise distances
        dists = []
        for i in range(len(phys_coords)):
            for j in range(i+1, len(phys_coords)):
                d = np.sqrt((coords_array[i,0]-coords_array[j,0])**2 + 
                           (coords_array[i,1]-coords_array[j,1])**2)
                if d > 0:
                    dists.append(d)
        
        if len(dists) > 0:
            min_spacing = np.min(dists)
            aperture = np.max(dists)
            
            # use k_max based on min spacing (Nyquist), but capped
            k_max = 2 * np.pi / min_spacing
            k_max = min(k_max, 1.0)
        else:
            k_max = 0.5
        
        k_range = np.linspace(-k_max, k_max, 150)
        kx, ky, arf = spac.compute_arf(k_range, k_range)
        
        # calculate metrics for display
        peak, sll, main_lobe_radius, dist_grid = spac.calculate_metrics(arf, kx, ky)
        weighted_penalty = spac.calculate_weighted_penalty(arf, dist_grid)
        
        im = ax[1].imshow(arf, extent=[-k_max, k_max, -k_max, k_max], origin='lower', cmap='inferno')
        ax[1].set_title(f"Array Response Function (k-space)\nSLL: {sll:.4f}, Weighted Penalty: {weighted_penalty:.4f}")
        ax[1].set_xlabel("kx (rad/m)")
        ax[1].set_ylabel("ky (rad/m)")
        plt.colorbar(im, ax=ax[1], label="Normalized Power")
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Saved visualization to {filename}")
        # plt.show() # commented out for headless environment

if __name__ == "__main__":
    pass
