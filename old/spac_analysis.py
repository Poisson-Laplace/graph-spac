import numpy as np
import matplotlib.pyplot as plt

class SPACAnalysis:
    def __init__(self, sensor_coords):
        
        self.sensor_coords = np.array(sensor_coords)
        self.n_sensors = len(sensor_coords)

    def compute_arf(self, kx_range, ky_range):
        
        kx_grid, ky_grid = np.meshgrid(kx_range, ky_range)
        arf = np.zeros_like(kx_grid, dtype=float)
        
        # vectorized calculation
        # k_vec: (nk, nk, 2)
        # r_vec: (n_sensors, 2)
        
        # we want to compute sum_j exp(i * (kx*xj + ky*yj))
        
        complex_sum = np.zeros_like(kx_grid, dtype=complex)
        
        for x, y in self.sensor_coords:
            # phase shift for each sensor
            phase = kx_grid * x + ky_grid * y
            complex_sum += np.exp(1j * phase)
            
        arf = np.abs(complex_sum / self.n_sensors)**2
        return kx_grid, ky_grid, arf

    def calculate_metrics(self, arf, kx_grid, ky_grid):
        
        # find peak (should be at 0,0 or very close)
        center_idx = np.unravel_index(np.argmax(arf), arf.shape)
        peak_val = arf[center_idx]
        
        # calculate distance of each point from the peak
        peak_kx = kx_grid[center_idx]
        peak_ky = ky_grid[center_idx]
        
        dist_grid = np.sqrt((kx_grid - peak_kx)**2 + (ky_grid - peak_ky)**2)
        
        # flatten for easier processing
        dists = dist_grid.flatten()
        vals = arf.flatten()
        
        # sort by distance
        sorted_indices = np.argsort(dists)
        sorted_dists = dists[sorted_indices]
        sorted_vals = vals[sorted_indices]
        
        # find the first minimum to define Main Lobe boundary
        main_lobe_radius = 0.0
        
        for i in range(1, len(sorted_vals) - 1):
            if sorted_vals[i] < 0.9 * peak_val:
                if sorted_vals[i+1] > sorted_vals[i]:
                    main_lobe_radius = sorted_dists[i]
                    break
        
        if main_lobe_radius == 0.0:
            main_lobe_radius = np.max(sorted_dists)

        # side lobes are everything outside this radius
        mask_side_lobes = dist_grid > main_lobe_radius
        side_lobes = arf[mask_side_lobes]
        
        if len(side_lobes) > 0:
            max_sll = np.max(side_lobes)
        else:
            max_sll = 0.0
            
        return peak_val, max_sll, main_lobe_radius, dist_grid
    
    def calculate_weighted_penalty(self, arf, dist_grid):
        
        # normalize distances to [0, 1] for stability
        max_dist = np.max(dist_grid)
        if max_dist > 0:
            norm_dist = dist_grid / max_dist
        else:
            norm_dist = dist_grid
        
        # weighted penalty: energy at position weighted by distance squared
        # this penalizes any energy far from center heavily
        total_energy = np.sum(arf)
        if total_energy > 0:
            weighted_penalty = np.sum(arf * (norm_dist ** 2)) / total_energy
        else:
            weighted_penalty = 1.0  # worst case
        
        return weighted_penalty

if __name__ == "__main__":
    # test with a simple cross array
    coords = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
    spac = SPACAnalysis(coords)
    k = np.linspace(-np.pi, np.pi, 1000)
    kx, ky, arf = spac.compute_arf(k, k)
    peak, sll = spac.calculate_metrics(arf, kx, ky)
    print(f"Peak: {peak}, SLL: {sll}")
