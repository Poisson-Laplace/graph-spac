

import matplotlib.pyplot as plt
import numpy as np
from spac_graph import SPACGraph, SPACResponseAnalyzer

class SPACVisualizer:
    
    
    def __init__(self, grid_manager):
        self.gm = grid_manager
    
    def plot_comprehensive(self, best_indices, target_depth, vs_halfspace, 
                          filename='spac_result.png'):
        
        # convert to physical coordinates (without jitter for visualization)
        phys_coords = self.gm.get_physical_coords(best_indices, jitter=False)
        graph = SPACGraph(phys_coords)
        analyzer = SPACResponseAnalyzer(graph, target_depth, vs_halfspace)
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Array Layout (top-left)
        ax1 = fig.add_subplot(2, 2, 1)
        self._plot_array_layout(ax1, best_indices, phys_coords)
        
        # 2. Distance Histogram (top-right)
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_distance_histogram(ax2, graph, analyzer)
        
        # 3. Azimuthal Rose (bottom-left)
        ax3 = fig.add_subplot(2, 2, 3, projection='polar')
        self._plot_azimuthal_rose(ax3, graph)
        
        # 4. k-space ARF (bottom-right)
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_arf(ax4, graph, analyzer)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved comprehensive visualization to {filename}")
        
        return fig
    
    def _plot_array_layout(self, ax, indices, phys_coords):
        
        rows, cols = self.gm.rows, self.gm.cols
        spacing = self.gm.grid_spacing
        
        # create grid image
        grid_img = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                if not self.gm.is_valid(r, c):
                    grid_img[r, c] = -1  # obstacle
        
        # mark sensors
        for r, c in indices:
            grid_img[r, c] = 1
        
        # convert to physical coordinates for display
        extent = [0, cols * spacing, rows * spacing, 0]
        
        ax.imshow(grid_img, cmap='RdYlGn', origin='upper', extent=extent,
                  vmin=-1, vmax=1, alpha=0.7)
        
        # plot sensors as scatter
        xs = [p[0] for p in phys_coords]
        ys = [p[1] for p in phys_coords]
        ax.scatter(xs, ys, s=150, c='blue', edgecolors='white', linewidth=2, zorder=5)
        
        # draw edges (connections)
        for i in range(len(phys_coords)):
            for j in range(i + 1, len(phys_coords)):
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], 
                       'b-', alpha=0.1, linewidth=0.5)
        
        # number sensors
        for i, (x, y) in enumerate(phys_coords):
            ax.annotate(str(i+1), (x, y), fontsize=8, ha='center', va='center',
                       color='white', fontweight='bold')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Array Configuration\n({len(indices)} sensors, '
                    f'Aperture={np.max([np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2) for i in range(len(xs)) for j in range(i+1, len(xs))]):.1f}m)')
        ax.set_aspect('equal')
    
    def _plot_distance_histogram(self, ax, graph, analyzer):
        
        distances = graph.get_distance_distribution()
        
        # log-scale bins
        if len(distances) > 0 and np.min(distances) > 0:
            log_bins = np.logspace(np.log10(np.min(distances) * 0.9),
                                   np.log10(np.max(distances) * 1.1),
                                   20)
            ax.hist(distances, bins=log_bins, color='steelblue', 
                   edgecolor='white', alpha=0.8)
            ax.set_xscale('log')
        
        # mark required range
        ax.axvline(analyzer.r_required_min, color='red', linestyle='--', 
                  linewidth=2, label=f'r_min required ({analyzer.r_required_min:.1f}m)')
        ax.axvline(analyzer.r_required_max, color='green', linestyle='--', 
                  linewidth=2, label=f'r_max required ({analyzer.r_required_max:.1f}m)')
        
        # mark actual range
        ax.axvline(graph.r_min, color='orange', linestyle='-', 
                  linewidth=2, label=f'r_min actual ({graph.r_min:.1f}m)')
        ax.axvline(graph.r_max, color='darkgreen', linestyle='-', 
                  linewidth=2, label=f'r_max actual ({graph.r_max:.1f}m)')
        
        ax.set_xlabel('Inter-sensor Distance (m)')
        ax.set_ylabel('Number of Pairs')
        ax.set_title(f'Edge Distance Distribution\nTarget Depth: {analyzer.target_depth}m, λ_max: {analyzer.lambda_max:.0f}m')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_azimuthal_rose(self, ax, graph):
        
        # folded geographic azimuths in [0, pi) — includes both directions
        angles_folded = graph.get_angle_distribution_geo()   # [0, pi)
        
        n_bins = 18  # 10-degree bins over [0, pi)
        hist, bin_edges = np.histogram(angles_folded, bins=n_bins, range=(0, np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = np.pi / n_bins
        
        # plot both directions (mirror at pi) for full 360° symmetric rose
        ax.bar(bin_centers, hist, width=width, alpha=0.75,
               color='steelblue', edgecolor='white', linewidth=0.5)
        ax.bar(bin_centers + np.pi, hist, width=width, alpha=0.75,
               color='steelblue', edgecolor='white', linewidth=0.5)
        
        ax.set_title('Azimuthal Distribution\n(Geographic: N=up, CW)')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
    
    def _plot_arf(self, ax, graph, analyzer):
        
        # determine appropriate k-range
        if graph.r_min > 0:
            k_max = min(np.pi / graph.r_min, 0.5)
        else:
            k_max = 0.3
        
        k_range = np.linspace(-k_max, k_max, 150)
        kx, ky, arf = analyzer.compute_arf(k_range)
        
        # plot - transpose for correct axis orientation (kx horizontal, ky vertical)
        im = ax.imshow(arf.T, extent=[-k_max, k_max, -k_max, k_max],
                       origin='lower', cmap='inferno', aspect='equal')
        
        # colorbar
        plt.colorbar(im, ax=ax, label='Normalized Power', shrink=0.8)
        
        # calculate SLL for title
        center = arf.shape[0] // 2
        peak = arf[center, center]
        
        # mask center for SLL
        mask_radius = arf.shape[0] // 8
        y, x = np.ogrid[:arf.shape[0], :arf.shape[1]]
        center_mask = ((x - center)**2 + (y - center)**2) <= mask_radius**2
        side = arf.copy()
        side[center_mask] = 0
        max_side = np.max(side)
        
        sll_db = -20 * np.log10(max_side / peak + 1e-10) if peak > 0 else 0
        
        ax.set_xlabel('kx (rad/m)')
        ax.set_ylabel('ky (rad/m)')
        ax.set_title(f'Array Response Function (k-space)\nSLL: {sll_db:.1f} dB')
        
        # add circle at wavelength of interest
        lambda_target = analyzer.lambda_max / 2  # half max wavelength
        k_target = 2 * np.pi / lambda_target
        if k_target < k_max:
            circle = plt.Circle((0, 0), k_target, fill=False, 
                               color='white', linestyle='--', linewidth=1.5,
                               label=f'λ = {lambda_target:.0f}m')
            ax.add_patch(circle)

if __name__ == "__main__":
    pass
