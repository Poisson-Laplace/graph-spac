

import numpy as np
from scipy.special import j0  # bessel function of first kind, order 0
import networkx as nx

class SPACGraph:
    
    
    def __init__(self, sensor_coords):
        
        self.coords = np.array(sensor_coords)
        self.n_sensors = len(sensor_coords)
        self.graph = self._build_graph()
        self.edges = self._compute_edges()
        
    def _build_graph(self):
        
        G = nx.complete_graph(self.n_sensors)
        for i, (x, y) in enumerate(self.coords):
            G.nodes[i]['pos'] = (x, y)
        return G
    
    def _compute_edges(self):
        
        edges = []
        for i in range(self.n_sensors):
            for j in range(i + 1, self.n_sensors):
                dx = self.coords[j, 0] - self.coords[i, 0]
                dy = self.coords[j, 1] - self.coords[i, 1]
                distance = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)  # mathematical: 0=East, CCW, in [-pi, pi]
                
                edges.append({
                    'i': i, 'j': j,
                    'distance': distance,
                    'angle': angle,
                    'dx': dx, 'dy': dy
                })
        return edges
    
    def get_distance_distribution(self):
        
        return np.array([e['distance'] for e in self.edges])
    
    def get_angle_distribution(self):
        
        return np.array([e['angle'] for e in self.edges])
    
    def get_angle_distribution_geo(self):
        
        angles_math = self.get_angle_distribution()
        # convert to geographic and include both directions (edge is bidirectional for display)
        geo = (np.pi / 2 - angles_math) % (2 * np.pi)            # [0, 2pi)
        geo_opp = (geo + np.pi) % (2 * np.pi)                    # opposite direction
        all_angles = np.concatenate([geo, geo_opp])
        # fold to [0, pi)
        folded = all_angles % np.pi
        return folded
    
    @property
    def r_min(self):
        
        dists = self.get_distance_distribution()
        return np.min(dists) if len(dists) > 0 else 0
    
    @property
    def r_max(self):
        
        dists = self.get_distance_distribution()
        return np.max(dists) if len(dists) > 0 else 0
    
    @property
    def aperture(self):
        
        return self.r_max

class SPACResponseAnalyzer:
    
    
    def __init__(self, spac_graph, target_depth=None, vs_halfspace=None):
        
        self.graph = spac_graph
        self.target_depth = target_depth or 50.0  # default 50m
        self.vs_halfspace = vs_halfspace or 500.0  # default 500 m/s
        
        # derived parameters
        self.lambda_max = 4 * self.target_depth  # max wavelength needed
        self.lambda_min = self.target_depth / 3  # min wavelength (approximate)
        
        # required distances based on wavelength
        # rule of thumb: r should be ~λ/3 to λ/2 for good J₀ sampling
        self.r_required_max = self.lambda_max / 2  # for lowest frequency
        self.r_required_min = self.lambda_min / 3  # for highest frequency
        
    def compute_arf(self, k_range):
        
        coords = self.graph.coords
        n = self.graph.n_sensors
        
        # meshgrid with indexing='xy' ensures:
        # kx varies along columns (axis 1, horizontal in imshow)
        # ky varies along rows (axis 0, vertical in imshow)
        kx_grid, ky_grid = np.meshgrid(k_range, k_range, indexing='xy')
        
        complex_sum = np.zeros_like(kx_grid, dtype=complex)
        for x, y in coords:
            phase = kx_grid * x + ky_grid * y
            complex_sum += np.exp(1j * phase)
        
        arf = np.abs(complex_sum / n) ** 2
        return kx_grid, ky_grid, arf
    
    def compute_bessel_coverage(self, freq_range):
        
        distances = self.graph.get_distance_distribution()
        
        # check if we have distances in the required range
        has_short = np.any(distances <= self.r_required_min * 1.5)
        has_long = np.any(distances >= self.r_required_max * 0.7)
        
        # ideal: Log-uniform distribution of distances
        log_dists = np.log10(distances[distances > 0])
        
        if len(log_dists) > 1:
            # check uniformity in log space
            log_range = np.max(log_dists) - np.min(log_dists)
            n_bins = min(10, len(log_dists))
            hist, _ = np.histogram(log_dists, bins=n_bins)
            uniformity = 1.0 - (np.std(hist) / (np.mean(hist) + 1e-10))
        else:
            log_range = 0
            uniformity = 0
        
        # combined score
        range_score = log_range / 2.0  # normalize: 2 decades = perfect
        depth_score = 1.0 if has_long else 0.5
        shallow_score = 1.0 if has_short else 0.5
        
        coverage_score = (range_score * 0.4 + 
                          uniformity * 0.3 + 
                          depth_score * 0.2 + 
                          shallow_score * 0.1)
        
        return min(coverage_score, 1.0), distances
    
    def compute_depth_penalty(self):
        
        r_max = self.graph.r_max
        required = self.lambda_max / 3
        
        if r_max >= required:
            return 0.0  # no penalty
        else:
            # penalty proportional to deficiency
            return (required - r_max) / required
    
    def compute_azimuthal_coverage(self, focus_direction=None):
        
        angles_math = self.graph.get_angle_distribution()  # [-pi, pi]
        
        if len(angles_math) == 0:
            return 0.0, 0.0
        
        # include both directions (edge i->j and j->i are physically the same
        # but must be counted at theta AND theta+pi for full coverage)
        angles_both = np.concatenate([angles_math, angles_math + np.pi])
        # wrap to [-pi, pi]
        angles_both = np.arctan2(np.sin(angles_both), np.cos(angles_both))
        
        # calculate isotropy score: how uniform is the angular distribution?
        n_bins = 18  # 20-degree bins over [-180, 180]
        hist, _ = np.histogram(angles_both, bins=n_bins, range=(-np.pi, np.pi))
        
        if np.sum(hist) > 0:
            hist_norm = hist / np.sum(hist)
        else:
            hist_norm = hist
        
        if np.mean(hist_norm) > 0:
            cv = np.std(hist_norm) / np.mean(hist_norm)
            isotropy_score = max(0, 1.0 - cv)
        else:
            isotropy_score = 0.0
        
        if focus_direction is not None:
            # focus_direction is geographic (0=North, CW positive, degrees)
            # convert to mathematical: math_angle = pi/2 - geo
            focus_math = np.pi / 2 - np.radians(focus_direction)
            perp_dir = focus_math + np.pi / 2
            # reward edges aligned with perpendicular direction
            angle_diff = np.abs(np.cos(angles_math - perp_dir))
            directional_score = np.mean(angle_diff)
        else:
            directional_score = isotropy_score
        
        return directional_score, isotropy_score
    
    def compute_focus_area_density(self, focus_point=None, focus_radius=None):
        
        if focus_point is None:
            return 1.0  # no focus, neutral score
        
        focus_radius = focus_radius or (self.graph.r_max / 4)
        
        coords = self.graph.coords
        fx, fy = focus_point
        
        distances_to_focus = np.sqrt((coords[:, 0] - fx)**2 + (coords[:, 1] - fy)**2)
        n_in_focus = np.sum(distances_to_focus <= focus_radius)
        
        # want some sensors near focus, but not all (need aperture too)
        ideal_ratio = 0.3  # 30% near focus is good
        actual_ratio = n_in_focus / self.graph.n_sensors
        
        # score based on how close to ideal
        score = 1.0 - abs(actual_ratio - ideal_ratio) / ideal_ratio
        return max(0.0, score)

if __name__ == "__main__":
    # test with a simple array
    coords = [(0, 0), (10, 0), (20, 0), (0, 10), (10, 10), (20, 10)]
    
    graph = SPACGraph(coords)
    print(f"Number of sensors: {graph.n_sensors}")
    print(f"Number of edges: {len(graph.edges)}")
    print(f"Distance range: {graph.r_min:.1f}m to {graph.r_max:.1f}m")
    
    analyzer = SPACResponseAnalyzer(graph, target_depth=30, vs_halfspace=500)
    print(f"Target λ_max: {analyzer.lambda_max:.1f}m")
    print(f"Required r_max: {analyzer.r_required_max:.1f}m")
    
    coverage, dists = analyzer.compute_bessel_coverage(freq_range=None)
    print(f"Bessel coverage score: {coverage:.3f}")
