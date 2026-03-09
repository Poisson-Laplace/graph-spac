

import numpy as np
import random
from spac_graph import SPACGraph, SPACResponseAnalyzer

class SPACOptimizer:
    
    
    def __init__(self, grid_manager, 
                 n_sensors=10,
                 target_depth=50.0,
                 vs_halfspace=500.0,
                 focus_point=None,
                 focus_direction=None,
                 population_size=50,
                 generations=100,
                 mutation_rate=0.15,
                 elite_ratio=0.1,
                 d_min=None,
                 random_seed=42):
        
        self.gm = grid_manager
        self.n_sensors = n_sensors
        self.target_depth = target_depth
        self.vs_halfspace = vs_halfspace
        self.focus_point = focus_point
        self.focus_direction = focus_direction
        
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = max(2, int(elite_ratio * population_size))
        
        # minimum inter-sensor distance (practical field constraint)
        self.d_min = d_min if d_min is not None else grid_manager.grid_spacing
        
        # reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.valid_indices = self.gm.get_accessible_indices()
        self.population = []
        
        # derived parameters
        self.lambda_max = 4 * target_depth
        self.r_required = self.lambda_max / 3
        
    def _indices_to_physical(self, indices):
        
        return self.gm.get_physical_coords(indices, jitter=True, jitter_amount=0.3)
    
    def initialize_population(self):
        
        self.population = []
        
        for _ in range(self.pop_size):
            if len(self.valid_indices) >= self.n_sensors:
                indices = random.sample(self.valid_indices, self.n_sensors)
                self.population.append(list(indices))
            else:
                raise ValueError(f"Not enough valid positions ({len(self.valid_indices)}) "
                               f"for {self.n_sensors} sensors")
    
    def _check_d_min(self, individual):
        
        phys_coords = self._indices_to_physical(individual)
        n = len(phys_coords)
        for i in range(n):
            for j in range(i + 1, n):
                dx = phys_coords[i][0] - phys_coords[j][0]
                dy = phys_coords[i][1] - phys_coords[j][1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < self.d_min:
                    return False
        return True

    def fitness(self, individual):
        
        # convert to physical coordinates
        phys_coords = self._indices_to_physical(individual)
        
        # build graph
        graph = SPACGraph(phys_coords)
        
        # hard penalty: d_min constraint
        if not self._check_d_min(individual):
            return 0.0
        
        # check for degenerate cases
        if graph.r_max < 1e-6:
            return 0.0
        
        # create analyzer
        analyzer = SPACResponseAnalyzer(
            graph, 
            target_depth=self.target_depth,
            vs_halfspace=self.vs_halfspace
        )
        
        # 1. Bessel Coverage Score (40%)
        bessel_score, _ = analyzer.compute_bessel_coverage(freq_range=None)
        
        # 2. Depth Penalty -> Score (25%)
        depth_penalty = analyzer.compute_depth_penalty()
        depth_score = 1.0 - depth_penalty
        
        # 3. k-space Quality (20%)
        # use appropriate k-range based on array geometry
        k_min = 2 * np.pi / (graph.r_max * 4) if graph.r_max > 0 else 0.01
        k_max = 2 * np.pi / graph.r_min if graph.r_min > 0 else 1.0
        k_max = min(k_max, 0.5)  # cap
        k_range = np.linspace(-k_max, k_max, 50)
        
        kx, ky, arf = analyzer.compute_arf(k_range)
        
        # calculate SLL
        center_idx = arf.shape[0] // 2, arf.shape[1] // 2
        peak_val = arf[center_idx]
        
        # mask center region
        n = arf.shape[0]
        y, x = np.ogrid[:n, :n]
        center_mask = ((x - n//2)**2 + (y - n//2)**2) <= (n//8)**2
        
        side_lobes = arf.copy()
        side_lobes[center_mask] = 0
        max_side_lobe = np.max(side_lobes)
        
        sll_db = -20 * np.log10(max_side_lobe / peak_val + 1e-10) if peak_val > 0 else 0
        kspace_score = min(sll_db / 20.0, 1.0)  # normalize: -20dB = perfect
        
        # 4. Azimuthal Coverage (10%) - includes isotropy when no noise direction
        azimuth_score, isotropy_score = analyzer.compute_azimuthal_coverage(self.focus_direction)
        
        # 5. Focus Area (5%)
        focus_score = analyzer.compute_focus_area_density(self.focus_point)
        
        # combined fitness
        # when no noise direction, isotropy is already included in azimuth_score
        fitness = (0.35 * bessel_score +
                   0.25 * depth_score +
                   0.20 * kspace_score +
                   0.15 * azimuth_score +  # increased for isotropy importance
                   0.05 * focus_score)
        
        return fitness
    
    def select_parents(self, fitnesses):
        
        selected = []
        tournament_size = 3
        
        for _ in range(self.pop_size - self.elite_count):
            candidates = random.sample(list(zip(self.population, fitnesses)), tournament_size)
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1, parent2):
        
        if random.random() > 0.8:
            return parent1[:], parent2[:]
        
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        
        # child 1
        child1 = [None] * n
        child1[start:end] = parent1[start:end]
        remaining = [x for x in parent2 if x not in child1]
        idx = 0
        for i in range(n):
            if child1[i] is None:
                child1[i] = remaining[idx]
                idx += 1
        
        # child 2
        child2 = [None] * n
        child2[start:end] = parent2[start:end]
        remaining = [x for x in parent1 if x not in child2]
        idx = 0
        for i in range(n):
            if child2[i] is None:
                child2[i] = remaining[idx]
                idx += 1
        
        return child1, child2
    
    def mutate(self, individual):
        
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(individual) - 1)
            new_pos = random.choice(self.valid_indices)
            
            # avoid duplicates
            attempts = 0
            while new_pos in individual and attempts < 10:
                new_pos = random.choice(self.valid_indices)
                attempts += 1
            
            if new_pos not in individual:
                individual[idx] = new_pos
        
        # spread mutation: encourage larger aperture
        if random.random() < self.mutation_rate / 2:
            # find current centroid
            phys = self._indices_to_physical(individual)
            centroid = np.mean(phys, axis=0)
            
            # find sensor closest to centroid
            distances = [np.sqrt((p[0]-centroid[0])**2 + (p[1]-centroid[1])**2) 
                        for p in phys]
            closest_idx = np.argmin(distances)
            
            # try to move it to a more peripheral position
            peripheral_candidates = [idx for idx in self.valid_indices 
                                    if idx not in individual]
            if peripheral_candidates:
                # prefer positions far from centroid
                cand_phys = self._indices_to_physical(peripheral_candidates)
                cand_dists = [np.sqrt((p[0]-centroid[0])**2 + (p[1]-centroid[1])**2) 
                             for p in cand_phys]
                
                # select from top 20% farthest
                n_top = max(1, len(peripheral_candidates) // 5)
                top_indices = np.argsort(cand_dists)[-n_top:]
                new_pos = peripheral_candidates[random.choice(top_indices)]
                individual[closest_idx] = new_pos
        
        return individual
    
    def run(self, verbose=True):
        
        self.initialize_population()
        
        best_ever = None
        best_fitness_ever = -1
        fitness_history = []
        
        for gen in range(self.generations):
            # evaluate fitness
            fitnesses = [self.fitness(ind) for ind in self.population]
            
            # track best
            max_fit = max(fitnesses)
            best_idx = fitnesses.index(max_fit)
            best_ind = self.population[best_idx]
            
            if max_fit > best_fitness_ever:
                best_fitness_ever = max_fit
                best_ever = best_ind[:]
            
            fitness_history.append(max_fit)
            
            if verbose and gen % 10 == 0:
                # get stats for best individual
                phys = self._indices_to_physical(best_ind)
                graph = SPACGraph(phys)
                print(f"Gen {gen:3d}: Fitness={max_fit:.4f} | "
                      f"r_min={graph.r_min:.1f}m, r_max={graph.r_max:.1f}m | "
                      f"Required r_max={self.r_required:.1f}m")
            
            # elitism: Keep best individuals
            sorted_pop = sorted(zip(self.population, fitnesses), 
                               key=lambda x: x[1], reverse=True)
            elites = [ind[:] for ind, _ in sorted_pop[:self.elite_count]]
            
            # selection
            parents = self.select_parents(fitnesses)
            
            # create next generation
            next_gen = elites[:]
            
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(parents, 2)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                next_gen.extend([c1, c2])
            
            self.population = next_gen[:self.pop_size]
        
        return best_ever, best_fitness_ever, fitness_history

if __name__ == "__main__":
    from grid_manager import GridManager
    
    gm = GridManager('matrix.xlsx', grid_spacing=5.0)
    
    optimizer = SPACOptimizer(
        gm,
        n_sensors=10,
        target_depth=30,
        vs_halfspace=400,
        generations=30,
        population_size=40
    )
    
    best, fitness, history = optimizer.run()
    print(f"\nBest fitness: {fitness:.4f}")
    print(f"Best array: {best}")
