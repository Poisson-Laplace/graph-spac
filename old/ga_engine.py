import numpy as np
import random
from spac_analysis import SPACAnalysis

class GAEngine:
    def __init__(self, grid_manager, population_size=50, n_sensors=10, generations=100, mutation_rate=0.1):
        self.gm = grid_manager
        self.pop_size = population_size
        self.n_sensors = n_sensors
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.valid_indices = self.gm.get_accessible_indices()
        
        # k-space range for evaluation
        self.k_range = np.linspace(-np.pi, np.pi, 50)

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            # randomly select n_sensors from valid indices
            if len(self.valid_indices) >= self.n_sensors:
                indices = random.sample(self.valid_indices, self.n_sensors)
                self.population.append(indices)
            else:
                raise ValueError("Not enough valid spots on the grid for the requested number of sensors.")

    def fitness(self, individual):
        
        # convert grid indices to physical coordinates WITH jittering
        phys_coords = self.gm.get_physical_coords(individual, jitter=True, jitter_amount=0.4)
        
        coords_array = np.array(phys_coords)
        
        # calculate all pairwise distances
        dists = []
        for i in range(len(phys_coords)):
            for j in range(i+1, len(phys_coords)):
                d = np.sqrt((coords_array[i,0]-coords_array[j,0])**2 + 
                           (coords_array[i,1]-coords_array[j,1])**2)
                if d > 0:
                    dists.append(d)
        
        if len(dists) == 0:
            return 0.0  # invalid array
        
        min_spacing = np.min(dists)
        max_spacing = np.max(dists)  # aperture
        mean_spacing = np.mean(dists)
        
        # spacing diversity score: reward arrays where sensors aren't clustered
        # ideal: min_spacing is not too small (prevents clustering)
        # using ratio of min to mean spacing
        spacing_score = min_spacing / mean_spacing if mean_spacing > 0 else 0
        
        # aperture score: reward larger apertures (better resolution)
        # normalize by grid size
        grid_diagonal = self.gm.grid_spacing * np.sqrt(self.gm.rows**2 + self.gm.cols**2)
        aperture_score = max_spacing / grid_diagonal if grid_diagonal > 0 else 0
        
        # now compute ARF
        spac = SPACAnalysis(phys_coords)
        
        # use fixed k-range for consistent evaluation
        # based on typical SPAC target: wavelengths from min_spacing to aperture
        k_max = np.pi / min_spacing  # nyquist
        k_max = min(k_max, 0.5)  # cap for numerical stability
        k_range = np.linspace(-k_max, k_max, 60)
        
        kx, ky, arf = spac.compute_arf(k_range, k_range)
        peak, sll, main_lobe_radius, dist_grid = spac.calculate_metrics(arf, kx, ky)
        
        # calculate distance-weighted penalty
        weighted_penalty = spac.calculate_weighted_penalty(arf, dist_grid)
        
        # combined fitness:
        # - Low SLL is good
        # - Low weighted_penalty is good  
        # - High spacing_score is good (sensors well distributed)
        # - High aperture_score is good (large array)
        
        sll_score = 1.0 / (1.0 + sll)
        concentration_score = 1.0 / (1.0 + weighted_penalty * 5)
        
        # weighted combination
        fitness = (0.2 * sll_score + 
                   0.3 * concentration_score + 
                   0.3 * spacing_score + 
                   0.2 * aperture_score)
        
        return fitness

    def select(self, fitnesses):
        
        tournament_size = 3
        selected = []
        for _ in range(self.pop_size):
            candidates = random.sample(list(zip(self.population, fitnesses)), tournament_size)
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        
        if random.random() < 0.8: # crossover probability
            point = random.randint(1, self.n_sensors - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        
        if random.random() < self.mutation_rate:
            idx_to_change = random.randint(0, self.n_sensors - 1)
            new_pos = random.choice(self.valid_indices)
            # ensure uniqueness if possible, though strict uniqueness might not be critical for SPAC, 
            # usually we don't want two sensors at exact same spot.
            if new_pos not in individual:
                individual[idx_to_change] = new_pos
        return individual

    def run(self):
        self.initialize_population()
        
        best_overall = None
        best_fitness_overall = -1
        
        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in self.population]
            
            # track best
            max_fit = max(fitnesses)
            best_ind = self.population[fitnesses.index(max_fit)]
            
            if max_fit > best_fitness_overall:
                best_fitness_overall = max_fit
                best_overall = best_ind
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Best Fitness = {max_fit:.4f}")
            
            # selection
            selected = self.select(fitnesses)
            
            # next gen
            next_gen = []
            for i in range(0, self.pop_size, 2):
                p1 = selected[i]
                p2 = selected[i+1] if i+1 < len(selected) else selected[0]
                
                c1, c2 = self.crossover(p1, p2)
                
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                
                next_gen.extend([c1, c2])
            
            self.population = next_gen[:self.pop_size]
            
        return best_overall, best_fitness_overall

if __name__ == "__main__":
    from grid_manager import GridManager
    gm = GridManager('matrix.xlsx') # ensure matrix.xlsx exists or it uses default
    ga = GAEngine(gm, generations=20)
    best, fit = ga.run()
    print("Best Array:", best)
