# GraphSPAC (vol5.2)
**Topological Design and Optimization of Seismic Arrays in Constrained Urban Environments**

GraphSPAC is an advanced, graph-theory-based optimization framework designed to generate ideal seismic sensor array geometries (such as SPAC, ESPAC) specifically for highly constrained urban spaces where classical circular or Euclidean arrays cannot be utilized. 

With **vol5.2**, the framework introduces a paradigm shift from simple *Geometric Fitting* to *Topological Synthesis*. Utilizing a Genetic Algorithm (NSGA-II) combined with Spectral Graph Theory, Dijkstra geodesic pathfinding, and Magnetic Repair operators, it shapes optimal array responses by intelligently navigating obstacles, accommodating directional noise, and escaping non-convex "Euclidean bottlenecks" in complex modern cities.

---

## 🚀 Key Features (vol5.2 Upgrade)

- **Geodesic Pathfinding (`ManifoldManager`)**: Entirely replaced blind Euclidean distance checks with true Dijkstra shortest-paths, seamlessly computing wavefield propagation boundaries precisely around complex city blocks and non-convex constraints.
- **Topological Awareness Engine (Spatial Seeding)**: Replaces traditional blind initialization. This motor aggressively seeds the Gen 0 population strictly across valid geographic nodes, easily preventing "Topological Locking" in hostile and fragmented environments.
- **Magnetic Mutation (`TopologicalRepair`)**: An intelligent repair operator that steps in during Crossover and Mutation. If a sensor drifts into a restricted zone (buildings/walls), it is instantly snapped back onto the valid manifold—preserving the ultra-fine continuous precision of the NSGA-II optimizer without discretizing the search space.
- **Tri-Modal Weighting Sets**: The framework now dynamically applies 3 pre-configured topologies depending on the scenario:
  - **Precision:** Focuses heavily on Side-Lobe Level (SLL) suppression in open areas.
  - **Discovery:** Maximizes Lag-Sampling Density (LSD) in highly constrained geometric bottlenecks.
  - **Robust:** Exponentially scales Algebraic Connectivity ($\lambda_2$) to prevent network fragmentation in "porous" or "island-like" topologies.
- **Dual-Kernel Support**: Supports both idealized Bessel ($J_0$) for isotropic ambient noise and Hankel ($H_0^{(1)}$) to steer sensors and create topological matched filters against strongly directional noise sources.
- **26 Benchmark Scenarios**: Features a massive suite of environments; from simple L-Shapes to advanced topologies like *The Sponge*, *Percolation Labyrinths*, and *2D Golomb Rulers*.

---

## 🛠 Usage & CLI

The entry point of the entire application is `main.py`. The suite features **26 pre-built challenging scenarios** showcasing various geometric and topological constraints.

### Basic Run
Run Scenario 26 (*The Sponge* — highly porous constraint test):
```bash
python main.py --scenario 26 --gens 200 --pop 100
```
This command inherently produces `sc26_layout_final.png` and `sc26_metrics.txt` into the `./results` folder.

### Testing Topological Awareness
You can disable the Spatial Seeding engine to witness how the baseline algorithm fails (clusters within the walls) in non-convex domains by passing the `--noseed` flag:
```bash
python main.py --scenario 26 --gens 200 --pop 100 --noseed
```

### Fully Custom Scenario Override (Sc 0)
You don't need to write code to test a new layout. You can feed your variables dynamically to Sc 0:
```bash
python main.py --scenario 0 --gens 100 --sensors 12 --focus 30.0 --spacing 5.0 --kernel hankel --noiseazimuth 45
```

### Animating the Generation Process (`--levels`)
If you define the `--levels` flag, the engine will dump the best array state at every single generation of the GA into a folder `sc0_levels/`, enabling you to easily compile an evolution animation of the sensors navigating the urban limits.
```bash
python main.py --scenario 0 --gens 100 --pop 50 --levels
```

---

## 📌 Output Panels Breakdown

When a scenario is completed, GraphSPAC produces a beautiful 5-panel High-Resolution PNG figure:
1. **Array Geometry**: The physical configuration, tracking the edges of the Fiedler Vector, demonstrating obstacles and bounds on the valid manifold.
2. **Co-Array Map**: The unique spatial differences ($\Delta x$, $\Delta y$).
3. **Lag-Sampling Density (LSD)**: The cross-section distance distributions versus the required bounds $r_{min}$ and $r_{max}$.
4. **ARF (Array Response Function)**: The 2D wavefield wavenumber representation (in dB to easily show the Side-Lobe Levels).
5. **Rose Diagram**: An angular histogram measuring spatial Isotropy (0 to 1).

---

## 📂 Project Structure

- `main.py`: CLI driver.
- `scenarios.py`: Houses the definitions for the 26 pre-built challenging geographical limitations, structured by their Tri-Modal intent.
- `nsga_optimizer.py`: The wrapper for pymoo handling NSGA-II parameters, custom fitness definitions, the **Topological Repair Operator**, and Golomb logic.
- `manifold_manager.py` / `grid_generators.py`: The brains of the Geodesic mapping. Handles Dijkstra shortest-path calculations and fractal/canyon terrain generation algorithms.
- `graph_metrics.py`: The engine implementing Laplacian matrices, algebraic connectivity ($\lambda_2$), SLL checks, ARF calculations, and Bessel/Hankel error penalization.
- `coarray.py`: Computes continuous pairwise operations.
- `visualizer.py`: Creates the stunning panoramic Multi-Angle Pareto plots and 5-panel topology figures.
- `generate_golomb_figures.py`: An automated script to generate and plot pairwise coordinate histograms compared with classical baselines.
