# GraphSPAC (vol3)

**Topological Design and Optimization of Seismic Arrays in Constrained Urban Environments**

GraphSPAC is an advanced, graph-theory-based optimization framework designed to generate ideal seismic sensor array geometries (such as SPAC, ESPAC) specifically for highly constrained urban spaces where classical circular arrays cannot be utilized. Utilizing a Genetic Algorithm (NSGA-II) combined with Spectral Graph Theory and matched filter constraints, it shapes array responses by dodging obstacles, accommodating directional noise, and escaping Euclidean bottlenecks.

---

## 🚀 Features (vol3)

- **NSGA-II Optimization:** Multi-objective genetic algorithm minimizing LSD entropy, Side-Lobe Level (SLL), and maximizing algebraic connectivity ($\lambda_2$).
- **Dual-Kernel Support:** Supports both idealized Bessel ($J_0$) for isotropic ambient noise and Hankel ($H_0^{(1)}$) to steer sensors and create topological matched filters against strongly directional noise sources.
- **Robustness against Constraints:** Works perfectly in "L-shaped" corridors, urban canyons, fragmented islands, and percolation-based fractal environments.
- **Parametric Flexibility:** Can be run completely openly `(Sc 0)` from the CLI, overriding numbers of sensors, depths, grid elements, spacing, and azimuthal angles.
- **Custom Excels (The Grid):** You can pass physical building layouts directly via `.xlsx` matrices where `0`s are obstacles and `1`s are accessible land.

## 🛠 Usage & CLI

The entry point of the entire application is `main.py`. The suite features 24 pre-built challenging scenarios showcasing various geometric and topological constraints.

### Basic Run
Run Scenario 11 (Deep Focus 50m constraint showcase):
```bash
python main.py --scenario 11 --gens 200 --pop 100
```
This command inherently produces `sc11_result.png`, `sc11_pareto.png` and `sc11_metrics.txt` into the `./results` folder.

### Fully Custom Scenario Override (Sc 0)
You don't need to write code to test a new layout. You can feed your variables dynamically to `Sc 0`:
```bash
python main.py --scenario 0 --gens 100 --sensors 12 --focus 30.0 --spacing 5.0 --kernel hankel --noiseazimuth 45
```

### Loading Custom Grids via Excel
If you have an area that is obstructed (e.g. your campus or a city street), create an Excel file where cells are either `1` or `0`.
```bash
python main.py --scenario 0 --yeryok 1 --matrix my_campus.xlsx --spacing 3.0 --gens 150
```

### Animating the Generation Process (`--levels`)
If you define the `--levels` flag, the engine will dump the best array state at every single generation of the GA into a folder `sc0_levels/`, enabling you to easily compile an evolution animation.
```bash
python main.py --scenario 0 --gens 100 --pop 50 --levels
```

---

## 📌 Output Panels Breakdown

When a scenario is completed, GraphSPAC produces a 5-panel High-Resolution PNG figure:
1. **Array Geometry:** The physical configuration, tracking the edges of the Fiedler Vector, demonstrating obstacles and bounds.
2. **Co-Array Map:** The unique spatial differences ($\Delta x, \Delta y$).
3. **Lag-Sampling Density (LSD):** The cross-section distance distributions versus the required bounds $r_{min}$ and $r_{max}$.
4. **ARF (Array Response Function):** The 2D wavefield wavenumber representation (in dB to easily show the Side-Lobe Levels).
5. **Rose Diagram:** An angular histogram measuring spatial Isotropy ($0$ to $1$).

---

## 📂 Project Structure

- `main.py`: CLI driver.
- `scenarios.py`: Houses the definitions for the 24 pre-built challenging geographical limitations.
- `nsga_optimizer.py`: The wrapper for `pymoo` handling NSGA-II parameters, custom fitness definitions, and deep focus compensations.
- `graph_metrics.py`: The engine implementing Laplacian matrices, algebraic connectivity ($\lambda_2$), SLL checks, ARF calculations, and Bessel/Hankel error penalization.
- `coarray.py`: Computes continuous pairwise operations.
- `grid_manager.py` / `grid_generators.py`: Handles continuous coordinate matching versus discrete matrix nodes, alongside complex fractal/canyon terrain generation algorithms. 
- `visualizer.py`: Creates the stunning panoramic Multi-Angle Pareto plots and 5-panel topology figures.
