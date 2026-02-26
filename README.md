# Graphspac

Graph-theoretic layout optimization for Spatial Autocorrelation (SPAC) seismic arrays.

This repository contains the source code for the "Graphspac" algorithm, implementing a Multi-Objective Genetic Algorithm to compute non-symmetrical, robust sensor coordinates over spatially obstructed urban boundaries. 

The accompanying paper is currently under review for *Mathematical Geosciences*.

## Core Dependencies
- `numpy`
- `matplotlib`
- `scipy`
- `pandas` (for parsing boundary matrices)

## Usage

You can test the core algorithm against constraints in the provided `matrix.xlsx` grid.

```bash
# run basic optimization tests
python run_scenarios.py

# specific convergence tracking
python convergence_tracker.py --scenario 2 --gens 60 --pop 50

# co-array spectral gap analysis vs classical geometries
python coarray_histogram.py --scenario 2 --gens 50 --pop 50

# topologocal robustness monte-carlo
python robustness_simulator.py --scenario 2 --gens 50 --pop 50 --trials 20
```

## Structure
- `ga_engine.py` / `spac_optimizer.py`: multi-objective evolutionary solver.
- `spac_graph.py`: complete graph ($K_n$) representations of sensor vertices and inter-station distances.
- `array_analyzer.py`: builds baseline primitive templates (Kennett spirals, nested triangles) for analytical cross-validation.
- `grid_manager.py`: handles hard spatial obstructions mapped from Excel/CSV boundaries.

Authors: Ferat
License: MIT
