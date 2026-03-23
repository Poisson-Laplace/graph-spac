import os
import sys

# Ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scenarios import DEFAULT_SCENARIO
from main import run_scenario

base_out_dir = "/home/ferat/Desktop/nevertrustclankers/graphspac/paperfigs/ucnoktalar"
os.makedirs(base_out_dir, exist_ok=True)

class Args:
    yeryok = 0
    matrix = None
    dmin = 0.1
    sensors = 10
    gens = 200
    pop = 100
    seed = 1249718046570
    noiseazimuth = None
    los = False
    golomb = False
    poincare = False
    compare = False
    robustness = False
    failure_analysis = False
    levels = False
    kernel = "bessel"

args = Args()

cases = [
    {
        "name": "Case1",
        "desc": "High lam2, zero SLL. Expected: Geometric stagnation, tiny cluster.",
        "weights": {"lam2": 1000.0, "sll": 0.0, "lsd": 0.0, "dr": 0.0, "graph_ent": 0.0, "eta": 0.0},
        "focus": 30.0,
        "gridsize": 30
    },
    {
        "name": "Case2",
        "desc": "Standard Precision Mode (wSLL=5, wLAM2=0.5, wLSD=1). Expected: Optimal Dynamic Equilibrium.",
        "weights": {"lam2": 0.5, "sll": 5.0, "lsd": 1.0, "dr": 0.5, "graph_ent": 0.5, "eta": 0.5},
        "focus": 30.0,
        "gridsize": 30
    },
    {
        "name": "Case3",
        "desc": "Zero lam2, extreme SLL=500. Expected: Sensors pushed to boundaries. Centrifugal disconnection.",
        "weights": {"lam2": 0.0, "sll": 500.0, "lsd": 0.0, "dr": 1000.0, "graph_ent": 0.0, "eta": 0.0},
        "focus": 400.0,
        "gridsize": 100
    }
]

for i, case in enumerate(cases):
    sc = DEFAULT_SCENARIO.copy()
    sc.update({
        "id": i + 1,
        "name": case["name"],
        "description": case["desc"],
        "domain": "open",
        "N": 10,
        "d_min": 0.1,
        "focus": case["focus"],
        "vs": 500.0,
        "kernel": "bessel",
        "mode": "precision",
        "weights": case["weights"],
        "spacing": 5.0,
        "gridsize": case["gridsize"]
    })
    
    out_dir = os.path.join(base_out_dir, case["name"])
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n--- Running {case['name']} ---")
    run_scenario(sc, args, out_dir=out_dir)

print("Done generating all 3 pathological cases!")
