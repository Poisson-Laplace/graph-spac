import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=0, xl=-5, xu=5)
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = [x[0]**2 + x[1]**2]
        out["my_metric"] = x[0] + x[1]

res = minimize(MyProblem(), NSGA2(pop_size=10), termination=('n_gen', 2), save_history=True)
pop0 = res.history[0].pop
print(pop0[0].get("my_metric"))
