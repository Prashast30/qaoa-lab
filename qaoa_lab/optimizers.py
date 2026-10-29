import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize


class Tracker:
    """Counts circuit evaluations and keeps the best-so-far cost after each one."""

    def __init__(self, fn, budget):
        self.fn = fn
        self.budget = budget
        self.count = 0
        self.best = np.inf
        self.history = []

    def record(self, val, extra=0):
        self.best = min(self.best, val)
        n = 1 + extra
        self.count += n
        self.history.extend([self.best] * n)

    def __call__(self, x):
        val = float(self.fn(np.asarray(x, dtype=float)))
        self.record(val)
        return val


def run_cobyla(tracker, x0):
    remaining = tracker.budget - tracker.count
    res = minimize(tracker, x0, method="COBYLA",
                   options={"maxiter": remaining, "rhobeg": 0.5})
    return res.x


def run_adam(tracker, x0, stepsize=0.1):
    # Each step is charged 1 forward pass + 2 evals per parameter, which is
    # what the parameter-shift rule would cost on real hardware.
    opt = qml.AdamOptimizer(stepsize=stepsize)
    x = pnp.array(x0, requires_grad=True)
    n = len(x0)
    while tracker.count + 1 + 2 * n <= tracker.budget:
        x, val = opt.step_and_cost(tracker.fn, x)
        tracker.record(float(val), extra=2 * n)
    return np.array(x)


def run_spsa(tracker, x0, rng, a=0.5, c=0.15, A=10, alpha=0.602, gamma=0.101):
    x = np.array(x0, dtype=float)
    k = 0
    while tracker.count + 2 <= tracker.budget:
        k += 1
        ak = a / (k + A) ** alpha
        ck = c / k ** gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        g = (tracker(x + ck * delta) - tracker(x - ck * delta)) / (2 * ck * delta)
        x = x - ak * g
    return x


def run_optimizer(name, tracker, x0, rng):
    if name == "cobyla":
        return run_cobyla(tracker, x0)
    if name == "adam":
        return run_adam(tracker, x0)
    if name == "spsa":
        return run_spsa(tracker, x0, rng)
    raise ValueError(name)