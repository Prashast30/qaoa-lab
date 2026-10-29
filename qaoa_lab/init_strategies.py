import numpy as np


def random_init(p, rng):
    return rng.uniform(0, np.pi, size=2 * p)


def linear_ramp(p, delta=0.8):
    """Annealing-style schedule: gammas rise, betas fall.
    Betas are negative because our mixer is RX(2*beta) = exp(-i*beta*X)."""
    frac = (np.arange(p) + 0.5) / p
    gammas = 0.5 * delta * frac
    betas = -delta * (1 - frac)
    return np.concatenate([gammas, betas])


def interp(x, p):
    """INTERP heuristic (Zhou et al.): turn optimised p-layer params into a
    starting point for p+1 layers."""
    def up(v):
        out = np.zeros(p + 1)
        for i in range(1, p + 2):
            left = v[i - 2] if i >= 2 else 0.0
            right = v[i - 1] if i <= p else 0.0
            out[i - 1] = (i - 1) / p * left + (p - i + 1) / p * right
        return out

    return np.concatenate([up(x[:p]), up(x[p:])])