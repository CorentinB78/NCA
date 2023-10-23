import numpy as np


def fermi(omegas, mu, beta):
    """
    Fermi function

    1 / (1 + e^{-beta (omegas - mu)})

    Entirely vectorized, supports infinite beta.
    """
    x = beta * (omegas - mu)
    ### for case beta=inf and omega=mu:
    x = np.nan_to_num(x, copy=False, nan=0.0, posinf=+np.inf, neginf=-np.inf)
    return 0.5 * (1.0 + np.tanh(-x * 0.5))
