import toolbox as tb
import numpy as np
from .utilities import *


def make_Delta_semicirc_tau(Gamma, D, E0, beta, nr_points, time_mesh):
    """
    Hybridization function in imaginary times for a bath with semicircular DOS.

    Returns the function on [0, beta].

    Arguments:
        Gamma -- Hybridization strength at Fermi level
        D -- half bandwidth
        E0 -- center of semicircle and Fermi level
        beta -- inverse temperature
        nr_points -- number of real frequencies for integration
        time_mesh -- Mesh of imaginary times on which values are returned

    Returns:
        1D array of same size as `time_mesh`
    """

    omegas = np.linspace(-D, D, nr_points)

    dos = np.sqrt(1.0 - (omegas / D) ** 2)
    dos *= Gamma / np.pi
    omegas += E0

    delta = gf_tau_from_dos(time_mesh.values(), beta, omegas, dos)

    return delta


def make_Delta_semicirc(Gamma, D, beta, Ef, time_mesh):

    big_freq_mesh = time_mesh.adjoint()

    assert big_freq_mesh.xmax >= 10 * D
    assert big_freq_mesh.delta <= 0.1 * D
    assert big_freq_mesh.delta <= 0.1 / beta

    ww = big_freq_mesh.values()
    dos = np.zeros(len(ww), dtype=float)
    for k, w in enumerate(ww):
        if np.abs(w) <= D:
            dos[k] = np.sqrt(D**2 - (w) ** 2) / D**2  # norm = pi/2

    less = 2j * dos * tb.fermi(ww, Ef, beta) * D * Gamma
    grea = 2j * dos * (tb.fermi(ww, Ef, beta) - 1.0) * D * Gamma

    _, delta_less = inv_fourier_transform(big_freq_mesh, less)
    _, delta_grea = inv_fourier_transform(big_freq_mesh, grea)

    return delta_less, delta_grea


def make_Delta_lorentzian(Gamma, D, beta, Ef, time_mesh, W=None, eps=None):

    # wmax = 10 * D
    # dw = min(0.1 * D, 0.1 / beta)
    # n = max(int(2 * wmax / dw), 10001)
    # n = max(n, int(time_mesh.xmax * wmax))
    # big_freq_mesh = Mesh(wmax, n if n % 2 == 1 else n + 1)

    big_freq_mesh = time_mesh.adjoint()

    assert big_freq_mesh.delta <= 0.1 * D
    assert big_freq_mesh.delta <= 0.1 / beta

    ww = big_freq_mesh.values()
    dos = D / ((ww) ** 2 + D**2) / np.pi  # norm = 1

    if W is not None:
        if eps is None:
            eps = W / 100.0
        assert big_freq_mesh.xmax > W + eps / 2.0
        window = planck_taper_window(big_freq_mesh, W, eps)
        dos *= window
        dos /= np.trapz(dos, dx=big_freq_mesh.delta)

    less = 2j * dos * tb.fermi(ww, Ef, beta) * np.pi * D * Gamma
    grea = 2j * dos * (tb.fermi(ww, Ef, beta) - 1.0) * np.pi * D * Gamma

    _, delta_less = inv_fourier_transform(big_freq_mesh, less)
    _, delta_grea = inv_fourier_transform(big_freq_mesh, grea)

    return delta_less, delta_grea
