import toolbox as tb
import numpy as np
from .function_tools import *
from matplotlib import pyplot as plt


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


def make_Delta_semicirc(Gamma, D, beta, Ef, time_mesh=None):
    """
    Lesser and Greater hybridization funcitons of bath with semicircular DOS.

    Arguments:
        Gamma -- coupling at zero energy
        D -- half bandwidth
        beta -- inverse temperature
        Ef -- Fermi level
        time_mesh -- mesh on which to return data

    Returns:
        delta_less, delta_grea
    """

    dw = D
    if np.abs(Ef) < D + 4.0 / beta:
        dw = min(dw, 1.0 / beta)
    dw = dw / 100.0
    wmax = 100.0 * D
    N = 2 * round(wmax / dw) + 1

    if N >= 1e7:
        print("/!\ [Semicirc] Large number of samples required")
        r = N * 1e-7
        dw *= np.sqrt(r)
        wmax /= np.sqrt(r)
        N = 2 * round(wmax / dw) + 1

    freq_mesh = Mesh(wmax, N, pt_on_value=D - wmax / (N - 1), adjust_nr_samples=False)

    ww = freq_mesh.values()
    dos = np.zeros(len(ww), dtype=float)
    for k, w in enumerate(ww):
        if np.abs(w) <= D:
            dos[k] = np.sqrt(D**2 - (w) ** 2) / D**2  # norm = pi/2

    less = 2j * dos * tb.fermi(ww, Ef, beta) * D * Gamma
    grea = -2j * dos * tb.one_minus_fermi(ww, Ef, beta) * D * Gamma

    time_mesh_comp, delta_less = inv_fourier_transform(freq_mesh, less)
    time_mesh_comp, delta_grea = inv_fourier_transform(freq_mesh, grea)

    if time_mesh is None:
        return time_mesh_comp, delta_less, delta_grea

    delta_grea = checked_interp(time_mesh, time_mesh_comp, delta_grea)
    delta_less = checked_interp(time_mesh, time_mesh_comp, delta_less)

    return delta_less, delta_grea


def make_Delta_lorentzian(Gamma, D, beta, Ef, time_mesh=None):
    """
    Lesser and Greater hybridization funcitons of bath with lorentzian DOS.

    Arguments:
        Gamma -- coupling at zero energy
        D -- half bandwidth
        beta -- inverse temperature
        Ef -- Fermi level
        time_mesh -- mesh on which to return data

    Returns:
        delta_less, delta_grea
    """

    dw = D
    if np.abs(Ef) < 10 * D:
        dw = min(dw, 1.0 / beta)
    dw = dw / 100.0
    wmax = D * 100.0
    N = 2 * round(wmax / dw) + 1

    if N >= 1e7:
        print("/!\ [Lorentzian] Large number of samples required")
        r = N * 1e-7
        dw *= np.sqrt(r)
        wmax /= np.sqrt(r)
        N = 2 * round(wmax / dw) + 1

    if np.abs(Ef) < 10 * D:
        freq_mesh = Mesh(wmax, N, pt_on_value=Ef)
    else:
        freq_mesh = Mesh(wmax, N)

    ww = freq_mesh.values()
    dos = D / ((ww) ** 2 + D**2) / np.pi  # norm = 1

    less = 2j * dos * tb.fermi(ww, Ef, beta) * np.pi * D * Gamma
    grea = -2j * dos * tb.one_minus_fermi(ww, Ef, beta) * np.pi * D * Gamma

    time_mesh_comp, delta_less = inv_fourier_transform(freq_mesh, less)
    time_mesh_comp, delta_grea = inv_fourier_transform(freq_mesh, grea)

    if time_mesh is None:
        return time_mesh_comp, delta_less, delta_grea

    delta_grea = checked_interp(time_mesh, time_mesh_comp, delta_grea)
    delta_less = checked_interp(time_mesh, time_mesh_comp, delta_less)

    return delta_less, delta_grea
