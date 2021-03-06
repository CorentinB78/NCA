import numpy as np
from .function_tools import *
from .utilities import print_warning_large_error

# TODO: swap outputs to respect default order: grea, less


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


def gf_tau_from_dos(taus, beta, omegas, dos):
    """
    Compute imaginary time Green function from a real frequency density of states.

    Arguments:
        taus -- 1D array, imaginary times
        beta -- float, inverse temperature
        omegas -- 1D array, real frequencies (increasing)
        dos -- 1D array, density of states at `omegas`

    Returns:
        1D array, values of imaginary time GF at `taus`
    """
    delta = omegas[1] - omegas[0]

    f = np.empty((len(taus), len(dos)), dtype=float)

    for k, tau in enumerate(taus):
        if tau < 0:
            f[k, :] = 0.0
        elif tau <= beta:

            mask = omegas >= 0.0
            f[k, :][mask] = (
                dos[mask]
                * fermi(-omegas[mask], 0.0, beta)
                * np.exp(-omegas[mask] * tau)
            )
            f[k, :][~mask] = (
                dos[~mask]
                * fermi(omegas[~mask], 0.0, beta)
                * np.exp(omegas[~mask] * (beta - tau))
            )

        else:
            f[k, :] = 0.0

    ### TODO: optimize this: avoid useless integrations
    return -integrate.simpson(f, axis=1, dx=delta)


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


def make_Delta_semicirc_w(Gamma, D, beta, Ef):
    """
    Frequency domain Lesser and Greater hybridization functions of bath with semicircular DOS.

    Arguments:
        Gamma -- coupling at zero energy
        D -- half bandwidth
        beta -- inverse temperature
        Ef -- Fermi level

    Returns:
        freq_mesh, delta_grea, delta_less
    """

    dw = D
    if np.abs(Ef) < D + 4.0 / beta:
        dw = min(dw, 1.0 / beta)
    dw = dw / 100.0
    wmax = 100.0 * D
    N = 2 * round(wmax / dw) + 1

    print_warning_large_error(
        f"[Semicirc] Large number of samples required N={N}", N, tolw=1e7, tole=1e8
    )
    if N >= 1e7:
        r = N * 1e-7
        dw *= np.sqrt(r)
        wmax /= np.sqrt(r)
        N = 2 * round(wmax / dw) + 1
        print(f"[Semicirc] Reduced to {N}.")

    freq_mesh = Mesh(wmax, N, pt_on_value=D - wmax / (N - 1), adjust_nr_samples=False)

    ww = freq_mesh.values()
    dos = np.zeros(len(ww), dtype=float)
    for k, w in enumerate(ww):
        if np.abs(w) <= D:
            dos[k] = np.sqrt(D**2 - (w) ** 2) / D**2  # norm = pi/2

    less = 2j * dos * fermi(ww, Ef, beta) * D * Gamma
    grea = -2j * dos * fermi(ww, Ef, -beta) * D * Gamma

    return freq_mesh, grea, less


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
    freq_mesh, grea, less = make_Delta_semicirc_w(Gamma, D, beta, Ef)

    time_mesh_comp, delta_less = inv_fourier_transform(freq_mesh, less)
    time_mesh_comp, delta_grea = inv_fourier_transform(freq_mesh, grea)

    if time_mesh is None:
        return time_mesh_comp, delta_less, delta_grea

    delta_grea = checked_interp(time_mesh, time_mesh_comp, delta_grea)
    delta_less = checked_interp(time_mesh, time_mesh_comp, delta_less)

    return delta_less, delta_grea


def make_Delta_lorentzian_w(Gamma, D, beta, Ef):
    """
    Frequency domain Lesser and Greater hybridization funcitons of bath with lorentzian DOS.

    Arguments:
        Gamma -- coupling at zero energy
        D -- half bandwidth
        beta -- inverse temperature
        Ef -- Fermi level

    Returns:
        freq_mesh, delta_grea, delta_less
    """

    dw = D
    if np.abs(Ef) < 10 * D:
        dw = min(dw, 1.0 / beta)
    dw = dw / 100.0
    wmax = D * 100.0
    N = 2 * round(wmax / dw) + 1

    print_warning_large_error(
        f"[Lorentzian] Large number of samples required N={N}", N, tolw=1e7, tole=1e8
    )
    if N >= 1e7:
        r = N * 1e-7
        dw *= np.sqrt(r)
        wmax /= np.sqrt(r)
        N = 2 * round(wmax / dw) + 1
        print(f"[Lorentzian] Reduced to {N}.")

    if np.abs(Ef) < 10 * D:
        freq_mesh = Mesh(wmax, N, pt_on_value=Ef)
    else:
        freq_mesh = Mesh(wmax, N)

    ww = freq_mesh.values()
    dos = D / ((ww) ** 2 + D**2) / np.pi  # norm = 1

    less = 2j * dos * fermi(ww, Ef, beta) * np.pi * D * Gamma
    grea = -2j * dos * fermi(ww, Ef, -beta) * np.pi * D * Gamma

    return freq_mesh, grea, less


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
    freq_mesh, grea, less = make_Delta_lorentzian_w(Gamma, D, beta, Ef)

    time_mesh_comp, delta_less = inv_fourier_transform(freq_mesh, less)
    time_mesh_comp, delta_grea = inv_fourier_transform(freq_mesh, grea)

    if time_mesh is None:
        return time_mesh_comp, delta_less, delta_grea

    delta_grea = checked_interp(time_mesh, time_mesh_comp, delta_grea)
    delta_less = checked_interp(time_mesh, time_mesh_comp, delta_less)

    return delta_less, delta_grea


def make_Delta_gaussian_w(Gamma, D, beta, Ef):
    """
    Frequency domain Lesser and Greater hybridization functions of bath with gaussian DOS.

    Arguments:
        Gamma -- coupling at zero energy
        D -- half bandwidth
        beta -- inverse temperature
        Ef -- Fermi level

    Returns:
        freq_mesh, delta_grea, delta_less
    """

    dw = D
    if np.abs(Ef) < 10 * D:
        dw = min(dw, 1.0 / beta)
    dw = dw / 100.0
    wmax = D * 10.0
    N = 2 * round(wmax / dw) + 1

    print_warning_large_error(
        f"[Gaussian] Large number of samples required N={N}", N, tolw=1e7, tole=1e8
    )
    if N >= 1e7:
        r = N * 1e-7
        dw *= np.sqrt(r)
        wmax /= np.sqrt(r)
        N = 2 * round(wmax / dw) + 1
        print(f"[Lorentzian] Reduced to {N}.")

    if np.abs(Ef) < 10 * D:
        freq_mesh = Mesh(wmax, N, pt_on_value=Ef)
    else:
        freq_mesh = Mesh(wmax, N)

    ww = freq_mesh.values()
    dos = np.exp(-((ww / D) ** 2) / 2.0) / D  # norm = sqrt(2 pi)

    less = 2j * dos * fermi(ww, Ef, beta) * D * Gamma
    grea = -2j * dos * fermi(ww, Ef, -beta) * D * Gamma

    return freq_mesh, grea, less


def make_Delta_gaussian(Gamma, D, beta, Ef, time_mesh=None):
    """
    Lesser and Greater hybridization functions of bath with gaussian DOS.

    Arguments:
        Gamma -- coupling at zero energy
        D -- half bandwidth
        beta -- inverse temperature
        Ef -- Fermi level
        time_mesh -- mesh on which to return data

    Returns:
        delta_less, delta_grea
    """
    freq_mesh, grea, less = make_Delta_gaussian_w(Gamma, D, beta, Ef)

    time_mesh_comp, delta_less = inv_fourier_transform(freq_mesh, less)
    time_mesh_comp, delta_grea = inv_fourier_transform(freq_mesh, grea)

    if time_mesh is None:
        return time_mesh_comp, delta_less, delta_grea

    delta_grea = checked_interp(time_mesh, time_mesh_comp, delta_grea)
    delta_less = checked_interp(time_mesh, time_mesh_comp, delta_less)

    return delta_less, delta_grea
