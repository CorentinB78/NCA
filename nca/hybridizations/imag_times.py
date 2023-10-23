import numpy as np
from scipy import integrate
from .utils import fermi


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

