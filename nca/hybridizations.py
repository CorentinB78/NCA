import numpy as np
from .function_tools import inv_fourier_transform, checked_interp, Mesh
from .utilities import print_warning_large_error, symmetrize
from scipy.special import roots_jacobi, roots_legendre
from scipy import integrate, interpolate


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


def make_Delta_semicirc_w(Gamma, D, beta, Ef, freq_mesh):
    """
    Frequency domain Lesser and Greater hybridization functions of bath with semicircular DOS.

    Arguments:
        Gamma -- coupling at zero energy
        D -- half bandwidth
        beta -- inverse temperature
        Ef -- Fermi level
        freq_mesh -- frequency mesh

    Returns:
        delta_grea, delta_less
    """
    ww = freq_mesh.values()
    dos = np.zeros(len(ww), dtype=float)
    for k, w in enumerate(ww):
        if np.abs(w) <= D:
            dos[k] = np.sqrt(D**2 - (w) ** 2) / D**2  # norm = pi/2

    less = 2j * dos * fermi(ww, Ef, beta) * D * Gamma
    grea = -2j * dos * fermi(ww, Ef, -beta) * D * Gamma

    return grea, less


def fourier_transform_semicirc(f, halfwidth, dt, nr_times, order):
    """
    Compute Fourier transform of function weighted by a semicircular density:
    \int_{-D}^{D} dw f(w) \sqrt{D^2 - w^2} / D^2 e^{iwt}
    for t > 0

    Using Gauss-Jacobi quadrature.

    Arguments:
        f -- function
        halfwidth -- float, half width of semicircular density
        dt -- float, time spacing
        nr_times -- int, number of times
        order -- int, order of quadrature

    Returns:
        1D array, Fourier transform at given time points
    """
    out = np.empty(nr_times, dtype=complex)
    ww, weights = roots_jacobi(order, 0.5, 0.5)
    weights *= f(halfwidth * ww)
    expo = np.exp(1j * halfwidth * ww * dt)
    fourier_factor = 1.0

    for i in range(nr_times):
        out[i] = np.sum(weights * fourier_factor)
        fourier_factor *= expo

    return out


def fourier_transform_semicirc_auto_old(
    f, halfwidth, dt, nr_times, tol=1e-10, max_order=100000, verbose=False
):
    """
    Compute Fourier transform of function weighted by a semicircular density:
    \int_{-D}^{D} dw f(w) \sqrt{D^2 - w^2} / D^2 e^{iwt}
    for t > 0

    Using Gauss-Jacobi quadrature with automatic choice of order to respect tolerance.

    Arguments:
        f -- function
        halfwidth -- float, half width of semicircular density
        dt -- float, time spacing
        nr_times -- int, number of times

    Keyword Arguments:
        tol -- float, tolerance (default: {1e-10})
        max_order -- maximum quadrature order (default: {100000})
        verbose -- of True, print convergence messages (default: {False})

    Returns:
        1D array, Fourier transform at given time points
    """
    n = int(halfwidth * dt * nr_times)
    n = max(n, 100)
    if verbose:
        print(f"{n}: \t ---")
    ft = fourier_transform_semicirc(f, halfwidth, dt, nr_times, n)

    err = np.inf
    while True:
        if 2 * n > max_order:
            print("/!\ [semicircular FT] max quadrature order reached")
            break

        n *= 2
        ft_new = fourier_transform_semicirc(f, halfwidth, dt, nr_times, n)
        err = np.max(np.abs(ft - ft_new))
        if verbose:
            print(f"{n}: \t {err}")
        ft = ft_new

        if err < tol:
            break

    return ft


def fourier_transform_semicirc_auto(
    f, halfwidth, dt, nr_times, tol=1e-10, order=1000, verbose=False
):
    """
    Computes the Fourier transform of w -> f(w) sqrt(D^2 - w^2) / D^2 defined on finite support [-D, D].
    TODO: check that

    Exact formula:
    $$
    f(t) = \frac{1}{D^2} \int_{-D}^{D} d\omega f(\omega) \sqrt{D^2 - \omega^2} e^{i\omega t}
    $$

    """
    # TODO: test it!
    def left(b):
        y, weights = roots_jacobi(order, 0.0, 0.5)
        slope = (b + 1.0) / 2.0
        x = slope * (y + 1) - 1.0
        weights *= f(halfwidth * x) * np.sqrt(1.0 - x)

        expo = np.exp(1j * halfwidth * x * dt)
        fourier_factor = 1.0

        out = np.empty(nr_times, dtype=complex)
        for i in range(nr_times):
            out[i] = np.sum(weights * fourier_factor)
            fourier_factor *= expo

        return slope**1.5 * out

    def right(a):
        y, weights = roots_jacobi(order, 0.5, 0.0)
        slope = (1.0 - a) / 2.0
        x = slope * (y - 1) + 1.0
        weights *= f(halfwidth * x) * np.sqrt(1.0 + x)

        expo = np.exp(1j * halfwidth * x * dt)
        fourier_factor = 1.0

        out = np.empty(nr_times, dtype=complex)
        for i in range(nr_times):
            out[i] = np.sum(weights * fourier_factor)
            fourier_factor *= expo

        return slope**1.5 * out

    def center(a, b):
        y, weights = roots_legendre(order)
        slope = (b - a) / 2.0
        x = slope * (y - 1.0) + b
        weights *= f(halfwidth * x) * np.sqrt(1.0 - x**2)

        expo = np.exp(1j * halfwidth * x * dt)
        fourier_factor = 1.0

        out = np.empty(nr_times, dtype=complex)
        for i in range(nr_times):
            out[i] = np.sum(weights * fourier_factor)
            fourier_factor *= expo

        return slope * out

    def left_rec(b, vals):
        a = (b - 1.0) / 2.0
        vals_l = left(a)
        vals_r = center(a, b)
        vals_new = vals_l + vals_r

        if np.max(np.abs(vals - vals_new)) > tol:
            return left_rec(a, vals_l) + center_rec(a, b, vals_r)
        else:
            return vals_new

    def right_rec(a, vals):
        b = (1.0 + a) / 2.0
        vals_l = right(b)
        vals_r = center(a, b)
        vals_new = vals_l + vals_r

        if np.max(np.abs(vals - vals_new)) > tol:
            return right_rec(b, vals_l) + center_rec(a, b, vals_r)
        else:
            return vals_new

    def center_rec(a, b, vals):
        m = (a + b) / 2.0
        vals_l = center(a, m)
        vals_r = center(m, b)
        vals_new = vals_l + vals_r

        if np.max(np.abs(vals - vals_new)) > tol:
            return center_rec(a, m, vals_l) + center_rec(m, b, vals_r)
        else:
            return vals_new

    nr_panels = int(dt * nr_times / (halfwidth * order))
    nr_panels = max(nr_panels, 2)
    if verbose:
        print("Initial nr of panels:", nr_panels)
    xi = np.linspace(-1.0, 1.0, nr_panels + 1)

    vals_l = left(xi[1])
    vals_r = right(xi[-2])
    out = left_rec(xi[1], vals_l) + right_rec(xi[-2], vals_r)

    for i in range(1, nr_panels - 1):
        vals = center(xi[i], xi[i + 1])
        out += center_rec(xi[i], xi[i + 1], vals)

    return out


def semicirc_lesser_times(mu, beta, halfwidth, dt, nr_times, tol=1e-10, verbose=True):
    """
    Computes g^<(t) of bath with semicircular density of states.

    Arguments:
        mu -- float, chem. potential
        beta -- float, inverse temperature
        halfwidth -- float, half width of semicircular density of states
        dt -- float, time spacing
        nr_times -- int, number of times

    Keyword Arguments:
        tol -- float, tolerance (default: {1e-10})
        max_order -- maximum quadrature order (default: {100000})
        verbose -- of True, print convergence messages (default: {False})

    Returns:
        times, values -- two 1D arrays
    """

    def f(w):
        return fermi(w, mu, beta)

    gf = fourier_transform_semicirc_auto(
        f,
        halfwidth,
        dt,
        nr_times,
        tol=tol * np.pi / 2.0,
        verbose=verbose,
    )
    gf *= 2.0j / np.pi
    gf = gf[::-1]

    times = np.arange(-nr_times + 1, 1) * dt

    return symmetrize(times, gf, 0.0, lambda y: -np.conj(y))


def semicirc_greater_times(mu, beta, halfwidth, dt, nr_times, tol=1e-10, verbose=True):
    """
    Computes g^>(t) of bath with semicircular density of states.

    Arguments:
        mu -- float, chem. potential
        beta -- float, inverse temperature
        halfwidth -- float, half width of semicircular density of states
        dt -- float, time spacing
        nr_times -- int, number of times

    Keyword Arguments:
        tol -- float, tolerance (default: {1e-10})
        max_order -- maximum quadrature order (default: {100000})
        verbose -- of True, print convergence messages (default: {False})

    Returns:
        times, values -- two 1D arrays
    """

    def f(w):
        return fermi(w, mu, -beta)

    gf = fourier_transform_semicirc_auto(
        f,
        halfwidth,
        dt,
        nr_times,
        tol=tol * np.pi / 2.0,
        verbose=verbose,
    )
    gf *= -2.0j / np.pi
    gf = gf[::-1]

    times = np.arange(-nr_times + 1, 1) * dt

    return symmetrize(times, gf, 0.0, lambda y: -np.conj(y))


def make_Delta_semicirc(Gamma, D, beta, Ef, time_mesh):
    """
    Lesser and Greater hybridization functions of bath with semicircular DOS.

    Arguments:
        Gamma -- coupling at zero energy
        D -- half bandwidth
        beta -- inverse temperature
        Ef -- Fermi level
        time_mesh -- mesh on which to return data

    Returns:
        delta_less, delta_grea
    """
    dt = time_mesh.delta
    N = len(time_mesh) // 2 + 1
    times, gfl = semicirc_lesser_times(Ef, beta, D, dt, N)
    times, gfg = semicirc_greater_times(Ef, beta, D, dt, N)

    np.testing.assert_allclose(times, time_mesh)

    gfl *= D * Gamma / 2.0
    gfg *= D * Gamma / 2.0

    return gfl, gfg


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

#######################################

def make_hyb_freqs(dos, beta, Ef, hyb_at_fermi_lvl):
    v2 = hyb_at_fermi_lvl / dos(Ef)
    def less(w):
        return 2j * v2 * dos(w) * fermi(w, Ef, beta)
    def grea(w):
        return -2j * v2 * dos(w) * fermi(w, Ef, -beta)
    return grea, less

def make_gaussian_dos(D):
    def out(w):
        return np.exp(-((w / D) ** 2) / 2.0) / (np.sqrt(2 * np.pi) * D)
    return out

def make_lorentzian_dos(D):
    def out(w):
        return D / (w**2 + D**2) / np.pi
    return out

def make_semicircular_dos(D):
    def out(w):
        if abs(w) < D:
            return 2 * np.sqrt(D**2 - w**2) / (np.pi * D**2)
        else:
            return 0.0
    return np.vectorize(out)

def make_hyb_times(dos, beta, Ef, hyb_at_fermi_lvl, time_mesh):
    freq_mesh = time_mesh.adjoint()
    grea_w, less_w = make_hyb_freqs(dos, beta, Ef, hyb_at_fermi_lvl)
    grea_w = grea_w(freq_mesh.values())
    less_w = less_w(freq_mesh.values())

    time_mesh, grea_t = inv_fourier_transform(freq_mesh, grea_w)
    _, less_t = inv_fourier_transform(freq_mesh, less_w)
    grea_t = interpolate.CubicSpline(time_mesh.values(), grea_t, extrapolate=False)
    less_t = interpolate.CubicSpline(time_mesh.values(), less_t, extrapolate=False)

    return grea_t, less_t

