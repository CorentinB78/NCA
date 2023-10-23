"""
Precision methods to make time resolved hybridization functions.
"""
import numpy as np
from ..utilities import symmetrize
from scipy.special import roots_jacobi, roots_legendre
from .utils import fermi

# TODO: swap outputs to respect default order: grea, less


def _fourier_transform_semicirc(f, halfwidth, dt, nr_times, order):
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


def _fourier_transform_semicirc_auto_old(
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
    ft = _fourier_transform_semicirc(f, halfwidth, dt, nr_times, n)

    err = np.inf
    while True:
        if 2 * n > max_order:
            print("/!\ [semicircular FT] max quadrature order reached")
            break

        n *= 2
        ft_new = _fourier_transform_semicirc(f, halfwidth, dt, nr_times, n)
        err = np.max(np.abs(ft - ft_new))
        if verbose:
            print(f"{n}: \t {err}")
        ft = ft_new

        if err < tol:
            break

    return ft


def _fourier_transform_semicirc_auto(
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

    gf = _fourier_transform_semicirc_auto(
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

    gf = _fourier_transform_semicirc_auto(
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
        delta_less, delta_grea on given time mesh
    """
    dt = time_mesh.delta
    N = len(time_mesh) // 2 + 1
    times, gfl = semicirc_lesser_times(Ef, beta, D, dt, N)
    times, gfg = semicirc_greater_times(Ef, beta, D, dt, N)

    np.testing.assert_allclose(times, time_mesh)

    gfl *= D * Gamma / 2.0
    gfg *= D * Gamma / 2.0

    return gfl, gfg