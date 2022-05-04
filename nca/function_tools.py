import numpy as np
from numpy import fft
from scipy import integrate, interpolate
import toolbox as tb
from matplotlib import pyplot as plt
from .utilities import print_warning_large_error


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2 ** ((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2 ** (len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def next_odd_regular(target):
    """
    Find the next odd reglar number greater than or equal to target.
    THese are composites of prime factors 3 and 5 only.
    """
    N = _next_regular(target)
    while (N % 2) == 0:
        N = _next_regular(N + 1)

    return N


class Mesh:
    def __init__(self, xmax, nr_samples, pt_on_value=False, adjust_nr_samples=True):
        if adjust_nr_samples:
            nr_samples = next_odd_regular(nr_samples)

        if nr_samples % 2 != 1:
            raise ValueError("nr of samples must be odd.")

        if pt_on_value > 0.0:
            delta = 2 * xmax / (nr_samples - 1)
            x_next = delta * (pt_on_value // delta + 1)
            xmax *= pt_on_value / x_next
            if pt_on_value > xmax:
                raise ValueError("`pt_on_value` cannot be reached. Increase xmax.")

        self.xmin = -xmax
        self.xmax = xmax
        self.nr_samples = nr_samples
        self.delta = (xmax - self.xmin) / (nr_samples - 1)
        self.data = None

        self.pt_on_value = pt_on_value
        # TODO: test pt_on_value
        self.idx_pt_on_value = int((pt_on_value - self.xmin) / self.delta)
        self.pt_on_value_adj = False

    def values(self):
        if self.data is None:
            self.data = np.linspace(self.xmin, self.xmax, self.nr_samples)
            self.data[self.nr_samples // 2] = 0.0  # enforce exact zero
        return self.data

    def adjoint(self):
        out = Mesh(
            2 * np.pi * (self.nr_samples - 1) / (2 * self.delta * self.nr_samples),
            self.nr_samples,
            adjust_nr_samples=False,
        )

        out.pt_on_value = self.pt_on_value
        out.idx_pt_on_value = self.idx_pt_on_value
        out.pt_on_value_adj = not self.pt_on_value_adj

        return out

    def __len__(self):
        return self.nr_samples

    def __eq__(self, other):
        return (self.delta == other.delta) and (self.nr_samples == other.nr_samples)

    def __array__(self):
        return self.values()


def interp(mesh_a, mesh_b, func_b, kind="linear", allow=True):
    if mesh_a is mesh_b:
        return func_b
    if not allow:
        raise RuntimeError
    return interpolate.interp1d(
        mesh_b.values(),
        func_b,
        kind=kind,
        assume_sorted=True,
        copy=False,
        bounds_error=False,
        fill_value=0.0,
    )(mesh_a.values())


def checked_interp(mesh_a, mesh_b, func_b, kind="cubic", tol=1e-3):
    if mesh_a is mesh_b:
        return func_b
    mesh_a_half = Mesh(
        mesh_a.xmax, 2 * (mesh_a.nr_samples // 4) + 1, adjust_nr_samples=False
    )

    vals1 = interp(mesh_a, mesh_b, func_b, kind=kind, allow=True)
    vals2 = interp(mesh_a_half, mesh_b, func_b, kind=kind, allow=True)
    check = interp(mesh_a, mesh_a_half, vals2, kind="linear", allow=True)

    err = np.trapz(np.abs(check - vals1), dx=mesh_a.delta)

    print_warning_large_error(
        f"Low number of samples for this interpolation. err={err}",
        err,
        tolw=tol,
        tole=1e-1,
    )

    return vals1


def product_functions(mesh_a, func_a, mesh_b, func_b):
    """
    Interpolate on the smaller mesh
    """
    if mesh_a.xmax > mesh_b.xmax:
        return product_functions(mesh_b, func_b, mesh_a, func_a)

    func_b = interp(mesh_a, mesh_b, func_b)
    return mesh_a, func_a * func_b


def sum_functions(mesh_a, func_a, mesh_b, func_b):
    """
    Interpolate on the larger mesh. Is this the good choice?
    """
    if mesh_a is None:
        return mesh_b, func_b

    if mesh_a.xmax < mesh_b.xmax:
        return sum_functions(mesh_b, func_b, mesh_a, func_a)

    func_b = interp(mesh_a, mesh_b, func_b)
    return mesh_a, func_a + func_b


def fourier_transform(mesh, f, axis=-1):
    adj_mesh = mesh.adjoint()
    f = np.swapaxes(f, -1, axis)
    g = fft.fftshift(fft.fft(f, axis=-1), axes=-1)[..., ::-1]
    g *= mesh.delta * np.exp(adj_mesh.values() * (1j * mesh.xmin))
    return adj_mesh, np.swapaxes(g, -1, axis)


def inv_fourier_transform(mesh, f, axis=-1):
    adj_mesh = mesh.adjoint()
    f = np.swapaxes(f, -1, axis)
    g = fft.fftshift(fft.fft(f, axis=-1), axes=-1)
    g *= (mesh.delta / (2 * np.pi)) * np.exp(adj_mesh.values() * (-1j * mesh.xmin))
    return adj_mesh, np.swapaxes(g, -1, axis)


def planck_taper_window(mesh, W, eps):
    Wp = W + eps / 2.0
    Wm = W - eps / 2.0
    assert Wm > 0.0
    out = np.empty(len(mesh))
    for k, x in enumerate(mesh.values()):
        if np.abs(x) >= Wp:
            out[k] = 0
        elif np.abs(x) > Wm:
            out[k] = 0.5 * (
                1.0
                - np.tanh((Wp - Wm) / (Wp - np.abs(x)) - (Wp - Wm) / (np.abs(x) - Wm))
            )
        else:
            out[k] = 1.0
    return out


def gf_tau_from_dos(taus, beta, omegas, dos):
    # TODO: test and fix issues of overflow for large omegas or large beta (see toolbox)
    delta = omegas[1] - omegas[0]

    f = np.empty((len(taus), len(dos)), dtype=float)

    for k, tau in enumerate(taus):
        if tau < 0:
            f[k, :] = 0.0
        elif tau < beta / 2.0:
            f[k, :] = dos * tb.fermi(-omegas, 0.0, beta) * np.exp(-omegas * tau)
        elif tau <= beta:
            f[k, :] = dos * tb.fermi(omegas, 0.0, beta) * np.exp(omegas * (beta - tau))
        else:
            f[k, :] = 0.0

    ### TODO: optimize this: avoid useless integrations
    return -integrate.simpson(f, axis=1, dx=delta)
