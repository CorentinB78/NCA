import numpy as np
from numpy import fft
from scipy import integrate
import toolbox as tb


class Mesh:
    def __init__(self, xmax, nr_samples, pt_on_value=False):
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
        )

        out.pt_on_value = self.pt_on_value
        out.idx_pt_on_value = self.idx_pt_on_value
        out.pt_on_value_adj = not self.pt_on_value_adj

        return out

    def __len__(self):
        return self.nr_samples


def fourier_transform(mesh, f, axis=-1):
    adj_mesh = mesh.adjoint()
    f = np.swapaxes(f, -1, axis)
    g = fft.fftshift(fft.fft(f, axis=-1), axes=-1)[..., ::-1]
    g *= mesh.delta * np.exp(1j * adj_mesh.values() * mesh.xmin)
    return adj_mesh, np.swapaxes(g, -1, axis)


def inv_fourier_transform(mesh, f, axis=-1):
    adj_mesh = mesh.adjoint()
    f = np.swapaxes(f, -1, axis)
    g = fft.fftshift(fft.fft(f, axis=-1), axes=-1)
    g *= mesh.delta * np.exp(-1j * adj_mesh.values() * mesh.xmin)
    g /= 2 * np.pi
    return adj_mesh, np.swapaxes(g, -1, axis)


def planck_taper_window(mesh, W, eps):
    Wp = W + eps / 2.0
    Wm = W - eps / 2.0
    assert Wp < mesh.xmax
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
