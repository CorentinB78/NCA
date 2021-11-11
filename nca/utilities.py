import numpy as np
from numpy import fft


class Mesh:
    def __init__(self, xmax, nr_samples):
        if nr_samples % 2 != 1:
            raise ValueError
        self.xmin = -xmax
        self.xmax = xmax
        self.nr_samples = nr_samples
        self.delta = (xmax - self.xmin) / (nr_samples - 1)
        self.data = None

    def values(self):
        if self.data is None:
            self.data = np.linspace(self.xmin, self.xmax, self.nr_samples)
        return self.data

    def adjoint(self):
        return Mesh(
            2 * np.pi * (self.nr_samples - 1) / (2 * self.delta * self.nr_samples),
            self.nr_samples,
        )

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
