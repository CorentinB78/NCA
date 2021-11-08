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
    
    def values(self):
        return np.linspace(self.xmin, self.xmax, self.nr_samples)

    def adjoint(self):
        adj_values = 2 * np.pi * fft.fftshift(fft.fftfreq(self.nr_samples, self.delta))
        return Mesh(adj_values[-1], len(adj_values))

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