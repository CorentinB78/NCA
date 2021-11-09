import toolbox as tb
import numpy as np
from .utilities import *

def make_Delta_semicirc(Gamma, D, beta, mu, time_mesh):
    
    freq_mesh = time_mesh.adjoint()
    assert(freq_mesh.xmax >= 10 * D)
    assert(freq_mesh.xmin <= 0.1 * D)
    assert(freq_mesh.xmin <= 0.1 / beta)
    
    ww = freq_mesh.values()
    dos = np.zeros(len(ww), dtype=float)
    for k, w in enumerate(ww):
        if np.abs(w) <= D:
            dos[k] = np.sqrt(D**2 - w**2) / D**2 # norm = pi/2

    less = 1j * dos * tb.fermi(ww, mu, beta) * D * Gamma
    grea = 1j * dos * (tb.fermi(ww, mu, beta) - 1.) * D * Gamma

    _, delta_less = inv_fourier_transform(freq_mesh, less)
    _, delta_grea = inv_fourier_transform(freq_mesh, grea)
    
    return delta_less, delta_grea


def make_Delta_lorentzian(Gamma, D, beta, mu, time_mesh):
    
    freq_mesh = time_mesh.adjoint()
    assert(freq_mesh.xmax >= 10 * D)
    assert(freq_mesh.xmin <= 0.1 * D)
    assert(freq_mesh.xmin <= 0.1 / beta)
    
    ww = freq_mesh.values()
    dos = D / (ww**2 + D**2) / np.pi # norm = 1

    less = 1j * dos * tb.fermi(ww, mu, beta) * np.pi * D * Gamma / 2.
    grea = 1j * dos * (tb.fermi(ww, mu, beta) - 1.) * np.pi * D * Gamma / 2.

    _, delta_less = inv_fourier_transform(freq_mesh, less)
    _, delta_grea = inv_fourier_transform(freq_mesh, grea)
    
    return delta_less, delta_grea
