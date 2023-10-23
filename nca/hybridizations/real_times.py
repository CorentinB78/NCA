import numpy as np
from scipy import interpolate
from ..function_tools import inv_fourier_transform
from .utils import fermi


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
    """
    Produce hybridization functions (lesser and greater) in the time domain from a density of state (DOS).

    The DOS is given as a callable and evaluated on a frequency grid for FFT.
    The result is interpolated to produce callables in the time domain for the greater and lesser hybridization functions.

    Arguments:
    * dos: callable in frequency domain
    * beta: inverse temperature
    * Ef: Fermi energy
    * hyb_at_fermi_lvl: strength of hybridization spectrum at Fermi level
    * time_mesh (Mesh): a time mesh on which FFT is performed

    Return:
    * greater: callable in time domain
    * lesser: callable in time domain
    """
    freq_mesh = time_mesh.adjoint()
    grea_w, less_w = make_hyb_freqs(dos, beta, Ef, hyb_at_fermi_lvl)
    grea_w = grea_w(freq_mesh.values())
    less_w = less_w(freq_mesh.values())

    time_mesh, grea_t = inv_fourier_transform(freq_mesh, grea_w)
    _, less_t = inv_fourier_transform(freq_mesh, less_w)
    grea_t = interpolate.CubicSpline(time_mesh.values(), grea_t, extrapolate=False)
    less_t = interpolate.CubicSpline(time_mesh.values(), less_t, extrapolate=False)

    return grea_t, less_t

