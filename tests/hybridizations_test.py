import unittest
from matplotlib import pyplot as plt
import numpy as np
from numpy import testing
from scipy import integrate

from nca import Mesh, fourier_transform
from nca.hybridizations.precision import make_Delta_semicirc
from nca.hybridizations import *


def single_point_FT(f, a, b, w0):
    res_cos = integrate.quad(f, a, b, weight='cos', wvar=w0, limit=100, complex_func=True)
    res_sin = integrate.quad(f, a, b, weight='sin', wvar=w0, limit=100, complex_func=True)
    return res_cos[0] + 1j * res_sin[0]


def test_gaussian_dos_test():
    dos = make_gaussian_dos(2.4)
    assert abs(integrate.quad(dos, -np.inf, np.inf)[0] - 1.0) < 1e-5


def test_lorentzian_dos_test():
    dos = make_lorentzian_dos(2.4)
    assert abs(integrate.quad(dos, -np.inf, np.inf)[0] - 1.0) < 1e-5


def test_semicircular_dos_test():
    ww = np.linspace(-10., 10., 10000)
    dos = make_semicircular_dos(2.4)

    assert abs(np.trapz(x=ww, y=dos(ww)) - 1.0) < 1e-5


def test_precision_semi_circ():
    t_mesh = Mesh(5000.0, 100000)
    Gamma = 3.0
    D = 2.0
    beta = 5.0
    Ef = 0.2
    delta_less, delta_grea = make_Delta_semicirc(Gamma, D, beta, Ef, t_mesh)

    w_mesh, delta_grea_w = fourier_transform(t_mesh, delta_grea)
    w_mesh, delta_less_w = fourier_transform(t_mesh, delta_less)

    idx0 = len(w_mesh) // 2
    assert np.abs(delta_grea_w[idx0] - delta_less_w[idx0] + 2.0j * Gamma) < 1e-5

    def A_ref(w):
        if np.abs(w) >= D:
            return 0.0
        return Gamma / (np.pi * D) * np.sqrt(D**2 - w**2)

    A_ref = np.vectorize(A_ref)

    delta_grea_w_ref = (
        -2j * np.pi * A_ref(w_mesh.values()) * fermi(w_mesh.values(), Ef, -beta)
    )
    delta_less_w_ref = (
        2j * np.pi * A_ref(w_mesh.values()) * fermi(w_mesh.values(), Ef, beta)
    )

    testing.assert_allclose(delta_grea_w, delta_grea_w_ref, atol=1e-1)
    testing.assert_allclose(delta_less_w, delta_less_w_ref, atol=1e-1)


def test_lorentzian():
    w_mesh = Mesh(1000.0, 1000000)
    t_mesh = w_mesh.adjoint()
    Gamma = 3.0
    D = 2.0
    beta = 5.0
    Ef = 0.2
    dos = make_lorentzian_dos(D) # centered on 0
    delta_grea, delta_less = make_hyb_times(dos, beta, Ef, Gamma, t_mesh)

    def delta_grea_w_ref(w):
        return -2j * Gamma * (Ef**2 + D**2) / (w**2 + D**2) * fermi(w, Ef, -beta)

    def delta_less_w_ref(w):
        return 2j * Gamma * (Ef**2 + D**2) / (w**2 + D**2) * fermi(w, Ef, beta)

    w0 = 1.0
    res_g = single_point_FT(delta_grea, -300. / D, 300. / D, w0)
    assert np.abs(res_g - delta_grea_w_ref(w0)) < 1e-2
    
    res_l = single_point_FT(delta_less, -300. / D, 300. / D, w0)
    assert np.abs(res_l - delta_less_w_ref(w0)) < 1e-2
    print(res_g, res_l)


def test_gaussian():
    w_mesh = Mesh(30.0, 100000)
    t_mesh = w_mesh.adjoint()
    print(t_mesh)
    Gamma = 3.0
    D = 2.0
    beta = 5.0
    Ef = 0.2
    dos = make_gaussian_dos(D)
    delta_grea, delta_less = make_hyb_times(dos, beta, Ef, Gamma, t_mesh)

    def delta_grea_w_ref(w):
        return -2j * Gamma * np.exp((Ef**2 - w**2) / (2.0 * D**2)) * fermi(w, Ef, -beta)

    def delta_less_w_ref(w):
        return 2j * Gamma * np.exp((Ef**2 - w**2) / (2.0 * D**2)) * fermi(w, Ef, beta)

    w0 = 1.5
    res_g = single_point_FT(delta_grea, -30. / D, 30. / D, w0)
    assert np.abs(res_g - delta_grea_w_ref(w0)) < 1e-3
    
    res_l = single_point_FT(delta_less, -30. / D, 30. / D, w0)
    assert np.abs(res_l - delta_less_w_ref(w0)) < 1e-3


def test_gf_tau_from_dos():
    beta = 2.0
    sigma = 1.0
    w = np.linspace(-10, 9.5, 50)
    norm = 1.0 / (1.0 + np.exp((sigma * beta) ** 2 / 4.0))
    dos = (
        norm
        / (sigma * np.sqrt(np.pi))
        * (np.exp(-((w / sigma) ** 2)) + np.exp(-((w / sigma) ** 2) - beta * w))
    )

    taus = np.array([0.0, 0.7, 1.3, 2.0])
    gf_tau_ref = -norm * np.exp((sigma * taus) ** 2 / 4.0)

    gf_tau = gf_tau_from_dos(taus, beta, w, dos)

    testing.assert_allclose(gf_tau, gf_tau_ref, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
