import numpy as np
from nca.hybridizations import *
from scipy import integrate

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
