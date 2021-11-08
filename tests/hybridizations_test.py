import unittest
from matplotlib import pyplot as plt
import numpy as np

from nca.hybridizations import *


class HybridizationTest(unittest.TestCase):

    def test_semi_circ(self):
        m = Mesh(100., 1001)
        Gamma = 3.0
        dl, dg = make_Delta_semicirc(Gamma, 2.0, 50.0, 0.2, m)

        mw, dgw = fourier_transform(m, dg)
        mw, dlw = fourier_transform(m, dl)

        tb.cpx_plot(mw.values(), dgw - dlw)
        tb.cpx_plot(mw.values(), dgw)
        tb.cpx_plot(mw.values(), -dlw)
        plt.xlim(-2., 2.)
        plt.show()

        self.assertAlmostEqual(-np.trapz(x=mw.values(), y=dgw - dlw).imag / np.pi, Gamma, 3)


    def test_lorentzian(self):
        m = Mesh(100., 30001)
        Gamma = 3.0
        dl, dg = make_Delta_lorentzian(Gamma, 2.0, 50.0, 0.2, m)

        mw, dgw = fourier_transform(m, dg)
        mw, dlw = fourier_transform(m, dl)

        tb.cpx_plot(mw.values(), dgw - dlw)
        tb.cpx_plot(mw.values(), dgw)
        tb.cpx_plot(mw.values(), -dlw)
        plt.xlim(-2., 2.)
        plt.show()

        self.assertAlmostEqual(-np.trapz(x=mw.values(), y=dgw - dlw).imag / np.pi, Gamma, 1)


if __name__ == '__main__':
    unittest.main()