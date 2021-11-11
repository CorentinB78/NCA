import unittest
from matplotlib import pyplot as plt
import numpy as np

from nca.hybridizations import *


class HybridizationTest(unittest.TestCase):
    def test_semi_circ(self):
        m = Mesh(500.0, 10001)
        Gamma = 3.0
        dl, dg = make_Delta_semicirc(
            Gamma, D=2.0, E0=0.0, beta=5.0, Ef=0.2, time_mesh=m
        )

        mw, dgw = fourier_transform(m, dg)
        mw, dlw = fourier_transform(m, dl)

        tb.cpx_plot(mw.values(), dgw - dlw)
        tb.cpx_plot(mw.values(), dgw)
        tb.cpx_plot(mw.values(), -dlw)
        plt.xlim(-2.0, 2.0)
        plt.show()

        self.assertAlmostEqual((dgw - dlw)[len(mw) // 2], -1j * Gamma, 3)

    def test_lorentzian(self):
        m = Mesh(500.0, 10001)
        Gamma = 3.0
        dl, dg = make_Delta_lorentzian(
            Gamma, D=2.0, E0=0.0, beta=5.0, Ef=0.2, time_mesh=m
        )

        mw, dgw = fourier_transform(m, dg)
        mw, dlw = fourier_transform(m, dl)

        tb.cpx_plot(mw.values(), dgw - dlw)
        tb.cpx_plot(mw.values(), dgw)
        tb.cpx_plot(mw.values(), -dlw)
        plt.xlim(-2.0, 2.0)
        plt.show()

        self.assertAlmostEqual((dgw - dlw)[len(mw) // 2], -1j * Gamma, 3)


if __name__ == "__main__":
    unittest.main()
