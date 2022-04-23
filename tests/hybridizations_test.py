import unittest
from matplotlib import pyplot as plt
import numpy as np
from numpy import testing
import toolbox as tb

from nca.hybridizations import *


class HybridizationTest(unittest.TestCase):
    def test_semi_circ(self):
        t_mesh = Mesh(500.0, 10001)
        Gamma = 3.0
        D = 2.0
        beta = 5.0
        mu = 0.2
        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, 0.0, beta, mu, t_mesh)

        w_mesh, delta_grea_w = fourier_transform(t_mesh, delta_grea)
        w_mesh, delta_less_w = fourier_transform(t_mesh, delta_less)

        def A_ref(w):
            if np.abs(w) >= D:
                return 0.0
            return Gamma / (np.pi * D) * np.sqrt(D**2 - w**2)

        A_ref = np.vectorize(A_ref)

        delta_grea_w_ref = (
            -2j
            * np.pi
            * A_ref(w_mesh.values())
            * tb.one_minus_fermi(w_mesh.values(), mu, beta)
        )
        delta_less_w_ref = (
            2j * np.pi * A_ref(w_mesh.values()) * tb.fermi(w_mesh.values(), mu, beta)
        )

        testing.assert_allclose(delta_grea_w, delta_grea_w_ref, rtol=1e-4, atol=1e-8)
        testing.assert_allclose(delta_less_w, delta_less_w_ref, rtol=1e-4, atol=1e-8)

        dgw_0 = np.trapz(delta_grea, dx=t_mesh.delta)
        dlw_0 = np.trapz(delta_less, dx=t_mesh.delta)

        self.assertAlmostEqual(0.5 * (dgw_0 - dlw_0), -1j * Gamma, 3)
        self.assertAlmostEqual(
            0.5 * (delta_grea_w - delta_less_w)[len(w_mesh) // 2], -1j * Gamma, 3
        )

    def test_lorentzian(self):
        t_mesh = Mesh(500.0, 10001)
        Gamma = 3.0
        D = 2.0
        beta = 5.0
        mu = 0.2
        delta_less, delta_grea = make_Delta_lorentzian(Gamma, D, 0.0, beta, mu, t_mesh)

        w_mesh, delta_grea_w = fourier_transform(t_mesh, delta_grea)
        w_mesh, delta_less_w = fourier_transform(t_mesh, delta_less)

        def A_ref(w):
            return Gamma * D**2 / (w**2 + D**2) / np.pi

        A_ref = np.vectorize(A_ref)

        delta_grea_w_ref = (
            -2j
            * np.pi
            * A_ref(w_mesh.values())
            * tb.one_minus_fermi(w_mesh.values(), mu, beta)
        )
        delta_less_w_ref = (
            2j * np.pi * A_ref(w_mesh.values()) * tb.fermi(w_mesh.values(), mu, beta)
        )

        testing.assert_allclose(delta_grea_w, delta_grea_w_ref, rtol=1e-4, atol=1e-8)
        testing.assert_allclose(delta_less_w, delta_less_w_ref, rtol=1e-4, atol=1e-8)

        dgw_0 = np.trapz(delta_grea, dx=t_mesh.delta)
        dlw_0 = np.trapz(delta_less, dx=t_mesh.delta)

        self.assertAlmostEqual(0.5 * (dgw_0 - dlw_0), -1j * Gamma, 3)
        self.assertAlmostEqual(
            0.5 * (delta_grea_w - delta_less_w)[len(w_mesh) // 2], -1j * Gamma, 3
        )


if __name__ == "__main__":
    unittest.main()
