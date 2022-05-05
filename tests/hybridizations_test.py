import unittest
from matplotlib import pyplot as plt
import numpy as np
from numpy import testing
import toolbox as tb

from nca.hybridizations import *


class TestHybridization(unittest.TestCase):
    def test_semi_circ(self):
        t_mesh = Mesh(5000.0, 1000000)
        Gamma = 3.0
        D = 2.0
        beta = 5.0
        mu = 0.2
        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, beta, mu, t_mesh)

        w_mesh, delta_grea_w = fourier_transform(t_mesh, delta_grea)
        w_mesh, delta_less_w = fourier_transform(t_mesh, delta_less)

        idx0 = len(w_mesh) // 2
        self.assertAlmostEqual(
            delta_grea_w[idx0] - delta_less_w[idx0], -2.0j * Gamma, 5
        )

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

        testing.assert_allclose(delta_grea_w, delta_grea_w_ref, atol=1e-1)
        testing.assert_allclose(delta_less_w, delta_less_w_ref, atol=1e-1)

    def test_lorentzian(self):
        t_mesh = Mesh(5000.0, 1000000)
        Gamma = 3.0
        D = 2.0
        beta = 5.0
        mu = 0.2
        delta_less, delta_grea = make_Delta_lorentzian(Gamma, D, beta, mu, t_mesh)

        w_mesh, delta_grea_w = fourier_transform(t_mesh, delta_grea)
        w_mesh, delta_less_w = fourier_transform(t_mesh, delta_less)

        idx0 = len(w_mesh) // 2
        self.assertAlmostEqual(
            delta_grea_w[idx0] - delta_less_w[idx0], -2.0j * Gamma, 5
        )

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

        testing.assert_allclose(delta_grea_w, delta_grea_w_ref, atol=1e-3)
        testing.assert_allclose(delta_less_w, delta_less_w_ref, atol=1e-3)

    def test_gaussian(self):
        w_mesh = Mesh(10.0, 100000)
        t_mesh = w_mesh.adjoint()
        Gamma = 3.0
        D = 2.0
        beta = 5.0
        mu = 0.2
        delta_less, delta_grea = make_Delta_gaussian(Gamma, D, beta, mu, t_mesh)

        _, delta_grea_w = fourier_transform(t_mesh, delta_grea)
        _, delta_less_w = fourier_transform(t_mesh, delta_less)

        idx0 = len(w_mesh) // 2
        self.assertAlmostEqual(
            delta_grea_w[idx0] - delta_less_w[idx0], -2.0j * Gamma, 5
        )

        def A_ref_0(w):
            return np.exp(-((w / D) ** 2) / 2.0)

        def A_ref(w):
            return Gamma * A_ref_0(w) / A_ref_0(0.0)

        A_ref = np.vectorize(A_ref)

        delta_grea_w_ref = (
            -2j * A_ref(w_mesh.values()) * tb.one_minus_fermi(w_mesh.values(), mu, beta)
        )
        delta_less_w_ref = (
            2j * A_ref(w_mesh.values()) * tb.fermi(w_mesh.values(), mu, beta)
        )

        testing.assert_allclose(delta_grea_w, delta_grea_w_ref, atol=1e-3)
        testing.assert_allclose(delta_less_w, delta_less_w_ref, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
