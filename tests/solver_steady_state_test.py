import unittest
from matplotlib import pyplot as plt
import numpy as np
from nca.utilities import *
from nca.hybridizations import *
from nca.solver_steady_state import *


class SolverSteadyStateTest(unittest.TestCase):
    def test_fluctuation_dissipation(self):
        mesh = Mesh(3000.0, 200001)
        # times = time_mesh.values()

        ### local model
        Gamma = 1.0
        eps = -1.0 * Gamma
        U = 3.0 * Gamma

        ### basis: 0, dn, up, updn
        H_loc = np.array([0.0, eps, eps, 2 * eps + U])

        beta = 3.0 / Gamma
        Ef = 0.3 * Gamma
        D = 6.0 * Gamma
        E0 = 0.0

        ### Hybridization
        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, E0, beta, Ef, mesh)

        S = NCA_Steady_State_Solver(
            H_loc, {0: delta_less, 1: delta_less}, {0: delta_grea, 1: delta_grea}, mesh
        )

        S.greater_loop(plot=False, verbose=True)
        S.lesser_loop(plot=False, verbose=True)

        G_grea = S.get_G_grea(0)
        G_less = S.get_G_less(0)
        # dos = 1j * (G_grea - G_less) / (2 * np.pi)

        # G_grea = gf.GfReTime(mesh=time_mesh, data=G_grea)
        # G_less = gf.GfReTime(mesh=time_mesh, data=G_less)
        # dos = gf.GfReTime(mesh=time_mesh, data=dos)

        self.assertAlmostEqual(
            G_grea[len(mesh) // 2] - G_less[len(mesh) // 2], -1.0j, 2
        )

        self.assertAlmostEqual(
            G_grea[len(mesh) // 2 - 100], -np.conj(G_grea[len(mesh) // 2 + 100]), 2
        )
        self.assertAlmostEqual(
            G_less[len(mesh) // 2 - 100], -np.conj(G_less[len(mesh) // 2 + 100]), 2
        )

        freq_mesh, G_grea_w = fourier_transform(mesh, G_grea)
        freq_mesh, G_less_w = fourier_transform(mesh, G_less)

        dos_w = 1.0j * (G_grea_w - G_less_w) / (2.0 * np.pi)

        mask = np.abs(freq_mesh.values() - Ef) < 1.0
        np.testing.assert_allclose(
            G_less_w[mask] / dos_w[mask] / (2j * np.pi),
            tb.fermi(freq_mesh.values()[mask], Ef, beta),
            atol=1e-2,
        )
        np.testing.assert_allclose(
            G_grea_w[mask] / dos_w[mask] / (-2j * np.pi),
            tb.fermi(-freq_mesh.values()[mask], -Ef, beta),
            atol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
