import unittest
from matplotlib import pyplot as plt
import numpy as np
from nca.utilities import *
from nca.hybridizations import *
from nca.solver_steady_state import *


class SolverSteadyStateTest(unittest.TestCase):

    def test_fluctuation_dissipation(self):
        mesh = Mesh(100., 10001)
        # times = time_mesh.values()

        ### local model
        Gamma = 1.0
        eps = -1.0 * Gamma
        U = 3. * Gamma

        ### basis: 0, dn, up, updn
        H_loc = np.array([0., eps, eps, 2*eps + U])

        beta = 10. / Gamma
        mu = 0.3 * Gamma
        D = 12.0 * Gamma

        ### Hybridization
        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, beta, mu, mesh)

        S = NCA_Steady_State_Solver(H_loc, 
                                    {0: delta_less, 1: delta_less}, 
                                    {0: delta_grea, 1: delta_grea}, mesh)

        S.greater_loop(plot=False, verbose=True)
        S.lesser_loop(plot=False, verbose=True)


        G_grea = S.get_G_grea(0)
        G_less = S.get_G_less(0)
        # dos = 1j * (G_grea - G_less) / (2 * np.pi)

        # G_grea = gf.GfReTime(mesh=time_mesh, data=G_grea)
        # G_less = gf.GfReTime(mesh=time_mesh, data=G_less)
        # dos = gf.GfReTime(mesh=time_mesh, data=dos)


        self.assertAlmostEqual(G_grea[len(mesh) // 2] - G_less[len(mesh) // 2], -1.0j, 3)

        self.assertAlmostEqual(G_grea[len(mesh) // 2 - 100], -np.conj(G_grea[len(mesh) // 2 + 100]) , 3)
        self.assertAlmostEqual(G_less[len(mesh) // 2 - 100], -np.conj(G_less[len(mesh) // 2 + 100]) , 3)


        freq_mesh, G_grea_w = fourier_transform(mesh, G_grea)
        freq_mesh, G_less_w = fourier_transform(mesh, G_less)

        dos_w = 1.j * (G_grea_w - G_less_w) / (2. * np.pi)

        mask = np.abs(freq_mesh.values() - mu) < 1.0
        np.testing.assert_allclose(G_less_w[mask] / dos_w[mask] / (2j * np.pi), tb.fermi(freq_mesh.values()[mask], mu, beta), atol = 1e-2)
        np.testing.assert_allclose(G_grea_w[mask] / dos_w[mask] / (-2j * np.pi), tb.fermi(-freq_mesh.values()[mask], -mu, beta), atol = 1e-2)



if __name__ == '__main__':
    unittest.main()