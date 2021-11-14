import unittest
from matplotlib import pyplot as plt
import numpy as np
from numpy.testing._private.utils import assert_equal
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

        ### normalization tests
        idx0 = len(S.times) // 2
        self.assertEqual(S.times[idx0], 0.0)

        for k in range(4):
            self.assertAlmostEqual(S.R_grea[idx0, k], -1j)
            self.assertAlmostEqual(
                np.trapz(S.R_grea_w[:, k], dx=S.freq_mesh.delta) / (2 * np.pi), -1j, 3
            )

        self.assertAlmostEqual(np.sum(S.R_grea[idx0, :]), -4j)
        self.assertAlmostEqual(
            np.trapz(np.sum(S.R_grea_w[:, :], axis=1), dx=S.freq_mesh.delta)
            / (2 * np.pi),
            -4j,
            3,
        )

    def test_orbital_in_state(self):
        H_loc = [0.0, 1.0, 1.0, 3.0]

        time_mesh = Mesh(10.0, 101)
        Delta = np.empty(101)

        S = NCA_Steady_State_Solver(
            H_loc,
            {0: Delta, 1: Delta},
            {0: Delta, 1: Delta},
            time_mesh,
        )

        # orbitals: 0 = up, 1 = down,
        # states: 0 = empty, 1 = up, 2 = down, 3 = both

        self.assertEqual(S.is_orb_in_state(0, 0), False)
        self.assertEqual(S.is_orb_in_state(1, 0), False)
        self.assertEqual(S.is_orb_in_state(0, 1), True)
        self.assertEqual(S.is_orb_in_state(1, 1), False)
        self.assertEqual(S.is_orb_in_state(0, 2), False)
        self.assertEqual(S.is_orb_in_state(1, 2), True)
        self.assertEqual(S.is_orb_in_state(0, 3), True)
        self.assertEqual(S.is_orb_in_state(1, 3), True)

    def test_self_energy(self):
        H_loc = [0.0, 1.0, 1.0, 3.0]
        time_mesh = Mesh(10.0, 101)
        times = time_mesh.values()
        Delta_less_up = np.cos(3.0 * times) * np.sin(0.1 * times + 5.0)
        Delta_grea_up = np.cos(4.0 * times) * np.sin(0.2 * times + 2.0)
        Delta_less_dn = np.cos(5.0 * times) * np.sin(0.6 * times - 1.0)
        Delta_grea_dn = np.cos(6.0 * times) * np.sin(0.5 * times + 3.0)

        S = NCA_Steady_State_Solver(
            H_loc,
            {0: Delta_less_up, 1: Delta_less_dn},
            {0: Delta_grea_up, 1: Delta_grea_dn},
            time_mesh,
        )

        S.R_grea[:, 0] = np.sin(5.0 * times) * np.cos(0.6 * times - 1.0)
        S.R_grea[:, 1] = np.sin(2.0 * times) * np.cos(0.3 * times - 4.0)
        S.R_grea[:, 2] = np.sin(7.0 * times) * np.cos(0.2 * times - 3.0)
        S.R_grea[:, 3] = np.sin(1.0 * times) * np.cos(0.5 * times - 9.0)

        S.self_energy_grea()

        # states: 0 = empty, 1 = up, 2 = down, 3 = both

        np.testing.assert_array_almost_equal(
            S.S_grea[:, 0],
            -1j * Delta_less_up[::-1] * S.R_grea[:, 1]
            - 1j * Delta_less_dn[::-1] * S.R_grea[:, 2],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_grea[:, 1],
            1j * Delta_grea_up * S.R_grea[:, 0]
            - 1j * Delta_less_dn[::-1] * S.R_grea[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_grea[:, 2],
            1j * Delta_grea_dn * S.R_grea[:, 0]
            - 1j * Delta_less_up[::-1] * S.R_grea[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_grea[:, 3],
            1j * Delta_grea_dn * S.R_grea[:, 1] + 1j * Delta_grea_up * S.R_grea[:, 2],
            10,
        )

        S.R_less[:, 0] = np.sin(5.0 * times) * np.cos(0.6 * times - 1.0)
        S.R_less[:, 1] = np.sin(2.0 * times) * np.cos(0.3 * times - 4.0)
        S.R_less[:, 2] = np.sin(7.0 * times) * np.cos(0.2 * times - 3.0)
        S.R_less[:, 3] = np.sin(1.0 * times) * np.cos(0.5 * times - 9.0)

        S.self_energy_less()

        # states: 0 = empty, 1 = up, 2 = down, 3 = both

        np.testing.assert_array_almost_equal(
            S.S_less[:, 0],
            1j * Delta_grea_up[::-1] * S.R_less[:, 1]
            + 1j * Delta_grea_dn[::-1] * S.R_less[:, 2],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_less[:, 1],
            -1j * Delta_less_up * S.R_less[:, 0]
            + 1j * Delta_grea_dn[::-1] * S.R_less[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_less[:, 2],
            -1j * Delta_less_dn * S.R_less[:, 0]
            + 1j * Delta_grea_up[::-1] * S.R_less[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_less[:, 3],
            -1j * Delta_less_dn * S.R_less[:, 1] - 1j * Delta_less_up * S.R_less[:, 2],
            10,
        )


if __name__ == "__main__":
    unittest.main()
