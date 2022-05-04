import unittest
import numpy as np
from numpy import testing
import nca
from nca import second_order
from matplotlib import pyplot as plt
import toolbox as tb


class TestInitialState(unittest.TestCase):
    def test_run(self):
        beta = 3.0
        freq_mesh, delta_grea, delta_less = nca.make_Delta_semicirc_w(
            0.5, 1.0, beta, 0.0
        )
        H_loc = np.array([0.0, -0.6, -0.6, np.inf])

        init = second_order.initial_density_matrix_for_steady_state(
            H_loc, freq_mesh, delta_grea, delta_less
        )

        print(init)

        self.assertEqual(init[1, 0], init[2, 0])
        self.assertAlmostEqual(np.sum(init[:, 0]), 1.0)
        self.assertAlmostEqual(np.sum(init[:, 1]), 1.0)
        self.assertAlmostEqual(
            init[0, 0] / init[1, 0],
            tb.one_minus_fermi(-0.6, 0.0, beta) / tb.fermi(-0.6, 0.0, beta),
            5,
        )


class TestRGreater(unittest.TestCase):
    def test_against_analytic(self):
        gamma = 2.0
        eps_d = -0.3
        time_mesh = nca.Mesh(10.0 / gamma, 10000)
        delta_grea = 1j * np.exp(-np.abs(time_mesh) * gamma)
        delta_less = delta_grea.copy()

        H_loc = np.array([0.0, eps_d, eps_d])

        R2_grea = second_order.make_R2_grea(H_loc, time_mesh, delta_grea, delta_less)

        self.assertEqual(R2_grea.shape, (3, len(time_mesh)))

        tt = time_mesh.values()
        R2_up_ref = (np.exp(1j * eps_d * tt) * np.exp(-gamma * np.abs(tt)) - 1.0) / (
            1j * eps_d - np.sign(tt) * gamma
        ) - tt
        R2_up_ref *= -1j * np.exp(-1j * eps_d * tt) / (1j * eps_d - np.sign(tt) * gamma)

        # tb.cpx_plot(time_mesh, R2_grea[1])
        # tb.cpx_plot(tt, R2_up_ref, "--")
        # plt.show()

        testing.assert_allclose(R2_grea[1], R2_up_ref, atol=1e-4)


class TestRLess(unittest.TestCase):
    def test_run(self):
        freq_mesh, delta_grea_w, delta_less_w = nca.make_Delta_semicirc_w(
            1.0, 2.0, 3.0, 0.0
        )
        time_mesh, delta_less, delta_grea = nca.make_Delta_semicirc(1.0, 2.0, 3.0, 0.0)

        H_loc = np.array([0.0, -0.6, -0.6])
        init_state = second_order.initial_density_matrix_for_steady_state(
            H_loc, freq_mesh, delta_grea_w, delta_less_w
        )
        init_state = init_state[:, 0]
        init_state = init_state[:3]
        print(init_state)

        data = second_order.make_R2_less(
            H_loc, time_mesh, delta_grea, delta_less, init_state
        )

        tb.cpx_plot(time_mesh, data)
        plt.show()


class TestGreenFunction(unittest.TestCase):
    def test_run(self):
        time_mesh, gf1, gf2 = second_order.GF2_grea()

        tb.cpx_plot(time_mesh, gf1 + gf2)
        # tb.cpx_plot(time_mesh, gf2, "--")
        plt.show()


if __name__ == "__main__":
    unittest.main()
