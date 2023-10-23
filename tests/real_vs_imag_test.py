import unittest
import numpy as np
from numpy import testing
from matplotlib import pyplot as plt
import nca

from nca.hybridizations import gf_tau_from_dos


class TestParams1(unittest.TestCase):
    def test_compare(self):
        beta = 10.0  # Inverse temperature
        mu = -1.0  # Chemical potential
        U = 30.0  # On-site density-density interaction
        h = 0.2  # Local magnetic field
        D = 50.0  # bandwidth
        Gamma = 1.0  # Hybridization energy

        H_loc = np.array([0.0, -mu - h, -mu + h, -2 * mu + U])

        ### Imag time
        tau_mesh = nca.Mesh(100.0, 10001, beta)

        delta = nca.make_Delta_semicirc_tau(Gamma, D, 0.0, beta, 1001, tau_mesh)

        S_imag = nca.SolverImagTime(beta, H_loc, {0: delta, 1: delta}, tau_mesh)

        S_imag.solve(verbose=True)

        taus, G_imag = S_imag.get_G_tau(0)
        self.assertAlmostEqual(taus[0], 0.0)
        self.assertAlmostEqual(taus[-1], beta)

        ### Real time
        time_mesh = nca.Mesh(500.0, 300001)

        dos = nca.make_semicircular_dos(D)
        delta_grea, delta_less = nca.make_hyb_times(dos, beta, 0.0, Gamma, time_mesh)

        S_real = nca.SolverSteadyState(2, H_loc, time_mesh)
        S_real.add_bath(0, delta_grea, delta_less)
        S_real.add_bath(1, delta_grea, delta_less)

        S_real.greater_loop(verbose=True)
        S_real.lesser_loop(verbose=True)

        freq_mesh, dos = S_real.get_DOS(0)

        idx_subset = np.arange(len(taus))[:: len(taus) // 30]
        idx_subset = np.append(idx_subset, len(taus) - 1)
        taus_subset = taus[idx_subset]
        self.assertGreater(len(taus_subset), 5)
        self.assertAlmostEqual(taus_subset[0], 0.0)
        self.assertAlmostEqual(taus_subset[-1], beta)

        G_real = gf_tau_from_dos(taus_subset, beta, freq_mesh.values(), dos)

        # ### Plot
        # plt.plot(taus_subset, G_real, "-x", label="real times")
        # plt.plot(taus, G_imag, label="imag times")

        # plt.legend()
        # plt.xlabel("tau")
        # plt.ylabel("G_imag")
        # plt.title(f"Max abs err: {np.max(np.abs(G_real - G_imag[idx_subset]))}")
        # plt.show()

        ### Compare
        testing.assert_allclose(G_real, G_imag[idx_subset], atol=1e-2)


class TestParamsInchworm(unittest.TestCase):
    def test_compare(self):
        beta = 5.0  # Inverse temperature
        mu = 2.0  # Chemical potential
        U = 5.0  # On-site density-density interaction
        h = 0.2  # Local magnetic field
        D = 5.0  # half bandwidth
        Gamma = 1.0  # Hybridization energy

        H_loc = np.array([0.0, -mu - h, -mu + h, -2 * mu + U])

        ### Imag time
        tau_mesh = nca.Mesh(100.0, 10001, beta)

        delta = nca.make_Delta_semicirc_tau(Gamma, D, 0.0, beta, 1001, tau_mesh)

        S_imag = nca.SolverImagTime(beta, H_loc, {0: delta, 1: delta}, tau_mesh)

        S_imag.solve(verbose=True)

        taus, G_imag = S_imag.get_G_tau(0)
        self.assertAlmostEqual(taus[0], 0.0)
        self.assertAlmostEqual(taus[-1], beta)

        ### define subset of taus
        idx_subset = np.arange(len(taus))[:: len(taus) // 30]
        idx_subset = np.append(idx_subset, len(taus) - 1)
        taus_subset = taus[idx_subset]
        self.assertGreater(len(taus_subset), 5)
        self.assertAlmostEqual(taus_subset[0], 0.0)
        self.assertAlmostEqual(taus_subset[-1], beta)

        ### Real time
        time_mesh = nca.Mesh(500.0, 300001)

        dos = nca.make_semicircular_dos(D)
        delta_grea, delta_less = nca.make_hyb_times(dos, beta, 0.0, Gamma, nca.Mesh(1000., 1000000))

        S_real = nca.SolverSteadyState(2, H_loc, time_mesh)

        S_real.add_bath(0, delta_grea, delta_less)
        S_real.add_bath(1, delta_grea, delta_less)

        S_real.greater_loop(verbose=True)
        S_real.lesser_loop(verbose=True)

        freq_mesh, dos = S_real.get_DOS(0)

        G_real = gf_tau_from_dos(taus_subset, beta, freq_mesh.values(), dos)

        # ### Plot
        # plt.plot(taus_subset, G_real, "-x", label="real times")
        # plt.plot(taus, G_imag, label="imag times")

        # plt.legend()
        # plt.xlabel("tau")
        # plt.ylabel("G_imag")
        # plt.title(f"Max abs err: {np.max(np.abs(G_real - G_imag[idx_subset]))}")
        # plt.show()

        ### Real time Alpert
        time_mesh = nca.Mesh(500.0, 300001)

        dos = nca.make_semicircular_dos(D)
        delta_grea, delta_less = nca.make_hyb_times(dos, beta, 0.0, Gamma, nca.Mesh(1000., 1000000))

        S_real = nca.SolverSteadyState(2, H_loc, time_mesh, order=8)

        S_real.add_bath(0, delta_grea, delta_less)
        S_real.add_bath(1, delta_grea, delta_less)

        S_real.greater_loop(verbose=True)
        S_real.lesser_loop(verbose=True)

        freq_mesh, dos = S_real.get_DOS(0)

        G_real_alpert = gf_tau_from_dos(taus_subset, beta, freq_mesh.values(), dos)

        ### Compare
        testing.assert_allclose(G_real, G_imag[idx_subset], atol=1e-2)
        testing.assert_allclose(G_real_alpert, G_imag[idx_subset], atol=1e-2)


if __name__ == "__main__":
    unittest.main()
