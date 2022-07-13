import unittest
import numpy as np
from numpy import testing
import nca
from matplotlib import pyplot as plt


class TestNcaVsInchworm(unittest.TestCase):
    def test_imag_time_nca(self):

        beta = 5.0  # Inverse temperature
        mu = 2.0  # Chemical potential
        U = 5.0  # On-site density-density interaction
        h = 0.2  # Local magnetic field
        D = 5.0  # half bandwidth
        Gamma = 1.0  # Hybridization energy

        tau_mesh = nca.Mesh(10.0 * beta, 30001, beta)

        ### basis: 0, up, dn, updn
        H_loc = np.array([0.0, -mu - h, -mu + h, -2 * mu + U])

        delta_tau = nca.make_Delta_semicirc_tau(Gamma, D, 0.0, beta, 300, tau_mesh)

        S = nca.SolverImagTime(beta, H_loc, {0: delta_tau, 1: delta_tau}, tau_mesh)

        S.solve(verbose=True)

        ### We want to check that R_nca(tau) == R_inch(tau) e^{w0 tau} for some unknown w0 (energy shift).
        ### To do so we compare R(tau) R(beta - tau) / R(beta)

        taus_inch, R_inch = self.get_inchworm_data()
        R_inch_norm = R_inch * R_inch[::-1] / R_inch[-1]

        for k in range(4):
            R_nca = np.interp(taus_inch, S.times, np.real(S.R_tau[:, k]))
            if R_nca[-1] == 0.0:
                raise ZeroDivisionError
            R_nca_norm = R_nca * R_nca[::-1] / R_nca[-1]

            # plt.plot(taus_inch, R_inch_norm[:, k], label="inch")
            # plt.plot(taus_inch, R_nca_norm, label="nca")
            # plt.legend()
            # plt.show()

            # print(R_inch_norm[:, k])
            # print(R_nca_norm)

            testing.assert_allclose(R_inch_norm[:, k], R_nca_norm, atol=1e-2)

    def get_inchworm_data(self):
        taus = np.array(
            [
                0.0,
                0.5005005,
                1.001001,
                1.5015015,
                2.002002,
                2.5025025,
                3.003003,
                3.5035035,
                4.004004,
                4.5045045,
            ]
        )

        R_tau = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [-0.44845404, -1.17743061, -0.97090024, -0.28605395],
                [-0.32490958, -1.5369445, -1.05971033, -0.16268471],
                [-0.30714028, -2.04870586, -1.19289107, -0.1393724],
                [-0.33547125, -2.76085819, -1.36378069, -0.14424291],
                [-0.3988408, -3.73832364, -1.5828727, -0.1655807],
                [-0.50796953, -5.09357882, -1.8536079, -0.20732064],
                [-0.69050845, -6.96211494, -2.19812602, -0.28146611],
                [-0.99517947, -9.58614129, -2.65580609, -0.41657934],
                [-1.60068009, -13.28346998, -3.31557195, -0.70514934],
            ]
        )

        return taus, R_tau


if __name__ == "__main__":
    unittest.main()
