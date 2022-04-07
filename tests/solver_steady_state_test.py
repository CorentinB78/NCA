import unittest
import numpy as np
from numpy import testing
from nca.utilities import *
from nca.hybridizations import *
from nca.solver_steady_state import *
from nca.fock_space import *


class SolverSteadyStateTest(unittest.TestCase):
    def compute_nca(self):
        mesh = Mesh(100.0, 200001)
        # times = time_mesh.values()

        ### local model
        Gamma = 1.0
        eps = -1.0
        U = 3.0

        ### basis: 0, dn, up, updn
        H_loc = np.array([0.0, eps, eps, 2 * eps + U])

        beta = 3.0
        Ef = 0.3
        D = 6.0
        E0 = 0.0

        ### Hybridization
        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, E0, beta, Ef, mesh)

        fock = FermionicFockSpace(["up", "dn"])
        fock.add_bath(0, delta_grea, delta_less)
        fock.add_bath(1, delta_grea, delta_less)
        hybs = fock.generate_hybridizations()

        print(fock.basis())

        S = NCA_Steady_State_Solver(H_loc, mesh, hybs, [0, 3])

        S.greater_loop(plot=False, verbose=True)
        S.lesser_loop(plot=False, verbose=True, max_iter=20)

        return fock, S

    def test_sanity_checks(self):
        beta = 3.0
        Ef = 0.3
        fock, S = self.compute_nca()

        ### check sanity check utility
        sanity_checks(S, fock)

        ### R & S

        ### Fourier transforms
        w_ref, R_less_w_ref = fourier_transform(S.time_mesh, S.R_less, axis=0)
        testing.assert_allclose(w_ref.values(), S.freqs)
        testing.assert_allclose(R_less_w_ref, S.R_less_w, atol=1e-4)

        _, R_grea_w_ref = fourier_transform(S.time_mesh, S.R_grea, axis=0)
        testing.assert_allclose(R_grea_w_ref, S.R_grea_w, atol=1e-4)

        _, S_less_w_ref = fourier_transform(S.time_mesh, S.S_less, axis=0)
        testing.assert_allclose(S_less_w_ref, S.S_less_w, atol=1e-4)

        _, S_grea_w_ref = fourier_transform(S.time_mesh, S.S_grea, axis=0)
        testing.assert_allclose(S_grea_w_ref, S.S_grea_w, atol=1e-4)

        ### symmetries: diagonal lessers and greaters are pure imaginary
        testing.assert_allclose(S.R_less_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.R_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.S_less_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.S_grea_w.real, 0.0, atol=1e-8)

        ### normalization
        idx0 = len(S.times) // 2
        self.assertEqual(S.times[idx0], 0.0)

        for k in range(4):
            self.assertAlmostEqual(S.R_grea[idx0, k], -1j)

        self.assertAlmostEqual(np.sum(S.R_less[idx0, :]), -4j, 2)

        ### Green functions

        G_grea = fock.get_G_grea(0, S)
        G_less = fock.get_G_less(0, S)
        Dos_w = fock.get_DOS(0, S)

        _, G_grea_w = fourier_transform(S.time_mesh, G_grea)
        _, G_less_w = fourier_transform(S.time_mesh, G_less)

        ### normalization and DoS
        Dos_w_ref = np.real(1j * (G_grea_w - G_less_w) / (2 * np.pi))
        testing.assert_allclose(Dos_w_ref, Dos_w, atol=1e-8)
        testing.assert_allclose(np.trapz(x=S.freqs, y=Dos_w), 1.0, atol=1e-6)

        ### Symmetries: diagonal lessers and greaters are pure imaginary and do not change sign
        testing.assert_allclose(G_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(G_less_w.real, 0.0, atol=1e-8)
        testing.assert_array_less(G_grea_w.imag, 1e-8)
        testing.assert_array_less(-G_less_w.imag, 1e-8)

        mask = np.abs(S.freqs - Ef) < 1.0
        np.testing.assert_allclose(
            G_less_w[mask] / Dos_w[mask] / (2j * np.pi),
            tb.fermi(S.freqs[mask], Ef, beta),
            atol=1e-2,
        )
        np.testing.assert_allclose(
            G_grea_w[mask] / Dos_w[mask] / (-2j * np.pi),
            tb.fermi(-S.freqs[mask], -Ef, beta),
            atol=1e-2,
        )

    def test_orbital_in_state(self):

        fock = FermionicFockSpace(["up", "dn"])

        # orbitals: 0 = up, 1 = down,
        # states: 0 = empty, 1 = up, 2 = down, 3 = both

        self.assertEqual(fock.is_orb_in_state(0, 0), False)
        self.assertEqual(fock.is_orb_in_state(1, 0), False)
        self.assertEqual(fock.is_orb_in_state(0, 1), True)
        self.assertEqual(fock.is_orb_in_state(1, 1), False)
        self.assertEqual(fock.is_orb_in_state(0, 2), False)
        self.assertEqual(fock.is_orb_in_state(1, 2), True)
        self.assertEqual(fock.is_orb_in_state(0, 3), True)
        self.assertEqual(fock.is_orb_in_state(1, 3), True)

        np.testing.assert_array_equal(fock.states_containing(0), ([1, 3], [0, 2]))
        np.testing.assert_array_equal(fock.states_containing(1), ([2, 3], [0, 1]))

    def test_self_energy(self):
        H_loc = [0.0, 1.0, 1.0, 3.0]
        time_mesh = Mesh(10.0, 101)
        times = time_mesh.values()
        Delta_less_up = np.cos(3.0 * times) * np.sin(0.1 * times + 5.0)
        Delta_grea_up = np.cos(4.0 * times) * np.sin(0.2 * times + 2.0)
        Delta_less_dn = np.cos(5.0 * times) * np.sin(0.6 * times - 1.0)
        Delta_grea_dn = np.cos(6.0 * times) * np.sin(0.5 * times + 3.0)

        fock = FermionicFockSpace(["up", "dn"])
        fock.add_bath(0, Delta_grea_up, Delta_less_up)
        fock.add_bath(1, Delta_grea_dn, Delta_less_dn)
        hybs = fock.generate_hybridizations()

        S = NCA_Steady_State_Solver(H_loc, time_mesh, hybs, [0, 3])

        S.R_grea[:, 0] = np.sin(5.0 * times) * np.cos(0.6 * times - 1.0)
        S.R_grea[:, 1] = np.sin(2.0 * times) * np.cos(0.3 * times - 4.0)
        S.R_grea[:, 2] = np.sin(7.0 * times) * np.cos(0.2 * times - 3.0)
        S.R_grea[:, 3] = np.sin(1.0 * times) * np.cos(0.5 * times - 9.0)

        S.self_energy_grea(S.is_even_state)
        S.self_energy_grea(~S.is_even_state)

        # states: 0 = empty, 1 = up, 2 = down, 3 = both

        np.testing.assert_array_almost_equal(
            S.S_grea[:, 0],
            1j * np.conj(Delta_less_up) * S.R_grea[:, 1]
            + 1j * np.conj(Delta_less_dn) * S.R_grea[:, 2],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_grea[:, 1],
            1j * Delta_grea_up * S.R_grea[:, 0]
            + 1j * np.conj(Delta_less_dn) * S.R_grea[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_grea[:, 2],
            1j * Delta_grea_dn * S.R_grea[:, 0]
            + 1j * np.conj(Delta_less_up) * S.R_grea[:, 3],
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

        S.self_energy_less(S.is_even_state)
        S.self_energy_less(~S.is_even_state)

        # states: 0 = empty, 1 = up, 2 = down, 3 = both

        np.testing.assert_array_almost_equal(
            S.S_less[:, 0],
            -1j * np.conj(Delta_grea_up) * S.R_less[:, 1]
            - 1j * np.conj(Delta_grea_dn) * S.R_less[:, 2],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_less[:, 1],
            -1j * Delta_less_up * S.R_less[:, 0]
            - 1j * np.conj(Delta_grea_dn) * S.R_less[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_less[:, 2],
            -1j * Delta_less_dn * S.R_less[:, 0]
            - 1j * np.conj(Delta_grea_up) * S.R_less[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.S_less[:, 3],
            -1j * Delta_less_dn * S.R_less[:, 1] - 1j * Delta_less_up * S.R_less[:, 2],
            10,
        )

    def test_values(self):
        beta = 1.0
        mu = 0.5
        U = 1.0
        D = 10.0
        Gamma = 1.0

        time_mesh = Mesh(1000.0, 100001)

        ### basis: 0, up, dn, updn
        H_loc = np.array([0.0, -mu, -mu, -2 * mu + U])

        delta_less, delta_grea = make_Delta_semicirc(
            Gamma, D, 0.0, beta, 0.0, time_mesh
        )

        fock = FermionicFockSpace(["up", "dn"])
        fock.add_bath(0, delta_grea, delta_less)
        fock.add_bath(1, delta_grea, delta_less)
        hybs = fock.generate_hybridizations()

        S = NCA_Steady_State_Solver(H_loc, time_mesh, hybs, [0, 3])

        S.greater_loop(tol=1e-5, verbose=True)
        S.lesser_loop(tol=1e-5, verbose=True)

        times_ref = np.linspace(-5.0, 5.0, 11)

        ### data from Renaud Garioud (Nov 2021)
        G_grea_ref = 0.5 * np.array(
            [
                3.81553968e-04 - 1.99104875e-04j,
                2.12026824e-03 - 1.81560990e-03j,
                9.81523185e-03 - 9.98405419e-03j,
                4.08214582e-02 - 4.67503833e-02j,
                1.62294099e-01 - 1.96646491e-01j,
                -3.52101334e-12 - 9.99328108e-01j,
                -1.62294099e-01 - 1.96646491e-01j,
                -4.08214582e-02 - 4.67503833e-02j,
                -9.81523185e-03 - 9.98405419e-03j,
                -2.12026824e-03 - 1.81560990e-03j,
                -3.81553968e-04 - 1.99104875e-04j,
            ]
        )
        G_less_ref = np.conj(G_grea_ref)

        G_grea = np.interp(times_ref, S.times, fock.get_G_grea(0, S))
        np.testing.assert_array_almost_equal(G_grea, G_grea_ref, 3)

        G_less = np.interp(times_ref, S.times, fock.get_G_less(0, S))
        np.testing.assert_array_almost_equal(G_less, G_less_ref, 3)


class SolverSteadyStateDysonTest(unittest.TestCase):
    def compute_nca(self):
        mesh = Mesh(300.0, 200001)

        ### local model
        Gamma = 1.0
        eps = -1.0
        U = 3.0

        ### basis: 0, dn, up, updn
        H_loc = np.array([0.0, eps, eps, 2 * eps + U])
        decay_t = 3000.0
        R0 = 1.0 / (
            mesh.adjoint().values()[:, None] - H_loc[None, :] + 2j * np.pi / decay_t
        )
        # R0 = (
        #     -1j
        #     * np.exp(-1j * H_loc[None, :] * mesh.values()[:, None])
        #     * np.exp(-np.abs(mesh.values()[:, None]) / decay_t)
        # )

        beta = 3.0
        Ef = 0.3
        D = 6.0
        E0 = 0.0

        ### Hybridization
        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, E0, beta, Ef, mesh)

        fock = FermionicFockSpace(["up", "dn"])
        fock.add_bath(0, delta_grea, delta_less)
        fock.add_bath(1, delta_grea, delta_less)
        hybs = fock.generate_hybridizations()

        print(fock.basis())

        S = NCA_Steady_State_Solver(R0, mesh, hybs, [0, 3])

        S.greater_loop(plot=False, verbose=True)
        S.lesser_loop(verbose=True, max_iter=100)

        return fock, S

    def test_sanity_checks(self):
        beta = 3.0
        Ef = 0.3
        fock, S = self.compute_nca()

        ### R & S

        ### Fourier transforms
        w_ref, R_less_w_ref = fourier_transform(S.time_mesh, S.R_less, axis=0)
        testing.assert_allclose(w_ref.values(), S.freqs)
        testing.assert_allclose(R_less_w_ref, S.R_less_w, atol=1e-4)

        _, R_grea_w_ref = fourier_transform(S.time_mesh, S.R_grea, axis=0)
        testing.assert_allclose(R_grea_w_ref, S.R_grea_w, atol=1e-4)

        _, S_less_w_ref = fourier_transform(S.time_mesh, S.S_less, axis=0)
        testing.assert_allclose(S_less_w_ref, S.S_less_w, atol=1e-4)

        _, S_grea_w_ref = fourier_transform(S.time_mesh, S.S_grea, axis=0)
        testing.assert_allclose(S_grea_w_ref, S.S_grea_w, atol=1e-4)

        ### symmetries: diagonal lessers and greaters are pure imaginary
        testing.assert_allclose(S.R_less_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.R_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.S_less_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.S_grea_w.real, 0.0, atol=1e-8)

        ### normalization
        idx0 = len(S.times) // 2
        self.assertEqual(S.times[idx0], 0.0)

        for k in range(4):
            self.assertAlmostEqual(S.R_grea[idx0, k], -1j)

        self.assertAlmostEqual(np.sum(S.R_less[idx0, :]), -4j, 2)

        ### Green functions

        G_grea = fock.get_G_grea(0, S)
        G_less = fock.get_G_less(0, S)
        Dos_w = fock.get_DOS(0, S)

        _, G_grea_w = fourier_transform(S.time_mesh, G_grea)
        _, G_less_w = fourier_transform(S.time_mesh, G_less)

        ### normalization and DoS
        Dos_w_ref = np.real(1j * (G_grea_w - G_less_w) / (2 * np.pi))
        testing.assert_allclose(Dos_w_ref, Dos_w, atol=1e-8)
        testing.assert_allclose(np.trapz(x=S.freqs, y=Dos_w), 1.0, atol=1e-6)

        ### Symmetries: diagonal lessers and greaters are pure imaginary and do not change sign
        testing.assert_allclose(G_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(G_less_w.real, 0.0, atol=1e-8)
        testing.assert_array_less(G_grea_w.imag, 1e-8)
        testing.assert_array_less(-G_less_w.imag, 1e-8)

        mask = np.abs(S.freqs - Ef) < 1.0
        np.testing.assert_allclose(
            G_less_w[mask] / Dos_w[mask] / (2j * np.pi),
            tb.fermi(S.freqs[mask], Ef, beta),
            atol=1e-2,
        )
        np.testing.assert_allclose(
            G_grea_w[mask] / Dos_w[mask] / (-2j * np.pi),
            tb.fermi(-S.freqs[mask], -Ef, beta),
            atol=1e-2,
        )

    def test_values(self):
        beta = 1.0
        mu = 0.5
        U = 1.0
        D = 10.0
        Gamma = 1.0

        time_mesh = Mesh(1000.0, 100001)

        ### basis: 0, up, dn, updn
        H_loc = np.array([0.0, -mu, -mu, -2 * mu + U])
        decay_t = 1000.0
        R0 = 1.0 / (
            time_mesh.adjoint().values()[:, None]
            - H_loc[None, :]
            + 2j * np.pi / decay_t
        )
        # R0 = (
        #     -1j
        #     * np.exp(-1j * time_mesh.values()[:, None] * H_loc[None, :])
        #     * np.exp(-np.abs(time_mesh.values()[:, None]) / decay_t)
        # )

        delta_less, delta_grea = make_Delta_semicirc(
            Gamma, D, 0.0, beta, 0.0, time_mesh
        )

        fock = FermionicFockSpace(["up", "dn"])
        fock.add_bath(0, delta_grea, delta_less)
        fock.add_bath(1, delta_grea, delta_less)
        hybs = fock.generate_hybridizations()

        S = NCA_Steady_State_Solver(R0, time_mesh, hybs, [0, 3])

        S.greater_loop(tol=1e-10, verbose=True, plot=False)
        S.lesser_loop(tol=1e-10, verbose=True)
        # plt.plot(S.times, S.R_less[:, 2].real)
        # plt.plot(S.times, S.R_less[:, 2].imag)
        # plt.show()

        times_ref = np.linspace(-5.0, 5.0, 11)

        ### data from Renaud Garioud (Nov 2021)
        G_grea_ref = 0.5 * np.array(
            [
                3.81553968e-04 - 1.99104875e-04j,
                2.12026824e-03 - 1.81560990e-03j,
                9.81523185e-03 - 9.98405419e-03j,
                4.08214582e-02 - 4.67503833e-02j,
                1.62294099e-01 - 1.96646491e-01j,
                -3.52101334e-12 - 9.99328108e-01j,
                -1.62294099e-01 - 1.96646491e-01j,
                -4.08214582e-02 - 4.67503833e-02j,
                -9.81523185e-03 - 9.98405419e-03j,
                -2.12026824e-03 - 1.81560990e-03j,
                -3.81553968e-04 - 1.99104875e-04j,
            ]
        )
        G_less_ref = np.conj(G_grea_ref)

        # plt.plot(times_ref, G_grea_ref.real, "o-")
        # plt.plot(S.times, fock.get_G_grea(0, S).real, "--")
        # plt.xlim(-10, 10)
        # plt.show()

        # plt.plot(times_ref, G_less_ref.real, "o-")
        # plt.plot(S.times, fock.get_G_less(0, S).real, "--")
        # plt.xlim(-10, 10)
        # plt.show()

        G_grea = np.interp(times_ref, S.times, fock.get_G_grea(0, S))
        np.testing.assert_array_almost_equal(G_grea, G_grea_ref, 2)

        G_less = np.interp(times_ref, S.times, fock.get_G_less(0, S))
        np.testing.assert_array_almost_equal(G_less, G_less_ref, 2)


# TODO: add comparison with imaginary times

if __name__ == "__main__":
    unittest.main()
