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

        ### Hybridization
        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, beta, Ef, mesh)

        fock = FermionicFockSpace(["up", "dn"])
        fock.add_bath(0, delta_grea, delta_less)
        fock.add_bath(1, delta_grea, delta_less)
        hybs = fock.generate_hybridizations()

        print(fock.basis())

        S = SolverSteadyState(H_loc, mesh, hybs, [0, 3])

        S.greater_loop(plot=False, verbose=True)
        S.lesser_loop(plot=False, verbose=True, max_iter=20)

        return fock, S

    def test_R_reta_non_reg(self):
        fock, S = self.compute_nca()

        data = np.loadtxt("tests/data_ref/R_reta_w.dat", dtype=complex)
        w_ref = data[:, 0].real
        R_reta_w_ref = data[:, 1:]

        R_reta_w_0 = tb.cpx_interp(w_ref, S.freq_meshes[0].values(), S.R_reta_w[:, 0])
        R_reta_w_1 = tb.cpx_interp(w_ref, S.freq_meshes[1].values(), S.R_reta_w[:, 1])

        testing.assert_allclose(R_reta_w_0, R_reta_w_ref[:, 0])
        testing.assert_allclose(R_reta_w_1, R_reta_w_ref[:, 1])

    def test_R_less_non_reg(self):
        fock, S = self.compute_nca()

        data = np.loadtxt("tests/data_ref/R_less_w.dat", dtype=complex)
        print(data.shape)
        w_ref = data[:, 0].real
        R_less_w_ref = data[:, 1:]

        R_less_w_0 = tb.cpx_interp(w_ref, S.freq_meshes[0].values(), S.R_less_w[:, 0])
        R_less_w_1 = tb.cpx_interp(w_ref, S.freq_meshes[1].values(), S.R_less_w[:, 1])

        testing.assert_allclose(R_less_w_0, R_less_w_ref[:, 0], atol=1e-10)
        testing.assert_allclose(R_less_w_1, R_less_w_ref[:, 1], atol=1e-10)

    def test_sanity_checks(self):
        beta = 3.0
        Ef = 0.3
        fock, S = self.compute_nca()

        ### check sanity check utility
        sanity_checks(S, fock)

        ### R & S

        ### Fourier transforms
        for i in range(4):
            w_ref, R_less_w_ref = fourier_transform(
                S.time_meshes[i], S.R_less[:, i], axis=0
            )
            testing.assert_allclose(w_ref.values(), S.freq_meshes[i].values())
            testing.assert_allclose(R_less_w_ref, S.R_less_w[:, i], atol=1e-4)

            _, R_grea_w_ref = fourier_transform(
                S.time_meshes[i], S.R_grea[:, i], axis=0
            )
            testing.assert_allclose(R_grea_w_ref, S.R_grea_w[:, i], atol=1e-4)

            _, S_less_w_ref = fourier_transform(
                S.time_meshes[i], S.S_less[:, i], axis=0
            )
            testing.assert_allclose(S_less_w_ref, S.S_less_w[:, i], atol=1e-4)

            _, S_grea_w_ref = fourier_transform(
                S.time_meshes[i], S.S_grea[:, i], axis=0
            )
            testing.assert_allclose(S_grea_w_ref, S.S_grea_w[:, i], atol=1e-4)

        ### symmetries: diagonal lessers and greaters are pure imaginary
        testing.assert_allclose(S.R_less_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.R_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.S_less_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.S_grea_w.real, 0.0, atol=1e-8)

        ### normalization
        idx0 = S.N // 2

        for k in range(4):
            self.assertAlmostEqual(S.R_grea[idx0, k], -1j)

        self.assertAlmostEqual(np.sum(S.R_less[idx0, :]), -4j, 2)

        ### Green functions

        m_grea, G_grea = fock.get_G_grea(0, S)
        m_less, G_less = fock.get_G_less(0, S)
        m_dos, Dos_w = fock.get_DOS(0, S)

        m_grea_w, G_grea_w = fourier_transform(m_grea, G_grea)
        m_less_w, G_less_w = fourier_transform(m_less, G_less)

        ### normalization and DoS
        Dos_w_ref = np.real(1j * (G_grea_w - G_less_w) / (2 * np.pi))
        testing.assert_allclose(Dos_w_ref, Dos_w, atol=1e-8)
        testing.assert_allclose(np.trapz(x=m_dos.values(), y=Dos_w), 1.0, atol=1e-6)

        ### Symmetries: diagonal lessers and greaters are pure imaginary and do not change sign
        testing.assert_allclose(G_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(G_less_w.real, 0.0, atol=1e-8)
        testing.assert_array_less(G_grea_w.imag, 1e-8)
        testing.assert_array_less(-G_less_w.imag, 1e-8)

        mask = np.abs(m_dos.values() - Ef) < 1.0
        np.testing.assert_allclose(
            G_less_w[mask] / Dos_w[mask] / (2j * np.pi),
            tb.fermi(m_dos.values()[mask], Ef, beta),
            atol=1e-2,
        )
        np.testing.assert_allclose(
            G_grea_w[mask] / Dos_w[mask] / (-2j * np.pi),
            tb.fermi(-m_dos.values()[mask], -Ef, beta),
            atol=1e-2,
        )

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

        S = SolverSteadyState(H_loc, time_mesh, hybs, [0, 3])

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

        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, beta, 0.0, time_mesh)

        fock = FermionicFockSpace(["up", "dn"])
        fock.add_bath(0, delta_grea, delta_less)
        fock.add_bath(1, delta_grea, delta_less)
        hybs = fock.generate_hybridizations()

        S = SolverSteadyState(H_loc, time_mesh, hybs, [0, 3])

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

        mesh, G_grea = fock.get_G_grea(0, S)
        G_grea = np.interp(times_ref, mesh.values(), G_grea)
        np.testing.assert_array_almost_equal(G_grea, G_grea_ref, 3)

        mesh, G_less = fock.get_G_less(0, S)
        G_less = np.interp(times_ref, mesh.values(), G_less)
        np.testing.assert_array_almost_equal(G_less, G_less_ref, 3)


class SolverSteadyStateInfiniteUTest(unittest.TestCase):
    def compute_nca(self):
        mesh = Mesh(100.0, 200001)
        # times = time_mesh.values()

        ### local model
        Gamma = 1.0
        eps = -1.0
        U = np.inf

        ### basis: 0, dn, up, updn
        H_loc = np.array([0.0, eps, eps])

        beta = 3.0
        Ef = 0.3
        D = 6.0

        ### Hybridization
        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, beta, Ef, mesh)

        fock = AIM_infinite_U()
        fock.add_bath(0, delta_grea, delta_less)
        fock.add_bath(1, delta_grea, delta_less)
        hybs = fock.generate_hybridizations()

        print(fock.basis())

        S = SolverSteadyState(H_loc, mesh, hybs, [0])

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
        for i in range(3):
            w_ref, R_less_w_ref = fourier_transform(
                S.time_meshes[i], S.R_less[:, i], axis=0
            )
            testing.assert_allclose(R_less_w_ref, S.R_less_w[:, i], atol=1e-4)

            _, R_grea_w_ref = fourier_transform(
                S.time_meshes[i], S.R_grea[:, i], axis=0
            )
            testing.assert_allclose(R_grea_w_ref, S.R_grea_w[:, i], atol=1e-4)

            _, S_less_w_ref = fourier_transform(
                S.time_meshes[i], S.S_less[:, i], axis=0
            )
            testing.assert_allclose(S_less_w_ref, S.S_less_w[:, i], atol=1e-4)

            _, S_grea_w_ref = fourier_transform(
                S.time_meshes[i], S.S_grea[:, i], axis=0
            )
            testing.assert_allclose(S_grea_w_ref, S.S_grea_w[:, i], atol=1e-4)

        ### symmetries: diagonal lessers and greaters are pure imaginary
        testing.assert_allclose(S.R_less_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.R_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.S_less_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.S_grea_w.real, 0.0, atol=1e-8)

        ### normalization
        idx0 = S.N // 2

        for k in range(3):
            self.assertAlmostEqual(S.R_grea[idx0, k], -1j)

        self.assertAlmostEqual(np.sum(S.R_less[idx0, :]), -3j, 2)

        ### Green functions

        m_grea, G_grea = fock.get_G_grea(0, S)
        m_less, G_less = fock.get_G_less(0, S)
        m_dos, Dos_w = fock.get_DOS(0, S)

        _, G_grea_w = fourier_transform(m_grea, G_grea)
        _, G_less_w = fourier_transform(m_less, G_less)

        ### normalization and DoS
        Dos_w_ref = np.real(1j * (G_grea_w - G_less_w) / (2 * np.pi))
        testing.assert_allclose(Dos_w_ref, Dos_w, atol=1e-8)
        # testing.assert_allclose(np.trapz(x=S.freqs, y=Dos_w), 1.0, atol=1e-6)

        ### Symmetries: diagonal lessers and greaters are pure imaginary and do not change sign
        testing.assert_allclose(G_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(G_less_w.real, 0.0, atol=1e-8)
        testing.assert_array_less(G_grea_w.imag, 1e-8)
        testing.assert_array_less(-G_less_w.imag, 1e-8)

        mask = np.abs(m_dos.values() - Ef) < 1.0
        np.testing.assert_allclose(
            G_less_w[mask] / Dos_w[mask] / (2j * np.pi),
            tb.fermi(m_dos.values()[mask], Ef, beta),
            atol=1e-2,
        )
        np.testing.assert_allclose(
            G_grea_w[mask] / Dos_w[mask] / (-2j * np.pi),
            tb.fermi(-m_dos.values()[mask], -Ef, beta),
            atol=1e-2,
        )


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
        inv_R0 = (
            mesh.adjoint().values()[None, :] - H_loc[:, None] + 2j * np.pi / decay_t
        )

        beta = 3.0
        Ef = 0.3
        D = 6.0

        ### Hybridization
        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, beta, Ef, mesh)

        fock = FermionicFockSpace(["up", "dn"])
        fock.add_bath(0, delta_grea, delta_less)
        fock.add_bath(1, delta_grea, delta_less)
        hybs = fock.generate_hybridizations()

        print(fock.basis())

        S = SolverSteadyState(inv_R0, mesh, hybs, [0, 3])

        S.greater_loop(plot=False, verbose=True)
        S.lesser_loop(verbose=True, max_iter=100)

        return fock, S

    def test_compare_types_of_input(self):
        time_mesh = Mesh(10.0, 100)
        w = time_mesh.adjoint().values()
        H_loc = np.array([0.0 - 0.1j, 2.0, 3.0 - 0.5j, -1.0])
        inv_R0 = [w + 0.1j, w - 2.0, w - 3.0 + 0.5j, w + 1.0]

        delta_grea = np.sin(w) + 1j * np.cos(w)
        delta_less = np.cos(w) + 1j * np.cos(w)

        fock = FermionicFockSpace(["up", "dn"])
        fock.add_bath(0, delta_grea, delta_less)
        fock.add_bath(1, delta_grea, delta_less)
        hybs = fock.generate_hybridizations()

        S_H = SolverSteadyState(H_loc, time_mesh, hybs, [0, 3])
        S_R = SolverSteadyState(inv_R0, time_mesh, hybs, [0, 3])

        testing.assert_allclose(S_H.inv_R0_reta_w, S_R.inv_R0_reta_w)

        S_H.initialize_grea()
        S_R.initialize_grea()

        testing.assert_allclose(S_H.R_reta_w, S_R.R_reta_w)

        S_H.fixed_pt_function_grea(S_H.R_reta_w)
        S_R.fixed_pt_function_grea(S_R.R_reta_w)

        testing.assert_allclose(S_H.R_reta_w, S_R.R_reta_w)
        testing.assert_allclose(S_H.R_grea, S_R.R_grea)
        testing.assert_allclose(S_H.S_reta_w, S_R.S_reta_w)
        testing.assert_allclose(S_H.S_grea, S_R.S_grea)

        S_H.initialize_less()
        S_R.initialize_less()

        testing.assert_allclose(S_H.R_less_w, S_R.R_less_w)
        testing.assert_array_equal(S_H.R_less_w, S_R.R_less_w)

        S_H.fixed_pt_function_less(S_H.R_less_w)
        S_R.fixed_pt_function_less(S_R.R_less_w)

        testing.assert_allclose(S_H.R_less_w, S_R.R_less_w)
        testing.assert_allclose(S_H.R_less, S_R.R_less)
        testing.assert_allclose(S_H.S_less_w, S_R.S_less_w)
        testing.assert_allclose(S_H.S_less, S_R.S_less)

    def test_sanity_checks(self):
        beta = 3.0
        Ef = 0.3
        fock, S = self.compute_nca()

        ### R & S

        ### Fourier transforms
        for i in range(4):
            w_ref, R_less_w_ref = fourier_transform(
                S.time_meshes[i], S.R_less[:, i], axis=0
            )
            testing.assert_allclose(R_less_w_ref, S.R_less_w[:, i], atol=1e-4)

            _, R_grea_w_ref = fourier_transform(
                S.time_meshes[i], S.R_grea[:, i], axis=0
            )
            testing.assert_allclose(R_grea_w_ref, S.R_grea_w[:, i], atol=1e-4)

            _, S_less_w_ref = fourier_transform(
                S.time_meshes[i], S.S_less[:, i], axis=0
            )
            testing.assert_allclose(S_less_w_ref, S.S_less_w[:, i], atol=1e-4)

            _, S_grea_w_ref = fourier_transform(
                S.time_meshes[i], S.S_grea[:, i], axis=0
            )
            testing.assert_allclose(S_grea_w_ref, S.S_grea_w[:, i], atol=1e-4)

        ### symmetries: diagonal lessers and greaters are pure imaginary
        testing.assert_allclose(S.R_less_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.R_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.S_less_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(S.S_grea_w.real, 0.0, atol=1e-8)

        ### normalization
        idx0 = S.N // 2

        for k in range(4):
            self.assertAlmostEqual(S.R_grea[idx0, k], -1j)

        self.assertAlmostEqual(np.sum(S.R_less[idx0, :]), -4j, 2)

        ### Green functions

        m_grea, G_grea = fock.get_G_grea(0, S)
        m_less, G_less = fock.get_G_less(0, S)
        m_dos, Dos_w = fock.get_DOS(0, S)

        _, G_grea_w = fourier_transform(m_grea, G_grea)
        _, G_less_w = fourier_transform(m_less, G_less)

        ### normalization and DoS
        Dos_w_ref = np.real(1j * (G_grea_w - G_less_w) / (2 * np.pi))
        testing.assert_allclose(Dos_w_ref, Dos_w, atol=1e-8)
        testing.assert_allclose(np.trapz(x=m_dos.values(), y=Dos_w), 1.0, atol=1e-6)

        ### Symmetries: diagonal lessers and greaters are pure imaginary and do not change sign
        testing.assert_allclose(G_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(G_less_w.real, 0.0, atol=1e-8)
        testing.assert_array_less(G_grea_w.imag, 1e-8)
        testing.assert_array_less(-G_less_w.imag, 1e-8)

        mask = np.abs(m_dos.values() - Ef) < 1.0
        np.testing.assert_allclose(
            G_less_w[mask] / Dos_w[mask] / (2j * np.pi),
            tb.fermi(m_dos.values()[mask], Ef, beta),
            atol=1e-2,
        )
        np.testing.assert_allclose(
            G_grea_w[mask] / Dos_w[mask] / (-2j * np.pi),
            tb.fermi(-m_dos.values()[mask], -Ef, beta),
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
        inv_R0 = (
            time_mesh.adjoint().values()[None, :]
            - H_loc[:, None]
            + 2j * np.pi / decay_t
        )

        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, beta, 0.0, time_mesh)

        fock = FermionicFockSpace(["up", "dn"])
        fock.add_bath(0, delta_grea, delta_less)
        fock.add_bath(1, delta_grea, delta_less)
        hybs = fock.generate_hybridizations()

        S = SolverSteadyState(inv_R0, time_mesh, hybs, [0, 3])

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

        m, G_grea = fock.get_G_grea(0, S)
        G_grea = np.interp(times_ref, m.values(), G_grea)
        np.testing.assert_array_almost_equal(G_grea, G_grea_ref, 2)

        m, G_less = fock.get_G_less(0, S)
        G_less = np.interp(times_ref, m.values(), G_less)
        np.testing.assert_array_almost_equal(G_less, G_less_ref, 2)


if __name__ == "__main__":
    unittest.main()
