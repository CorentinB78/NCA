import unittest
import numpy as np
from numpy import testing
from scipy import integrate
from nca.function_tools import *
from nca.hybridizations import *
from nca.solver_steady_state import *
from nca.state_space import *


class TestGreenFunction(unittest.TestCase):
    def test_greater(self):
        mesh = Mesh(10.0, 100)
        x = mesh.values()
        R_grea = np.array([np.cos(x), np.sin(x), np.cos(x + 0.5), np.sin(x + 0.5)]).T
        R_less = np.array([np.cos(x - 0.7), np.sin(x - 0.7), np.cos(x), np.sin(x)]).T

        s = StateSpace(2)
        m, G = greater_gf(0, s, mesh, R_grea, R_less, 3.0)
        G_ref = 1j * (np.cos(-x - 0.7) * np.sin(x) + np.cos(-x) * np.sin(x + 0.5)) / 3.0

        self.assertIs(m, mesh)
        testing.assert_allclose(G, G_ref)

    def test_lesser(self):
        mesh = Mesh(10.0, 100)
        x = mesh.values()
        R_grea = np.array([np.cos(x), np.sin(x), np.cos(x + 0.5), np.sin(x + 0.5)]).T
        R_less = np.array([np.cos(x - 0.7), np.sin(x - 0.7), np.cos(x), np.sin(x)]).T

        s = StateSpace(2)
        m, G = lesser_gf(0, s, mesh, R_grea, R_less, 3.0)
        G_ref = (
            -1j * (np.sin(x - 0.7) * np.cos(-x) + np.sin(x) * np.cos(-x + 0.5)) / 3.0
        )

        self.assertIs(m, mesh)
        testing.assert_allclose(G, G_ref)


class TestParams1(unittest.TestCase):

    S = None

    @classmethod
    def setUpClass(cls):
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

        S = SolverSteadyState(2, H_loc, mesh)
        S.add_bath(0, delta_grea, delta_less)
        S.add_bath(1, delta_grea, delta_less)

        S.greater_loop(plot=False, verbose=True)
        S.lesser_loop(plot=False, verbose=True, max_iter=20)

        cls.S = S

    def test_R_reta_non_reg(self):
        S = self.S

        data = np.loadtxt("tests/data_ref/R_reta_w.dat", dtype=complex)
        w_ref = data[:, 0].real
        R_reta_w_ref = data[:, 1:]

        R_grea_w_0 = tb.cpx_interp(w_ref, S.freq_mesh.values(), S.get_R_grea_w()[:, 0])
        R_grea_w_1 = tb.cpx_interp(w_ref, S.freq_mesh.values(), S.get_R_grea_w()[:, 1])

        testing.assert_allclose(
            1j * R_grea_w_0, 2j * R_reta_w_ref[:, 0].imag, atol=1e-4, rtol=1e-2
        )
        testing.assert_allclose(
            1j * R_grea_w_1, 2j * R_reta_w_ref[:, 1].imag, atol=1e-4, rtol=1e-2
        )

    def test_R_less_non_reg(self):
        S = self.S

        data = np.loadtxt("tests/data_ref/R_less_w.dat", dtype=complex)
        print(data.shape)
        w_ref = data[:, 0].real
        R_less_w_ref = data[:, 1:]

        R_less_w_0 = tb.cpx_interp(w_ref, S.freq_mesh.values(), S.get_R_less_w()[:, 0])
        R_less_w_1 = tb.cpx_interp(w_ref, S.freq_mesh.values(), S.get_R_less_w()[:, 1])

        testing.assert_allclose(
            1j * R_less_w_0, R_less_w_ref[:, 0], atol=1e-4, rtol=1e-2
        )
        testing.assert_allclose(
            1j * R_less_w_1, R_less_w_ref[:, 1], atol=1e-4, rtol=1e-2
        )

    def test_RS_symmetries(self):
        S = self.S

        ### symmetries: diagonal lessers and greaters are negative (pure imaginary)
        testing.assert_array_less(S.get_R_less_w(), 1e-8)
        testing.assert_array_less(S.get_R_grea_w(), 1e-8)
        testing.assert_array_less(S.get_S_less_w(), 1e-8)
        testing.assert_array_less(S.get_S_grea_w(), 1e-8)

    def test_R_normalization(self):
        S = self.S
        idx0 = S.N // 2

        for k in range(4):
            self.assertAlmostEqual(S.get_R_grea()[idx0, k], -1j, 4)

        self.assertAlmostEqual(np.sum(S.get_R_less()[idx0, :]), -4j, 4)

    def test_get_normalization_error(self):
        norm_err = self.S.get_normalization_error()
        testing.assert_allclose(norm_err, 0.0, atol=1e-4)

    def test_green_functions(self):
        S = self.S

        m_grea, G_grea = S.get_G_grea(0)
        m_less, G_less = S.get_G_less(0)
        m_dos, Dos_w = S.get_DOS(0)

        m_grea_w, G_grea_w = fourier_transform(m_grea, G_grea)
        m_less_w, G_less_w = fourier_transform(m_less, G_less)

        ### normalization and DoS
        Dos_w_ref = np.real(1j * (G_grea_w - G_less_w) / (2 * np.pi))
        testing.assert_allclose(Dos_w_ref, Dos_w, atol=1e-8)
        testing.assert_allclose(integrate.simpson(x=m_dos.values(), y=Dos_w), 1.0, atol=1e-4)

        ### Symmetries: diagonal lessers and greaters are pure imaginary and do not change sign
        testing.assert_allclose(G_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(G_less_w.real, 0.0, atol=1e-8)
        testing.assert_array_less(G_grea_w.imag, 1e-8)
        testing.assert_array_less(-G_less_w.imag, 1e-8)

    def test_fluctuation_dissipation_thm(self):
        S = self.S
        beta = 3.0
        Ef = 0.3

        m_grea, G_grea = S.get_G_grea(0)
        m_less, G_less = S.get_G_less(0)
        m_dos, Dos_w = S.get_DOS(0)

        _, G_grea_w = fourier_transform(m_grea, G_grea)
        _, G_less_w = fourier_transform(m_less, G_less)

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

    def test_get_R_reta_w(self):
        R_reta_w = self.S.get_R_reta_w()
        R_grea_w = self.S.get_R_grea_w()

        testing.assert_allclose(2 * np.imag(R_reta_w), R_grea_w, atol=1e-8)

    def test_get_S_reta_w(self):
        S_reta_w = self.S.get_S_reta_w()
        S_grea_w = self.S.get_S_grea_w()

        testing.assert_allclose(2 * np.imag(S_reta_w), S_grea_w, atol=1e-8)


class TestSelfEnergy(unittest.TestCase):
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
            S.get_S_grea()[:, 0],
            1j * np.conj(Delta_less_up) * S.get_R_grea()[:, 1]
            + 1j * np.conj(Delta_less_dn) * S.get_R_grea()[:, 2],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.get_S_grea()[:, 1],
            1j * Delta_grea_up * S.get_R_grea()[:, 0]
            + 1j * np.conj(Delta_less_dn) * S.get_R_grea()[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.get_S_grea()[:, 2],
            1j * Delta_grea_dn * S.get_R_grea()[:, 0]
            + 1j * np.conj(Delta_less_up) * S.get_R_grea()[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.get_S_grea()[:, 3],
            1j * Delta_grea_dn * S.get_R_grea()[:, 1]
            + 1j * Delta_grea_up * S.get_R_grea()[:, 2],
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
            S.get_S_less()[:, 0],
            -1j * np.conj(Delta_grea_up) * S.get_R_less()[:, 1]
            - 1j * np.conj(Delta_grea_dn) * S.get_R_less()[:, 2],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.get_S_less()[:, 1],
            -1j * Delta_less_up * S.get_R_less()[:, 0]
            - 1j * np.conj(Delta_grea_dn) * S.get_R_less()[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.get_S_less()[:, 2],
            -1j * Delta_less_dn * S.get_R_less()[:, 0]
            - 1j * np.conj(Delta_grea_up) * S.get_R_less()[:, 3],
            10,
        )

        np.testing.assert_array_almost_equal(
            S.get_S_less()[:, 3],
            -1j * Delta_less_dn * S.get_R_less()[:, 1]
            - 1j * Delta_less_up * S.get_R_less()[:, 2],
            10,
        )


class TestParamsRenaud(unittest.TestCase):
    def setUp(self):
        beta = 1.0
        mu = 0.5
        U = 1.0
        D = 10.0
        Gamma = 1.0

        time_mesh = Mesh(1000.0, 100001)

        ### basis: 0, up, dn, updn
        H_loc = np.array([0.0, -mu, -mu, -2 * mu + U])

        delta_less, delta_grea = make_Delta_semicirc(Gamma, D, beta, 0.0, time_mesh)

        S = SolverSteadyState(2, H_loc, time_mesh)
        S.add_bath(0, delta_grea, delta_less)
        S.add_bath(1, delta_grea, delta_less)

        S.greater_loop(tol=1e-5, verbose=True)
        S.lesser_loop(tol=1e-5, verbose=True)

        self.S = S

    def test_values(self):
        S = self.S

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

        mesh, G_grea = S.get_G_grea(0)
        G_grea = np.interp(times_ref, mesh.values(), G_grea)
        np.testing.assert_array_almost_equal(G_grea, G_grea_ref, 3)

        mesh, G_less = S.get_G_less(0)
        G_less = np.interp(times_ref, mesh.values(), G_less)
        np.testing.assert_array_almost_equal(G_less, G_less_ref, 3)


class TestInfiniteU(unittest.TestCase):

    S = None

    @classmethod
    def setUpClass(cls):
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

        S = AIM_infinite_U(H_loc, mesh)
        S.add_bath(0, delta_grea, delta_less)
        S.add_bath(1, delta_grea, delta_less)

        S.greater_loop(plot=False, verbose=True)
        S.lesser_loop(plot=False, verbose=True, max_iter=20)

        cls.S = S

    @unittest.skip("sanity check should be removed")
    def test_sanity_checks(self):
        sanity_checks(self.S, self.fock)

    def test_RS_symmetries(self):
        S = self.S

        ### symmetries: diagonal lessers and greaters are negative (pure imaginary)
        testing.assert_array_less(S.get_R_less_w(), 1e-8)
        testing.assert_array_less(S.get_R_grea_w(), 1e-8)
        testing.assert_array_less(S.get_S_less_w(), 1e-8)
        testing.assert_array_less(S.get_S_grea_w(), 1e-8)

    def test_R_normalization(self):
        S = self.S
        idx0 = S.N // 2

        for k in range(3):
            self.assertAlmostEqual(S.get_R_grea()[idx0, k], -1j, 4)

        self.assertAlmostEqual(np.sum(S.get_R_less()[idx0, :]), -3j, 4)

    def test_green_functions(self):
        S = self.S

        m_grea, G_grea = S.get_G_grea(0)
        m_less, G_less = S.get_G_less(0)
        m_dos, Dos_w = S.get_DOS(0)

        _, G_grea_w = fourier_transform(m_grea, G_grea)
        _, G_less_w = fourier_transform(m_less, G_less)

        ### normalization and DoS
        Dos_w_ref = np.real(1j * (G_grea_w - G_less_w) / (2 * np.pi))
        testing.assert_allclose(Dos_w_ref, Dos_w, atol=1e-8)
        # testing.assert_allclose(integrate.simpson(x=S.freqs, y=Dos_w), 1.0, atol=1e-6)

        ### Symmetries: diagonal lessers and greaters are pure imaginary and do not change sign
        testing.assert_allclose(G_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(G_less_w.real, 0.0, atol=1e-8)
        testing.assert_array_less(G_grea_w.imag, 1e-8)
        testing.assert_array_less(-G_less_w.imag, 1e-8)

    def test_fluctuation_dissipation_thm(self):
        S = self.S
        beta = 3.0
        Ef = 0.3

        m_grea, G_grea = S.get_G_grea(0)
        m_less, G_less = S.get_G_less(0)
        m_dos, Dos_w = S.get_DOS(0)

        _, G_grea_w = fourier_transform(m_grea, G_grea)
        _, G_less_w = fourier_transform(m_less, G_less)

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


class TestExtendedR0(unittest.TestCase):

    S = None

    @classmethod
    def setUpClass(cls):
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

        S = SolverSteadyState(2, inv_R0, mesh)

        S.add_bath(0, delta_grea, delta_less)
        S.add_bath(1, delta_grea, delta_less)

        S.greater_loop(plot=False, verbose=True)
        S.lesser_loop(verbose=True, max_iter=100)

        cls.S = S

    def test_RS_symmetries(self):
        S = self.S

        ### symmetries: diagonal lessers and greaters are negative (pure imaginary)
        testing.assert_array_less(S.get_R_less_w(), 1e-8)
        testing.assert_array_less(S.get_R_grea_w(), 1e-8)
        testing.assert_array_less(S.get_S_less_w(), 1e-8)
        testing.assert_array_less(S.get_S_grea_w(), 1e-8)

    def test_R_normalization(self):
        S = self.S
        idx0 = S.N // 2

        for k in range(4):
            self.assertAlmostEqual(S.get_R_grea()[idx0, k], -1j, 4)

        self.assertAlmostEqual(np.sum(S.get_R_less()[idx0, :]), -4j, 4)

    def test_green_functions(self):
        S = self.S

        m_grea, G_grea = S.get_G_grea(0)
        m_less, G_less = S.get_G_less(0)
        m_dos, Dos_w = S.get_DOS(0)

        _, G_grea_w = fourier_transform(m_grea, G_grea)
        _, G_less_w = fourier_transform(m_less, G_less)

        ### normalization and DoS
        Dos_w_ref = np.real(1j * (G_grea_w - G_less_w) / (2 * np.pi))
        testing.assert_allclose(Dos_w_ref, Dos_w, atol=1e-8)
        testing.assert_allclose(integrate.simpson(x=m_dos.values(), y=Dos_w), 1.0, atol=1e-4)

        ### Symmetries: diagonal lessers and greaters are pure imaginary and do not change sign
        testing.assert_allclose(G_grea_w.real, 0.0, atol=1e-8)
        testing.assert_allclose(G_less_w.real, 0.0, atol=1e-8)
        testing.assert_array_less(G_grea_w.imag, 1e-8)
        testing.assert_array_less(-G_less_w.imag, 1e-8)

    def test_fluctuation_dissipation_thm(self):
        S = self.S
        beta = 3.0
        Ef = 0.3

        m_grea, G_grea = S.get_G_grea(0)
        m_less, G_less = S.get_G_less(0)
        m_dos, Dos_w = S.get_DOS(0)

        _, G_grea_w = fourier_transform(m_grea, G_grea)
        _, G_less_w = fourier_transform(m_less, G_less)

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


class TestExtendedR0ParamsRenaud(unittest.TestCase):
    def setUp(self):
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

        S = SolverSteadyState(2, inv_R0, time_mesh)
        S.add_bath(0, delta_grea, delta_less)
        S.add_bath(1, delta_grea, delta_less)

        S.greater_loop(tol=1e-10, verbose=True, plot=False)
        S.lesser_loop(tol=1e-10, verbose=True)
        # plt.plot(S.times, S.get_R_less()[:, 2].real)
        # plt.plot(S.times, S.get_R_less()[:, 2].imag)
        # plt.show()

        self.S = S

    def test_values(self):
        S = self.S

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

        m, G_grea = S.get_G_grea(0)
        G_grea = np.interp(times_ref, m.values(), G_grea)
        np.testing.assert_array_almost_equal(G_grea, G_grea_ref, 2)

        m, G_less = S.get_G_less(0)
        G_less = np.interp(times_ref, m.values(), G_less)
        np.testing.assert_array_almost_equal(G_less, G_less_ref, 2)


class TestExtendedR0VsHloc(unittest.TestCase):
    @unittest.skip("Test will be removed")
    def test_compare_types_of_input(self):
        time_mesh = Mesh(10.0, 100)
        w = time_mesh.adjoint().values()
        H_loc = np.array([0.0 - 0.1j, 2.0, 3.0 - 0.5j, -1.0])
        inv_R0 = [w + 0.1j, w - 2.0, w - 3.0 + 0.5j, w + 1.0]

        delta_grea = np.sin(w) + 1j * np.cos(w)
        delta_less = np.cos(w) + 1j * np.cos(w)

        S_H = SolverSteadyState(2, H_loc, time_mesh)
        S_H.add_bath(0, delta_grea, delta_less)
        S_H.add_bath(1, delta_grea, delta_less)

        S_R = SolverSteadyState(2, inv_R0, time_mesh)
        S_R.add_bath(0, delta_grea, delta_less)
        S_R.add_bath(1, delta_grea, delta_less)

        testing.assert_allclose(S_H.core.inv_R0_reta_w, S_R.core.inv_R0_reta_w)

        S_H.core.initialize_grea()
        S_R.core.initialize_grea()

        testing.assert_allclose(S_H.core.R_grea_w, S_R.core.R_grea_w)

        S_H.core.fixed_pt_function_grea(S_H.core.R_grea_w)
        S_R.core.fixed_pt_function_grea(S_R.core.R_grea_w)

        testing.assert_allclose(S_H.core.R_grea_w, S_R.core.R_grea_w)


if __name__ == "__main__":
    unittest.main()
