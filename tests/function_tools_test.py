import unittest
from nca.function_tools import *
import numpy as np
from numpy import testing


class TestRegularNumber(unittest.TestCase):
    def test_odd_regular(self):
        self.assertEqual(next_odd_regular(0), 1)
        self.assertEqual(next_odd_regular(2), 3)
        self.assertEqual(next_odd_regular(3), 3)
        self.assertEqual(next_odd_regular(4), 5)
        self.assertEqual(next_odd_regular(5), 5)
        self.assertEqual(next_odd_regular(10), 15)
        self.assertEqual(next_odd_regular(32), 45)
        self.assertEqual(next_odd_regular(381), 405)
        self.assertEqual(next_odd_regular(405), 405)


class MeshTest(unittest.TestCase):
    def test_regular_spacing(self):
        m = Mesh(50.0, 300)
        testing.assert_allclose(np.diff(m.values()), m.delta)

    def test_zero(self):
        m = Mesh(50.0, 300)
        L = len(m)
        self.assertEqual(m.values()[L // 2], 0.0)

    def test_max_values(self):
        m = Mesh(50.0, 300)
        self.assertEqual(m.values()[0], -50.0)
        self.assertEqual(m.values()[-1], 50.0)
        self.assertEqual(m.xmin, -50.0)
        self.assertEqual(m.xmax, 50.0)

    def test_length(self):
        m = Mesh(50.0, 300)
        self.assertEqual(len(m), len(m.values()))
        self.assertEqual(len(m) % 2, 1)
        self.assertGreater(len(m), 300)

    def test_adjoint(self):
        m = Mesh(50.0, 300)
        m_adj = m.adjoint()
        L = len(m)
        self.assertEqual(L, len(m_adj))
        self.assertAlmostEqual(m_adj.xmax / m_adj.delta, m.xmax / m.delta)
        self.assertAlmostEqual(m.delta * m_adj.delta, 2 * np.pi / L)

    def test_adjoint_of_adjoint(self):
        m = Mesh(50.0, 300)
        m_adj = m.adjoint()
        m_adj_2 = m_adj.adjoint()

        testing.assert_allclose(m.values(), m_adj_2.values())

    def test_pt_on_value(self):
        x = 67.34
        m = Mesh(100.0, 101, pt_on_value=x)

        idx = np.argmin(np.abs(m.values() - x))
        self.assertAlmostEqual(x, m.values()[idx])

        val_unmod = np.linspace(-100.0, 100.0, 101)
        idx = np.argmin(np.abs(val_unmod - x))
        self.assertNotEqual(x, val_unmod[idx])

    def test_no_adjust(self):

        # even mesh
        with self.assertRaises(ValueError):
            m = Mesh(10.0, 100, adjust_nr_samples=False)

        # given length
        m = Mesh(10.0, 101, adjust_nr_samples=False)
        self.assertEqual(len(m), 101)

        # adjoint
        m_adj = m.adjoint()
        L = len(m)
        self.assertEqual(L, len(m_adj))
        self.assertAlmostEqual(m_adj.xmax / m_adj.delta, m.xmax / m.delta)
        self.assertAlmostEqual(m.delta * m_adj.delta, 2 * np.pi / L)


class TestInterp(unittest.TestCase):
    def test_linear_complex(self):
        x = Mesh(10, 1000)

        def f(x):
            return np.cos(x) + 1j * np.sin(x + 0.5)

        x2 = Mesh(7.0, 10)
        y2 = interp(x2, x, f(x.values()), kind="linear")

        testing.assert_allclose(y2, f(x2.values()), atol=1e-4)

    def test_cubic_complex(self):
        x = Mesh(10, 100)

        def f(x):
            return np.cos(x) + 1j * np.sin(x + 0.5)

        x2 = Mesh(7.0, 10)
        y2 = interp(x2, x, f(x.values()), kind="cubic")

        testing.assert_allclose(y2, f(x2.values()), atol=1e-4)

    def test_extrapol(self):
        x = Mesh(10.0, 100)

        def f(x):
            return np.cos(x) + 1j * np.sin(x + 0.5)

        x2 = Mesh(15.0, 10)
        y2 = interp(x2, x, f(x.values()))
        mask = np.abs(x2.values()) > 10.0

        testing.assert_array_equal(y2[mask], 0.0)


class TestFunctions(unittest.TestCase):
    def test_product(self):
        m_a = Mesh(1.0, 1000)
        m_b = Mesh(3.0, 1000)
        f_a = np.cos(m_a.values())
        f_b = np.sin(m_b.values())

        m, f = product_functions(m_a, f_a, m_b, f_b)

        self.assertIs(m, m_a)
        testing.assert_allclose(
            f, np.cos(m_a.values()) * np.sin(m_a.values()), atol=1e-5
        )

    def test_sum(self):
        m_a = Mesh(1.0, 1000)
        m_b = Mesh(3.0, 1000)
        f_a = np.exp(-np.abs(m_a.values()) * 10)
        f_b = np.sin(m_b.values())

        m, f = sum_functions(m_a, f_a, m_b, f_b)

        self.assertIs(m, m_b)
        testing.assert_allclose(
            f, np.sin(m_b.values()) + np.exp(-np.abs(m_b.values()) * 10.0), atol=1e-4
        )


class FourierTest(unittest.TestCase):
    def test_direct(self):
        times = Mesh(100.0, 1001)
        f_t = np.exp(-((times.values() + 1.0) ** 2))

        freqs, f_w = fourier_transform(times, f_t)

        f_w_ref = np.sqrt(np.pi) * np.exp(
            -freqs.values() ** 2 / 4.0 - 1.0j * freqs.values()
        )

        np.testing.assert_allclose(f_w, f_w_ref, atol=1e-10)

    def test_inverse(self):
        freqs = Mesh(100.0, 1001)
        f_w = np.sqrt(np.pi) * np.exp(
            -freqs.values() ** 2 / 4.0 - 1.0j * freqs.values()
        )

        times, f_t = inv_fourier_transform(freqs, f_w)

        f_t_ref = np.exp(-((times.values() + 1.0) ** 2))

        np.testing.assert_allclose(f_t, f_t_ref, atol=1e-10)


class WindowTest(unittest.TestCase):
    def test_A(self):
        m = Mesh(10.0, 10001)
        window = planck_taper_window(m, 5.0, 2.0)

        x = m.values()
        mask_in = np.abs(x) < 4.0
        mask_out = np.abs(x) > 6.0

        np.testing.assert_allclose(window[mask_in], 1.0)
        np.testing.assert_allclose(window[mask_out], 0.0)

        der = np.diff(window) / np.diff(x)
        x_der = (x[1:] + x[:-1]) / 2.0
        der2 = np.diff(der) / np.diff(x_der)

        np.testing.assert_array_less(np.abs(der), 2.0)
        np.testing.assert_array_less(np.abs(der2), 6.0)


if __name__ == "__main__":
    unittest.main()
