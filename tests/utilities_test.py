import unittest
from nca.utilities import *
import toolbox as tb
from matplotlib import pyplot as plt


class MeshTest(unittest.TestCase):
    def test_pt_on_value(self):
        x = 67.34
        m = Mesh(100.0, 101, pt_on_value=x)

        idx = np.argmin(np.abs(m.values() - x))
        self.assertAlmostEqual(x, m.values()[idx])

        val_unmod = np.linspace(-100.0, 100.0, 101)
        idx = np.argmin(np.abs(val_unmod - x))
        self.assertNotEqual(x, val_unmod[idx])


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
        m = Mesh(10.0, 101)
        window = planck_taper_window(m, 5.0, 2.0)
        Wm = 4.0
        Wp = 6.0

        for i in [25, 30, 75, 80]:
            x = m.values()[i]
            ref = 1.0 / (
                1.0
                + np.exp((Wp - Wm) / (Wp - np.abs(x)) - (Wp - Wm) / (np.abs(x) - Wm))
            )
            self.assertAlmostEqual(window[i], ref)


if __name__ == "__main__":
    unittest.main()
