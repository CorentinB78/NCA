import unittest
from nca.utilities import *
import toolbox as tb
from matplotlib import pyplot as plt

class FourierTest(unittest.TestCase):

    def testA(self):
        m = Mesh(3., 101)
        print(len(m))
        f = np.exp(-(m.values() - 1.j - 0.2) ** 2)

        mg, g = inv_fourier_transform(m, f)
        mh, h = fourier_transform(mg, g)

        # mg, g = fourier_transform(m, f)
        # mh, h = inv_fourier_transform(mg, g)

        tb.cpx_plot(mh.values(), h, label='new')
        tb.cpx_plot(m.values(), f, label='orig')
        plt.axvline(0.)
        plt.legend()
        plt.show()

        tb.cpx_plot(mg.values(), g, label='tr')
        tb.cpx_plot(*tb.inv_fourier_transform(m.values(), f), ls='--', label='old')

        plt.legend()
        plt.show()

class WindowTest(unittest.TestCase):

    def test_A(self):
        m = Mesh(10., 101)
        window = planck_taper_window(m, 5., 2.)
        Wm = 4.
        Wp = 6.

        for i in [25, 30, 75, 80]:
            x = m.values()[i]
            ref = 1. / (1. + np.exp((Wp - Wm) / (Wp - np.abs(x)) - (Wp - Wm) / (np.abs(x) - Wm)))
            self.assertAlmostEqual(window[i], ref)

if __name__ == '__main__':
    unittest.main()