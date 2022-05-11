import unittest
from nca.fock_space import is_orb_in_state, states_containing, greater_gf, lesser_gf
from nca.function_tools import Mesh
import numpy as np
from numpy import testing


class TestOrbitals(unittest.TestCase):
    def test_orbital_in_state(self):

        N = 2
        # orbitals: 0 = up, 1 = down,
        # states: 0 = empty, 1 = up, 2 = down, 3 = both

        self.assertEqual(is_orb_in_state(1, 0), False)
        self.assertEqual(is_orb_in_state(0, 0), False)
        self.assertEqual(is_orb_in_state(0, 1), True)
        self.assertEqual(is_orb_in_state(1, 1), False)
        self.assertEqual(is_orb_in_state(0, 2), False)
        self.assertEqual(is_orb_in_state(1, 2), True)
        self.assertEqual(is_orb_in_state(0, 3), True)
        self.assertEqual(is_orb_in_state(1, 3), True)

        np.testing.assert_array_equal(states_containing(0, N), ([1, 3], [0, 2]))
        np.testing.assert_array_equal(states_containing(1, N), ([2, 3], [0, 1]))


class TestGreenFunction(unittest.TestCase):
    def test_greater(self):
        mesh = Mesh(10.0, 100)
        x = mesh.values()
        R_grea = np.array([np.cos(x), np.sin(x), np.cos(x + 0.5), np.sin(x + 0.5)]).T
        R_less = np.array([np.cos(x - 0.7), np.sin(x - 0.7), np.cos(x), np.sin(x)]).T

        m, G = greater_gf(0, 2, [mesh] * 4, R_grea, R_less, 3.0)
        G_ref = 1j * (np.cos(-x - 0.7) * np.sin(x) + np.cos(-x) * np.sin(x + 0.5)) / 3.0

        self.assertIs(m, mesh)
        testing.assert_allclose(G, G_ref)

    def test_lesser(self):
        mesh = Mesh(10.0, 100)
        x = mesh.values()
        R_grea = np.array([np.cos(x), np.sin(x), np.cos(x + 0.5), np.sin(x + 0.5)]).T
        R_less = np.array([np.cos(x - 0.7), np.sin(x - 0.7), np.cos(x), np.sin(x)]).T

        m, G = lesser_gf(0, 2, [mesh] * 4, R_grea, R_less, 3.0)
        G_ref = (
            -1j * (np.sin(x - 0.7) * np.cos(-x) + np.sin(x) * np.cos(-x + 0.5)) / 3.0
        )

        self.assertIs(m, mesh)
        testing.assert_allclose(G, G_ref)


if __name__ == "__main__":
    unittest.main()
