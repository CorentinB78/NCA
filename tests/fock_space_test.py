import unittest
from nca.fock_space import is_orb_in_state, states_containing
import numpy as np


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


if __name__ == "__main__":
    unittest.main()
