import numpy as np
from .function_tools import fourier_transform, inv_fourier_transform, AlpertMeshFunction, alpert_fourier_transform, inv_ft_to_alpert
from scipy import integrate


class CoreSolverSteadyState:
    # TODO: implement non-diagonal hybridization functions & local Hamiltonian
    def __init__(self, local_evol, time_mesh, list_even_states, list_odd_states, M, order):
        """
        Non-Crossing Approximation (NCA) solver for steady states in real frequencies --- core functions.

        For now only diagonal hybridizations and local hamiltonians are supported.

        Arguments:
            local_evol -- list of local evolution for each state. A local evolution can be a complex number representing energy and damping (negative imag part), or the values of 1/R_0^{reta}(w) on the frequency mesh adjoint to `time_mesh`.
            time_mesh -- an instance of `Mesh` for time coordinates
            list_even_states -- list of int representing the even states
            list_odd_states -- list of int representing the odd states

        Attributes:
            hybridizations -- dictionnary describing hybridization processes. Keys are the initial states. Values are lists of all processes starting at the same state. Each process is a tuple (intermediate_state, delta_grea, delta_less) where delta_grea/less are 1D arrays containing hybridization functions (as sampled on `time_mesh`). delta_grea is the one participating to the greater SE, while delta_less is for the lesser SE. The process changes the local system from the initial state to the intermediate state then back to the starting state. Conjugate processes are not added automatically.
        """
        # TODO: sanity checks
        assert len(list_even_states) + len(list_odd_states) == len(local_evol)

        self.N = len(time_mesh)
        N = self.N
        self.D = len(local_evol)
        self.Z_loc = self.D
        self.M = M
        self.order = order

        self.even_states = list_even_states
        self.odd_states = list_odd_states

        self.D_half = max(len(self.even_states), len(self.odd_states))

        self.is_even_state = np.array(
            [(s in list_even_states) for s in range(self.D)], dtype=bool
        )

        self.time_mesh = time_mesh
        self.freq_mesh = time_mesh.adjoint()

        self.inv_R0_reta_w = np.empty((N, self.D), dtype=complex)
        for s, g in enumerate(local_evol):
            if isinstance(g, complex) or isinstance(g, float):
                self.inv_R0_reta_w[:, s] = self.freq_mesh.values() - g
            else:
                self.inv_R0_reta_w[:, s] = g

        self.R_grea_w = np.zeros((N, self.D), dtype=float)  # imaginary part only
        self.R_less_w = np.zeros((N, self.D), dtype=float)  # imaginary part only

        self.nr_grea_feval = 0
        self.nr_less_feval = 0

        self.state_parity_table = np.empty(self.D, dtype=int)
        k_odd = 0
        k_even = 0
        for i in range(self.D):
            if i in self.even_states:
                self.state_parity_table[i] = k_even
                k_even += 1
            else:
                self.state_parity_table[i] = k_odd
                k_odd += 1

        self.normalization_error = []

    ### Initial guesses ###

    def initialize_grea(self):
        """
        Make a first guess for R^> based on magnitude of hybridizations.

        Populates R_grea_w
        """
        even = self.is_even_state

        delta_magn = 0.0

        for a in self.even_states:
            for b, delta, _ in self.hybridizations[a]:
                delta_magn += np.abs(delta.values_left[0]) ** 2
        delta_magn = np.sqrt(delta_magn)

        self.R_grea_w[:, even] = np.imag(
            2.0 / (self.inv_R0_reta_w[:, even] + 1.0j * delta_magn)
        )

    def initialize_less(self):
        """
        Make a first guess for R^< based on magnitude of hybridizations.

        Populates R_less_w
        """
        even = self.is_even_state

        delta_magn = 0.0
        idx0 = self.N // 2

        for a in self.even_states:
            for b, _, delta in self.hybridizations[a]:
                delta_magn += np.abs(delta[idx0]) ** 2
        delta_magn = np.sqrt(delta_magn)

        for i in range(self.D):
            if even[i]:
                self.R_less_w[:, i] = np.imag(
                    1.0 / (self.inv_R0_reta_w[:, i] + 1.0j * delta_magn)
                )

        self.normalize_less_w()

    ### Iteration functions

    def _self_energy_grea(self, parity_flag, R_grea):
        """
        Compute S^> from R^> in time domain.

        Arguments:
            parity_flag -- boolean, True for even states, False for odd states
            R_grea -- 2D array (times, states) for R^> for the complementary states

        Returns:
            2D array (times, states)
        """
        states = self.even_states if parity_flag else self.odd_states

        S_grea = [R_grea[0].get_empty_duplicate() for _ in states]

        for k, a in enumerate(states):
            for b, delta, _ in self.hybridizations[a]:
                S_grea[k] += delta * R_grea[self.state_parity_table[b]]

            S_grea[k] *= 1j

        return S_grea

    def iteration_grea(self, parity_flag, M, order):
        """
        Perform one greater iteration.

        If parity flag is True, use R^> in frequency domain of odd states to update the same quantity for even states, by going to the time domain, updating S^> and applying the self-consistency.
        If the parity flag is False, the roles of odd and even states are inverted.

        Arguments:
            parity_flag -- boolean
        """
        group_1, group_2 = self.even_states, self.odd_states
        if parity_flag:
            group_1, group_2 = group_2, group_1

        R_grea = []
        for k, s in enumerate(group_1):
            R_grea.append(inv_ft_to_alpert(self.freq_mesh.xmin, self.freq_mesh.delta, self.R_grea_w[:, s], M=M, order=order))
            R_grea[k] *= 1j

        S_grea = self._self_energy_grea(parity_flag, R_grea)
        del R_grea

        for k in range(len(S_grea)):
            _, f = alpert_fourier_transform(S_grea[k], wmin=self.freq_mesh.xmin, N=len(self.freq_mesh))
            S_grea[k] = f


        for k, s in enumerate(group_2):
            r = self.inv_R0_reta_w[:, s] - S_grea[k]
            self.R_grea_w[:, s] = np.imag(2.0 / r)

    def _self_energy_less(self, parity_flag, R_less):
        """
        Compute S^< from R^< in time domain.

        Arguments:
            parity_flag -- boolean, True for even states, False for odd states
            R_less -- 2D array (times, states) for R^< for the complementary states

        Returns:
            2D array (times, states)
        """
        states = self.even_states if parity_flag else self.odd_states

        S_less = np.zeros((self.N, len(states)), dtype=complex)

        for k, a in enumerate(states):
            for b, _, delta in self.hybridizations[a]:
                S_less[:, k] += -1j * delta[:] * R_less[:, self.state_parity_table[b]]

        return S_less

    def iteration_less(self, parity_flag):
        """
        Perform one lesser iteration.

        If parity flag is True, use R^< in frequency domain of odd states to update the same quantity for even states, by going to the time domain, updating S^< and applying the self-consistency.
        If the parity flag is False, the roles of odd and even states are inverted.

        Arguments:
            parity_flag -- boolean
        """
        group_1, group_2 = self.even_states, self.odd_states
        if parity_flag:
            group_1, group_2 = group_2, group_1

        R_less = np.empty((self.N, self.D_half), dtype=complex)

        for k, s in enumerate(group_1):
            _, R_less[:, k] = inv_fourier_transform(
                self.freq_mesh, self.R_less_w[:, s], axis=0
            )
        R_less *= 1.0j

        S_less = self._self_energy_less(parity_flag, R_less)
        del R_less

        _, S_less = fourier_transform(self.time_mesh, S_less, axis=0)

        for k, s in enumerate(group_2):
            self.R_less_w[:, s] = self.R_reta_sqr_w[:, s] * S_less[:, k].imag

    def normalize_less_w(self):
        """
        Normalize R^< according to the partition function

        Raises:
            ZeroDivisionError: norm is zero
        """
        Z = 0.0
        for i in range(self.D):
            Z += -integrate.simpson(dx=self.freq_mesh.delta, y=self.R_less_w[:, i])
        Z /= 2 * np.pi
        if Z == 0.0:
            raise ZeroDivisionError
        self.R_less_w *= self.Z_loc / Z

    ### Loops ###

    def fixed_pt_function_grea(self, R_grea_w):
        """
        Fixed point function for greater loop

        Arguments:
            R_grea_w -- 2D array

        Returns:
            New R^>
        """
        self.R_grea_w[...] = R_grea_w

        self.iteration_grea(False, self.M, self.order)
        self.iteration_grea(True, self.M, self.order)

        self.normalization_error.append(self.get_normalization_error())

        self.nr_grea_feval += 2

        return self.R_grea_w.copy()

    def fixed_pt_function_less(self, R_less_w):
        """
        Fixed point function for lesser loop

        Arguments:
            R_less_w -- 2D array

        Returns:
            New R^<
        """

        self.R_less_w[...] = R_less_w.reshape((-1, self.D))

        self.iteration_less(False)
        self.iteration_less(True)

        self.normalize_less_w()

        self.nr_less_feval += 2

        return self.R_less_w.copy()

    ### Utilities ###

    def get_normalization_error(self):
        """
        Normalization error for R^>

        Returns:
            error
        """
        norm = np.empty(self.D, dtype=float)
        for i in range(self.D):
            norm[i] = integrate.simpson(self.R_grea_w[:, i], dx=self.freq_mesh.delta)
        return np.abs(norm + 2.0 * np.pi)
