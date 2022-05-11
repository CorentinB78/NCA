import numpy as np
from .function_tools import fourier_transform, inv_fourier_transform
import toolbox as tb


class CoreSolverSteadyState:
    def __init__(self, local_evol, time_mesh, list_even_states, list_odd_states):
        """
        Real time Non-Crossing Approximation (NCA) solver for steady states.

        For now only diagonal hybridizations and local hamiltonians are supported. TODO.

        * local_evol: list of local evolution for each state. A local evolution can be a complex number representing energy and damping (positive imag part), or the values of 1/R_0^{reta}(w) on the frequency mesh adjoint to `time_mesh`.
        * time_mesh: an instance of `Mesh`.
        * list_even_states: TODO
        """
        # TODO: sanity checks
        assert len(list_even_states) + len(list_odd_states) == len(local_evol)

        self.N = len(time_mesh)
        N = self.N
        self.D = len(local_evol)
        self.Z_loc = self.D

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
        even = self.is_even_state

        delta_magn = 0.0
        idx0 = self.N // 2

        for a in self.even_states:
            for b, delta, _ in self.hybridizations[a]:
                delta_magn += np.abs(delta[idx0]) ** 2
        delta_magn = np.sqrt(delta_magn)

        self.R_grea_w[:, even] = np.imag(
            2.0 / (self.inv_R0_reta_w[:, even] + 1.0j * delta_magn)
        )

    def initialize_less(self):
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

    ### Self consistency relations

    def self_consistency_grea(self, parity_flag):
        """
        parity_flag: True for odd->even, False for even->odd
        """
        group_1, group_2 = self.even_states, self.odd_states
        if parity_flag:
            group_1, group_2 = group_2, group_1

        R_grea = np.empty((self.N, self.D_half), dtype=complex)

        for k, s in enumerate(group_1):
            _, R_grea[:, k] = inv_fourier_transform(self.freq_mesh, self.R_grea_w[:, s])
        R_grea *= 1j

        S_grea = np.zeros((self.N, self.D_half), dtype=complex)

        for k, a in enumerate(group_2):
            for b, delta, _ in self.hybridizations[a]:
                S_grea[:, k] += 1j * delta[:] * R_grea[:, self.state_parity_table[b]]

        del R_grea

        idx0 = self.N // 2
        S_grea[:idx0, :] = 0.0
        S_grea[idx0, :] *= 0.5
        _, S_grea = fourier_transform(self.time_mesh, S_grea, axis=0)

        for k, s in enumerate(group_2):
            r = self.inv_R0_reta_w[:, s] - S_grea[:, k]
            self.R_grea_w[:, s] = np.imag(2.0 / r)

    def self_consistency_less(self, parity_flag):
        """
        parity_flag: True for odd->even, False for even->odd
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

        S_less = np.zeros((self.N, self.D_half), dtype=complex)

        for k, a in enumerate(group_2):
            for b, _, delta in self.hybridizations[a]:
                S_less[:, k] += -1j * delta[:] * R_less[:, self.state_parity_table[b]]

        del R_less

        _, S_less = fourier_transform(self.time_mesh, S_less, axis=0)

        for k, s in enumerate(group_2):
            self.R_less_w[:, s] = self.R_reta_sqr_w[:, s] * S_less[:, k].imag

    def normalize_less_w(self):
        Z = 0.0
        for i in range(self.D):
            Z += -np.trapz(dx=self.freq_mesh.delta, y=self.R_less_w[:, i])
        Z /= 2 * np.pi
        if Z == 0.0:
            raise ZeroDivisionError
        self.R_less_w *= self.Z_loc / Z

    ### Loops ###

    def fixed_pt_function_grea(self, R_grea_w):
        self.R_grea_w[...] = R_grea_w

        self.self_consistency_grea(False)
        self.self_consistency_grea(True)

        self.normalization_error.append(self.get_normalization_error())

        self.nr_grea_feval += 2

        return self.R_grea_w.copy()

    def fixed_pt_function_less(self, R_less_w):

        self.R_less_w[...] = R_less_w.reshape((-1, self.D))

        self.self_consistency_less(False)
        self.self_consistency_less(True)

        self.normalize_less_w()

        self.nr_less_feval += 2

        return self.R_less_w.copy()

    ### Utilities ###

    def check_quality_grid(self, tol_delta):
        _, _, w, der2 = tb.derivate_twice(self.freqs, self.R_grea_w, axis=0)
        der2 = np.trapz(x=w, y=np.abs(der2), axis=0)
        err_delta = self.freq_mesh.delta**2 * der2 / 12.0

        print(f"Quality grid: delta error ={err_delta}")
        print(f"Quality grid: norm error = {self.get_normalization_error()}")

        dw = np.sqrt(12 * tol_delta / np.max(der2))
        print(f"Max time advised: {np.pi / dw}")

    def get_normalization_error(self):
        norm = np.empty(self.D, dtype=float)
        for i in range(self.D):
            norm[i] = np.trapz(self.R_grea_w[:, i], dx=self.freq_mesh.delta)
        return np.abs(norm + 2.0 * np.pi)
