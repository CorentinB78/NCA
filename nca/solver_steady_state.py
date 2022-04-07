import numpy as np
from matplotlib import pyplot as plt
from .utilities import fourier_transform, inv_fourier_transform
from .fixed_point_loop_solver import fixed_point_loop


class NCA_Steady_State_Solver:
    def __init__(
        self, local_evol, time_mesh, hybridizations, list_even_states, energy_shift=None
    ):
        """
        Real time Non-Crossing Approximation (NCA) solver for steady states.

        For now only diagonal hybridizations and local hamiltonians are supported. TODO.

        * local_evol: diagonal of the (D, D) hamiltonian matrix, or retarded propagator in frequencies in a (N, D) array (D is the number of states in the local system, N the number of times/frequencies)
        * time_mesh: an instance of `Mesh`.
        * hybridizations: list of hybridization processes. Each process is a tuple (a, b, delta_grea, delta_less) where a, b are states (identified by an int within range(D)) and delta_grea/less are 1D arrays containing hybridization functions (as sampled on `time_mesh`). delta_grea is the one participating to the greater SE, while delta_less is for the lesser SE. The process changes the local system from a to b then back to a. Conjugate processes are not added automatically.
        Optionnaly, several processes can be regrouped if they share the same hybridization functions, then a and b should be 1D arrays.
        * list_even_states: TODO
        """
        # TODO: sanity checks

        N = len(time_mesh)
        local_evol = np.asarray(local_evol)
        self.H_loc = None
        self.R0_reta_w = None

        if local_evol.ndim == 1:
            self.H_loc = local_evol
        elif local_evol.ndim == 2 and local_evol.shape[0] == N:
            self.R0_reta_w = local_evol
        else:
            raise ValueError(
                "`local_evol` should have a shape (D) for hamiltonian or (N, D) for propagator."
            )

        self.D = local_evol.shape[-1]
        self.Z_loc = self.D
        self.is_even_state = np.array(
            [(s in list_even_states) for s in range(self.D)], dtype=bool
        )

        self.hybridizations = hybridizations
        self.time_mesh = time_mesh
        self.times = self.time_mesh.values()

        self.R_less = np.zeros((N, self.D), dtype=complex)
        self.R_grea = np.zeros((N, self.D), dtype=complex)

        self.S_less = np.zeros((N, self.D), dtype=complex)
        self.S_grea = np.zeros((N, self.D), dtype=complex)

        self.freq_mesh = self.time_mesh.adjoint()
        self.freqs = self.freq_mesh.values()

        self.R_grea_w = np.zeros((N, self.D), dtype=complex)
        self.R_reta_w = np.zeros((N, self.D), dtype=complex)
        self.R_less_w = np.zeros((N, self.D), dtype=complex)

        self.S_less_w = np.zeros((N, self.D), dtype=complex)
        self.S_grea_w = np.zeros((N, self.D), dtype=complex)
        self.S_reta_w = np.zeros((N, self.D), dtype=complex)

        self.energy_shift = np.zeros(self.D)
        if energy_shift is not None:
            self.energy_shift += energy_shift

        self.nr_grea_feval = 0
        self.nr_less_feval = 0

        self.normalization_error = []

    ############# greater ##############
    def go_to_times_grea(self, states):
        """\tilde R^>(w) ---> R^>(t)"""
        _, self.R_grea[:, states] = inv_fourier_transform(
            self.freq_mesh, 2j * self.R_reta_w[:, states].imag, axis=0
        )

    def normalize_grea(self, states):
        idx0 = len(self.time_mesh) // 2
        self.normalization_error.append(
            np.mean(np.abs(self.R_grea[idx0, states] + 1.0j))
        )
        self.R_grea[:, states] /= 1.0j * self.R_grea[idx0, states]

    def self_energy_grea(self, states):
        """R^>(t) ---> S^>(t)"""
        self.S_grea[:, states] = 0.0

        for states_a, states_b, delta, _ in self.hybridizations:
            states_a, states_b = np.atleast_1d(states_a, states_b)
            for (a, b) in zip(states_a, states_b):
                if states[a]:
                    if states[b]:
                        raise RuntimeError(
                            "An hybridization process couples two states of same parity."
                        )
                    self.S_grea[:, a] += (
                        1j
                        * delta[:]
                        * self.R_grea[:, b]
                        * np.exp(
                            1j
                            * (self.energy_shift[a] - self.energy_shift[b])
                            * self.times
                        )
                    )

    def back_to_freqs_grea(self, states):
        """S^>(t) ---> \tilde S^>(w)"""
        idx0 = len(self.time_mesh) // 2
        self.S_reta_w[:idx0, states] = 0.0
        self.S_reta_w[idx0:, states] = self.S_grea[idx0:, states]
        self.S_reta_w[idx0, states] *= 0.5
        _, self.S_reta_w[:, states] = fourier_transform(
            self.time_mesh, self.S_reta_w[:, states], axis=0
        )

    def propagator_grea(self, states, eta=0.0):
        """\tilde S^>(w) ---> \tilde R^>(w)"""
        if self.H_loc is None:
            self.R_reta_w[:, states] = 1.0 - self.R0_reta_w[:, states] * (
                self.S_reta_w[:, states] - 1.0j * eta
            )
            if not np.all(np.isfinite(self.R_reta_w[:, states])):
                raise ZeroDivisionError
            self.R_reta_w[:, states] = (
                self.R0_reta_w[:, states] / self.R_reta_w[:, states]
            )
        else:
            self.R_reta_w[:, states] = (
                self.freq_mesh.values()[:, None]
                + self.energy_shift[states]
                - self.H_loc[states]
                - self.S_reta_w[:, states]
                + 1.0j * eta
            )
            if not np.all(np.isfinite(self.R_reta_w[:, states])):
                raise ZeroDivisionError
            self.R_reta_w[:, states] = 1.0 / self.R_reta_w[:, states]

    def initialize_grea(self, eta=0.0):
        even = self.is_even_state

        delta_magn = 0.0
        idx0 = len(self.times) // 2

        for states_a, states_b, delta, _ in self.hybridizations:
            states_a, states_b = np.atleast_1d(states_a, states_b)
            for (a, b) in zip(states_a, states_b):
                if even[a]:
                    delta_magn += np.abs(delta[idx0]) ** 2
        delta_magn = np.sqrt(delta_magn)

        if self.H_loc is None:
            self.R_reta_w[:, even] = self.R0_reta_w[:, even] / (
                1.0 + 1.0j * (delta_magn + eta) * self.R0_reta_w[:, even]
            )
        else:
            self.R_reta_w[:, even] = 1.0 / (
                self.freqs[:, None] + 1.0j * (delta_magn + eta)
            )

    def fixed_pt_function_grea(self, R_reta_w, eta=0.0):
        self.R_reta_w[...] = R_reta_w

        even = self.is_even_state
        odd = ~self.is_even_state

        self.go_to_times_grea(even)
        self.normalize_grea(even)
        self.self_energy_grea(odd)
        self.back_to_freqs_grea(odd)
        self.propagator_grea(odd, eta=eta)
        self.go_to_times_grea(odd)
        self.normalize_grea(odd)
        self.self_energy_grea(even)
        self.back_to_freqs_grea(even)
        self.propagator_grea(even, eta=eta)

        self.nr_grea_feval += 2

        return self.R_reta_w.copy()

    def greater_loop(
        self,
        tol=1e-8,
        min_iter=5,
        max_iter=100,
        eta=1.0,
        eta_perm=0.0,
        plot=False,
        verbose=False,
    ):
        def err_func(R):
            return np.trapz(np.mean(np.abs(R), axis=1), dx=self.freq_mesh.delta)

        def callback_func(R, n_iter):
            plt.plot(
                self.freq_mesh.values(),
                -R[:, 0].imag,
                label=str(n_iter),
            )

        self.initialize_grea(eta=eta + eta_perm)

        eta_i = eta
        q = np.log(100.0) / np.log(min_iter)
        for _ in range(min_iter):
            self.fixed_pt_function_grea(self.R_reta_w, eta=eta_i + eta_perm)
            eta_i /= q

        fixed_point_loop(
            self.fixed_pt_function_grea,
            self.R_reta_w,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            callback_func=callback_func if plot else None,
            err_func=err_func,
            f_kwargs={"eta": eta_perm},
        )

        if plot:
            plt.legend()
            plt.xlim(-20, 15)

        self.R_grea_w[:] = 2j * self.R_reta_w.imag
        self.S_grea_w[:] = 2j * self.S_reta_w.imag

    ########## lesser ############
    def go_to_times_less(self, states):
        """R^<(w) ---> R^<(t)"""
        _, self.R_less[:, states] = inv_fourier_transform(
            self.freq_mesh, self.R_less_w[:, states], axis=0
        )

    def self_energy_less(self, states):
        """R^<(t) ---> S^<(t)"""
        self.S_less[:, states] = 0.0

        for states_a, states_b, _, delta in self.hybridizations:
            states_a, states_b = np.atleast_1d(states_a, states_b)
            for (a, b) in zip(states_a, states_b):
                if states[a]:
                    if states[b]:
                        raise RuntimeError(
                            "An hybridization process couples two states of same parity."
                        )
                    self.S_less[:, a] += (
                        -1j
                        * delta[:]
                        * self.R_less[:, b]
                        * np.exp(
                            1j
                            * (self.energy_shift[a] - self.energy_shift[b])
                            * self.times
                        )
                    )

    def back_to_freqs_less(self, states):
        """S^<(t) ---> S^<(w)"""
        _, self.S_less_w[:, states] = fourier_transform(
            self.time_mesh, self.S_less[:, states], axis=0
        )

    def propagator_less(self, states):
        """S^<(w) ---> R^<(w)"""
        self.R_less_w[:, states] = (
            self.R_reta_w[:, states]
            * self.S_less_w[:, states]
            * np.conj(self.R_reta_w[:, states])
        )

    def normalize_less_t(self):
        idx0 = len(self.time_mesh) // 2
        Z = 1j * np.sum(self.R_less[idx0])
        if Z == 0.0:
            print(self.R_less[idx0])
            raise ZeroDivisionError
        self.R_less_w *= self.Z_loc / Z
        self.R_less *= self.Z_loc / Z

    def normalize_less_w(self):
        Z = 1j * np.sum(np.trapz(x=self.freqs, y=self.R_less_w, axis=0)) / (2 * np.pi)
        if Z == 0.0:
            raise ZeroDivisionError
        self.R_less_w *= self.Z_loc / Z
        self.R_less *= self.Z_loc / Z

    def initialize_less(self, eta=0.0):
        even = self.is_even_state

        delta_magn = 0.0
        idx0 = len(self.times) // 2

        for states_a, states_b, _, delta in self.hybridizations:
            states_a, states_b = np.atleast_1d(states_a, states_b)
            for (a, b) in zip(states_a, states_b):
                if even[a]:
                    delta_magn += np.abs(delta[idx0]) ** 2
        delta_magn = np.sqrt(delta_magn)

        if self.H_loc is None:
            self.R_less_w[:, even] = np.imag(
                self.R0_reta_w[:, even]
                / (1.0 + 1.0j * (delta_magn + eta) * self.R0_reta_w[:, even])
            )
        else:
            self.R_less_w[:, even] = np.imag(
                1.0
                / (
                    self.freqs[:, None]
                    + self.energy_shift[None, even]
                    - self.H_loc[None, even]
                    + 1.0j * (delta_magn + eta)
                )
            )

        self.normalize_less_w()

    def fixed_pt_function_less(self, R_less_w):
        even = self.is_even_state
        odd = ~self.is_even_state

        self.R_less_w[...] = R_less_w.reshape((-1, self.D))

        self.go_to_times_less(even)
        # self.normalize_less_t()
        self.self_energy_less(odd)
        self.back_to_freqs_less(odd)
        self.propagator_less(odd)
        # self.normalize_less_w()

        self.go_to_times_less(odd)
        # self.normalize_less_t()
        self.self_energy_less(even)
        self.back_to_freqs_less(even)
        self.propagator_less(even)
        self.normalize_less_w()

        self.nr_less_feval += 2

        return self.R_less_w.copy()

    def lesser_loop(self, tol=1e-8, max_iter=100, plot=False, verbose=False, alpha=1.0):
        def err_func(R):
            return np.trapz(np.mean(np.abs(R), axis=1), dx=self.freq_mesh.delta)

        def callback_func(R, n_iter):
            plt.plot(
                self.freq_mesh.values(),
                R[:, 0].imag,
                label=str(n_iter),
                color="b" if n_iter % 2 else "r",
            )

        self.initialize_less()

        fixed_point_loop(
            self.fixed_pt_function_less,
            self.R_less_w,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            callback_func=callback_func if plot else None,
            err_func=err_func,
            alpha=alpha,
        )

        self.go_to_times_less(range(self.D))

        if plot:
            plt.legend()
            plt.xlim(-20, 15)
