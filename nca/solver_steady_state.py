import numpy as np
import itertools
from .utilities import *
from .fixed_point_loop_solver import fixed_point_loop

import matplotlib
from matplotlib import pyplot as plt


class NCA_Steady_State_Solver:
    def __init__(self, H_loc, time_mesh, hybridizations, list_even_states):
        """
        Real time Non-Crossing Approximation (NCA) solver for steady states.

        For now only diagonal hybridizations and local hamiltonians are supported. TODO.

        H_loc: diagonal of the (D, D) hamiltonian matrix, D is the number of states in the local system.
        time_mesh: an instance of `Mesh`.
        hybridizations: list of hybridization processes. Each process is a tuple (a, b, delta_grea, delta_less) where
            a, b are states (identified by an int within range(D)) and delta_grea/less are 1D arrays containing hybridization functions (as sampled on `time_mesh`). delta_grea is the one participating to the greater SE, while delta_less is for the lesser SE. The process changes the local system from a to b then back to a. Conjugate processes are not added automatically.
            Optionnaly, several processes can be regrouped if they share the same hybridization functions, then a and b should be 1D arrays.
        list_even_states: TODO
        """

        self.H_loc = np.asarray(H_loc)
        self.D = len(self.H_loc)
        self.Z_loc = self.D
        self.is_even_state = np.array(
            [(s in list_even_states) for s in range(self.D)], dtype=bool
        )

        self.hybridizations = hybridizations
        self.time_mesh = time_mesh
        self.times = self.time_mesh.values()
        N = len(self.time_mesh)

        self.R_less = np.zeros((N, self.D), dtype=complex)
        self.R_grea = np.zeros((N, self.D), dtype=complex)

        self.S_less = np.zeros((N, self.D), dtype=complex)
        self.S_grea = np.zeros((N, self.D), dtype=complex)

        self.freq_mesh = self.time_mesh.adjoint()
        self.freqs = self.freq_mesh.values()

        diff_H_loc = np.diff(np.unique(self.H_loc))
        diff_H_loc = diff_H_loc[np.isfinite(diff_H_loc)]
        # if len(diff_H_loc) > 0:
        #     assert self.freq_mesh.delta < 0.1 * np.min(diff_H_loc)
        #     assert self.freq_mesh.xmax > 10 * np.max(diff_H_loc)

        self.R_grea_w = np.zeros((N, self.D), dtype=complex)
        self.R_grea_reta_w = np.zeros((N, self.D), dtype=complex)
        self.R_less_w = np.zeros((N, self.D), dtype=complex)

        self.S_less_w = np.zeros((N, self.D), dtype=complex)
        self.S_grea_w = np.zeros((N, self.D), dtype=complex)
        self.S_grea_reta_w = np.zeros((N, self.D), dtype=complex)

        self.nr_grea_feval = 0
        self.nr_less_feval = 0

    ############# greater ##############
    def go_to_times_grea(self, states):
        """\tilde R^>(w) ---> R^>(t)"""
        _, self.R_grea[:, states] = inv_fourier_transform(
            self.freq_mesh, 2j * self.R_grea_reta_w[:, states].imag, axis=0
        )

    def normalize_grea(self, states):
        idx0 = len(self.time_mesh) // 2
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
                    self.S_grea[:, a] += 1j * delta[:] * self.R_grea[:, b]

    def back_to_freqs_grea(self, states):
        """S^>(t) ---> \tilde S^>(w)"""
        idx0 = len(self.time_mesh) // 2
        self.S_grea_reta_w[:idx0, states] = 0.0
        self.S_grea_reta_w[idx0:, states] = self.S_grea[idx0:, states]
        self.S_grea_reta_w[idx0, states] *= 0.5
        _, self.S_grea_reta_w[:, states] = fourier_transform(
            self.time_mesh, self.S_grea_reta_w[:, states], axis=0
        )

    def propagator_grea(self, states, eta=0.0):
        """\tilde S^>(w) ---> \tilde R^>(w)"""
        self.R_grea_reta_w[:, states] = (
            self.freq_mesh.values()[:, None]
            - self.H_loc[states]
            - self.S_grea_reta_w[:, states]
            + 1.0j * eta
        )
        if not np.all(np.isfinite(self.R_grea_reta_w[:, states])):
            raise RuntimeError("WARNING: division by zero")
        self.R_grea_reta_w[:, states] = 1.0 / self.R_grea_reta_w[:, states]

    def initialize_grea(self, eta=0.0):
        even = self.is_even_state
        odd = ~self.is_even_state
        self.R_grea[:, odd] = (
            -1j
            * np.exp(-1j * self.H_loc[odd] * self.times[:, None])
            * np.exp(-((4.0 * self.times[:, None] / self.time_mesh.xmax) ** 2))
        )

        self.normalize_grea(odd)
        self.self_energy_grea(even)
        self.back_to_freqs_grea(even)
        self.propagator_grea(even, eta=eta)

    def fixed_pt_function_grea(self, R_grea_reta_w, eta=0.0):
        self.R_grea_reta_w[...] = R_grea_reta_w

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

        return self.R_grea_reta_w.copy()

    def greater_loop(
        self, tol=1e-8, min_iter=5, max_iter=100, eta=1.0, plot=False, verbose=False
    ):
        def err_func(R):
            return np.trapz(np.mean(np.abs(R), axis=1), dx=self.freq_mesh.delta)

        def callback_func(R, n_iter):
            plt.plot(
                self.freq_mesh.values(),
                -R[:, 0].imag,
                label=str(n_iter),
            )

        self.initialize_grea(eta=eta)

        eta_i = eta
        q = np.log(100.0) / np.log(min_iter)
        for _ in range(min_iter):
            self.fixed_pt_function_grea(self.R_grea_reta_w, eta=eta_i)
            eta_i /= q

        fixed_point_loop(
            self.fixed_pt_function_grea,
            self.R_grea_reta_w,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            callback_func=callback_func if plot else None,
            err_func=err_func,
        )

        if plot:
            plt.legend()
            plt.xlim(-20, 15)

        self.R_grea_w[:] = 2j * self.R_grea_reta_w.imag
        self.S_grea_w[:] = 2j * self.S_grea_reta_w.imag

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
                    self.S_less[:, a] += -1j * delta[:] * self.R_less[:, b]

    def back_to_freqs_less(self, states):
        """S^<(t) ---> S^<(w)"""
        _, self.S_less_w[:, states] = fourier_transform(
            self.time_mesh, self.S_less[:, states], axis=0
        )

    def propagator_less(self, states):
        """S^<(w) ---> R^<(w)"""
        self.R_less_w[:, states] = (
            self.R_grea_reta_w[:, states]
            * self.S_less_w[:, states]
            * np.conj(self.R_grea_reta_w[:, states])
        )

    def normalize_less_t(self):
        idx0 = len(self.time_mesh) // 2
        Z = 1j * np.mean(self.R_less[idx0])
        if Z == 0.0:
            print(self.R_less[idx0])
            raise ZeroDivisionError
        self.R_less_w /= Z
        self.R_less /= Z

    def normalize_less_w(self):
        Z = 1j * np.sum(np.trapz(x=self.freqs, y=self.R_less_w, axis=0)) / (2 * np.pi)
        if Z == 0.0:
            raise ZeroDivisionError
        self.R_less_w *= self.Z_loc / Z
        # self.R_less /= Z

    def initialize_less(self):
        """Run only when R^> is known"""
        even = self.is_even_state
        odd = ~self.is_even_state

        self.R_less[:, odd] = (
            -1j
            * np.exp(1j * self.H_loc[odd] * self.times[:, None])
            * np.exp(-((4.0 * self.times[:, None] / self.time_mesh.xmax) ** 2))
        )

        # self.normalize_less_t()
        self.self_energy_less(even)
        self.back_to_freqs_less(even)
        self.propagator_less(even)
        # self.normalize_less_w()

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

    def lesser_loop(self, tol=1e-8, max_iter=100, plot=False, verbose=False):
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
            alpha=0.75,
        )

        if plot:
            plt.legend()
            plt.xlim(-20, 15)


class FermionicFockSpace:
    # TODO: method for list of even states
    def __init__(self, orbital_names):
        self.orbital_names = orbital_names
        self.nr_orbitals = len(orbital_names)
        self.baths = []

    def state_string(self, state):
        s = []

        for k in range(self.nr_orbitals):
            if (state % 2) == 1:
                s.append(self.orbital_names[k])
            state = state // 2

        return ",".join(s)

    def basis(self):
        all_states = np.arange(2 ** self.nr_orbitals)
        out = [self.state_string(s) for s in all_states]
        return out

    def add_bath(self, orbital, delta_grea, delta_less):
        """Only baths coupled to a single orbital for now"""
        self.baths.append((orbital, delta_grea, delta_less))

    def is_orb_in_state(self, orbital, state):
        return (state // 2 ** orbital) % 2 == 1

    def states_containing(self, orbital):
        all_states = np.arange(2 ** self.nr_orbitals)
        contains = self.is_orb_in_state(orbital, all_states)
        return all_states[contains], all_states[~contains]

    def generate_hybridizations(self):
        hyb = []

        for orbital, delta_grea, delta_less in self.baths:
            states_a, states_b = self.states_containing(orbital)

            ### particle processes
            hyb.append((states_a, states_b, delta_grea, delta_less))

            ### hole processes
            hyb.append((states_b, states_a, np.conj(delta_less), np.conj(delta_grea)))

        return hyb

    def get_G_grea(self, orbital, solver):
        if orbital >= self.nr_orbitals:
            raise ValueError

        states_yes, states_no = self.states_containing(orbital)
        G_grea = np.sum(
            solver.R_less[::-1, states_no] * solver.R_grea[:, states_yes],
            axis=1,
        )

        G_grea *= 1j / solver.Z_loc
        return G_grea

    def get_G_less(self, orbital, solver):
        if orbital >= self.nr_orbitals:
            raise ValueError

        states_yes, states_no = self.states_containing(orbital)
        G_less = np.sum(
            solver.R_less[:, states_yes] * solver.R_grea[::-1, states_no],
            axis=1,
        )

        G_less *= -1j / solver.Z_loc
        return G_less

    def get_DOS(self, orbital, solver):
        G_less = self.get_G_less(orbital, solver)
        G_grea = self.get_G_grea(orbital, solver)

        dos = 1j * (G_grea - G_less) / (2 * np.pi)
        return np.real(fourier_transform(solver.time_mesh, dos)[1])
