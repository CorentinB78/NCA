import numpy as np
import itertools
from .utilities import *
from .fixed_point_loop_solver import fixed_point_loop

import matplotlib
from matplotlib import pyplot as plt

# TODO: Generalize interface: allow any processes
# TODO: Generalize equations: allow non diagonal and sectors
# TODO: Separate self-consistent solver to allow using other root finding techniques


class NCA_Steady_State_Solver_Dyson:
    def __init__(self, R0_grea, R0_less, time_mesh, hybridizations):
        """
        R0_grea, R0_less: time functions returning 1D arrays, 1 component per state.
        """
        self.time_mesh = time_mesh
        self.times = self.time_mesh.values()

        N = len(self.time_mesh)
        self.D = R0_grea.shape[1]
        assert R0_grea.shape == (N, self.D)
        assert R0_less.shape == (N, self.D)

        self.R0_grea = R0_grea
        self.R0_less = R0_less
        self.R0_reta = R0_grea.copy()

        _, self.R0_less_w = fourier_transform(self.time_mesh, self.R0_less, axis=0)
        idx0 = len(self.time_mesh) // 2
        self.R0_reta[:idx0, :] = 0.0
        self.R0_reta[idx0, :] *= 0.5
        _, self.R0_reta_w = fourier_transform(self.time_mesh, self.R0_reta, axis=0)

        self.Z_loc = self.D

        self.hybridizations = hybridizations
        self.R_less = np.zeros((N, self.D), dtype=complex)
        self.R_grea = np.zeros((N, self.D), dtype=complex)

        self.S_less = np.zeros((N, self.D), dtype=complex)
        self.S_grea = np.zeros((N, self.D), dtype=complex)

        self.freq_mesh = self.time_mesh.adjoint()
        self.freqs = self.freq_mesh.values()

        self.R_grea_w = np.zeros((N, self.D), dtype=complex)
        self.R_grea_reta_w = np.zeros((N, self.D), dtype=complex)
        self.R_less_w = np.zeros((N, self.D), dtype=complex)

        self.S_less_w = np.zeros((N, self.D), dtype=complex)
        self.S_grea_w = np.zeros((N, self.D), dtype=complex)
        self.S_grea_reta_w = np.zeros((N, self.D), dtype=complex)

        self.nr_grea_feval = 0
        self.nr_less_feval = 0

    ############# greater ##############
    def go_to_times_grea(self):
        """\tilde R^>(w) ---> R^>(t)"""
        _, self.R_grea = inv_fourier_transform(
            self.freq_mesh, 2j * self.R_grea_reta_w.imag, axis=0
        )

    def normalize_grea(self):
        idx0 = len(self.time_mesh) // 2
        for k in range(self.D):
            self.R_grea[:, k] /= 1.0j * self.R_grea[idx0, k]

    def self_energy_grea(self):
        """R^>(t) ---> S^>(t)"""
        self.S_grea[:, :] = 0.0

        for state_a, state_b, delta, _ in self.hybridizations:
            state_a, state_b = np.atleast_1d(state_a, state_b)
            self.S_grea[:, state_a] += 1j * delta[:, None] * self.R_grea[:, state_b]

    def back_to_freqs_grea(self):
        """S^>(t) ---> \tilde S^>(w)"""
        idx0 = len(self.time_mesh) // 2
        self.S_grea_reta_w[:idx0, :] = 0.0
        self.S_grea_reta_w[idx0:, :] = self.S_grea[idx0:, :]
        self.S_grea_reta_w[idx0, :] *= 0.5
        _, self.S_grea_reta_w = fourier_transform(
            self.time_mesh, self.S_grea_reta_w, axis=0
        )

    def propagator_grea(self, eta=0.0):
        """\tilde S^>(w) ---> \tilde R^>(w)"""
        self.R_grea_reta_w[:, :] = (
            1.0 - self.R0_reta_w * self.S_grea_reta_w + 1.0j * eta
        )
        if not np.all(np.isfinite(self.R_grea_reta_w)):
            raise RuntimeError("WARNING: division by zero")
        self.R_grea_reta_w = self.R0_reta_w / self.R_grea_reta_w

    def initialize_grea(self, eta=0.0):
        self.R_grea = self.R0_grea * np.exp(
            -((4.0 * self.times[:, None] / self.time_mesh.xmax) ** 2)
        )

        self.normalize_grea()
        self.self_energy_grea()
        self.back_to_freqs_grea()
        self.propagator_grea(eta=eta)

    def fixed_pt_function_grea(self, R_grea_reta_w, eta=0.0):
        self.R_grea_reta_w = R_grea_reta_w

        self.go_to_times_grea()
        self.normalize_grea()
        self.self_energy_grea()
        self.back_to_freqs_grea()
        self.propagator_grea(eta=eta)

        self.nr_grea_feval += 1

        return self.R_grea_reta_w

    def greater_loop(
        self, tol=1e-8, min_iter=5, max_iter=100, eta=1.0, plot=False, verbose=False
    ):
        def err_func(R):
            return np.trapz(np.mean(np.abs(R), axis=1), dx=self.freq_mesh.delta)

        def callback_func(R, n_iter):
            plt.plot(
                self.freq_mesh.values(), -R[:, 2].imag, label=str(n_iter), alpha=0.3
            )

        self.initialize_grea()

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
            plt.xlim(-5, 5)

        self.R_grea_w[:] = 2j * self.R_grea_reta_w.imag
        self.S_grea_w[:] = 2j * self.S_grea_reta_w.imag

    ########## lesser ############
    def go_to_times_less(self):
        """R^<(w) ---> R^<(t)"""
        _, self.R_less = inv_fourier_transform(self.freq_mesh, self.R_less_w, axis=0)

    def self_energy_less(self):
        """R^<(t) ---> S^<(t)"""
        self.S_less[:, :] = 0.0

        for state_a, state_b, _, delta in self.hybridizations:
            state_a, state_b = np.atleast_1d(state_a, state_b)
            self.S_less[:, state_a] += -1j * delta[:, None] * self.R_less[:, state_b]

    def back_to_freqs_less(self):
        """S^<(t) ---> S^<(w)"""
        _, self.S_less_w = fourier_transform(self.time_mesh, self.S_less, axis=0)

    def propagator_less(self):
        """S^<(w) ---> R^<(w)"""
        self.R_less_w = (
            self.cst_term_less
            + self.R_grea_reta_w * self.S_less_w * np.conj(self.R_grea_reta_w)
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
        self.R_less = self.R0_less * np.exp(
            -((4.0 * self.times[:, None] / self.time_mesh.xmax) ** 2)
        )

        self.cst_term_less = 1 + self.R_grea_reta_w * self.S_grea_reta_w
        self.cst_term_less = (
            self.cst_term_less * self.R0_less_w * np.conj(self.cst_term_less)
        )

        # self.normalize_less_t()
        self.self_energy_less()
        self.back_to_freqs_less()
        self.propagator_less()
        self.normalize_less_w()

    def fixed_pt_function_less(self, R_less_w):
        self.R_less_w = R_less_w

        self.go_to_times_less()
        # self.normalize_less_t()
        self.self_energy_less()
        self.back_to_freqs_less()
        self.propagator_less()
        self.normalize_less_w()

        self.nr_less_feval += 1

        return self.R_less_w

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
            alpha=0.5,
        )

        if plot:
            plt.legend()
            plt.xlim(-20, 15)
