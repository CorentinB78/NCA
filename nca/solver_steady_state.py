import numpy as np
from matplotlib import pyplot as plt
from .function_tools import fourier_transform, inv_fourier_transform
from .fixed_point_loop_solver import fixed_point_loop
import toolbox as tb


class SolverSteadyState:
    def __init__(self, local_evol, time_mesh, hybridizations, list_even_states):
        """
        Real time Non-Crossing Approximation (NCA) solver for steady states.

        For now only diagonal hybridizations and local hamiltonians are supported. TODO.

        * local_evol: list of local evolution for each state. A local evolution can be a complex number representing energy and damping (positive imag part), or the values of 1/R_0^{reta}(w) on the frequency mesh adjoint to `time_mesh`.
        * time_mesh: an instance of `Mesh`.
        * hybridizations: list of hybridization processes. Each process is a tuple (a, b, delta_grea, delta_less) where a, b are states (identified by an int within range(D)) and delta_grea/less are 1D arrays containing hybridization functions (as sampled on `time_mesh`). delta_grea is the one participating to the greater SE, while delta_less is for the lesser SE. The process changes the local system from a to b then back to a. Conjugate processes are not added automatically.
        Optionnaly, several processes can be regrouped if they share the same hybridization functions, then a and b should be 1D arrays.
        * list_even_states: TODO
        """
        # TODO: sanity checks

        self.N = len(time_mesh)
        N = self.N
        self.D = len(local_evol)
        self.Z_loc = self.D

        self.is_even_state = np.array(
            [(s in list_even_states) for s in range(self.D)], dtype=bool
        )

        self.hybridizations = hybridizations
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

        self.normalization_error = []

    ############# greater ##############

    def normalize_grea(self, states):
        idx0 = self.N // 2
        self.R_grea[:, states] /= 1.0j * self.R_grea[idx0, states]

    def self_consistency_grea(self, states):
        """R^>(t) ---> S^>(t)"""
        other_states = ~states

        R_grea = np.empty((self.N, self.D), dtype=complex)

        for i in range(self.D):
            if other_states[i]:
                _, R_grea[:, i] = inv_fourier_transform(
                    self.freq_mesh, self.R_grea_w[:, i]
                )
                R_grea[:, i] *= 1j

        S_grea = np.zeros((self.N, self.D), dtype=complex)

        for a in np.arange(self.D)[states]:
            for b, delta, _ in self.hybridizations[a]:
                S_grea[:, a] += 1j * delta[:] * R_grea[:, b]

        del R_grea

        idx0 = self.N // 2
        S_grea[:idx0, states] = 0.0
        S_grea[idx0, states] *= 0.5
        for i in range(self.D):
            if states[i]:
                _, S_grea[:, i] = fourier_transform(
                    self.time_mesh, S_grea[:, i], axis=0
                )

        for i in range(self.D):
            if states[i]:
                inv_R_reta_w = self.inv_R0_reta_w[:, i] - S_grea[:, i]
                if not np.all(np.isfinite(inv_R_reta_w)):
                    raise ZeroDivisionError
                self.R_grea_w[:, i] = np.imag(2.0 / inv_R_reta_w)

    def get_R_grea_w(self):
        return self.R_grea_w.copy()

    def get_R_grea(self):
        _, R_grea = inv_fourier_transform(self.freq_mesh, self.R_grea_w, axis=0)
        return R_grea * 1j

    def get_R_reta_w(self):
        R_grea = self.get_R_grea()
        idx0 = self.N // 2

        R_grea[:idx0, :] = 0.0
        R_grea[idx0, :] *= 0.5
        _, R_reta_w = fourier_transform(self.time_mesh, R_grea, axis=0)
        return R_reta_w

    def initialize_grea(self):
        even = self.is_even_state

        delta_magn = 0.0
        idx0 = self.N // 2

        for a in np.arange(self.D)[even]:
            for b, delta, _ in self.hybridizations[a]:
                delta_magn += np.abs(delta[idx0]) ** 2
        delta_magn = np.sqrt(delta_magn)

        self.R_grea_w[:, even] = np.imag(
            2.0 / (self.inv_R0_reta_w[:, even] + 1.0j * delta_magn)
        )

    def fixed_pt_function_grea(self, R_grea_w):
        self.R_grea_w[...] = R_grea_w

        even = self.is_even_state
        odd = ~self.is_even_state

        self.self_consistency_grea(odd)
        self.self_consistency_grea(even)

        self.normalization_error.append(self.get_normalization_error())

        self.nr_grea_feval += 2

        return self.R_grea_w.copy()

    def greater_loop(
        self,
        tol=1e-8,
        max_iter=100,
        plot=False,
        verbose=False,
    ):
        def err_func(R):
            e = 0.0
            for i in range(self.D):
                e += np.trapz(np.abs(R[:, i]), dx=self.freq_mesh.delta)
            return e / self.D

        def callback_func(R, n_iter):
            plt.plot(
                self.freq_mesh.values(),
                -R[:, 0].imag,
                label=str(n_iter),
            )

        self.initialize_grea()

        fixed_point_loop(
            self.fixed_pt_function_grea,
            self.R_grea_w,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            callback_func=callback_func if plot else None,
            err_func=err_func,
        )

        if plot:
            plt.legend()
            plt.xlim(-20, 15)

        self.R_reta_sqr_w = np.abs(self.get_R_reta_w()) ** 2

        # self.S_grea_w = 2.0 * self.S_reta_w.imag

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

    ########## lesser ############
    def self_consistency_less(self, states):
        """R^<(t) ---> S^<(t)"""
        other_states = ~states

        R_less = np.empty((self.N, self.D), dtype=complex)

        for i in range(self.D):
            if other_states[i]:
                _, R_less[:, i] = inv_fourier_transform(
                    self.freq_mesh, self.R_less_w[:, i], axis=0
                )
                R_less[:, i] *= 1.0j

        S_less = np.zeros((self.N, self.D), dtype=complex)

        for a in np.arange(self.D)[states]:
            for b, _, delta in self.hybridizations[a]:
                S_less[:, a] += -1j * delta[:] * R_less[:, b]

        del R_less

        for i in range(self.D):
            if states[i]:
                _, ft = fourier_transform(self.time_mesh, S_less[:, i])
                S_less[:, i] = ft.imag

        self.R_less_w[:, states] = self.R_reta_sqr_w[:, states] * S_less[:, states]

    def get_R_less_w(self):
        return self.R_less_w.copy()

    def get_R_less(self):
        _, R_less = inv_fourier_transform(self.freq_mesh, self.R_less_w, axis=0)
        return R_less * 1j

    def normalize_less_w(self):
        Z = 0.0
        for i in range(self.D):
            Z += -np.trapz(dx=self.freq_mesh.delta, y=self.R_less_w[:, i])
        Z /= 2 * np.pi
        if Z == 0.0:
            raise ZeroDivisionError
        self.R_less_w *= self.Z_loc / Z

    def initialize_less(self):
        even = self.is_even_state

        delta_magn = 0.0
        idx0 = self.N // 2

        for a in np.arange(self.D)[even]:
            for b, _, delta in self.hybridizations[a]:
                delta_magn += np.abs(delta[idx0]) ** 2
        delta_magn = np.sqrt(delta_magn)

        for i in range(self.D):
            if even[i]:
                self.R_less_w[:, i] = np.imag(
                    1.0 / (self.inv_R0_reta_w[:, i] + 1.0j * delta_magn)
                )

        self.normalize_less_w()

    def fixed_pt_function_less(self, R_less_w):
        even = self.is_even_state
        odd = ~self.is_even_state

        self.R_less_w[...] = R_less_w.reshape((-1, self.D))

        self.self_consistency_less(odd)
        self.self_consistency_less(even)

        self.normalize_less_w()

        self.nr_less_feval += 2

        return self.R_less_w.copy()

    def lesser_loop(self, tol=1e-8, max_iter=100, plot=False, verbose=False, alpha=1.0):
        def err_func(R):
            e = 0.0
            for i in range(self.D):
                e += np.trapz(np.abs(R[:, i]), dx=self.freq_mesh.delta)
            return e / self.D

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

        if plot:
            plt.legend()
            plt.xlim(-20, 15)
