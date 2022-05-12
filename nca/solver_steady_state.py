import numpy as np
from matplotlib import pyplot as plt
from .function_tools import fourier_transform, inv_fourier_transform
from .state_space import StateSpace
from .core_steady_state import CoreSolverSteadyState
from .fixed_point_loop_solver import fixed_point_loop


def greater_gf(orbital, state_space, time_mesh, R_grea, R_less, Z):
    R_grea = np.asarray(R_grea, dtype=complex)
    R_less = np.asarray(R_less, dtype=complex)

    states_yes, states_no = state_space.get_state_pairs_from_orbital(orbital)

    G_grea = 0.0
    for i in range(len(states_yes)):
        s_no, s_yes = states_no[i], states_yes[i]
        G_grea += R_less[::-1, s_no] * R_grea[:, s_yes]

    G_grea *= 1j / Z
    return time_mesh, G_grea


def lesser_gf(orbital, state_space, time_mesh, R_grea, R_less, Z):
    R_grea = np.asarray(R_grea, dtype=complex)
    R_less = np.asarray(R_less, dtype=complex)

    states_yes, states_no = state_space.get_state_pairs_from_orbital(orbital)

    G_less = 0.0
    for i in range(len(states_yes)):
        s_no, s_yes = states_no[i], states_yes[i]
        G_less += R_grea[::-1, s_no] * R_less[:, s_yes]

    G_less *= -1j / Z
    return time_mesh, G_less


class SolverSteadyState:
    # TODO: method for list of even states
    def __init__(
        self,
        nr_orbitals,
        local_evol,
        time_mesh,
        orbital_names=None,
        forbidden_states=None,
    ):
        """
        Real time Non-Crossing Approximation (NCA) solver for steady states.

        For now only diagonal hybridizations and local hamiltonians are supported. TODO.

        * local_evol: list of local evolution for each state. A local evolution can be a complex number representing energy and damping (positive imag part), or the values of 1/R_0^{reta}(w) on the frequency mesh adjoint to `time_mesh`.
        * time_mesh: an instance of `Mesh`.
        * hybridizations: list of hybridization processes. Each process is a tuple (a, b, delta_grea, delta_less) where a, b are states (identified by an int within range(D)) and delta_grea/less are 1D arrays containing hybridization functions (as sampled on `time_mesh`). delta_grea is the one participating to the greater SE, while delta_less is for the lesser SE. The process changes the local system from a to b then back to a. Conjugate processes are not added automatically.
        Optionnaly, several processes can be regrouped if they share the same hybridization functions, then a and b should be 1D arrays.
        * list_even_states: TODO
        """
        self.state_space = StateSpace(nr_orbitals, orbital_names, forbidden_states)

        self.D = len(local_evol)
        self.N = len(time_mesh)
        assert self.D == 2**nr_orbitals - len(self.state_space.forbidden_states)

        self._hybs = {}
        for s in self.state_space.all_states:
            self._hybs[s] = []

        even, odd = self.state_space.get_states_by_parity()

        self.core = CoreSolverSteadyState(local_evol, time_mesh, even, odd)

        self.time_mesh = time_mesh
        self.freq_mesh = self.core.freq_mesh

        self._lock_hybs = False

    def add_bath(self, orbital, delta_grea, delta_less):
        """Only baths coupled to a single orbital for now"""
        if self._lock_hybs:
            raise RuntimeError

        states_a, states_b = self.state_space.get_state_pairs_from_orbital(orbital)

        for a, b in zip(states_a, states_b):
            self._hybs[a].append((b, delta_grea, delta_less))
            self._hybs[b].append((a, np.conj(delta_less), np.conj(delta_grea)))

    def greater_loop(
        self,
        tol=1e-8,
        max_iter=100,
        plot=False,
        verbose=False,
    ):
        self._lock_hybs = True
        self.core.hybridizations = self._hybs

        def err_func(R):
            e = np.sum(np.trapz(np.abs(R), dx=self.freq_mesh.delta, axis=0))
            return e / self.D

        def callback_func(R, n_iter):
            plt.plot(
                self.freq_mesh.values(),
                -R[:, 0].imag,
                label=str(n_iter),
            )

        self.core.initialize_grea()

        fixed_point_loop(
            self.core.fixed_pt_function_grea,
            self.core.R_grea_w,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            callback_func=callback_func if plot else None,
            err_func=err_func,
        )

        if plot:
            plt.legend()
            plt.xlim(-20, 15)

        self.core.R_reta_sqr_w = np.abs(self.get_R_reta_w()) ** 2

        # self.core.S_grea_w = 2.0 * self.core.S_reta_w.imag

    def lesser_loop(self, tol=1e-8, max_iter=100, plot=False, verbose=False, alpha=1.0):
        self._lock_hybs = True
        self.core.hybridizations = self._hybs

        def err_func(R):
            e = np.sum(np.trapz(np.abs(R), dx=self.freq_mesh.delta, axis=0))
            return e / self.D

        def callback_func(R, n_iter):
            plt.plot(
                self.freq_mesh.values(),
                R[:, 0].imag,
                label=str(n_iter),
                color="b" if n_iter % 2 else "r",
            )

        self.core.initialize_less()

        fixed_point_loop(
            self.core.fixed_pt_function_less,
            self.core.R_less_w,
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

    ### getters ###

    def get_R_grea_w(self):
        """
        Returns the *imaginary part* of R^>(w) in a float array, which is a pure imaginary quantity.
        """
        return self.core.R_grea_w.copy()

    def get_R_grea(self):
        """
        Returns R^>(t) in a complex array
        """
        _, R_grea = inv_fourier_transform(self.freq_mesh, self.core.R_grea_w, axis=0)
        return R_grea * 1j

    def get_R_reta_w(self):
        """
        Returns R^R(w) in a complex array
        """
        R_grea = self.get_R_grea()
        idx0 = self.N // 2

        R_grea[:idx0, :] = 0.0
        R_grea[idx0, :] *= 0.5
        _, R_reta_w = fourier_transform(self.time_mesh, R_grea, axis=0)
        return R_reta_w

    def get_R_less_w(self):
        """
        Returns the *imaginary part* of R^<(w) in a float array, which is a pure imaginary quantity.
        """
        return self.core.R_less_w.copy()

    def get_R_less(self):
        """
        Returns R^<(t) in a complex array
        """
        _, R_less = inv_fourier_transform(self.freq_mesh, self.core.R_less_w, axis=0)
        return R_less * 1j

    def get_S_grea_w(self):
        """
        Returns the *imaginary part* of S^>(w) in a float array, which is a pure imaginary quantity.
        """
        return 2 * np.imag(self.get_S_reta_w())

    def get_S_reta_w(self):
        """
        Returns S^R(w) in a complex array
        """
        out = self.core.inv_R0_reta_w - 1.0 / self.get_R_reta_w()
        return out

    def get_S_less_w(self):
        """
        Returns the *imaginary part* of S^<(w) in a float array, which is a pure imaginary quantity.
        """
        out = self.core.R_less_w / self.core.R_reta_sqr_w
        return np.real(out)

    def get_G_grea(self, orbital):
        """Returns G^>(t) on time grid used in solver"""
        return greater_gf(
            orbital,
            self.state_space,
            self.time_mesh,
            self.get_R_grea(),
            self.get_R_less(),
            self.core.Z_loc,
        )

    def get_G_less(self, orbital):
        """Returns G^<(t) on time grid used in solver"""
        return lesser_gf(
            orbital,
            self.state_space,
            self.time_mesh,
            self.get_R_grea(),
            self.get_R_less(),
            self.core.Z_loc,
        )

    def get_G_grea_w(self, orbital):
        m, g = self.get_G_grea(orbital)
        m, g = fourier_transform(m, g)
        return m, g

    def get_G_less_w(self, orbital):
        m, g = self.get_G_less(orbital)
        m, g = fourier_transform(m, g)
        return m, g

    def get_G_reta_w(self, orbital):
        """
        Returns the retarded Green function in frequencies
        """
        m, G_less = self.get_G_less(orbital)
        m2, G_grea = self.get_G_grea(orbital)

        assert m is m2

        G_reta = G_grea - G_less
        idx0 = self.N // 2
        G_reta[:idx0] = 0.0
        G_reta[idx0] *= 0.5
        m, G_reta_w = fourier_transform(m, G_reta)
        return m, G_reta_w

    def get_DOS(self, orbital):
        """Returns density of states"""
        m, G_less = self.get_G_less(orbital)
        m2, G_grea = self.get_G_grea(orbital)

        assert m is m2

        dos = 1j * (G_grea - G_less) / (2 * np.pi)
        m, dos = fourier_transform(m, dos)
        return m, np.real(dos)

    def get_normalization_error(self):
        return self.core.get_normalization_error()


def AIM_infinite_U(local_evol, time_mesh):
    return SolverSteadyState(
        2, local_evol, time_mesh, orbital_names=["up", "dn"], forbidden_states=[3]
    )
