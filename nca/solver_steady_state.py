import numpy as np
from .function_tools import fourier_transform, inv_fourier_transform
from .state_space import StateSpace
from .core_steady_state import CoreSolverSteadyState
from .fixed_point_loop_solver import fixed_point_loop


def greater_gf(orbital, state_space, time_mesh, R_grea, R_less, Z):
    """
    Compute a local greater Green function (GF) from R^< and R^>.

    Arguments:
        orbital -- integer indicating which GF to compute
        state_space -- a StateSpace instance
        time_mesh -- a Mesh instance (not used)
        R_grea -- 2D array of shape (times, states)
        R_less -- 2D array of shape (times, states)
        Z -- partition function for normalization

    Returns:
        time_mesh -- Mesh instance
        G_grea -- 1D array
    """
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
    """
    Compute a local lesser Green function (GF) from R^< and R^>.

    Arguments:
        orbital -- integer indicating which GF to compute
        state_space -- a StateSpace instance
        time_mesh -- a Mesh instance (not used)
        R_grea -- 2D array of shape (times, states)
        R_less -- 2D array of shape (times, states)
        Z -- partition function for normalization

    Returns:
        time_mesh -- Mesh instance
        G_grea -- 1D array
    """
    # TODO: time_mesh is not used!
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
    # TODO: implement non-diagonal hybridization functions & local Hamiltonian
    def __init__(
        self,
        nr_orbitals,
        local_evol,
        time_mesh,
        orbital_names=None,
        forbidden_states=None,
    ):
        """
        Non-Crossing Approximation (NCA) solver for steady states using real frequencies.

        For now only diagonal hybridizations and local hamiltonians are supported.

        Arguments:
            nr_orbitals -- int, number of orbitals in local system
            local_evol -- list of local evolution for each state. A local evolution can be a complex number representing energy and damping (negative imag part), or the values of 1/R_0^{reta}(w) on the frequency mesh adjoint to `time_mesh`.
            time_mesh -- Mesh instance for time coordinates

        Keyword Arguments:
            orbital_names -- list of strings to name orbitals (default: {None})
            forbidden_states -- list of states (int) which have infinite energy, and are thus forbidden (default: {None})
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
        """
        Connect a bath to the local system and describe the corresponding hybridization functions.

        Only bath connected to a single site.

        Arguments:
            orbital -- int, which orbital to connect the bath to
            delta_grea -- 1D array, greater hybridization function on time mesh of the solver
            delta_less -- 1D array, lesser hybridization function on time mesh of the solver

        Raises:
            RuntimeError: A bath was added after starting calculation
        """
        if self._lock_hybs:
            raise RuntimeError("A bath cannot be added after starting calculation")

        states_a, states_b = self.state_space.get_state_pairs_from_orbital(orbital)

        for a, b in zip(states_a, states_b):
            self._hybs[a].append((b, delta_grea, delta_less))
            self._hybs[b].append((a, np.conj(delta_less), np.conj(delta_grea)))

    def greater_loop(
        self, tol=1e-8, max_iter=100, verbose=False, alpha=1.0, return_iterations=False
    ):
        """
        Perform self-consistency loop for R^> and S^>.

        Should be done before the lesser loop.

        Keyword Arguments:
            tol -- tolerance to reach for the loop to stop (default: {1e-8})
            max_iter -- maximum number of iterations (default: {100})
            verbose -- if True, print information at each iteration (default: {False})
            alpha -- level of mixing (default: {1.0} no mixing)
            return_iterations -- if True, return a list of R at each iteration
        """
        self._lock_hybs = True
        self.core.hybridizations = self._hybs

        def err_func(R):
            e = np.sum(np.abs(R)) * self.freq_mesh.delta / self.D
            return e

        callback_func = None

        self.core.initialize_grea()

        if return_iterations:
            # after initialization
            R_all = [self.core.R_grea_w.copy()]
            S_all = [self.get_S_grea_w().copy()]

            def callback_func(R, n_iter):
                R_all.append(R.copy())
                S_all.append(self.get_S_grea_w().copy())

        fixed_point_loop(
            self.core.fixed_pt_function_grea,
            self.core.R_grea_w,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            callback_func=callback_func,
            err_func=err_func,
            alpha=alpha,
        )

        self.core.R_reta_sqr_w = np.abs(self.get_R_reta_w()) ** 2

        # self.core.S_grea_w = 2.0 * self.core.S_reta_w.imag

        if return_iterations:
            return R_all, S_all

    def lesser_loop(
        self, tol=1e-8, max_iter=100, verbose=False, alpha=1.0, return_iterations=False
    ):
        """
        Perform self-consistency loop for R^< and S^<.

        Greater loop should be done before.

        Keyword Arguments:
            tol -- tolerance to reach for the loop to stop (default: {1e-8})
            max_iter -- maximum number of iterations (default: {100})
            verbose -- if True, print information at each iteration (default: {False})
            alpha -- level of mixing (default: {1.0} no mixing)
            return_iterations -- if True, return a list of R at each iteration
        """
        self._lock_hybs = True
        self.core.hybridizations = self._hybs

        def err_func(R):
            e = np.sum(np.abs(R)) * self.freq_mesh.delta / self.D
            return e

        callback_func = None

        self.core.initialize_less()

        if return_iterations:
            # after initialization
            R_all = [self.core.R_less_w.copy()]
            S_all = [self.get_S_less_w().copy()]

            def callback_func(R, n_iter):
                R_all.append(R.copy())
                S_all.append(self.get_S_less_w().copy())

        fixed_point_loop(
            self.core.fixed_pt_function_less,
            self.core.R_less_w,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            callback_func=callback_func,
            err_func=err_func,
            alpha=alpha,
        )

        if return_iterations:
            return R_all, S_all

    ### R and S getters ###

    def get_R_grea_w(self):
        """
        Return R^> in frequency domain.
        As it is a pure imaginary quantity, only the imaginary part is returned.

        Returns:
            2D array with shape (frequencies, states)
        """
        return self.core.R_grea_w.copy()

    def get_R_grea(self):
        """
        Returns R^>(t) in time domain

        Returns:
            2D array with shape (times, states)
        """
        _, R_grea = inv_fourier_transform(self.freq_mesh, self.core.R_grea_w, axis=0)
        return R_grea * 1j

    def get_R_reta_w(self):
        """
        Return R^R in frequency domain.

        Returns:
            2D array with shape (frequencies, states)
        """
        R_grea = self.get_R_grea()
        idx0 = self.N // 2

        R_grea[:idx0, :] = 0.0
        R_grea[idx0, :] *= 0.5
        _, R_reta_w = fourier_transform(self.time_mesh, R_grea, axis=0)
        return R_reta_w

    def get_R_less_w(self):
        """
        Return R^< in frequency domain.
        As it is a pure imaginary quantity, only the imaginary part is returned.

        Returns:
            2D array with shape (frequencies, states)
        """
        return self.core.R_less_w.copy()

    def get_R_less(self):
        """
        Returns R^< in time domain

        Returns:
            2D array with shape (times, states)
        """
        _, R_less = inv_fourier_transform(self.freq_mesh, self.core.R_less_w, axis=0)
        return R_less * 1j

    def get_S_grea_w(self):
        """
        Return S^> in frequency domain.
        As it is a pure imaginary quantity, only the imaginary part is returned.

        Returns:
            2D array of floats with shape (frequencies, states)
        """
        return 2 * np.imag(self.get_S_reta_w())

    def get_S_reta_w(self):
        """
        Return S^R in frequency domain.

        Returns:
            2D array with shape (frequencies, states)
        """
        # TODO: use more stable method, e.g. going through real time and NCA self-energy formula
        out = self.core.inv_R0_reta_w - 1.0 / self.get_R_reta_w()
        return out

    def get_S_less_w(self):
        """
        Return S^< in frequency domain.
        As it is a pure imaginary quantity, only the imaginary part is returned.

        Returns:
            2D array of floats with shape (frequencies, states)
        """
        out = self.core.R_less_w / self.core.R_reta_sqr_w
        return np.real(out)

    ### Green functions getters ###

    def get_G_grea(self, orbital):
        """
        Compute greater Green functions in time domain

        Arguments:
            orbital -- int

        Returns:
            mesh -- a Mesh instance for time coordinates
            gf -- 1D array
        """
        return greater_gf(
            orbital,
            self.state_space,
            self.time_mesh,
            self.get_R_grea(),
            self.get_R_less(),
            self.core.Z_loc,
        )

    def get_G_less(self, orbital):
        """
        Compute lesser Green functions in time domain

        Arguments:
            orbital -- int

        Returns:
            mesh -- a Mesh instance for time coordinates
            gf -- 1D array
        """
        return lesser_gf(
            orbital,
            self.state_space,
            self.time_mesh,
            self.get_R_grea(),
            self.get_R_less(),
            self.core.Z_loc,
        )

    def get_G_grea_w(self, orbital):
        """
        Compute greater Green functions in frequency domain

        Arguments:
            orbital -- int

        Returns:
            mesh -- a Mesh instance for frequency coordinates
            gf -- 1D array
        """
        m, g = self.get_G_grea(orbital)
        m, g = fourier_transform(m, g)
        return m, g

    def get_G_less_w(self, orbital):
        """
        Compute lesser Green functions in frequency domain

        Arguments:
            orbital -- int

        Returns:
            mesh -- a Mesh instance for frequency coordinates
            gf -- 1D array
        """
        m, g = self.get_G_less(orbital)
        m, g = fourier_transform(m, g)
        return m, g

    def get_G_reta_w(self, orbital):
        """
        Compute retarded Green functions in frequency domain

        Arguments:
            orbital -- int

        Returns:
            mesh -- a Mesh instance for frequency coordinates
            gf -- 1D array
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
        """
        Compute the density of states in frequency domain from the retarded Green function

        Arguments:
            orbital -- int

        Returns:
            mesh -- a Mesh instance for frequency coordinates
            gf -- 1D array
        """
        m, G_less = self.get_G_less(orbital)
        m2, G_grea = self.get_G_grea(orbital)

        assert m is m2

        dos = 1j * (G_grea - G_less) / (2 * np.pi)
        m, dos = fourier_transform(m, dos)
        return m, np.real(dos)

    def get_occupation(self, orbital):
        """
        Compute occupation of given orbital.

        Arguments:
            orbital -- int

        Returns:
            occupation -- float
        """
        time_mesh, G_less = self.get_G_less(orbital)

        idx0 = len(time_mesh) // 2
        return G_less[idx0].imag

    def get_normalization_error(self):
        return self.core.get_normalization_error()


######## Shortcuts for often used solvers ########


def AIM_infinite_U(local_evol, time_mesh):
    """
    Return solver for a single-site Anderson impurity model with infinite Hubbard interaction

    Arguments:
        local_evol -- list of local evolutions (length 3, see doc of `SolverSteadyState`)
        time_mesh -- Mesh instance for time coordinates

    Returns:
        a SolverSteadyState instance
    """
    return SolverSteadyState(
        2, local_evol, time_mesh, orbital_names=["up", "dn"], forbidden_states=[3]
    )
