import numpy as np
import string
from .function_tools import fourier_transform, product_functions, sum_functions


def is_orb_in_state(orbital, state):
    """
    Return True if orbital `orbital` is occupied in state `state`.
    """
    return (state // 2**orbital) % 2 == 1


def states_containing(orbital, nr_orbitals):
    """
    Return two lists, the first containing the states for which `orbital` is occupated, the second containing the other states.

    States of same index only differs from the occupation of `orbital`.
    # TODO: test
    """
    all_states = np.arange(2**nr_orbitals)
    contains = is_orb_in_state(orbital, all_states)
    return all_states[contains], all_states[~contains]


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


class StateSpace:
    def __init__(self, nr_orbitals, orbital_names=None, forbidden_states=None):
        if orbital_names is not None:
            assert len(orbital_names) == nr_orbitals
            self.orbital_names = orbital_names
        else:
            self.orbital_names = list(string.ascii_lowercase)[:nr_orbitals]

        self.nr_orbitals = nr_orbitals

        if forbidden_states is not None:
            self.forbidden_states = forbidden_states
        else:
            self.forbidden_states = []

        all_states = np.arange(2**self.nr_orbitals)
        self._basis = [self.get_state_label(s) for s in all_states]

        idx = []
        for i in range(len(all_states)):
            if all_states[i] in self.forbidden_states:
                idx.append(i)

        self._all_states = np.delete(all_states, idx)

    def __contains__(self, state):
        if state < 0 or state >= 2**self.nr_orbitals:
            return False
        if state in self.forbidden_states:
            return False
        return True

    def get_state_label(self, state):
        """
        Represent a state as a string of the filled orbital names
        """
        s = []

        for k in range(self.nr_orbitals):
            if (state % 2) == 1:
                s.append(self.orbital_names[k])
            state = state // 2

        return ",".join(s)

    @property
    def basis(self):
        return self._basis

    @property
    def all_states(self):
        return self._all_states

    def get_state_pairs_from_orbital(self, orbital):
        if orbital >= self.nr_orbitals:
            raise ValueError
        a, b = states_containing(orbital, self.nr_orbitals)

        if len(self.forbidden_states) > 0:
            idx = []
            for i in range(len(a)):
                if a[i] in self.forbidden_states or b[i] in self.forbidden_states:
                    idx.append(i)

            a = np.delete(a, idx)
            b = np.delete(b, idx)

        return a, b


class FermionicFockSpace:
    # TODO: method for list of even states
    def __init__(self, orbital_names, forbidden_states=None):
        self.state_space = StateSpace(
            len(orbital_names), orbital_names, forbidden_states
        )
        self.hybs = {}
        for s in self.state_space.all_states:
            self.hybs[s] = []
        self.baths = []

    def add_bath(self, orbital, delta_grea, delta_less):
        """Only baths coupled to a single orbital for now"""
        states_a, states_b = self.state_space.get_state_pairs_from_orbital(orbital)

        for a, b in zip(states_a, states_b):
            self.hybs[a].append((b, delta_grea, delta_less))
            self.hybs[b].append((a, np.conj(delta_less), np.conj(delta_grea)))

    def generate_hybridizations(self):
        return self.hybs

    def get_G_grea(self, orbital, solver):
        """Returns G^>(t) on time grid used in solver"""
        return greater_gf(
            orbital,
            self.state_space,
            solver.time_mesh,
            solver.get_R_grea(),
            solver.get_R_less(),
            solver.Z_loc,
        )

    def get_G_less(self, orbital, solver):
        """Returns G^<(t) on time grid used in solver"""
        return lesser_gf(
            orbital,
            self.state_space,
            solver.time_mesh,
            solver.get_R_grea(),
            solver.get_R_less(),
            solver.Z_loc,
        )

    def get_G_grea_w(self, orbital, solver):
        m, g = self.get_G_grea(orbital, solver)
        m, g = fourier_transform(m, g)
        return m, g

    def get_G_less_w(self, orbital, solver):
        m, g = self.get_G_less(orbital, solver)
        m, g = fourier_transform(m, g)
        return m, g

    def get_G_reta_w(self, orbital, solver):
        """
        Returns the retarded Green function in frequencies
        """
        m, G_less = self.get_G_less(orbital, solver)
        m2, G_grea = self.get_G_grea(orbital, solver)

        assert m is m2

        G_reta = G_grea - G_less
        idx0 = solver.N // 2
        G_reta[:idx0] = 0.0
        G_reta[idx0] *= 0.5
        m, G_reta_w = fourier_transform(m, G_reta)
        return m, G_reta_w

    def get_DOS(self, orbital, solver):
        """Returns density of states"""
        m, G_less = self.get_G_less(orbital, solver)
        m2, G_grea = self.get_G_grea(orbital, solver)

        assert m is m2

        dos = 1j * (G_grea - G_less) / (2 * np.pi)
        m, dos = fourier_transform(m, dos)
        return m, np.real(dos)


def AIM_infinite_U():
    return FermionicFockSpace(["up", "dn"], [3])


def report_allclose(a, b, *args, **kwargs):
    print(np.max(np.abs(a - b)))


def report_less(a, b, *args, **kwargs):
    diff = a - b
    mask = diff >= 0.0
    if mask.any():
        print(np.max(diff[mask]))
    else:
        print(0.0)


def sanity_checks(S, fock=None):

    ### R & S

    ### Fourier transforms
    for i in range(S.D):
        w_ref, R_less_w_ref = fourier_transform(
            S.time_meshes[i], S.R_less[:, i], axis=0
        )

        report_allclose(R_less_w_ref, S.R_less_w[:, i], atol=1e-4)

        _, R_grea_w_ref = fourier_transform(S.time_meshes[i], S.R_grea[:, i], axis=0)
        report_allclose(R_grea_w_ref, S.R_grea_w[:, i], atol=1e-4)

        _, S_less_w_ref = fourier_transform(S.time_meshes[i], S.S_less[:, i], axis=0)
        report_allclose(S_less_w_ref, S.S_less_w[:, i], atol=1e-4)

        _, S_grea_w_ref = fourier_transform(S.time_meshes[i], S.S_grea[:, i], axis=0)
        report_allclose(S_grea_w_ref, S.S_grea_w[:, i], atol=1e-4)

    ### symmetries: diagonal lessers and greaters are pure imaginary
    report_allclose(S.R_less_w.real, 0.0, atol=1e-8)
    report_allclose(S.R_grea_w.real, 0.0, atol=1e-8)
    report_allclose(S.S_less_w.real, 0.0, atol=1e-8)
    report_allclose(S.S_grea_w.real, 0.0, atol=1e-8)

    ### normalization
    idx0 = S.N // 2

    for k in range(S.D):
        report_allclose(S.R_grea[idx0, k], -1j)

    report_allclose(np.sum(S.R_less[idx0, :]), -1j * S.Z_loc, 2)

    ### Green functions
    if fock is not None:

        for k in range(fock.nr_orbitals):
            m_grea, G_grea = fock.get_G_grea(k, S)
            m_less, G_less = fock.get_G_less(k, S)
            m_dos, Dos_w = fock.get_DOS(k, S)

            _, G_grea_w = fourier_transform(m_grea, G_grea)
            _, G_less_w = fourier_transform(m_less, G_less)

            ### normalization and DoS
            Dos_w_ref = np.real(1j * (G_grea_w - G_less_w) / (2 * np.pi))
            report_allclose(Dos_w_ref, Dos_w, atol=1e-8)
            report_allclose(np.trapz(x=m_dos.values(), y=Dos_w), 1.0, atol=1e-6)

            ### Symmetries: diagonal lessers and greaters are pure imaginary and do not change sign
            report_allclose(G_grea_w.real, 0.0, atol=1e-8)
            report_allclose(G_less_w.real, 0.0, atol=1e-8)
            report_less(G_grea_w.imag, 1e-8)
            report_less(-G_less_w.imag, 1e-8)
