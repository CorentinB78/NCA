import numpy as np
from .utilities import fourier_transform, product_functions, sum_functions


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


def greater_gf(orbital, nr_orbitals, time_meshes, R_grea, R_less, energy_shift, Z):
    if orbital >= nr_orbitals:
        raise ValueError

    states_yes, states_no = states_containing(orbital, nr_orbitals)

    mesh = None
    G_grea = 0.0
    for i in range(len(states_yes)):
        s_no, s_yes = states_no[i], states_yes[i]
        m, gg = product_functions(
            time_meshes[s_no],
            R_less[::-1, s_no],
            time_meshes[s_yes],
            R_grea[:, s_yes],
        )

        gg = gg * np.exp(1j * (energy_shift[s_no] - energy_shift[s_yes]) * m.values())

        mesh, G_grea = sum_functions(mesh, G_grea, m, gg)

    G_grea *= 1j / Z
    return mesh, G_grea


def lesser_gf(orbital, nr_orbitals, time_meshes, R_grea, R_less, energy_shift, Z):
    if orbital >= nr_orbitals:
        raise ValueError

    states_yes, states_no = states_containing(orbital, nr_orbitals)

    mesh = None
    G_less = 0.0
    for i in range(len(states_yes)):
        s_no, s_yes = states_no[i], states_yes[i]
        m, gg = product_functions(
            time_meshes[s_no],
            R_grea[::-1, s_no],
            time_meshes[s_yes],
            R_less[:, s_yes],
        )

        gg = gg * np.exp(1j * (energy_shift[s_no] - energy_shift[s_yes]) * m.values())

        mesh, G_less = sum_functions(mesh, G_less, m, gg)

    G_less *= -1j / Z
    return mesh, G_less


class FermionicFockSpace:
    # TODO: method for list of even states
    def __init__(self, orbital_names):
        self.orbital_names = orbital_names
        self.nr_orbitals = len(orbital_names)
        self.baths = []

    def state_string(self, state):
        """
        Represent a state as a string of the filled orbital names
        """
        s = []

        for k in range(self.nr_orbitals):
            if (state % 2) == 1:
                s.append(self.orbital_names[k])
            state = state // 2

        return ",".join(s)

    def basis(self):
        all_states = np.arange(2**self.nr_orbitals)
        out = [self.state_string(s) for s in all_states]
        return out

    def add_bath(self, orbital, delta_grea, delta_less):
        """Only baths coupled to a single orbital for now"""
        self.baths.append((orbital, delta_grea, delta_less))

    def generate_hybridizations(self):
        hyb = []

        for orbital, delta_grea, delta_less in self.baths:
            states_a, states_b = states_containing(orbital, self.nr_orbitals)

            ### particle processes
            hyb.append((states_a, states_b, delta_grea, delta_less))

            ### hole processes
            hyb.append((states_b, states_a, np.conj(delta_less), np.conj(delta_grea)))

        return hyb

    def get_G_grea(self, orbital, solver):
        """Returns G^>(t) on time grid used in solver"""
        return greater_gf(
            orbital,
            self.nr_orbitals,
            solver.time_meshes,
            solver.R_grea,
            solver.R_less,
            solver.energy_shift,
            solver.Z_loc,
        )

    def get_G_less(self, orbital, solver):
        """Returns G^<(t) on time grid used in solver"""
        return lesser_gf(
            orbital,
            self.nr_orbitals,
            solver.time_meshes,
            solver.R_grea,
            solver.R_less,
            solver.energy_shift,
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

    def get_DOS(self, orbital, solver):
        """Returns density of states"""
        m, G_less = self.get_G_less(orbital, solver)
        m2, G_grea = self.get_G_grea(orbital, solver)

        assert m is m2

        dos = 1j * (G_grea - G_less) / (2 * np.pi)
        m, dos = fourier_transform(m, dos)
        return m, np.real(dos)


class AIM_infinite_U(FermionicFockSpace):
    # TODO: method for list of even states
    def __init__(self):
        self.orbital_names = ["up", "dn"]
        self.nr_orbitals = 2
        self.baths = []

    def state_string(self, state):
        """
        Represent a state as a string of the filled orbital names
        """
        if state >= 3:
            raise ValueError(f"State {state} does not exist")

        s = []

        for k in range(self.nr_orbitals):
            if (state % 2) == 1:
                s.append(self.orbital_names[k])
            state = state // 2

        return ",".join(s)

    def basis(self):
        all_states = np.arange(2**self.nr_orbitals)[:3]
        out = [self.state_string(s) for s in all_states]
        return out

    def is_orb_in_state(self, orbital, state):
        """
        Return a mask over the list of states indicating which ones have an orbital occupated.
        """
        if state >= 3:
            raise ValueError(f"State {state} does not exist")

        return (state // 2**orbital) % 2 == 1

    def generate_hybridizations(self):
        hybs = []

        for orbital, delta_grea, delta_less in self.baths:
            if orbital == 0:
                hybs.append((1, 0, delta_grea, delta_less))
                hybs.append((0, 1, np.conj(delta_less), np.conj(delta_grea)))
            elif orbital == 1:
                hybs.append((2, 0, delta_grea, delta_less))
                hybs.append((0, 2, np.conj(delta_less), np.conj(delta_grea)))
            else:
                raise RuntimeError

        return hybs

    def get_G_grea(self, orbital, solver):
        """Returns G^>(t) on time grid used in solver"""
        if orbital >= self.nr_orbitals:
            raise ValueError

        mesh, G_grea = product_functions(
            solver.time_meshes[0],
            solver.R_less[::-1, 0],
            solver.time_meshes[1],
            solver.R_grea[:, 1],
        )

        G_grea *= 1j * np.exp(
            -1j * (solver.energy_shift[1] - solver.energy_shift[0]) * mesh.values()
        )

        G_grea /= solver.Z_loc
        return mesh, G_grea

    def get_G_less(self, orbital, solver):
        """Returns G^<(t) on time grid used in solver"""
        if orbital >= self.nr_orbitals:
            raise ValueError

        mesh, G_less = product_functions(
            solver.time_meshes[1],
            solver.R_less[:, 1],
            solver.time_meshes[0],
            solver.R_grea[::-1, 0],
        )
        G_less *= -1j * np.exp(
            -1j * (solver.energy_shift[1] - solver.energy_shift[0]) * mesh.values()
        )

        G_less /= solver.Z_loc
        return mesh, G_less


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
