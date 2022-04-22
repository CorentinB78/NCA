import numpy as np
from .utilities import fourier_transform


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
        if orbital >= self.nr_orbitals:
            raise ValueError

        states_yes, states_no = states_containing(orbital, self.nr_orbitals)
        G_grea = np.sum(
            solver.R_less[::-1, states_no]
            * solver.R_grea[:, states_yes]
            * np.exp(
                1j
                * (solver.energy_shift[states_no] - solver.energy_shift[states_yes])
                * solver.times[:, None]
            ),
            axis=1,
        )

        G_grea *= 1j / solver.Z_loc
        return G_grea

    def get_G_less(self, orbital, solver):
        """Returns G^<(t) on time grid used in solver"""
        if orbital >= self.nr_orbitals:
            raise ValueError

        states_yes, states_no = states_containing(orbital, self.nr_orbitals)
        G_less = np.sum(
            solver.R_less[:, states_yes]
            * solver.R_grea[::-1, states_no]
            * np.exp(
                1j
                * (solver.energy_shift[states_no] - solver.energy_shift[states_yes])
                * solver.times[:, None]
            ),
            axis=1,
        )

        G_less *= -1j / solver.Z_loc
        return G_less

    def get_G_grea_w(self, orbital, solver):
        g = self.get_G_grea(orbital, solver)
        _, g = fourier_transform(solver.time_mesh, g)
        return g

    def get_G_less_w(self, orbital, solver):
        g = self.get_G_less(orbital, solver)
        _, g = fourier_transform(solver.time_mesh, g)
        return g

    def get_DOS(self, orbital, solver):
        """Returns density of states on frequency grid used in solver"""
        G_less = self.get_G_less(orbital, solver)
        G_grea = self.get_G_grea(orbital, solver)

        dos = 1j * (G_grea - G_less) / (2 * np.pi)
        return np.real(fourier_transform(solver.time_mesh, dos)[1])


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

        G_grea = 1j * solver.R_less[::-1, 0] * solver.R_grea[:, 1]
        G_grea *= np.exp(
            -1j * (solver.energy_shift[1] - solver.energy_shift[0]) * solver.times
        )

        G_grea /= solver.Z_loc
        return G_grea

    def get_G_less(self, orbital, solver):
        """Returns G^<(t) on time grid used in solver"""
        if orbital >= self.nr_orbitals:
            raise ValueError

        G_less = -1j * solver.R_less[:, 1] * solver.R_grea[::-1, 0]
        G_less *= np.exp(
            -1j * (solver.energy_shift[1] - solver.energy_shift[0]) * solver.times
        )

        G_less /= solver.Z_loc
        return G_less


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
    w_ref, R_less_w_ref = fourier_transform(S.time_mesh, S.R_less, axis=0)

    report_allclose(w_ref.values(), S.freqs)
    report_allclose(R_less_w_ref, S.R_less_w, atol=1e-4)

    _, R_grea_w_ref = fourier_transform(S.time_mesh, S.R_grea, axis=0)
    report_allclose(R_grea_w_ref, S.R_grea_w, atol=1e-4)

    _, S_less_w_ref = fourier_transform(S.time_mesh, S.S_less, axis=0)
    report_allclose(S_less_w_ref, S.S_less_w, atol=1e-4)

    _, S_grea_w_ref = fourier_transform(S.time_mesh, S.S_grea, axis=0)
    report_allclose(S_grea_w_ref, S.S_grea_w, atol=1e-4)

    ### symmetries: diagonal lessers and greaters are pure imaginary
    report_allclose(S.R_less_w.real, 0.0, atol=1e-8)
    report_allclose(S.R_grea_w.real, 0.0, atol=1e-8)
    report_allclose(S.S_less_w.real, 0.0, atol=1e-8)
    report_allclose(S.S_grea_w.real, 0.0, atol=1e-8)

    ### normalization
    idx0 = len(S.times) // 2

    for k in range(S.D):
        report_allclose(S.R_grea[idx0, k], -1j)

    report_allclose(np.sum(S.R_less[idx0, :]), -1j * S.Z_loc, 2)

    ### Green functions
    if fock is not None:

        for k in range(fock.nr_orbitals):
            G_grea = fock.get_G_grea(k, S)
            G_less = fock.get_G_less(k, S)
            Dos_w = fock.get_DOS(k, S)

            _, G_grea_w = fourier_transform(S.time_mesh, G_grea)
            _, G_less_w = fourier_transform(S.time_mesh, G_less)

            ### normalization and DoS
            Dos_w_ref = np.real(1j * (G_grea_w - G_less_w) / (2 * np.pi))
            report_allclose(Dos_w_ref, Dos_w, atol=1e-8)
            report_allclose(np.trapz(x=S.freqs, y=Dos_w), 1.0, atol=1e-6)

            ### Symmetries: diagonal lessers and greaters are pure imaginary and do not change sign
            report_allclose(G_grea_w.real, 0.0, atol=1e-8)
            report_allclose(G_less_w.real, 0.0, atol=1e-8)
            report_less(G_grea_w.imag, 1e-8)
            report_less(-G_less_w.imag, 1e-8)
