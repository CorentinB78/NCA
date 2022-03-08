import numpy as np
import itertools
from .utilities import *

import matplotlib
from matplotlib import pyplot as plt


class NCA_Steady_State_Solver:
    def __init__(self, H_loc, time_mesh, hybridizations):
        """
        Real time Non-Crossing Approximation (NCA) solver for steady states.

        For now only diagonal hybridizations and local hamiltonians are supported. TODO.

        H_loc: diagonal of the (D, D) hamiltonian matrix, D is the number of states in the local system.
        time_mesh: an instance of `Mesh`.
        hybridizations: list of hybridization processes. Each process is a tuple (a, b, delta_grea, delta_less) where
            a, b are states (identified by an int within range(D)) and delta_grea/less are 1D arrays containing hybridization functions (as sampled on `time_mesh`). delta_grea is the one participating to the greater SE, while delta_less is for the lesser SE. The process changes the local system from a to b then back to a. Conjugate processes are not added automatically.
            Optionnaly, several processes can be regrouped if they share the same hybridization functions, then a and b should be 1D arrays.
        """

        self.H_loc = np.asarray(H_loc)
        self.D = len(self.H_loc)
        self.Z_loc = self.D

        self.hybridizations = hybridizations
        self.time_mesh = time_mesh
        self.times = self.time_mesh.values()
        N = len(self.time_mesh)

        self.R_less = np.zeros((N, self.D), dtype=complex)
        self.R_grea = np.zeros((N, self.D), dtype=complex)

        self.S_less = np.zeros((N, self.D), dtype=complex)
        self.S_grea = np.zeros((N, self.D), dtype=complex)

        ### initialization
        for k, t in enumerate(self.time_mesh.values()):
            self.R_grea[k, :] = -1j * np.exp(-1j * self.H_loc * t)
            self.R_less[k, :] = -1j * np.exp(1j * self.H_loc * t)

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

        self.N1 = 0
        self.N2 = 0

    def self_energy_grea(self):
        self.S_grea[:, :] = 0.0

        for state_a, state_b, delta, _ in self.hybridizations:
            state_a, state_b = np.atleast_1d(state_a, state_b)
            self.S_grea[:, state_a] += 1j * delta[:, None] * self.R_grea[:, state_b]

    def self_energy_less(self):
        self.S_less[:, :] = 0.0

        for state_a, state_b, _, delta in self.hybridizations:
            state_a, state_b = np.atleast_1d(state_a, state_b)
            self.S_less[:, state_a] += -1j * delta[:, None] * self.R_less[:, state_b]

    def greater_loop(
        self, tol=1e-8, min_iter=5, max_iter=100, eta=1.0, plot=False, verbose=False
    ):
        n = 0
        err = +np.inf
        eta_i = eta
        q = np.log(100.0) / np.log(min_iter)
        idx0 = len(self.time_mesh) // 2

        if plot:
            plt.plot(
                self.freq_mesh.values(),
                -self.R_grea_reta_w[:, 0].imag,
                label=str(self.N1),
            )

        while (err > tol and n < max_iter) or n < min_iter:

            self.self_energy_grea()

            # temporarly store \tilde S^>(t)
            self.S_grea_reta_w[:idx0, :] = 0.0
            self.S_grea_reta_w[idx0:, :] = self.S_grea[idx0:, :]
            self.S_grea_reta_w[idx0, :] *= 0.5
            _, self.S_grea_reta_w = fourier_transform(
                self.time_mesh, self.S_grea_reta_w, axis=0
            )

            # self.S_grea_reta_w[:] -= self.S_grea_reta_w[0, :]  # vanish at inf freq

            R_grea_reta_w_prev = self.R_grea_reta_w.copy()
            self.R_grea_reta_w[:, :] = (
                self.freq_mesh.values()[:, None]
                - self.H_loc[None, :]
                - self.S_grea_reta_w[:, :]
                + 1.0j * eta_i
            )
            if not np.all(np.isfinite(self.R_grea_reta_w[:, :])):
                raise RuntimeError("WARNING: division by zero")
            self.R_grea_reta_w[:, :] = 1.0 / self.R_grea_reta_w[:, :]

            err = np.trapz(
                np.mean(np.abs(self.R_grea_reta_w - R_grea_reta_w_prev), axis=1),
                dx=self.freq_mesh.delta,
            )
            if verbose:
                print(self.N1, err)
            _, self.R_grea = inv_fourier_transform(
                self.freq_mesh, 2j * self.R_grea_reta_w.imag, axis=0
            )

            for k in range(self.D):
                self.R_grea[:, k] /= 1.0j * self.R_grea[idx0, k]

            self.N1 += 1
            n += 1
            if n >= min_iter:
                eta_i = 0.0
            else:
                eta_i /= q

            if plot:
                plt.plot(
                    self.freq_mesh.values(),
                    -self.R_grea_reta_w[:, 0].imag,
                    label=str(self.N1),
                )

        if verbose:
            print("Done.")
            print()

        if plot:
            plt.legend()
            plt.xlim(-20, 15)

        if err > tol:
            print(f"WARNING: poor convergence, err={err}")

        self.R_grea_w[:] = 2j * self.R_grea_reta_w.imag
        self.S_grea_w[:] = 2j * self.S_grea_reta_w.imag

    def normalize_less(self):
        Z = 1j * np.mean(self.R_less[len(self.time_mesh) // 2])
        if Z == 0.0:
            print(self.R_less[len(self.time_mesh) // 2])
            raise ZeroDivisionError
        self.R_less_w /= Z
        self.R_less /= Z

    def lesser_loop(self, tol=1e-8, max_iter=100, plot=False, verbose=False):
        # self.normalize_less()
        n = 0
        err = +np.inf

        while err > tol and n < max_iter:
            self.self_energy_less()
            _, self.S_less_w = fourier_transform(self.time_mesh, self.S_less, axis=0)
            # self.S_less_w.set_from_fourier(self.S_less)

            R_less_w_prev = self.R_less_w.copy()
            # if self.N2 > 10:
            #     self.R_less_w[:] = 0.25 * R_less_w_prev + 0.75 * self.R_grea_reta_w * self.S_less_w * np.conj(self.R_grea_reta_w)
            # elif self.N2 > 5:
            #     self.R_less_w[:] = (R_less_w_prev + self.R_grea_reta_w * self.S_less_w * np.conj(self.R_grea_reta_w)) / 2.
            # else:
            #     self.R_less_w[:] = self.R_grea_reta_w * self.S_less_w * np.conj(self.R_grea_reta_w)

            self.R_less_w[:] = (
                0.25 * R_less_w_prev
                + 0.75
                * self.R_grea_reta_w
                * self.S_less_w
                * np.conj(self.R_grea_reta_w)
            )
            _, self.R_less = inv_fourier_transform(
                self.freq_mesh, self.R_less_w, axis=0
            )
            # self.R_less.set_from_fourier(self.R_less_w)
            self.normalize_less()

            err = np.trapz(
                np.mean(np.abs(self.R_less_w - R_less_w_prev), axis=1),
                dx=self.freq_mesh.delta,
            )
            if verbose:
                print(self.N2, err)

            self.N2 += 1
            n += 1

            if plot:
                plt.plot(
                    self.freq_mesh.values(),
                    self.S_less_w[:, 0].imag,
                    label=str(self.N2),
                )

        if verbose:
            print("Done.")
            print()

        if plot:
            plt.legend()
            plt.xlim(-20, 15)

        if err > tol:
            print(f"WARNING: poor convergence, err={err}")


class FermionicFockSpace:
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
