import numpy as np
import itertools
from .utilities import *

import matplotlib
from matplotlib import pyplot as plt


class NCA_Steady_State_Solver:
    def __init__(self, H_loc, delta_less_dict, delta_grea_dict, time_mesh):

        self.H_loc = np.asarray(H_loc)
        self.D = len(self.H_loc)
        self.Z_loc = self.D

        self.nr_orbitals = int(np.log2(self.D))
        if 2 ** self.nr_orbitals != self.D:
            raise ValueError

        self.delta_less_dict = delta_less_dict
        self.delta_grea_dict = delta_grea_dict
        self.time_mesh = time_mesh
        self.times = self.time_mesh.values()
        N = len(self.time_mesh)

        self.R_less = np.empty((N, self.D), dtype=complex)
        self.R_grea = np.empty((N, self.D), dtype=complex)
        self.R_grea_reta = np.zeros((N, self.D), dtype=complex)

        self.S_less = np.empty((N, self.D), dtype=complex)
        self.S_grea = np.empty((N, self.D), dtype=complex)
        self.S_grea_reta = np.zeros((N, self.D), dtype=complex)

        ### initialization
        active_orb = self.H_loc < np.inf
        for k, t in enumerate(self.time_mesh.values()):
            if t >= 0.0:
                self.R_grea_reta[k, active_orb] = -1j * np.exp(
                    -1.0j * self.H_loc[active_orb] * t
                )
            self.R_less[k, active_orb] = -1j * np.exp(1.0j * self.H_loc[active_orb] * t)

            self.R_grea_reta[k, ~active_orb] = 0.0
            self.R_less[k, ~active_orb] = 0.0

        self.freq_mesh = self.time_mesh.adjoint()
        self.freqs = self.freq_mesh.values()

        diff_H_loc = np.diff(np.unique(self.H_loc))
        diff_H_loc = diff_H_loc[np.isfinite(diff_H_loc)]
        if len(diff_H_loc) > 0:
            assert self.freq_mesh.delta < 0.1 * np.min(diff_H_loc)
            assert self.freq_mesh.xmax > 10 * np.max(diff_H_loc)

        self.R_grea_w = np.empty((N, self.D), dtype=complex)
        self.R_grea_reta_w = np.empty((N, self.D), dtype=complex)
        self.R_less_w = np.empty((N, self.D), dtype=complex)

        self.S_less_w = np.empty((N, self.D), dtype=complex)
        self.S_grea_w = np.empty((N, self.D), dtype=complex)
        self.S_grea_reta_w = np.empty((N, self.D), dtype=complex)

        self.N1 = 0
        self.N2 = 0

    def is_orb_in_state(self, orbital, basis_state):
        return (basis_state // 2 ** orbital) % 2 == 1

    def self_energy_grea(self):
        self.S_grea_reta[:] = 0.0

        for k in range(self.D):
            for x in range(self.nr_orbitals):
                if self.is_orb_in_state(x, k):
                    if self.H_loc[k - 2 ** x] < np.inf:
                        self.S_grea_reta[:, k] += (
                            1j
                            * self.delta_grea_dict[x]
                            * self.R_grea_reta[:, k - 2 ** x]
                        )
                else:
                    if self.H_loc[k + 2 ** x] < np.inf:
                        self.S_grea_reta[:, k] += (
                            -1j
                            * self.delta_less_dict[x][::-1]
                            * self.R_grea_reta[:, k + 2 ** x]
                        )

    def self_energy_less(self):
        self.S_less[:] = 0.0

        for k in range(self.D):
            for x in range(self.nr_orbitals):
                if self.is_orb_in_state(x, k):
                    if self.H_loc[k - 2 ** x] < np.inf:
                        self.S_less[:, k] += (
                            -1j * self.delta_less_dict[x] * self.R_less[:, k - 2 ** x]
                        )
                else:
                    if self.H_loc[k + 2 ** x] < np.inf:
                        self.S_less[:, k] += (
                            1j
                            * self.delta_grea_dict[x][::-1]
                            * self.R_less[:, k + 2 ** x]
                        )

    def greater_loop(
        self, tol=1e-8, min_iter=5, max_iter=100, eta=1.0, plot=False, verbose=False
    ):
        n = 0
        err = +np.inf
        active_orb = self.H_loc < np.inf
        eta_i = eta
        q = np.log(100.0) / np.log(min_iter)

        if plot:
            plt.plot(
                self.freq_mesh.values(),
                -self.R_grea_reta_w[:, 0].imag,
                label=str(self.N1),
            )

        while (err > tol and n < max_iter) or n < min_iter:

            self.self_energy_grea()
            _, self.S_grea_reta_w = fourier_transform(
                self.time_mesh, self.S_grea_reta, axis=0
            )

            R_grea_reta_w_prev = self.R_grea_reta_w.copy()
            self.R_grea_reta_w[:, ~active_orb] = 0.0
            self.R_grea_reta_w[:, active_orb] = (
                self.freq_mesh.values()[:, None]
                - self.H_loc[None, active_orb]
                - self.S_grea_reta_w[:, active_orb]
                + 1.0j * eta_i
            )
            if not np.all(np.isfinite(self.R_grea_reta_w[:, active_orb])):
                raise RuntimeError("WARNING: division by zero")
            self.R_grea_reta_w[:, active_orb] = 1.0 / self.R_grea_reta_w[:, active_orb]

            err = np.trapz(
                np.mean(np.abs(self.R_grea_reta_w - R_grea_reta_w_prev), axis=1),
                dx=self.freq_mesh.delta,
            )
            if verbose:
                print(self.N1, err)
            _, self.R_grea_reta = inv_fourier_transform(
                self.freq_mesh, self.R_grea_reta_w, axis=0
            )

            # TODO: why is the normalization not converging to 1?
            # print(2.j * self.R_grea_reta[len(self.time_mesh) // 2, :])
            for k in range(self.D):
                if active_orb[k]:
                    self.R_grea_reta[k] /= (
                        2.0j * self.R_grea_reta[len(self.time_mesh) // 2, k]
                    )

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
        _, self.R_grea = inv_fourier_transform(self.freq_mesh, self.R_grea_w, axis=0)
        # self.R_grea.set_from_fourier(self.R_grea_w)

        self.S_grea_w[:] = 2j * self.S_grea_reta_w.imag
        _, self.S_grea = inv_fourier_transform(self.freq_mesh, self.S_grea_w, axis=0)
        # self.S_grea.set_from_fourier(self.S_grea_w)

    def normalize_less(self):
        Z = 1j * np.mean(
            self.R_less[len(self.time_mesh) // 2], where=(self.H_loc < np.inf)
        )
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

    def get_G_grea(self, orb):
        if orb >= self.nr_orbitals:
            raise ValueError
        G_grea = np.zeros(len(self.time_mesh), dtype=complex)

        for state in range(self.D):
            if not self.is_orb_in_state(orb, state):
                G_grea += self.R_less[::-1, state] * self.R_grea[:, state + 2 ** orb]

        G_grea *= 1j / self.Z_loc
        return G_grea

    def get_G_less(self, orb):
        if orb >= self.nr_orbitals:
            raise ValueError
        G_less = np.zeros(len(self.time_mesh), dtype=complex)

        for state in range(self.D):
            if self.is_orb_in_state(orb, state):
                G_less += self.R_less[:, state] * self.R_grea[::-1, state - 2 ** orb]

        G_less *= -1j / self.Z_loc
        return G_less

    def get_DOS(self, orb):
        G_less = self.get_G_less(orb)
        G_grea = self.get_G_grea(orb)

        dos = 1j * (G_grea - G_less) / (2 * np.pi)
        return np.real(fourier_transform(self.time_mesh, dos)[1])
