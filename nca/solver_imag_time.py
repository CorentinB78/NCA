import numpy as np
from .utilities import *

from matplotlib import pyplot as plt
import toolbox as tb


class SolverImagTime:
    def __init__(self, beta, H_loc, delta_dict, time_mesh):

        self.beta = beta
        if (time_mesh.pt_on_value_adj) or (time_mesh.pt_on_value != self.beta):
            print(time_mesh.pt_on_value - self.beta)
            raise ValueError

        self.time_mesh = time_mesh
        self.times = self.time_mesh.values()
        N = len(self.time_mesh)

        self.H_loc = np.asarray(H_loc).copy()
        self.energy_shift = 0.0
        self.D = len(self.H_loc)

        min_en_threshold = -np.log(1e-10) / self.time_mesh.xmax  # > 0
        min_energy = np.min(self.H_loc.real)
        if min_energy < min_en_threshold:
            self.H_loc += min_en_threshold - min_energy
            self.energy_shift += min_en_threshold - min_energy

        self.nr_orbitals = int(np.log2(self.D))
        if 2 ** self.nr_orbitals != self.D:
            raise ValueError

        self.delta_dict = delta_dict

        self.R_tau = np.zeros((N, self.D), dtype=complex)
        self.S_tau = np.zeros((N, self.D), dtype=complex)

        ### initialization
        active_orb = self.H_loc < np.inf
        for k, tau in enumerate(self.time_mesh.values()):
            self.R_tau[k, active_orb] = -np.exp(-np.abs(self.H_loc[active_orb]) * tau)

        self.freq_mesh = self.time_mesh.adjoint()
        self.freqs = self.freq_mesh.values()

        self.R_iw = np.zeros((N, self.D), dtype=complex)
        self.S_iw = np.zeros((N, self.D), dtype=complex)

        self.N_loops = 0

    def is_orb_in_state(self, orbital, basis_state):
        return (basis_state // 2 ** orbital) % 2 == 1

    def self_energy(self):
        self.S_tau[:] = 0.0

        for k in range(self.D):
            for x in range(self.nr_orbitals):
                if self.is_orb_in_state(x, k):
                    if self.H_loc[k - 2 ** x] < np.inf:
                        self.S_tau[:, k] += (
                            -self.delta_dict[x] * self.R_tau[:, k - 2 ** x]
                        )
                else:
                    if self.H_loc[k + 2 ** x] < np.inf:
                        self.S_tau[:, k] += (
                            self.delta_dict[x][::-1] * self.R_tau[:, k + 2 ** x]
                        )

    def solve(self, tol=1e-8, min_iter=5, max_iter=100, plot=False, verbose=False):
        n = 0
        err = +np.inf
        active_orb = self.H_loc < np.inf
        idx0 = len(self.time_mesh) // 2
        min_en_threshold = -np.log(1e-10) / self.time_mesh.xmax  # > 0

        if plot:
            plt.plot(
                self.freq_mesh.values(),
                -self.R_iw[:, 0].real,
                label=str(self.N_loops),
            )

        while (err > tol and n < max_iter) or n < min_iter:

            self.self_energy()

            # temporarly store \tilde S^>(t)
            self.S_iw[:idx0, :] = 0.0
            self.S_iw[idx0:, :] = self.S_tau[idx0:, :]
            self.S_iw[idx0, :] *= 0.5
            _, self.S_iw = fourier_transform(self.time_mesh, self.S_iw, axis=0)

            R_iw_prev = self.R_iw.copy()
            self.R_iw[:, ~active_orb] = 0.0
            self.R_iw[:, active_orb] = (
                self.H_loc[None, active_orb] + self.S_iw[:, active_orb]
            )
            min_energy = np.min(self.R_iw[:, active_orb].real)
            if min_energy <= min_en_threshold:
                self.R_iw[:, active_orb] += min_en_threshold - min_energy
                self.H_loc += min_en_threshold - min_energy
                self.energy_shift += min_en_threshold - min_energy

            self.R_iw[:, active_orb] *= -1
            self.R_iw[:, active_orb] += 1j * self.freq_mesh.values()[:, None]
            self.R_iw[:, active_orb] = 1.0 / self.R_iw[:, active_orb]

            err = np.trapz(
                np.mean(np.abs(self.R_iw - R_iw_prev), axis=1),
                dx=self.freq_mesh.delta,
            )
            if verbose:
                print(self.N_loops, err)

            ### symetrized FT
            _, self.R_tau = inv_fourier_transform(
                self.freq_mesh, 2 * self.R_iw.real, axis=0
            )

            for k in range(self.D):
                if active_orb[k]:
                    self.R_tau[:, k] /= -self.R_tau[idx0, k]

            self.N_loops += 1
            n += 1

            if plot:
                plt.plot(
                    self.freq_mesh.values(),
                    -self.R_iw[:, 0].imag,
                    label=str(self.N_loops),
                )

        if verbose:
            print("Done.")
            print()

        if plot:
            plt.legend()
            plt.xlim(-20, 15)

        if err > tol:
            print(f"WARNING: poor convergence, err={err}")

    def get_unshifted_R_tau(self):
        return self.R_tau * np.exp(self.time_mesh.values()[:, None] * self.energy_shift)

    def get_unshifted_S_tau(self):
        return self.S_tau * np.exp(self.time_mesh.values()[:, None] * self.energy_shift)

    def get_Z_ratio(self):
        """Return Z / Z_bath"""
        idx = self.time_mesh.idx_pt_on_value
        return -np.sum(self.R_tau[idx, :].real * np.exp(self.beta * self.energy_shift))

    def get_density_matrix(self):
        idx = self.time_mesh.idx_pt_on_value
        rho = self.get_unshifted_R_tau()[idx, :]
        return rho / np.sum(rho)

    def get_G_tau(self, orb):
        if orb >= self.nr_orbitals:
            raise ValueError

        times_cut, R_tau_cut = tb.vcut(
            self.time_mesh.values(), self.R_tau, 0.0, self.beta, axis=0
        )

        G_tau = np.zeros(len(times_cut), dtype=float)

        for state in range(self.D):
            if not self.is_orb_in_state(orb, state):
                G_tau += np.real(
                    R_tau_cut[::-1, state] * R_tau_cut[:, state + 2 ** orb]
                )

        idx = self.time_mesh.idx_pt_on_value
        G_tau /= np.sum(self.R_tau[idx, :].real)
        return times_cut, G_tau
