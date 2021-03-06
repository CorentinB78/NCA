import numpy as np
from .function_tools import *
from .utilities import print_warning_large_error

from matplotlib import pyplot as plt


### TODO: restrict real quantities to float
### TODO: write notes on implementation (energy shift, def of R(tau > beta) , etc)
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
            self.energy_shift += min_en_threshold - min_energy

        self.nr_orbitals = int(np.log2(self.D))
        if 2**self.nr_orbitals != self.D:
            raise ValueError

        self.delta_dict = delta_dict

        self.R_tau = np.zeros((N, self.D), dtype=complex)
        self.S_tau = np.zeros((N, self.D), dtype=complex)

        ### initialization
        active_orb = self.H_loc < np.inf
        for k, tau in enumerate(self.time_mesh.values()):
            self.R_tau[k, active_orb] = -np.exp(
                -(self.H_loc[active_orb] + self.energy_shift) * np.abs(tau)
            )

        self.freq_mesh = self.time_mesh.adjoint()
        self.freqs = self.freq_mesh.values()

        self.R_iw = np.zeros((N, self.D), dtype=complex)
        self.S_iw = np.zeros((N, self.D), dtype=complex)

        self.N_loops = 0

    def is_orb_in_state(self, orbital, basis_state):
        return (basis_state // 2**orbital) % 2 == 1

    def self_energy(self):
        self.S_tau[:] = 0.0
        idx0 = len(self.time_mesh) // 2
        idxb = self.time_mesh.idx_pt_on_value

        for k in range(self.D):
            for x in range(self.nr_orbitals):
                if self.is_orb_in_state(x, k):
                    if self.H_loc[k - 2**x] < np.inf:
                        self.S_tau[idx0 : idxb + 1, k] += (
                            -self.delta_dict[x][idx0 : idxb + 1]
                            * self.R_tau[idx0 : idxb + 1, k - 2**x]
                        )
                else:
                    if self.H_loc[k + 2**x] < np.inf:
                        self.S_tau[idx0 : idxb + 1, k] += (
                            -self.delta_dict[x][idxb : idx0 - 1 : -1]
                            * self.R_tau[idx0 : idxb + 1, k + 2**x]
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

            _, self.S_iw = fourier_transform(self.time_mesh, self.S_tau, axis=0)

            R_iw_prev = self.R_iw.copy()
            self.R_iw[:, ~active_orb] = 0.0
            self.R_iw[:, active_orb] = (
                self.H_loc[None, active_orb]
                + self.S_iw[:, active_orb]
                + self.energy_shift
            )

            min_energy = np.min(self.R_iw[:, active_orb].real)
            if min_energy <= min_en_threshold:
                self.energy_shift += min_en_threshold - min_energy
                self.S_tau[:, active_orb] *= np.exp(
                    (min_energy - min_en_threshold) * self.time_mesh.values()[:, None]
                )

                ### redo with updated shift
                _, self.S_iw = fourier_transform(self.time_mesh, self.S_tau, axis=0)

                self.R_iw[:, active_orb] = (
                    self.H_loc[None, active_orb]
                    + self.S_iw[:, active_orb]
                    + self.energy_shift
                )

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
                    -self.R_iw[:, 0].real,
                    label=str(self.N_loops),
                )

        if verbose:
            print("Done.")
            print()

        if plot:
            plt.legend()
            plt.xlim(-20, 15)

        print_warning_large_error(
            f"Imag. times fixed point loop: Poor convergence. Error={err}",
            err,
            tolw=tol,
            tole=1e-3,
        )

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

        times_cut, R_tau_cut = vcut(
            self.time_mesh.values(),
            self.R_tau,
            0.0 - self.time_mesh.delta / 2.0,  # including 0.
            self.beta + self.time_mesh.delta / 2.0,  # including beta
            axis=0,
        )

        G_tau = np.zeros(len(times_cut), dtype=float)

        for state in range(self.D):
            if not self.is_orb_in_state(orb, state):
                G_tau += np.real(
                    R_tau_cut[::-1, state] * R_tau_cut[:, state + 2**orb]
                )

        idx = self.time_mesh.idx_pt_on_value
        G_tau /= np.sum(self.R_tau[idx, :].real)
        return times_cut, G_tau


def vcut(coord, values, left=None, right=None, axis=-1):
    """
    Cut the coordinate and values arrays of a sampled function so as to reduce
    its coordinate range to [`left`, `right`].

    Return views.
    None means infinity.
    """
    coord_out = np.asarray(coord)
    values_out = np.swapaxes(values, 0, axis)

    if left is not None:
        left_i = np.searchsorted(coord_out, [left])[0]
        coord_out = coord_out[left_i:]
        values_out = values_out[left_i:]
    if right is not None:
        right_i = np.searchsorted(coord_out, [right])[0]
        if right_i < len(coord_out):
            coord_out = coord_out[:right_i]
            values_out = values_out[:right_i]

    values_out = np.swapaxes(values_out, 0, axis)

    return coord_out, values_out
