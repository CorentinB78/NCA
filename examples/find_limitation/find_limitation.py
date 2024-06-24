"""
Simple Anderson impurity model coupled to a thermal bath.
Find hardest instance possible.
"""
import nca
from collections import namedtuple
from time import time as now
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt


def solve(eps, beta, omega_max, delta_omega):
    start = now()
    print()
    print(f"### omega_max={omega_max}, delta_omega={delta_omega} ###")
    nb_samples = int(0.5 + 2 * omega_max / delta_omega)
    if nb_samples >= 100_000_000:
        raise RuntimeError(f"nb_samples={nb_samples} is >= 3e8")
    time_mesh = nca.Mesh(omega_max, nb_samples).adjoint()
    # print(time_mesh)

    ### local (diagonal) Hamiltonian
    H_loc = np.array([0.0, eps, eps])  # basis: 0, up, dn, (updn -> forbidden)

    ### Hybridization to a semicircular bath
    Gamma = 1.0  # Hybridization strength
    # beta = 100.0  # inverse temperature
    Ef = 0.0  # Fermi level
    D = 20.0  # half bandwidth

    dos = nca.make_gaussian_dos(D)
    # dos = nca.make_semicircular_dos(D)
    hyb_grea, hyb_less = nca.make_hyb_times(dos, beta, Ef, Gamma, time_mesh)

    ### solver
    S = nca.AIM_infinite_U(H_loc, time_mesh, order=6)

    S.add_bath(0, hyb_grea, hyb_less)  # orbital 0 -> up
    S.add_bath(1, hyb_grea, hyb_less)  # orbital 1 -> down

    ### calculation
    S.greater_loop(max_iter=20, verbose=False)

    R_grea_w = S.get_R_grea_w()
    runtime = now() - start
    print(f"Runtime: {runtime} s")
    return S.freq_mesh, R_grea_w, runtime


def discrepancy(omegas_1, values_1, omegas_2, values_2):
    omegas_1 = np.asarray(omegas_1)
    omegas_2 = np.asarray(omegas_2)
    # wmax = min(omegas_1[-1], omegas_2[-1])
    wmax = 20.  # half bandwidth
    interp_1 = CubicSpline(omegas_1, values_1, axis=0, extrapolate=False)
    interp_2 = CubicSpline(omegas_2, values_2, axis=0, extrapolate=False)
    discr_a, err_a = quad(lambda w: np.abs(interp_1(w)[0] - interp_2(w)[0])**2, -wmax, wmax)
    discr_b, err_b = quad(lambda w: np.abs(interp_1(w)[1] - interp_2(w)[1])**2, -wmax, wmax)
    discr = np.sqrt(discr_a + discr_b)
    err = np.sqrt(err_a**2 + err_b**2) / (2 * discr)
    # print(f"Discrepancy = {discr}, err = {err}")
    return discr, err


def solve_auto_inner(eps, beta, omega_max, delta_omega_init=0.01, tol=1e-3):
    delta_omega = delta_omega_init
    nr_failures = 0

    omegas_ref, R_grea_w_ref, runtime = solve(eps, beta, omega_max, delta_omega)

    for i in range(100):
        omegas = omegas_ref
        R_grea_w = R_grea_w_ref
        delta_omega /= 2.0
        omegas_ref, R_grea_w_ref, runtime = solve(eps, beta, omega_max, delta_omega)

        discr, err = discrepancy(omegas, R_grea_w, omegas_ref, R_grea_w_ref)
        if discr - err <= tol <= discr + err:
            nr_failures += 1
            if nr_failures > 3:
                raise RuntimeError(f"[Inner loop] Integration error is too big: discr={discr}, err={err}")
            else:
                print(fr"[Inner loop] /!\ Integration error is too big: discr={discr}, err={err}")
        if discr <= tol:
            print(f"[Inner loop] Converged in {i} iterations. omega_max={omega_max}, delta_omega={delta_omega}")
            break

        if i == 99:
            raise RuntimeError("Inner loop did not converge")

    return omegas_ref, R_grea_w_ref, delta_omega, runtime


def solve_auto(eps, beta, omega_max_init=100., delta_omega_init=0.01, tol=1e-3):
    start = now()
    omega_max = omega_max_init
    nr_failures = 0

    omegas_ref, R_grea_w_ref, delta_omega, runtime_best = solve_auto_inner(eps, beta, omega_max, delta_omega_init, tol=tol)

    for i in range(100):
        omegas = omegas_ref
        R_grea_w = R_grea_w_ref
        omega_max *= 2.0
        delta_omega_init = 4 * delta_omega
        omegas_ref, R_grea_w_ref, delta_omega, runtime_best = solve_auto_inner(eps, beta, omega_max, delta_omega_init, tol=tol)

        discr, err = discrepancy(omegas, R_grea_w, omegas_ref, R_grea_w_ref)
        if discr - err <= tol <= discr + err:
            nr_failures += 1
            if nr_failures > 3:
                raise RuntimeError(f"[Outer loop] Integration error is too big: discr={discr}, err={err}")
            else:
                print(fr"[Outer loop] /!\ Integration error is too big: discr={discr}, err={err}")
        if discr <= tol:
            print(f"[Outer loop] Converged in {i} iterations. omega_max={omega_max}, delta_omega={delta_omega}")
            break

        if i == 99:
            raise RuntimeError("Outer loop did not converge")

    runtime = now() - start
    print(f"Full runtime = {runtime} s")

    Result = namedtuple('Result', 'omegas, R_grea_w, omegas_prev, R_grea_w_prev, omega_max, delta_omega, runtime_total, runtime_best')
    return Result(omegas_ref, R_grea_w_ref, omegas, R_grea_w, omega_max, delta_omega, runtime, runtime_best)


if __name__ == '__main__':

    res = solve_auto(eps=-4., beta=200., omega_max_init=100., tol=1e-4)


    for k in range(2):
        plt.plot(res.omegas_prev, res.R_grea_w_prev[:, k] / 2., '--', label=f"R^R_{k}")
        plt.plot(res.omegas, res.R_grea_w[:, k] / 2., label=f"R^R_{k}")

    plt.xlim(-15, 5)
    plt.legend()
    plt.title(r"$R^R(\omega)$")
    plt.xlabel(r"$\omega$")
    plt.show()

