"""
Second order perturbation theory for the Anderson impurity model at U = +inf.
"""
import numpy as np
from scipy import linalg, integrate
import toolbox as tb


def initial_density_matrix_for_steady_state(
    H_loc, freq_mesh, delta_grea_w, delta_less_w
):
    """
    Computes initial states that lead for a steady state in second order perturbation theory.

    Only Anderson impurity model.
    """

    en_diff = H_loc[:, None] - H_loc[None, :]

    delta_grea_en = np.interp(
        en_diff, freq_mesh.values(), delta_grea_w.imag, left=0.0, right=0.0
    )
    delta_less_en = np.interp(
        en_diff, freq_mesh.values(), delta_less_w.imag, left=0.0, right=0.0
    )

    M = np.zeros((4, 4), dtype=float)
    i_0, i_up, i_dn, i_2 = 0, 1, 2, 3
    M[0, 1] = delta_grea_en[i_up, i_0]
    M[0, 2] = delta_grea_en[i_dn, i_0]
    M[1, 0] = -delta_less_en[i_up, i_0]
    M[2, 0] = -delta_less_en[i_dn, i_0]

    if len(H_loc) > 3:
        M[1, 3] = delta_grea_en[i_2, i_up]
        M[2, 3] = delta_grea_en[i_2, i_dn]
        M[3, 1] = -delta_less_en[i_2, i_up]
        M[3, 2] = -delta_less_en[i_2, i_dn]

    for i in range(len(H_loc)):
        M[i, i] = -np.sum(M[:, i])

    out = linalg.null_space(M)
    for i in range(out.shape[1]):
        out[:, i] /= np.sum(out[:, i])

    return out


def make_R2_grea(H_loc, time_mesh, delta_grea, delta_less):
    tt = time_mesh.values()
    idx0 = len(tt) // 2
    tt = tt[idx0:]

    integrand_empty = delta_less[::-1][idx0:] * (
        np.exp(1j * tt * (H_loc[1] - H_loc[0]))
        + np.exp(1j * tt * (H_loc[2] - H_loc[0]))
    )
    integrand_up = delta_grea[idx0:] * np.exp(-1j * tt * (H_loc[0] - H_loc[1]))
    integrand_dn = delta_grea[idx0:] * np.exp(-1j * tt * (H_loc[0] - H_loc[2]))

    R2_empty = tt * integrate.cumtrapz(integrand_empty, dx=time_mesh.delta, initial=0.0)
    R2_empty -= integrate.cumtrapz(
        tt * integrand_empty, dx=time_mesh.delta, initial=0.0
    )
    R2_empty *= np.exp(-1j * H_loc[0] * tt)

    R2_up = tt * integrate.cumtrapz(integrand_up, dx=time_mesh.delta, initial=0.0)
    R2_up -= integrate.cumtrapz(tt * integrand_up, dx=time_mesh.delta, initial=0.0)
    R2_up *= -np.exp(-1j * H_loc[1] * tt)

    R2_dn = tt * integrate.cumtrapz(integrand_dn, dx=time_mesh.delta, initial=0.0)
    R2_dn -= integrate.cumtrapz(tt * integrand_dn, dx=time_mesh.delta, initial=0.0)
    R2_dn *= -np.exp(-1j * H_loc[2] * tt)

    out = np.stack([R2_empty, R2_up, R2_dn])
    _, out = tb.symmetrize(tt, out, 0.0, lambda x: -np.conj(x))

    return out


def make_R2_less(H_loc, time_mesh, delta_grea, delta_less, init_state):

    tt = time_mesh.values()
    idx0 = len(tt) // 2
    t = tt[idx0:]

    less_empty = np.exp(1j * (H_loc[1] - H_loc[0]) * tt) + np.exp(
        1j * (H_loc[2] - H_loc[0]) * tt
    )
    less_empty *= init_state[0] * delta_less

    grea_occup = init_state[1] * delta_grea * np.exp(1j * (H_loc[1] - H_loc[0]) * tt)
    grea_occup += init_state[2] * delta_grea * np.exp(1j * (H_loc[2] - H_loc[0]) * tt)

    diverging_term = integrate.trapz(less_empty + grea_occup, dx=time_mesh.delta)
    print("diverging term:", diverging_term, "should be zero.")

    linear_term = integrate.trapz(
        less_empty[idx0:] + grea_occup[idx0:], dx=time_mesh.delta
    )
    linear_term *= t

    cst_term = integrate.trapz(
        (less_empty + grea_occup) * np.abs(tt), dx=time_mesh.delta
    )

    last_term = t * integrate.cumtrapz(
        grea_occup[::-1][idx0:], dx=time_mesh.delta, initial=0.0
    )
    last_term += -integrate.cumtrapz(
        t * grea_occup[::-1][idx0:], dx=time_mesh.delta, initial=0.0
    )

    R2_less = linear_term + last_term
    R2_less *= -np.exp(-1j * H_loc[0] * t)

    _, out = tb.symmetrize(t, R2_less, 0.0, lambda x: -np.conj(x))

    return out


def GF2_grea():
    import nca

    freq_mesh, delta_grea_w, delta_less_w = nca.make_Delta_semicirc_w(
        1.0, 2.0, 3.0, 0.0
    )
    time_mesh, delta_less, delta_grea = nca.make_Delta_semicirc(1.0, 2.0, 3.0, 0.0)

    H_loc = np.array([0.0, -0.6, -0.6])
    init_state = initial_density_matrix_for_steady_state(
        H_loc, freq_mesh, delta_grea_w, delta_less_w
    )
    init_state = init_state[:, 0]
    init_state = init_state[:3]
    print(init_state)

    R2_less = make_R2_less(H_loc, time_mesh, delta_grea, delta_less, init_state)
    R2_grea = make_R2_grea(H_loc, time_mesh, delta_grea, delta_less)

    gf1 = init_state[0] * np.exp(1j * (H_loc[0] * time_mesh.values())) * R2_grea[1]
    gf2 = R2_less[::-1] * np.exp(-1j * (H_loc[1] * time_mesh.values()))

    return time_mesh, gf1, gf2
