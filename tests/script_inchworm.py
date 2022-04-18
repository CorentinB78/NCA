import inchworm
import numpy as np
import triqs
from triqs.gf import *
from triqs.operators import c, c_dag, n
import time
from h5 import HDFArchive
from triqs.utility import mpi

# ==== System Parameters ====
beta = 5.           # Inverse temperature
mu = 2.             # Chemical potential
U = 5.              # On-site density-density interaction
# U = 4.              # On-site density-density interaction
h = 0.2             # Local magnetic field
# h = 0.             # Local magnetic field
Gamma = 1.          # Hybridization energy
D = 5.

block_names = ['up', 'dn']
n_orb = 1

# ==== Local Hamiltonian ====
h_0 = - mu*( n('up',0) + n('dn',0) ) - h*( n('up',0) - n('dn',0) )
h_int = U * n('up',0) * n('dn',0)
h_imp = h_0 + h_int

# ==== Green function structure ====
gf_struct = [ (s, n_orb) for s in block_names ]

# ==== Hybridization Function ====
n_iw = int(10 * beta)
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
Delta = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
Delta << 0.;

Delta['up'] << Gamma * D * 0.5 * SemiCircular(D)
Delta['dn'] << Gamma * D * 0.5 * SemiCircular(D)
# for iw in Delta['up'].mesh:
#     Delta['up'][iw] = -1j * Gamma * np.sign(iw.imag)
#     Delta['dn'][iw] = -1j * Gamma * np.sign(iw.imag)

# # ==== Non-Interacting Impurity Green function  ====
# G0_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
# G0_iw['up'] << inverse(iOmega_n + mu + h - Delta['up'])
# G0_iw['dn'] << inverse(iOmega_n + mu - h - Delta['dn'])




for job in range(10):

    # --------- Construct the CTHYB solver ----------
    constr_params = {
            'beta' : beta,
            'gf_struct' : gf_struct,
            'n_iw' : n_iw,
            'n_tau_green' : 10,
            'n_tau_inch' : 1000,
            'n_tau' : 10001
            }
    S = inchworm.Solver(**constr_params)

    # --------- Initialize Delta_tau ----------
    # S.Delta_tau << Fourier(Delta) # Not possible as Delta_tau is purely real
    for bl in ['up', 'dn']:
        S.Delta_tau[bl] << make_gf_from_fourier(Delta[bl], constr_params['n_tau']).real

    # --------- Solve! ----------
    assert(h_imp - h_imp.real == 0.0 * n('up', 0))

    solve_params = {
            'h_imp' : h_imp.real,
            'n_cycles' : 3000,
            'measure_order_histogram' : True,
            #'length_cycle' : 100,
            #'n_callibration_cycles' : 100,
            #'verbosity' : 3 if mpi.is_master_node() else 0
            "max_order": 1,
            "random_seed": 6354 + 123 * job + 37837 * mpi.rank
            }
    start = time.time()
    # S.solve(**solve_params)
    S.solve_inchworm(solve_params)
    # S.solve_green(**solve_params)
    end = time.time()

    print(end - start)

    # -------- Save in archive ---------
    if mpi.is_master_node():
        with HDFArchive(f"data/apr_2022/res_smallsteps_inchworm_{job}.h5",'w') as results:
            # results["G"] = S.G_tau

            # import inspect
            import __main__
            results.create_group("Solver_Info")
            info_grp = results["Solver_Info"]
            info_grp["solver_name"] = "inchworm"
            # info_grp["constr_params"] = constr_params
            # info_grp["solve_params"] = solve_params
            info_grp["solver"] = S
            #info_grp["solver_version"] = version.version
            #info_grp["solver_git_hash"] = version.inchworm_hash
            #info_grp["triqs_git_hash"] = version.triqs_hash
            # info_grp["script"] = inspect.getsource(__main__)
            info_grp["num_threads"] = mpi.world.Get_size()
            info_grp["run_time"] = end - start
