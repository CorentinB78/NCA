import nca
import numpy as np
import time


mesh = nca.Mesh(100.0, 200001)
# times = time_mesh.values()

### local model
Gamma = 1.0
eps = -1.0
U = 3.0

### basis: 0, dn, up, updn
H_loc = np.array([0.0, eps, eps, 2 * eps + U])

beta = 3.0
Ef = 0.3
D = 6.0
E0 = 0.0

### Hybridization
delta_less, delta_grea = nca.make_Delta_semicirc(Gamma, D, E0, beta, Ef, mesh)

fock = nca.FermionicFockSpace(["up", "dn"])
fock.add_bath(0, delta_grea, delta_less)
fock.add_bath(1, delta_grea, delta_less)
hybs = fock.generate_hybridizations()

print(fock.basis())

S = nca.SolverSteadyState(H_loc, mesh, hybs, [0, 3])

start = time.time()
S.greater_loop(plot=False, verbose=True)
S.lesser_loop(plot=False, verbose=True, max_iter=20)
runtime = time.time() - start

print(f"Run time: {runtime}")