"""
Example of a simple Anderson impurity model coupled to a thermal bath.
"""
import nca
import numpy as np
import matplotlib.pyplot as plt

mesh = nca.Mesh(100.0, 200001)

### local (diagonal) Hamiltonian
eps = -2.0
U = 3.0
H_loc = np.array([0.0, eps, eps, 2 * eps + U])  # basis: 0, up, dn, updn

### Hybridization to a semicircular bath
Gamma = 1.0  # Hybridization strength
beta = 3.0  # inverse temperature
Ef = 0.0  # Fermi level
D = 6.0  # half bandwidth
delta_less, delta_grea = nca.make_Delta_semicirc(Gamma, D, beta, Ef, mesh)

### solver
S = nca.SolverSteadyState(2, H_loc, mesh)

S.add_bath(0, delta_grea, delta_less)  # orbital 0 -> up
S.add_bath(1, delta_grea, delta_less)  # orbital 1 -> down
# /!\ carefull with order of delta_grea, delta_less

basis = S.state_space.basis
print("local basis:", basis)
print("list of orbitals:", S.state_space.orbital_names)

# rename for nicer legends
basis = ["empty", "up", "down", "full"]
S.state_space.orbital_names = ["up", "down"]

### calculation
S.greater_loop(max_iter=20, verbose=True)
S.lesser_loop(max_iter=20, verbose=True)

### plot results
R_grea_w = S.get_R_grea_w()

for k in range(4):
    plt.plot(S.freq_mesh, R_grea_w[:, k], label=basis[k])

plt.xlim(-10, 10)
plt.legend()
plt.title("R^>(w)")
plt.xlabel("w")
plt.show()

m, dos = S.get_DOS(0)  # DOS takes an orbital, 0 -> up, 1 -> dn

plt.plot(m, dos)
plt.xlim(-10, 10)
plt.title("Density of states")
plt.xlabel("w")
plt.show()
