"""
Example of a simple Anderson impurity model coupled to a thermal bath.
"""
import nca
import numpy as np
import matplotlib.pyplot as plt

time_mesh = nca.Mesh(200.0, int(4e5)).adjoint()
print(time_mesh)

### local (diagonal) Hamiltonian
U = 5.0
eps = - U / 2.
H_loc = np.array([0.0, eps, eps, 2 * eps + U])  # basis: 0, up, dn, updn

### Hybridization to a semicircular bath
Gamma = 1.0  # Hybridization strength
beta = 100.0  # inverse temperature
Ef = 0.0  # Fermi level
D = 20.0  # half bandwidth
dos = nca.make_gaussian_dos(D)
hyb_grea, hyb_less = nca.make_hyb_times(dos, beta, Ef, Gamma, time_mesh)

### solver
S = nca.SolverSteadyState(2, H_loc, time_mesh, order=6)

S.add_bath(0, hyb_grea, hyb_less)  # orbital 0 -> up
S.add_bath(1, hyb_grea, hyb_less)  # orbital 1 -> down

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

plt.xlim(-20, 10)
plt.legend()
plt.title(r"$R^>(\omega)$")
plt.xlabel(r"$\omega$")
plt.show()

R_less_w = S.get_R_less_w()

for k in range(4):
    plt.plot(S.freq_mesh, R_less_w[:, k], label=basis[k])

plt.xlim(-20, 10)
plt.legend()
plt.title(r"$R^<(\omega)$")
plt.xlabel(r"$\omega$")
plt.show()

m, dos = S.get_DOS(0)  # DOS takes an orbital, 0 -> up, 1 -> dn

plt.plot(m, dos)
plt.xlim(-20, 20)
plt.title("Density of states")
plt.xlabel(r"$\omega$")
plt.show()
