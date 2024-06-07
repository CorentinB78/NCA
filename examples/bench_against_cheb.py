"""
Example of a simple Anderson impurity model coupled to a thermal bath.
"""
import nca
import numpy as np
import matplotlib.pyplot as plt

x = 3
time_mesh = nca.Mesh(x*200.0, x*x*int(4e5)).adjoint()
print(time_mesh)

### local (diagonal) Hamiltonian
eps0 = 0.0
eps1 = -2.5
H_loc = np.array([eps0, eps1, eps1])  # basis: 0, up, dn, (updn -> forbidden)

### Hybridization to a semicircular bath
Gamma = 1.0  # Hybridization strength
beta = 100.0  # inverse temperature
Ef = 0.0  # Fermi level
D = 20.0  # half bandwidth

dos = nca.make_gaussian_dos(D)
# dos = nca.make_semicircular_dos(D)
hyb_grea, hyb_less = nca.make_hyb_times(dos, beta, Ef, Gamma, time_mesh)

### solver
S = nca.AIM_infinite_U(H_loc, time_mesh, order=6)
S.state_space.orbital_names = ["up", "down"]

S.add_bath(0, hyb_grea, hyb_less)  # orbital 0 -> up
S.add_bath(1, hyb_grea, hyb_less)  # orbital 1 -> down


### calculation
S.greater_loop(max_iter=20, verbose=True)

### plot results
R_grea_w = S.get_R_grea_w()

for k in range(2):
    plt.plot(S.freq_mesh, R_grea_w[:, k] / 2., label=f"R^R_{k}")

plt.xlim(-20, 10)
plt.legend()
plt.title(r"$R^R(\omega)$")
plt.xlabel(r"$\omega$")
plt.show()

S.lesser_loop(max_iter=20, verbose=True)
R_less_w = S.get_R_less_w()

for k in range(2):
    plt.plot(S.freq_mesh, R_less_w[:, k], label=f"R^<_{k}")

plt.xlim(-20, 10)
plt.legend()
plt.title(r"$R^<(\omega)$")
plt.xlabel(r"$\omega$")
plt.show()

m, dos = S.get_DOS(0)  # DOS takes an orbital, 0 -> up, 1 -> dn

plt.plot(m, np.pi * Gamma * dos)
plt.xlim(-20, 20)
plt.title("Density of states")
plt.xlabel(r"$\omega$")
plt.show()
