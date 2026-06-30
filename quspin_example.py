import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import fermions_basis_1d

# 1. Define custom parameters
N_sites = 4         # A 2x2 square cluster (4 sites total)
t_hopping = 1.0     # Hopping amplitude
U_repulsion = 4.0   # On-site interaction strength

# Define our 2D grid bonds manually for a 2x2 square lattice:
# Sites are numbered 0, 1, 2, 3
# Horizontal bonds: (0-1), (2-3) | Vertical bonds: (0-2), (1-3)
hop_bonds = [[-t_hopping, 0, 1], [-t_hopping, 2, 3], [-t_hopping, 0, 2], [-t_hopping, 1, 3]]
int_bonds = [[U_repulsion, i, i] for i in range(N_sites)]

# 2. Set up the exact basis (Enforcing conservation shrinks the matrix size dramatically!)
# We will look at "half-filling": 2 up electrons and 2 down electrons
basis = fermions_basis_1d(N=N_sites, Nf=(2, 2)) 

# 3. Build the operators for the Fermi-Hubbard Hamiltonian equation
# Kinetic hopping terms for spin-up and spin-down
static_list = [
    ["+-", hop_bonds],   # spin-up hopping
    ["-+", [[-t[0], t[2], t[1]] for t in hop_bonds]], # hermitian conjugate
    ["|+-", hop_bonds],  # spin-down hopping
    ["|-+", [[-t[0], t[2], t[1]] for t in hop_bonds]],
    ["n|n", int_bonds]   # On-site U interaction: n_up * n_down
]

# Build the final exact Hamiltonian matrix
H = hamiltonian(static_list, [], basis=basis, dtype=np.float64)

# 4. Run the Exact Diagonalization Solver
# eigsh finds the absolute lowest eigenvalue (Ground State) perfectly
E_exact, psi_exact = H.eigsh(k=1, which="SA")

print(f"=== QuSpin Exact Diagonalization ===")
print(f"Lattice size: {N_sites} sites (Half-filled)")
print(f"Exact Matrix Dimension: {basis.Ns} x {basis.Ns}")
print(f"Exact Ground State Energy: {E_exact[0]:.6f}")