import openfermion as of

# 1. Define your custom variables flexibly
x_dim = 2                  # Width of 2D grid
y_dim = 3                  # Height of 2D grid (Total 6 sites, 12 spin-orbitals)
t_hopping = 1.2            # Kinetic hopping energy
u_coulomb = 4.0            # On-site repulsion
mu_chemical = 0.5          # Chemical potential (controls filling fraction)
periodic_bounds = False    # Set to True for a closed loop/torus shape

# 2. Build the customized 2D Fermi-Hubbard Hamiltonian in one line
hubbard_hamiltonian = of.fermi_hubbard(
    x_dimension=x_dim,
    y_dimension=y_dim,
    tunneling=t_hopping,
    coulomb=u_coulomb,
    chemical_potential=mu_chemical,
    periodic=periodic_bounds,
    spinless=False
)

# 3. Quick sanity check: Print the total number of terms generated
print(f"Lattice Size: {x_dim}x{y_dim}")
print(f"Total interaction terms generated: {len(hubbard_hamiltonian.terms)}")