import pennylane as qml

# Load the specific 1x4 FermiHubbard open chain dataset
datasets = qml.data.load(
    "qspin", 
    sysname="FermiHubbard", 
    periodicity="open", 
    lattice="rectangular", 
    layout="2x2"
)

# Extract the target dataset from the returned list
fermi_data = datasets[0]

# View all available attributes (e.g., hamiltonian, ground_states, energies)
print(fermi_data)