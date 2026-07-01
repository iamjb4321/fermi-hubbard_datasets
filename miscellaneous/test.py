import pennylane as qml

print("Connecting to PennyLane servers to load dataset...")

# Load the specific 1x4 FermiHubbard open chain dataset
datasets = qml.data.load(
    "qspin", 
    sysname="FermiHubbard", 
    periodicity="open", 
    lattice="rectangular", 
    layout="2x2"
)

print("Dataset loaded successfully! Extracting data...")


# Extract the target dataset from the returned list
fermi_data = datasets[0]

# For seeing the whole dataset
#print("Available attributes:", fermi_data.attrs.keys())


# 1. Grab the list of Hamiltonians and select the first one [0]
H = fermi_data.hamiltonians[0]  # <--- Fixed plural name and added index
wires = H.wires

# 2. Define a quantum device using the dataset's wire configuration
dev = qml.device("default.qubit", wires=wires)

# 3. Create a QNode to measure the expectation value of the Fermi-Hubbard Hamiltonian
@qml.qnode(dev)
def circuit(params):
    # Prepare a simple ansatz (e.g., a parameterized state)
    qml.DoubleExcitation(params, wires=list(wires)[:4])
    return qml.expval(H)

# Initialize dummy parameters and execute
params = 0.15
energy_expectation = circuit(params)

print(f"Calculated Expectation Energy: {energy_expectation}")
print(f"Dataset Ground Truth Energy: {fermi_data.ground_energies[0]}")