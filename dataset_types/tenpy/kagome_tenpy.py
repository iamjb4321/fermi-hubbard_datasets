import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.models.spins import SpinModel
from tenpy.algorithms.exact_diag import ExactDiag

def run_kagome_ed():
    # 1. Define model parameters
    # The Kagome unit cell has 3 sites, so Lx=2, Ly=1 gives 2 * 1 * 3 = 6 sites total.
    model_params = {
        'lattice': 'Kagome',
        'Lx': 2,
        'Ly': 1,
        'S': 0.5,           # Spin-1/2
        'Jx': 1.0,          # Heisenberg isotropic couplings (J_x = J_y = J_z)
        'Jy': 1.0,
        'Jz': 1.0,
        'bc_MPS': 'finite', # Required for ExactDiag
        'bc_y': 'open',
        'conserve': 'Sz'    # Conserve total Sz to reduce the matrix size blocks
    }
    
    print("Initializing SpinModel on a Kagome lattice...")
    model = SpinModel(model_params)
    
    # 2. Pick a target charge sector (optional, but highly recommended)
    # Let's target the half-filling sector (total Sz = 0)
    # Total sites = 6, so 3 up and 3 down spins
    num_sites = model.lat.N_sites
    print(f"Total number of sites in the cluster: {num_sites}")
    
    # Let's create an initial state to extract the total charge representation
    # 'up' and 'down' repeated to fill total sites
    init_state = ['up', 'down'] * (num_sites // 2)
    
    # Construct an MPS briefly just to read out the exact charge sector representation safely
    from tenpy.networks.mps import MPS
    psi_mock = MPS.from_product_state(model.lat.mps_sites(), init_state)
    target_sector = psi_mock.get_total_charge(True)
    
    # 3. Initialize Exact Diagonalization
    print("Building full Hamiltonian from MPO inside specified charge sector...")
    ED = ExactDiag(model, charge_sector=target_sector)
    ED.build_full_H_from_mpo()
    
    # 4. Perform the full diagonalization
    print("Running exact diagonalization...")
    ED.full_diagonalization()
    
    # 5. Extract ground state and eigenvalues
    E_ground, psi_ground = ED.groundstate()
    all_eigenvalues = ED.E
    
    print("\n--- ED Results ---")
    print(f"Ground state energy: {E_ground:.8f}")
    print(f"Energy per site: {E_ground / num_sites:.8f}")
    print(f"Number of eigenvalues in this sector: {len(all_eigenvalues)}")
    print(f"Lowest 5 eigenvalues: {all_eigenvalues[:5]}")

if __name__ == "__main__":
    run_kagome_ed()