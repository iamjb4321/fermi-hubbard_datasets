import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.models.spins import SpinModel
from tenpy.algorithms.exact_diag import ExactDiag

def run_kagome_ed(lx, U=4.0, t=1.0):
    # 1. Define model parameters
    # The Kagome unit cell has 3 sites, so Lx=2, Ly=1 gives 2 * 1 * 3 = 6 sites total.
    model_params = {
        'lattice': 'Kagome',
        'Lx': lx,
        'Ly': 1,
        'S': 0.5,           # Spin-1/2
        'Jx': 1.0,          # Heisenberg isotropic couplings (J_x = J_y = J_z)
        'Jy': 1.0,
        'Jz': 1.0,
        'bc_MPS': 'finite', # Required for ExactDiag
        'bc_x': "periodic",
        'bc_y': 'periodic',

        #completely ignored by SpinModel; MUST USE FermiHubbardModel import 
        "t": t,                  
        "U": U,           
        "mu": 0.0,                 
        "V": 0.0, 

        'conserve': 'Sz'    # Conserve total Sz to reduce the matrix size blocks
    }
    
    print("Initializing SpinModel on a Kagome lattice...")
    model = SpinModel(model_params)
    
    # 2. Pick a target charge sector (optional, but highly recommended)
    # Let's target the half-filling sector (total Sz = 0)
    # Total sites = 6, so 3 up and 3 down spins

    # 2. Pick a target charge sector
    num_sites = model.lat.N_sites
    print(f"Total number of sites in the cluster: {num_sites}")
    
    # Generate an alternating state of exact length num_sites
    init_state = ['up' if i % 2 == 0 else 'down' for i in range(num_sites)]
    
    # Construct an MPS briefly to extract the charge sector representation
    from tenpy.networks.mps import MPS
    psi_mock = MPS.from_product_state(model.lat.mps_sites(), init_state)
    target_sector = psi_mock.get_total_charge(True)

    # 3. Initialize Exact Diagonalization
    print("Building exact diagonalization object...")
    # Increase max_size to prevent the safe-abort mechanism
    ED = ExactDiag(model, charge_sector=target_sector, max_size=2e9)

    # print("Building full Hamiltonian from MPO inside specified charge sector...")
    # ED = ExactDiag(model, charge_sector=target_sector)
    print("Constructing the matrix from the MPO structure...")
    ED.build_full_H_from_mpo()
    
    # 4. Perform the full diagonalization
    print("Running sparse exact diagonalization (Lanczos)...")
    # k=1 finds the single lowest ground state. Increase k if you want excited states.
    E_ground_array, psi_ground_list = ED.sparse_diag(k=1, which="SR")

    # print("Running exact diagonalization...")
    # ED.full_diagonalization()
    
    # 5. Extract ground state and eigenvalues
    E_ground = E_ground_array[0]
    psi_ground = psi_ground_list[0]
    
    print("\n--- ED Results ---")
    print(f"Ground state energy: {E_ground:.8f}")
    print(f"Energy per site: {E_ground / num_sites:.8f}")


if __name__ == "__main__":
    # params: lx, U, t
    for i in [0.0, 2.0, 6.0, 10.0]:
        for j in [-0.3, 0.3, 1.0]:
            t = j

            run_kagome_ed(5,i, t)
            run_kagome_ed(3, i, t)
    