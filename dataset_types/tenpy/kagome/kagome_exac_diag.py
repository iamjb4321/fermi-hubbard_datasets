import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.networks.mps import MPS

def run_kagome_ed(lx, U=4.0, t=1.0):
    # 1. Define model parameters correctly for FermiHubbardModel
    model_params = {
        'lattice': 'Kagome',
        'Lx': lx,
        'Ly': 1,
        'bc_MPS': 'finite', 
        'bc_x': "periodic",
        'bc_y': 'periodic',
        
        # Fermionic parameters
        't': t,          # Nearest-neighbor hopping
        'U': U,          # On-site interaction
        'mu': 0.0,       # Chemical potential
        'V': 0.0,        # Nearest-neighbor interaction
        
        # Correct TeNPy syntax for Hubbard conservation laws
        'cons_N': 'N',     # Conserve total electron number
        'cons_Sz': 'Sz'    # Conserve total spin projection
    }
    
    print(f"\nInitializing FermiHubbardModel (Lx={lx}, U={U:.1f}, t={t:.1f})...")
    model = FermiHubbardModel(model_params)
    
    num_sites = model.lat.N_sites
    print(f"Total number of sites in the cluster: {num_sites}")
    
    # 2. Pick a valid charge sector using correct SpinHalfFermionSite labels
    # Valid options are: 'empty', 'up', 'down', 'full'
    init_state = ['up' if i % 2 == 0 else 'down' for i in range(num_sites)]
    
    # Construct mock MPS using exact unit cell width to suppress warnings
    psi_mock = MPS.from_product_state(
        model.lat.mps_sites(), 
        init_state, 
        unit_cell_width=model.lat.N_cells
    )
    target_sector = psi_mock.get_total_charge(True)
    
    # 3. Initialize Exact Diagonalization with large max_size allowance
    print("Building exact diagonalization object...")
    ED = ExactDiag(model, charge_sector=target_sector, max_size=2e9)
    
    print("Constructing the matrix from the MPO structure...")
    ED.build_full_H_from_mpo()
    
    # 4. Perform the sparse diagonalization
    print("Running sparse exact diagonalization (Lanczos)...")
    E_ground_array, psi_ground_list = ED.sparse_diag(k=1, which="SR")
    
    # 5. Extract ground state and eigenvalues
    E_ground = E_ground_array[0]
    
    print("--- ED Results ---")
    print(f"Ground state energy: {E_ground:.8f}")
    print(f"Energy per site: {E_ground / num_sites:.8f}")

if __name__ == "__main__":
    # Scan parameter space looping over your chosen grid
    for i in [0.0, 2.0, 6.0, 10.0]:      # Mapped to interaction U
        for j in [-0.3, 0.3, 1.0]:       # Mapped to hopping t
            t_val = j
            
            # Execute calculations sequentially
            run_kagome_ed(5, U=i, t=t_val) 
            run_kagome_ed(3, U=i, t=t_val)
