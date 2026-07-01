import numpy as np
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

def run_kagome_dmrg(lx, U=4.0, t=1.0):
    # 1. Define model parameters for FermiHubbardModel
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
        
        # Enable conservation laws natively for DMRG speedups
        'cons_N': 'N',     
        'cons_Sz': 'Sz'    
    }
    
    # Initialize Model
    model = FermiHubbardModel(model_params)
    num_sites = model.lat.N_sites
    
    # 2. Initialize a valid alternating product state for half-filling
    init_state = ['up' if i % 2 == 0 else 'down' for i in range(num_sites)]
    
    psi = MPS.from_product_state(
        model.lat.mps_sites(), 
        init_state, 
        unit_cell_width=model.lat.N_cells
    )
    
    # 3. Configure the DMRG Engine options
    dmrg_params = {
        'mixer': True,             # Crucial for frustrated lattices to break local minima
        'max_sweeps': 30,          # Small clusters converge rapidly
        'max_bond_dimension': 200, # More than enough to yield the exact ground state for 15 sites
        'verbose': 0               # Keeps your result.txt file clean of sweep iterations
    }
    
    # 4. Run the local optimization sweep
    info = dmrg.run(psi, model, dmrg_params)
    E_ground = info['E']
    
    # 5. Output cleanly formatted results
    print(f"Lx={lx:<2} | U={U:<4.1f} | t={t:<4.1f} | E_G={E_ground:<12.8f} | E/site={E_ground/num_sites:<10.8f}")

if __name__ == "__main__":
    print("Starting Hubbard-Kagome Parameter Sweep via DMRG...")
    print("-" * 65)
    
    # Scan parameter space looping over your chosen grid
    for i in [0.0, 2.0, 6.0, 10.0]:      # Mapped to interaction U
        for j in [-0.3, 0.3, 1.0]:       # Mapped to hopping t
            t_val = j
            
            # Efficiently processes both configurations without memory inflation
            run_kagome_dmrg(5, U=i, t=t_val) 
            run_kagome_dmrg(3, U=i, t=t_val)
