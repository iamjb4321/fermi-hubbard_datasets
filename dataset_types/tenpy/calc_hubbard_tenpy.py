import tenpy
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import numpy as np

# 1. Choose how many different U points to calculate
num_simulations = 5  
u_values = np.linspace(0.0, 8.0, num_simulations)

# Lists to store our final calculated data
calculated_energies = []
results_info = []

print("=== Starting DMRG Ground State Calculations ===")

for i in range(num_simulations):
    current_u = u_values[i]
    
    # Configure the custom Hamiltonian parameters
    model_params = {
        "lattice": "Square",       
        "Lx": 3,                   # 3x2 grid keeps calculation times fast on a laptop
        "Ly": 2,                   
        "bc_x": "open",            
        "bc_y": "periodic",        
        
        "t": 1.0,                  
        "U": current_u,           
        "mu": 0.0,                 
        "V": 0.0,                  
        
        "cons_N": "N",            
        "cons_Sz": "Sz",          
    }
    
    # Initialize the specific mathematical model
    model = FermiHubbardModel(model_params)
    
    # 2. Build the initial starting state guess (filling the lattice half-full)
    # This pattern alternates up and down spins on the grid sites
    init_state = [["up"], ["down"]] * (model.lat.N_sites // 2)
    psi = MPS.from_lat_product_state(model.lat, init_state)
    
    # 3. Configure the DMRG algorithm solver options
    dmrg_params = {
        'mixer': True,             # Prevents the solver from getting stuck in local minimums
        'max_sweeps': 15,          # Maximum optimization iterations per U value
        'verbose': 0               # Silences the massive debug output to keep terminal clean
    }
    
    print(f"Running simulation {i+1}/{num_simulations} for U = {current_u:.2f}...")
    
    # 4. RUN THE SOLVER: Calculates the minimal energy E for the Hamiltonian equation
    info = dmrg.run(psi, model, dmrg_params)
    
    # Extract the resulting Ground State Energy
    ground_state_energy = info['E']
    
    # Save our calculated data
    calculated_energies.append(ground_state_energy)
    results_info.append((current_u, ground_state_energy))

print("\n=== Calculations Complete! ===")
print("Here is your custom dataset:")
print("-" * 40)
print(f"{'U Value':<15} | {'Ground State Energy':<20}")
print("-" * 40)
for u_val, energy in results_info:
    print(f"{u_val:<15.2f} | {energy:<20.6f}")