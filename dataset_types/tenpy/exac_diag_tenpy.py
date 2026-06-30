import tenpy
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.algorithms.exact_diag import ExactDiag

# 1. Flexible parameters
model_params = {
    "lattice": "Square",       
    "Lx": 2,                   # 2x2 = 4 sites
    "Ly": 2,                   
    "bc_x": "open",            
    "bc_y": "open",        
    
    "t": 1.0,                  
    "U": 4.0,           
    "mu": 0.0,                 
    "V": 0.0,                  
    
    "cons_N": "N",            
    "cons_Sz": "Sz",          
}

# 2. Generate the model
model = FermiHubbardModel(model_params)

print("=== Running Built-In Exact Diagonalization ===")

# 3. Setup the Exact Diagonalization Solver
try:
    sector = model.default_charge()
    ed_solver = ExactDiag(model, charge_sector=sector)
except AttributeError:
    ed_solver = ExactDiag(model)

# 4. FIX: Run full_diagonalization FIRST to populate the internal states
ed_solver.full_diagonalization()

# 5. Now you can safely call groundstate() to extract the values!
exact_energy, psi_ground = ed_solver.groundstate()

print(f"Lattice Geometry: {type(model.lat).__name__} ({model.lat.N_sites} sites)")
print(f"Mathematically Exact Ground State Energy = {exact_energy:.6f}")

#NOT WORKIGN YET 