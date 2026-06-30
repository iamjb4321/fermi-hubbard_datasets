import tenpy
from tenpy.models.hubbard import FermiHubbardModel
import numpy as np

hamiltonians = []
u_values = np.linspace(0.0, 10.0, 5) # Let's test with 5 first to verify

for i in range(len(u_values)):
    model_params = {
        "lattice": "Square",       
        "Lx": 4,                   
        "Ly": 2,                   
        "bc_x": "open",            
        "bc_y": "periodic",        
        
        # Custom physics variables
        "t": 1.0,                  
        "U": u_values[i],           
        "mu": 0.0,                 
        "V": 0.5,                  
        
        # FIX 1: TeNPy FermiHubbardModel uses these keys for particle/spin conservation
        "cons_N": "N",            # Conserve total particle number
        "cons_Sz": "Sz",          # Conserve total spin Z component
    }

    custom_model = FermiHubbardModel(model_params)
    hamiltonians.append(custom_model)

for model in hamiltonians:
    # FIX 2: Dynamic type check for the lattice name and safer options lookup
    lat_name = type(model.lat).__name__
    u_val = model.options.get('U', 0.0)
    
    print(f"Lattice: {lat_name} | Sites: {model.lat.N_sites} | U value: {u_val:.2f}")