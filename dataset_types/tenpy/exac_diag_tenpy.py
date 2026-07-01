import numpy as np
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.algorithms.exact_diag import ExactDiag

# 1. Setup parameter dictionary for a 2D 2x2 lattice
model_params = {

    "lattice": "Square",       
    "Lx": 2,                   # Keep it tiny (2x2 = 4 sites) for exact matrix math!
    "Ly": 2,                   
    "bc_x": "open",            
    "bc_y": "open",   
    "bc_MPS": "finite",      # REQUIRED for ExactDiag     
    
    "t": 1.0,                  
    "U": 4.0,           
    "mu": 0.0,                 
    "V": 0.0,                  
    
    "cons_N": "N",            
    "cons_Sz": "Sz",          
}

# 2. Instantiate the 2D Model
model_2d = FermiHubbardModel(model_params)

# Quick sanity check: Print total sites (should be 4) 
# and how the 1D snake path maps the 2D coordinates [x, y]
print(f"Total sites in the 1D MPS path: {model_2d.lat.N_sites}")
print("MPS Index -> Lattice Coordinates Mapping:")
for mps_idx in range(model_2d.lat.N_sites):
    lat_idx = model_2d.lat.mps2lat_idx(mps_idx)
    print(f"  MPS index {mps_idx} maps to grid coordinate: {lat_idx}")

# 3. Hand off the MPO structure to ExactDiag
ed_2d = ExactDiag(model_2d)
ed_2d.build_full_H_from_mpo()

# 4. Diagonalize
ed_2d.full_diagonalization() # can comment out/delete this (and the 2 
                                # prints below) if you don't want all 
                                # the eignevalues

print("\n--- 2x2 Spectrum Results ---")
print("All eigenvalues:\n", ed_2d.E)

E_g, psi_g = ed_2d.groundstate()
print(f"\n2x2 Ground state energy: {E_g:.6f}")


# ------------Getting other data-----------------

# Extracting the 1st excited state to measure things on 
E_0, psi_0 = ed_2d.groundstate()

print("\n================ MEASUREMENTS ================")

# -------------------------------------------------------------
# 0. Convert the ED Dense States into standard MPS Objects
# -------------------------------------------------------------
# Convert the ground state vector
E_0, psi_0_dense = ed_2d.groundstate()
psi_0_mps = ed_2d.full_to_mps(psi_0_dense)

# Convert the 1st excited state vector
E_1 = ed_2d.E[1]
psi_1_dense = ed_2d.V.take_slice(1, axes='ps*') 
psi_1_mps = ed_2d.full_to_mps(psi_1_dense)

# -------------------------------------------------------------
# 1. Local Electron Densities per Site
# -------------------------------------------------------------
# The density operator for the Fermi-Hubbard model is "N"
# 1. Convert your dense statevectors into standard MPS objects first
E_0, psi_0_dense = ed_2d.groundstate()
psi_0_mps = ed_2d.full_to_mps(psi_0_dense)

E_1 = ed_2d.E[1]
psi_1_dense = ed_2d.V.take_slice(1, axes='ps*') 
psi_1_mps = ed_2d.full_to_mps(psi_1_dense)

# 2. Call expectation_value on the MPS objects instead of ed_2d
density_ground = psi_0_mps.expectation_value("Ntot")
density_excited = psi_1_mps.expectation_value("Ntot")

print("\n[Local Electron Density <N_i>]")
for mps_idx in range(model_2d.lat.N_sites):
    lat_idx = model_2d.lat.mps2lat_idx(mps_idx)
    print(f"Site {mps_idx} {lat_idx[:2]}: "
          f"Ground={density_ground[mps_idx]:.4f} | "
          f"1st Excited={density_excited[mps_idx]:.4f}")


# -------------------------------------------------------------
# 2. Spin-Spin Correlations <Sz_i Sz_j>
# -------------------------------------------------------------
# This returns a 2D matrix where index [i, j] represents the correlation
# Call correlation_function on your converted MPS state vector
sz_sz_ground = psi_0_mps.correlation_function("Sz", "Sz")

print("\n[Ground State Spin-Spin Correlations <Sz_i Sz_j>]")
# Let's print out the full matrix layout nicely
print("     ", "  ".join([f"Site{i}" for i in range(model_2d.lat.N_sites)]))
for i in range(model_2d.lat.N_sites):
    row_str = " ".join([f"{sz_sz_ground[i, j]:.4f}" for j in range(model_2d.lat.N_sites)])
    print(f"Site{i}: {row_str}")

# Quick interpretation example:
print(f"\nCorrelation between Site 0 and Site 1: {sz_sz_ground[0, 1]:.4f}")