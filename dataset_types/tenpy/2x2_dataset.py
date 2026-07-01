import json
import random
import numpy as np
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.algorithms.exact_diag import ExactDiag
import os

def generate_llm_datapoint(t_val, u_val):
    """Runs TeNPy for specific parameters and formats it into an LLM prompt-response pair."""
    
    model_params = {
        "lattice": "Square",       
        "Lx": 2, "Ly": 2,          
        "bc_x": "open", "bc_y": "open",   
        "bc_MPS": "finite",        
        "t": t_val, 
        "U": u_val,           
        "mu": 0.0, "V": 0.0,                  
        "cons_N": "N", "cons_Sz": "Sz",          
    }

    # 1. Solve the physics
    model_2d = FermiHubbardModel(model_params)
    ed_2d = ExactDiag(model_2d)
    ed_2d.build_full_H_from_mpo()
    ed_2d.full_diagonalization()
    
    E_0, psi_0_dense = ed_2d.groundstate()
    psi_0_mps = ed_2d.full_to_mps(psi_0_dense)
    
    density_ground = psi_0_mps.expectation_value("Ntot")
    sz_sz_ground = psi_0_mps.correlation_function("Sz", "Sz")
    
    # 2. Design the prompt (what the user asks the LLM)
    prompt = (
        f"Analyze a 2x2 Fermi-Hubbard lattice model with open boundary conditions. "
        f"The kinetic hopping parameter t is {t_val:.2f}, and the on-site Coulomb repulsion U is {u_val:.2f}. "
        f"What are the ground state energy, local electron densities, and spin-spin correlations?"
    )
    
    # 3. Design the target response (what the LLM should learn to output)
    response = (
        f"For a 2x2 Fermi-Hubbard grid with t={t_val:.2f} and U={u_val:.2f}:\n\n"
        f"Ground State Energy: {E_0:.6f}\n\n"
        f"Local Electron Densities per site:\n"
        f"  Site 0 [0,0]: {density_ground[0]:.4f}\n"
        f"  Site 1 [0,1]: {density_ground[1]:.4f}\n"
        f"  Site 2 [1,0]: {density_ground[2]:.4f}\n"
        f"  Site 3 [1,1]: {density_ground[3]:.4f}\n\n"
        f"Spin-Spin Correlations <Sz_i Sz_j> Matrix:\n"
        f"  [{sz_sz_ground[0,0]:.4f}, {sz_sz_ground[0,1]:.4f}, {sz_sz_ground[0,2]:.4f}, {sz_sz_ground[0,3]:.4f}]\n"
        f"  [{sz_sz_ground[1,0]:.4f}, {sz_sz_ground[1,1]:.4f}, {sz_sz_ground[1,2]:.4f}, {sz_sz_ground[1,3]:.4f}]\n"
        f"  [{sz_sz_ground[2,0]:.4f}, {sz_sz_ground[2,1]:.4f}, {sz_sz_ground[2,2]:.4f}, {sz_sz_ground[2,3]:.4f}]\n"
        f"  [{sz_sz_ground[3,0]:.4f}, {sz_sz_ground[3,1]:.4f}, {sz_sz_ground[3,2]:.4f}, {sz_sz_ground[3,3]:.4f}]"
    )
    
    # 4. Wrap into OpenAI / HuggingFace standard shareable format
    return {
        "messages": [
            {"role": "system", "content": "You are an expert in quantum information science and tensor network simulations."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }

# --- Generation Loop ---
# 1. Define your target folder and file name
target_folder = "data"  # You can also use absolute paths like "C:/Users/.../data"
file_name = "fermi_hubbard_2x2_dataset.jsonl"

# 2. Use os.path.join to create the correct path for your operating system
output_file = os.path.join(target_folder, file_name)

# 3. Create the directory automatically if it doesn't exist yet
os.makedirs(target_folder, exist_ok=True)

num_samples = 50  # Start with 50 pairs to test, scale up to thousands later

print(f"Generating {num_samples} LLM training samples...")
with open(output_file, "w", encoding="utf-8") as f:
    for _ in range(num_samples):
        # Generate random values to sample parameter space
        t_sample = 1.0  # Keep hopping normalized
        u_sample = random.uniform(0.0, 10.0) # Sweep from weak to strong interaction
        
        try:
            datapoint = generate_llm_datapoint(t_sample, u_sample)
            f.write(json.dumps(datapoint) + "\n")
        except Exception as e:
            # Handle potential edge case solver failures gracefully
            continue

print(f"Dataset compiled and written to {output_file}!")