# importing necessary libraries
import pennylane as qml
from pennylane import numpy as np
import config, vqe_core

# setting up the molecule and its configuration
molecule_name = "LiH_R8"
molecule_config = config.MOLECULES[molecule_name]

# building the Hamiltonian and calculating the exact energy
hamiltonian, qubits = vqe_core.build_hamiltonian(molecule_config)
hamiltonian_matrix = qml.matrix(hamiltonian, wire_order=range(qubits))

eigenvalues = np.linalg.eigvalsh(hamiltonian_matrix)
exact_energy = eigenvalues[0]

print(f"Exact energy for {molecule_name}: {exact_energy}")