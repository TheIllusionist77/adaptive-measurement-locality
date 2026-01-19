# importing necessary libraries
import pennylane as qml
import config, vqe_core
from pennylane import qchem
from protocols import *

def main():
    """Runs a single experiment with the given configuration."""

    # defining experiment configuration
    molecule_name = "LiH_R8"
    depth = 4
    protocol_config = {"type": "adaptive"}
    seed = 42

    # getting molecular configuration
    molecule_config = config.MOLECULES[molecule_name]

    # building Hamiltonian and Hartree-Fock state
    hamiltonian, qubits = vqe_core.build_hamiltonian(molecule_config)
    hf_state = qchem.hf_state(molecule_config["active_electrons"], qubits)

    # setting up quantum device and printing target energy
    dev = qml.device("default.mixed" if config.USE_NOISE else "lightning.qubit", wires=qubits)
    print(f"Target energy for {molecule_name}: {molecule_config["ground_state"]} Ha")

    # building ansatz and initializing parameters
    ansatz = vqe_core.build_ansatz(hf_state, qubits, config.NOISE_PARAMS if config.USE_NOISE else None)
    theta = vqe_core.initialize_params(depth, qubits, seed)

    # selecting and initializing protocol
    protocol_type = protocol_config["type"]
    if protocol_type == "adaptive":
        protocol = AdaptiveProtocol(dev, hamiltonian, ansatz, depth, molecule_config["ground_state"], qubits, verbose=True)
    elif protocol_type == "fixed":
        k = protocol_config["k"]
        protocol = FixedKProtocol(dev, hamiltonian, ansatz, depth, molecule_config["ground_state"], qubits, k, verbose=True)
    elif protocol_type == "global":
        protocol = GlobalProtocol(dev, hamiltonian, ansatz, depth, molecule_config["ground_state"], qubits, verbose=True)
        
    _, final_avg_energy, checkpoint = protocol.run(theta)
    while checkpoint is not None:
        _, final_avg_energy, checkpoint = protocol.run(None, checkpoint=checkpoint)

    print(f"Final average energy for {molecule_name}: {final_avg_energy:.5f} Ha!")

if __name__ == "__main__":
    main()