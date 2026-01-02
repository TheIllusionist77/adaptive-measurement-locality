# importing necessary libraries
import pennylane as qml
import config, vqe_core
from pennylane import qchem
from protocols import *

def run_experiment():
    """Runs a single experiment with the given configuration."""

    # experimental configuration
    molecule_name = "LiH_R8"
    depth = 4
    protocol_config = {"type": "adaptive"}
    seed = 42

    # get molecule configuration
    molecule_config = config.MOLECULES[molecule_name]

    # build Hamiltonian and Hartree-Fock state
    hamiltonian, qubits = vqe_core.build_hamiltonian(molecule_config)
    hf_state = qchem.hf_state(molecule_config["electrons"], qubits)

    # setting up the quantum device and calculating the Hartree-Fock energy
    dev = qml.device("default.mixed" if config.USE_NOISE else "default.qubit", wires=qubits)

    @qml.qnode(dev)
    def hf_energy():
        qml.BasisState(hf_state, wires=range(qubits))
        return qml.expval(hamiltonian)
    
    print(f"Hartree-Fock energy for {molecule_name}: {hf_energy():.8f} Ha")
    print(f"Target energy for {molecule_name}: {molecule_config["ground_state"]} Ha")

    # building the ansatz and initializing parameters
    ansatz = vqe_core.build_ansatz(hf_state, qubits, config.NOISE_PARAMS if config.USE_NOISE else None)
    theta = vqe_core.initialize_params(depth, qubits, seed)

    # select and run protocol
    protocol_type = protocol_config["type"]
    if protocol_type == "adaptive":
        protocol = AdaptiveProtocol(dev, hamiltonian, ansatz, depth,
                                    molecule_config["ground_state"], qubits, verbose=True)
    elif protocol_type == "fixed":
        k = protocol_config["k"]
        protocol = FixedKProtocol(dev, hamiltonian, ansatz, depth,
                                  molecule_config["ground_state"], qubits, k, verbose=True)
    elif protocol_type == "global":
        protocol = GlobalProtocol(dev, hamiltonian, ansatz, depth,
                                  molecule_config["ground_state"], qubits, verbose=True)
        
    protocol.run(theta)

if __name__ == "__main__":
    run_experiment()