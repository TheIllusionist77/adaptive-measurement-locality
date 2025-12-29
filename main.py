# importing necessary libraries
import pennylane as qml
import config, vqe_core
from protocols import *

# defining the main function to run the VQE protocol
def main():
    # setting up the molecule and its configuration
    molecule_name = "LiH_R8"
    molecule_config = config.MOLECULES[molecule_name]

    # building the Hamiltonian and the Hartree-Fock state
    hamiltonian, qubits = vqe_core.build_hamiltonian(molecule_config)
    hf_state = vqe_core.get_hf_state(molecule_config["electrons"], qubits)

    # setting up the quantum device and calculating the Hartree-Fock energy
    dev = qml.device("default.mixed", wires=qubits)

    @qml.set_shots(config.SHOTS_PER_STEP)
    @qml.qnode(dev)
    def hf_energy():
        qml.BasisState(hf_state, wires=range(qubits))
        return qml.expval(hamiltonian)

    print(f"Hartree-Fock energy for {molecule_name}: {hf_energy():.8f} Ha")
    print(f"Target energy for {molecule_name}: {molecule_config["ground_state"]} Ha")

    # running the VQE protocol to find the ground state energy
    ansatz = vqe_core.build_ansatz(hf_state, qubits, config.NOISE_PARAMS)
    depth = config.DEPTHS[0]

    # initializing the parameters
    theta = vqe_core.initialize_params(depth, qubits, config.SEED, config.INIT_SCALE)
    density_matrix_circuit = vqe_core.build_density_matrix_circuit(dev, ansatz, depth)

    # running the VQE protocol and printing the final energy
    energy_threshold = molecule_config["ground_state"] * config.THRESHOLD_SCALAR
    protocol = AdaptiveProtocol(dev, hamiltonian, ansatz, depth, energy_threshold, density_matrix_circuit, qubits)
    log = protocol.run(theta)

    print(f"Final energy for {molecule_name}: {log["full_energy"][-1]:.8f} Ha")

if __name__ == "__main__":
    main()