# importing necessary libraries
import pennylane as qml
import config, vqe_core
from protocols import *

# defining the main function to run the VQE protocol
def main():
    # setting up the molecule and its configuration
    molecule_name = "H2"
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
    ansatz = vqe_core.build_ansatz(hf_state, qubits, noise_params=config.NOISE_PARAMS)
    depth = config.DEPTHS[0]

    # building the cost functions and initializing the parameters
    training_cost = vqe_core.build_cost_function(dev, hamiltonian, ansatz, depth, k=4)
    full_cost = vqe_core.build_cost_function(dev, hamiltonian, ansatz, depth, k=None)
    theta = vqe_core.initialize_params(depth, qubits, config.SEED, config.INIT_SCALE)

    density_matrix_circuit = vqe_core.build_density_matrix_circuit(dev, ansatz, depth)

    # running the VQE protocol and printing the final energy
    energy_threshold = molecule_config["ground_state"] * config.THRESHOLD_SCALAR
    protocol = FixedKProtocol(training_cost, full_cost, config.LEARNING_RATE, config.MAX_STEPS, energy_threshold, k=4,
                              density_matrix_circuit=density_matrix_circuit, qubits=qubits)
    results = protocol.run(theta)

    print(f"Final energy for {molecule_name}: {results["final_energy"]:.8f} Ha")

if __name__ == "__main__":
    main()