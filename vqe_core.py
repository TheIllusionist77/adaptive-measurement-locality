# importing necessary libraries
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import config

def build_hamiltonian(molecule_config):
    """
    Builds the hamiltonian for a given molecule configuration.
    
    :param molecule_config: A dictionary containing the symbols and coordinates of the molecule.
    """

    hamiltonian, qubits = qchem.molecular_hamiltonian(
        molecule_config["symbols"],
        np.array(molecule_config["coordinates"]),
        unit="angstrom",
        active_electrons=molecule_config["active_electrons"],
        active_orbitals=molecule_config["active_orbitals"]
    )

    return hamiltonian, qubits

def build_ansatz(hf_state, qubits, noise_params=None):
    """
    Builds the ansatz for the VQE algorithm.
    
    :param hf_state: The Hartree-Fock state of the system.
    :param qubits: The number of qubits in the system.
    :param noise_params: Dictionary containing noise model parameters.
    """

    def ansatz(params, depth):
        qml.BasisState(hf_state, wires=range(qubits))

        param_idx = 0
        for d in range(depth):
            for q in range(qubits):
                qml.RY(params[param_idx], wires=q)
                param_idx += 1

                if noise_params:
                    if noise_params.get("depolarizing_1q", 0.0) > 0.0:
                        qml.DepolarizingChannel(noise_params["depolarizing_1q"], wires=q)
                    if noise_params.get("amplitude", 0.0) > 0.0:
                        qml.AmplitudeDamping(noise_params["amplitude"], wires=q)
                    if noise_params.get("phase", 0.0) > 0.0:
                        qml.PhaseDamping(noise_params["phase"], wires=q)

                qml.RZ(params[param_idx], wires=q)
                param_idx += 1

                if noise_params:
                    if noise_params.get("depolarizing_1q", 0.0) > 0.0:
                        qml.DepolarizingChannel(noise_params["depolarizing_1q"], wires=q)
                    if noise_params.get("amplitude", 0.0) > 0.0:
                        qml.AmplitudeDamping(noise_params["amplitude"], wires=q)
                    if noise_params.get("phase", 0.0) > 0.0:
                        qml.PhaseDamping(noise_params["phase"], wires=q)

            for q in range(qubits - 1):
                qml.CNOT(wires=[q, q + 1])

                if noise_params:
                    if noise_params.get("depolarizing_2q", 0.0) > 0.0:
                        qml.DepolarizingChannel(noise_params["depolarizing_2q"], wires=q)
                        qml.DepolarizingChannel(noise_params["depolarizing_2q"], wires=q+1)

    return ansatz

def build_cost_function(dev, hamiltonian, ansatz, depth, k=None):
    """
    Builds the cost function for the VQE algorithm, optionally with a locality filter.
    
    :param dev: The quantum device used for the VQE algorithm.
    :param hamiltonian: The hamiltonian of the system.
    :param ansatz: The ansatz used in the VQE algorithm.
    :param depth: The depth of the ansatz circuit.
    :param k: The locality parameter for the hamiltonian.
    """

    observable = locality_filter(hamiltonian, k) if k else hamiltonian

    @qml.set_shots(config.GRAD_SHOTS)
    @qml.qnode(dev)
    def cost_function(params):
        ansatz(params, depth)
        return qml.expval(observable)

    return cost_function

def initialize_params(depth, qubits, seed, init_scale):
    """
    Initializes the parameters for the VQE algorithm.
    
    :param depth: The depth of the ansatz circuit.
    :param qubits: The number of qubits in the system.
    :param seed: The seed for the random number generator.
    :param init_scale: The scale factor for the initial parameters.
    """

    np.random.seed(seed)
    num_params = depth * qubits * 2

    return np.random.randn(num_params) * init_scale

def get_pauli_weight(observable):
    """
    Calculates the Pauli weight of a given observable.
    
    :param observable: The observable for which the Pauli weight is to be calculated.
    """

    if isinstance(observable, qml.Identity):
        return 0
    elif isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        return 1
    elif isinstance(observable, qml.ops.op_math.Prod):
        return sum(get_pauli_weight(op) for op in observable.operands)
    else:
        raise ValueError("Unknown Pauli operator type!")

def locality_filter(hamiltonian, k):
    """
    Filters the terms in the hamiltonian based on the locality parameter k.
    
    :param hamiltonian: The hamiltonian of the system.
    :param k: The locality parameter.
    """

    filtered_terms = []
    for term in hamiltonian.operands:
        obs = term.base
        coeff = term.scalar
        weight = get_pauli_weight(obs)

        if weight <= k:
            filtered_terms.append(coeff * obs)
    
    if len(filtered_terms) == 0:
        return 0.0 * qml.Identity(hamiltonian.wires[0])

    return qml.sum(*filtered_terms)

def build_shadow_circuit(dev, ansatz, depth):
    """
    Builds a quantum circuit that returns the classical shadow measurements.
    
    :param dev: The quantum device used for the circuit.
    :param ansatz: The ansatz used in the circuit.
    :param depth: The depth of the ansatz circuit.
    """

    @qml.set_shots(config.SHADOW_SHOTS)
    @qml.qnode(dev)
    def shadow_circuit(params):
        ansatz(params, depth)
        return qml.classical_shadow(wires=dev.wires)

    return shadow_circuit