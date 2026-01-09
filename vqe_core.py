# importing necessary libraries
import pennylane as qml
import numpy as np
import jax
import config
from pennylane import qchem

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

    def apply_noise(wires):
        if noise_params:
            p_dep = noise_params["dep_1q"] if len(wires) == 1 else noise_params["dep_2q"]
            for w in wires:
                qml.DepolarizingChannel(p_dep, wires=w)

    def ansatz(params, depth):
        qml.BasisState(hf_state, wires=range(qubits))

        param_idx = 0
        for d in range(depth):
            for q in range(qubits):
                qml.RX(params[param_idx], wires=q)
                param_idx += 1
                apply_noise([q])

                qml.RY(params[param_idx], wires=q)
                param_idx += 1
                apply_noise([q])

                qml.RZ(params[param_idx], wires=q)
                param_idx += 1
                apply_noise([q])

            for q in range(qubits):
                qml.IsingZZ(params[param_idx], wires=[q, (q+1) % qubits])
                param_idx += 1
                apply_noise([q, (q+1) % qubits])

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
    @qml.qnode(dev, cache=False, interface="jax")
    def cost_function(params):
        ansatz(params, depth)
        return qml.expval(observable)

    return cost_function

def initialize_params(depth, qubits, seed):
    """
    Initializes the parameters for the VQE algorithm.
    
    :param depth: The depth of the ansatz circuit.
    :param qubits: The number of qubits in the system.
    :param seed: The seed for the random number generator.
    """

    key = jax.random.PRNGKey(seed)
    num_params = depth * qubits * 4

    return jax.random.normal(key, (num_params,)) * np.pi * 2

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