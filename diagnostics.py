# importing necessary libraries
import pennylane as qml
from pennylane import numpy as np
import config

def compute_subsystem_entropies(shadow_data, total_qubits):
    """
    Computes the Rényi-2 entropies for all adjacent two-qubit subsystems using classical shadows.
    
    :param shadow_data: The classical shadow data.
    :param total_qubits: Total number of qubits in the system.
    """

    bits, recipes = shadow_data
    shadow = qml.ClassicalShadow(bits, recipes)
    entropies = {}

    for i in range(total_qubits - 1):
        subsystem = (i, i + 1)

        entropy_value = shadow.entropy(wires=list(subsystem), alpha=2, base=2)
        entropies[subsystem] = float(entropy_value)
    
    return entropies

def compute_diagnostics(grad, entropies):
    """
    Computes various diagnostics for the optimization process.
    
    :param grad: The gradient array.
    :param entropies: The dictionary of subsystem entropies.
    """

    gradient_mean = float(np.mean(np.abs(grad)))
    gradient_std = 1.0 / np.sqrt(config.GRAD_SHOTS)
    gradient_snr = gradient_mean / gradient_std

    diagnostics = {
        "gradient_snr": gradient_snr,
        "avg_entropy": float(np.mean(list(entropies.values())))
    }

    return diagnostics

def get_diagnostics(data):
    """
    Formats diagnostics information for logging.
    
    :param data: The dictionary containing diagnostics data.
    """

    return (f"[Diagnostics] SNR = {data["gradient_snr"]:.2f}, Avg Entropy: {data["avg_entropy"]:.3f}")