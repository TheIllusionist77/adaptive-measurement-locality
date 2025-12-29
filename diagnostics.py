# importing necessary libraries
import pennylane as qml
from pennylane import numpy as np
import config

def compute_subsystem_entropies(density_matrix, total_qubits):
    """
    Computes the Rényi-2 entropies for all adjacent two-qubit subsystems.
    
    :param density_matrix: The full density matrix.
    :param total_qubits: Total number of qubits in the system.
    """

    entropies = {}

    for i in range(total_qubits - 1):
        subsystem = (i, i + 1)
        rho = qml.math.reduce_dm(density_matrix, indices=list(subsystem))

        purity = np.trace(rho @ rho).real
        purity = np.clip(purity, 0.0, 1.0)

        entropies[subsystem] = -np.log2(purity)

    return entropies

def compute_diagnostics(grad, density_matrix, total_qubits):
    """
    Computes various diagnostics for the optimization process.
    
    :param grad: The gradient array.
    :param density_matrix: The full density matrix.
    :param total_qubits: Total number of qubits in the system.
    """

    entropies = compute_subsystem_entropies(density_matrix, total_qubits)

    gradient_mean = float(np.mean(np.abs(grad)))
    gradient_std = 1.0 / np.sqrt(config.SHOTS_PER_STEP)

    if gradient_std > 1e-10:
        gradient_snr = gradient_mean / gradient_std
    else:
        gradient_snr = float("inf") if gradient_mean > 1e-10 else 0.0

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