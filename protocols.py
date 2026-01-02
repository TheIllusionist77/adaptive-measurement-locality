# importing necessary libraries
import pennylane as qml
import time, config, diagnostics, vqe_core

class GlobalProtocol:
    """A global protocol for optimizing quantum circuits."""

    def __init__(self, dev, hamiltonian, ansatz, depth, ground_state, qubits, verbose=False):
        """
        Initialize the global protocol with the given parameters.
        
        :param dev: The quantum device used for the VQE algorithm.
        :param hamiltonian: The hamiltonian of the system.
        :param ansatz: The ansatz used in the VQE algorithm.
        :param depth: The depth of the ansatz circuit.
        :param ground_state: The ground state energy of the molecule.
        :param qubits: The number of qubits in the system.
        :param verbose: Whether to print progress information.
        """
        
        self.dev = dev
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.depth = depth
        self.ground_state = ground_state
        self.qubits = qubits
        self.k = qubits
        self.verbose = verbose

        self.training_cost = vqe_core.build_cost_function(dev, hamiltonian, ansatz, depth)
        self.full_cost = vqe_core.build_cost_function(dev, hamiltonian, ansatz, depth)
        self.grad_fn = qml.grad(self.training_cost)
        self.shadow_circuit = vqe_core.build_shadow_circuit(dev, ansatz, depth)

        self.log = {
            "step": [],
            "train_energy": [],
            "full_energy": [],
            "shots_used": [],
            "wall_time": [],
            "gradient_snr": [],
            "avg_entropy": [],
            "locality_k": []
        }

    def get_step_info(self, data):
        """
        Get string representation of the current step information.
        
        :param data: The dictionary containing diagnostics data.
        """

        if self.verbose:
            print(f"Step {self.log["step"][-1]} (k={self.k}): Training Energy = {self.log["train_energy"][-1]:.8f} Ha, Full Energy = {self.log["full_energy"][-1]:.8f} Ha")
            print(diagnostics.get_diagnostics(data) + f", Shots Used = {self.log["shots_used"][-1]}, Time = {round(self.log["wall_time"][-1], 3)}s")

    def log_step(self, step, train_energy, full_energy, shots_used, wall_time, data):
        """
        Log the data for the current step.
        
        :param step: Current step number.
        :param train_energy: Training energy at the current step.
        :param full_energy: Full energy at the current step.
        :param shots_used: Number of shots used at the current step.
        :param wall_time: Wall time taken for the current step.
        :param data: Diagnostics data for the current step.
        """

        total_shots = self.log["shots_used"][-1] + shots_used if self.log["shots_used"] else shots_used
        total_time = self.log["wall_time"][-1] + wall_time if self.log["wall_time"] else wall_time

        self.log["step"].append(step)
        self.log["train_energy"].append(train_energy)
        self.log["full_energy"].append(full_energy)
        self.log["shots_used"].append(total_shots)
        self.log["wall_time"].append(total_time)
        self.log["gradient_snr"].append(data["gradient_snr"])
        self.log["avg_entropy"].append(data["avg_entropy"])
        self.log["locality_k"].append(self.k)

    def adjust_k(self, data):
        """Subclasses may override to adjust k based on diagnostics."""
        pass

    def run(self, theta):
        """
        Run the protocol to optimize the quantum circuit parameters.
        
        :param theta: Initial parameters for the quantum circuit.
        """

        # optimization loop
        for step in range(1, config.MAX_STEPS + 1):
            step_start = time.time()

            with qml.Tracker(self.dev) as tracker:
                shadow_data = self.shadow_circuit(theta)
                entropies = diagnostics.compute_subsystem_entropies(shadow_data, self.qubits)

                train_energy = self.training_cost(theta)
                full_energy = self.full_cost(theta)
                
                grad = self.grad_fn(theta)
                data = diagnostics.compute_diagnostics(grad, entropies)
                theta = theta - config.LEARNING_RATE * grad

            self.log_step(step, train_energy, full_energy, tracker.totals.get("shots"), time.time() - step_start, data)
            self.get_step_info(data)

            # adjust k based on diagnostics, if needed
            old_k = self.k
            self.adjust_k(data)

            if old_k != self.k:
                self.training_cost = vqe_core.build_cost_function(self.dev, self.hamiltonian, self.ansatz, self.depth, self.k)
                self.grad_fn = qml.grad(self.training_cost)

                if self.verbose:
                    modification = "deescalated" if old_k > self.k else "escalated"
                    print(f"[Locality] k {modification} from {old_k} to {self.k}.")

            if step >= config.CONVERGENCE_WINDOW:
                recent_energies = self.log["full_energy"][-config.CONVERGENCE_WINDOW:]
                avg_energy = sum(recent_energies) / len(recent_energies)

                lower_bound = self.ground_state
                upper_bound = self.ground_state / config.ENERGY_THRESHOLD
                
                if lower_bound <= avg_energy <= upper_bound:
                    if self.verbose:
                        print(f"Converged at step {step}!")
                    break
        
        return self.log

class FixedKProtocol(GlobalProtocol):
    """A fixed-k protocol for optimizing quantum circuits."""

    def __init__(self, dev, hamiltonian, ansatz, depth, ground_state, qubits, k, verbose=False):
        """
        Override to initialize the fixed-k protocol with the given parameters.
        
        :param k: The fixed locality parameter.
        """

        super().__init__(dev, hamiltonian, ansatz, depth, ground_state, qubits, verbose)
        self.k = k
        self.training_cost = vqe_core.build_cost_function(dev, hamiltonian, ansatz, depth, k)
        self.grad_fn = qml.grad(self.training_cost)
    
class AdaptiveProtocol(GlobalProtocol):
    """An adaptive protocol that adjusts k during optimization based on diagnostics."""

    def __init__(self, dev, hamiltonian, ansatz, depth, ground_state, qubits, verbose=False):
        """Override to initialize the adaptive protocol with the given parameters."""

        super().__init__(dev, hamiltonian, ansatz, depth, ground_state, qubits, verbose)
        self.k = 1
        self.escalation_counter = 0
        self.deescalation_counter = 0
        self.training_cost = vqe_core.build_cost_function(dev, hamiltonian, ansatz, depth, self.k)
        self.grad_fn = qml.grad(self.training_cost)

    def adjust_k(self, data):
        """
        Override to determine whether to escalate, deescalate, or maintain k based on diagnostics.
        
        :param data: Diagnostics data for the current step.
        """

        gradient_snr = data["gradient_snr"]
        entropy = data["avg_entropy"]

        snr_raise = gradient_snr >= config.ESCALATION_THRESHOLDS.get("gradient_snr")
        entropy_raise = entropy <= config.ESCALATION_THRESHOLDS.get("avg_entropy")
        snr_lower = gradient_snr <= config.DEESCALATION_THRESHOLDS.get("gradient_snr")
        entropy_lower = entropy >= config.DEESCALATION_THRESHOLDS.get("avg_entropy")

        raise_condition = snr_raise and entropy_raise and self.k < self.qubits
        lower_condition = (snr_lower or entropy_lower) and self.k > 1

        if raise_condition:
            self.escalation_counter += 1
            self.deescalation_counter = 0

            if self.escalation_counter >= config.HYSTERESIS:
                self.k += 1
                self.escalation_counter = 0
        elif lower_condition:
            self.deescalation_counter += 1
            self.escalation_counter = 0

            if self.deescalation_counter >= config.HYSTERESIS:
                self.k -= 1
                self.deescalation_counter = 0
        else:
            self.escalation_counter = 0
            self.deescalation_counter = 0