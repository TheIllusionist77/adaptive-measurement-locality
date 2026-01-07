# importing necessary libraries
import pennylane as qml
import time, jax, optax
import config, diagnostics, vqe_core
import jax.numpy as jnp

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
        self.optimizer = optax.sgd(config.LEARNING_RATE)
        self.grad_fn = jax.jit(jax.grad(self.training_cost))

        self.log = {
            "step": [],
            "train_energy": [],
            "full_energy": [],
            "shots_used": [],
            "wall_time": [],
            "grad_var": [],
            "grad_align": [],
            "improvement": [],
            "locality_k": []
        }

    def get_step_info(self, data):
        """
        Get string representation of the current step information.
        
        :param data: The dictionary containing diagnostics data.
        """

        if self.verbose:
            print(f"Step {self.log["step"][-1]} (k={self.k}): Training Energy = {self.log["train_energy"][-1]:.4f} Ha, Full Energy = {self.log["full_energy"][-1]:.4f} Ha")
            print(diagnostics.get_diagnostics(data) + f", Shots Used = {self.log["shots_used"][-1]}")

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
        self.log["train_energy"].append(float(train_energy))
        self.log["full_energy"].append(float(full_energy))
        self.log["shots_used"].append(total_shots)
        self.log["wall_time"].append(total_time)
        self.log["grad_var"].append(float(data["grad_var"]))
        self.log["grad_align"].append(float(data["grad_align"]))
        self.log["improvement"].append(float(data["improvement"]))
        self.log["locality_k"].append(self.k)

    def adjust_k(self):
        """Subclasses may override to adjust k based on diagnostics."""
        pass

    def run(self, theta, progress_cb=None):
        """
        Run the protocol to optimize the quantum circuit parameters.
        
        :param theta: Initial parameters for the quantum circuit.
        :param progress_cb: Callback function to increment progress bars.
        """

        # initializing various things
        prev_train = None
        prev_full = None
        align_ema = 0.0
        opt_state = self.optimizer.init(theta)

        # optimization loop
        for step in range(1, config.MAX_STEPS + 1):
            step_start = time.time()

            with qml.Tracker(self.dev) as tracker:
                curr_train = self.training_cost(theta)
                curr_full = self.full_cost(theta)
                
                grad = self.grad_fn(theta)
                data = diagnostics.compute_diagnostics(grad, curr_train, curr_full, prev_train, prev_full, align_ema)
                updates, opt_state = self.optimizer.update(grad, opt_state, theta)
                theta = optax.apply_updates(theta, updates)

            # log step information and update previous parameters
            self.log_step(step, curr_train, curr_full, tracker.totals.get("shots"), time.time() - step_start, data)
            self.get_step_info(data)

            prev_train = curr_train
            prev_full = curr_full
            align_ema = data["grad_align"]

            if progress_cb:
                progress_cb(1)

            # adjust k based on diagnostics, if needed
            old_k = self.k
            if step > config.IMPROVEMENT_STEPS:
                self.adjust_k()

            if old_k != self.k:
                self.training_cost = vqe_core.build_cost_function(self.dev, self.hamiltonian, self.ansatz, self.depth, self.k)
                self.grad_fn = jax.jit(jax.grad(self.training_cost))
                align_ema = 0.0

                if self.verbose:
                    modification = "deescalated" if old_k > self.k else "escalated"
                    print(f"[Locality] k {modification} from {old_k} to {self.k}.")

            # check if optimizer has converged
            if step >= config.CONVERGENCE_WINDOW:
                recent_energies = self.log["full_energy"][-config.CONVERGENCE_WINDOW:]
                avg_energy = sum(recent_energies) / len(recent_energies)

                lower_bound = self.ground_state - config.ENERGY_THRESHOLD
                upper_bound = self.ground_state + config.ENERGY_THRESHOLD
                
                if lower_bound <= avg_energy <= upper_bound:
                    if self.verbose:
                        print(f"Converged at step {step}!")
                    if progress_cb:
                        progress_cb(config.MAX_STEPS - step)
                    break
        
        recent_energies = self.log["full_energy"][-config.CONVERGENCE_WINDOW:]
        final_avg_energy = sum(recent_energies) / len(recent_energies)
        
        return self.log, final_avg_energy

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
        self.grad_fn = jax.jit(jax.grad(self.training_cost))
    
class AdaptiveProtocol(GlobalProtocol):
    """An adaptive protocol that adjusts k during optimization based on diagnostics."""

    def __init__(self, dev, hamiltonian, ansatz, depth, ground_state, qubits, verbose=False):
        """Override to initialize the adaptive protocol with the given parameters."""

        super().__init__(dev, hamiltonian, ansatz, depth, ground_state, qubits, verbose)
        self.k = 1
        self.escalation_counter = 0
        self.deescalation_counter = 0
        self.training_cost = vqe_core.build_cost_function(dev, hamiltonian, ansatz, depth, self.k)
        self.grad_fn = jax.jit(jax.grad(self.training_cost))

    def adjust_k(self):
        """Determine whether to escalate, deescalate, or maintain k based on diagnostics."""

        grad_var = self.log["grad_var"][-1]
        grad_align = self.log["grad_align"][-1]
        improvement = jnp.array(self.log["improvement"][-config.IMPROVEMENT_STEPS:])
        step_bias = self.log["step"][-1] / (config.CONVERGENCE_WINDOW * 2)
        
        improving = jnp.mean(improvement) / jnp.mean(jnp.abs(improvement)) <= config.IMPROVEMENT_THRESHOLD
        raise_condition = grad_align <= config.ALIGN_THRESHOLD and not improving
        lower_condition = grad_var <= (config.VAR_THRESHOLD / (2 ** (step_bias))) and not improving

        if raise_condition:
            self.escalation_counter += 1
        else:
            self.escalation_counter = 0

        if lower_condition:
            self.deescalation_counter += 1
        else:
            self.deescalation_counter = 0

        if self.escalation_counter >= (config.HYSTERESIS + self.k ** 2) and self.k < self.qubits:
            self.k += 1
            self.escalation_counter = 0
            self.deescalation_counter = 0
        elif self.deescalation_counter >= (config.HYSTERESIS + step_bias) and self.k > 1:
            self.k -= 1
            self.escalation_counter = 0
            self.deescalation_counter = 0