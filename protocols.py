# importing necessary libraries
import pennylane as qml
import time, config, diagnostics

class GlobalProtocol:
    """
    A global protocol for optimizing quantum circuits.
    """

    def __init__(self, training_cost, full_cost, learning_rate, max_steps, energy_threshold,
                 density_matrix_circuit=None, qubits=None):
        """
        Initialize the global protocol with the given parameters.
        
        :param training_cost: The cost function used for training.
        :param full_cost: The full cost function used for evaluation.
        :param learning_rate: The learning rate for the optimizer.
        :param max_steps: The maximum number of steps for the optimization.
        :param energy_threshold: The energy threshold for convergence.
        :param density_matrix_circuit: The density matrix circuit for diagnostics.
        :param qubits: The number of qubits in the system.
        """

        self.training_cost = training_cost
        self.full_cost = full_cost
        self.optimizer = qml.GradientDescentOptimizer(stepsize=learning_rate)
        self.max_steps = max_steps
        self.energy_threshold = energy_threshold
        self.convergence_window = config.CONVERGENCE_WINDOW
        self.density_matrix_circuit = density_matrix_circuit
        self.qubits = qubits

    def get_step_info(self, step, train_energy, full_energy):
        """
        Get string representation of the current step information.
        
        :param step: Current step number.
        :param train_energy: Training energy at the current step.
        :param full_energy: Full energy at the current step.
        """

        return f"Step {step}: Training Energy = {train_energy:.8f} Ha, Full Energy = {full_energy:.8f} Ha"

    def initialize_log(self):
        """
        Initialize the log dictionary to store optimization data.
        """

        log = {
            "step": [],
            "training_energy": [],
            "full_energy": [],
            "shots_used": [],
            "wall_time": []
        }

        if self.density_matrix_circuit:
            log.update({
                "gradient_snr": [],
                "gradient_norm": [],
                "gradient_std": [],
                "gradient_max": [],
                "avg_entropy": [],
                "max_entropy": []
            })
        
        return log
    
    def log_step(self, log, step, train_energy, full_energy, shots_used, wall_time, data=None):
        """
        Log the data for the current step.
        
        :param log: Log dictionary to store data.
        :param step: Current step number.
        :param train_energy: Training energy at the current step.
        :param full_energy: Full energy at the current step.
        :param shots_used: Number of shots used at the current step.
        :param wall_time: Wall time taken for the current step.
        :param data: Diagnostics data for the current step.
        """

        log["step"].append(step)
        log["training_energy"].append(train_energy)
        log["full_energy"].append(full_energy)
        log["shots_used"].append(shots_used)
        log["wall_time"].append(wall_time)

        if data:
            log["gradient_snr"].append(data["gradient_snr"])
            log["gradient_norm"].append(data["gradient_norm"])
            log["gradient_std"].append(data["gradient_std"])
            log["gradient_max"].append(data["gradient_max"])
            log["avg_entropy"].append(data["avg_entropy"])
            log["max_entropy"].append(data["max_entropy"])

    def get_final_results(self, theta, log, converged):
        """
        Get the final results after optimization.

        :param theta: Final parameters after optimization.
        :param log: Log dictionary containing optimization data.
        :param converged: Boolean indicating whether the optimization converged.
        """

        total_shots = sum(log["shots_used"])
        total_time = sum(log["wall_time"])

        return {
            "final_params": theta,
            "log": log,
            "final_energy": log["full_energy"][-1],
            "steps": len(log["step"]) - 1,
            "converged": converged,
            "total_shots": total_shots,
            "total_time": total_time
        }

    def run(self, init_params):
        """
        Run the protocol to optimize the quantum circuit parameters.
        
        :param init_params: Initial parameters for the quantum circuit.
        """

        # initialize parameters and log
        theta = init_params.copy()
        log = self.initialize_log()

        step_start = time.time()
        initial_train = self.training_cost(theta)
        initial_full = self.full_cost(theta)

        if self.density_matrix_circuit:
            grad = qml.grad(self.training_cost)(theta)
            density_matrix = self.density_matrix_circuit(theta)
            data = diagnostics.compute_diagnostics(grad, density_matrix, self.qubits)
            self.log_step(log, 0, initial_train, initial_full, 2 * config.SHOTS_PER_STEP, time.time() - step_start, data)
            print(self.get_step_info(0, initial_train, initial_full))
            print(diagnostics.get_diagnostics(data, 0))
        else:
            self.log_step(log, 0, initial_train, initial_full, 2 * config.SHOTS_PER_STEP, time.time() - step_start)
            print(self.get_step_info(0, initial_train, initial_full))

        # optimization loop
        converged = False
        for step in range(1, self.max_steps + 1):
            step_start = time.time()
            theta, train_energy = self.optimizer.step_and_cost(self.training_cost, theta)
            full_energy = self.full_cost(theta)

            if self.density_matrix_circuit:
                grad = qml.grad(self.training_cost)(theta)
                density_matrix = self.density_matrix_circuit(theta)
                data = diagnostics.compute_diagnostics(grad, density_matrix, self.qubits)
                self.log_step(log, step, train_energy, full_energy, 2 * config.SHOTS_PER_STEP, time.time() - step_start, data)
                print(self.get_step_info(step, train_energy, full_energy))
                print(diagnostics.get_diagnostics(data, step))
            else:
                self.log_step(log, step, train_energy, full_energy, 2 * config.SHOTS_PER_STEP, time.time() - step_start)
                print(self.get_step_info(step, train_energy, full_energy))

            if step >= self.convergence_window:
                recent_energies = log["full_energy"][-self.convergence_window:]
                avg_energy = sum(recent_energies) / len(recent_energies)
                
                if avg_energy < self.energy_threshold:
                    print(f"Converged at step {step}!")
                    converged = True
                    break
        
        return self.get_final_results(theta, log, converged)
    
class FixedKProtocol(GlobalProtocol):
    """
    A fixed-k protocol for optimizing quantum circuits.
    """

    def __init__(self, training_cost, full_cost, learning_rate, max_steps, energy_threshold, k,
                 density_matrix_circuit=None, qubits=None):
        """
        Initialize the fixed-k protocol with the given parameters.
        
        :param k: The fixed locality parameter.
        """

        super().__init__(training_cost, full_cost, learning_rate, max_steps, energy_threshold,
                         density_matrix_circuit, qubits)
        self.k = k

    def get_step_info(self, step, train_energy, full_energy):
        """
        Override to include k in the step info.
        """

        return f"Step {step} (k={self.k}): Training Energy = {train_energy:.8f} Ha, Full Energy = {full_energy:.8f} Ha"
    
    def initialize_log(self):
        """
        Override to include k in the log.
        """

        log = super().initialize_log()
        log["k"] = []
        return log
    
    def log_step(self, log, step, train_energy, full_energy, shots_used, wall_time, data=None):
        """
        Override to log k in each step.
        """

        super().log_step(log, step, train_energy, full_energy, shots_used, wall_time, data)
        log["k"].append(self.k)
    
    def get_final_results(self, theta, log, converged):
        """
        Override to include k in final results.
        """

        results = super().get_final_results(theta, log, converged)
        results["k"] = self.k
        return results