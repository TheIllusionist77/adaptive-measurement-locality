# importing necessary libraries
import pennylane as qml
import pandas as pd
import os, json, multiprocessing
import config, vqe_core
from pennylane import qchem
from tqdm import tqdm
from datetime import datetime
from protocols import *

def generate_experiment_queue():
    """Generates a list of all experiment configurations to run."""

    queue = []
    for molecule_name in config.EXPERIMENT_CONFIG["molecules"]:
        for depth in config.EXPERIMENT_CONFIG["depths"]:
            for protocol_config in config.EXPERIMENT_CONFIG["protocols"]:
                for seed in config.EXPERIMENT_CONFIG["seeds"]:
                    protocol_type = protocol_config["type"]

                    if protocol_type == "fixed":
                        k = protocol_config["k"]
                        exp_id = f"{molecule_name}_d{depth}_{protocol_type}_k{k}_seed{seed}"
                    else:
                        exp_id = f"{molecule_name}_d{depth}_{protocol_type}_seed{seed}"

                    experiment = {
                        "id": exp_id,
                        "molecule": molecule_name,
                        "depth": depth,
                        "protocol": protocol_config,
                        "seed": seed
                    }

                    queue.append(experiment)
    
    return queue

def check_completed_experiments():
    """Generates a list of all experiment configurations already completed."""

    completed = set()
    if not os.path.exists(config.OUTPUT_DIR):
        return completed
    
    for exp_id in os.listdir(config.OUTPUT_DIR):
        path = os.path.join(config.OUTPUT_DIR, exp_id)

        if os.path.isdir(path):
            csv_exists = os.path.exists(os.path.join(path, "log.csv"))
            json_exists = os.path.exists(os.path.join(path, "metadata.json"))
            
            if csv_exists and json_exists:
                completed.add(exp_id)

    return completed

def run_experiment(exp_config):
    """
    Runs a single experiment with the given configuration.
    
    :param exp_config: Dictionary containing experimental configuration.
    """

    # extract experiment configuration from dictionary
    exp_id = exp_config["id"]
    molecule_name = exp_config["molecule"]
    depth = exp_config["depth"]
    protocol_config = exp_config["protocol"]
    seed = exp_config["seed"]

    # get molecule configuration
    molecule_config = config.MOLECULES[molecule_name]

    # building the Hamiltonian and the Hartree-Fock state
    hamiltonian, qubits = vqe_core.build_hamiltonian(molecule_config)
    hf_state = qchem.hf_state(molecule_config["electrons"], qubits)

    # setting up the quantum device and calculating the Hartree-Fock energy
    dev = qml.device("default.mixed" if config.USE_NOISE else "default.qubit", wires=qubits)

    @qml.qnode(dev)
    def hf_energy():
        qml.BasisState(hf_state, wires=range(qubits))
        return qml.expval(hamiltonian)
    
    hf_energy_value = hf_energy()

    # building the ansatz and initializing parameters
    ansatz = vqe_core.build_ansatz(hf_state, qubits, config.NOISE_PARAMS if config.USE_NOISE else None)
    theta = vqe_core.initialize_params(depth, qubits, seed)

    # select and run protocol
    protocol_type = protocol_config["type"]
    if protocol_type == "adaptive":
        protocol = AdaptiveProtocol(dev, hamiltonian, ansatz, depth,
                                    molecule_config["ground_state"], qubits)
    elif protocol_type == "fixed":
        k = protocol_config["k"]
        protocol = FixedKProtocol(dev, hamiltonian, ansatz, depth,
                                  molecule_config["ground_state"], qubits, k)
    elif protocol_type == "global":
        protocol = GlobalProtocol(dev, hamiltonian, ansatz, depth,
                                  molecule_config["ground_state"], qubits)
        
    log = protocol.run(theta)
    result = {
        "exp_id": exp_id,
        "log": log,
        "metadata": {
            "molecule": molecule_name,
            "depth": depth,
            "protocol": protocol_config,
            "seed": seed,
            "qubits": qubits,
            "hf_energy": float(hf_energy_value),
            "target_energy": molecule_config["ground_state"],
            "final_energy": float(log["full_energy"][-1]),
            "total_steps": int(len(log["step"])),
            "converged": bool(len(log["step"]) < config.MAX_STEPS),
            "time": datetime.now().isoformat()
        }
    }

    return result

def save_results(result):
    """
    Saves experiment results to disk.
    
    :param result: Dictionary containing experimental results.
    """
    
    exp_id = result["exp_id"]
    exp_dir = os.path.join(config.OUTPUT_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    # save experiment log as .csv
    log_df = pd.DataFrame(result["log"])
    log_df.to_csv(os.path.join(exp_dir, "log.csv"), index=False)

    # save experiment metadata as .json
    with open(os.path.join(exp_dir, "metadata.json"), "w") as file:
        json.dump(result["metadata"], file, indent=2)

def main():
    """Main function to conduct all experiments across multiple cores."""

    all_experiments = generate_experiment_queue()
    print(f"Total experiments in configuration: {len(all_experiments)}")

    completed = check_completed_experiments()
    print(f"Already completed experiments: {len(completed)}")

    remaining = [exp for exp in all_experiments if exp["id"] not in completed]
    if len(remaining) == 0:
        print("All experiments have been completed!")
        return
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"Running {len(remaining)} experiments on {config.NUM_CORES} cores...")

    with multiprocessing.Pool(config.NUM_CORES) as pool:
        with tqdm(total=len(remaining), desc="Overall Progress", position=0) as pbar:
            for result in pool.imap_unordered(run_experiment, remaining):
                save_results(result)
                pbar.set_postfix({
                    "Last": result["exp_id"][:30],
                    "Energy": f"{result["metadata"]["final_energy"]:.5f}"
                })
                pbar.update(1)

if __name__ == "__main__":
    main()