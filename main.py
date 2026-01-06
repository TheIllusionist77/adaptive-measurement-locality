# importing necessary libraries
import pennylane as qml
import pandas as pd
import os, json, multiprocessing, random
import config, vqe_core
from pennylane import qchem
from tqdm import tqdm
from datetime import datetime
from protocols import *

def init_worker(tqdm_lock):
    """
    Initializes a multiprocessing worker with a unique ID and tqdm lock.
    
    :param tqdm_lock: Multiprocessing lock for thread-safe progress updates.
    """

    global WORKER_ID
    WORKER_ID = multiprocessing.current_process()._identity[0]
    tqdm.set_lock(tqdm_lock)

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
    
    random.shuffle(queue)
    return queue

def check_completed_experiments():
    """Generates a list of all experiment configurations already completed."""

    completed = set()
    json_path = os.path.join(config.OUTPUT_DIR, "metadata.json")
    
    if os.path.exists(json_path):
        with open(json_path, "r") as file:
            metadata = json.load(file)
            completed = set(metadata.keys())

    return completed

def run_experiment(exp_config):
    """
    Runs a single experiment with the given configuration.
    
    :param exp_config: Dictionary containing experimental configuration.
    """

    global WORKER_ID
    worker_id = WORKER_ID

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
    hf_state = qchem.hf_state(molecule_config["active_electrons"], qubits)

    # setting up the quantum device
    dev = qml.device("default.mixed" if config.USE_NOISE else "lightning.qubit", wires=qubits)

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
        
    BAR_FORMAT = "{desc}{percentage:6.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    with tqdm(total=config.MAX_STEPS, desc=f"Process {worker_id:03d}: {exp_id}".ljust(40),
              bar_format=BAR_FORMAT, position=worker_id, leave=False, dynamic_ncols=True) as pbar:
        log, final_avg_energy = protocol.run(theta, progress_cb=lambda n: pbar.update(n))
        
    result = {
        "exp_id": exp_id,
        "log": log,
        "metadata": {
            "molecule": molecule_name,
            "depth": depth,
            "protocol": protocol_config,
            "seed": seed,
            "qubits": qubits,
            "target_energy": molecule_config["ground_state"],
            "final_avg_energy": final_avg_energy,
            "total_steps": log["step"][-1],
            "total_shots": log["shots_used"][-1],
            "total_time": log["wall_time"][-1],
            "converged": bool(len(log["step"]) < config.MAX_STEPS),
            "time": datetime.now().isoformat()
        }
    }

    return result

def save_results(result, lock):
    """
    Saves experiment results to disk.
    
    :param result: Dictionary containing experimental results.
    :param lock: Multiprocessing lock for thread-safe file writing.
    """
    
    # extract experiment IDs and define the output directory
    exp_id = result["exp_id"]
    logs_dir = os.path.join(config.OUTPUT_DIR, "logs")

    # create the output directories, if needed
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # save experiment log as .csv
    log_df = pd.DataFrame(result["log"])
    log_df.to_csv(os.path.join(logs_dir, f"{exp_id}_log.csv"), index=False)

    # save experiment metadata as .json
    json_path = os.path.join(config.OUTPUT_DIR, "metadata.json")
    with lock:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        metadata[exp_id] = result["metadata"]
        with open(json_path, "w") as file:
            json.dump(metadata, file, indent=4)

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

    file_lock = multiprocessing.Lock()
    tqdm_lock = multiprocessing.RLock()

    with multiprocessing.Pool(config.NUM_CORES, init_worker, (tqdm_lock,), maxtasksperchild=1) as pool:
        with tqdm(total=len(remaining), desc="Overall Progress", position=0) as pbar:
            for result in pool.imap_unordered(run_experiment, remaining):
                save_results(result, file_lock)
                pbar.set_postfix({"Last": result["exp_id"]})
                pbar.update(1)

if __name__ == "__main__":
    main()