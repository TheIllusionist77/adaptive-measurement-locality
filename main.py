# importing necessary libraries
import pennylane as qml
import pandas as pd
import os, json, multiprocessing, random, pickle, time
import config, vqe_core
from pennylane import qchem
from tqdm import tqdm
from datetime import datetime
from collections import deque
from protocols import *

def init_worker(tqdm_lock, position_queue):
    """
    Initializes a multiprocessing worker.
    
    :param tqdm_lock: Multiprocessing lock for thread-safe progress updates.
    :param position_queue: Multiprocessing queue for managing progress bar positions.
    """

    global POSITION_QUEUE
    POSITION_QUEUE = position_queue
    tqdm.set_lock(tqdm_lock)

def generate_experiment_queue():
    """Generates a list of all experiment configurations to run."""

    checkpointed_experiments = []
    new_experiments = []

    existing_checkpoints = []
    if os.path.exists(config.CHECKPOINT_DIR):
        existing_checkpoints = [f.replace(".pkl", "") for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith(".pkl")]
    
    for molecule_name in config.EXPERIMENT_CONFIG["molecules"]:
        for depth in config.EXPERIMENT_CONFIG["depths"]:
            for protocol_config in config.EXPERIMENT_CONFIG["protocols"]:
                for seed in config.EXPERIMENT_CONFIG["seeds"]:
                    protocol_type = protocol_config["type"]
                    qubits = config.MOLECULES[molecule_name]["active_orbitals"] * 2
                    noise = "noisy" if config.USE_NOISE else "noiseless"

                    if protocol_type == "fixed":
                        k = protocol_config["k"]
                        if k >= qubits:
                            continue
                        exp_id = f"{molecule_name}_d{depth}_{protocol_type}_k{k}_{noise}_seed{seed}"
                    else:
                        exp_id = f"{molecule_name}_d{depth}_{protocol_type}_{noise}_seed{seed}"

                    experiment = {
                        "id": exp_id,
                        "molecule": molecule_name,
                        "depth": depth,
                        "protocol": protocol_config,
                        "seed": seed
                    }

                    if exp_id in existing_checkpoints:
                        checkpointed_experiments.append(experiment)
                    else:
                        new_experiments.append(experiment)

    random.shuffle(new_experiments)
    return checkpointed_experiments + new_experiments

def check_completed_experiments():
    """Returns a set of all experiment configurations already completed."""

    completed = set()
    json_path = os.path.join(config.OUTPUT_DIR, "metadata.json")
    
    if os.path.exists(json_path):
        with open(json_path, "r") as file:
            metadata = json.load(file)
            completed = set(metadata.keys())

    return completed

def load_checkpoint(exp_id):
    """
    Loads a checkpoint if it exists.
    
    :param exp_id: An experimental identification string.
    """

    path = os.path.join(config.CHECKPOINT_DIR, f"{exp_id}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        return None
    
def save_checkpoint(exp_id, checkpoint):
    """
    Saves a checkpoint to disk.
    
    :param exp_id: An experimental identification string.
    :param checkpoint: Dictionary containing checkpoint information.
    """

    with open(os.path.join(config.CHECKPOINT_DIR, f"{exp_id}.pkl"), "wb") as f:
        pickle.dump(checkpoint, f)

def delete_checkpoint(exp_id):
    """
    Deletes a checkpoint if it exists.
    
    :param exp_id: An experimental identification string.
    """

    path = os.path.join(config.CHECKPOINT_DIR, f"{exp_id}.pkl")
    if os.path.exists(path):
        os.remove(path)

def run_experiment(exp_config):
    """
    Runs a single experiment with the given configuration.
    
    :param exp_config: Dictionary containing experimental configuration.
    """

    # assigning progress bar position
    global POSITION_QUEUE
    position = POSITION_QUEUE.get()

    try:
        # extracting experiment configuration from dictionary
        exp_id = exp_config["id"]
        molecule_name = exp_config["molecule"]
        depth = exp_config["depth"]
        protocol_config = exp_config["protocol"]
        seed = exp_config["seed"]

        # getting molecular configuration
        molecule_config = config.MOLECULES[molecule_name]

        # building Hamiltonian and Hartree-Fock state
        hamiltonian, qubits = vqe_core.build_hamiltonian(molecule_config)
        hf_state = qchem.hf_state(molecule_config["active_electrons"], qubits)

        # setting up quantum device
        dev = qml.device("default.mixed" if config.USE_NOISE else "lightning.qubit", wires=qubits)

        # building ansatz and initializing parameters
        ansatz = vqe_core.build_ansatz(hf_state, qubits, config.NOISE_PARAMS if config.USE_NOISE else None)
        checkpoint = load_checkpoint(exp_id)
        theta = vqe_core.initialize_params(depth, qubits, seed) if checkpoint is None else None

        # selecting and initializing protocol
        protocol_type = protocol_config["type"]
        if protocol_type == "adaptive":
            protocol = AdaptiveProtocol(dev, hamiltonian, ansatz, depth, molecule_config["ground_state"], qubits)
        elif protocol_type == "fixed":
            k = protocol_config["k"]
            protocol = FixedKProtocol(dev, hamiltonian, ansatz, depth, molecule_config["ground_state"], qubits, k)
        elif protocol_type == "global":
            protocol = GlobalProtocol(dev, hamiltonian, ansatz, depth, molecule_config["ground_state"], qubits)

        # running experiment and displaying progress bars
        bar_format = "{desc:<50}{percentage:8.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        start_step = checkpoint["log"]["step"][-1] if checkpoint else 0

        with tqdm(total=config.MAX_STEPS, initial=start_step, desc=f"Worker {position:03d}: {exp_id}",
                  bar_format=bar_format, position=position, leave=False, dynamic_ncols=True) as pbar:
            log, final_avg_energy, checkpoint = protocol.run(theta, progress_cb=lambda n: pbar.update(n), checkpoint=checkpoint)
        
        # managing experiment data
        if checkpoint is not None:
            save_checkpoint(exp_id, checkpoint)
            return {"needs_restart": True, "exp_config": exp_config}
        
        result = {
            "exp_id": exp_id,
            "log": log,
            "metadata": {
                "molecule": molecule_name,
                "depth": depth,
                "protocol": protocol_config,
                "noisy": config.USE_NOISE,
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

        delete_checkpoint(exp_id)
        return result
    finally:
        POSITION_QUEUE.put(position)

def save_results(result, lock):
    """
    Saves experiment results to disk.
    
    :param result: Dictionary containing experimental results.
    :param lock: Multiprocessing lock for thread-safe file writing.
    """
    
    # extracting experiment ID and defining output directory
    exp_id = result["exp_id"]
    logs_dir = os.path.join(config.OUTPUT_DIR, "logs")

    # creating output directories, if needed
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # saving experiment log as .csv
    log_df = pd.DataFrame(result["log"])
    log_df.to_csv(os.path.join(logs_dir, f"{exp_id}.csv"), index=False)

    # saving experiment metadata as .json
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
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print(f"Running {len(remaining)} experiments on {config.NUM_CORES} cores...")

    # setting up locks and position manager for progress bars
    file_lock = multiprocessing.Lock()
    tqdm_lock = multiprocessing.RLock()

    manager = multiprocessing.Manager()
    position_queue = manager.Queue()
    for i in range(1, config.NUM_CORES + 1):
        position_queue.put(i)

    # creating worker pool with position queue
    with multiprocessing.Pool(config.NUM_CORES, init_worker, (tqdm_lock, position_queue), maxtasksperchild=1) as pool:
        with tqdm(total=len(remaining), desc="Overall Progress", position=0) as pbar:
            pending = deque(remaining)
            futures = []

            while pending or futures:
                while pending and len(futures) < config.NUM_CORES:
                    exp = pending.popleft()
                    future = pool.apply_async(run_experiment, (exp,))
                    futures.append(future)
                
                # processing checkpointed and completed experiments
                for i, future in enumerate(futures):
                    if future.ready():
                        result = future.get()
                        futures.pop(i)

                        if result.get("needs_restart"):
                            pending.appendleft(result["exp_config"])
                        else:
                            save_results(result, file_lock)
                            pbar.set_postfix({"Last": result["exp_id"]})
                            pbar.update(1)
                        break
                else:
                    time.sleep(0.1)

if __name__ == "__main__":
    main()