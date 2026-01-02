# defining experiment configuration
EXPERIMENT_CONFIG = {
    "molecules": ["LiH_R8", "LiH"],
    "protocols": [
        {"type": "adaptive"},
        {"type": "fixed", "k": 2},
        {"type": "fixed", "k": 4},
        {"type": "fixed", "k": 6},
        {"type": "global"}
    ],
    "seeds": [42, 67, 123, 654, 2025, 8900, 42069, 97531, 101010, 123125],
    "depths": [2, 4, 6, 8]
}
OUTPUT_DIR = "results"
NUM_CORES = 10

# defining VQE configuration parameters
GRAD_SHOTS = 1024
SHADOW_SHOTS = 4096
INIT_SCALE = 0.01
CONVERGENCE_WINDOW = 10
ENERGY_THRESHOLD = 1.01

# defining optimizer parameters
MAX_STEPS = 200
LEARNING_RATE = 0.1

# defining adaptive protocol parameters
HYSTERESIS = 2
ESCALATION_THRESHOLDS = {"gradient_snr": 0.9, "avg_entropy": 0.6}
DEESCALATION_THRESHOLDS = {"gradient_snr": 0.3, "avg_entropy": 1.2}

# defining noise model parameters
USE_NOISE = False
NOISE_PARAMS = {
    "depolarizing_1q": 0.001,
    "depolarizing_2q": 0.01,
    "amplitude": 0.0005,
    "phase": 0.001
}

# defining the molecules and their properties
MOLECULES = {
    "LiH_R8": {
        "symbols": ["Li", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5949]],
        "electrons": 4,
        "ground_state": -7.86384,
        "active_electrons": 2,
        "active_orbitals": 4
    },
    "LiH": {
        "symbols": ["Li", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5949]],
        "electrons": 4,
        "ground_state": -7.88240,
        "active_electrons": None,
        "active_orbitals": None
    }
}