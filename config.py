# defining VQE configuration parameters
GRAD_SHOTS = 1024
SHADOW_SHOTS = 4096
SEED = 42
DEPTHS = [4]
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
NOISE_PARAMS = {
    "depolarizing_1q": 0.001,
    "depolarizing_2q": 0.01,
    "amplitude": 0.0005,
    "phase": 0.001
}

# defining the molecules and their properties
MOLECULES = {
    "H2": {
        "symbols": ["H", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
        "electrons": 2,
        "ground_state": -1.13727,
        "active_electrons": None,
        "active_orbitals": None
    },
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
        "ground_state": -7.88263,
        "active_electrons": None,
        "active_orbitals": None
    }
}