# defining VQE configuration parameters
SHOTS_PER_STEP = 4096
SEED = 42
DEPTHS = [4]
INIT_SCALE = 0.01
CONVERGENCE_WINDOW = 5
THRESHOLD_SCALAR = 0.99

# defining optimizer parameters
MAX_STEPS = 1000
LEARNING_RATE = 0.4

# defining noise model parameters
NOISE_PARAMS = {
    "depolarizing_1q": 0.0001,
    "depolarizing_2q": 0.001,
    "amplitude": 0.00005,
    "phase": 0.0001
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