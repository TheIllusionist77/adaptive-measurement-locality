# importing necessary libraries
import jax
import jax.numpy as jnp

# enabling 64-bit precision for jax
jax.config.update("jax_enable_x64", True)

# defining experiment configuration
EXPERIMENT_CONFIG = {
    "molecules": ["LiH_R8", "BeH2_R8"],
    "protocols": [
        {"type": "adaptive"},
        {"type": "fixed", "k": 1},
        {"type": "fixed", "k": 2},
        {"type": "fixed", "k": 3},
        {"type": "fixed", "k": 4},
        {"type": "fixed", "k": 5},
        {"type": "fixed", "k": 6},
        {"type": "fixed", "k": 7},
        {"type": "global"}
    ],
    "seeds": [42],
    "depths": [1, 2, 3, 4, 5, 6, 7, 8]
}
OUTPUT_DIR = "results"
NUM_CORES = 10

# defining VQE configuration parameters
GRAD_SHOTS = 1024
INIT_SCALE = jnp.pi * 2
CONVERGENCE_WINDOW = 10
ENERGY_THRESHOLD = 0.008
IMPROVEMENT_STEPS = 5

# defining optimizer parameters
MAX_STEPS = 500
LEARNING_RATE = 0.1

# defining adaptive protocol parameters
HYSTERESIS = 3
EMA_ALPHA = 0.5
VAR_THRESHOLD = 5e-3
ALIGN_THRESHOLD = 0.5
IMPROVEMENT_THRESHOLD = -0.2

# defining noise model parameters
USE_NOISE = False
NOISE_PARAMS = {
    "dep_1q": 0.001,
    "dep_2q": 0.01,
}

# defining the molecules and their properties
MOLECULES = {
    "LiH_R8": {
        "symbols": ["Li", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5949]],
        "electrons": 4,
        "ground_state": -7.8620,
        "active_electrons": 2,
        "active_orbitals": 4
    },
    "BeH2_R8": {
        "symbols": ["Be", "H", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3264], [0.0, 0.0, -1.3264]],
        "electrons": 6,
        "ground_state": -15.5603,
        "active_electrons": 4,
        "active_orbitals": 4
    },
    "LiH": {
        "symbols": ["Li", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5949]],
        "electrons": 4,
        "ground_state": -7.8620,
        "active_electrons": 4,
        "active_orbitals": 6
    },
    "BeH2_R12": {
        "symbols": ["Be", "H", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3264], [0.0, 0.0, -1.3264]],
        "electrons": 6,
        "ground_state": -15.5603,
        "active_electrons": 6,
        "active_orbitals": 6
    }
}