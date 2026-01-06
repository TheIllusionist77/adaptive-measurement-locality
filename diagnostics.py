# importing necessary libraries
import config
import jax.numpy as jnp

def compute_diagnostics(grad, curr_train, curr_full, prev_train, prev_full, align_ema):
    """
    Computes various diagnostics for the optimization process.
    
    :param grad: The current step's gradient array.
    :param curr_train: The current step's training energy.
    :param curr_full: The current step's full energy.
    :param prev_train: The previous step's training energy.
    :param prev_full: The previous step's full energy.
    :param align_ema: The previous step's gradient alignment value.
    """

    grad_var = float(jnp.var(grad))
    delta_full = 0.0

    if prev_train is not None and prev_full is not None:
        delta_train = curr_train - prev_train
        delta_full = curr_full - prev_full
        
        if abs(delta_train) >= 1e-4:
            grad_align = float(delta_full / delta_train)
            align_ema = (1 - config.EMA_ALPHA) * align_ema + config.EMA_ALPHA * grad_align
        else:
            align_ema *= 1 - config.EMA_ALPHA

    diagnostics = {
        "grad_var": grad_var,
        "grad_align": align_ema,
        "improvement": delta_full
    }

    return diagnostics

def get_diagnostics(data):
    """
    Formats diagnostics information for logging.
    
    :param data: The dictionary containing diagnostics data.
    """

    return f"   Grad Var = {data["grad_var"]:.6f}, Grad Align = {data["grad_align"]:.2f}, Improvement = {data["improvement"]:.4f} Ha"