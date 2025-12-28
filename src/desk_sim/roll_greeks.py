import numpy as np
from desk_sim.roll_pricer import price_from_state_mc

def delta_from_state_fd(
    product,
    market,
    grid_remaining,
    level_now: np.ndarray,                  # (n_assets,)
    obs_indices_remaining: np.ndarray,
    n_paths: int,
    rel_bump: float = 0.01,
    rng_seed: int = 0
) -> np.ndarray:
    """
    Delta per asset at current state using bump-and-reprice with common random numbers.
    """
    n_assets = level_now.shape[0]
    base_rng = np.random.default_rng(rng_seed)
    base = price_from_state_mc(product, market, grid_remaining, level_now, obs_indices_remaining, n_paths, base_rng)

    deltas = np.empty(n_assets, dtype=float)
    for i in range(n_assets):
        bumped = level_now.copy()
        bumped[i] *= (1.0 + rel_bump)
        bumped_rng = np.random.default_rng(rng_seed)
        price_b = price_from_state_mc(product, market, grid_remaining, bumped, obs_indices_remaining, n_paths, bumped_rng)
        deltas[i] = (price_b - base) / (level_now[i] * rel_bump)
    return deltas
