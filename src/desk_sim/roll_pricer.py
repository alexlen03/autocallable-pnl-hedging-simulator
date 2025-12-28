import numpy as np
from desk_sim.instruments import AutocallableWorstOf, payoff_and_tau_from_levels
from desk_sim.market import MarketParams, TimeGrid
from desk_sim.dynamics import simulate_bs_normalised_levels

def price_from_state_mc(
    product: AutocallableWorstOf,
    market: MarketParams,
    grid_remaining: TimeGrid,
    level_now: np.ndarray,              # shape (n_assets,), normalised current level vs S0
    obs_indices_remaining: np.ndarray,  # indices in remaining grid
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Price at 'now' given current normalised levels, by simulating future *relative* moves.
    """
    if rng is None:
        rng = np.random.default_rng()

    # simulate future relative paths starting at 1
    paths_rel = simulate_bs_normalised_levels(grid_remaining, market, n_paths, rng=rng)
    # scale by current level
    paths = paths_rel * level_now[None, None, :]

    r = float(market.rate)
    disc = np.empty(n_paths, dtype=float)
    for p in range(n_paths):
        payoff, tau = payoff_and_tau_from_levels(product, paths[p], obs_indices_remaining)
        disc[p] = np.exp(-r * tau) * payoff

    return float(np.mean(disc))
