import numpy as np

from desk_sim.instruments import AutocallableWorstOf, payoff_and_tau_from_levels
from desk_sim.market import MarketParams, TimeGrid, obs_times_to_indices
from desk_sim.dynamics import simulate_bs_normalised_levels


def price_autocallable_mc(
    product: AutocallableWorstOf,
    market: MarketParams,
    grid: TimeGrid,
    n_paths: int,
    rng: np.random.Generator | None = None,
    return_diag: bool = False
):
    """
    Monte Carlo price of a worst-of autocallable:
        Price = E[ exp(-r*tau) * Payoff(tau) ]

    Returns:
        price (float) or (price, diagnostics dict) if return_diag=True
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")
    if rng is None:
        rng = np.random.default_rng()

    # 1) simulate normalised levels
    paths = simulate_bs_normalised_levels(grid=grid, market=market, n_paths=n_paths, rng=rng)
    # paths shape: (n_paths, n_steps, n_assets)

    # 2) map observation times to indices
    obs_idx = obs_times_to_indices(grid, product.obs_times)

    # 3) compute discounted payoffs path-by-path
    r = float(market.rate)
    disc_payoffs = np.empty(n_paths, dtype=float)

    call_count = 0
    taus = np.empty(n_paths, dtype=float)

    for p in range(n_paths):
        payoff, tau = payoff_and_tau_from_levels(product, paths[p], obs_idx)
        taus[p] = tau
        disc_payoffs[p] = np.exp(-r * tau) * payoff

        # autocall if tau < maturity (by construction in TP1)
        if tau < product.maturity - 1e-15:
            call_count += 1

    price = float(np.mean(disc_payoffs))

    if not return_diag:
        return price

    diagnostics = {
        "n_paths": n_paths,
        "call_probability": call_count / n_paths,
        "avg_tau": float(np.mean(taus)),
        "avg_discounted_payoff": float(np.mean(disc_payoffs)),
        "std_discounted_payoff": float(np.std(disc_payoffs, ddof=1)),
    }
    return price, diagnostics
