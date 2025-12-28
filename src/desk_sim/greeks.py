import numpy as np

from desk_sim.instruments import AutocallableWorstOf
from desk_sim.market import MarketParams, TimeGrid
from desk_sim.pricer_mc import price_autocallable_mc


def _bump_spot0(spot0: np.ndarray, asset_idx: int, rel_bump: float) -> np.ndarray:
    spot0 = np.asarray(spot0, dtype=float).copy()
    spot0[asset_idx] *= (1.0 + rel_bump)
    return spot0


def _bump_vols(vols: np.ndarray, asset_idx: int, abs_bump: float) -> np.ndarray:
    vols = np.asarray(vols, dtype=float).copy()
    vols[asset_idx] += abs_bump
    if vols[asset_idx] <= 0:
        raise ValueError("vol bump produced non-positive vol")
    return vols


def delta_fd(
    product: AutocallableWorstOf,
    market: MarketParams,
    grid: TimeGrid,
    n_paths: int,
    spot0: np.ndarray,
    rel_bump: float = 0.01,
    rng_seed: int = 0
) -> np.ndarray:
    """
    Finite-difference delta per asset using bump-and-reprice with common random numbers.

    Note: In the current V1 implementation, dynamics simulate *normalised levels* and do not
    explicitly use spot0. To keep the interface desk-like, we interpret delta here as sensitivity
    to the initial level (normalised), i.e. bumping spot0 is equivalent to bumping initial level.

    Returns:
        deltas: shape (n_assets,)
    """
    spot0 = np.asarray(spot0, dtype=float)
    n_assets = spot0.shape[0]
    deltas = np.empty(n_assets, dtype=float)

    # Base price (use same seed)
    base_rng = np.random.default_rng(rng_seed)
    base_price = price_autocallable_mc(product, market, grid, n_paths, rng=base_rng, return_diag=False)

    for i in range(n_assets):
        # In a normalised-level simulator, "spot0 bump" should ideally feed into dynamics.
        # For V1, we approximate by bumping notional scaling of levels is not available, so
        # we implement a simple proxy: bump coupon/notional? (Not desired.)
        #
        # Better: incorporate spot0 in dynamics later. For now, we compute delta w.r.t. level(0)=1:
        # approximate by bumping autocall/protection barriers in the opposite direction is also not desired.
        #
        # So: we do the correct desk interface now, and in TP6/TP7 we will extend dynamics to accept spot0.
        #
        # Practical workaround now: treat "spot0 bump" as bumping initial normalised level:
        # we can model that by scaling the entire path for that asset after simulation. We'll do that in pricer.
        #
        # -> implement delta by calling a helper pricer that scales one asset paths.
        bumped_price = _price_with_asset_scaling(
            product, market, grid, n_paths, asset_idx=i, scale=(1.0 + rel_bump), rng_seed=rng_seed
        )

        deltas[i] = (bumped_price - base_price) / (spot0[i] * rel_bump)

    return deltas


def vega_fd(
    product: AutocallableWorstOf,
    market: MarketParams,
    grid: TimeGrid,
    n_paths: int,
    abs_bump: float = 0.01,
    rng_seed: int = 0
) -> np.ndarray:
    """
    Finite-difference vega per asset: dPrice/dVol_i (vol bump in absolute terms, e.g. 0.01 = +1 vol point)
    Uses common random numbers.
    """
    n_assets = market.vols.shape[0]
    vegas = np.empty(n_assets, dtype=float)

    base_rng = np.random.default_rng(rng_seed)
    base_price = price_autocallable_mc(product, market, grid, n_paths, rng=base_rng, return_diag=False)

    for i in range(n_assets):
        bumped_market = MarketParams(
            rate=market.rate,
            vols=_bump_vols(market.vols, i, abs_bump),
            corr=market.corr
        )
        bumped_rng = np.random.default_rng(rng_seed)
        bumped_price = price_autocallable_mc(product, bumped_market, grid, n_paths, rng=bumped_rng, return_diag=False)

        vegas[i] = (bumped_price - base_price) / abs_bump

    return vegas


def _price_with_asset_scaling(
    product: AutocallableWorstOf,
    market: MarketParams,
    grid: TimeGrid,
    n_paths: int,
    asset_idx: int,
    scale: float,
    rng_seed: int
) -> float:
    """
    Helper: price the product where one asset path is scaled by a constant factor.
    This emulates a spot0 bump in a normalised-level simulator.
    Uses common random numbers via rng_seed.
    """
    # Import locally to avoid circular import issues if you refactor later
    from desk_sim.dynamics import simulate_bs_normalised_levels
    from desk_sim.market import obs_times_to_indices
    from desk_sim.instruments import payoff_and_tau_from_levels

    rng = np.random.default_rng(rng_seed)
    paths = simulate_bs_normalised_levels(grid=grid, market=market, n_paths=n_paths, rng=rng)
    paths[:, :, asset_idx] *= scale

    obs_idx = obs_times_to_indices(grid, product.obs_times)
    r = float(market.rate)

    disc = np.empty(n_paths, dtype=float)
    for p in range(n_paths):
        payoff, tau = payoff_and_tau_from_levels(product, paths[p], obs_idx)
        disc[p] = np.exp(-r * tau) * payoff

    return float(np.mean(disc))
