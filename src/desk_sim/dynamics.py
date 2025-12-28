import numpy as np
from desk_sim.market import MarketParams, TimeGrid


def simulate_bs_normalised_levels(
    grid: TimeGrid,
    market: MarketParams,
    n_paths: int,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Simulate correlated Black–Scholes *normalised* levels:
        level_i(t) = S_i(t) / S_i(0)

    Returns:
        paths: shape (n_paths, n_steps, n_assets)
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")
    if rng is None:
        rng = np.random.default_rng()

    n_steps = grid.times.shape[0]
    n_assets = market.vols.shape[0]

    r = float(market.rate)
    vols = market.vols.astype(float)

    # Cholesky for correlation
    L = np.linalg.cholesky(market.corr)

    dt = float(grid.dt)
    sqrt_dt = np.sqrt(dt)

    # paths
    paths = np.empty((n_paths, n_steps, n_assets), dtype=float)
    paths[:, 0, :] = 1.0  # normalised start

    drift = (r - 0.5 * vols**2) * dt  # shape (n_assets,)

    for t in range(1, n_steps):
        # Z: (n_paths, n_assets) iid standard normals
        Z = rng.standard_normal(size=(n_paths, n_assets))
        # correlated increments
        dW = Z @ L.T  # (n_paths, n_assets)

        incr = drift + vols * sqrt_dt * dW  # log-increment
        paths[:, t, :] = paths[:, t - 1, :] * np.exp(incr)

    return paths

