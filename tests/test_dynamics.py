import numpy as np
from desk_sim.market import MarketParams, make_time_grid
from desk_sim.dynamics import simulate_bs_normalised_levels


def test_shapes_and_initial_level():
    market = MarketParams(
        rate=0.02,
        vols=np.array([0.2, 0.25]),
        corr=np.array([[1.0, 0.5], [0.5, 1.0]])
    )
    grid = make_time_grid(maturity=1.0, steps_per_year=252)
    paths = simulate_bs_normalised_levels(grid, market, n_paths=1000, rng=np.random.default_rng(0))

    assert paths.shape == (1000, 253, 2)
    assert np.allclose(paths[:, 0, :], 1.0)


def test_paths_positive():
    market = MarketParams(
        rate=0.0,
        vols=np.array([0.4, 0.4]),
        corr=np.array([[1.0, 0.2], [0.2, 1.0]])
    )
    grid = make_time_grid(maturity=0.5, steps_per_year=252)
    paths = simulate_bs_normalised_levels(grid, market, n_paths=2000, rng=np.random.default_rng(1))
    assert np.all(paths > 0.0)


def test_empirical_correlation_roughly_matches():
    # rough check on 1-step log-returns
    market = MarketParams(
        rate=0.0,
        vols=np.array([0.3, 0.3]),
        corr=np.array([[1.0, 0.7], [0.7, 1.0]])
    )
    grid = make_time_grid(maturity=1/252, steps_per_year=252)  # 2 points: t0, t1
    paths = simulate_bs_normalised_levels(grid, market, n_paths=50_000, rng=np.random.default_rng(2))

    logret = np.log(paths[:, 1, :] / paths[:, 0, :])  # (n_paths, 2)
    emp_corr = np.corrcoef(logret.T)[0, 1]

    assert emp_corr > 0.6  # loose bound
