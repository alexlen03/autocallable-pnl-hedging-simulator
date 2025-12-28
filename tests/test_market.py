import numpy as np
from desk_sim.market import MarketParams, make_time_grid, obs_times_to_indices


def test_make_time_grid():
    grid = make_time_grid(maturity=1.0, steps_per_year=252)
    assert grid.times[0] == 0.0
    assert abs(grid.times[-1] - 1.0) < 1e-12
    assert len(grid.times) == 253


def test_obs_times_to_indices_quarterly():
    grid = make_time_grid(maturity=1.0, steps_per_year=252)
    obs_times = np.array([0.25, 0.5, 0.75, 1.0])
    idx = obs_times_to_indices(grid, obs_times)

    # Should be close to 63, 126, 189, 252
    assert idx[0] in range(62, 65)
    assert idx[1] in range(125, 128)
    assert idx[2] in range(188, 191)
    assert idx[3] == 252


def test_market_params_shapes():
    market = MarketParams(
        rate=0.02,
        vols=np.array([0.2, 0.25]),
        corr=np.array([[1.0, 0.5], [0.5, 1.0]])
    )
    assert market.vols.shape == (2,)
    assert market.corr.shape == (2, 2)
