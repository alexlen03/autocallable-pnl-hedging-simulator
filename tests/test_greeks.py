import numpy as np
from desk_sim.instruments import AutocallableWorstOf
from desk_sim.market import MarketParams, make_time_grid
from desk_sim.greeks import vega_fd, delta_fd


def test_greeks_run_and_shapes():
    product = AutocallableWorstOf(
        maturity=1.0,
        obs_times=np.array([0.5, 1.0]),
        coupon_rate=0.06,
        autocall_barrier=1.0,
        protection_barrier=0.6,
        notional=100.0,
    )
    market = MarketParams(
        rate=0.01,
        vols=np.array([0.2, 0.25]),
        corr=np.array([[1.0, 0.4], [0.4, 1.0]])
    )
    grid = make_time_grid(maturity=1.0, steps_per_year=252)

    spot0 = np.array([100.0, 100.0])

    deltas = delta_fd(product, market, grid, n_paths=5000, spot0=spot0, rel_bump=0.01, rng_seed=1)
    vegas = vega_fd(product, market, grid, n_paths=5000, abs_bump=0.01, rng_seed=1)

    assert deltas.shape == (2,)
    assert vegas.shape == (2,)
    assert np.all(np.isfinite(deltas))
    assert np.all(np.isfinite(vegas))
