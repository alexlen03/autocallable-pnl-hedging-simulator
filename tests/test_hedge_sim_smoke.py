import numpy as np
from desk_sim.instruments import AutocallableWorstOf
from desk_sim.market import MarketParams, make_time_grid
from desk_sim.hedge_sim import run_delta_hedge_one_path

def test_hedge_sim_runs():
    product = AutocallableWorstOf(
        maturity=0.25,
        obs_times=np.array([0.25]),
        coupon_rate=0.05,
        autocall_barrier=1.0,
        protection_barrier=0.6,
        notional=100.0,
    )
    market = MarketParams(
        rate=0.0,
        vols=np.array([0.2, 0.2]),
        corr=np.array([[1.0, 0.2],[0.2, 1.0]])
    )
    grid = make_time_grid(product.maturity, steps_per_year=252)

    df = run_delta_hedge_one_path(
        product, market, grid,
        n_paths_pricing=1000,
        rel_bump=0.01,
        rng_seed_path=1,
        rng_seed_pricer=2
    )
    assert len(df) > 0
    assert "pnl_total" in df.columns
