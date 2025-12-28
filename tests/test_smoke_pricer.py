import numpy as np
from desk_sim.instruments import AutocallableWorstOf
from desk_sim.market import MarketParams, make_time_grid
from desk_sim.pricer_mc import price_autocallable_mc


def test_pricer_runs_and_returns_reasonable_number():
    product = AutocallableWorstOf(
        maturity=1.0,
        obs_times=np.array([0.5, 1.0]),
        coupon_rate=0.06,
        autocall_barrier=1.0,
        protection_barrier=0.6,
        notional=100.0,
    )

    market = MarketParams(
        rate=0.0,
        vols=np.array([0.2, 0.2]),
        corr=np.array([[1.0, 0.5], [0.5, 1.0]])
    )

    grid = make_time_grid(maturity=1.0, steps_per_year=252)

    price = price_autocallable_mc(
        product=product,
        market=market,
        grid=grid,
        n_paths=10_000,
        rng=np.random.default_rng(123),
        return_diag=False
    )

    # very loose sanity bounds
    assert 0.0 < price < 200.0
