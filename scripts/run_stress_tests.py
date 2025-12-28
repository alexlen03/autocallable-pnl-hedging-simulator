import numpy as np

from desk_sim.instruments import AutocallableWorstOf
from desk_sim.market import MarketParams, make_time_grid
from desk_sim.pricer_mc import price_autocallable_mc
from desk_sim.scenarios import base_scenario, vol_up, vol_down, corr_breakdown


def main():
    product = AutocallableWorstOf(
        maturity=1.0,
        obs_times=np.array([0.25, 0.5, 0.75, 1.0]),
        coupon_rate=0.08,
        autocall_barrier=1.0,
        protection_barrier=0.6,
        notional=100.0,
    )

    base_market = MarketParams(
        rate=0.02,
        vols=np.array([0.25, 0.30]),
        corr=np.array([[1.0, 0.5], [0.5, 1.0]])
    )

    grid = make_time_grid(product.maturity, steps_per_year=252)

    scenarios = {
        "base": base_scenario(base_market),
        "vol_up": vol_up(base_market, 0.2),
        "vol_down": vol_down(base_market, 0.2),
        "corr_breakdown": corr_breakdown(base_market, 0.0),
    }

    print("Stress test pricing:")
    for name, mkt in scenarios.items():
        price = price_autocallable_mc(
            product, mkt, grid,
            n_paths=30_000,
            rng=np.random.default_rng(0)
        )
        print(f"{name:15s}: {price:8.4f}")


if __name__ == "__main__":
    main()
