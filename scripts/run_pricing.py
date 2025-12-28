import numpy as np

from desk_sim.instruments import AutocallableWorstOf
from desk_sim.market import MarketParams, make_time_grid
from desk_sim.pricer_mc import price_autocallable_mc

print("RUN PRICING STARTED")


def main():
    product = AutocallableWorstOf(
        maturity=1.0,
        obs_times=np.array([0.25, 0.5, 0.75, 1.0]),
        coupon_rate=0.08,
        autocall_barrier=1.0,
        protection_barrier=0.6,
        notional=100.0,
    )

    market = MarketParams(
        rate=0.02,
        vols=np.array([0.25, 0.30]),
        corr=np.array([[1.0, 0.5], [0.5, 1.0]])
    )

    grid = make_time_grid(maturity=product.maturity, steps_per_year=252)

    price, diag = price_autocallable_mc(
        product=product,
        market=market,
        grid=grid,
        n_paths=50_000,
        rng=np.random.default_rng(0),
        return_diag=True
    )

    print(f"Price: {price:.4f}")
    for k, v in diag.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
