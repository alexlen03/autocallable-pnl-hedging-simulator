import numpy as np

from desk_sim.instruments import AutocallableWorstOf
from desk_sim.market import MarketParams, make_time_grid
from desk_sim.pricer_mc import price_autocallable_mc
from desk_sim.greeks import delta_fd, vega_fd


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

    spot0 = np.array([100.0, 100.0])

    deltas = delta_fd(product, market, grid, n_paths=30_000, spot0=spot0, rel_bump=0.01, rng_seed=0)
    vegas = vega_fd(product, market, grid, n_paths=30_000, abs_bump=0.01, rng_seed=0)

    print("Deltas:", deltas)
    print("Vegas:", vegas)

if __name__ == "__main__":
    main()
