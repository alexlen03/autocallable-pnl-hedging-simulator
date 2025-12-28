import os
import numpy as np
import matplotlib.pyplot as plt

from desk_sim.instruments import AutocallableWorstOf
from desk_sim.market import MarketParams, make_time_grid
from desk_sim.hedge_sim import run_delta_hedge_one_path


def main():
    print("Running desk simulation...")

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

    grid = make_time_grid(product.maturity, steps_per_year=252)

    df = run_delta_hedge_one_path(
        product=product,
        market=market,
        full_grid=grid,
        n_paths_pricing=5000,   # augmente plus tard si tu veux
        rel_bump=0.01,
        rng_seed_path=42,
        rng_seed_pricer=0
    )

    df = df.dropna().copy()
    df["cum_pnl"] = df["pnl_total"].cumsum()

    os.makedirs("reports/figures", exist_ok=True)

    # cumulative PnL
    plt.figure()
    plt.plot(df["time"], df["cum_pnl"])
    plt.xlabel("time (years)")
    plt.ylabel("cumulative PnL")
    plt.title("Delta-hedged PnL (1 path)")
    plt.savefig("reports/figures/pnl_total.png", dpi=200)
    plt.close()

    # hedge error histogram
    plt.figure()
    plt.hist(df["pnl_total"], bins=40)
    plt.xlabel("daily total PnL")
    plt.ylabel("count")
    plt.title("Hedge error distribution")
    plt.savefig("reports/figures/hedge_error_hist.png", dpi=200)
    plt.close()

    print(df[["time", "V_product", "pnl_total"]].tail())
    print("Saved figures to reports/figures/")


if __name__ == "__main__":
    main()
