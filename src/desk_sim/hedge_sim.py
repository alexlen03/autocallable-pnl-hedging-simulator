import numpy as np
import pandas as pd

from desk_sim.market import make_remaining_grid, remaining_obs_times, obs_times_to_indices
from desk_sim.roll_pricer import price_from_state_mc
from desk_sim.roll_greeks import delta_from_state_fd
from desk_sim.dynamics import simulate_bs_normalised_levels

def run_delta_hedge_one_path(
    product,
    market,
    full_grid,
    n_paths_pricing: int = 20000,
    rel_bump: float = 0.01,
    rng_seed_path: int = 123,
    rng_seed_pricer: int = 0
) -> pd.DataFrame:
    """
    Simulate one realised path, reprice daily, compute delta, hedge, and compute PnL.
    Returns a DataFrame with time series.
    """
    rng_path = np.random.default_rng(rng_seed_path)
    realised = simulate_bs_normalised_levels(full_grid, market, n_paths=1, rng=rng_path)[0]
    # realised shape: (n_steps, n_assets)

    n_steps, n_assets = realised.shape
    times = full_grid.times

    # Hedge state: q in underlyings, cash B
    q = np.zeros(n_assets)
    B = 0.0

    rows = []

    # Initial valuation and hedge
    for t_idx in range(n_steps - 1):
        now_time = float(times[t_idx])
        level_now = realised[t_idx, :].copy()

        # build remaining product + grid
        rem_grid = make_remaining_grid(full_grid, t_idx)
        rem_obs_times = remaining_obs_times(product.obs_times, now_time)

        # if no remaining obs times, keep maturity only (your payoff already handles maturity)
        if rem_obs_times.size == 0:
            obs_idx = np.array([len(rem_grid.times) - 1], dtype=int)
            product_rem = product  # maturity in payoff handled by last point
        else:
            # product for remaining horizon: shift maturity
            product_rem = type(product)(
                maturity=float(product.maturity - now_time),
                obs_times=rem_obs_times,
                coupon_rate=product.coupon_rate,
                autocall_barrier=product.autocall_barrier,
                protection_barrier=product.protection_barrier,
                notional=product.notional,
            )
            obs_idx = obs_times_to_indices(rem_grid, product_rem.obs_times)

        # price and delta at current state
        V = price_from_state_mc(
            product_rem, market, rem_grid, level_now, obs_idx,
            n_paths=n_paths_pricing,
            rng=np.random.default_rng(rng_seed_pricer + t_idx),
        )

        delta = delta_from_state_fd(
            product_rem, market, rem_grid, level_now, obs_idx,
            n_paths=n_paths_pricing,
            rel_bump=rel_bump,
            rng_seed=rng_seed_pricer + t_idx,
        )

        # Underlying "prices" for hedge: use normalised levels as proxy prices
        S = level_now

        # Re-hedge: target q = -delta (bank short product → hedge with -delta; adjust sign to your convention)
        q_target = -delta
        dq = q_target - q

        # self-financing: buy/sell underlying -> adjust cash
        # cost = dq · S
        B -= float(np.dot(dq, S))
        q = q_target

        # Move one step forward to compute hedge PnL increment
        S_next = realised[t_idx + 1, :]
        dt = float(times[t_idx + 1] - times[t_idx])

        # accrue cash at risk-free rate (simple continuous accrual)
        B *= float(np.exp(market.rate * dt))

        hedge_value = float(np.dot(q, S) + B)
        hedge_value_next = float(np.dot(q, S_next) + B)  # after spot move, same q, same B after accrual

        rows.append({
            "t_idx": t_idx,
            "time": now_time,
            "V_product": V,
            "delta_0": float(delta[0]),
            "delta_1": float(delta[1]) if n_assets > 1 else np.nan,
            "q0": float(q[0]),
            "q1": float(q[1]) if n_assets > 1 else np.nan,
            "cash": float(B),
            "S0": float(S[0]),
            "S1": float(S[1]) if n_assets > 1 else np.nan,
            "hedge_value": hedge_value,
            "hedge_value_next": hedge_value_next,
        })

    df = pd.DataFrame(rows)

    # Simple PnL proxy: changes in (product value + hedge value)
    # Align next step product value by shifting
    df["V_product_next"] = df["V_product"].shift(-1)
    df["pnl_product"] = df["V_product_next"] - df["V_product"]
    df["pnl_hedge"] = df["hedge_value_next"] - df["hedge_value"]
    df["pnl_total"] = df["pnl_product"] + df["pnl_hedge"]

    return df
