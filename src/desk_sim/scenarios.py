import numpy as np
from desk_sim.market import MarketParams


def base_scenario(market: MarketParams) -> MarketParams:
    return market


def vol_up(market: MarketParams, bump: float = 0.1) -> MarketParams:
    return MarketParams(
        rate=market.rate,
        vols=market.vols * (1.0 + bump),
        corr=market.corr,
    )


def vol_down(market: MarketParams, bump: float = 0.1) -> MarketParams:
    return MarketParams(
        rate=market.rate,
        vols=market.vols * (1.0 - bump),
        corr=market.corr,
    )


def corr_breakdown(market: MarketParams, target_corr: float = 0.0) -> MarketParams:
    n = market.vols.shape[0]
    corr = np.full((n, n), target_corr)
    np.fill_diagonal(corr, 1.0)
    return MarketParams(
        rate=market.rate,
        vols=market.vols,
        corr=corr,
    )
