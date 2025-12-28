"""
Autocallable worst-of product definition and payoff logic.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AutocallableWorstOf:
    maturity: float                 # years
    obs_times: np.ndarray           # shape (n_obs,), in years
    coupon_rate: float              # annualised coupon (simple)
    autocall_barrier: float         # e.g. 1.0
    protection_barrier: float       # e.g. 0.6
    notional: float = 100.0

    def __post_init__(self):

        if self.maturity <= 0:
            raise ValueError("maturity must be positive")    
        if np.any(self.obs_times <= 0) or np.any(self.obs_times > self.maturity):
            raise ValueError("obs_times must be in (0, maturity]")    
        if not np.all(np.diff(self.obs_times) > 0):
            raise ValueError("obs_times must be strictly increasing")
        if self.autocall_barrier <= 0:
            raise ValueError("autocall_barrier must be positive")
        if self.protection_barrier <= 0:
            raise ValueError("protection_barrier must be positive")
        if self.protection_barrier > self.autocall_barrier:
            raise ValueError("protection_barrier should not exceed autocall_barrier")
        if self.notional <= 0:
            raise ValueError("notional must be positive")

def payoff_and_tau_from_levels(
    product: AutocallableWorstOf,
    levels: np.ndarray,          # shape (n_steps, n_assets): this is the trajectory 
    obs_indices: np.ndarray      # indices of observation dates
) -> tuple[float, float]:
    
    """
    Returns:
        payoff: cash amount paid
        tau: redemption time in years
    """

    n_steps, n_assets = levels.shape
    if n_assets < 2:
        raise ValueError("Worst-of autocallable requires at least 2 assets")

    # Early redemption (autocall)
    for k, idx in enumerate(obs_indices):
        worst = float(np.min(levels[idx, :]))
        if worst >= product.autocall_barrier:
            tau = float(product.obs_times[k])
            payoff = product.notional * (1.0 + product.coupon_rate * tau)
            return payoff, tau

    # No autocall: payoff at maturity
    worst_T = float(np.min(levels[-1, :]))
    tau = product.maturity

    if worst_T >= product.protection_barrier:
        payoff = product.notional
    else:
        payoff = product.notional * worst_T

    return payoff, tau
