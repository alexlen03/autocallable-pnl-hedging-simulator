#vol; cor; rates

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MarketParams:
    rate: float                  # risk-free rate r
    vols: np.ndarray             # shape (n_assets,)
    corr: np.ndarray             # shape (n_assets, n_assets)

    def __post_init__(self):
        if self.vols.ndim != 1:
            raise ValueError("vols must be a 1D array (n_assets,)")
        n = self.vols.shape[0]
        if self.corr.shape != (n, n):
            raise ValueError("corr must have shape (n_assets, n_assets)")
        if not np.allclose(np.diag(self.corr), 1.0):
            raise ValueError("corr must have 1.0 on the diagonal")
        # light sanity check: symmetric
        if not np.allclose(self.corr, self.corr.T):
            raise ValueError("corr must be symmetric")


@dataclass(frozen=True)
class TimeGrid:
    times: np.ndarray            # shape (n_steps,)
    dt: float                    # time step in years

    def __post_init__(self):
        if self.times.ndim != 1:
            raise ValueError("times must be 1D")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.times[0] != 0.0:
            raise ValueError("times must start at 0.0")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("times must be strictly increasing")

def make_time_grid(maturity: float, steps_per_year: int = 252) -> TimeGrid:
    """
    Builds a uniform grid from 0 to maturity inclusive.
    """
    if maturity <= 0:
        raise ValueError("maturity must be > 0")
    if steps_per_year <= 0:
        raise ValueError("steps_per_year must be > 0")

    dt = 1.0 / float(steps_per_year)
    n_steps = int(round(maturity * steps_per_year)) + 1
    times = np.linspace(0.0, maturity, n_steps)
    return TimeGrid(times=times, dt=dt)

def obs_times_to_indices(grid: TimeGrid, obs_times: np.ndarray) -> np.ndarray:
    """
    Map observation times (in years) to indices in the time grid.

    Uses nearest grid point, with a small tolerance.
    Returns an array of indices of shape (n_obs,).
    """
    obs_times = np.asarray(obs_times, dtype=float)
    if obs_times.ndim != 1:
        raise ValueError("obs_times must be 1D")
    if np.any(obs_times < 0) or np.any(obs_times > grid.times[-1]):
        raise ValueError("obs_times must be within [0, maturity]")
    if not np.all(np.diff(obs_times) > 0):
        raise ValueError("obs_times must be strictly increasing")

    # nearest index for each obs_time
    idx = np.searchsorted(grid.times, obs_times, side="left")

    idx = np.clip(idx, 0, len(grid.times) - 1)

    # choose nearest between idx and idx-1
    idx_minus = np.clip(idx - 1, 0, len(grid.times) - 1)
    choose_minus = np.abs(grid.times[idx_minus] - obs_times) < np.abs(grid.times[idx] - obs_times)
    idx = np.where(choose_minus, idx_minus, idx)

    # ensure mapping is strictly increasing
    if not np.all(np.diff(idx) > 0):
        raise ValueError("mapped obs_indices are not strictly increasing; adjust grid or obs_times")

    return idx.astype(int)
