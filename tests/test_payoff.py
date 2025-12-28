import numpy as np
import pytest
from desk_sim.instruments import AutocallableWorstOf, payoff_and_tau_from_levels


def test_autocall_first_observation():
    product = AutocallableWorstOf(
        maturity=1.0,
        obs_times=np.array([0.5, 1.0]),
        coupon_rate=0.06,
        autocall_barrier=1.0,
        protection_barrier=0.6,
        notional=100.0,
    )

    levels = np.ones((5, 2))  # always above barrier
    obs_indices = np.array([2, 4])

    payoff, tau = payoff_and_tau_from_levels(product, levels, obs_indices)

    assert tau == pytest.approx(0.5)
    assert payoff == pytest.approx(100.0 * (1.0 + 0.06 * 0.5))


def test_no_autocall_protected():
    product = AutocallableWorstOf(
        maturity=1.0,
        obs_times=np.array([1.0]),
        coupon_rate=0.06,
        autocall_barrier=1.0,
        protection_barrier=0.6,
        notional=100.0,
    )

    levels = np.array([
        [1.0, 1.0],
        [0.7, 0.8],
    ])
    obs_indices = np.array([1])

    payoff, tau = payoff_and_tau_from_levels(product, levels, obs_indices)

    assert tau == pytest.approx(1.0)
    assert payoff == pytest.approx(100.0)


def test_no_autocall_unprotected():
    product = AutocallableWorstOf(
        maturity=1.0,
        obs_times=np.array([1.0]),
        coupon_rate=0.06,
        autocall_barrier=1.0,
        protection_barrier=0.6,
        notional=100.0,
    )

    levels = np.array([
        [1.0, 1.0],
        [0.4, 0.9],
    ])
    obs_indices = np.array([1])

    payoff, tau = payoff_and_tau_from_levels(product, levels, obs_indices)

    assert tau == pytest.approx(1.0)
    assert payoff == pytest.approx(40.0)
