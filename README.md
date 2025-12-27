# autocallable-pnl-hedging-

# Autocallable PnL & Hedging Desk Simulator

## Objective
This project implements a **desk-style simulator** for pricing, hedging, and managing the PnL of an **equity worst-of autocallable**.

The goal is not to build a sophisticated academic model, but to replicate the **workflow of an equity derivatives trading desk**:
- Monte Carlo pricing of a path-dependent product
- Greeks estimation via bump-and-reprice
- Discrete delta hedging
- Daily PnL attribution
- Stress analysis on volatility and correlation

---

## Product Description
The product is a **worst-of autocallable** on two equities.

### Key features
- Two underlyings (worst-of)
- Fixed notional (default: 100)
- Discrete observation dates
- Early redemption (autocall) if the worst-of breaches a high barrier
- Capital protection at maturity above a low barrier
- Linear capital loss below the protection barrier

### Payoff logic
At each observation date:
- Compute the **worst-of performance**
- If worst-of ≥ autocall barrier:
  - Product redeems early
  - Pays notional plus coupon
  - Simulation stops

If no autocall occurs:
- At maturity:
  - If worst-of ≥ protection barrier → repay notional
  - Else → repay notional × worst-of

The payoff is paid at a **random redemption time** (stopping time).

---

## Pricing Methodology
The product is priced as the **expected discounted value of its cashflows**:

\[
\text{Price} = \mathbb{E}\left[e^{-r \tau} \cdot \text{Payoff}(\tau)\right]
\]

where:
- \(\tau\) is the (random) redemption time
- the expectation is estimated via **Monte Carlo simulation**

### Market model (V1)
- Multi-asset Black–Scholes
- Constant volatilities
- Constant correlation
- Constant risk-free rate

Monte Carlo is used because the payoff is **path-dependent** and involves early stopping.

---

## Desk Simulation
Beyond pricing, the project simulates a **trading desk view** of the product:
- Daily revaluation
- Finite-difference Greeks (delta, vega)
- Discrete delta hedging
- Separation of product PnL and hedge PnL
- Aggregated daily PnL

This highlights key desk risks:
- Gamma risk near barriers
- Volatility risk (short vega)
- Correlation sensitivity in worst-of structures

---

## Project Structure
