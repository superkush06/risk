"""Value at Risk: parametric, historical, and Monte Carlo."""

from __future__ import annotations

import math

import numpy as np


def _phi_inv(p: float) -> float:
    """Inverse standard-normal CDF via Beasley-Springer-Moro approximation."""
    # Coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /                (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def parametric_var(mu: float, sigma: float, *, alpha: float = 0.95,
                   horizon: int = 1) -> float:
    """Gaussian VaR: VaR_alpha = -(mu - z * sigma) for one-sided loss.

    `mu`, `sigma` are per-period; horizon scales by sqrt(t).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    z = _phi_inv(alpha)
    scale = math.sqrt(horizon)
    return -(mu * horizon - z * sigma * scale)


def historical_var(returns: np.ndarray, *, alpha: float = 0.95) -> float:
    """Empirical VaR: -quantile_(1-alpha) of historical returns."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    returns = np.asarray(returns, dtype=float)
    return float(-np.quantile(returns, 1.0 - alpha))


def monte_carlo_var(simulate, *, n_paths: int = 100_000, alpha: float = 0.95,
                    seed: int = 0) -> float:
    """Monte Carlo VaR: `simulate(rng, n)` -> array of P&L draws."""
    rng = np.random.default_rng(seed)
    pnl = simulate(rng, n_paths)
    return historical_var(pnl, alpha=alpha)
