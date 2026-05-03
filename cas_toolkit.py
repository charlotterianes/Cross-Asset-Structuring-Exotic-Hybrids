"""
============================================================================
Cross-Asset Structuring & Exotic Hybrids: single-file toolkit
============================================================================

A self-contained Python module covering pricing, risk, and idea-generation
tools for the structured products typically traded on Asian bank desks.

Sections
--------
1. Market data utilities         (discount factors, simple curves)
2. Vanilla pricers               (Black-Scholes, Garman-Kohlhagen, Black-76)
3. Volatility surface            (SVI fit, SABR, arbitrage checks, Dupire LV)
4. Numerical methods             (GBM paths, MC engine, bump-and-reval Greeks,
                                   Crank-Nicolson PDE)
5. Exotic products               (DCI, ELN, FCN, Autocallable, TARF,
                                   Range Accrual, Accumulator)
6. Hybrid products               (Quanto, Composite, simplified PRDC)
7. Risk                          (P&L explain, scenario ladders)
8. Demo                          (end-to-end run for every section)

Dependencies: numpy, scipy. No external bank libraries.

Author: Charlotte Riane
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, brentq


# ============================================================================
# 1. MARKET DATA UTILITIES
# ============================================================================

def df(rate: float, t: float) -> float:
    """Continuously compounded discount factor."""
    return float(np.exp(-rate * t))


@dataclass
class FlatCurve:
    """Minimal flat zero curve. Production code would use a bootstrapped curve."""
    rate: float

    def discount(self, t: float) -> float:
        return df(self.rate, t)

    def fwd(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            return self.rate
        return (np.log(self.discount(t1) / self.discount(t2))) / (t2 - t1)


@dataclass
class PiecewiseFlatCurve:
    """Piecewise-flat forward curve. Times in years, forwards in cont-comp."""
    times: np.ndarray   # length n+1, starting at 0
    forwards: np.ndarray  # length n

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.forwards = np.asarray(self.forwards, dtype=float)
        assert len(self.times) == len(self.forwards) + 1

    def discount(self, t: float) -> float:
        if t <= 0:
            return 1.0
        integral = 0.0
        for i, f in enumerate(self.forwards):
            t0, t1 = self.times[i], self.times[i + 1]
            if t <= t0:
                break
            seg = min(t, t1) - t0
            integral += f * seg
            if t <= t1:
                break
        return float(np.exp(-integral))


# ============================================================================
# 2. VANILLA PRICERS
# ============================================================================

def _bs_d1d2(S, K, T, sigma, r, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, sigma: float,
             r: float, q: float = 0.0, opt: str = 'call') -> float:
    """Black-Scholes price for an equity option with continuous dividend yield q."""
    if T <= 0:
        return max(S - K, 0) if opt == 'call' else max(K - S, 0)
    d1, d2 = _bs_d1d2(S, K, T, sigma, r, q)
    if opt == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, sigma: float,
              r: float, q: float = 0.0, opt: str = 'call') -> dict:
    """First and key second-order Greeks for BS equity option."""
    d1, d2 = _bs_d1d2(S, K, T, sigma, r, q)
    pdf = norm.pdf(d1)
    sign = 1 if opt == 'call' else -1
    return dict(
        delta=sign * np.exp(-q * T) * (norm.cdf(d1) if opt == 'call' else norm.cdf(d1) - 1),
        gamma=np.exp(-q * T) * pdf / (S * sigma * np.sqrt(T)),
        vega=S * np.exp(-q * T) * pdf * np.sqrt(T),
        theta=(-S * np.exp(-q * T) * pdf * sigma / (2 * np.sqrt(T))
               - sign * r * K * np.exp(-r * T) * (norm.cdf(d2) if opt == 'call' else norm.cdf(-d2))
               + sign * q * S * np.exp(-q * T) * (norm.cdf(d1) if opt == 'call' else norm.cdf(-d1))),
        rho=sign * K * T * np.exp(-r * T) * (norm.cdf(d2) if opt == 'call' else norm.cdf(-d2)),
        vanna=-np.exp(-q * T) * pdf * d2 / sigma,
        volga=S * np.exp(-q * T) * pdf * np.sqrt(T) * d1 * d2 / sigma,
    )


def gk_price(S: float, K: float, T: float, sigma: float,
             r_d: float, r_f: float, opt: str = 'call') -> float:
    """Garman-Kohlhagen FX option price. r_d = domestic, r_f = foreign."""
    return bs_price(S, K, T, sigma, r_d, r_f, opt)


def gk_greeks(S: float, K: float, T: float, sigma: float,
              r_d: float, r_f: float, opt: str = 'call') -> dict:
    """FX Greeks. Equivalent to BS with q -> r_f."""
    return bs_greeks(S, K, T, sigma, r_d, r_f, opt)


def black76_price(F: float, K: float, T: float, sigma: float,
                  r: float, opt: str = 'call') -> float:
    """Black-76 forward-style pricing for caps, floors, swaptions."""
    if T <= 0:
        return df(r, T) * (max(F - K, 0) if opt == 'call' else max(K - F, 0))
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return df(r, T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df(r, T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price: float, S: float, K: float, T: float,
                r: float, q: float = 0.0, opt: str = 'call') -> float:
    """Solve for BS implied vol using Brent's method."""
    def fn(s):
        return bs_price(S, K, T, s, r, q, opt) - price
    try:
        return brentq(fn, 1e-6, 5.0, xtol=1e-8)
    except ValueError:
        return float('nan')


# ============================================================================
# 3. VOLATILITY SURFACE
# ============================================================================

def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                       m: float, sig: float) -> np.ndarray:
    """Raw SVI total variance: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sig^2))."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))


def fit_svi_slice(strikes: np.ndarray, ivols: np.ndarray, T: float,
                  forward: float) -> dict:
    """Fit raw SVI to a single expiry slice. Returns parameter dict."""
    k = np.log(np.asarray(strikes) / forward)
    w_market = np.asarray(ivols) ** 2 * T

    def loss(params):
        a, b, rho, m, sig = params
        if b < 0 or sig <= 0 or abs(rho) >= 1 or b * (1 + abs(rho)) > 4 / T:
            return 1e10
        w_model = svi_total_variance(k, a, b, rho, m, sig)
        if np.any(w_model < 0):
            return 1e10
        return float(np.sum((w_model - w_market) ** 2))

    x0 = [float(np.median(w_market)), 0.1, -0.3, 0.0, 0.1]
    bounds = [(-1, 1), (1e-6, 5), (-0.999, 0.999), (-2, 2), (1e-3, 5)]
    res = minimize(loss, x0, bounds=bounds, method='L-BFGS-B')
    a, b, rho, m, sig = res.x
    return dict(a=a, b=b, rho=rho, m=m, sigma=sig, T=T, forward=forward,
                rmse=np.sqrt(res.fun / len(k)))


def svi_implied_vol(k: float, params: dict) -> float:
    """Recover implied vol from fitted SVI parameters at log-moneyness k."""
    w = svi_total_variance(np.array([k]), params['a'], params['b'],
                           params['rho'], params['m'], params['sigma'])[0]
    return float(np.sqrt(max(w, 1e-10) / params['T']))


def butterfly_arbitrage_check(strikes: np.ndarray, ivols: np.ndarray,
                              T: float, forward: float) -> bool:
    """
    Check density positivity (no butterfly arbitrage).
    Computes second derivative of call price with respect to strike.
    Returns True if surface is arbitrage-free.
    """
    K = np.asarray(strikes)
    sigmas = np.asarray(ivols)
    prices = np.array([bs_price(forward, k, T, s, 0, 0, 'call')
                       for k, s in zip(K, sigmas)])
    # Discrete second derivative: density proxy
    if len(K) < 3:
        return True
    d2 = np.diff(np.diff(prices) / np.diff(K)) / np.diff(K[:-1])
    return bool(np.all(d2 >= -1e-6))


def sabr_implied_vol(F: float, K: float, T: float, alpha: float,
                     beta: float, rho: float, nu: float) -> float:
    """Hagan's SABR lognormal implied volatility approximation."""
    if abs(F - K) < 1e-12:
        # ATM formula
        FK_beta = F ** (1 - beta)
        term1 = alpha / FK_beta
        term2 = ((1 - beta) ** 2 / 24) * alpha ** 2 / FK_beta ** 2
        term3 = 0.25 * rho * beta * nu * alpha / FK_beta
        term4 = (2 - 3 * rho ** 2) * nu ** 2 / 24
        return float(term1 * (1 + (term2 + term3 + term4) * T))
    FK = F * K
    log_FK = np.log(F / K)
    FK_beta = FK ** ((1 - beta) / 2)
    z = (nu / alpha) * FK_beta * log_FK
    if abs(z) < 1e-10:
        x_z = 1.0
    else:
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho)) / z
    pre = alpha / (FK_beta * (1 + ((1 - beta) ** 2 / 24) * log_FK ** 2
                              + ((1 - beta) ** 4 / 1920) * log_FK ** 4))
    correction = (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / FK_beta ** 2
                       + 0.25 * rho * beta * nu * alpha / FK_beta
                       + (2 - 3 * rho ** 2) * nu ** 2 / 24) * T)
    return float(pre * (1 / x_z) * correction)


def fit_sabr_slice(strikes: np.ndarray, ivols: np.ndarray, F: float, T: float,
                   beta: float = 1.0) -> dict:
    """Fit SABR (alpha, rho, nu) with beta fixed (typically 1.0 FX, 0.5 rates)."""
    K = np.asarray(strikes)
    iv = np.asarray(ivols)

    def loss(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        model = np.array([sabr_implied_vol(F, k, T, alpha, beta, rho, nu) for k in K])
        return float(np.sum((model - iv) ** 2))

    x0 = [0.2, 0.0, 0.3]
    bounds = [(1e-4, 5), (-0.999, 0.999), (1e-4, 5)]
    res = minimize(loss, x0, bounds=bounds, method='L-BFGS-B')
    alpha, rho, nu = res.x
    return dict(alpha=alpha, beta=beta, rho=rho, nu=nu,
                rmse=np.sqrt(res.fun / len(K)))


def dupire_local_vol(svi_params: dict, K: float, T: float,
                     forward_fn: Callable[[float], float],
                     r: float = 0.0, eps: float = 1e-4) -> float:
    """
    Dupire local volatility from a parameterised IV surface.
    Uses finite differences in (k, T) on total variance.

    Args:
        svi_params: dict-of-dicts keyed by T or callable returning SVI params at T
        K: strike
        T: maturity
        forward_fn: function returning forward at maturity T
        r: risk-free rate (only used for forward growth in time bumping here)
    """
    F = forward_fn(T)
    k = np.log(K / F)

    # Compute total variance and its derivatives by finite differences
    def w_at(kk, tt):
        params = svi_params if not callable(svi_params) else svi_params(tt)
        return float(svi_total_variance(np.array([kk]), params['a'], params['b'],
                                         params['rho'], params['m'], params['sigma'])[0])

    w = w_at(k, T)
    dw_dk = (w_at(k + eps, T) - w_at(k - eps, T)) / (2 * eps)
    d2w_dk2 = (w_at(k + eps, T) - 2 * w + w_at(k - eps, T)) / (eps ** 2)
    dw_dT = (w_at(k, T + eps) - w_at(k, max(T - eps, 1e-6))) / (2 * eps) if T > eps else \
            (w_at(k, T + eps) - w) / eps

    # Dupire formula (Gatheral form)
    denom = (1 - (k / w) * dw_dk
             + 0.25 * (-0.25 - 1 / w + (k / w) ** 2) * dw_dk ** 2
             + 0.5 * d2w_dk2)
    if denom <= 0 or dw_dT <= 0:
        return float('nan')
    return float(np.sqrt(dw_dT / denom))


# ============================================================================
# 4. NUMERICAL METHODS
# ============================================================================

def gbm_paths(S0: float, mu: float, sigma: float, T: float,
              n_steps: int, n_paths: int, antithetic: bool = True,
              seed: Optional[int] = None) -> np.ndarray:
    """Simulate geometric Brownian motion paths under given drift mu."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z_half, -Z_half])
        if Z.shape[0] < n_paths:
            extra = rng.standard_normal((n_paths - Z.shape[0], n_steps))
            Z = np.vstack([Z, extra])
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    paths = S0 * np.exp(log_paths)
    return np.hstack([np.full((n_paths, 1), S0), paths])


def correlated_gbm_paths(S0: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                         corr: np.ndarray, T: float, n_steps: int,
                         n_paths: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Multi-asset correlated GBM. Returns array of shape (n_paths, n_steps+1, n_assets).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(S0)
    dt = T / n_steps
    L = np.linalg.cholesky(corr)

    paths = np.zeros((n_paths, n_steps + 1, n_assets))
    paths[:, 0, :] = S0

    for t in range(1, n_steps + 1):
        z = rng.standard_normal((n_paths, n_assets))
        cz = z @ L.T
        increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * cz
        paths[:, t, :] = paths[:, t - 1, :] * np.exp(increments)
    return paths


def bumped_greeks(pricer_fn: Callable, base_kwargs: dict,
                  bumps: Optional[dict] = None) -> dict:
    """
    Generic finite-difference Greeks. Pricer must accept kwargs.

    bumps: dict mapping kwarg name -> (size, type) where type in {'absolute', 'relative'}.
    Default bumps cover spot, sigma, r.
    """
    if bumps is None:
        bumps = {}
        if 'S' in base_kwargs: bumps['S'] = (0.01, 'relative')
        if 'spot' in base_kwargs: bumps['spot'] = (0.01, 'relative')
        if 'sigma' in base_kwargs: bumps['sigma'] = (0.01, 'absolute')
        if 'r' in base_kwargs: bumps['r'] = (0.0001, 'absolute')

    base_px = pricer_fn(**base_kwargs)
    out = {'pv': base_px}
    for key, (size, btype) in bumps.items():
        up_kw, dn_kw = dict(base_kwargs), dict(base_kwargs)
        if btype == 'relative':
            shift = base_kwargs[key] * size
        else:
            shift = size
        up_kw[key] = base_kwargs[key] + shift
        dn_kw[key] = base_kwargs[key] - shift
        up_px = pricer_fn(**up_kw)
        dn_px = pricer_fn(**dn_kw)
        out[f'd_{key}'] = (up_px - dn_px) / (2 * shift)
        out[f'd2_{key}'] = (up_px - 2 * base_px + dn_px) / (shift ** 2)
    return out


def crank_nicolson_european(S0: float, K: float, T: float, sigma: float,
                            r: float, q: float, opt: str = 'call',
                            S_max_mult: float = 4.0, M: int = 200,
                            N: int = 200) -> float:
    """
    Crank-Nicolson finite difference solver for European vanilla.
    Used here as a sanity check against BS closed form.
    """
    S_max = S_max_mult * max(S0, K)
    dS = S_max / M
    dt = T / N
    S_grid = np.linspace(0, S_max, M + 1)

    # Terminal payoff
    if opt == 'call':
        V = np.maximum(S_grid - K, 0)
    else:
        V = np.maximum(K - S_grid, 0)

    # Build tridiagonal coefficients (interior nodes only)
    j = np.arange(1, M)
    alpha = 0.25 * dt * (sigma ** 2 * j ** 2 - (r - q) * j)
    beta = -0.5 * dt * (sigma ** 2 * j ** 2 + r)
    gamma = 0.25 * dt * (sigma ** 2 * j ** 2 + (r - q) * j)

    # Tridiagonal matrices
    A = np.diag(1 - beta) - np.diag(alpha[1:], -1) - np.diag(gamma[:-1], 1)
    B = np.diag(1 + beta) + np.diag(alpha[1:], -1) + np.diag(gamma[:-1], 1)

    for _ in range(N):
        rhs = B @ V[1:M]
        # Boundary at S=0 and S=S_max
        if opt == 'call':
            V0_new, VM_new = 0.0, S_max - K * np.exp(-r * dt)
        else:
            V0_new, VM_new = K * np.exp(-r * dt), 0.0
        rhs[0] += alpha[0] * (V[0] + V0_new)
        rhs[-1] += gamma[-1] * (V[M] + VM_new)
        V_new = np.linalg.solve(A, rhs)
        V[0] = V0_new
        V[M] = VM_new
        V[1:M] = V_new

    # Interpolate to S0
    return float(np.interp(S0, S_grid, V))


# ============================================================================
# 5. EXOTIC PRODUCTS
# ============================================================================

@dataclass
class DCI:
    """
    Dual Currency Investment.
    Client deposits notional in CCY1, gets enhanced coupon, but if spot
    breaches strike at maturity, principal converts to CCY2 at strike.
    Decomposition: deposit + short OTM put on CCY1/CCY2.
    """
    notional: float
    spot: float           # CCY1/CCY2 (e.g. SGDUSD)
    strike: float
    tenor_days: int
    coupon_rate: float    # annualised, paid to client
    sigma: float
    r_dom: float          # CCY1 rate (deposit currency)
    r_for: float          # CCY2 rate

    @property
    def T(self) -> float:
        return self.tenor_days / 365.0

    def fair_coupon(self) -> float:
        """Coupon making the structure zero-cost to the bank (before margin)."""
        put_px = gk_price(self.spot, self.strike, self.T, self.sigma,
                          self.r_dom, self.r_for, 'put')
        n_puts = self.notional / self.strike
        premium = n_puts * put_px
        return self.r_dom + (premium / self.notional) / self.T

    def client_payoff(self, spot_T: float) -> dict:
        coupon_amt = self.notional * self.coupon_rate * self.T
        if spot_T >= self.strike:
            return dict(ccy='CCY1', principal=self.notional,
                        coupon=coupon_amt,
                        total_in_ccy1=self.notional + coupon_amt)
        principal_ccy2 = self.notional / self.strike
        coupon_ccy2 = coupon_amt / self.strike
        return dict(ccy='CCY2', principal=principal_ccy2, coupon=coupon_ccy2,
                    total_in_ccy1=(principal_ccy2 + coupon_ccy2) * spot_T)

    def bank_pv(self) -> float:
        """PV to bank of the embedded short put (positive = bank receives)."""
        put_px = gk_price(self.spot, self.strike, self.T, self.sigma,
                          self.r_dom, self.r_for, 'put')
        return (self.notional / self.strike) * put_px


@dataclass
class ELN:
    """
    Equity-Linked Note: bond + short put on single name.
    Client receives coupon, takes equity-like downside if stock < strike.
    """
    notional: float
    spot: float
    strike: float
    tenor_days: int
    coupon_rate: float
    sigma: float
    r: float
    q: float = 0.0

    @property
    def T(self) -> float:
        return self.tenor_days / 365.0

    def fair_coupon(self) -> float:
        put_px = bs_price(self.spot, self.strike, self.T, self.sigma,
                          self.r, self.q, 'put')
        n_puts = self.notional / self.strike
        premium = n_puts * put_px
        return self.r + (premium / self.notional) / self.T

    def client_payoff(self, spot_T: float) -> float:
        coupon_amt = self.notional * self.coupon_rate * self.T
        if spot_T >= self.strike:
            return self.notional + coupon_amt
        # Take physical or cash-settled equivalent at strike
        n_shares = self.notional / self.strike
        return n_shares * spot_T + coupon_amt


@dataclass
class FCN:
    """
    Fixed Coupon Note (worst-of basket). Bond + short worst-of put.
    High coupon comes from selling dispersion / correlation risk.
    """
    notional: float
    spots: np.ndarray         # initial spots, length N
    sigmas: np.ndarray
    divs: np.ndarray
    corr: np.ndarray          # N x N
    strike_pct: float         # KI level, e.g. 0.65 = 65% of initial
    tenor_yrs: float
    coupon_rate: float
    r: float

    def price_mc(self, n_paths: int = 50_000, seed: int = 42) -> dict:
        spots = np.asarray(self.spots, dtype=float)
        sigmas = np.asarray(self.sigmas, dtype=float)
        divs = np.asarray(self.divs, dtype=float)
        n_steps = max(int(self.tenor_yrs * 252), 1)

        paths = correlated_gbm_paths(
            S0=spots, mu=self.r - divs, sigma=sigmas, corr=self.corr,
            T=self.tenor_yrs, n_steps=n_steps, n_paths=n_paths, seed=seed)
        # Worst-of final ratio
        ratios = paths[:, -1, :] / spots
        min_ratio = ratios.min(axis=1)

        # Coupon paid regardless (in vanilla FCN); principal at risk if min < strike
        df_T = np.exp(-self.r * self.tenor_yrs)
        coupon_pv = self.notional * self.coupon_rate * self.tenor_yrs * df_T
        # Principal: full if min >= strike, else worst-of share-settled at strike
        principal_payoff = np.where(min_ratio >= self.strike_pct,
                                     self.notional,
                                     self.notional * (min_ratio / self.strike_pct))
        principal_pv = df_T * principal_payoff.mean()
        total_client_pv = principal_pv + coupon_pv
        bank_pv = self.notional - total_client_pv
        return dict(client_pv=total_client_pv, bank_pv=bank_pv,
                    coupon_pv=coupon_pv, principal_pv=principal_pv,
                    se=self.notional * principal_payoff.std() / np.sqrt(n_paths))


@dataclass
class WorstOfAutocallable:
    """
    Worst-of Phoenix Autocallable.
    - Coupon paid on observation date if all underlyings >= coupon_barrier.
    - Autocalls (early redeems at par) if all >= autocall_barrier on obs date.
    - At maturity, if KI ever touched and worst < initial, client takes
      worst-of equity loss; otherwise par.
    """
    spots: np.ndarray
    sigmas: np.ndarray
    divs: np.ndarray
    corr: np.ndarray
    r: float
    coupon: float                  # cash amount per obs date if condition met
    coupon_barrier: float          # fraction of initial
    autocall_barrier: float
    ki_barrier: float
    obs_dates_yrs: Sequence[float]
    maturity_yrs: float
    notional: float = 1.0
    n_paths: int = 20_000
    seed: int = 42

    def price(self) -> dict:
        spots = np.asarray(self.spots, dtype=float)
        sigmas = np.asarray(self.sigmas, dtype=float)
        divs = np.asarray(self.divs, dtype=float)
        n_steps = max(int(self.maturity_yrs * 252), 1)
        dt = self.maturity_yrs / n_steps
        obs_steps = sorted(set(min(n_steps, max(1, int(round(t * 252))))
                               for t in self.obs_dates_yrs))

        paths = correlated_gbm_paths(
            S0=spots, mu=self.r - divs, sigma=sigmas, corr=self.corr,
            T=self.maturity_yrs, n_steps=n_steps, n_paths=self.n_paths,
            seed=self.seed)
        ratios = paths / spots  # (n_paths, n_steps+1, n_assets)
        worst_path = ratios.min(axis=2)  # (n_paths, n_steps+1)

        pvs = np.zeros(self.n_paths)
        for p in range(self.n_paths):
            wp = worst_path[p]
            # KI: continuous monitoring on min across path
            ki_hit = bool((wp < self.ki_barrier).any())
            autocalled = False
            pv = 0.0
            for step in obs_steps:
                t = step * dt
                df_t = np.exp(-self.r * t)
                if wp[step] >= self.coupon_barrier:
                    pv += df_t * self.coupon * self.notional
                if wp[step] >= self.autocall_barrier:
                    pv += df_t * self.notional
                    autocalled = True
                    break
            if not autocalled:
                df_T = np.exp(-self.r * self.maturity_yrs)
                final = wp[-1]
                if ki_hit and final < 1.0:
                    pv += df_T * self.notional * final
                else:
                    pv += df_T * self.notional
            pvs[p] = pv
        return dict(pv=float(pvs.mean()),
                    se=float(pvs.std() / np.sqrt(self.n_paths)))


@dataclass
class TARF:
    """
    Target Redemption Forward.
    Strip of weekly/monthly fixings. Client receives (strike - spot) when
    spot < strike; pays leverage * (spot - strike) when spot > strike.
    Knocks out when cumulative client profit hits target.

    Famous for blowing up corporates when spot trends against them because
    the loss leg is unbounded but the gain leg is capped at target.
    """
    spot: float
    strike: float
    target_profit: float
    n_fixings: int
    fixing_interval_days: int
    sigma: float
    r_dom: float
    r_for: float
    leverage: float = 2.0
    notional_per_fixing: float = 1.0

    def price(self, n_paths: int = 50_000, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        dt = self.fixing_interval_days / 365.0
        drift = (self.r_dom - self.r_for - 0.5 * self.sigma ** 2) * dt
        diff = self.sigma * np.sqrt(dt)

        bank_pvs = np.zeros(n_paths)
        ko_steps = np.zeros(n_paths)
        for i in range(n_paths):
            s = self.spot
            cum_profit = 0.0
            path_pv = 0.0
            ko_at = self.n_fixings
            for k in range(1, self.n_fixings + 1):
                z = rng.standard_normal()
                s = s * np.exp(drift + diff * z)
                t = k * dt
                df_t = np.exp(-self.r_dom * t)
                if s < self.strike:
                    gain_to_client = (self.strike - s) * self.notional_per_fixing
                    cum_profit += gain_to_client
                    path_pv -= df_t * gain_to_client   # bank pays
                    if cum_profit >= self.target_profit:
                        ko_at = k
                        break
                else:
                    loss_to_client = (s - self.strike) * self.leverage * self.notional_per_fixing
                    path_pv += df_t * loss_to_client    # bank receives
            bank_pvs[i] = path_pv
            ko_steps[i] = ko_at
        return dict(bank_pv=float(bank_pvs.mean()),
                    se=float(bank_pvs.std() / np.sqrt(n_paths)),
                    avg_fixings_to_ko=float(ko_steps.mean()),
                    pct_ko=float((ko_steps < self.n_fixings).mean()))


@dataclass
class RangeAccrual:
    """
    Pays daily coupon when spot/rate is inside [lower, upper] range.
    Sensitivity: short volatility of underlier (wider realised range = fewer days inside).
    """
    spot: float
    lower: float
    upper: float
    daily_coupon: float
    tenor_days: int
    sigma: float
    r: float
    q: float = 0.0

    def price(self, n_paths: int = 50_000, seed: int = 42) -> dict:
        T = self.tenor_days / 365.0
        paths = gbm_paths(self.spot, self.r - self.q, self.sigma, T,
                          self.tenor_days, n_paths, antithetic=True, seed=seed)
        # Days inside range across the path (excluding initial day)
        in_range = (paths[:, 1:] >= self.lower) & (paths[:, 1:] <= self.upper)
        days_in = in_range.sum(axis=1)
        # PV: discount each daily coupon
        dt = 1 / 365.0
        ts = np.arange(1, self.tenor_days + 1) * dt
        dfs = np.exp(-self.r * ts)
        pv_per_path = (in_range * dfs).sum(axis=1) * self.daily_coupon
        return dict(pv=float(pv_per_path.mean()),
                    se=float(pv_per_path.std() / np.sqrt(n_paths)),
                    avg_days_in_range=float(days_in.mean()))


@dataclass
class Accumulator:
    """
    Daily share accumulator with knockout.
    Each day: client buys 1x notional at strike if spot >= strike,
    2x notional if spot < strike (the "I-kill-you-later" feature).
    Knocks out if spot >= ko_barrier.
    """
    spot: float
    strike: float
    ko_barrier: float
    n_days: int
    sigma: float
    r: float
    q: float = 0.0
    leverage_below: float = 2.0

    def price(self, n_paths: int = 30_000, seed: int = 42) -> dict:
        T = self.n_days / 365.0
        paths = gbm_paths(self.spot, self.r - self.q, self.sigma, T,
                          self.n_days, n_paths, antithetic=True, seed=seed)
        client_pvs = np.zeros(n_paths)
        for p in range(n_paths):
            pv = 0.0
            for d in range(1, self.n_days + 1):
                s = paths[p, d]
                t = d / 365.0
                df_t = np.exp(-self.r * t)
                if s >= self.ko_barrier:
                    break
                # Client buys at strike: payoff = (spot - strike) per share
                size = self.leverage_below if s < self.strike else 1.0
                pv += df_t * size * (s - self.strike)
            client_pvs[p] = pv
        return dict(client_pv=float(client_pvs.mean()),
                    se=float(client_pvs.std() / np.sqrt(n_paths)))


# ============================================================================
# 6. HYBRID PRODUCTS
# ============================================================================

def quanto_call(S: float, K: float, T: float, sigma_S: float,
                sigma_FX: float, rho: float,
                r_dom: float, r_for: float, q: float = 0.0,
                opt: str = 'call') -> float:
    """
    Quanto vanilla: payoff in domestic currency on foreign underlying.
    Drift adjustment: mu_S = r_for - q - rho * sigma_S * sigma_FX
    Discount in domestic currency.
    """
    mu_S = r_for - q - rho * sigma_S * sigma_FX
    d1 = (np.log(S / K) + (mu_S + 0.5 * sigma_S ** 2) * T) / (sigma_S * np.sqrt(T))
    d2 = d1 - sigma_S * np.sqrt(T)
    if opt == 'call':
        return np.exp(-r_dom * T) * (S * np.exp(mu_S * T) * norm.cdf(d1)
                                      - K * norm.cdf(d2))
    return np.exp(-r_dom * T) * (K * norm.cdf(-d2)
                                  - S * np.exp(mu_S * T) * norm.cdf(-d1))


def composite_call(S: float, K: float, T: float, sigma_S: float,
                   sigma_FX: float, rho: float, FX0: float,
                   r_dom: float, r_for: float, q: float = 0.0,
                   opt: str = 'call') -> float:
    """
    Composite option: foreign underlying converted to domestic at maturity FX rate.
    Effective vol: sqrt(sigma_S^2 + sigma_FX^2 + 2*rho*sigma_S*sigma_FX)
    Effective spot: S * FX0 (today's converted value)
    """
    sigma_eff = np.sqrt(sigma_S ** 2 + sigma_FX ** 2 + 2 * rho * sigma_S * sigma_FX)
    return bs_price(S * FX0, K, T, sigma_eff, r_dom, q, opt)


def prdc_3factor_mc(
    notional: float,
    fx0: float, strike_fx: float,
    r_dom_curve: FlatCurve, r_for_curve: FlatCurve,
    a_dom: float, sigma_dom: float,
    a_for: float, sigma_for: float,
    sigma_fx: float,
    corr: np.ndarray,        # 3x3 between (r_dom, r_for, fx)
    coupon_floor: float, coupon_cap: float,
    tenor_yrs: float, n_coupons: int,
    n_paths: int = 5000, seed: int = 42
) -> dict:
    """
    Simplified Power Reverse Dual Currency note.
    Coupon at each date = max(min(c1 * FX/strike - c2, cap), floor).
    Three-factor model: Hull-White domestic, Hull-White foreign, log-normal FX.

    This is illustrative. Production calibration is much heavier (joint
    calibration to swaptions in each ccy plus FX vol surface).
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr)
    n_steps = max(int(tenor_yrs * 52), 1)   # weekly steps
    dt = tenor_yrs / n_steps
    coupon_steps = [int(round((i + 1) * n_steps / n_coupons)) for i in range(n_coupons)]

    pvs = np.zeros(n_paths)
    for p in range(n_paths):
        x_d, x_f = 0.0, 0.0   # short rate factors (deviations from initial curve)
        fx = fx0
        pv = 0.0
        cum_disc = 0.0
        coupon_idx = 0

        for step in range(1, n_steps + 1):
            z = rng.standard_normal(3)
            cz = L @ z
            # Hull-White Euler
            x_d = x_d - a_dom * x_d * dt + sigma_dom * np.sqrt(dt) * cz[0]
            x_f = x_f - a_for * x_f * dt + sigma_for * np.sqrt(dt) * cz[1]
            # FX log-normal under domestic risk-neutral with quanto adjustment
            r_d_inst = r_dom_curve.rate + x_d
            r_f_inst = r_for_curve.rate + x_f
            fx = fx * np.exp((r_d_inst - r_f_inst - 0.5 * sigma_fx ** 2) * dt
                              + sigma_fx * np.sqrt(dt) * cz[2])
            cum_disc += r_d_inst * dt

            if coupon_idx < n_coupons and step == coupon_steps[coupon_idx]:
                # Coupon formula: tied to fx ratio
                raw = (fx / strike_fx) - 1.0
                coupon = max(min(raw, coupon_cap), coupon_floor)
                pv += notional * coupon * np.exp(-cum_disc)
                coupon_idx += 1
        # Principal repayment
        pv += notional * np.exp(-cum_disc)
        pvs[p] = pv
    return dict(pv=float(pvs.mean()),
                se=float(pvs.std() / np.sqrt(n_paths)))


# ============================================================================
# 7. RISK
# ============================================================================

def pnl_explain(prev_greeks: dict, prev_market: dict, today_market: dict,
                today_pv: float, prev_pv: float, dt_days: float) -> dict:
    """
    Decompose today's PV change into Greek contributions.
    Anything left in 'unexplained' is what traders investigate.
    """
    dS = today_market.get('S', today_market.get('spot', 0)) - \
         prev_market.get('S', prev_market.get('spot', 0))
    dSig = today_market.get('sigma', 0) - prev_market.get('sigma', 0)
    dR = today_market.get('r', today_market.get('r_dom', 0)) - \
         prev_market.get('r', prev_market.get('r_dom', 0))
    dt = dt_days / 365.0

    delta_pnl = prev_greeks.get('delta', 0) * dS
    gamma_pnl = 0.5 * prev_greeks.get('gamma', 0) * dS ** 2
    vega_pnl = prev_greeks.get('vega', 0) * dSig
    theta_pnl = prev_greeks.get('theta', 0) * dt
    rho_pnl = prev_greeks.get('rho', 0) * dR
    vanna_pnl = prev_greeks.get('vanna', 0) * dS * dSig
    volga_pnl = 0.5 * prev_greeks.get('volga', 0) * dSig ** 2

    explained = (delta_pnl + gamma_pnl + vega_pnl + theta_pnl
                 + rho_pnl + vanna_pnl + volga_pnl)
    actual = today_pv - prev_pv
    return dict(
        actual=actual, explained=explained, unexplained=actual - explained,
        delta=delta_pnl, gamma=gamma_pnl, vega=vega_pnl, theta=theta_pnl,
        rho=rho_pnl, vanna=vanna_pnl, volga=volga_pnl,
    )


def scenario_ladder(pricer_fn: Callable, base_kwargs: dict,
                    spot_key: str = 'S', vol_key: str = 'sigma',
                    spot_shifts: Sequence[float] = (-0.20, -0.10, -0.05, 0,
                                                     0.05, 0.10, 0.20),
                    vol_shifts: Sequence[float] = (-0.05, -0.02, 0,
                                                    0.02, 0.05)) -> dict:
    """
    Run a 2D grid of (spot, vol) shifts. Returns dict-of-dicts of PVs and PnLs.
    spot_shifts are relative, vol_shifts are absolute vol points.
    """
    base_pv = pricer_fn(**base_kwargs)
    grid = {}
    for ds in spot_shifts:
        row = {}
        for dv in vol_shifts:
            kw = dict(base_kwargs)
            kw[spot_key] = base_kwargs[spot_key] * (1 + ds)
            kw[vol_key] = base_kwargs[vol_key] + dv
            pv = pricer_fn(**kw)
            row[f'{dv:+.2f}'] = dict(pv=pv, pnl=pv - base_pv)
        grid[f'{ds:+.0%}'] = row
    return dict(base_pv=base_pv, ladder=grid)


# ============================================================================
# 8. DEMO
# ============================================================================

def _sep(title: str) -> None:
    print('\n' + '=' * 70)
    print(f' {title}')
    print('=' * 70)


def run_demo() -> None:
    np.set_printoptions(precision=4, suppress=True)

    # ---- Vanillas ----
    _sep('VANILLA PRICERS')
    px_call = bs_price(100, 100, 1.0, 0.20, 0.04, 0.02, 'call')
    px_cn = crank_nicolson_european(100, 100, 1.0, 0.20, 0.04, 0.02, 'call')
    print(f'BS call (closed form):   {px_call:.4f}')
    print(f'BS call (Crank-Nicolson): {px_cn:.4f}  (sanity check)')
    g = bs_greeks(100, 100, 1.0, 0.20, 0.04, 0.02, 'call')
    print('Greeks:', {k: round(v, 4) for k, v in g.items()})

    fx_put = gk_price(1.35, 1.42, 30/365, 0.07, 0.035, 0.045, 'put')
    print(f'\nFX put (GK, SGDUSD-style): {fx_put:.4f}')

    iv = implied_vol(px_call, 100, 100, 1.0, 0.04, 0.02, 'call')
    print(f'Implied vol round-trip: {iv:.4f}  (should be 0.20)')

    # ---- Vol surface ----
    _sep('VOL SURFACE: SVI FIT')
    strikes = np.array([80, 90, 95, 100, 105, 110, 120])
    ivols = np.array([0.28, 0.24, 0.22, 0.20, 0.21, 0.23, 0.27])
    svi = fit_svi_slice(strikes, ivols, T=1.0, forward=100.0)
    print('SVI params:', {k: round(float(v), 4) for k, v in svi.items()})
    iv_check = svi_implied_vol(0.0, svi)
    print(f'SVI ATM vol: {iv_check:.4f}')
    arb_free = butterfly_arbitrage_check(strikes, ivols, 1.0, 100.0)
    print(f'Butterfly arb-free: {arb_free}')

    _sep('VOL SURFACE: SABR FIT')
    sabr = fit_sabr_slice(strikes, ivols, F=100.0, T=1.0, beta=1.0)
    print('SABR params:', {k: round(float(v), 4) for k, v in sabr.items()})

    # ---- DCI ----
    _sep('DCI (Dual Currency Investment)')
    # USDSGD setup: client deposits SGD, alternate currency USD.
    # Strike 1.32 = 2.2% OTM put on USDSGD (i.e. client takes USD if SGD strengthens).
    dci = DCI(notional=1_000_000, spot=1.35, strike=1.32,
              tenor_days=30, coupon_rate=0.06,
              sigma=0.07, r_dom=0.035, r_for=0.045)
    print(f'DCI fair coupon (annualised): {dci.fair_coupon():.4%}')
    print(f'Bank PV of embedded short put: {dci.bank_pv():.2f}')
    print('Client payoff if spot=1.30:', dci.client_payoff(1.30))
    print('Client payoff if spot=1.40:', dci.client_payoff(1.40))

    # ---- ELN ----
    _sep('ELN (Equity-Linked Note)')
    eln = ELN(notional=100_000, spot=100, strike=90, tenor_days=180,
              coupon_rate=0.10, sigma=0.30, r=0.04, q=0.02)
    print(f'ELN fair coupon: {eln.fair_coupon():.4%}')
    print(f'Client payoff if stock ends at 80: {eln.client_payoff(80):.2f}')
    print(f'Client payoff if stock ends at 100: {eln.client_payoff(100):.2f}')

    # ---- FCN ----
    _sep('FCN (Worst-of Fixed Coupon Note)')
    fcn = FCN(notional=100_000,
              spots=np.array([100, 100, 100]),
              sigmas=np.array([0.25, 0.30, 0.28]),
              divs=np.array([0.02, 0.02, 0.02]),
              corr=np.array([[1, 0.6, 0.5],
                             [0.6, 1, 0.55],
                             [0.5, 0.55, 1]]),
              strike_pct=0.65, tenor_yrs=1.0, coupon_rate=0.12, r=0.04)
    res = fcn.price_mc(n_paths=10_000)
    print(f"FCN client PV: {res['client_pv']:.2f}, bank PV: {res['bank_pv']:.2f}")

    # ---- Autocallable ----
    _sep('WORST-OF PHOENIX AUTOCALLABLE')
    ac = WorstOfAutocallable(
        spots=np.array([100, 100]),
        sigmas=np.array([0.25, 0.30]),
        divs=np.array([0.02, 0.02]),
        corr=np.array([[1, 0.6], [0.6, 1]]),
        r=0.04, coupon=0.04,
        coupon_barrier=0.65, autocall_barrier=1.0,
        ki_barrier=0.60,
        obs_dates_yrs=[0.5, 1.0, 1.5, 2.0],
        maturity_yrs=2.0, n_paths=5_000)
    out = ac.price()
    print(f"Autocallable PV: {out['pv']:.4f} +/- {1.96 * out['se']:.4f}")

    # ---- TARF ----
    _sep('TARF (Target Redemption Forward)')
    tarf = TARF(spot=1.35, strike=1.32, target_profit=0.10,
                n_fixings=24, fixing_interval_days=14,
                sigma=0.08, r_dom=0.035, r_for=0.045, leverage=2.0)
    out = tarf.price(n_paths=10_000)
    print(f"TARF bank PV: {out['bank_pv']:.4f} +/- {1.96 * out['se']:.4f}")
    print(f"% of paths that knocked out: {out['pct_ko']:.1%}")
    print(f"Avg fixings to KO: {out['avg_fixings_to_ko']:.1f}")

    # ---- Range Accrual ----
    _sep('RANGE ACCRUAL')
    ra = RangeAccrual(spot=1.35, lower=1.30, upper=1.40,
                      daily_coupon=100, tenor_days=90,
                      sigma=0.07, r=0.035)
    out = ra.price(n_paths=10_000)
    print(f"Range accrual PV: {out['pv']:.2f} +/- {1.96 * out['se']:.2f}")
    print(f"Avg days in range: {out['avg_days_in_range']:.1f} / 90")

    # ---- Accumulator ----
    _sep('ACCUMULATOR')
    acc = Accumulator(spot=100, strike=98, ko_barrier=103,
                      n_days=126, sigma=0.25, r=0.04, q=0.02,
                      leverage_below=2.0)
    out = acc.price(n_paths=5_000)
    print(f"Accumulator client PV: {out['client_pv']:.4f} +/- {1.96 * out['se']:.4f}")

    # ---- Quanto ----
    _sep('QUANTO + COMPOSITE')
    qc = quanto_call(S=4500, K=4500, T=1.0, sigma_S=0.18, sigma_FX=0.08,
                     rho=0.30, r_dom=0.035, r_for=0.04, q=0.015)
    cc = composite_call(S=4500, K=4500*1.35, T=1.0, sigma_S=0.18,
                        sigma_FX=0.08, rho=0.30, FX0=1.35,
                        r_dom=0.035, r_for=0.04, q=0.015)
    print(f'Quanto SPX call (paid in SGD): {qc:.4f}')
    print(f'Composite SPX call (FX-converted at maturity): {cc:.4f}')

    # ---- Simplified PRDC ----
    _sep('SIMPLIFIED PRDC')
    prdc = prdc_3factor_mc(
        notional=1_000_000,
        fx0=150.0, strike_fx=140.0,
        r_dom_curve=FlatCurve(0.04), r_for_curve=FlatCurve(0.001),
        a_dom=0.05, sigma_dom=0.01,
        a_for=0.05, sigma_for=0.005,
        sigma_fx=0.10,
        corr=np.array([[1.0, 0.2, -0.3],
                       [0.2, 1.0, 0.1],
                       [-0.3, 0.1, 1.0]]),
        coupon_floor=0.0, coupon_cap=0.08,
        tenor_yrs=5.0, n_coupons=10, n_paths=2000)
    print(f"PRDC PV: {prdc['pv']:.2f} +/- {1.96 * prdc['se']:.2f}")

    # ---- P&L Explain ----
    _sep('P&L EXPLAIN (one-day)')
    prev_kw = dict(S=100, K=100, T=1.0, sigma=0.20, r=0.04, q=0.02, opt='call')
    today_kw = dict(prev_kw)
    today_kw.update(S=101.5, sigma=0.21, T=1.0 - 1/365)
    prev_pv = bs_price(**prev_kw)
    today_pv = bs_price(**today_kw)
    prev_g = bs_greeks(prev_kw['S'], prev_kw['K'], prev_kw['T'],
                       prev_kw['sigma'], prev_kw['r'], prev_kw['q'], 'call')
    explain = pnl_explain(
        prev_greeks=prev_g,
        prev_market=dict(S=100, sigma=0.20, r=0.04),
        today_market=dict(S=101.5, sigma=0.21, r=0.04),
        today_pv=today_pv, prev_pv=prev_pv, dt_days=1)
    for k, v in explain.items():
        print(f'  {k:14s}: {v:+.4f}')

    # ---- Scenario ladder ----
    _sep('SCENARIO LADDER (spot x vol)')
    base_kw = dict(S=100, K=100, T=1.0, sigma=0.20, r=0.04, q=0.02, opt='call')
    ladder = scenario_ladder(bs_price, base_kw)
    print(f"Base PV: {ladder['base_pv']:.4f}")
    print('PnL grid (rows=spot shift, cols=vol shift):')
    print(f"{'':>8s}", end='')
    vol_keys = list(next(iter(ladder['ladder'].values())).keys())
    for vk in vol_keys:
        print(f"{vk:>10s}", end='')
    print()
    for spot_k, row in ladder['ladder'].items():
        print(f"{spot_k:>8s}", end='')
        for vk in vol_keys:
            print(f"{row[vk]['pnl']:>+10.3f}", end='')
        print()

    _sep('DEMO COMPLETE')


if __name__ == '__main__':
    run_demo()
