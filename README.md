# Cross-Asset Structuring & Exotic Hybrids

A working repository implementing pricing, risk, and idea-generation tools for the structured products typically traded on Asian bank desks: DCI, FCN, autocallables, TARFs, range accruals, and rates/FX hybrids.

The aim is not to re-implement Black-Scholes for the hundredth time, but to build a coherent toolkit that mirrors how a structuring desk actually works: market-data layer, vol surfaces, vanillas, exotics, hybrids, risk attribution. Asia/FX product weighting is deliberate.

A practitioner-oriented companion to Bouzoubaa & Osseiran's *Exotic Options and Hybrids*, with implementations weighted toward products that dominate Asian structuring flow.

## Status

**Currently in `cas_toolkit.py`:**

- [x] Black-Scholes, Garman-Kohlhagen, Black-76 with full Greeks (delta, gamma, vega, theta, rho, vanna, volga)
- [x] Implied vol solver (Brent)
- [x] Raw SVI fit with butterfly arbitrage check
- [x] SABR fit (Hagan lognormal approximation)
- [x] Dupire local vol from parameterised IV surface
- [x] Crank-Nicolson PDE solver for European vanilla (sanity-check vs analytical)
- [x] GBM path generator with antithetic variates
- [x] Multi-asset correlated GBM via Cholesky
- [x] Dual Currency Investment (fair coupon solver, payoff scenarios)
- [x] Equity-Linked Note (single-name)
- [x] Worst-of Fixed Coupon Note (Monte Carlo)
- [x] Worst-of Phoenix Autocallable with continuous KI monitoring
- [x] Target Redemption Forward with leveraged loss leg and cumulative-profit KO
- [x] Range Accrual
- [x] Daily share Accumulator with KO
- [x] Quanto and composite options
- [x] Simplified 3-factor PRDC (Hull-White × Hull-White × log-normal FX)
- [x] P&L explain decomposition (Greek attribution + unexplained residual)
- [x] Scenario ladder generator (spot × vol grid)
- [x] Generic bump-and-reval Greek wrapper

**On the roadmap:**

- [ ] Curve bootstrapping from real SOR/SORA/USD-OIS deposit and swap quotes
- [ ] Heston stochastic vol calibration
- [ ] Variance and volatility swap pricing
- [ ] Cliquet structures
- [ ] Snowball / KIKO accumulator variants
- [ ] Longstaff-Schwartz for Bermudan exercise
- [ ] FX vol surface in delta-strike convention with smile interpolation
- [ ] VBA companion module (UDF wrappers + EOD batch pricing)

## Quickstart

```bash
git clone https://github.com/charlotterianes/Cross-Asset-Structuring-Exotic-Hybrids.git
cd Cross-Asset-Structuring-Exotic-Hybrids
pip install -r requirements.txt
python cas_toolkit.py
```

Running the script executes a demo across every product. Sample output:

```
DCI (Dual Currency Investment)
DCI fair coupon (annualised): 5.2606%
Bank PV of embedded short put: 1447.04

WORST-OF PHOENIX AUTOCALLABLE
Autocallable PV: 0.9294 +/- 0.0057

TARF (Target Redemption Forward)
TARF bank PV: 1.8045 +/- 0.0403
% of paths that knocked out: 51.0%
```

## Product index

| Product | Underlying | Decomposition | Key Greeks on bank book |
|---|---|---|---|
| DCI | FX pair | Deposit + short OTM put | Short vega, short gamma, long theta |
| ELN | Single name equity | Bond + short put | Short vega, short gamma |
| FCN | Equity basket | Bond + short worst-of put | Short vega, long correlation, short skew |
| Autocallable | Equity basket | Coupon + autocall option + KI put | Short vega, long correlation, short skew |
| TARF | FX pair | Strip of leveraged forwards + cumulative-profit KO | Short vega, complex path-dependent gamma |
| Range Accrual | FX or rates | Strip of digitals | Short vega of underlying |
| Accumulator | Single name | Daily forwards with KO and leveraged loss leg | Short vega, complex path-dependent |
| Quanto | Cross-asset | Foreign payoff in domestic ccy | Adjusted by quanto correlation |
| PRDC | Rates × FX | Coupon tied to FX, principal in dom ccy | Three-factor: dom rate, for rate, FX |

For each product, term sheet conventions and risk discussion are in `docs/product_termsheets/` (in progress).

## Example usage

### Pricing a DCI

```python
from cas_toolkit import DCI

dci = DCI(
    notional=1_000_000,   # SGD
    spot=1.35,            # USDSGD
    strike=1.32,          # OTM put on USDSGD
    tenor_days=30,
    coupon_rate=0.06,     # 6% annualised, quoted to client
    sigma=0.07,
    r_dom=0.035,          # SGD rate
    r_for=0.045,          # USD rate
)

print(f"Fair coupon: {dci.fair_coupon():.4%}")
print(f"Bank PV (margin): {dci.bank_pv():.2f}")
print("Payoff if SGD strengthens to 1.30:", dci.client_payoff(1.30))
```

### Worst-of phoenix autocallable

```python
import numpy as np
from cas_toolkit import WorstOfAutocallable

ac = WorstOfAutocallable(
    spots=np.array([100, 100]),
    sigmas=np.array([0.25, 0.30]),
    divs=np.array([0.02, 0.02]),
    corr=np.array([[1.0, 0.6], [0.6, 1.0]]),
    r=0.04,
    coupon=0.04,             # 4% per observation date
    coupon_barrier=0.65,     # pay coupon if worst-of >= 65%
    autocall_barrier=1.00,   # auto-redeem if worst-of >= 100%
    ki_barrier=0.60,         # 60% KI for principal protection
    obs_dates_yrs=[0.5, 1.0, 1.5, 2.0],
    maturity_yrs=2.0,
    n_paths=20_000,
)

result = ac.price()
print(f"PV: {result['pv']:.4f} +/- {1.96 * result['se']:.4f}")
```

### P&L explain on a vanilla position

```python
from cas_toolkit import bs_price, bs_greeks, pnl_explain

prev = dict(S=100, K=100, T=1.0, sigma=0.20, r=0.04, q=0.02, opt='call')
today = dict(prev); today.update(S=101.5, sigma=0.21, T=prev['T'] - 1/365)

prev_g = bs_greeks(**{k: prev[k] for k in ['S','K','T','sigma','r','q','opt']})
explain = pnl_explain(
    prev_greeks=prev_g,
    prev_market=dict(S=100, sigma=0.20, r=0.04),
    today_market=dict(S=101.5, sigma=0.21, r=0.04),
    today_pv=bs_price(**today), prev_pv=bs_price(**prev), dt_days=1,
)
# Returns: actual, explained, unexplained, plus delta/gamma/vega/theta/rho/vanna/volga PnL
```

## Methodology notes

**Why Asia weighting.** Asian bank desks see disproportionate flow in DCIs (FX-linked deposits to PB clients), FCNs and autocallables (yield-enhancement on regional indices and single names), and TARFs (corporate FX hedging). Standard textbooks weight toward Western equity exotics: cliquets, lookbacks, Bermudans on US single names. The product mix here reflects what actually trades in Singapore and Hong Kong.

**Why SVI not just SABR.** SVI has a cleaner arbitrage-free parametrisation for equity skews where the smile is asymmetric and fat-tailed. SABR remains the right tool for FX and rates where ATM implied vol moves with the forward (the so-called "sticky delta" behaviour SABR captures naturally). Both are included so the right tool is available per asset class.

**Why bump-and-reval.** Production books often use AAD or pathwise differentiation for performance, but bump-and-reval is the universal fallback that works on any pricer regardless of internal structure. The wrapper here is generic enough to handle any product in this repo.

**Why path-dependent products use continuous KI monitoring.** Phoenix autocallables and similar trades have continuously monitored KI barriers in real term sheets. Approximating with discrete observation dates undervalues the KI risk. The implementation here checks the worst-of ratio at every step.

## File structure

```
cas_toolkit.py                Single-file toolkit (all sections)
requirements.txt              numpy, scipy
README.md                     This file
docs/                         Term sheets and methodology (WIP)
notebooks/                    Walkthroughs and scenario analyses (WIP)
```

The single-file structure is deliberate for the initial commit; modularisation into `src/cas/{vanilla,exotics,hybrids,risk,volsurface,numerics}` is on the roadmap once the API stabilises.

## References

Bouzoubaa, M. and Osseiran, A. (2010). *Exotic Options and Hybrids: A Guide to Structuring, Pricing and Trading*. Wiley.

Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.

Hagan, P., Kumar, D., Lesniewski, A. and Woodward, D. (2002). Managing smile risk. *Wilmott Magazine*.

Andersen, L. and Piterbarg, V. (2010). *Interest Rate Modeling, Volume III: Products and Risk Management*. Atlantic Financial Press.

## Disclaimer

Educational and research code. Not for production use. Models are simplified for clarity; production pricing libraries at investment banks include calibration discipline, day-count conventions, holiday calendars, basis adjustments, and numerical safeguards that this toolkit deliberately abstracts away.
