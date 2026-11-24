"""P&L engine: daily gross / net P&L of a vol-scaled position, simulated with vectorbt."""
from __future__ import annotations

import numpy as np

REL_TOL = 1e-9          # held value must equal the requested notional to this relative tolerance


def _simulate(n, r_next, cost, init_cash):
    """vectorbt run on the price path 1, (1+r_0), (1+r_0)(1+r_1), ... holding notional ``n_t`` over row t.
    Returns (shares held after the order of each row, fee of each row, price path)."""
    import vectorbt as vbt              # lazy: ~6 s import and ~14 s numba JIT on first use
    price = np.r_[1.0, np.cumprod(1.0 + r_next)]
    pf = vbt.Portfolio.from_orders(price, size=np.r_[n, np.nan], size_type="targetvalue", direction="both",
                                   fees=cost, init_cash=init_cash, freq="1D", raise_reject=True,
                                   allow_partial=False, min_size=0.0)
    rec = pf.orders.records_arr
    shares = np.zeros(len(price))
    fee = np.zeros(len(price))
    np.add.at(shares, rec["idx"], np.where(rec["side"] == 0, rec["size"], -rec["size"]))
    np.add.at(fee, rec["idx"], rec["fees"])
    return np.cumsum(shares), fee, price


def simulate(pos, y_fwd, sigma, cost):
    """Daily (gross, net) P&L; row t is the P&L of the position held on row t.

    Row t holds notional n_t = pos_t / sigma_t (fixed capital of 1, so n may be far above 1) from the close of
    row t to the next close and earns n_t * r_t with r_t = y_fwd_t * sigma_t; every trade pays ``cost`` x traded
    value. vectorbt charges that on the true traded value |n_t - n_{t-1}(1 + r_{t-1})| (it includes the price drift
    of the old holding), not |n_t - n_{t-1}|. TargetValue is used, not TargetPercent: the latter caps long
    exposure at 1x equity. Raises if any order is rejected / partially filled or the held notional is off.
    """
    pos, y, sg = (np.asarray(a, dtype=float) for a in (pos, y_fwd, sigma))
    if not (np.isfinite(pos).all() and np.isfinite(y).all() and np.isfinite(sg).all() and (sg > 0).all()):
        raise ValueError("engine.simulate: pos, y_fwd and sigma must be finite and sigma > 0")
    if len(pos) == 0:
        return np.empty(0), np.empty(0)
    n, r = pos / sg, y * sg
    if (r <= -1).any():
        raise ValueError("engine.simulate: next-day return <= -100%")
    # cash only has to fund purchases: bound all cumulative traded value, so it never binds
    shares, fee, price = _simulate(n, r, cost, init_cash=10.0 * (np.abs(n).sum() * (1 + np.abs(r).max()) + 1.0))
    if not np.allclose(shares[:-1] * price[:-1], n, rtol=REL_TOL, atol=REL_TOL * (1.0 + np.abs(n).max())):
        raise RuntimeError("engine.simulate: held notional differs from the requested one (order clipped)")
    gross = shares[:-1] * np.diff(price)
    return gross, gross - fee[:-1]
