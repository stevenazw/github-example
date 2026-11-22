"""HAC t-stats and the circular-shift null scale for single-factor tests."""
import math

import numpy as np


class _Pre:
    """Quantities that depend only on S (not on y), restricted to usable columns."""

    def __init__(self, S, min_obs):
        S = np.asarray(S, dtype=float)
        m = np.isfinite(S)
        n = m.sum(axis=0)
        Sz = np.where(m, S, 0.0)
        mean = Sz.sum(axis=0) / np.maximum(n, 1)
        st_all = np.where(m, S - mean, 0.0)
        sxx_all = (st_all * st_all).sum(axis=0)
        self.N = S.shape[1]
        self.n_all = n
        self.cols = np.flatnonzero((n >= min_obs) & (sxx_all > 0))
        self.m = m[:, self.cols].astype(float)
        self.st = np.ascontiguousarray(st_all[:, self.cols])
        self.st2 = self.st * self.st
        self.sxx = sxx_all[self.cols]
        self.n = n[self.cols]


def _core(pre, Y, lags, block=64):
    """sxy and Newey-West V for every column of pre and every column of Y (T, B).

    Returns (sxy, V, ybar), each (B, Nv). Columns are processed in blocks for cache.
    """
    T, B = Y.shape
    Nv = len(pre.cols)
    sxy = np.empty((B, Nv))
    V = np.empty((B, Nv))
    ybar = np.empty((B, Nv))
    w = [1.0 - l / (lags + 1.0) for l in range(1, lags + 1)]
    for c0 in range(0, Nv, block):
        sl = slice(c0, min(c0 + block, Nv))
        st, st2 = pre.st[:, sl], pre.st2[:, sl]
        # sum of valid s~ is 0, so sxy = s~'y; ybar_j = (m'y)/n
        sx = st.T @ Y
        yb = (pre.m[:, sl].T @ Y) / pre.n[sl, None]
        bc = sx / pre.sxx[sl, None]
        sxy[:, sl], ybar[:, sl] = sx.T, yb.T
        for b in range(B):
            # u = s~ * (y - ybar) - b * s~^2 (zero where invalid since s~ = 0)
            u = st * (Y[:, b:b + 1] - yb[:, b]) - bc[:, b] * st2
            v = np.einsum("tn,tn->n", u, u)
            for l, wl in enumerate(w, start=1):
                v += 2.0 * wl * np.einsum("tn,tn->n", u[l:], u[:-l])
            V[b, sl] = v
    return sxy, V, ybar


def hac_t(S, y, lags, min_obs=250):
    """HAC (Newey-West) t of the univariate OLS slope of y on each column of S.

    Returns (t, ic, n), each (N,). NaN in S marks unavailable days.
    """
    S = np.asarray(S, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("y must be finite")
    pre = _Pre(S, min_obs)
    t = np.full(pre.N, np.nan)
    ic = np.full(pre.N, np.nan)
    if len(pre.cols):
        sxy, V, ybar = _core(pre, y[:, None], lags)
        sxy, V, ybar = sxy[0], V[0], ybar[0]
        syy = pre.m.T @ (y * y) - pre.n * ybar**2
        ok = V > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = np.where(ok, sxy / np.sqrt(np.where(ok, V, 1.0)), np.nan)
            ii = np.where(ok, sxy / np.sqrt(pre.sxx * syy), np.nan)
        t[pre.cols], ic[pre.cols] = tt, ii
    return t, ic, pre.n_all.copy()


def null_sd(S, y, lags, min_obs=250, n_boot=1000, min_shift=60, seed=0):
    """Circular-shift null scale of the HAC t of each column of S.

    y is rolled by k ~ U{min_shift..T-min_shift}, t* recomputed for every column, and the sd of t*
    over the n_boot replicates (ddof=0, finite values only) is returned per column, shape (N,).
    Unusable columns (fewer than min_obs finite values, zero variance, no valid null t) are NaN.
    The shifts are rng = default_rng(seed); ks = rng.integers(min_shift, T-min_shift+1, n_boot).
    """
    S = np.asarray(S, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("y must be finite")
    T = len(y)
    if T - 2 * min_shift < 1:
        raise ValueError("T - 2*min_shift must be >= 1")
    rng = np.random.default_rng(seed)
    ks = rng.integers(min_shift, T - min_shift + 1, size=n_boot)
    pre = _Pre(S, min_obs)
    sd_all = np.full(pre.N, np.nan)
    if len(pre.cols):
        Y = y[(np.arange(T)[:, None] - ks[None, :]) % T]  # column b == np.roll(y, ks[b])
        sxy, V, _ = _core(pre, Y, lags)
        ok = V > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ok, sxy / np.sqrt(np.where(ok, V, 1.0)), np.nan)
        fin = np.isfinite(t)
        c = fin.sum(axis=0)
        t0 = np.where(fin, t, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = t0.sum(axis=0) / c
            sd = np.sqrt(np.where(fin, (t - mu) ** 2, 0.0).sum(axis=0) / c)
        sd_all[pre.cols] = np.where((c > 0) & np.isfinite(sd) & (sd > 0), sd, np.nan)
    return sd_all


_erfc = np.vectorize(math.erfc, otypes=[float])


def norm_sf(z):
    """Standard normal upper-tail probability (vectorized)."""
    return 0.5 * _erfc(np.asarray(z, dtype=float) / math.sqrt(2.0))


def two_sided_p(t):
    return 2.0 * norm_sf(np.abs(np.asarray(t, dtype=float)))
