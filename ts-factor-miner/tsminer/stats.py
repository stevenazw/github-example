"""HAC t-stats, circular-shift null for max|t|, multiple-testing and EB helpers."""
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


def shift_null(S, y, lags, min_obs=250, n_boot=1000, min_shift=60, seed=0, groups=None):
    """Circular-shift null: y is rolled by k ~ U{min_shift..T-min_shift}, t* recomputed.

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
    Y = y[(np.arange(T)[:, None] - ks[None, :]) % T]  # column b == np.roll(y, ks[b])
    K = None if groups is None else int(np.max(groups)) + 1
    max_abs = np.full(n_boot, np.nan)
    mean_t2 = np.full(n_boot, np.nan)
    gm = None if groups is None else np.full((n_boot, K), np.nan)
    t_var = float("nan")
    if len(pre.cols):
        sxy, V, _ = _core(pre, Y, lags)
        ok = V > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ok, sxy / np.sqrt(np.where(ok, V, 1.0)), np.nan)
        fin = np.isfinite(t)
        t2 = np.where(fin, t * t, 0.0)
        cnt = fin.sum(axis=1)
        has = cnt > 0
        max_abs[has] = np.sqrt(t2.max(axis=1))[has]
        mean_t2[has] = (t2.sum(axis=1) / np.maximum(cnt, 1))[has]
        if groups is not None:
            G = np.zeros((len(pre.cols), K))
            G[np.arange(len(pre.cols)), np.asarray(groups)[pre.cols]] = 1.0
            gc = fin.astype(float) @ G
            with np.errstate(divide="ignore", invalid="ignore"):
                gm = np.where(gc > 0, (t2 @ G) / gc, np.nan)
        if fin.any():
            t_var = float(np.var(t[fin]))
    return {"max_abs_t": max_abs, "mean_t2": mean_t2, "group_mean_t2": gm, "t_var": t_var}


def fwer_p(t_obs, max_abs_t_null):
    """Single-step maxT p-values: (1 + #{null >= |t|}) / (B + 1); NaN stays NaN."""
    t = np.asarray(t_obs, dtype=float)
    null = np.sort(np.asarray(max_abs_t_null, dtype=float))
    B = len(null)
    a = np.abs(t)
    cnt = B - np.searchsorted(null, np.where(np.isfinite(a), a, 0.0), side="left")
    return np.where(np.isfinite(a), (1.0 + cnt) / (B + 1.0), np.nan)


_erfc = np.vectorize(math.erfc, otypes=[float])


def norm_sf(z):
    """Standard normal upper-tail probability (vectorized)."""
    return 0.5 * _erfc(np.asarray(z, dtype=float) / math.sqrt(2.0))


def two_sided_p(t):
    return 2.0 * norm_sf(np.abs(np.asarray(t, dtype=float)))


def bh_q(p):
    """Benjamini-Hochberg adjusted p-values; NaN ignored and preserved."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    m = len(idx)
    if m == 0:
        return q
    order = idx[np.argsort(p[idx], kind="stable")]
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.minimum(adj, 1.0)
    return q


def holm(p):
    """Holm step-down adjusted p-values; NaN ignored and preserved."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    adj = p[order] * (m - np.arange(m))
    out[order] = np.minimum(np.maximum.accumulate(adj), 1.0)
    return out


def eb_shrink(t, ic, v0):
    """Empirical-Bayes shrinkage of IC given t-stats; v0 = null E[t^2] (~1)."""
    t = np.asarray(t, dtype=float)
    ic = np.asarray(ic, dtype=float)
    ok = np.isfinite(t) & np.isfinite(ic) & (ic != 0)
    shrink = np.full(t.shape, np.nan)
    post_ic = np.full(t.shape, np.nan)
    p_pos = np.full(t.shape, np.nan)
    if not ok.any():
        return {"tau2": float("nan"), "shrink": shrink, "post_ic": post_ic, "p_pos": p_pos}
    tv, iv = t[ok], ic[ok]
    n_eff = (tv / iv) ** 2
    tau2 = max(0.0, float(np.mean(tv**2)) - v0) / float(np.mean(n_eff))
    sh = n_eff * tau2 / (v0 + n_eff * tau2)
    shrink[ok] = sh
    post_ic[ok] = sh * iv
    p_pos[ok] = 1.0 - norm_sf(tv * np.sqrt(sh / v0))
    return {"tau2": float(tau2), "shrink": shrink, "post_ic": post_ic, "p_pos": p_pos}
