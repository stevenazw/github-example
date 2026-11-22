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

    Each candidate's t* is studentized by its own null sd (sd_j = std of t*[:, j] over the same
    B replicates, ddof=0): z* = t* / sd_j, so heavy-tailed candidates get no extra chance at the
    max. Summaries (max_abs_z, mean_z2, group_mean_z2) are on z*; t_sd (length N, NaN for unusable
    columns) is returned so observed z = t / t_sd; t_var is the pooled variance of raw t* (diagnostic).

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
    mean_z2 = np.full(n_boot, np.nan)
    gm = None if groups is None else np.full((n_boot, K), np.nan)
    t_var = float("nan")
    t_sd = np.full(S.shape[1], np.nan)
    if len(pre.cols):
        sxy, V, _ = _core(pre, Y, lags)
        ok = V > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ok, sxy / np.sqrt(np.where(ok, V, 1.0)), np.nan)
        fin_t = np.isfinite(t)
        if fin_t.any():
            t_var = float(np.var(t[fin_t]))
        c = fin_t.sum(axis=0)
        t0 = np.where(fin_t, t, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = t0.sum(axis=0) / c
            sd = np.sqrt((np.where(fin_t, (t - mu) ** 2, 0.0)).sum(axis=0) / c)
        sd = np.where((c > 0) & np.isfinite(sd) & (sd > 0), sd, np.nan)
        t_sd[pre.cols] = sd
        with np.errstate(divide="ignore", invalid="ignore"):
            z = t / sd
        fin = np.isfinite(z)
        z2 = np.where(fin, z * z, 0.0)
        cnt = fin.sum(axis=1)
        has = cnt > 0
        max_abs[has] = np.sqrt(z2.max(axis=1))[has]
        mean_z2[has] = (z2.sum(axis=1) / np.maximum(cnt, 1))[has]
        if groups is not None:
            G = np.zeros((len(pre.cols), K))
            G[np.arange(len(pre.cols)), np.asarray(groups)[pre.cols]] = 1.0
            gc = fin.astype(float) @ G
            with np.errstate(divide="ignore", invalid="ignore"):
                gm = np.where(gc > 0, (z2 @ G) / gc, np.nan)
    return {"max_abs_z": max_abs, "mean_z2": mean_z2, "group_mean_z2": gm, "t_sd": t_sd, "t_var": t_var}


def fwer_p(t_obs, max_abs_t_null):
    """Single-step maxT p-values (pass studentized z and the max_abs_z null): (1 + #{null >= |t|}) / (B + 1); NaN stays NaN."""
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


def eb_two_group(z, ic, v0, n_iter=200, tol=1e-8):
    """Two-groups empirical Bayes: z_j ~ pi0 N(0, v0) + (1-pi0) N(0, v0 + n_j tau2), n_j = (z_j/ic_j)^2.

    Fitted by EM on columns with finite z and finite non-zero ic; other outputs are NaN.
    p_signal = posterior P(non-null); post_ic = p_signal * shrink * ic;
    p_pos = p_signal * Phi(z sqrt(shrink / v0)) + (1 - p_signal) * 0.5.
    tau2 == 0 means the data support no non-null candidate (pi0 is then unidentified): the model
    degenerates to all-null, reported as pi0 = 1, p_signal = shrink = post_ic = 0, p_pos = 0.5.
    """
    z = np.asarray(z, dtype=float)
    ic = np.asarray(ic, dtype=float)
    ok = np.isfinite(z) & np.isfinite(ic) & (ic != 0)
    out = {k: np.full(z.shape, np.nan) for k in ("p_signal", "shrink", "post_ic", "p_pos")}
    out.update(pi0=float("nan"), tau2=float("nan"))
    if not ok.any():
        return out
    zv, iv = z[ok], ic[ok]
    n = (zv / iv) ** 2
    z2 = zv * zv
    pi0 = 0.9
    tau2 = max(float(np.mean(z2)) - v0, 0.05) / ((1.0 - pi0) * float(np.mean(n)))

    def logphi(var):
        return -0.5 * (np.log(2.0 * np.pi * var) + z2 / var)

    def resp(pi0, tau2):
        a = np.log(pi0) + logphi(v0)
        b = np.log(max(1.0 - pi0, 1e-300)) + logphi(v0 + n * tau2)
        mx = np.maximum(a, b)
        return np.exp(b - mx) / (np.exp(a - mx) + np.exp(b - mx))

    for _ in range(n_iter):
        r = resp(pi0, tau2)
        pi_new = 1.0 - float(np.mean(r))
        pi_new = min(max(pi_new, 1e-12), 1.0 - 1e-12)
        den = float((r * n).sum())
        tau_new = max(0.0, float((r * (z2 - v0)).sum())) / den if den > 0 else 0.0
        done = abs(pi_new - pi0) < tol and abs(tau_new - tau2) < tol
        pi0, tau2 = pi_new, tau_new
        if done:
            break
    if not tau2 > 0:
        out["p_signal"][ok] = 0.0
        out["shrink"][ok] = 0.0
        out["post_ic"][ok] = 0.0
        out["p_pos"][ok] = 0.5
        out["pi0"], out["tau2"] = 1.0, 0.0
        return out
    r = resp(pi0, tau2)
    sh = n * tau2 / (v0 + n * tau2)
    out["p_signal"][ok] = r
    out["shrink"][ok] = sh
    out["post_ic"][ok] = r * sh * iv
    out["p_pos"][ok] = r * (1.0 - norm_sf(zv * np.sqrt(sh / v0))) + (1.0 - r) * 0.5
    out["pi0"], out["tau2"] = float(pi0), float(tau2)
    return out
