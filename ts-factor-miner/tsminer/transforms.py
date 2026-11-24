"""Causal signal transforms, candidate construction and forward target."""
import numpy as np
import pandas as pd

TRANSFORM_NAMES = ("chg_1", "chg_5", "chg_20", "chg_60", "chg_252", "z_60", "z_252")

_ALLOWED = {
    "D": TRANSFORM_NAMES,
    "W": ("chg_5", "chg_20", "chg_60", "chg_252", "z_252"),
    "M": ("chg_20", "chg_60", "chg_252", "z_252"),
}


def allowed_transforms(freq):
    """Transforms meaningful for a series of the given native frequency."""
    if freq not in _ALLOWED:
        raise ValueError(f"unknown freq: {freq!r}")
    return _ALLOWED[freq]


def _raw_change(x, k, kind):
    """Unstandardized k-day change: log diff for prices (NaN where x_t or x_{t-k} <= 0), else arithmetic diff."""
    if kind == "price":
        lx = np.log(x.where(x > 0))
        return lx - lx.shift(k)
    return x - x.shift(k)


def chg_std_window(k, std_window=252):
    """Window of the rolling std that standardizes a k-day change: at least 3 non-overlapping k-day spans.

    Adjacent k-day changes overlap heavily, so a window shorter than ~3k gives an unstable std.
    """
    return max(int(std_window), 3 * int(k))


def _finish(z, clip):
    z = z.where(np.isfinite(z))
    return z.clip(-clip, clip)


def make_signal(x, name, kind, std_window=252, clip=4.0):
    """Causal standardized signal from series x; kind in {"price", "level"}."""
    if kind not in ("price", "level"):
        raise ValueError(f"unknown kind: {kind!r}")
    prefix, _, num = name.partition("_")
    if prefix not in ("chg", "z") or not num.isdigit() or int(num) < 1:
        raise ValueError(f"unknown transform: {name!r}")
    k = int(num)
    if prefix == "chg":
        d = _raw_change(x, k, kind)
        w = chg_std_window(k, std_window)
        sd = d.rolling(w, min_periods=w // 2).std()
        out = d / sd.where(sd > 0)
    else:
        r = x.rolling(k, min_periods=k)
        sd = r.std()
        out = (x - r.mean()) / sd.where(sd > 0)
    return _finish(out, clip)


def build_candidates(X, meta, std_window=252, clip=4.0):
    """Expand each raw series into its allowed transforms. Returns (S, cand_meta)."""
    missing = [c for c in X.columns if c not in meta.index]
    if missing:
        raise KeyError(f"series missing from meta: {missing}")
    cols, rows = [], []
    for sid in X.columns:
        kind, freq, group = meta.at[sid, "kind"], meta.at[sid, "freq"], meta.at[sid, "group"]
        for tr in allowed_transforms(freq):
            cid = f"{sid}|{tr}"
            cols.append(make_signal(X[sid], tr, kind, std_window, clip).rename(cid))
            rows.append((cid, sid, tr, kind, freq, group))
    S = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=X.index)
    cand_meta = pd.DataFrame(rows, columns=["id", "series", "transform", "kind", "freq", "group"]).set_index("id")
    return S, cand_meta


def forward_target(r, vol_halflife=20, min_periods=60):
    """y_fwd[t] = r[t+1] / sigma_t with sigma_t from data up to t. Returns (y_fwd, sigma)."""
    sigma = r.ewm(halflife=vol_halflife, min_periods=min_periods).std()
    y = r.shift(-1) / sigma.where(sigma > 0)
    return y, sigma
