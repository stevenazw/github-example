import numpy as np
import pandas as pd
import pytest

from tsminer.transforms import (TRANSFORM_NAMES, _raw_change, allowed_transforms,
                                build_candidates, chg_std_window, forward_target, make_signal)


def make_x(n=700, seed=0, kind="price"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    if kind == "price":
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    return pd.Series(np.cumsum(rng.normal(0, 0.1, n)), index=idx)


@pytest.mark.parametrize("kind", ["price", "level"])
@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test_causality(name, kind):
    x = make_x(kind=kind)
    full = make_signal(x, name, kind)
    for t0 in (300, 450, 599):
        part = make_signal(x.iloc[:t0], name, kind)
        pd.testing.assert_series_equal(part, full.iloc[:t0])


def test_chg_log_for_price_arith_for_level():
    x = make_x(kind="price")
    np.testing.assert_allclose(_raw_change(x, 5, "price").dropna(), (np.log(x) - np.log(x.shift(5))).dropna())
    np.testing.assert_allclose(_raw_change(x, 5, "level").dropna(), (x - x.shift(5)).dropna())
    # the public signal is a scaled version of the right raw change (before clipping)
    sig = make_signal(x, "chg_5", "price", clip=1e9)
    d = _raw_change(x, 5, "price")
    sd = d.rolling(252, min_periods=126).std()
    pd.testing.assert_series_equal(sig, d / sd, check_names=False)
    sig_l = make_signal(x, "chg_5", "level", clip=1e9)
    d2 = x - x.shift(5)
    pd.testing.assert_series_equal(sig_l, d2 / d2.rolling(252, min_periods=126).std(), check_names=False)


def test_price_with_nonpositive_is_nan_pointwise():
    x = make_x(kind="price")
    x.iloc[400] = -1.0
    sig = make_signal(x, "chg_5", "price")  # must not raise
    # the non-positive point and the point 5 rows later (its lag partner) are NaN
    assert np.isnan(sig.iloc[400]) and np.isnan(sig.iloc[405])
    assert np.isfinite(sig.iloc[399]) and np.isfinite(sig.iloc[406])
    assert not np.isinf(sig).any()


def test_price_nonpositive_in_future_does_not_change_past():
    x = make_x(kind="price")
    t0 = 400
    x2 = x.copy()
    x2.iloc[t0 + 20] = -1.0
    for name in TRANSFORM_NAMES:
        a = make_signal(x, name, "price")
        b = make_signal(x2, name, "price")
        pd.testing.assert_series_equal(a.iloc[:t0 + 1], b.iloc[:t0 + 1])


def test_z_window_nan_and_constant():
    x = make_x(kind="level")
    z = make_signal(x, "z_60", "level")
    assert z.iloc[:59].isna().all() and z.iloc[59:].notna().all()
    c = pd.Series(5.0, index=x.index)
    zc = make_signal(c, "z_60", "level")
    assert zc.isna().all() and not np.isinf(zc).any()
    cc = make_signal(c, "chg_5", "level")
    assert cc.isna().all() and not np.isinf(cc).any()


def test_clip():
    x = make_x(kind="level")
    x.iloc[400] += 50.0  # jump
    for name in TRANSFORM_NAMES:
        s = make_signal(x, name, "level", clip=3.0)
        assert s.abs().max() <= 3.0
    assert make_signal(x, "chg_1", "level", clip=3.0).abs().max() == 3.0


def test_bad_inputs():
    x = make_x()
    with pytest.raises(ValueError):
        make_signal(x, "chg_1", "bogus")
    with pytest.raises(ValueError):
        make_signal(x, "foo_3", "price")
    with pytest.raises(ValueError):
        make_signal(x, "nonsense", "price")


def test_allowed_transforms():
    assert allowed_transforms("D") == TRANSFORM_NAMES
    assert TRANSFORM_NAMES == ("chg_1", "chg_5", "chg_20", "chg_60", "chg_252", "z_60", "z_252")
    assert allowed_transforms("W") == ("chg_5", "chg_20", "chg_60", "chg_252", "z_252")
    assert allowed_transforms("M") == ("chg_20", "chg_60", "chg_252", "z_252")
    with pytest.raises(ValueError):
        allowed_transforms("Q")


def test_build_candidates():
    X = pd.DataFrame({"a": make_x(seed=1), "b": make_x(seed=2, kind="level"), "c": make_x(seed=3)})
    meta = pd.DataFrame({"kind": ["price", "level", "level"], "freq": ["D", "W", "M"],
                         "group": ["g1", "g2", "g1"]}, index=["a", "b", "c"])
    S, cm = build_candidates(X, meta)
    n_exp = sum(len(allowed_transforms(f)) for f in meta["freq"])
    assert S.shape == (len(X), n_exp) and len(cm) == n_exp == 16
    assert list(S.columns) == list(cm.index)
    assert "a|chg_1" in S.columns and "b|chg_1" not in S.columns and "c|chg_5" not in S.columns
    row = cm.loc["b|z_252"]
    assert (row["series"], row["transform"], row["kind"], row["freq"], row["group"]) == ("b", "z_252", "level", "W", "g2")
    assert list(cm.columns) == ["series", "transform", "kind", "freq", "group"]
    pd.testing.assert_series_equal(S["a|z_60"], make_signal(X["a"], "z_60", "price"), check_names=False)
    with pytest.raises(KeyError):
        build_candidates(X.assign(d=1.0), meta)


def test_build_candidates_counts_per_freq():
    X = pd.DataFrame({"a": make_x(seed=1), "b": make_x(seed=2), "c": make_x(seed=3)})
    for freq, n in (("D", 7), ("W", 5), ("M", 4)):
        meta = pd.DataFrame({"kind": ["price"] * 3, "freq": [freq] * 3, "group": ["g"] * 3}, index=list(X.columns))
        S, cm = build_candidates(X, meta)
        assert S.shape[1] == len(cm) == 3 * n
        assert (cm["transform"] == "chg_252").sum() == 3


def test_chg_std_window_rule():
    for k in (1, 5, 20, 60):
        assert chg_std_window(k, 252) == 252
    assert chg_std_window(252, 252) == 756
    assert chg_std_window(100, 252) == 300 and chg_std_window(84, 252) == 252
    assert chg_std_window(5, 120) == 120


def _old_signal(x, k, kind, std_window=252, clip=4.0):
    """Reference: the pre-change chg_k (window = std_window regardless of k)."""
    d = _raw_change(x, k, kind)
    sd = d.rolling(std_window, min_periods=std_window // 2).std()
    out = d / sd.where(sd > 0)
    return out.where(np.isfinite(out)).clip(-clip, clip)


@pytest.mark.parametrize("kind", ["price", "level"])
def test_small_k_outputs_unchanged(kind):
    x = make_x(n=900, kind=kind)
    for k in (1, 5, 20, 60):
        pd.testing.assert_series_equal(make_signal(x, f"chg_{k}", kind), _old_signal(x, k, kind), check_names=False)
    for sw in (180, 200, 252):   # 3 * 60 <= std_window: window unchanged
        pd.testing.assert_series_equal(make_signal(x, "chg_60", kind, std_window=sw),
                                       _old_signal(x, 60, kind, std_window=sw), check_names=False)


@pytest.mark.parametrize("kind", ["price", "level"])
def test_chg_252_first_valid_and_window(kind):
    x = make_x(n=1200, kind=kind)
    s = make_signal(x, "chg_252", kind)
    first = int(np.flatnonzero(s.notna().to_numpy())[0])
    assert first == 252 + 378 - 1            # first raw change at 252, then 378 = 756 // 2 values for the std
    assert make_signal(x, "chg_60", kind).first_valid_index() == x.index[60 + 126 - 1]
    d = _raw_change(x, 252, kind)
    sig = make_signal(x, "chg_252", kind, clip=1e9)
    pd.testing.assert_series_equal(sig, d / d.rolling(756, min_periods=378).std(), check_names=False)
    # with the old 252 window the first signal would come far earlier
    assert _old_signal(x, 252, kind).first_valid_index() == x.index[252 + 126 - 1]


def test_chg_252_clip_and_kinds():
    x = make_x(n=1200, kind="level")
    x.iloc[1000:] += 30.0
    s = make_signal(x, "chg_252", "level", clip=3.0)
    assert s.abs().max() == 3.0
    xp = make_x(n=1200, kind="price")
    np.testing.assert_allclose(_raw_change(xp, 252, "price").dropna(), (np.log(xp) - np.log(xp.shift(252))).dropna())


def test_chg_252_longer_window_clips_less_on_trend():
    """Persistent one-sided drift: a 756 window sees the regime, so fewer values pile up at the clip."""
    rng = np.random.default_rng(7)
    n = 2500
    idx = pd.bdate_range("2005-01-01", periods=n)
    x = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, n))), index=idx)
    clip = 4.0
    new = make_signal(x, "chg_252", "price", clip=clip)
    old = _old_signal(x, 252, "price", clip=clip)

    def frac(s):
        v = s.dropna()
        return float((v.abs() >= clip).mean()), len(v)
    f_new, n_new = frac(new)
    f_old, n_old = frac(old)
    print(f"clip fraction: 756-window {f_new:.4f} (n={n_new}), 252-window {f_old:.4f} (n={n_old})")
    assert f_new < f_old


def test_forward_target_no_lookahead():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=400)
    r = pd.Series(rng.normal(0, 0.005, 400), index=idx)
    y, sig = forward_target(r)
    assert np.isnan(y.iloc[-1])
    t = 250
    r2 = r.copy()
    r2.iloc[t + 2:] = rng.normal(0, 0.05, len(r) - t - 2)
    y2, sig2 = forward_target(r2)
    assert y2.iloc[t] == y.iloc[t]
    assert y2.iloc[:t + 1].equals(y.iloc[:t + 1])
    # sigma_t does not depend on r[t+1]
    r3 = r.copy()
    r3.iloc[t + 1] = 0.5
    y3, sig3 = forward_target(r3)
    assert sig3.iloc[t] == sig.iloc[t] and sig3.iloc[:t + 1].equals(sig.iloc[:t + 1])
    ok = y.notna()
    np.testing.assert_allclose((y * sig)[ok], r.shift(-1)[ok], rtol=1e-12)
    assert sig.iloc[:59].isna().all() and y.iloc[:59].isna().all()


def test_forward_target_zero_vol_nan():
    r = pd.Series(0.0, index=pd.bdate_range("2020-01-01", periods=120))
    y, sig = forward_target(r)
    assert y.isna().all()
