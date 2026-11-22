import numpy as np
import pandas as pd
import pytest

from tsminer.transforms import (TRANSFORM_NAMES, _raw_change, allowed_transforms,
                                build_candidates, forward_target, make_signal)


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


def test_price_with_nonpositive_falls_back_to_arith():
    x = make_x(kind="level")
    x.iloc[10] = -1.0
    sig = make_signal(x, "chg_5", "price")
    pd.testing.assert_series_equal(sig, make_signal(x, "chg_5", "level"))
    assert np.isfinite(sig.dropna()).all()


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
    assert allowed_transforms("W") == ("chg_5", "chg_20", "chg_60", "z_252")
    assert allowed_transforms("M") == ("chg_20", "chg_60", "z_252")
    with pytest.raises(ValueError):
        allowed_transforms("Q")


def test_build_candidates():
    X = pd.DataFrame({"a": make_x(seed=1), "b": make_x(seed=2, kind="level"), "c": make_x(seed=3)})
    meta = pd.DataFrame({"kind": ["price", "level", "level"], "freq": ["D", "W", "M"],
                         "group": ["g1", "g2", "g1"]}, index=["a", "b", "c"])
    S, cm = build_candidates(X, meta)
    n_exp = sum(len(allowed_transforms(f)) for f in meta["freq"])
    assert S.shape == (len(X), n_exp) and len(cm) == n_exp == 13
    assert list(S.columns) == list(cm.index)
    assert "a|chg_1" in S.columns and "b|chg_1" not in S.columns and "c|chg_5" not in S.columns
    row = cm.loc["b|z_252"]
    assert (row["series"], row["transform"], row["kind"], row["freq"], row["group"]) == ("b", "z_252", "level", "W", "g2")
    assert list(cm.columns) == ["series", "transform", "kind", "freq", "group"]
    pd.testing.assert_series_equal(S["a|z_60"], make_signal(X["a"], "z_60", "price"), check_names=False)
    with pytest.raises(KeyError):
        build_candidates(X.assign(d=1.0), meta)


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
