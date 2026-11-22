import numpy as np
import pytest

from tsminer.stats import (bh_q, eb_shrink, fwer_p, hac_t, holm, norm_sf,
                           shift_null, two_sided_p)


def naive_hac_t(S, y, lags, min_obs):
    T, N = S.shape
    t = np.full(N, np.nan)
    ic = np.full(N, np.nan)
    for j in range(N):
        m = np.isfinite(S[:, j])
        n = m.sum()
        if n < min_obs:
            continue
        s = np.where(m, S[:, j] - S[m, j].mean(), 0.0)
        yy = np.where(m, y - y[m].mean(), 0.0)
        sxx, syy, sxy = (s * s).sum(), (yy * yy).sum(), (s * yy).sum()
        if sxx == 0:
            continue
        e = yy - sxy / sxx * s
        u = s * e
        V = (u * u).sum()
        for l in range(1, lags + 1):
            V += 2 * (1 - l / (lags + 1)) * (u[l:] * u[:-l]).sum()
        if V <= 0:
            continue
        t[j] = sxy / np.sqrt(V)
        ic[j] = sxy / np.sqrt(sxx * syy)
    return t, ic


def ar1(rng, T, N, phi):
    e = rng.standard_normal((T, N))
    x = np.empty((T, N))
    x[0] = e[0]
    for i in range(1, T):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_hac_matches_naive_with_nans():
    rng = np.random.default_rng(1)
    T, N = 600, 12
    S = rng.standard_normal((T, N)).cumsum(0) * 0.1 + rng.standard_normal((T, N))
    S[rng.random((T, N)) < 0.1] = np.nan
    S[:200, 3] = np.nan            # NaN prefix
    S[:, 5] = np.nan               # all-NaN column
    S[:450, 7] = np.nan            # too few obs -> NaN
    y = rng.standard_normal(T) + 0.1 * np.nan_to_num(S[:, 0])
    t, ic, n = hac_t(S, y, 5, min_obs=250)
    tn, icn = naive_hac_t(S, y, 5, 250)
    np.testing.assert_allclose(t, tn, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(ic, icn, rtol=1e-6, equal_nan=True)
    assert np.isnan(t[5]) and np.isnan(t[7]) and np.isnan(t[3]) == (400 < 250)
    np.testing.assert_array_equal(n, np.isfinite(S).sum(0))


def test_hac_rejects_nonfinite_y():
    y = np.ones(300)
    y[5] = np.nan
    with pytest.raises(ValueError):
        hac_t(np.zeros((300, 2)), y, 5)


def test_null_calibration_iid():
    rng = np.random.default_rng(2)
    T, N = 3000, 2000
    t, _, _ = hac_t(rng.standard_normal((T, N)), rng.standard_normal(T), 5)
    assert 0.9 <= t.std() <= 1.1


def test_null_calibration_persistent_signal_vol_clustering():
    rng = np.random.default_rng(3)
    T, N = 3000, 2000
    S = ar1(rng, T, N, 0.98)
    # piecewise volatility regimes
    vol = np.repeat(rng.choice([0.5, 1.0, 2.0], size=T // 100), 100)
    y = rng.standard_normal(T) * vol
    t, _, _ = hac_t(S, y, 5)
    assert 0.9 <= t.std() <= 1.1


def test_fwer_control_pure_noise():
    rng = np.random.default_rng(4)
    T, N, B, R = 500, 40, 199, 150
    hits = 0
    for r in range(R):
        f = ar1(rng, T, 3, 0.7)               # common factors -> correlated, autocorrelated columns
        S = f @ rng.standard_normal((3, N)) + ar1(rng, T, N, 0.5)
        y = rng.standard_normal(T)
        t, _, _ = hac_t(S, y, 5, min_obs=100)
        null = shift_null(S, y, 5, min_obs=100, n_boot=B, min_shift=30, seed=r)["max_abs_t"]
        hits += np.nanmin(fwer_p(t, null)) <= 0.05
    assert 0.01 <= hits / R <= 0.11


def test_planted_signal_detected():
    rng = np.random.default_rng(5)
    T, N = 3000, 300
    y = rng.standard_normal(T)
    S = rng.standard_normal((T, N + 1))
    S[:, -1] = 0.1 * y + np.sqrt(1 - 0.01) * rng.standard_normal(T)
    t, _, _ = hac_t(S, y, 5)
    null = shift_null(S, y, 5, n_boot=300, seed=0)["max_abs_t"]
    p = fwer_p(t, null)
    assert p[-1] <= 0.05
    assert np.argmin(p) == N


def test_shift_null_group_mean_consistent():
    rng = np.random.default_rng(6)
    T, N, B = 400, 9, 20
    S = rng.standard_normal((T, N))
    S[:100, 2] = np.nan
    S[:, 4] = np.nan                      # invalid column excluded from group means
    y = rng.standard_normal(T)
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    res = shift_null(S, y, 3, min_obs=100, n_boot=B, min_shift=20, seed=7, groups=groups)
    assert res["group_mean_t2"].shape == (B, 3)
    # recompute by hand on the same shifts (documented rng contract)
    ks = np.random.default_rng(7).integers(20, T - 20 + 1, size=B)
    for b, k in enumerate(ks):
        t, _ = naive_hac_t(S, np.roll(y, k), 3, 100)
        t2 = t**2
        assert res["max_abs_t"][b] == pytest.approx(np.nanmax(np.abs(t)), rel=1e-6)
        assert res["mean_t2"][b] == pytest.approx(np.nanmean(t2), rel=1e-6)
        for g in range(3):
            assert res["group_mean_t2"][b, g] == pytest.approx(np.nanmean(t2[groups == g]), rel=1e-6)
    # weighted-average identity with valid counts
    valid = np.isfinite(naive_hac_t(S, y, 3, 100)[0])
    w = np.array([valid[groups == g].sum() for g in range(3)])
    np.testing.assert_allclose((res["group_mean_t2"] * w).sum(1) / w.sum(), res["mean_t2"], rtol=1e-9)
    assert shift_null(S, y, 3, min_obs=100, n_boot=5, min_shift=20)["group_mean_t2"] is None


def test_shift_null_bad_args():
    with pytest.raises(ValueError):
        shift_null(np.zeros((100, 2)), np.zeros(100), 3, min_shift=50)


def test_bh_and_holm_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # BH: sorted .005,.01,.03,.04 -> *4/rank = .02,.02,.04,.04
    np.testing.assert_allclose(bh_q(p), [0.02, 0.04, 0.04, 0.02])
    # Holm: sorted *(4,3,2,1) = .02,.03,.06,.04 -> cummax .02,.03,.06,.06
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    pn = np.array([0.01, np.nan, 0.04, 0.03, 0.005])
    qb, qh = bh_q(pn), holm(pn)
    assert np.isnan(qb[1]) and np.isnan(qh[1])
    np.testing.assert_allclose(qb[[0, 2, 3, 4]], bh_q(p))
    np.testing.assert_allclose(qh[[0, 2, 3, 4]], holm(p))
    np.testing.assert_allclose(bh_q(np.array([0.9, 0.8])), [0.9, 0.9])
    np.testing.assert_allclose(holm(np.array([0.6, 0.7])), [1.0, 1.0])


def test_fwer_p_hand_example():
    null = np.array([1.0, 2.0, 3.0, 4.0])
    t = np.array([2.5, -4.0, 0.5, np.nan, 5.0])
    np.testing.assert_allclose(fwer_p(t, null), [3 / 5, 2 / 5, 1.0, np.nan, 1 / 5])
    np.testing.assert_allclose(fwer_p(np.array([2.0]), null), [4 / 5])  # ties count (>=)


def test_normal_helpers():
    np.testing.assert_allclose(norm_sf(np.array([0.0, 1.959964])), [0.5, 0.025], atol=1e-6)
    np.testing.assert_allclose(two_sided_p(np.array([-1.959964, np.nan])), [0.05, np.nan], atol=1e-6, equal_nan=True)


def test_eb_shrink_noise():
    rng = np.random.default_rng(8)
    N = 500
    n_eff = rng.uniform(500, 3000, N)
    t = rng.standard_normal(N)
    ic = t / np.sqrt(n_eff)
    r = eb_shrink(t, ic, 1.0)
    assert r["tau2"] < 1e-4  # a real IC^2 variance of 0.05^2 would give ~2.5e-3
    assert np.nanmax(np.abs(r["post_ic"])) < 0.1 * np.nanmax(np.abs(ic))
    assert np.all(r["shrink"] < 0.05)


def test_eb_shrink_half_signal():
    rng = np.random.default_rng(9)
    N = 400
    n_eff = np.full(N, 2000.0)
    ic_true = np.where(np.arange(N) < N // 2, rng.normal(0, 0.05, N), 0.0)
    ic = ic_true + rng.standard_normal(N) / np.sqrt(n_eff)
    t = ic * np.sqrt(n_eff)
    t[0], ic[0] = np.nan, np.nan
    r = eb_shrink(t, ic, 1.0)
    ok = np.isfinite(t)
    assert r["tau2"] > 0 and np.isnan(r["shrink"][0])
    assert np.all((r["shrink"][ok] > 0) & (r["shrink"][ok] < 1))
    assert np.all(np.sign(r["post_ic"][ok]) == np.sign(ic[ok]))
    assert np.all(np.abs(r["post_ic"][ok]) < np.abs(ic[ok]))
    i_big, i_small = np.nanargmax(np.abs(t)), np.nanargmin(np.abs(t))
    assert abs(r["p_pos"][i_big] - 0.5) > abs(r["p_pos"][i_small] - 0.5)


def test_eb_shrink_tau2_zero():
    r = eb_shrink(np.array([0.1, -0.2]), np.array([0.01, -0.02]), 1.0)
    assert r["tau2"] == 0
    np.testing.assert_allclose(r["post_ic"], 0)
    np.testing.assert_allclose(r["p_pos"], 0.5)
