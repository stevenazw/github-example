import numpy as np
import pytest

from tsminer.stats import hac_t, norm_sf, null_sd, two_sided_p


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


def test_null_sd_pure_noise_near_one():
    rng = np.random.default_rng(4)
    T, N = 1500, 60
    S = rng.standard_normal((T, N))
    y = rng.standard_normal(T)
    sd = null_sd(S, y, 5, min_obs=250, n_boot=400, min_shift=60, seed=1)
    assert sd.shape == (N,) and np.isfinite(sd).all()
    assert 0.9 <= sd.mean() <= 1.1


def test_null_sd_heavy_tailed_sparse_column():
    rng = np.random.default_rng(8)
    T = 1500
    normal = rng.standard_normal((T, 3))
    spike = rng.standard_normal((T, 3)) * (rng.random((T, 3)) < 0.05) * 6.0   # ~95% zeros
    y = rng.standard_normal(T)
    sd = null_sd(np.hstack([normal, spike]), y, 5, min_obs=100, n_boot=300, min_shift=60, seed=1)
    assert np.isfinite(sd).all() and (sd > 0).all()


def test_null_sd_matches_naive_and_marks_unusable():
    rng = np.random.default_rng(6)
    T, N, B = 400, 4, 20
    S = rng.standard_normal((T, N))
    S[:, 2] = np.nan                       # unusable
    S[:350, 3] = np.nan                    # too few obs
    y = rng.standard_normal(T)
    sd = null_sd(S, y, 3, min_obs=100, n_boot=B, min_shift=20, seed=7)
    ks = np.random.default_rng(7).integers(20, T - 20 + 1, size=B)
    tt = np.array([naive_hac_t(S, np.roll(y, k), 3, 100)[0] for k in ks])
    for j in (0, 1):
        assert sd[j] == pytest.approx(np.std(tt[:, j]), rel=1e-6)
    assert np.isnan(sd[2]) and np.isnan(sd[3])


def test_null_sd_reproducible_same_seed():
    rng = np.random.default_rng(9)
    S, y = rng.standard_normal((500, 5)), rng.standard_normal(500)
    a = null_sd(S, y, 5, min_obs=100, n_boot=50, min_shift=30, seed=3)
    b = null_sd(S, y, 5, min_obs=100, n_boot=50, min_shift=30, seed=3)
    c = null_sd(S, y, 5, min_obs=100, n_boot=50, min_shift=30, seed=4)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_null_sd_too_short_raises():
    with pytest.raises(ValueError):
        null_sd(np.zeros((100, 2)), np.zeros(100), 3, min_shift=50)


def test_normal_helpers():
    np.testing.assert_allclose(norm_sf(np.array([0.0, 1.959964])), [0.5, 0.025], atol=1e-6)
    np.testing.assert_allclose(two_sided_p(np.array([-1.959964, np.nan])), [0.05, np.nan], atol=1e-6, equal_nan=True)
