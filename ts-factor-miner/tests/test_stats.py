import numpy as np
import pytest

from tsminer.stats import (bh_q, eb_two_group, fwer_p, hac_t, holm, norm_sf,
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
        res = shift_null(S, y, 5, min_obs=100, n_boot=B, min_shift=30, seed=r)
        hits += np.nanmin(fwer_p(t / res["t_sd"], res["max_abs_z"])) <= 0.05
    assert 0.01 <= hits / R <= 0.11


def test_planted_signal_detected():
    rng = np.random.default_rng(5)
    T, N = 3000, 300
    y = rng.standard_normal(T)
    S = rng.standard_normal((T, N + 1))
    S[:, -1] = 0.1 * y + np.sqrt(1 - 0.01) * rng.standard_normal(T)
    t, _, _ = hac_t(S, y, 5)
    res = shift_null(S, y, 5, n_boot=300, seed=0)
    p = fwer_p(t / res["t_sd"], res["max_abs_z"])
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
    assert res["group_mean_z2"].shape == (B, 3)
    # recompute by hand on the same shifts (documented rng contract)
    ks = np.random.default_rng(7).integers(20, T - 20 + 1, size=B)
    tt = np.array([naive_hac_t(S, np.roll(y, k), 3, 100)[0] for k in ks])      # (B, N)
    fin = np.isfinite(tt)
    sd = np.full(tt.shape[1], np.nan)
    for j in np.flatnonzero(fin.any(axis=0)):
        sd[j] = np.std(tt[fin[:, j], j])
    np.testing.assert_allclose(res["t_sd"], sd, rtol=1e-6)
    assert np.isnan(res["t_sd"][4])
    z2 = (tt / sd) ** 2
    for b in range(B):
        assert res["max_abs_z"][b] == pytest.approx(np.nanmax(np.sqrt(z2[b])), rel=1e-6)
        assert res["mean_z2"][b] == pytest.approx(np.nanmean(z2[b]), rel=1e-6)
        for g in range(3):
            assert res["group_mean_z2"][b, g] == pytest.approx(np.nanmean(z2[b][groups == g]), rel=1e-6)
    # weighted-average identity with valid counts
    valid = np.isfinite(naive_hac_t(S, y, 3, 100)[0])
    w = np.array([valid[groups == g].sum() for g in range(3)])
    np.testing.assert_allclose((res["group_mean_z2"] * w).sum(1) / w.sum(), res["mean_z2"], rtol=1e-9)
    assert shift_null(S, y, 3, min_obs=100, n_boot=5, min_shift=20)["group_mean_z2"] is None


def test_studentized_max_is_balanced_across_heavy_tails():
    rng = np.random.default_rng(8)
    T, half, B = 1500, 20, 400
    normal = rng.standard_normal((T, half))
    spike = rng.standard_normal((T, half)) * (rng.random((T, half)) < 0.05) * 6.0   # ~95% zeros
    S = np.hstack([normal, spike])
    y = rng.standard_normal(T)
    res = shift_null(S, y, 5, min_obs=100, n_boot=B, min_shift=60, seed=1)
    sd = res["t_sd"]
    print("t_sd normal:", sd[:half].mean(), "t_sd heavy:", sd[half:].mean())
    assert np.isfinite(sd).all() and (sd > 0).all()
    # which column attains max|z*| in each replicate (same shifts as the contract)
    ks = np.random.default_rng(1).integers(60, T - 60 + 1, size=B)
    Z = np.array([hac_t(S, np.roll(y, k), 5, 100)[0] for k in ks]) / sd
    share = np.mean(np.argmax(np.abs(Z), axis=1) >= half)
    print("heavy-tail share of argmax |z*|:", share)
    assert 0.3 <= share <= 0.7


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


def _z_ic(z, rng, lo=1000, hi=3000):
    n_eff = rng.uniform(lo, hi, len(z))
    return z, z / np.sqrt(n_eff)


def test_eb_two_group_sparse():
    rng = np.random.default_rng(9)
    N, K = 1000, 20
    z = rng.standard_normal(N)
    z[:K] = rng.choice([-1, 1], K) * 5.0 + rng.standard_normal(K)
    z, ic = _z_ic(z, rng)
    r = eb_two_group(z, ic, 1.0)
    print("sparse: pi0=%.4f tau2=%.3g med p_signal sig=%.3f noise=%.3f sum|post_ic| sig=%.4f noise=%.4f" % (
        r["pi0"], r["tau2"], np.median(r["p_signal"][:K]), np.median(r["p_signal"][K:]),
        np.abs(r["post_ic"][:K]).sum(), np.abs(r["post_ic"][K:]).sum()))
    assert 0.9 <= r["pi0"] < 1.0
    assert np.median(r["p_signal"][:K]) > 0.9
    assert np.median(r["p_signal"][K:]) < 0.1
    assert np.abs(r["post_ic"][K:]).sum() < np.abs(r["post_ic"][:K]).sum()


def test_eb_two_group_pure_noise():
    rng = np.random.default_rng(10)
    z, ic = _z_ic(rng.standard_normal(1000), rng)
    r = eb_two_group(z, ic, 1.0)
    ratio = np.mean(np.abs(r["post_ic"])) / np.mean(np.abs(ic))
    print("noise: pi0=%.4f tau2=%.3g ratio=%.4f p_pos range=(%.3f, %.3f)" % (
        r["pi0"], r["tau2"], ratio, r["p_pos"].min(), r["p_pos"].max()))
    assert ratio < 0.2
    assert np.all((r["p_pos"] > 0.2) & (r["p_pos"] < 0.8))


def test_eb_two_group_sign_and_nan():
    rng = np.random.default_rng(11)
    z = rng.standard_normal(300)
    z[:10] += 5.0
    z, ic = _z_ic(z, rng)
    base = eb_two_group(z, ic, 1.0)
    ok = np.isfinite(z)
    assert np.all(np.sign(base["post_ic"][ok]) == np.sign(ic[ok]))
    z2, ic2 = z.copy(), ic.copy()
    z2[50], ic2[60], ic2[70] = np.nan, np.nan, 0.0
    r = eb_two_group(z2, ic2, 1.0)
    bad = [50, 60, 70]
    for k in ("p_signal", "shrink", "post_ic", "p_pos"):
        assert np.isnan(r[k][bad]).all()
        assert np.isfinite(np.delete(r[k], bad)).all()
    assert np.all(np.sign(np.delete(r["post_ic"], bad)) == np.sign(np.delete(ic2, bad)))
    assert np.isnan(eb_two_group(np.full(3, np.nan), np.ones(3), 1.0)["pi0"])
