import json
import shutil

import numpy as np
import pandas as pd
import pytest

from tsminer.config import PortfolioConfig, load_config, load_portfolio_config
from tsminer.confirm import confirm
from tsminer.databank import Panel
from tsminer.engine import simulate
from tsminer.pipeline import mine, positions, prepare
from tsminer import portfolio as pf
from tsminer.portfolio import (LAG, applied_states, compute_target, factor_positions, load_target, master_calendar,
                               place_factor, portfolio_position, raw_states, state_cumic, state_hyst, state_slope,
                               summarize)
from tsminer.site import _load_run, build_asset_data

from test_pipeline import TOML, make_panel

SMALL = PortfolioConfig(ic_window=60, slope_lookback=20, cumic_ma=30)       # short windows: small hand-made cases
ALL = ("always", "slope", "hyst", "cumic")


# ------------------------------------------------------------------ fixtures (same pattern as test_site)
@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("pfcfg") / "test.toml"
    p.write_text(TOML + "\n[verdict]\nz_weak = 2.0\n")
    return p


@pytest.fixture(scope="module")
def cfg(cfg_path):
    return load_config(cfg_path)


@pytest.fixture(scope="module")
def panel():
    return make_panel(1, plant=True, short=True, dup=True)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, cfg, cfg_path, panel):
    d = tmp_path_factory.mktemp("pfrun") / "run"
    mine(cfg, d, panel=panel, config_path=cfg_path)
    assert len(pd.read_csv(d / "shortlist.csv")) >= 2
    return d


@pytest.fixture(scope="module")
def opened_dir(tmp_path_factory, run_dir):
    d = tmp_path_factory.mktemp("pfopen") / "run"
    shutil.copytree(run_dir, d)
    confirm(d)
    return d


@pytest.fixture(scope="module")
def noisy_run(tmp_path_factory, cfg, cfg_path):
    """Another run of the same target (same target series), unopened."""
    base = tmp_path_factory.mktemp("pfnoisy")
    pc = make_panel(1, plant=True, short=True)
    rng = np.random.default_rng(99)
    pc.X["sig"] = pc.X["sig"] + np.cumsum(rng.normal(0, 0.6, len(pc.X)))
    pc.X = pc.X.rename(columns={"sig": "noisy"})
    pc.meta = pc.meta.rename(index={"sig": "noisy"})
    pc.meta.loc["noisy", ["ticker", "description"]] = ["noisy", "desc noisy"]
    mine(cfg, base / "c", panel=pc, config_path=cfg_path)
    return base / "c"


def _td(S, signs, y, sigma=None, cost=1e-4, buffer=0.5, elig_oos=None, elig_full=None):
    """Hand-made loaded target: ``S`` list of signals on the rows, ``y`` the forward target."""
    R = len(y)
    m = {"dates": pd.bdate_range("2015-01-01", periods=R), "y": np.asarray(y, float),
         "sigma": np.full(R, 0.005) if sigma is None else sigma, "cost": cost, "buffer": buffer}
    fs = []
    for i, (s, g) in enumerate(zip(S, signs)):
        s = np.asarray(s, float)
        ef = np.isfinite(s) if elig_full is None else elig_full[i]
        eo = np.zeros(R, bool) if elig_oos is None else elig_oos[i]
        fs.append({"id": f"f{i}", "title": f"t{i}", "series": f"x{i}", "transform": "chg_1", "group": "g",
                   "sign": float(g), "run": "r", "opened": True, "s": s, "elig_full": ef, "elig_oos": eo})
    return {"target": "T", "slug": "t", "m": m, "factors": fs}


def _planted(seed, after="noise", n=4000, a=1200, b=2200, ic=0.3):
    rng = np.random.default_rng(seed)
    s, e = rng.normal(size=n), rng.normal(size=n)
    k = ic / np.sqrt(1 - ic**2)
    y = e.copy()
    y[a:b] += k * s[a:b]                                  # the factor predicts only on rows a..b-1
    if after == "rev":
        y[b:] -= k * s[b:]                                # ... and works against its direction afterwards
    return s, y


# ------------------------------------------------------------------ 1. no look-ahead in the state
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_state_on_row_t_never_uses_rows_from_t_minus_1(sign):
    s, y = _planted(5, n=1500, a=500, b=1000)
    s = s.copy()
    s[:30] = np.nan                                       # a late start and a hole in the factor's own rows
    s[700:712] = np.nan
    base = applied_states(s, y, sign, PortfolioConfig())
    rng = np.random.default_rng(1)
    for t in (330, 400, 520, 716, 800, 1100, 1400):
        s2, y2 = s.copy(), y.copy()
        fin = np.isfinite(s2)
        fin[:t - 1] = False                               # values may change from row t-1 on; finiteness is kept
        s2[fin] = rng.normal(size=int(fin.sum())) * 10
        y2[t - 1:] = rng.normal(size=len(y) - t + 1) * 10
        alt = applied_states(s2, y2, sign, PortfolioConfig())
        for m in ALL:
            assert np.array_equal(alt[m][:t + 1], base[m][:t + 1]), (m, t)      # the state applied on row <= t
        assert any(not np.array_equal(alt[m][t + 1:], base[m][t + 1:]) for m in ("slope", "hyst", "cumic"))


def test_state_ignores_later_finiteness_changes():
    """The signal itself disappearing on rows >= t-1 changes nothing for the states applied on rows <= t."""
    s, y = _planted(6, n=1200, a=300, b=800)
    base = applied_states(s, y, 1.0, PortfolioConfig())
    t = 600
    s2 = s.copy()
    s2[t - 1:] = np.nan
    alt = applied_states(s2, y, 1.0, PortfolioConfig())
    for m in ("slope", "hyst", "cumic"):
        assert np.array_equal(alt[m][:t + 1], base[m][:t + 1])


def test_lag_is_two_rows_and_always_is_not_shifted():
    s, y = _planted(7, n=800, a=100, b=700)
    raw = raw_states(s, y, 1.0, SMALL)
    app = applied_states(s, y, 1.0, SMALL)
    assert LAG == 2
    for m in ("slope", "hyst", "cumic"):
        assert raw[m].any()
        np.testing.assert_array_equal(app[m][2:], raw[m][:-2])
        assert not app[m][:2].any()
    assert app["always"].all() and raw["always"].all()


# ------------------------------------------------------------------ 2. method semantics on hand-made series
def test_slope():
    ric = np.array([0.1, 0.2, 0.3, 0.1, 0.5, 0.3, 0.3])
    # L = 2, valid iff higher than 2 rows earlier: 0.3>0.1, 0.1<0.2, 0.5>0.3, 0.3>0.1, 0.3<0.5
    np.testing.assert_array_equal(state_slope(ric, 2), [0, 0, 1, 0, 1, 1, 0])
    assert not state_slope(np.array([0.1, 0.1, 0.1, 0.1]), 2).any()              # equal is not higher
    nan = np.array([np.nan, 0.1, 0.2, 0.3, np.nan, 0.5])
    np.testing.assert_array_equal(state_slope(nan, 1), [0, 0, 1, 1, 0, 0])      # NaN on either side: invalid
    assert not state_slope(np.arange(3.0), 3).any()                              # not enough rows for L


def test_hyst_transitions():
    L = 2
    # invalid start; enter needs ric>0 and rising; leave needs ric<0 and falling; otherwise keep
    seq = np.array([-0.5, -0.4, 0.1, -0.1, -0.3, -0.2])
    #  k2: 0.1 > 0 and > -0.5 -> valid (enter)
    #  k3: -0.1 is negative but RISING vs k1 (-0.4) -> stays valid
    #  k4: -0.3 negative and falling vs k2 (0.1) -> invalid (leave)
    #  k5: -0.2 negative (and rising) while invalid -> stays invalid
    np.testing.assert_array_equal(state_hyst(seq, L), [0, 0, 1, 1, 0, 0])
    seq = np.array([0.3, 0.2, 0.1, 0.4, 0.05, 0.02])
    #  k2: 0.1 positive but FALLING vs k0 (0.3) while invalid -> stays invalid
    #  k3: 0.4 positive and rising vs k1 -> valid
    #  k4/k5: positive but falling while valid -> stays valid
    np.testing.assert_array_equal(state_hyst(seq, L), [0, 0, 0, 1, 1, 1])
    # the eight (state, level, change) cases, L = 2: ric = [a0, a1, a2, x]; the state at k2 is set up by a0..a2 and
    # the final row compares x with a1
    table = [([0, 0, 0, 0.2], 0, 1),           # invalid, positive and rising -> enters
             ([0.5, 0.3, 0.0, 0.2], 0, 0),     # invalid, positive but falling (0.2 < 0.3) -> stays invalid
             ([0, -0.3, 0, -0.1], 0, 0),       # invalid, negative but rising -> stays invalid
             ([0, -0.3, 0, -0.5], 0, 0),       # invalid, negative and falling -> stays invalid
             ([0, 0, 0.1, 0.3], 1, 1),         # valid, positive and rising -> stays valid
             ([0, 0.5, 0.1, 0.2], 1, 1),       # valid, positive but falling (0.2 < 0.5) -> stays valid
             ([-0.5, -0.4, 0.1, -0.1], 1, 1),  # valid, negative but rising (-0.1 > -0.4) -> stays valid
             ([-0.5, 0.0, 0.1, -0.3], 1, 0)]   # valid, negative and falling -> leaves
    for ric, k2, want in table:
        st = state_hyst(np.array(ric), 2)
        assert st.tolist() == [False, False, bool(k2), bool(want)], ric
    # NaN forces invalid and resets; the machine starts again from invalid
    np.testing.assert_array_equal(state_hyst(np.array([0.1, 0.2, np.nan, 0.3, 0.4]), 1), [0, 1, 0, 0, 1])
    assert not state_hyst(np.array([0.5, 0.6, 0.7]), 5).any()                   # warm-up: nothing before k = L


def test_cumic():
    c = np.array([1, 1, 1, -1, -1, -1, 3.0])
    #  C = 1 2 3 2 1 0 3 ; 3-row moving average from k=2: 2, 2.33, 2, 1, 1.33
    np.testing.assert_array_equal(state_cumic(c, 3), [0, 0, 1, 0, 0, 0, 1])
    assert not state_cumic(np.zeros(10), 3).any()                                # on the average is not above it
    assert not state_cumic(np.array([1.0, 2.0]), 3).any()                        # full window required
    assert state_cumic(np.array([1.0, 2.0, 3.0]), 3).tolist() == [False, False, True]
    assert not state_cumic(np.array([1.0, np.nan, 1.0, 1.0]), 2).any()          # NaN poisons the curve: invalid


def test_warmup_nan_and_always_in_applied_states():
    s, y = _planted(3, n=700, a=0, b=700)
    pc = PortfolioConfig()
    s = s.copy()
    s[:50] = np.nan
    s[400:410] = np.nan
    app = applied_states(s, y, 1.0, pc)
    assert app["always"].tolist() == np.isfinite(s).tolist()                    # valid wherever there is a signal
    assert not app["slope"][:50 + pc.ic_window + pc.slope_lookback].any()        # rolling IC and its lag not ready
    assert not app["hyst"][:50 + pc.ic_window + pc.slope_lookback].any()
    assert not app["cumic"][:50 + pc.cumic_ma].any()
    assert app["cumic"].any() and app["slope"].any()
    # rows where the factor has no signal are never valid either way (the portfolio also requires a signal)
    for m in ("slope", "hyst", "cumic"):
        assert not app[m][:LAG + 50].any()


def test_direction_sign_matters():
    s, y = _planted(8, n=1500, a=0, b=1500)
    right, wrong = applied_states(s, y, 1.0, PortfolioConfig()), applied_states(s, y, -1.0, PortfolioConfig())
    for m in ("hyst", "cumic"):
        assert right[m][600:].mean() > 0.95 and wrong[m][600:].mean() < 0.05


# ------------------------------------------------------------------ 3. planted and noise factors
@pytest.mark.parametrize("after", ["noise", "rev"])
def test_planted_factor_valid_during_its_stretch(after):
    """IC 0.3 on rows 1200..2199 only. After the full-window lag the hysteresis machine and the cumulative-IC
    machine are valid; well after it stops (rolling window flushed: > 2200 + 252 + margin) a factor that now
    works against its direction is invalid, while a pure-noise continuation is a coin flip (a noise factor is
    valid about half the time by symmetry, so 'mostly invalid' can only be demanded of the reversed case)."""
    during, post = {"hyst": [], "cumic": []}, {"hyst": [], "cumic": []}
    for seed in range(10):
        s, y = _planted(seed, after)
        st = applied_states(s, y, 1.0, PortfolioConfig())
        for m in during:
            during[m].append(st[m][1500:2200].mean())
            post[m].append(st[m][2200 + 252 + 100:].mean())
    for m in during:
        assert np.mean(during[m]) > 0.95 and np.min(during[m]) > 0.9
        if after == "rev":
            assert np.max(post[m]) < 0.1
        else:
            # 10 seeds x ~1450 rows of a slowly moving state: effective n ~ 10 x 10; sd of the mean ~ 0.05
            assert 0.3 < np.mean(post[m]) < 0.65 and np.mean(post[m]) < np.mean(during[m]) - 0.3


def test_noise_factor_is_valid_about_half_the_time_under_slope():
    share = []
    for seed in range(10):
        s, y = _planted(seed, n=4000, a=0, b=0)                                 # no relation at all
        st = applied_states(s, y, 1.0, PortfolioConfig())["slope"]
        share.append(st[PortfolioConfig().ic_window + PortfolioConfig().slope_lookback + LAG:].mean())
    # by symmetry P(ric[t] > ric[t-L]) = 1/2. The indicator is strongly autocorrelated (window overlap, ~100+ rows):
    # effective n ~ 10 x 3700/150 ~ 250, sd of the pooled share ~ 0.03; the tolerance is 0.08 (> 2.5 sd).
    assert abs(np.mean(share) - 0.5) < 0.08


# ------------------------------------------------------------------ 4. portfolio arithmetic
def test_equal_weights_over_valid_factors_only():
    S = np.array([[0.5, -2.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 3.0, 0.0], [2.0, -2.0, np.nan]])
    P = np.array([[0.5, -0.4, 0.9], [1.0, 0.2, 1.0], [-0.5, 0.7, 0.0], [0.3, -0.6, 0.0]])
    sign = np.array([1.0, -1.0, 1.0])
    V = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 1], [1, 1, 1]], bool)
    cont, n = portfolio_position("cont", V, S, P, sign)
    np.testing.assert_array_equal(n, [2, 0, 2, 3])
    np.testing.assert_allclose(cont, [(0.5 + 0.4) / 2, 0.0, (-0.5 + 0.0) / 2, (0.3 + 0.6 + 0.0) / 3], atol=1e-15)
    a, _ = portfolio_position("binA", V, S, P, sign)                       # votes: +1*sign(s) (NaN -> 0)
    np.testing.assert_allclose(a, [(1 + 1) / 2, 0.0, (-1 + 0) / 2, (1 + 1 + 0) / 3], atol=1e-15)
    b, _ = portfolio_position("binB", V, S, P, sign)
    np.testing.assert_array_equal(b, np.sign(a))
    # a tie of votes is exactly 0, nothing valid is flat
    tie, _ = portfolio_position("binB", np.array([[1, 1]], bool), np.array([[1.0, 1.0]]), np.zeros((1, 2)),
                                np.array([1.0, -1.0]))
    assert tie.tolist() == [0.0]
    for v in ("cont", "binA", "binB"):
        pos, n0 = portfolio_position(v, np.zeros((3, 3), bool), S[:3], P[:3], sign)
        assert (pos == 0).all() and (n0 == 0).all()
    with pytest.raises(ValueError):
        portfolio_position("x", V, S, P, sign)


def test_buffered_position_mirror_and_history():
    rng = np.random.default_rng(2)
    s = rng.normal(size=300)
    s[:20] = np.nan
    p = factor_positions(s, 0.5)
    assert (p[:20] == 0).all()
    np.testing.assert_array_equal(p[20:], positions(s[20:], 1.0, 0.5))
    np.testing.assert_array_equal(positions(s[20:], -1.0, 0.5), -positions(s[20:], 1.0, 0.5))   # exact mirror
    assert not np.array_equal(p, np.sign(s))


def _rand_case(R=500, seed=4):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=R)
    S = [rng.normal(size=R) for _ in range(3)]
    S[1][:100] = np.nan                                                   # starts later
    S[2][300:320] = np.nan                                                # a hole
    return S, y


def test_always_cont_is_the_plain_equal_weight_portfolio():
    S, y = _rand_case()
    signs = [1.0, -1.0, 1.0]
    td = _td(S, signs, y)
    res = compute_target(td, SMALL)
    # independent: each factor's buffered position from its first finite row, mean over the factors with a signal
    pos = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for s, g in zip(S, signs):
        f = np.flatnonzero(np.isfinite(s))
        p = np.zeros(len(y))
        p[f[0]:] = g * positions(s[f[0]:], 1.0, 0.5)
        fin = np.isfinite(s)
        pos += np.where(fin, p, 0.0)
        cnt += fin
    pos = pos / cnt
    sg = td["m"]["sigma"]
    gross, net = simulate(pos, y, sg, 1e-4)
    v = res["variants"][("always", "cont", "full")]
    assert (v["a"], v["b"]) == (0, len(y) - 1)
    np.testing.assert_allclose(v["position"], pos, atol=1e-15)
    np.testing.assert_allclose(v["net"], net, atol=1e-15)
    np.testing.assert_allclose(v["gross"], gross, atol=1e-15)
    np.testing.assert_array_equal(v["n_valid"], cnt)


def test_variants_from_states_match_a_row_by_row_brute_force():
    S, y = _rand_case(R=700, seed=9)
    # let the factors have different quality so that the states differ
    y = y + 0.4 * np.nan_to_num(S[0]) * (np.arange(700) < 350) - 0.4 * np.nan_to_num(S[1])
    signs = [1.0, -1.0, 1.0]
    td = _td(S, signs, y)
    res = compute_target(td, SMALL)
    st = [applied_states(s, y, g, SMALL) for s, g in zip(S, signs)]
    for k in ("slope", "hyst", "cumic"):
        for ver in ("cont", "binA", "binB"):
            v = res["variants"][(k, ver, "full")]
            want = np.zeros(700)
            nv = np.zeros(700)
            for t in range(700):
                votes = []
                for i, s in enumerate(S):
                    if st[i][k][t] and np.isfinite(s[t]):
                        p = factor_positions(s, 0.5)[t]
                        votes.append(signs[i] * p if ver == "cont" else signs[i] * np.sign(s[t]))
                nv[t] = len(votes)
                if votes:
                    want[t] = np.sign(np.sum(votes)) if ver == "binB" else np.mean(votes)
            np.testing.assert_allclose(v["position"], want, atol=1e-14)
            np.testing.assert_array_equal(v["n_valid"], nv)
    assert any(res["variants"][(k, "cont", "full")]["n_valid"].min() == 0 for k in ("slope", "hyst", "cumic"))
    assert any(res["variants"][(k, "cont", "full")]["n_valid"].max() >= 2 for k in ("slope", "hyst", "cumic"))


def test_binB_is_the_sign_of_binA():
    S, y = _rand_case(R=600, seed=11)
    res = compute_target(_td(S, [1.0, 1.0, -1.0], y), SMALL)
    for k in ALL:
        a, b = res["variants"][(k, "binA", "full")]["position"], res["variants"][(k, "binB", "full")]["position"]
        np.testing.assert_array_equal(b, np.sign(a))
        assert set(np.unique(b)) <= {-1.0, 0.0, 1.0} and np.abs(a).max() <= 1.0


def test_offsetting_factors_net_to_zero_position_and_zero_cost():
    rng = np.random.default_rng(1)
    s, y = rng.normal(size=400), rng.normal(size=400)
    td = _td([s, s], [1.0, -1.0], y, cost=1e-2)                           # the same signal with opposite direction
    res = compute_target(td, SMALL)
    for ver in ("cont", "binA", "binB"):
        v = res["variants"][("always", ver, "full")]
        assert (v["position"] == 0).all()
        assert (v["gross"] == 0).all() and (v["net"] == 0).all()
    # costs are charged on the netted position: one factor alone trades and pays, the pair does not
    alone = compute_target(_td([s], [1.0], y, cost=1e-2), SMALL)["variants"][("always", "cont", "full")]
    assert (alone["gross"] - alone["net"]).sum() > 0
    s2 = rng.normal(size=400)
    pair = compute_target(_td([s, s2, s2], [1.0, 1.0, -1.0], y, cost=1e-2), SMALL)["variants"][("always", "cont", "full")]
    one = compute_target(_td([s], [1.0], y, cost=1e-2), SMALL)["variants"][("always", "cont", "full")]
    np.testing.assert_allclose(pair["position"], one["position"] / 3, atol=1e-15)    # s2 and -s2 cancel inside the mean


def test_summarize_numbers():
    net = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02])
    gross = net + 0.001
    pos = np.array([0.0, 1.0, 1.0, -1.0, 0.0, 0.5])
    ret = np.array([0.1, 0.2, -0.1, 0.0, 0.3, 0.1])
    dates = pd.bdate_range("2020-01-01", periods=6)
    r = summarize(gross, net, pos, np.array([0, 1, 2, 2, 0, 1]), ret, dates)
    k = 0.10 / np.sqrt(252) / net.std(ddof=1)
    nav = 1 + np.cumsum(k * net)
    assert r["cum_pnl_10vol"] == pytest.approx(k * net.sum())
    assert r["max_dd_10vol"] == pytest.approx(np.max(np.maximum.accumulate(nav) - nav))
    assert r["sharpe_net"] == pytest.approx(net.mean() / net.std(ddof=1) * np.sqrt(252))
    assert r["avg_abs_dpos"] == pytest.approx(np.mean([1, 0, 2, 1, 0.5]))
    assert r["share_flat"] == pytest.approx(2 / 6) and r["avg_n_valid"] == pytest.approx(1.0)
    assert r["corr_target"] == pytest.approx(np.corrcoef(net, ret)[0, 1])
    assert r["target_sharpe"] == pytest.approx(ret.mean() / ret.std(ddof=1) * np.sqrt(252))
    assert (r["start"], r["end"], r["n_rows"]) == ("2020-01-01", "2020-01-08", 6)


# ------------------------------------------------------------------ 5. samples
def test_oos_sample_uses_each_factor_only_on_its_own_oos_rows():
    S, y = _rand_case(R=600, seed=21)
    S = [np.where(np.isfinite(s), s, 0.5) for s in S]
    eo = [np.zeros(600, bool) for _ in S]
    eo[0][400:] = True
    eo[1][250:] = True
    eo[2][500:550] = True
    td = _td(S, [1.0, -1.0, 1.0], y, elig_oos=eo)
    res = compute_target(td, SMALL)
    v = res["variants"][("always", "binA", "oos")]
    assert (v["a"], v["b"]) == (250, 599)                                     # starts at the earliest OOS start
    want_n = np.array([int(eo[0][t]) + int(eo[1][t]) + int(eo[2][t]) for t in range(250, 600)])
    np.testing.assert_array_equal(v["n_valid"], want_n)
    t = 300                                                                   # only factor 1 is in
    assert v["position"][t - 250] == -np.sign(S[1][t])
    t = 520                                                                   # all three
    assert v["position"][t - 250] == pytest.approx((np.sign(S[0][t]) - np.sign(S[1][t]) + np.sign(S[2][t])) / 3)
    full = res["variants"][("always", "binA", "full")]
    assert full["a"] == 0 and full["n_valid"].max() == 3
    # the state machine may use the factor's history from before its OOS start: no warm-up reset at row 400
    y2 = y + 0.5 * S[0]
    td = _td([S[0]], [1.0], y2, elig_oos=[eo[0]])
    r2 = compute_target(td, SMALL)
    assert r2["variants"][("cumic", "cont", "oos")]["n_valid"][:5].tolist() == [1] * 5
    assert r2["variants"][("hyst", "cont", "oos")]["n_valid"][:5].tolist() == [1] * 5


def test_no_eligible_oos_rows_means_no_oos_sample():
    S, y = _rand_case(R=300, seed=2)
    res = compute_target(_td(S, [1, 1, 1], y), SMALL)
    assert {k[2] for k in res["variants"]} == {"full"}
    assert all(s["sample"] == "full" for s in res["summary"])
    assert np.isnan(res["stats"]["valid_share"]["oos"]["always"])


def test_opened_run_rows(run_dir, opened_dir):
    ctx = _load_run(opened_dir, 0)
    m = master_calendar([ctx])
    cand = pd.read_csv(opened_dir / "candidates.csv").set_index("id")
    for cid in ("sig|chg_1", "short|chg_1"):
        j = ctx["col_of"][cid]
        s, ef, eo = place_factor(m, ctx, j)
        assert ef.sum() == int(np.isfinite(ctx["S_np"][:, j]).sum())
        assert eo.sum() == cand.loc[cid, "n_oos"] and (ef | ~eo).all() and (eo <= ef).all()
        assert eo.sum() > 0 and ef[eo].all() and np.flatnonzero(eo).min() > np.flatnonzero(ef).min()
        ref = np.flatnonzero(ef)
        assert np.flatnonzero(eo).tolist() == ref[len(ref) - int(cand.loc[cid, "n_oos"]):].tolist()   # the last rows


def test_unopened_run_never_reads_its_oos_rows(tmp_path, run_dir, panel, cfg):
    """A run without confirm.json: its factors only have IS + OBS rows (nothing else is read), contribute nothing
    to ``oos``, and stop at their OOS start in ``full``. Random data after the last OOS start changes nothing."""
    assert not (run_dir / "confirm.json").exists()
    before = load_target([run_dir])
    cand = pd.read_csv(run_dir / "candidates.csv").set_index("id")
    for f in before["factors"]:
        assert not f["opened"] and not f["elig_oos"].any()
        assert f["elig_full"].sum() == cand.loc[f["id"], "n_is"] + cand.loc[f["id"], "n_obs"]
        assert np.array_equal(np.isfinite(f["s"]), f["elig_full"])              # signal is NaN on the OOS rows
    res_b = compute_target(before, pf.PortfolioConfig())
    assert {k[2] for k in res_b["variants"]} == {"full"}
    prep = prepare(panel, cfg)
    cols = list(prep["S"].columns)
    c = max(int(prep["rows"][int(prep["e2"][cols.index(f["id"])])]) for f in before["factors"])
    assert c < len(panel.target) - 5
    rng = np.random.default_rng(5)
    tgt, X = panel.target.copy(), panel.X.copy()
    tgt.iloc[c + 1:] = rng.normal(0, 0.005, len(tgt) - c - 1)
    for col in X.columns:
        fin = np.isfinite(X[col].to_numpy()) & (np.arange(len(X)) >= c)
        X.loc[fin, col] = rng.normal(0, 50, int(fin.sum()))
    d2 = tmp_path / "run2"
    shutil.copytree(run_dir, d2)
    Panel(target=tgt, X=X, meta=panel.meta, excluded=panel.excluded, info=panel.info).save(d2 / "panel")
    after = load_target([d2])
    res_a = compute_target(after, pf.PortfolioConfig())
    assert [f["id"] for f in after["factors"]] == [f["id"] for f in before["factors"]]
    last = max(np.flatnonzero(f["elig_full"]).max() for f in before["factors"])
    for fa, fb in zip(after["factors"], before["factors"]):
        np.testing.assert_array_equal(fa["s"][:last + 1], fb["s"][:last + 1])
        np.testing.assert_array_equal(fa["elig_full"], fb["elig_full"][:len(fa["elig_full"])])
    assert res_a["variants"].keys() == res_b["variants"].keys()
    for k, vb in res_b["variants"].items():
        va = res_a["variants"][k]
        assert (va["a"], va["b"]) == (vb["a"], vb["b"]) and vb["b"] <= last
        for key in ("position", "n_valid", "gross", "net"):
            np.testing.assert_array_equal(va[key], vb[key])
    assert res_a["summary"] == res_b["summary"]


def test_mixed_opened_and_unopened_runs(opened_dir, noisy_run):
    ctxs = [_load_run(opened_dir, 0), _load_run(noisy_run, 1)]
    m = master_calendar(ctxs)
    jo, ju = ctxs[0]["col_of"]["sig|chg_1"], ctxs[1]["col_of"]["noisy|chg_1"]
    so, efo, eoo = place_factor(m, ctxs[0], jo)
    su, efu, eou = place_factor(m, ctxs[1], ju)
    prep = ctxs[1]["prep"]
    assert not eou.any() and eoo.any()
    assert efu.sum() == int((prep["is_mask"][:, ju] | prep["obs_mask"][:, ju]).sum()) < int(np.isfinite(ctxs[1]["S_np"][:, ju]).sum())
    assert np.flatnonzero(efu).max() < np.flatnonzero(eoo).min() + 400           # sanity: ends where its OOS starts
    td = _td([so, su], [1.0, 1.0], m["y"], sigma=m["sigma"], cost=m["cost"], buffer=m["buffer"],
             elig_full=[efo, efu], elig_oos=[eoo, eou])
    res = compute_target(td, pf.PortfolioConfig())
    v = res["variants"][("always", "binA", "oos")]
    assert v["n_valid"].max() == 1                                            # the unopened factor never counts in oos
    w = res["variants"][("always", "binA", "full")]
    assert w["n_valid"].max() == 2 and w["n_valid"][-1] == 1                  # ... and has stopped by the end in full


def test_runs_of_one_target_must_agree(tmp_path, run_dir, noisy_run):
    master_calendar([_load_run(run_dir, 0), _load_run(noisy_run, 1)])          # the same target series: fine
    bad = tmp_path / "bad"
    shutil.copytree(noisy_run, bad)
    p = Panel.load(bad / "panel")
    t = p.target.copy()
    t.iloc[100] += 0.01
    Panel(target=t, X=p.X, meta=p.meta, excluded=p.excluded, info=p.info).save(bad / "panel")
    with pytest.raises(ValueError, match="target series differs"):
        master_calendar([_load_run(run_dir, 0), _load_run(bad, 1)])
    for key, val in (("buffer", "0.4"), ("cost_per_side", "0.0002")):
        d = tmp_path / f"cfg_{key}"
        shutil.copytree(noisy_run, d)
        txt = (d / "config.toml").read_text()
        if key == "buffer":
            txt += f"\n[position]\nbuffer = {val}\n"
        else:
            txt = txt.replace("cost_per_side = 0.00005", f"cost_per_side = {val}")
        (d / "config.toml").write_text(txt)
        with pytest.raises(ValueError, match=key):
            master_calendar([_load_run(run_dir, 0), _load_run(d, 1)])


# ------------------------------------------------------------------ 6. the final layer is the dashboard's stage 4
@pytest.mark.parametrize("which", [["run"], ["open"], ["run", "noisy"], ["open", "noisy"]])
def test_final_layer_equals_dashboard_stage_4(run_dir, opened_dir, noisy_run, which):
    dirs = [{"run": run_dir, "open": opened_dir, "noisy": noisy_run}[k] for k in which]
    data = build_asset_data(dirs)
    want = [f["id"] for f in data["factors"] if f["stage"] == 4]
    got = [f["id"] for f in load_target(dirs)["factors"]]
    assert want and sorted(got) == sorted(want) and len(set(got)) == len(got)
    td = load_target(dirs)
    by = {f["id"]: f for f in data["factors"]}
    for f in td["factors"]:                                                   # direction and OOS status come along
        assert f["sign"] == by[f["id"]]["sign"] and f["opened"] == by[f["id"]]["oos_opened"]
    if which == ["run", "noisy"]:
        assert any(f["stage"] == 3 for f in data["factors"])                  # merged factors are not in the portfolio


# ------------------------------------------------------------------ config, CLI and outputs
def test_portfolio_config(tmp_path):
    p = tmp_path / "p.toml"
    p.write_text("[portfolio]\nic_window = 252\nslope_lookback = 63\ncumic_ma = 126\n"
                 'methods = ["always", "slope", "hyst", "cumic"]\npositions = ["cont", "binA", "binB"]\n')
    pc = load_portfolio_config(p)
    assert (pc.ic_window, pc.slope_lookback, pc.cumic_ma) == (252, 63, 126)
    assert pc.methods == list(ALL) and pc.positions == ["cont", "binA", "binB"]
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent / "configs" / "portfolio.toml"
    assert load_portfolio_config(repo) == pc                                  # the shipped config carries these values
    for txt, msg in (("[portfolio]\nbogus = 1\n", "unknown key"), ("[other]\nx = 1\n", "unknown top-level"),
                     ("", "missing required section"), ('[portfolio]\nmethods = ["nope"]\n', "methods"),
                     ('[portfolio]\npositions = []\n', "positions"), ("[portfolio]\nic_window = 1\n", "ic_window"),
                     ("[portfolio]\nic_window = 2.5\n", "ic_window"), ("[portfolio]\nic_window = true\n", "bool"),
                     ('[portfolio]\nmethods = ["slope", "slope"]\n', "methods")):
        p.write_text(txt)
        with pytest.raises(ValueError, match=msg):
            load_portfolio_config(p)


def test_cli_outputs(tmp_path, run_dir, noisy_run, capsys):
    from tsminer.__main__ import main
    out = tmp_path / "out"
    pcfg = tmp_path / "p.toml"
    pcfg.write_text("[portfolio]\nic_window = 100\nslope_lookback = 30\ncumic_ma = 50\n")
    assert main(["portfolio", str(pcfg), str(run_dir), str(noisy_run), "--out", str(out)]) == 0
    files = {p.name for p in out.iterdir()}
    assert files == {"pnl_test.csv", "summary_test.csv", "state_changes_test.csv", "current_test.csv",
                     "portfolio.json", "report.md"}
    pnl = pd.read_csv(out / "pnl_test.csv")
    assert list(pnl.columns) == ["date", "method", "position_version", "sample", "n_valid", "position", "gross", "net"]
    sm = pd.read_csv(out / "summary_test.csv")
    assert len(sm) == 4 * 3 * 1 and set(sm["sample"]) == {"full"}                # both runs unopened: no oos sample
    assert len(pnl.groupby(["method", "position_version", "sample"])) == len(sm)
    for col in ("start", "end", "years", "sharpe_net", "sharpe_gross", "max_dd_10vol", "cum_pnl_10vol", "avg_abs_dpos",
                "avg_n_valid", "share_flat", "corr_target", "target_sharpe"):
        assert col in sm.columns
    # the summary agrees with the P&L file
    g = pnl[(pnl.method == "always") & (pnl.position_version == "cont")]
    row = sm[(sm.method == "always") & (sm.position_version == "cont")].iloc[0]
    assert row["sharpe_net"] == pytest.approx(g["net"].mean() / g["net"].std() * np.sqrt(252))
    ch = pd.read_csv(out / "state_changes_test.csv")
    assert list(ch.columns) == ["id", "method", "date", "new_state"] and set(ch["new_state"]) <= {"有效", "失效"}
    assert (ch[ch.method == "always"].groupby("id").size() == 1).all()          # one entry per factor, never a switch
    cur = pd.read_csv(out / "current_test.csv")
    assert {"id", "title", "series", "transform", "group", "sign", "latest_ric", "slope_state", "slope_since",
            "hyst_state", "cumic_since"} <= set(cur.columns)
    doc = json.loads((out / "portfolio.json").read_text())
    t = doc["targets"]["test"]
    assert doc["parameters"]["ic_window"] == 100 and doc["parameters"]["lag_rows"] == 2
    assert [f["id"] for f in t["factors"]] == list(cur["id"]) and len(t["summary"]) == len(sm)
    assert {r["run"] for r in t["runs"]} == {run_dir.name, noisy_run.name} and t["runs"][0]["candidates_sha256"]
    assert "有效" in (out / "report.md").read_text() and "选择偏差" in (out / "report.md").read_text()
    assert main(["portfolio", str(pcfg), str(tmp_path / "nope"), "--out", str(out)]) == 1
