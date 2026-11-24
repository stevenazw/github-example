import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsminer import rolling
from tsminer.__main__ import main
from tsminer.config import RollingConfig, load_config, load_rolling_config
from tsminer.databank import META_COLUMNS, Panel
from tsminer.engine import simulate
from tsminer.pipeline import positions
from tsminer.rolling import (build_pool, group_weights, load_sources, mean_hac_t, mining_rows, remine_rows,
                             roll, roll_spec, select, window_stats)

ROOT = Path(__file__).resolve().parent.parent
N = 2600
GROUPS = ["FX", "Equity", "Rates", "Vol"]
COST = 0.00005

TOML = f"""
[target]
name = "TEST"
kind = "spot"
ticker = "TEST Curncy"
field = "PX_LAST"
cost_per_side = {COST}

[lags]
chain = 1

[stats]
hac_lags = 5
min_obs = 250
n_boot = 200
min_shift = 60
seed = 7
"""

# 2-year windows, 6-month steps; the last year is sealed; first re-mining date on a Saturday (-> next trading day)
RC = RollingConfig(window_years=2, step_months=6, first_date="2013-01-05", holdout_years=1, stay_z=1.0,
                   group_cap=0.3334, coverage=0.8, grid={"window_years": [2, 3], "step_months": [6, 12]})


def make_panel(seed, n_noise=12, regime=None, ic=0.3, n=N):
    """Synthetic panel of random-walk series (7 candidates each). regime = (a, b): the series 'sig' predicts the
    target only for its increments d[t], a <= t < b; x[t] -> r[t+2], i.e. after the 1-row execution shift the held
    signal at row t predicts y_fwd[t] = r[t+1] / sigma_t."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    d = rng.normal(0, 1, (n, n_noise + 1))
    eps = rng.normal(0, 0.005, n)
    r = eps.copy()
    if regime is not None:
        a, b = regime
        t = np.arange(a, b)
        r[t + 2] = 0.005 * (ic * d[t, 0] + np.sqrt(1 - ic**2) * rng.normal(0, 1, len(t)))
    cols = ["sig"] + [f"n{i}" for i in range(n_noise)]
    X = pd.DataFrame(np.cumsum(d, axis=0), index=idx, columns=cols)
    meta = pd.DataFrame({
        "ticker": X.columns, "field": "PX_LAST", "kind": "level", "freq": "D",
        "group": [GROUPS[i % len(GROUPS)] for i in range(X.shape[1])], "lag": 1, "lag_rule": "chain",
        "description": [f"desc {c}" for c in X.columns], "first_obs": idx[0], "n_obs": n,
    }, index=pd.Index(X.columns, name="series_id"))[META_COLUMNS]
    return Panel(target=pd.Series(r, index=idx, name="TEST"), X=X, meta=meta,
                 excluded=pd.DataFrame({"ticker": ["a"], "reason": ["no_data"]}),
                 info={"db_version": "v-test", "published_at": "2026-01-01", "target": "TEST",
                       "built_at": "2026-01-01T00:00:00", "sample_start": str(idx[0].date()),
                       "sample_end": str(idx[-1].date()), "n_rows": n})


@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("cfg") / "test.toml"
    p.write_text(TOML)
    return p


@pytest.fixture(scope="module")
def cfg(cfg_path):
    return load_config(cfg_path)


def pool_of(cfg, panel, last_date=None):
    return build_pool([("src", cfg, panel)], last_date if last_date is not None else panel.target.index[-1])


def run_spec(pool, rc=RC, w=2, m=6):
    return roll_spec(pool, rc, w, m, {})


def save_run(base, name, panel, cfg_path):
    d = Path(base) / name
    panel.save(d / "panel")
    shutil.copyfile(cfg_path, d / "config.toml")
    return d


@pytest.fixture(scope="module")
def regime_panel():
    return make_panel(3, regime=(900, 1400))


@pytest.fixture(scope="module")
def regime_pool(cfg, regime_panel):
    return pool_of(cfg, regime_panel, regime_panel.target.index[N - 260])   # holdout: the last ~year


@pytest.fixture(scope="module")
def regime_res(regime_pool):
    return run_spec(regime_pool)


# ------------------------------------------------------------------ 1. no look-ahead
def test_selection_ignores_everything_after_the_mining_rows(cfg, cfg_path, regime_panel, regime_pool):
    base = pool_of(cfg, regime_panel, regime_panel.target.index[-1])
    full = run_spec(base)
    dates = base["dates"]
    k = 3
    Dk = dates[remine_rows(dates, RC.first_date, 6, len(dates) - 1)[k]]
    # scramble signals and returns on every calendar date strictly after D_k (the held row D_k + 1 is a holding row)
    p = regime_panel
    rng = np.random.default_rng(99)
    after = p.target.index > Dk
    X2, r2 = p.X.copy(), p.target.copy()
    X2[after] = np.cumsum(rng.normal(0, 3, (after.sum(), X2.shape[1])), axis=0) + 50.0
    r2[after] = rng.normal(0, 0.02, after.sum())
    alt = run_spec(pool_of(cfg, Panel(r2, X2, p.meta, p.excluded, p.info)))
    cols = ["period", "id", "sign", "weight", "z", "status"]
    a = full["members"].query("period <= @k")[cols].reset_index(drop=True)
    b = alt["members"].query("period <= @k")[cols].reset_index(drop=True)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)                   # membership, signs, weights and z: bit-identical
    assert full["members"].query("period > @k")["z"].tolist() != alt["members"].query("period > @k")["z"].tolist()


def test_nothing_after_the_cutoff_is_read_or_written(cfg_path, tmp_path):
    p = make_panel(5, regime=(900, 1400))
    run = save_run(tmp_path / "x", "src", p, cfg_path)
    cut = p.target.index[-1] - pd.DateOffset(years=RC.holdout_years)
    X2, r2 = p.X.copy(), p.target.copy()
    after = p.target.index > cut
    X2[after] = 1e6
    r2[after] = 0.5                                       # absurd values in the sealed year
    run2 = save_run(tmp_path / "y", "src", Panel(r2, X2, p.meta, p.excluded, p.info), cfg_path)
    outs = []
    for r, o in ((run, "o1"), (run2, "o2")):
        roll(RC, [r], tmp_path / o, log=lambda *_: None)
        outs.append({n: pd.read_csv(tmp_path / o / f"{n}.csv") for n in ("periods", "members", "exits", "pnl")})
    for n in outs[0]:
        pd.testing.assert_frame_equal(outs[0][n], outs[1][n])
    pnl = outs[0]["pnl"]
    assert pd.Timestamp(pnl["date"].max()) < cut          # the last holding row's next-day return is dated <= C
    info = json.loads((tmp_path / "o1" / "roll.json").read_text())
    assert info["cutoff"] == str(cut.date())


# ------------------------------------------------------------------ 2. timing
def test_mining_rows_end_before_the_remining_row_and_holding_rows_start_after_it(regime_pool, regime_res):
    dates = regime_pool["dates"]
    Ds = remine_rows(dates, RC.first_date, 6, len(dates) - 1)
    assert dates[Ds[0]] == pd.Timestamp("2013-01-07")       # first_date is a Saturday: next trading day
    for i in Ds:
        lo, hi = mining_rows(dates, i, 2)
        assert hi == i - 1                                    # row t needs r[t+1]: known at D only for t <= D - 1
        assert dates[lo] >= dates[i] - pd.DateOffset(years=2) > dates[lo - 1]
    for i, (_, r) in zip(Ds, regime_res["periods"].iterrows()):
        assert r["date"] == dates[i] and r["hold_start"] == dates[i + 1]


def test_holding_rows_partition_the_evaluation_span(regime_pool, regime_res):
    dates = regime_pool["dates"]
    per, pnl = regime_res["periods"], regime_res["pnl"]
    Ds = remine_rows(dates, RC.first_date, 6, len(dates) - 1)
    pos = {d: i for i, d in enumerate(dates)}
    s = [pos[d] for d in per["hold_start"]]
    e = [pos[d] for d in per["hold_end"]]
    assert all(s[k + 1] == e[k] + 1 for k in range(len(s) - 1))                     # no gap, no overlap
    assert e[:-1] == Ds[1:]                                                           # period k ends at D_{k+1}
    assert s[0] == Ds[0] + 1 and e[-1] == len(dates) - 1
    assert pnl["date"].tolist() == dates[Ds[0] + 1:].tolist()


def test_remining_dates_step_by_calendar_months():
    d = pd.bdate_range("2010-01-01", periods=2000)
    Ds = remine_rows(d, "2012-01-31", 6, len(d) - 1)
    want = [d[d.searchsorted(pd.Timestamp("2012-01-31") + pd.DateOffset(months=6 * k))] for k in range(len(Ds))]
    assert list(d[Ds]) == want and len(Ds) >= 6                # 2012-01-31 + 6m -> 2012-07-31, no drift afterwards


# ------------------------------------------------------------------ 3. regime factor and noise
def test_regime_factor_is_selected_with_a_lag_and_exits_after_it_stops_working(regime_pool, regime_res):
    a, b = 900, 1400                                          # panel positions of the regime
    p = make_panel(3, regime=(a, b))
    start, end = p.target.index[a], p.target.index[b]
    mem, per = regime_res["members"], regime_res["periods"]
    D = per.set_index("period")["date"]
    me = mem[mem["id"] == "src/sig|chg_1"]                    # the planted candidate
    held = me["period"].tolist()
    assert held, "the planted regime factor was never selected"
    assert held == list(range(held[0], held[-1] + 1))         # one spell
    assert (D[held] > start).all() and D[held[0]] > start + pd.Timedelta(days=60)   # needs months of evidence first
    assert me["status"].iloc[0] == "new" and (me["status"].iloc[1:] == "stay").all() and (me["sign"] == 1).all()
    # hysteresis keeps it after the stretch ended while the window still holds enough of it, but not for ever:
    # once the window (2y) has left the stretch it must be out
    assert D[held[-1]] < end + pd.DateOffset(years=2) and held[-1] < per["period"].iloc[-1]
    ex = regime_res["exits"]
    assert ((ex["id"] == "src/sig|chg_1") & (ex["reason"] == "stay_rule")).any()
    # it earns its IC while the regime lasts and no longer once the regime is over
    inside = me[(per.set_index("period").loc[me["period"], "hold_end"].to_numpy() <= end)]
    after = me[per.set_index("period").loc[me["period"], "hold_start"].to_numpy() > end]
    assert len(inside) >= 2 and inside["ic_signed_next"].mean() > 0.15
    assert len(after) >= 1 and after["ic_signed_next"].mean() < 0.1


def test_pure_noise_pool_has_no_edge(cfg):
    p = make_panel(17, n_noise=20)
    res = run_spec(pool_of(cfg, p, p.target.index[N - 260]))
    s = res["summary"]
    years = s["n_rows"] / 252
    tol = 3.0 / np.sqrt(years)                                # 3 standard errors of an annualised Sharpe
    assert abs(s["net_sharpe"]) < tol, (s["net_sharpe"], tol)
    assert abs(s["hac_t_net"]) < 3.0


# ------------------------------------------------------------------ 4. hysteresis
def _stat(z, sign, sharpe, N_):
    nan = np.full(N_, np.nan)
    return {"z": np.array(z, float), "sign": np.array(sign, float), "sharpe_net": np.array(sharpe, float),
            "t": nan, "ic": nan, "null_sd": nan, "eligible": np.ones(N_, bool)}


def test_hysteresis_keeps_incumbents_but_does_not_admit_equals(cfg):
    rng = np.random.default_rng(0)
    Sw = rng.normal(size=(300, 6))                           # independent columns: no de-duplication
    ids = [f"c{j}" for j in range(6)]
    # 0: incumbent, z=1.2 (stay_z <= |z| < z_weak), sign unchanged        -> stays
    # 1: NOT incumbent, same z, sharpe fine                              -> does not enter
    # 2: incumbent, sign flipped, strong (|z| 3 >= z_weak, sharpe fine)  -> exits (and is not re-admitted now)
    # 3: incumbent, |z| = 0.9 < stay_z                                    -> exits
    # 4: new, |z| 1.6, sharpe 0.1 <= is_min_net_sharpe                    -> does not enter
    # 5: new, |z| 1.6, sharpe fine                                        -> enters
    st = _stat([1.2, 1.2, -3.0, 0.9, 1.6, -1.6], [1, 1, -1, 1, 1, -1], [np.nan, 0.9, 1.0, np.nan, 0.1, 0.9], 6)
    prev = {0: 1, 2: 1, 3: 1}
    r = select(st, prev, Sw, ids, RC, cfg.verdict)
    assert r["held"] == {0: "stay", 5: "new"}
    assert r["exits"][2] == ("stay_rule", "sign_flip") and r["exits"][3] == ("stay_rule", "low_z")
    assert 1 not in r["held"] and 4 not in r["held"]
    assert r["n_stay"] == 1 and r["n_new_cand"] == 1


def test_failed_incumbent_can_re_enter_later_via_the_entry_rule(cfg):
    Sw = np.random.default_rng(1).normal(size=(300, 2))
    ids = ["a", "b"]
    low = _stat([0.5, 0.0], [1, 1], [0.9, 0.9], 2)
    assert select(low, {0: 1}, Sw, ids, RC, cfg.verdict)["held"] == {}
    strong = _stat([2.0, 0.0], [1, 1], [0.9, 0.9], 2)       # a later date, no longer an incumbent
    assert select(strong, {}, Sw, ids, RC, cfg.verdict)["held"] == {0: "new"}


# ------------------------------------------------------------------ 5. weights
def test_weights_sum_to_one_and_respect_the_cap():
    groups = ["A"] * 6 + ["B"] * 2 + ["C"] + ["D"]
    w = group_weights(groups, 0.3334)
    assert w.sum() == pytest.approx(1.0)
    g = np.array(groups)
    tot = {k: w[g == k].sum() for k in "ABCD"}
    assert max(tot.values()) <= 0.3334 + 1e-9               # A (equal weight 0.6) is cut to the cap
    assert tot["A"] == pytest.approx(0.3334)
    for k in "ABCD":                                         # equal within a group
        assert np.ptp(w[g == k]) == pytest.approx(0.0, abs=1e-15)
    # the excess goes to the other groups, equally per factor: B (2) vs C (1) vs D (1)
    assert w[g == "B"][0] == pytest.approx(w[g == "C"][0]) == pytest.approx(w[g == "D"][0])


def test_weights_iterate_until_feasible():
    groups = ["A"] * 8 + ["B"] * 4 + ["C"] * 1 + ["D"] * 1       # redistribution pushes B over the cap too
    w = group_weights(groups, 0.34)
    g = np.array(groups)
    assert w.sum() == pytest.approx(1.0)
    assert all(w[g == k].sum() <= 0.34 + 1e-9 for k in "ABCD")


def test_weights_equal_factor_weight_when_cap_not_binding():
    w = group_weights(["A", "B", "C", "A", "B", "C"], 0.3334)
    np.testing.assert_allclose(w, 1 / 6)


def test_weights_fewer_than_three_groups_are_equal_per_group():
    w = group_weights(["A"] * 5 + ["B"], 0.3334)
    assert w.sum() == pytest.approx(1.0)
    assert w[:5] == pytest.approx(0.1) and w[5] == pytest.approx(0.5)
    np.testing.assert_allclose(group_weights(["A", "A", "A"], 0.3334), 1 / 3)       # one group: everything in it
    assert len(group_weights([], 0.3334)) == 0


def test_no_factors_means_flat(cfg, monkeypatch):
    p = make_panel(21, n_noise=3)
    pool = pool_of(cfg, p, p.target.index[N - 260])
    z = np.full(pool["S"].shape[1], np.nan)
    monkeypatch.setattr(rolling, "window_stats", lambda *a, **k: _stat(z, z, z, len(z)))
    res = run_spec(pool)
    assert (res["pnl"][["position", "gross", "net"]] == 0).all().all()
    assert (res["periods"]["n_held"] == 0).all() and res["members"].empty


# ------------------------------------------------------------------ 6. composite
def test_mirrored_sign_identity_of_positions():
    s = np.random.default_rng(2).normal(size=500)
    s[100:130] = np.nan
    np.testing.assert_array_equal(positions(s, -1.0, 0.5), -positions(s, 1.0, 0.5))
    np.testing.assert_array_equal(positions(s, -1.0, 0.0), -positions(s, 1.0, 0.0))


def test_composite_is_the_weighted_sum_of_single_factor_buffered_positions(cfg, regime_pool, regime_res):
    pool, res = regime_pool, regime_res
    dates = pool["dates"]
    per, mem, pnl = res["periods"], res["members"], res["pnl"]
    at = {d: i for i, d in enumerate(dates)}
    col = {c: j for j, c in enumerate(pool["ids"])}
    checked = 0
    for k, r in per.iterrows():
        h = slice(at[r["hold_start"]], at[r["hold_end"]] + 1)
        m = mem[mem["period"] == k]
        assert m["weight"].sum() == pytest.approx(1.0 if len(m) else 0.0)
        want = np.zeros(h.stop - h.start)
        for _, f in m.iterrows():
            want += f["weight"] * positions(pool["S"][:, col[f["id"]]], float(f["sign"]), cfg.position.buffer)[h]
        got = pnl.set_index("date").loc[r["hold_start"]:r["hold_end"], "position"].to_numpy()
        np.testing.assert_allclose(got, want, atol=1e-12)
        checked += len(m)
    assert checked > 0


def test_costs_are_charged_on_the_netted_composite_position(cfg, monkeypatch):
    p = make_panel(8, n_noise=1)
    pool = pool_of(cfg, p, p.target.index[N - 260])
    pool["S"] = np.column_stack([pool["S"][:, 0], pool["S"][:, 0]])   # two identical candidates ...
    pool["ids"], pool["meta"] = pool["ids"][:2], pool["meta"].iloc[:2].copy()
    pool["meta"]["group"] = ["A", "B"]
    n = pool["S"].shape[1]
    z, sign, sh = np.array([3.0, 3.0]), np.array([1.0, -1.0]), np.array([1.0, 1.0])   # ... held with opposite signs
    monkeypatch.setattr(rolling, "window_stats", lambda *a, **k: _stat(z, sign, sh, n))
    monkeypatch.setattr(rolling, "dedupe", lambda ids, df, th: {i: i for i in ids})
    res = run_spec(pool)
    m = res["members"]
    assert (m.groupby("period")["id"].count() == 2).all()
    pnl = res["pnl"]
    assert (pnl["position"] == 0).all()
    assert (pnl["gross"] == 0).all() and (pnl["net"] == 0).all()     # no trade, hence no cost
    single = simulate(positions(pool["S"][:, 0], 1.0, cfg.position.buffer)[-200:], pool["y"][-200:],
                      pool["sig"][-200:], COST)
    assert (single[0] - single[1]).sum() > 0                          # each leg alone would have paid costs


def test_one_engine_run_over_all_holding_rows_charges_switches_on_the_netted_position(regime_pool, regime_res):
    pool, pnl = regime_pool, regime_res["pnl"]
    n = len(pnl)
    rows = np.arange(len(pool["dates"]) - n, len(pool["dates"]))
    g, net = simulate(pnl["position"].to_numpy(), pool["y"][rows], pool["sig"][rows], COST)
    np.testing.assert_allclose(pnl["net"], net, atol=1e-12)
    np.testing.assert_allclose(pnl["gross"], g, atol=1e-12)


# ------------------------------------------------------------------ 7. incumbent priority in de-duplication
def test_incumbent_keeps_its_cluster_against_a_stronger_correlated_entrant(cfg):
    rng = np.random.default_rng(4)
    a = rng.normal(size=400)
    Sw = np.column_stack([a, a + 0.1 * rng.normal(size=400), rng.normal(size=400)])       # corr(a, b) ~ 0.995
    ids = ["inc", "new_strong", "other"]
    st = _stat([1.2, 4.0, 2.0], [1, 1, 1], [np.nan, 1.0, 1.0], 3)
    r = select(st, {0: 1}, Sw, ids, RC, cfg.verdict)
    assert r["held"] == {0: "stay", 2: "new"}                # the weaker incumbent is the representative
    assert 1 not in r["held"]
    ctrl = select(st, {}, Sw, ids, RC, cfg.verdict)          # without incumbency the strong one wins the cluster
    assert ctrl["held"] == {1: "new", 2: "new"}
    # an incumbent absorbed by a higher-priority incumbent is logged as a de-duplication exit
    both = select(_stat([1.2, 2.0, 2.0], [1, 1, 1], [np.nan] * 3, 3), {0: 1, 1: 1}, Sw, ids, RC, cfg.verdict)
    assert 0 in both["exits"] and both["exits"][0][0] == "dedupe" and 1 in both["held"]


# ------------------------------------------------------------------ statistic, config, plumbing
def test_window_stats_is_the_mine_step1_statistic(cfg, regime_pool):
    from tsminer.stats import hac_t, null_sd
    pool = regime_pool
    lo, hi = 700, 1300
    s = window_stats(pool, lo, hi, RC)
    j = list(pool["ids"]).index("src/sig|chg_1")
    st = cfg.stats
    Sw, yw = pool["S"][lo:hi + 1], pool["y"][lo:hi + 1]
    t, ic, _ = hac_t(Sw[:, [j]], yw, st.hac_lags, st.min_obs)
    sd = null_sd(Sw[:, [j]], yw, st.hac_lags, st.min_obs, st.n_boot, st.min_shift, st.seed)
    assert s["z"][j] == pytest.approx(t[0] / sd[0], rel=1e-9) and s["sign"][j] == np.sign(ic[0])
    assert abs(s["z"][j]) > 1.5 and np.isfinite(s["sharpe_net"][j])
    # eligibility: coverage, min_obs and a finite value in the last rows
    S2 = Sw.copy()
    pool2 = {**pool, "S": np.vstack([pool["S"][:lo], S2, pool["S"][hi + 1:]])}
    pool2["S"][lo:lo + 300, 3] = np.nan                       # 50% coverage < 0.8
    pool2["S"][hi - 2:hi + 1, 4] = np.nan                     # alive earlier but not in the last 5 rows ...
    pool2["S"][hi - 4:hi - 2, 4] = np.nan                     # ... all 5 last rows missing
    s2 = window_stats(pool2, lo, hi, RC)
    assert not s2["eligible"][3] and not s2["eligible"][4] and s2["eligible"][j]


def test_mean_hac_t_matches_a_hand_computation():
    x = np.random.default_rng(6).normal(0.1, 1, 400)
    u = x - x.mean()
    lags = 3
    lrv = u @ u + 2 * sum((1 - l / (lags + 1)) * (u[l:] @ u[:-l]) for l in range(1, lags + 1))
    assert mean_hac_t(x, lags) == pytest.approx(x.mean() * len(x) / np.sqrt(lrv))
    assert mean_hac_t(x, 0) == pytest.approx(x.mean() / x.std(ddof=0) * np.sqrt(len(x)))


def test_sources_must_share_target_and_settings(cfg_path, tmp_path):
    p = make_panel(1, n_noise=2, n=600)
    a = save_run(tmp_path, "a", p, cfg_path)
    b = save_run(tmp_path, "b", p, cfg_path)
    assert [n for n, _, _ in load_sources([a, b])] == ["a", "b"]
    r2 = p.target.copy()
    r2.iloc[10] += 0.001
    c = save_run(tmp_path, "c", Panel(r2, p.X, p.meta, p.excluded, p.info), cfg_path)
    with pytest.raises(ValueError, match="target series"):
        load_sources([a, c])
    d = save_run(tmp_path, "d", p, cfg_path)
    (d / "config.toml").write_text(TOML.replace(str(COST), "0.0001"))
    with pytest.raises(ValueError, match=r"\[target\]"):
        load_sources([a, d])
    e = save_run(tmp_path, "e", p, cfg_path)
    (e / "config.toml").write_text(TOML.replace("seed = 7", "seed = 8"))
    with pytest.raises(ValueError, match=r"\[stats\]"):
        load_sources([a, e])


def test_pooled_ids_are_unique_across_sources(cfg, cfg_path, tmp_path):
    p = make_panel(2, n_noise=2, n=900)
    pool = build_pool([("x", cfg, p), ("y", cfg, p)], p.target.index[-1])      # the same candidates in two sources
    assert len(set(pool["ids"])) == len(pool["ids"]) == 2 * 3 * 7
    assert set(pool["meta"]["source"]) == {"x", "y"} and {"series", "transform", "group"} <= set(pool["meta"])


def test_rolling_config_loader(tmp_path):
    rc = load_rolling_config(ROOT / "configs" / "ty_roll.toml")
    assert (rc.window_years, rc.step_months, rc.first_date, rc.holdout_years) == (3, 6, "2005-01-01", 3)
    assert (rc.stay_z, rc.group_cap, rc.coverage) == (1.0, 0.3334, 0.8)
    assert rc.grid == {"window_years": [3, 5], "step_months": [6, 12]}
    good = (ROOT / "configs" / "ty_roll.toml").read_text()
    for bad, msg in [("coverage = 0.8", "coverage = 1.5"), ("stay_z = 1.0", "stay_z = 0.0"),
                     ("window_years = 3          #", "bogus = 1\nwindow_years = 3          #"),
                     ('first_date = "2005-01-01"', 'first_date = "not a date"'), ("holdout_years = 3", "holdout_years = -1"),
                     ("[rolling.grid]", "[rolling.grid]\nfoo = [1]")]:
        p = tmp_path / "bad.toml"
        p.write_text(good.replace(bad, msg, 1))
        with pytest.raises(ValueError):
            load_rolling_config(p)


def test_cli_roll_writes_every_output_and_the_grid(cfg_path, tmp_path, capsys):
    p = make_panel(9, n_noise=4, regime=(700, 1100), n=1900)
    run = save_run(tmp_path, "r1", p, cfg_path)
    rc = tmp_path / "roll.toml"
    rc.write_text('[rolling]\nwindow_years = 2\nstep_months = 6\nfirst_date = "2013-01-01"\nholdout_years = 1\n'
                  'stay_z = 1.0\ngroup_cap = 0.3334\ncoverage = 0.8\n[rolling.grid]\nwindow_years = [2]\n'
                  'step_months = [6, 12]\n')
    out = tmp_path / "out"
    assert main(["roll", str(rc), str(run), "--out", str(out), "--grid"]) == 0
    for f in ("roll.json", "periods.csv", "members.csv", "exits.csv", "pnl.csv", "report.md"):
        assert (out / f).exists()
    info = json.loads((out / "roll.json").read_text())
    assert info["specs"] == ["2y6m", "2y12m"] and info["primary"] == "2y6m"
    assert info["sources"][0]["run"] == "r1" and len(info["sources"][0]["panel_pkl_sha256"]) == 64
    assert set(pd.read_csv(out / "pnl.csv")["spec"]) == {"2y6m", "2y12m"}
    assert "滚动挖掘报告" in (out / "report.md").read_text(encoding="utf-8")
    assert main(["roll", str(rc), str(tmp_path / "nope"), "--out", str(out)]) == 1
