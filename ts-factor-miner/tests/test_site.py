import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest

from tsminer.config import load_config
from tsminer.confirm import confirm
from tsminer.pipeline import mine, prepare
from tsminer import site as site_mod
from tsminer.site import (SEG_NAMES, RIC_WINDOW, rolling_ic, build_asset_data, build_site, direction_text, downsample_idx, factor_nav,
                          choose_keeper, factor_stage, flag_texts, formula_text, group_same_factors, is_same_factor, slugify,
                          stage_note, transform_label)
from tsminer.databank import Panel

from test_pipeline import HIGH_COST, TOML, _cut_panel, make_panel

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_HOSTS = {"cdnjs.cloudflare.com", "cdn.jsdelivr.net", "unpkg.com", "fonts.googleapis.com", "fonts.gstatic.com"}


@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("sitecfg") / "test.toml"
    p.write_text(TOML + "\n[verdict]\nz_weak = 2.0\n")      # pin the pre-1.5 bar: the fixtures' funnel stays as designed
    return p


@pytest.fixture(scope="module")
def cfg(cfg_path):
    return load_config(cfg_path)


@pytest.fixture(scope="module")
def panel():
    return make_panel(1, plant=True, short=True, dup=True)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, cfg, cfg_path, panel):
    d = tmp_path_factory.mktemp("siterun") / "run"
    mine(cfg, d, panel=panel, config_path=cfg_path)
    assert len(pd.read_csv(d / "shortlist.csv")) >= 2
    return d


@pytest.fixture(scope="module")
def opened_dir(tmp_path_factory, run_dir):
    d = tmp_path_factory.mktemp("siteopen") / "run"
    shutil.copytree(run_dir, d)
    confirm(d)
    return d


@pytest.fixture(scope="module")
def locked_data(run_dir):
    """The single source of a single-run asset (the per-source dict)."""
    return build_asset_data(run_dir)


@pytest.fixture(scope="module")
def opened_data(opened_dir):
    return build_asset_data(opened_dir)


@pytest.fixture(scope="module")
def site_dir(tmp_path_factory, run_dir, opened_dir):
    other = tmp_path_factory.mktemp("siteother") / "run"
    shutil.copytree(opened_dir, other)
    rj = json.loads((other / "run.json").read_text())
    rj["target"] = "OTHER-1"
    (other / "run.json").write_text(json.dumps(rj))
    out = tmp_path_factory.mktemp("site")
    build_site([run_dir, other], out)
    return out


# ------------------------------------------------------------------ 1. isolation
def test_unopened_has_no_oos(run_dir, locked_data, cfg, panel):
    assert not any(f["oos_opened"] for f in locked_data["factors"])
    assert locked_data["oos_counts"] == {"confirmed": 0, "consistent": 0, "failed": 0, "insufficient": 0,
                                         "unopened": locked_data["funnel"]["n_final"]}
    assert locked_data["factors"]
    prep = prepare(panel, cfg)
    cal = panel.target.index
    col_of = {c: i for i, c in enumerate(prep["S"].columns)}
    for f in locked_data["factors"]:
        assert 2 not in f["nav"]["seg"]
        assert f["oos_verdict"] is None and f["oos_confirm"] is None
        assert f["segments"][2] == {"seg": 2, "name": "样本外", "opened": False}
        j = col_of[f["id"]]
        last_obs = np.flatnonzero(prep["obs_mask"][:, j] | prep["is_mask"][:, j]).max()
        # the return of row p is realised on the next calendar row; that is the last date allowed
        assert f["nav"]["dates"][-1] <= str(cal[prep["rows"][last_obs] + 1].date())
        assert f["nav"]["dates"][-1] > str(cal[prep["rows"][last_obs]].date())


@pytest.mark.parametrize("cid", ["sig|chg_1", "short|chg_1"])
def test_unopened_output_ignores_oos_rows(tmp_path, run_dir, locked_data, cfg, panel, cid):
    assert cid in {f["id"] for f in locked_data["factors"]} or cid not in set(
        pd.read_csv(run_dir / "shortlist.csv")["id"])
    ids = set(pd.read_csv(run_dir / "shortlist.csv")["id"])
    # use a shortlist factor if the named one was folded into a cluster
    cid = cid if cid in ids else sorted(ids)[0]
    p2, c = _cut_panel(panel, cfg, cid, 4242, "e2")
    assert c < len(panel.target) - 5
    d2 = tmp_path / "run2"
    shutil.copytree(run_dir, d2)
    p2.save(d2 / "panel")
    b = build_asset_data(d2)
    fa = next(f for f in locked_data["factors"] if f["id"] == cid)
    fb = next(f for f in b["factors"] if f["id"] == cid)
    assert fa == fb


def test_opened(opened_dir, opened_data):
    assert all(f["oos_opened"] for f in opened_data["factors"])
    cf = pd.read_csv(opened_dir / "confirm.csv").set_index("id")
    assert opened_data["factors"]
    for f in opened_data["factors"]:
        assert set(f["nav"]["seg"]) == {0, 1, 2}                # also the dropped factors: shown, but no verdict
        assert f["segments"][2]["n"] > 0
        if f["stage"] == 4:
            assert f["oos_verdict"] == cf.loc[f["id"], "verdict"]
        else:
            assert f["oos_verdict"] is None and f["oos_confirm"] is None
    final = [f for f in opened_data["factors"] if f["stage"] == 4]
    assert {f["id"] for f in final} == set(cf.index)
    cnt = cf.loc[[f["id"] for f in final], "verdict"].value_counts()
    assert opened_data["oos_counts"]["confirmed"] == int(cnt.get("CONFIRMED", 0))
    order = {"CONFIRMED": 0, "CONSISTENT": 1, "INSUFFICIENT": 2, "FAILED": 3}
    ranks = [order[f["oos_verdict"]] for f in final]
    assert ranks == sorted(ranks)
    st = [f["stage"] for f in opened_data["factors"]]
    assert st == sorted(st, reverse=True)


# ------------------------------------------------------------------ 2. net value definition
@pytest.fixture(scope="module")
def raw(opened_dir, cfg, panel):
    """Un-downsampled inputs of one factor over all three segments."""
    sl = pd.read_csv(opened_dir / "shortlist.csv").iloc[0]
    prep = prepare(panel, cfg)
    j = list(prep["S"].columns).index(sl["id"])
    rows = prep["rows"]
    m = prep["is_mask"][:, j] | prep["obs_mask"][:, j] | prep["oos_mask"][:, j]
    idx = np.flatnonzero(m)
    seg = np.where(prep["is_mask"][idx, j], 0, np.where(prep["obs_mask"][idx, j], 1, 2))
    cal = panel.target.index
    args = dict(s=prep["S"].to_numpy(dtype=float)[rows][idx, j], y=prep["y_fwd"].to_numpy(dtype=float)[rows][idx],
                sig=prep["sigma"].to_numpy(dtype=float)[rows][idx], seg=seg,
                dates=np.asarray(cal[rows[idx] + 1].strftime("%Y-%m-%d")), sign=float(sl["sign"]),
                buffer=cfg.position.buffer, cost=cfg.target.cost_per_side)
    return args, factor_nav(**args)


def test_nav_definition(raw):
    a, res = raw
    ret, nav, seg = res["ret"], res["nav"], a["seg"]
    assert nav[0] == pytest.approx(1 + ret[0])
    ins = ret[seg == 0]
    assert np.std(ins, ddof=1) * np.sqrt(252) == pytest.approx(0.10, rel=1e-6)
    np.testing.assert_allclose(np.diff(nav), ret[1:], rtol=1e-9, atol=1e-12)
    cum = sum(s["cum_ret"] for s in res["segments"])
    assert cum == pytest.approx(nav[-1] - 1)
    for s in res["segments"]:
        r = ret[seg == s["seg"]]
        assert s["n"] == len(r)
        assert s["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
        v = nav[seg == s["seg"]]
        assert s["max_dd"] == pytest.approx(float((np.maximum.accumulate(v) - v).max()))


def test_nav_zero_vol_in_sample_is_skipped():
    n = 50
    assert factor_nav(np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n, dtype=int),
                      np.array(["2020-01-01"] * n), 1.0, 0.5, 0.0) is None


# ------------------------------------------------------------------ 3. downsampling
def test_downsample_idx_keeps_boundaries(raw):
    a, _ = raw
    seg = a["seg"]
    assert len(seg) > 1500                      # the synthetic factor does exercise the downsampling
    keep = downsample_idx(seg)
    assert len(keep) <= 1500 + 3 + 1
    assert keep[-1] == len(seg) - 1
    for k in (0, 1, 2):
        w = np.flatnonzero(seg == k)
        assert w[0] in keep and w[-1] in keep
    assert list(keep) == sorted(set(keep))


def test_downsampled_json(opened_data):
    for f in opened_data["factors"]:
        nav, segs = f["nav"], f["segments"]
        assert len(nav["dates"]) == len(nav["values"]) == len(nav["seg"]) <= 1500 + 4
        assert nav["seg"] == sorted(nav["seg"])
        d = nav["dates"]
        for s in segs:
            assert s["start"] in d and s["end"] in d
        assert d[-1] == segs[2]["end"]
        assert all(round(v, 4) == v for v in nav["values"])


# ------------------------------------------------------------------ 3b. rolling IC (display only)
def _ric_inputs(n=400, seed=3):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    y = 0.5 * s + rng.normal(size=n)                    # planted: the signal points the right way
    return s, y


def test_rolling_ic_window_and_value():
    s, y = _ric_inputs()
    r = rolling_ic(s, y, 1.0)
    assert RIC_WINDOW == 252 and len(r) == len(s)
    assert np.isnan(r[:251]).all() and np.isfinite(r[251:]).all()       # full window only
    for t in (251, 300, 399):
        assert r[t] == pytest.approx(np.corrcoef(s[t - 251:t + 1], y[t - 251:t + 1])[0, 1], abs=1e-12)
    # hand-checkable: a perfectly linear window correlates to exactly +1, its mirror to -1
    a = np.arange(300.0)
    assert rolling_ic(a, 2 * a + 1, 1.0)[251:] == pytest.approx(1.0)
    assert rolling_ic(a, -a, 1.0)[251:] == pytest.approx(-1.0)
    assert np.isnan(rolling_ic(a[:251], a[:251], 1.0)).all()            # fewer rows than a window
    assert np.isnan(rolling_ic(np.ones(300), a, 1.0)).all()             # no variance: NaN, no warning


def test_rolling_ic_is_causal():
    s, y = _ric_inputs()
    base = rolling_ic(s, y, 1.0)
    t = 330
    s2, y2 = s.copy(), y.copy()
    rng = np.random.default_rng(9)
    s2[t + 1:], y2[t + 1:] = rng.normal(size=len(s) - t - 1) * 50, rng.normal(size=len(s) - t - 1) * 50
    alt = rolling_ic(s2, y2, 1.0)
    np.testing.assert_array_equal(alt[:t + 1], base[:t + 1])           # rows up to t are untouched by later rows
    assert not np.array_equal(alt[t + 1:], base[t + 1:])


def test_rolling_ic_direction():
    s, y = _ric_inputs()
    pos, neg = rolling_ic(s, y, 1.0), rolling_ic(s, y, -1.0)
    np.testing.assert_array_equal(neg[251:], -pos[251:])
    assert np.isnan(neg[:251]).all()
    assert np.nanmean(pos) > 0.3 and np.nanmin(pos) > 0.15              # planted relationship, sign +1: clearly positive
    assert np.nanmax(neg) < -0.15


def test_rolling_ic_in_site_data_matches_direct_computation(locked_data, opened_data, run_dir, opened_dir, cfg, panel):
    """At the last stored point the value is the sign-adjusted correlation of the trailing 252 rows of the factor's
    visible rows; for an unopened run those are in-sample + observation rows only (no OOS row inside the window)."""
    prep = prepare(panel, cfg)
    cols = list(prep["S"].columns)
    S = prep["S"].to_numpy(dtype=float)[prep["rows"]]
    y = prep["y_fwd"].to_numpy(dtype=float)[prep["rows"]]
    for data, opened in ((locked_data, False), (opened_data, True)):
        checked = 0
        for f in data["factors"]:
            nav = f["nav"]
            assert len(nav["ric"]) == len(nav["dates"]) == len(nav["values"])      # shares the net value's points
            assert all(v is None or round(v, 3) == v for v in nav["ric"])
            j = cols.index(f["id"])
            m = prep["is_mask"][:, j] | prep["obs_mask"][:, j] | (prep["oos_mask"][:, j] if opened else False)
            idx = np.flatnonzero(m)
            if len(idx) < RIC_WINDOW:
                assert all(v is None for v in nav["ric"])
                continue
            w = idx[-RIC_WINDOW:]
            want = f["sign"] * np.corrcoef(S[w, j], y[w])[0, 1]
            assert nav["ric"][-1] == pytest.approx(want, abs=6e-4)
            first = next(i for i, v in enumerate(nav["ric"]) if v is not None)
            assert first >= 1 and all(v is None for v in nav["ric"][:first])      # the first year is blank
            checked += 1
        assert checked


def test_rolling_ic_hides_unopened_segment(locked_data, tmp_path, run_dir, cfg, panel):
    """An unopened OOS exposes no rolling IC: no point of segment 2 and no window that reaches into it (altering
    the OOS rows changes nothing, also not the rolling IC)."""
    for f in locked_data["factors"]:
        assert 2 not in f["nav"]["seg"] and len(f["nav"]["ric"]) == len(f["nav"]["seg"])
        assert f["segments"][2] == {"seg": 2, "name": "样本外", "opened": False}
    cid = locked_data["factors"][0]["id"]
    p2, c = _cut_panel(panel, cfg, cid, 4242, "e2")
    d2 = tmp_path / "run2"
    shutil.copytree(run_dir, d2)
    p2.save(d2 / "panel")
    b = build_asset_data(d2)
    fa = next(f for f in locked_data["factors"] if f["id"] == cid)
    fb = next(f for f in b["factors"] if f["id"] == cid)
    assert any(v is not None for v in fa["nav"]["ric"]) and fa["nav"]["ric"] == fb["nav"]["ric"]


def test_rolling_ic_is_display_only(opened_dir, opened_data, monkeypatch):
    """With the rolling IC switched off (all NaN) everything else is identical: stage, verdict, order, net value."""
    monkeypatch.setattr(site_mod, "rolling_ic", lambda s, y, sign, window=RIC_WINDOW: np.full(len(s), np.nan))
    off = build_asset_data(opened_dir)
    assert all(v is None for f in off["factors"] for v in f["nav"]["ric"])
    assert any(v is not None for f in opened_data["factors"] for v in f["nav"]["ric"])

    def strip(d):
        d = _strip(d)
        return {**d, "factors": [{**f, "nav": {k: v for k, v in f["nav"].items() if k != "ric"}} for f in d["factors"]]}
    assert strip(off) == strip(opened_data)
    assert [(f["id"], f["stage"], f["oos_verdict"]) for f in off["factors"]] == \
           [(f["id"], f["stage"], f["oos_verdict"]) for f in opened_data["factors"]]


# ------------------------------------------------------------------ 4. formulas
def _meta(**kw):
    base = {"ticker": "USGG2YR Index", "field": "PX_LAST", "kind": "price", "lag": 1}
    base.update(kw)
    return pd.Series(base)


def test_formula_price_and_level(cfg):
    t = "\n".join(formula_text(_meta(), "chg_60", cfg, 1.0))
    assert "ln(x[t] / x[t−60])" in t and "USGG2YR Index" in t and "滞后 1 日可得" in t
    t = "\n".join(formula_text(_meta(kind="level"), "chg_5", cfg, 1.0))
    assert "ln(" not in t and "x[t] − x[t−5]" in t


def test_formula_chg_std_window_follows_k(cfg):
    t = "\n".join(formula_text(_meta(), "chg_252", cfg, 1.0))
    assert "ln(x[t] / x[t−252])" in t and "std_756(r)" in t and "std_252" not in t
    t = "\n".join(formula_text(_meta(kind="level"), "chg_252", cfg, 1.0))
    assert "std_756(d)" in t
    assert "std_252(r)" in "\n".join(formula_text(_meta(), "chg_60", cfg, 1.0))
    assert transform_label("chg_252") == "252 日变化"


def test_formula_z_and_position_sign(cfg):
    t = "\n".join(formula_text(_meta(), "z_252", cfg, -1.0))
    assert "mean_252(" in t and "std_252(" in t
    assert "pos[t] = −1 × s[t−1] ÷ σ[t]" in t and "信号形成后第 1 个收盘成交" in t     # default execution_delay = 1
    assert "pos[t] = +1 × s[t−1] ÷ σ[t]" in "\n".join(formula_text(_meta(), "z_60", cfg, 1.0))


def test_formula_execution_delay_variants(tmp_path):
    for d, want in ((0, "s[t]"), (1, "s[t−1]"), (2, "s[t−2]")):
        p = tmp_path / f"c{d}.toml"
        p.write_text(TOML + f"\n[position]\nexecution_delay = {d}\n")
        t = "\n".join(formula_text(_meta(), "chg_5", load_config(p), 1.0))
        assert f"pos[t] = +1 × {want} ÷ σ[t]" in t
        assert ("信号形成后第" in t) == (d > 0)
        if d:
            assert f"信号形成后第 {d} 个收盘成交" in t and "s[t−" + str(d) + "]" in t


def test_formula_follows_config(cfg_path, tmp_path):
    txt = (TOML.replace("std_window = 252", "std_window = 120").replace("clip = 4.0", "clip = 3.0")
           .replace("vol_halflife = 20", "vol_halflife = 33") + "\n[position]\nbuffer = 0.8\n")
    p = tmp_path / "c.toml"
    p.write_text(txt)
    c2 = load_config(p)
    t = "\n".join(formula_text(_meta(), "chg_20", c2, 1.0))
    assert "std_120(r)" in t and "±3" in t and "半衰期 33" in t and "缓冲带 0.8" in t
    assert "±4" not in t and "std_252" not in t


def test_formula_special_series(cfg):
    assert "TY 期货连续合约收益指数" in formula_text(_meta(ticker="chain:TY", field="return_index"), "chg_20", cfg)[0]
    assert "TY 跨月价差 ÷ 主力合约价格" in formula_text(_meta(ticker="carry:TY", kind="level"), "z_60", cfg)[0]
    sp = formula_text(_meta(ticker="spread:A-B", field="AAA Index-BBB Index", kind="level"), "chg_5", cfg)[0]
    assert "AAA Index − BBB Index" in sp


def test_direction_flags_slug():
    assert direction_text(1, "TY") == "因子上升 → 做多 TY"
    assert direction_text(-1.0, "TY") == "因子上升 → 做空 TY"
    assert flag_texts("xlag_weak;obs_short") == ["对数据可得时点敏感", "观察期样本不足"]
    assert flag_texts("") == [] and flag_texts(float("nan")) == []
    assert slugify("EURUSD") == "eurusd" and slugify("TY") == "ty"
    with pytest.raises(ValueError):
        slugify("下载")
    with pytest.raises(ValueError):
        slugify("Settings")


def test_factor_text_fields(opened_data):
    for f in opened_data["factors"]:
        assert f["direction"].startswith("因子上升 → ") and "TEST" in f["direction"]
        assert any("pos[t]" in line for line in f["formula"])
        sg = "+1" if f["sign"] > 0 else "−1"
        assert f"pos[t] = {sg} ×" in "\n".join(f["formula"])


# ------------------------------------------------------------------ 5. site files
def test_site_files(site_dir):
    idx = (site_dir / "index.html").read_text(encoding="utf-8")
    art = (site_dir / "artifact.html").read_text(encoding="utf-8")
    assert idx.startswith("<!doctype html>")
    low = art.lower()
    for bad in ("<!doctype", "<html", "<head", "<body"):
        assert bad not in low
    assert "<title>因子跟踪看板</title>" in art[:300]
    assert '<meta name="claude:viewport-fit-sides" content="cover">' in art[:400]
    assert art.index("<style>") > art.index("<meta name=\"claude")
    for slug in ("test", "other-1"):
        assert slug in idx and slug in art
        data = json.loads((site_dir / "data" / f"{slug}.json").read_text(encoding="utf-8"))
        assert data["slug"] == slug and data["factors"] and "sources" not in data
        assert len(data["runs"]) == 1 and data["funnel"]["n_is"] == len(data["factors"])
    # same template: the page content of both files is identical
    style = art[art.index("<style>"):art.index("</style>")]
    assert style in idx and art[art.index('<div id="app">'):] in idx
    assert "fetch('data/'" in art


# ------------------------------------------------------------------ 5b. criteria and the tradability step
def test_new_run_has_criteria_and_trade_keep(site_dir, run_dir, locked_data):
    vd = load_config(run_dir / "config.toml").verdict
    want = {"z_weak": vd.z_weak, "obs_min_retention": vd.obs_min_retention, "min_net_sharpe": vd.min_net_sharpe,
            "is_min_net_sharpe": vd.is_min_net_sharpe, "dedupe_corr": vd.dedupe_corr, "execution_delay": 1}
    assert want["min_net_sharpe"] == 0.3 and want["is_min_net_sharpe"] == 0.3
    assert locked_data["criteria"] == want
    run = json.loads((run_dir / "run.json").read_text())
    assert locked_data["funnel"]["n_final"] == run["funnel"]["shortlist"]
    for slug in ("test", "other-1"):
        d = json.loads((site_dir / "data" / f"{slug}.json").read_text(encoding="utf-8"))
        assert d["criteria"] == want and d["funnel"]["n_final"] == run["funnel"]["shortlist"]
    art = (site_dir / "artifact.html").read_text(encoding="utf-8")
    assert "入选条件" in art and "criteria" in art
    assert "cost" + "_negative" not in art and "扣成本后收益为负" not in art
    assert "min_net_sharpe" in art and "扣成本后净 Sharpe ≥ " in art
    assert "扣成本后净 Sharpe > " in art and "is_min_net_sharpe" in art


def test_old_style_run_without_trade_keys(tmp_path, run_dir):
    old = tmp_path / "old"
    shutil.copytree(run_dir, old)
    rj = json.loads((old / "run.json").read_text())
    del rj["funnel"]["trade_keep"], rj["funnel"]["trade_drop"], rj["config"]["verdict"]["min_net_sharpe"]
    del rj["config"]["verdict"]["is_min_net_sharpe"]
    (old / "run.json").write_text(json.dumps(rj))
    cdf = pd.read_csv(old / "candidates.csv")
    assert not (cdf["trade_status"] == "DROP").any()
    cdf.drop(columns=["trade_status", "sharpe_gross_dev", "sharpe_net_dev"]).to_csv(old / "candidates.csv", index=False)
    out = tmp_path / "site"
    build_site([old], out)                                           # must not raise
    d = json.loads((out / "data" / "test.json").read_text(encoding="utf-8"))
    assert "min_net_sharpe" not in d["criteria"] and "is_min_net_sharpe" not in d["criteria"] and all(f["stage"] != 2 for f in d["factors"])
    assert d["funnel"]["n_final"] == len(pd.read_csv(old / "shortlist.csv"))
    assert d["criteria"]["z_weak"] == rj["config"]["verdict"]["z_weak"]
    assert d["factors"]


def test_old_run_without_execution_delay(tmp_path, run_dir):
    """A run from before execution_delay existed: counts as 0, no error, no execution text anywhere."""
    old = tmp_path / "old"
    shutil.copytree(run_dir, old)
    rj = json.loads((old / "run.json").read_text())
    del rj["config"]["position"]["execution_delay"]
    (old / "run.json").write_text(json.dumps(rj))
    out = tmp_path / "site"
    build_site([old], out)                                           # must not raise
    d = json.loads((out / "data" / "test.json").read_text(encoding="utf-8"))
    assert "execution_delay" not in d["criteria"] and d["factors"]
    for f in d["factors"]:
        t = "\n".join(f["formula"])
        assert "pos[t] = " in t and "s[t−1]" not in t and "信号形成后第" not in t and " × s[t] ÷ σ[t]" in t
    art = (out / "artifact.html").read_text(encoding="utf-8")
    assert "成交：信号形成后第" in art and "execText" in art


def test_new_run_site_shows_execution(site_dir, locked_data):
    assert locked_data["criteria"]["execution_delay"] == 1
    for f in locked_data["factors"]:
        t = "\n".join(f["formula"])
        assert "s[t−1] ÷ σ[t]" in t and "信号形成后第 1 个收盘成交" in t
    d = json.loads((site_dir / "data" / "test.json").read_text(encoding="utf-8"))
    assert d["criteria"]["execution_delay"] == 1


def test_flag_text_has_no_cost_flag():
    assert flag_texts("unstable") == ["逐年方向不稳定"]


# ------------------------------------------------------------------ 6. publishing constraints
def test_publish_constraints(site_dir):
    art = (site_dir / "artifact.html").read_text(encoding="utf-8")
    hosts = {urlparse(u).hostname for u in re.findall(r'(?:src|href)="(https?://[^"]+)"', art)}
    assert hosts and hosts <= ALLOWED_HOSTS
    scripts = re.findall(r'<script[^>]*\ssrc="([^"]+)"', art)
    assert all(urlparse(s).hostname in ALLOWED_HOSTS - {"fonts.googleapis.com", "fonts.gstatic.com"} for s in scripts)
    assert ':root:not([data-theme="light"])' in art and ':root[data-theme="dark"]' in art
    assert "@media (prefers-color-scheme: dark)" in art
    assert "max(16px, env(safe-area-inset-left" in art and "max(16px, env(safe-area-inset-right" in art
    assert "top: env(safe-area-inset-top, 0px)" in art
    for bad in ("alert(", "confirm(", "prompt(", "window.print"):
        assert bad not in art
    # every colour literal lives inside the token blocks, not in component rules
    css = re.search(r"<style>(.*?)</style>", art, re.S).group(1)
    tokens_end = css.index("* { box-sizing")
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", css[tokens_end:])


# ------------------------------------------------------------------ 7. duplicates and CLI
def _cli(*args, cwd=ROOT):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-m", "tsminer", "site", *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_cli(tmp_path, run_dir, opened_dir):
    out = tmp_path / "s"
    ok = _cli(run_dir, "--out", out)
    assert ok.returncode == 0, ok.stderr
    assert (out / "index.html").exists() and (out / "data" / "test.json").exists()
    again = _cli(opened_dir, "--out", out)                               # appended to the registry: one library
    assert again.returncode == 0, again.stderr
    assert len(json.loads((out / "runs.json").read_text())) == 2
    assert _cli("--out", out).returncode == 0                            # rebuild from the registry alone
    empty = _cli("--out", tmp_path / "none")
    assert empty.returncode != 0 and "no run directories" in empty.stderr


# ------------------------------------------------------------- 8. formula variables
def test_formula_text_for_variable_series(cfg):
    expr = "(ts('A','PUT') + ts('A','CALL')) / 2 - rvol(chain('TY','return_index'), 20)"
    row = _meta(ticker="var:ty_vrp", field=expr, kind="level", lag=1)
    lines = formula_text(row, "chg_20", cfg, 1.0)
    assert lines[0] == f"x[t] = {expr}"
    assert lines[1] == "d[t] = x[t] − x[t−20]" and lines[-2] == "pos[t] = +1 × s[t−1] ÷ σ[t]"
    assert len(lines) == len(formula_text(_meta(), "chg_20", cfg, 1.0))          # the other lines are unchanged
    assert formula_text(row, "z_60", cfg, -1.0)[0] == f"x[t] = {expr}"


def test_variable_card_title_and_formula_wraps(tmp_path, cfg, cfg_path):
    panel = make_panel(1, plant=True, short=True)
    expr = "ts('A') - ts('B') + " + " + ".join(f"ts('LONGTICKER{i} Index')" for i in range(12))
    panel.X = panel.X.rename(columns={"sig": "var:sig"})
    panel.meta = panel.meta.rename(index={"sig": "var:sig"})
    panel.meta.loc["var:sig", ["ticker", "field", "description"]] = ["var:sig", expr, "信号变量"]
    d = tmp_path / "run"
    mine(cfg, d, panel=panel, config_path=cfg_path)
    data = build_asset_data(d)
    cards = [f for f in data["factors"] if f["series"] == "var:sig"]
    assert cards
    for f in cards:
        assert f["formula"][0] == f"x[t] = {expr}"
        assert f["title"].startswith("信号变量 · ") and f["description"] == "信号变量"
    build_site([d], tmp_path / "o")
    css = re.search(r"<style>(.*?)</style>", (tmp_path / "o" / "artifact.html").read_text(encoding="utf-8"), re.S).group(1)
    rule = re.search(r"pre\.formula\s*\{([^}]*)\}", css).group(1)
    assert "pre-wrap" in rule and "overflow-wrap: anywhere" in rule
    rep = (d / "report.md").read_text(encoding="utf-8")
    assert "信号变量" in rep.split("## Shortlist")[1].split("## IC 衰减")[0]




# ------------------------------------------------------------- 10. layers: every factor that passed the IS bar
def test_factor_stage_cases():
    base = {"id": "a", "obs_status": "KEEP", "trade_status": "KEEP"}
    assert factor_stage({**base, "obs_status": "DROP", "trade_status": ""}, False) == 1
    assert factor_stage({**base, "trade_status": "DROP"}, False) == 2
    assert factor_stage(base, False) == 3
    assert factor_stage(base, True) == 4
    assert factor_stage({**base, "obs_status": "INSUFFICIENT"}, True) == 4
    old = {"id": "a", "obs_status": "KEEP"}                             # run from before the tradability screen
    assert factor_stage(old, False) == 3 and factor_stage(old, True) == 4
    assert factor_stage({"id": "a", "obs_status": "DROP"}, True) == 1


def test_stage_note_numbers():
    crit = {"obs_min_retention": 0.5, "min_net_sharpe": 0.3}
    assert stage_note({"retention": 0.2345}, 1, crit, {}) == "观察期 IC 保留比例 0.23，低于 0.5"
    assert stage_note({"retention": -0.3}, 1, crit, {}) == "观察期方向相反（保留比例 -0.30）"
    assert stage_note({"sharpe_net_dev": 0.0549}, 2, crit, {}) == "扣成本后净 Sharpe 0.05，低于 0.3"
    assert stage_note({}, 3, crit, "名称 · 20 日变化") == "与「名称 · 20 日变化」高度相关，去重时并入"
    assert stage_note({}, 4, crit, {}) == ""


def test_factors_are_all_non_reject_candidates(run_dir, opened_dir):
    for d_ in (run_dir, opened_dir):
        src = build_asset_data(d_)
        cand = pd.read_csv(d_ / "candidates.csv")
        want = sorted(cand.loc[cand["verdict"] != "REJECT", "id"])
        ids = [f["id"] for f in src["factors"]]
        assert len(ids) == len(set(ids)) and set(ids) <= set(want)         # each at most once
        assert len(ids) + src["n_merged"] == len(want) and not src["skipped"]
        fn = src["funnel"]
        assert fn["n_is"] >= fn["n_obs"] >= fn["n_trade"] >= fn["n_final"]
        by = pd.Series([f["stage"] for f in src["factors"]]).value_counts()
        assert fn == {"n_is": len(ids), "n_obs": int(sum(by.get(k, 0) for k in (2, 3, 4))),
                      "n_trade": int(sum(by.get(k, 0) for k in (3, 4))), "n_final": int(by.get(4, 0))}
        assert not any(k in fn for k in ("tested", "is_pass", "is_weak", "obs_drop"))
        assert [f["stage"] for f in src["factors"]] == sorted((f["stage"] for f in src["factors"]), reverse=True)
        assert any(f["stage"] < 4 for f in src["factors"]) and all(f["is_verdict"] in ("PASS", "WEAK_PASS")
                                                                     for f in src["factors"])
        for f in src["factors"]:
            assert f["nav"]["values"] and f["segments"][0]["n"] > 0 and f["formula"]
            assert bool(f["stage_note"]) == (f["stage"] < 4)
        shortlist = set(pd.read_csv(d_ / "shortlist.csv")["id"])
        assert {f["id"] for f in src["factors"] if f["stage"] == 4} == shortlist   # single run: same de-duplication


@pytest.fixture(scope="module")
def obs_flip_run(tmp_path_factory, cfg, cfg_path):
    """'sig' reverses its relation to the target right after its IS rows: dropped by the observation period."""
    base = make_panel(41, plant=True, short=True, ic_sig=0.15)
    prep = prepare(base, cfg)
    j = list(prep["S"].columns).index("sig|chg_1")
    c1 = int(prep["rows"][int(prep["e1"][j])])
    d = tmp_path_factory.mktemp("flip") / "run"
    mine(cfg, d, panel=make_panel(41, plant=True, short=True, ic_sig=0.15, flip_at=c1), config_path=cfg_path)
    assert len(pd.read_csv(d / "shortlist.csv")) >= 1
    confirm(d)
    return d


@pytest.fixture(scope="module")
def costly_run(tmp_path_factory, panel):
    """Same IC evidence but a high cost: the net Sharpe is too low for the tradability screen."""
    d = tmp_path_factory.mktemp("costly")
    p = d / "c.toml"
    p.write_text(TOML.replace("cost_per_side = 0.00005", f"cost_per_side = {HIGH_COST}")
                 + "\n[verdict]\nis_min_net_sharpe = -1e9\n")      # step-1 Sharpe bar off: isolates step 3
    mine(load_config(p), d / "run", panel=panel, config_path=p)
    return d / "run"


def test_dropped_by_observation_period(obs_flip_run):
    src = build_asset_data(obs_flip_run)
    f = next(x for x in src["factors"] if x["id"] == "sig|chg_1")
    r = pd.read_csv(obs_flip_run / "candidates.csv").set_index("id").loc["sig|chg_1"]
    assert f["stage"] == 1 and r["obs_status"] == "DROP" and r["retention"] < 0
    assert f["stage_note"] == f"观察期方向相反（保留比例 {r['retention']:.2f}）"
    assert f["oos_verdict"] is None and f["oos_confirm"] is None
    assert f["nav"]["values"] and set(f["nav"]["seg"]) == {0, 1, 2} and f["segments"][2]["n"] > 0     # OOS shown
    assert all(x["oos_verdict"] for x in src["factors"] if x["stage"] == 4)


def test_dropped_by_tradability(costly_run):
    src = build_asset_data(costly_run)
    r = pd.read_csv(costly_run / "candidates.csv").set_index("id")
    f = next(x for x in src["factors"] if x["id"] == "sig|chg_1")
    assert f["stage"] == 2 and r.loc["sig|chg_1", "trade_status"] == "DROP"
    lim = json.loads((costly_run / "run.json").read_text())["config"]["verdict"]["min_net_sharpe"]
    assert f["stage_note"] == f"扣成本后净 Sharpe {r.loc['sig|chg_1', 'sharpe_net_dev']:.2f}，低于 {lim:g}"
    assert f["oos_verdict"] is None and f["nav"]["values"] and 2 not in f["nav"]["seg"]
    assert src["funnel"]["n_trade"] < src["funnel"]["n_obs"]


def test_merged_factor_note_names_the_representative(run_dir):
    src = build_asset_data(run_dir)
    merged = [f for f in src["factors"] if f["stage"] == 3]
    assert merged                                                   # the synthetic panel has a near copy of 'sig'
    final_titles = {f["title"] for f in src["factors"] if f["stage"] == 4}
    for f in merged:
        assert f["stage_note"].startswith("与「") and f["stage_note"][2:].split("」")[0] in final_titles


def test_dropped_factor_in_unopened_run_ignores_oos(tmp_path, costly_run, cfg, panel):
    """A dropped factor of an unconfirmed run: no OOS row, and random OOS data changes nothing."""
    ccfg = load_config(costly_run / "config.toml")
    before = build_asset_data(costly_run)
    dropped = [f for f in before["factors"] if f["stage"] == 2]
    assert dropped and all(2 not in f["nav"]["seg"] and f["segments"][2] == {"seg": 2, "name": "样本外", "opened": False}
                           for f in before["factors"])
    cdf = pd.read_csv(costly_run / "candidates.csv")
    prep = prepare(panel, ccfg)
    cols = list(prep["S"].columns)
    c = max(int(prep["rows"][int(prep["e2"][cols.index(i)])]) for i in cdf.loc[cdf["verdict"] != "REJECT", "id"])
    rng = np.random.default_rng(5)
    tgt, X = panel.target.copy(), panel.X.copy()
    tgt.iloc[c + 1:] = rng.normal(0, 0.005, len(tgt) - c - 1)
    for col in X.columns:
        fin = np.isfinite(X[col].to_numpy()) & (np.arange(len(X)) >= c)
        X.loc[fin, col] = rng.normal(0, 50, int(fin.sum()))
    d2 = tmp_path / "run2"
    shutil.copytree(costly_run, d2)
    Panel(target=tgt, X=X, meta=panel.meta, excluded=panel.excluded, info=panel.info).save(d2 / "panel")
    after = build_asset_data(d2)
    assert _strip(after) == _strip(before)


# ------------------------------------------------------------- 11. one library per target, no sources
def _strip(d):
    return {k: v for k, v in d.items() if k not in ("created_at", "runs")}


def _series(rng, n=400, start="2015-01-01"):
    return pd.Series(rng.normal(0, 1, n), index=pd.bdate_range(start, periods=n))


def test_identity_of_signals():
    rng = np.random.default_rng(3)
    a = _series(rng)
    noisy = a + rng.normal(0, 0.0316, len(a))
    assert 0.999 < np.corrcoef(a, noisy)[0, 1] < 0.9999
    assert is_same_factor("chg_20", a, "chg_20", a.copy())
    assert is_same_factor("chg_20", a, "chg_20", -a)                      # reversed sign: still the same factor
    assert is_same_factor("chg_20", a, "chg_20", noisy)
    loose = a + rng.normal(0, 0.1425, len(a))
    assert 0.985 < np.corrcoef(a, loose)[0, 1] < 0.995
    assert not is_same_factor("chg_20", a, "chg_20", loose)               # 0.99: different
    assert not is_same_factor("chg_20", a, "z_60", a.copy())              # transform names differ
    late = a.copy()
    late.index = late.index + pd.offsets.BDay(200)                        # only 200 common dates
    assert len(a.index.intersection(late.index)) < 250
    assert not is_same_factor("chg_20", a, "chg_20", late)
    mid = pd.Series(a.to_numpy(), index=pd.bdate_range("2015-01-01", periods=len(a)))
    assert is_same_factor("chg_20", a, "chg_20", mid.iloc[:260])          # 260 common dates are enough


def test_identity_is_transitive():
    rng = np.random.default_rng(11)
    a = _series(rng, 2000)
    b = a + rng.normal(0, 0.0345, len(a))
    c = b + rng.normal(0, 0.0345, len(a))
    r = lambda x, y: abs(np.corrcoef(x, y)[0, 1])
    assert r(a, b) >= 0.999 and r(b, c) >= 0.999 and r(a, c) < 0.999       # a-c alone would not merge
    other = _series(rng, 2000)
    es = [{"transform": "chg_5", "dev": s} for s in (a, other, b, c)]
    assert group_same_factors(es) == [[0, 2, 3], [1]]


def test_choose_keeper_rules():
    m = lambda **kw: {"opened": False, "run_idx": 0, "z": 3.0, "id": "x", **kw}
    assert choose_keeper([m(run_idx=0), m(run_idx=1)]) == 1                        # newer run
    assert choose_keeper([m(run_idx=2), m(run_idx=1)]) == 0
    assert choose_keeper([m(run_idx=0, opened=True), m(run_idx=1)]) == 0           # opened OOS beats newer
    assert choose_keeper([m(run_idx=0, opened=True), m(run_idx=1, opened=True)]) == 1
    assert choose_keeper([m(z=-4.0), m(z=3.0)]) == 0                               # same run: larger |z|


@pytest.fixture(scope="module")
def alias_dirs(tmp_path_factory, cfg, cfg_path):
    """Run B: the same data as run_dir, but the planted series 'sig' is called 'alias' (identical values) and the
    near copy 'sig2' is absent. Unopened, and an opened copy."""
    base = tmp_path_factory.mktemp("alias")
    pb = make_panel(1, plant=True, short=True)
    pb.X = pb.X.rename(columns={"sig": "alias"})
    pb.meta = pb.meta.rename(index={"sig": "alias"})
    pb.meta.loc["alias", ["ticker", "description"]] = ["alias", "desc alias"]
    mine(cfg, base / "b", panel=pb, config_path=cfg_path)
    shutil.copytree(base / "b", base / "b_open")
    confirm(base / "b_open")
    return base / "b", base / "b_open"


@pytest.fixture(scope="module")
def noisy_run(tmp_path_factory, cfg, cfg_path):
    """Run C: 'alias' is 'sig' plus noise (corr of the signals ~0.86: related, not the same)."""
    base = tmp_path_factory.mktemp("noisy")
    pc = make_panel(1, plant=True, short=True)
    rng = np.random.default_rng(99)
    pc.X["sig"] = pc.X["sig"] + np.cumsum(rng.normal(0, 0.6, len(pc.X)))
    pc.X = pc.X.rename(columns={"sig": "noisy"})
    pc.meta = pc.meta.rename(index={"sig": "noisy"})
    pc.meta.loc["noisy", ["ticker", "description"]] = ["noisy", "desc noisy"]
    mine(cfg, base / "c", panel=pc, config_path=cfg_path)
    assert "noisy|chg_1" in set(pd.read_csv(base / "c" / "shortlist.csv")["id"])
    return base / "c"


def _by_id(data):
    return {f["id"]: f for f in data["factors"]}


def test_same_underlying_signal_appears_once(run_dir, alias_dirs):
    b, _ = alias_dirs
    for order in ([run_dir, b], [b, run_dir]):
        d = build_asset_data(order)
        ids = [f["id"] for f in d["factors"]]
        assert len(ids) == len(set(ids))
        keeper = "alias|chg_1" if order[-1] == b else "sig|chg_1"              # both unopened: the newer run wins
        other = "sig|chg_1" if keeper == "alias|chg_1" else "alias|chg_1"
        assert keeper in ids and other not in ids
        assert other in _by_id(d)[keeper]["also_as"]
        assert not any(i in _by_id(d)[keeper]["also_as"] for i in [keeper])
        assert d["n_merged"] >= 1
        cands = [pd.read_csv(p / "candidates.csv") for p in order]
        total = sum(int((c["verdict"] != "REJECT").sum()) for c in cands)
        assert len(ids) + d["n_merged"] == total                              # merged ones are accounted for


def test_opened_run_wins_the_merge(run_dir, alias_dirs, opened_dir):
    b, b_open = alias_dirs
    d = build_asset_data([b_open, run_dir])                    # opened run first (older), unopened run is newer
    f = _by_id(d)
    assert "alias|chg_1" in f and "sig|chg_1" not in f and "sig|chg_1" in f["alias|chg_1"]["also_as"]
    assert f["alias|chg_1"]["oos_opened"] is True and set(f["alias|chg_1"]["nav"]["seg"]) == {0, 1, 2}
    d = build_asset_data([run_dir, b_open])
    assert "alias|chg_1" in _by_id(d) and "sig|chg_1" not in _by_id(d)
    d = build_asset_data([opened_dir, b])                       # the opened one is the older run and still wins
    f = _by_id(d)
    assert "sig|chg_1" in f and "alias|chg_1" not in f and "alias|chg_1" in f["sig|chg_1"]["also_as"]
    assert f["sig|chg_1"]["oos_opened"] is True


def test_pooled_dedupe_across_runs(run_dir, noisy_run):
    d = build_asset_data([run_dir, noisy_run])
    f = _by_id(d)
    assert "noisy|chg_1" in f and "sig|chg_1" in f                   # related but not the same: both are kept ...
    n, s = f["noisy|chg_1"], f["sig|chg_1"]
    finals = {x["title"]: x for x in d["factors"] if x["stage"] == 4}
    assert n["stage"] == 3 and n["t_dev"] < s["t_dev"]                # ... and the weaker one is merged into a stronger
    rep_title = n["stage_note"][2:].split("」")[0]
    assert n["stage_note"] == f"与「{rep_title}」高度相关，去重时并入" and rep_title in finals
    assert finals[rep_title]["t_dev"] > n["t_dev"]
    assert d["funnel"]["n_final"] == len(finals)


def test_pooled_verdicts_match_confirm_csv(opened_dir, alias_dirs, run_dir):
    b, _ = alias_dirs
    cf = pd.read_csv(opened_dir / "confirm.csv").set_index("id")
    d = build_asset_data([opened_dir, b])
    hit = 0
    for f in d["factors"]:
        if f["stage"] == 4 and f["id"] in cf.index:
            assert f["oos_opened"] and f["oos_verdict"] == cf.loc[f["id"], "verdict"]
            c = f["oos_confirm"]
            assert c["t_signed"] == pytest.approx(cf.loc[f["id"], "t_signed"], rel=1e-9)
            assert c["retention_oos"] == pytest.approx(cf.loc[f["id"], "retention_oos"], rel=1e-9)
            hit += 1
    assert hit >= 2
    for f in d["factors"]:
        if f["stage"] == 4 and not f["oos_opened"]:
            assert f["oos_verdict"] is None and f["oos_confirm"] is None
        if f["stage"] < 4:
            assert f["oos_verdict"] is None
    oc = d["oos_counts"]
    assert oc["unopened"] == sum(1 for f in d["factors"] if f["stage"] == 4 and not f["oos_opened"])
    assert oc["confirmed"] + oc["consistent"] + oc["failed"] + oc["insufficient"] + oc["unopened"] == d["funnel"]["n_final"]


def test_oos_verdict_function():
    from tsminer.confirm import oos_verdict
    assert oos_verdict(float("nan"), 0.1, 0.1, 0.5) == "INSUFFICIENT"
    assert oos_verdict(3.0, 0.1, 0.1, 0.5) == "CONFIRMED"
    assert oos_verdict(0.2, 0.06, 0.1, 0.5) == "CONSISTENT"
    assert oos_verdict(0.2, 0.04, 0.1, 0.5) == "FAILED"
    assert oos_verdict(0.2, -0.06, 0.1, 0.5) == "FAILED"
    assert oos_verdict(3.0, 0.1, 0.1, 0.5, sufficient=False) == "INSUFFICIENT"


def test_unopened_runs_stay_isolated_in_the_pool(tmp_path, panel, cfg, opened_dir, noisy_run):
    b = noisy_run                                               # its factors are not copies of the opened run's
    before = build_asset_data([opened_dir, b])
    unopened = [f for f in before["factors"] if not f["oos_opened"]]
    assert unopened and all(2 not in f["nav"]["seg"] for f in unopened)
    assert any(f["oos_opened"] for f in before["factors"])
    b2 = tmp_path / "b2"
    shutil.copytree(b, b2)
    pb = Panel.load(b / "panel")
    prep = prepare(pb, load_config(b / "config.toml"))
    cols = list(prep["S"].columns)
    cdf = pd.read_csv(b / "candidates.csv")
    c = max(int(prep["rows"][int(prep["e2"][cols.index(i)])]) for i in cdf.loc[cdf["verdict"] != "REJECT", "id"])
    assert c < len(pb.target) - 5
    rng = np.random.default_rng(8)
    tgt, X = pb.target.copy(), pb.X.copy()
    tgt.iloc[c + 1:] = rng.normal(0, 0.005, len(tgt) - c - 1)
    for col in X.columns:
        fin = np.isfinite(X[col].to_numpy()) & (np.arange(len(X)) >= c)
        X.loc[fin, col] = rng.normal(0, 50, int(fin.sum()))
    Panel(target=tgt, X=X, meta=pb.meta, excluded=pb.excluded, info=pb.info).save(b2 / "panel")
    after = build_asset_data([opened_dir, b2])
    assert _strip(after) == _strip(before)                      # the opened run's factors too are unaffected


def test_counts_and_uniqueness(run_dir, alias_dirs, noisy_run, opened_dir):
    b, _ = alias_dirs
    runs = [run_dir, b, noisy_run, opened_dir]
    d = build_asset_data(runs)
    ids = [f["id"] for f in d["factors"]]
    assert len(ids) == len(set(ids))
    fn = d["funnel"]
    assert fn["n_is"] >= fn["n_obs"] >= fn["n_trade"] >= fn["n_final"]
    total = sum(int((pd.read_csv(p / "candidates.csv")["verdict"] != "REJECT").sum()) for p in runs)
    assert fn["n_is"] + d["n_merged"] == total and fn["n_is"] == len(ids)
    assert d["runs"] == [p.name for p in runs] and not d["criteria_mixed"]


def test_criteria_mixed_flag(tmp_path, run_dir):
    other = tmp_path / "other"
    shutil.copytree(run_dir, other)
    rj = json.loads((other / "run.json").read_text())
    rj["config"]["verdict"]["z_weak"] = 1.9
    (other / "run.json").write_text(json.dumps(rj))
    assert build_asset_data([run_dir, other])["criteria_mixed"] is True        # the newest run's gates are used
    assert build_asset_data([other, run_dir])["criteria"]["z_weak"] == json.loads(
        (run_dir / "run.json").read_text())["config"]["verdict"]["z_weak"]


def test_registry(tmp_path, run_dir, alias_dirs):
    b, _ = alias_dirs
    out = tmp_path / "site"
    build_site([run_dir], out)
    ids_a = {f["id"] for f in json.loads((out / "data" / "test.json").read_text(encoding="utf-8"))["factors"]}
    build_site([b], out)                                              # only the new run is given
    d = json.loads((out / "data" / "test.json").read_text(encoding="utf-8"))
    reg = json.loads((out / "runs.json").read_text())
    assert reg == [str(run_dir.resolve()), str(b.resolve())]
    assert d["runs"] == [run_dir.name, b.name] and "alias|chg_1" in {f["id"] for f in d["factors"]} \
        and {i for i in ids_a if not i.startswith("sig|")} <= {f["id"] for f in d["factors"]}
    build_site([run_dir], out)                                        # known run: no duplicate registry entry
    assert len(json.loads((out / "runs.json").read_text())) == 2
    build_site([b], out, reset=True)
    d = json.loads((out / "data" / "test.json").read_text(encoding="utf-8"))
    assert json.loads((out / "runs.json").read_text()) == [str(b.resolve())] and d["runs"] == [b.name]
    assert not any(f["series"] == "sig" for f in d["factors"])
    # a registered directory that is gone: error naming it
    gone = tmp_path / "gone"
    shutil.copytree(b, gone)
    out2 = tmp_path / "site2"
    build_site([gone], out2)
    shutil.rmtree(gone)
    with pytest.raises(ValueError, match=str(gone.resolve())):
        build_site([run_dir], out2)
    build_site([run_dir], out2, reset=True)
    with pytest.raises(ValueError, match="no run directories"):
        build_site([], tmp_path / "empty")
    r = _cli(run_dir, "--out", out2)
    assert r.returncode == 0
    shutil.rmtree(gone, ignore_errors=True)
    reg2 = json.loads((out2 / "runs.json").read_text())
    assert reg2 == [str(run_dir.resolve())]


def test_cli_missing_registered_run_fails(tmp_path, run_dir, alias_dirs):
    b, _ = alias_dirs
    gone = tmp_path / "gone"
    shutil.copytree(b, gone)
    out = tmp_path / "s"
    assert _cli(gone, "--out", out).returncode == 0
    shutil.rmtree(gone)
    r = _cli(run_dir, "--out", out)
    assert r.returncode == 1 and str(gone.resolve()) in r.stderr and "--reset" in r.stderr


def test_no_sources_and_no_candidate_totals(tmp_path, run_dir, alias_dirs):
    b, _ = alias_dirs
    out = tmp_path / "s"
    build_site([run_dir, b], out)
    data = (out / "data" / "test.json").read_text(encoding="utf-8")
    assert '"sources"' not in data and '"tested"' not in data and "检" + "验" not in data
    assert "来源" not in data
    for name in ("artifact.html", "index.html"):
        html = (out / name).read_text(encoding="utf-8")
        assert "检" + "验" not in html and "来源" not in html and "sources" not in html


def test_universe_name_is_not_a_config_key(tmp_path, cfg_path):
    p = tmp_path / "c.toml"
    p.write_text(TOML + '\n[universe]\nname = "x"\n')
    with pytest.raises(ValueError, match="unknown key"):
        load_config(p)
