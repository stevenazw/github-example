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
from tsminer.site import (SEG_NAMES, build_asset_data, build_site, direction_text, downsample_idx, factor_nav,
                          flag_texts, formula_text, slugify)
from tsminer.databank import Panel

from test_pipeline import TOML, _cut_panel, make_panel

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_HOSTS = {"cdnjs.cloudflare.com", "cdn.jsdelivr.net", "unpkg.com", "fonts.googleapis.com", "fonts.gstatic.com"}


@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("sitecfg") / "test.toml"
    p.write_text(TOML)
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
    assert locked_data["oos_opened"] is False and locked_data["oos_counts"] is None
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
    assert opened_data["oos_opened"] is True
    cf = pd.read_csv(opened_dir / "confirm.csv").set_index("id")
    assert opened_data["factors"]
    for f in opened_data["factors"]:
        assert set(f["nav"]["seg"]) == {0, 1, 2}
        assert f["oos_verdict"] == cf.loc[f["id"], "verdict"]
        assert f["segments"][2]["n"] > 0
    cnt = cf.loc[[f["id"] for f in opened_data["factors"]], "verdict"].value_counts()
    assert opened_data["oos_counts"]["confirmed"] == int(cnt.get("CONFIRMED", 0))
    order = {"CONFIRMED": 0, "CONSISTENT": 1, "INSUFFICIENT": 2, "FAILED": 3}
    ranks = [order[f["oos_verdict"]] for f in opened_data["factors"]]
    assert ranks == sorted(ranks)


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


def test_formula_z_and_position_sign(cfg):
    t = "\n".join(formula_text(_meta(), "z_252", cfg, -1.0))
    assert "mean_252(" in t and "std_252(" in t
    assert "pos[t] = −1 × s[t] ÷ σ[t]" in t
    assert "pos[t] = +1 × s[t] ÷ σ[t]" in "\n".join(formula_text(_meta(), "z_60", cfg, 1.0))


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
        assert data["slug"] == slug and data["factors"]
    # same template: the page content of both files is identical
    style = art[art.index("<style>"):art.index("</style>")]
    assert style in idx and art[art.index('<div id="app">'):] in idx
    assert "fetch('data/'" in art


# ------------------------------------------------------------------ 5b. criteria and the tradability step
def test_new_run_has_criteria_and_trade_keep(site_dir, run_dir, locked_data):
    vd = load_config(run_dir / "config.toml").verdict
    want = {"z_weak": vd.z_weak, "obs_min_retention": vd.obs_min_retention, "min_net_sharpe": vd.min_net_sharpe}
    assert want["min_net_sharpe"] == 0.3
    assert locked_data["criteria"] == want
    run = json.loads((run_dir / "run.json").read_text())
    assert locked_data["funnel"]["trade_keep"] == run["funnel"]["trade_keep"]
    for slug in ("test", "other-1"):
        d = json.loads((site_dir / "data" / f"{slug}.json").read_text(encoding="utf-8"))
        assert d["criteria"] == want and d["funnel"]["trade_keep"] == run["funnel"]["trade_keep"]
    art = (site_dir / "artifact.html").read_text(encoding="utf-8")
    assert "入选条件" in art and "criteria" in art
    assert "cost" + "_negative" not in art and "扣成本后收益为负" not in art
    assert "min_net_sharpe" in art and "扣成本后净 Sharpe ≥ " in art


def test_old_style_run_without_trade_keys(tmp_path, run_dir):
    old = tmp_path / "old"
    shutil.copytree(run_dir, old)
    rj = json.loads((old / "run.json").read_text())
    del rj["funnel"]["trade_keep"], rj["funnel"]["trade_drop"], rj["config"]["verdict"]["min_net_sharpe"]
    (old / "run.json").write_text(json.dumps(rj))
    out = tmp_path / "site"
    build_site([old], out)                                           # must not raise
    d = json.loads((out / "data" / "test.json").read_text(encoding="utf-8"))
    assert "trade_keep" not in d["funnel"] and "min_net_sharpe" not in d["criteria"]
    assert d["criteria"]["z_weak"] == rj["config"]["verdict"]["z_weak"]
    assert d["factors"]


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
def test_duplicate_target_raises(tmp_path, run_dir, opened_dir):
    with pytest.raises(ValueError, match="duplicate target"):
        build_site([run_dir, opened_dir], tmp_path / "o")


def test_cli(tmp_path, run_dir, opened_dir):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    ok = subprocess.run([sys.executable, "-m", "tsminer", "site", str(run_dir), "--out", str(tmp_path / "s")],
                        cwd=ROOT, env=env, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert (tmp_path / "s" / "index.html").exists() and (tmp_path / "s" / "data" / "test.json").exists()
    dup = subprocess.run([sys.executable, "-m", "tsminer", "site", str(run_dir), str(opened_dir), "--out",
                          str(tmp_path / "s2")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert dup.returncode != 0 and "duplicate" in dup.stderr


# ------------------------------------------------------------- 8. formula variables
def test_formula_text_for_variable_series(cfg):
    expr = "(ts('A','PUT') + ts('A','CALL')) / 2 - rvol(chain('TY','return_index'), 20)"
    row = _meta(ticker="var:ty_vrp", field=expr, kind="level", lag=1)
    lines = formula_text(row, "chg_20", cfg, 1.0)
    assert lines[0] == f"x[t] = {expr}"
    assert lines[1] == "d[t] = x[t] − x[t−20]" and lines[-2] == "pos[t] = +1 × s[t] ÷ σ[t]"
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
