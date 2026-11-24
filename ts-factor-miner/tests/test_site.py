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
                          factor_stage, flag_texts, formula_text, infer_source_name, slugify, stage_note)
from tsminer.databank import Panel

from test_pipeline import HIGH_COST, TOML, _cut_panel, make_panel

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
    """The single source of a single-run asset (the per-source dict)."""
    return build_asset_data(run_dir)["sources"][0]


@pytest.fixture(scope="module")
def opened_data(opened_dir):
    return build_asset_data(opened_dir)["sources"][0]


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
    b = build_asset_data(d2)["sources"][0]
    fa = next(f for f in locked_data["factors"] if f["id"] == cid)
    fb = next(f for f in b["factors"] if f["id"] == cid)
    assert fa == fb


def test_opened(opened_dir, opened_data):
    assert opened_data["oos_opened"] is True
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
        assert data["slug"] == slug and len(data["sources"]) == 1 and data["sources"][0]["factors"]
        assert data["summary"]["n_factors"] == len(data["sources"][0]["factors"])
    # same template: the page content of both files is identical
    style = art[art.index("<style>"):art.index("</style>")]
    assert style in idx and art[art.index('<div id="app">'):] in idx
    assert "fetch('data/'" in art


# ------------------------------------------------------------------ 5b. criteria and the tradability step
def test_new_run_has_criteria_and_trade_keep(site_dir, run_dir, locked_data):
    vd = load_config(run_dir / "config.toml").verdict
    want = {"z_weak": vd.z_weak, "obs_min_retention": vd.obs_min_retention, "min_net_sharpe": vd.min_net_sharpe,
            "dedupe_corr": vd.dedupe_corr}
    assert want["min_net_sharpe"] == 0.3
    assert locked_data["criteria"] == want
    run = json.loads((run_dir / "run.json").read_text())
    assert locked_data["funnel"]["n_final"] == run["funnel"]["shortlist"]
    for slug in ("test", "other-1"):
        d = json.loads((site_dir / "data" / f"{slug}.json").read_text(encoding="utf-8"))["sources"][0]
        assert d["criteria"] == want and d["funnel"]["n_final"] == run["funnel"]["shortlist"]
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
    cdf = pd.read_csv(old / "candidates.csv")
    assert not (cdf["trade_status"] == "DROP").any()
    cdf.drop(columns=["trade_status", "sharpe_gross_dev", "sharpe_net_dev"]).to_csv(old / "candidates.csv", index=False)
    out = tmp_path / "site"
    build_site([old], out)                                           # must not raise
    d = json.loads((out / "data" / "test.json").read_text(encoding="utf-8"))["sources"][0]
    assert "min_net_sharpe" not in d["criteria"] and all(f["stage"] != 2 for f in d["factors"])
    assert d["funnel"]["n_final"] == len(pd.read_csv(old / "shortlist.csv"))
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
def test_duplicate_source_name_raises(tmp_path, run_dir, opened_dir):
    """Same target and same (inferred) source name: refuse, and say how to tell the runs apart."""
    with pytest.raises(ValueError, match=r"duplicate source name.*universe"):
        build_site([run_dir, opened_dir], tmp_path / "o")
    assert not (tmp_path / "o" / "artifact.html").exists()


def test_cli(tmp_path, run_dir, opened_dir):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    ok = subprocess.run([sys.executable, "-m", "tsminer", "site", str(run_dir), "--out", str(tmp_path / "s")],
                        cwd=ROOT, env=env, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert (tmp_path / "s" / "index.html").exists() and (tmp_path / "s" / "data" / "test.json").exists()
    dup = subprocess.run([sys.executable, "-m", "tsminer", "site", str(run_dir), str(opened_dir), "--out",
                          str(tmp_path / "s2")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert dup.returncode != 0 and "duplicate source name" in dup.stderr and "universe" in dup.stderr


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
    data = build_asset_data(d)["sources"][0]
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


# ------------------------------------------------------------- 9. several sources per target
def _named(src, dst, name):
    """Copy of run directory ``src`` whose config.toml carries ``[universe] name``."""
    shutil.copytree(src, dst)
    cp = dst / "config.toml"
    cp.write_text(cp.read_text(encoding="utf-8") + f'\n[universe]\nname = "{name}"\n', encoding="utf-8")
    return dst


@pytest.fixture(scope="module")
def ab_dirs(tmp_path_factory, run_dir, opened_dir):
    """A: confirmed, B: not confirmed; same target and same factor ids."""
    base = tmp_path_factory.mktemp("ab")
    return _named(opened_dir, base / "a", "A"), _named(run_dir, base / "b", "B")


def test_source_name_inference():
    assert infer_source_name("自定义", ["var:x"]) == "自定义"
    assert infer_source_name("  ", ["var:x"]) == "变量清单"
    assert infer_source_name("", ["a", "var:x"]) == "变量清单"
    assert infer_source_name(None, ["a", "chain:TY"]) == "原始序列扫描"
    assert infer_source_name("", []) == "原始序列扫描"


def test_two_sources_one_target(tmp_path, ab_dirs, run_dir):
    a, b = ab_dirs
    ref = build_asset_data(run_dir)["sources"][0]
    for order in ([a, b], [b, a]):
        out = tmp_path / "".join(p.name for p in order)
        build_site(order, out)
        files = sorted(p.name for p in (out / "data").iterdir())
        assert files == ["test.json"]
        d = json.loads((out / "data" / "test.json").read_text(encoding="utf-8"))
        assert [s["name"] for s in d["sources"]] == [p.name.upper() for p in order]
        assert [s["run_dir"] for s in d["sources"]] == [p.name for p in order]
        assert d["summary"]["n_factors"] == sum(len(s["factors"]) for s in d["sources"])
        for s in d["sources"]:
            assert s["funnel"] == ref["funnel"] and s["criteria"] == ref["criteria"]
            assert {f["id"] for f in s["factors"]} == {f["id"] for f in ref["factors"]}
            if not s["oos_opened"]:                                # same order as the run on its own
                assert s["factors"] == ref["factors"]
            assert (s["is_frac"], s["obs_frac"]) == (ref["is_frac"], ref["obs_frac"])
        man = json.loads(re.search(r"var MANIFEST = (.*?);\n", (out / "artifact.html").read_text(encoding="utf-8")).group(1))
        assert [s["name"] for s in man["assets"][0]["sources"]] == [p.name.upper() for p in order]


def test_sources_are_isolated_from_each_other(tmp_path, ab_dirs, run_dir, opened_dir):
    a, b = ab_dirs
    d = build_asset_data([a, b])
    sa, sb = d["sources"]
    assert sa["oos_opened"] is True and sb["oos_opened"] is False
    assert sa["oos_counts"] is not None and sb["oos_counts"] is None
    assert all(set(f["nav"]["seg"]) == {0, 1, 2} and bool(f["oos_verdict"]) == (f["stage"] == 4) for f in sa["factors"])
    assert sb["factors"] and all(2 not in f["nav"]["seg"] and f["oos_verdict"] is None and f["oos_confirm"] is None
                                 and f["segments"][2] == {"seg": 2, "name": "样本外", "opened": False}
                                 for f in sb["factors"])
    # each source equals what it is on its own: an opened neighbour changes nothing
    nokey = lambda src: {k: v for k, v in src.items() if k != "key"}
    assert nokey(sb) == nokey(build_asset_data(b)["sources"][0]) and nokey(sa) == nokey(build_asset_data(a)["sources"][0])
    # asset-level totals only count the opened source
    assert d["summary"]["oos_counts"] == sa["oos_counts"] and d["summary"]["n_opened"] == 1
    assert d["summary"]["n_factors"] == len(sa["factors"]) + len(sb["factors"])


def test_unconfirmed_source_ignores_random_oos_rows(tmp_path, ab_dirs, cfg, panel):
    a, b = ab_dirs
    before = build_asset_data([a, b])["sources"][1]
    prep = prepare(panel, cfg)
    cols = list(prep["S"].columns)
    cdf = pd.read_csv(b / "candidates.csv")
    ids = cdf.loc[cdf["verdict"] != "REJECT", "id"]            # every factor shown, not just the shortlist
    c = max(int(prep["rows"][int(prep["e2"][cols.index(i)])]) for i in ids)     # from here on: OOS for every factor
    assert c < len(panel.target) - 5
    rng = np.random.default_rng(31337)
    tgt, X = panel.target.copy(), panel.X.copy()
    tgt.iloc[c + 1:] = rng.normal(0, 0.005, len(tgt) - c - 1)
    for col in X.columns:
        fin = np.isfinite(X[col].to_numpy()) & (np.arange(len(X)) >= c)
        X.loc[fin, col] = rng.normal(0, 50, int(fin.sum()))
    Panel(target=tgt, X=X, meta=panel.meta, excluded=panel.excluded, info=panel.info).save(b / "panel")
    try:
        after = build_asset_data([a, b])["sources"][1]
    finally:
        panel.save(b / "panel")
    assert after["oos_opened"] is False and after == before
    assert any(f["stage"] < 4 for f in after["factors"])        # dropped factors are part of the comparison
    assert all(2 not in f["nav"]["seg"] for f in after["factors"])


def test_same_factor_ids_in_two_sources(tmp_path, ab_dirs):
    a, b = ab_dirs
    out = tmp_path / "s"
    build_site([a, b], out)                                              # must not raise
    d = json.loads((out / "data" / "test.json").read_text(encoding="utf-8"))
    ids = [{f["id"] for f in s["factors"]} for s in d["sources"]]
    assert ids[0] & ids[1]                                                # the same ids do occur twice
    keys = [s["key"] for s in d["sources"]]
    assert len(set(keys)) == 2
    assert len({(s["key"], f["id"]) for s in d["sources"] for f in s["factors"]}) == d["summary"]["n_factors"]
    art = (out / "artifact.html").read_text(encoding="utf-8")
    man = json.loads(re.search(r"var MANIFEST = (.*?);\n", art).group(1))
    assert [s["key"] for s in man["assets"][0]["sources"]] == keys
    assert "'card-' + a.slug + '-' + s.key + '-'" in art                  # DOM ids carry the source key


def test_variable_source_name_from_var_series(tmp_path, cfg, cfg_path, run_dir):
    panel = make_panel(1, plant=True, short=True)
    panel.X = panel.X.rename(columns={"sig": "var:sig"})
    panel.meta = panel.meta.rename(index={"sig": "var:sig"})
    panel.meta.loc["var:sig", ["ticker", "field", "description"]] = ["var:sig", "ts('A')", "信号变量"]
    d = tmp_path / "vars"
    mine(cfg, d, panel=panel, config_path=cfg_path)
    out = tmp_path / "o"
    build_site([run_dir, d], out)
    data = json.loads((out / "data" / "test.json").read_text(encoding="utf-8"))
    assert [s["name"] for s in data["sources"]] == ["原始序列扫描", "变量清单"]
    assert all(f["series"] != "var:sig" for f in data["sources"][0]["factors"])


def test_two_targets_with_sources(tmp_path, ab_dirs, opened_dir):
    a, b = ab_dirs
    other = tmp_path / "other"
    shutil.copytree(opened_dir, other)
    rj = json.loads((other / "run.json").read_text())
    rj["target"] = "OTHER-1"
    (other / "run.json").write_text(json.dumps(rj))
    out = tmp_path / "s"
    r = build_site([a, other, b], out)
    assert [x["slug"] for x in r["assets"]] == ["test", "other-1"]
    assert [len(x["sources"]) for x in r["assets"]] == [2, 1]


# ------------------------------------------------------------- 10. layers: every factor that passed the IS bar
def test_factor_stage_cases():
    base = {"id": "a", "obs_status": "KEEP", "trade_status": "KEEP", "rep": "a"}
    assert factor_stage({**base, "obs_status": "DROP", "trade_status": "", "rep": ""}, {"a"}) == 1
    assert factor_stage({**base, "trade_status": "DROP", "rep": ""}, set()) == 2
    assert factor_stage({**base, "rep": "b"}, {"b"}) == 3
    assert factor_stage(base, {"a"}) == 4
    assert factor_stage({**base, "obs_status": "INSUFFICIENT"}, {"a"}) == 4
    old = {"id": "a", "obs_status": "KEEP", "rep": "b"}                 # run from before the tradability screen
    assert factor_stage(old, {"b"}) == 3 and factor_stage({**old, "rep": "a"}, {"a"}) == 4
    assert factor_stage({"id": "a", "obs_status": "DROP"}, set()) == 1


def test_stage_note_numbers():
    crit = {"obs_min_retention": 0.5, "min_net_sharpe": 0.3}
    assert stage_note({"retention": 0.2345}, 1, crit, {}) == "观察期 IC 保留比例 0.23，低于 0.5"
    assert stage_note({"retention": -0.3}, 1, crit, {}) == "观察期方向相反（保留比例 -0.30）"
    assert stage_note({"sharpe_net_dev": 0.0549}, 2, crit, {}) == "扣成本后净 Sharpe 0.05，低于 0.3"
    assert stage_note({"rep": "r|chg_1"}, 3, crit, {"r|chg_1": "名称 · 20 日变化"}) == "与「名称 · 20 日变化」高度相关，去重时并入"
    assert stage_note({}, 4, crit, {}) == ""


def test_factors_are_all_non_reject_candidates(ab_dirs, run_dir, opened_dir):
    for d_ in (run_dir, opened_dir):
        src = build_asset_data(d_)["sources"][0]
        cand = pd.read_csv(d_ / "candidates.csv")
        want = sorted(cand.loc[cand["verdict"] != "REJECT", "id"])
        assert sorted(f["id"] for f in src["factors"]) == want                  # each exactly once
        assert not src["skipped"]
        fn = src["funnel"]
        assert fn["n_is"] >= fn["n_obs"] >= fn["n_trade"] >= fn["n_final"]
        assert fn["n_final"] == len(pd.read_csv(d_ / "shortlist.csv"))
        by = pd.Series([f["stage"] for f in src["factors"]]).value_counts()
        assert fn == {"n_is": len(want), "n_obs":
                      int(sum(by.get(k, 0) for k in (2, 3, 4))),
                      "n_trade": int(sum(by.get(k, 0) for k in (3, 4))), "n_final": int(by.get(4, 0))}
        keys = ("tested", "is_pass", "is_weak", "obs_drop")
        assert not any(k in src["funnel"] for k in keys)
        assert [f["stage"] for f in src["factors"]] == sorted((f["stage"] for f in src["factors"]), reverse=True)
        assert any(f["stage"] < 4 for f in src["factors"]) and all(f["is_verdict"] in ("PASS", "WEAK_PASS")
                                                                     for f in src["factors"])
        for f in src["factors"]:
            assert f["nav"]["values"] and f["segments"][0]["n"] > 0 and f["formula"]
            assert bool(f["stage_note"]) == (f["stage"] < 4)


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
    p.write_text(TOML.replace("cost_per_side = 0.00005", f"cost_per_side = {HIGH_COST}"))
    mine(load_config(p), d / "run", panel=panel, config_path=p)
    return d / "run"


def test_dropped_by_observation_period(obs_flip_run):
    src = build_asset_data(obs_flip_run)["sources"][0]
    f = next(x for x in src["factors"] if x["id"] == "sig|chg_1")
    r = pd.read_csv(obs_flip_run / "candidates.csv").set_index("id").loc["sig|chg_1"]
    assert f["stage"] == 1 and r["obs_status"] == "DROP" and r["retention"] < 0
    assert f["stage_note"] == f"观察期方向相反（保留比例 {r['retention']:.2f}）"
    assert f["oos_verdict"] is None and f["oos_confirm"] is None
    assert f["nav"]["values"] and set(f["nav"]["seg"]) == {0, 1, 2} and f["segments"][2]["n"] > 0     # OOS shown
    assert all(x["oos_verdict"] for x in src["factors"] if x["stage"] == 4)


def test_dropped_by_tradability(costly_run):
    src = build_asset_data(costly_run)["sources"][0]
    r = pd.read_csv(costly_run / "candidates.csv").set_index("id")
    f = next(x for x in src["factors"] if x["id"] == "sig|chg_1")
    assert f["stage"] == 2 and r.loc["sig|chg_1", "trade_status"] == "DROP"
    lim = json.loads((costly_run / "run.json").read_text())["config"]["verdict"]["min_net_sharpe"]
    assert f["stage_note"] == f"扣成本后净 Sharpe {r.loc['sig|chg_1', 'sharpe_net_dev']:.2f}，低于 {lim:g}"
    assert f["oos_verdict"] is None and f["nav"]["values"] and 2 not in f["nav"]["seg"]
    assert src["funnel"]["n_trade"] < src["funnel"]["n_obs"]


def test_merged_factor_note_names_the_representative(run_dir):
    src = build_asset_data(run_dir)["sources"][0]
    merged = [f for f in src["factors"] if f["stage"] == 3]
    assert merged                                                   # the synthetic panel has a near copy of 'sig'
    titles = {f["id"]: f["title"] for f in src["factors"]}
    rep = pd.read_csv(run_dir / "candidates.csv").set_index("id")["rep"]
    for f in merged:
        assert f["stage_note"] == f"与「{titles[rep[f['id']]]}」高度相关，去重时并入"


def test_dropped_factor_in_unopened_run_ignores_oos(tmp_path, costly_run, cfg, panel):
    """A dropped factor of an unconfirmed run: no OOS row, and random OOS data changes nothing."""
    ccfg = load_config(costly_run / "config.toml")
    before = build_asset_data(costly_run)["sources"][0]
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
    after = build_asset_data(d2)["sources"][0]
    assert {**after, "run_dir": ""} == {**before, "run_dir": ""}


def test_no_candidate_totals_anywhere(tmp_path, ab_dirs):
    out = tmp_path / "s"
    build_site(list(ab_dirs), out)
    data = (out / "data" / "test.json").read_text(encoding="utf-8")
    assert '"tested"' not in data and "检" + "验" not in data
    for name in ("artifact.html", "index.html"):
        assert "检" + "验" not in (out / name).read_text(encoding="utf-8")
