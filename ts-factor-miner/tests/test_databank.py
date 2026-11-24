"""Tests for tsminer.config and tsminer.databank (pure functions first, one integration test)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from tsminer.config import LagsConfig, load_config
from tsminer.databank import (META_COLUMNS, STALE_LIMIT, align_to_calendar, infer_freq, infer_kind,
                              resolve_lag, resolve_lag_rule)

CONFIGS = ROOT / "configs"


def make_calendar(n=40, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


# ------------------------------------------------------------ align_to_calendar
@pytest.mark.parametrize("lag", [0, 1, 4])
def test_lag_semantics_never_earlier(lag):
    cal = make_calendar()
    # each observation value encodes its own calendar row
    x = pd.Series(np.arange(len(cal), dtype=float), index=cal)
    out = align_to_calendar(x, cal, lag=lag, stale_limit=5)
    for t in range(len(cal)):
        if t < lag:
            assert np.isnan(out.iloc[t])
        else:
            assert out.iloc[t] == t - lag          # appears exactly lag rows later, never earlier


@pytest.mark.parametrize("lag", [0, 1, 4])
def test_single_observation_appears_at_index_plus_lag(lag):
    cal = make_calendar()
    k = 10
    x = pd.Series([7.0], index=[cal[k]])
    out = align_to_calendar(x, cal, lag=lag, stale_limit=3)
    assert out.iloc[: k + lag].isna().all()
    assert out.iloc[k + lag] == 7.0


def test_asof_off_calendar_observation():
    cal = make_calendar()
    fri = 2                                             # 2020-01-03, a Friday
    sat = cal[fri] + pd.Timedelta(days=1)               # Saturday: not on the calendar
    assert cal[fri].dayofweek == 4 and sat.dayofweek == 5
    x = pd.Series([1.0, 2.0], index=[cal[0], sat])
    out = align_to_calendar(x, cal, lag=0, stale_limit=5)
    assert out.iloc[0] == 1.0 and out.iloc[fri] == 1.0   # carried forward on days x has no obs
    assert out.iloc[fri + 1] == 2.0                      # visible on the next calendar day (Monday)
    out1 = align_to_calendar(x, cal, lag=1, stale_limit=5)
    assert out1.iloc[fri + 1] == 1.0 and out1.iloc[fri + 2] == 2.0


def test_obs_before_calendar_start_used_for_first_day():
    cal = make_calendar()
    x = pd.Series([3.0], index=[cal[0] - pd.Timedelta(days=2)])
    out = align_to_calendar(x, cal, lag=0, stale_limit=5)
    assert out.iloc[0] == 3.0
    assert out.iloc[4] == 3.0 and np.isnan(out.iloc[5])   # staleness still applies


def test_stale_truncation_boundary():
    cal = make_calendar()
    k, limit = 5, 5
    x = pd.Series([1.0], index=[cal[k]])
    out = align_to_calendar(x, cal, lag=0, stale_limit=limit)
    assert out.iloc[k + limit] == 1.0                  # exactly at the limit: still valid
    assert np.isnan(out.iloc[k + limit + 1])           # one past: NaN
    out2 = align_to_calendar(x, cal, lag=2, stale_limit=limit)   # truncation happens before the shift
    assert out2.iloc[k + limit + 2] == 1.0 and np.isnan(out2.iloc[k + limit + 3])


def test_stale_limits_constants():
    assert STALE_LIMIT == {"D": 5, "W": 10}


@pytest.mark.parametrize("lag", [0, 1, 4])
def test_no_lookahead_invariance(lag):
    cal = make_calendar(60)
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=len(cal)), index=cal)
    d_idx = 30
    base = align_to_calendar(x, cal, lag, 5)
    x2 = x.copy()
    x2.iloc[d_idx + 1:] = rng.normal(size=len(cal) - d_idx - 1) + 100   # change everything after d
    alt = align_to_calendar(x2, cal, lag, 5)
    safe = d_idx + lag                                  # rows <= index(d)+lag-1 ... plus d's own row
    pd.testing.assert_series_equal(base.iloc[:safe], alt.iloc[:safe])
    assert base.iloc[safe] == alt.iloc[safe]            # d itself is unchanged and visible at d+lag
    assert base.iloc[safe + 1] != alt.iloc[safe + 1]    # sanity: the change shows up one row later


def test_nan_and_duplicate_inputs_and_negative_lag():
    cal = make_calendar()
    x = pd.Series([1.0, np.nan, 2.0, 5.0], index=[cal[0], cal[1], cal[2], cal[2]])
    out = align_to_calendar(x, cal, 0, 5)
    assert out.iloc[1] == 1.0 and out.iloc[2] == 5.0   # NaN ignored; last duplicate wins
    with pytest.raises(ValueError):
        align_to_calendar(x, cal, -1, 5)


# ------------------------------------------------------------------- infer_freq
def test_infer_freq():
    daily = pd.bdate_range("2020-01-01", periods=300)
    daily = daily.delete([10, 11, 12, 100, 101])        # holiday gaps
    assert infer_freq(daily) == "D"
    assert infer_freq(pd.date_range("2020-01-07", periods=100, freq="W-TUE")) == "W"
    assert infer_freq(pd.date_range("2000-01-31", periods=60, freq="ME")) == "other"
    assert infer_freq(pd.bdate_range("2020-01-01", periods=20)) == "other"   # too short


# ----------------------------------------------------------- infer_kind / lag
def test_infer_kind():
    assert infer_kind("price", "price") == "price"
    assert infer_kind("index_pt", "index_level") == "price"
    assert infer_kind("fwd_pts", "price") == "level"
    assert infer_kind("pct", "price") == "level"
    assert infer_kind("pct", "yield") == "level"
    assert infer_kind("contracts", "index_level") == "price"


def test_resolve_lag_priority_and_none():
    lags = LagsConfig(chain=1, class_={"Positioning/CFTC_COT": 4}, ticker={"ABC Index": 9})
    assert resolve_lag("ABC Index", "Positioning", "CFTC_COT", lags) == 9     # ticker wins
    assert resolve_lag("XYZ Index", "Positioning", "CFTC_COT", lags) == 4     # then class
    assert resolve_lag("XYZ Index", "Market_Data", "VOL", lags) is None       # no rule -> None, no default
    assert resolve_lag("XYZ Index", "Positioning", "Other", lags) is None
    assert resolve_lag_rule("ABC Index", "Positioning", "CFTC_COT", lags) == (9, "ticker:ABC Index")
    assert resolve_lag_rule("XYZ Index", "Positioning", "CFTC_COT", lags) == (4, "class:Positioning/CFTC_COT")
    assert resolve_lag_rule("XYZ Index", "Market_Data", "VOL", lags) is None
    # a plain dict is accepted as well
    assert resolve_lag("T", "A", "B", {"chain": 1, "class": {"A/B": 3}, "ticker": {}}) == 3
    assert resolve_lag("T", "A", "C", {"chain": 1}) is None


# ------------------------------------------------------------------ load_config
def _write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


MINIMAL = """
[target]
name = "T"
kind = "futures_chain"
root = "TY"
cost_per_side = 0.0001

[lags]
chain = 1
"""


@pytest.mark.parametrize("name", ["ty", "eurusd"])
def test_load_shipped_configs(name):
    cfg = load_config(CONFIGS / f"{name}.toml")
    assert cfg.target.name.lower() == name
    assert cfg.universe.select and cfg.universe.spread
    assert cfg.lags.chain >= 0
    assert cfg.lags.class_["Positioning/CFTC_COT"] == 4
    assert cfg.lags.ticker["NFCIINDX Index"] == 6
    d = cfg.to_dict()
    assert d["lags"]["class"]["FED/Primary_Dealer"] == 8 and d["lags"]["chain"] == cfg.lags.chain
    json.dumps(d)                                       # must be JSON serialisable
    assert (cfg.sample.is_frac, cfg.sample.obs_frac) == (0.70, 0.15)
    assert cfg.position.buffer == 0.5
    assert (cfg.verdict.obs_min_retention, cfg.verdict.year_hit_min) == (0.5, 0.6)


@pytest.mark.parametrize("name", ["ty", "eurusd"])
def test_shipped_configs_availability_table_and_universe(name):
    cfg = load_config(CONFIGS / f"{name}.toml")
    assert cfg.lags.class_["Market_Data/Government_Bond"] == 0 and cfg.lags.class_["Market_Data/VOL"] == 1
    assert cfg.lags.class_["Macro/US"] == 1
    assert cfg.lags.class_["FED/Balance_Sheet"] == 2
    sel = {(e.get("class_two"), str(e.get("class_three"))): e for e in cfg.universe.select}
    for e in cfg.universe.select:
        assert "MM_Future" not in e.get("class_three", []) and "Commodity_Future" not in e.get("class_three", [])
        if e.get("class_two") == "Commodity":
            assert e["class_three"] == ["Commodity_Index", "Spot"]
    assert any(e.get("class_two") == "Commodity" for e in cfg.universe.select)


def test_minimal_config_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.stats.seed == 20261122 and cfg.verdict.dedupe_corr == 0.7 and cfg.stats.min_obs == 250
    assert cfg.lags.chain == 1 and cfg.lags.class_ == {} and cfg.lags.ticker == {}
    assert (cfg.sample.is_frac, cfg.sample.obs_frac) == (0.70, 0.15) and cfg.position.buffer == 0.5
    assert (cfg.verdict.obs_min_retention, cfg.verdict.year_hit_min) == (0.5, 0.6)


def test_lags_section_is_an_input(tmp_path):
    base = MINIMAL.split("[lags]")[0]
    for bad in ("D = 1", "W = 6", "frequency_default = 1"):       # frequency defaults no longer exist
        with pytest.raises(ValueError):
            load_config(_write(tmp_path, base + f"[lags]\nchain = 1\n{bad}\n"))
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, base + "[lags]\n[lags.class]\n\"A/B\" = 1\n"))    # chain missing
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, base))                                                  # no [lags] at all
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, base + "[lags]\nchain = -1\n"))
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, base + '[lags]\nchain = 1\n[lags.class]\n"bad" = 1\n'))
    ok = load_config(_write(tmp_path, base + '[lags]\nchain = 2\n[lags.class]\n"A/B" = 3\n[lags.ticker]\n"X" = 0\n'))
    assert ok.lags.chain == 2 and ok.lags.class_ == {"A/B": 3} and ok.lags.ticker == {"X": 0}


OLD_KEY = "hold" + "out_frac"        # the removed two-segment key must now be rejected as unknown


def test_sample_fractions(tmp_path):
    s = lambda body: load_config(_write(tmp_path, MINIMAL + f"\n[sample]\n{body}\n"))
    assert s("is_frac = 0.6\nobs_frac = 0.2").sample.obs_frac == 0.2
    for bad in ("is_frac = 0.8\nobs_frac = 0.2", "is_frac = 0.9\nobs_frac = 0.3", "is_frac = 0\nobs_frac = 0.2",
                "is_frac = 0.7\nobs_frac = 0", "is_frac = -0.1", OLD_KEY + " = 0.2", "is_frac = 0.7\n" + OLD_KEY + " = 0.2",
                'is_frac = "x"'):
        with pytest.raises(ValueError):
            s(bad)


def test_position_and_verdict_new_keys(tmp_path):
    assert load_config(_write(tmp_path, MINIMAL + "\n[position]\nbuffer = 0\n")).position.buffer == 0
    for bad in ("buffer = -0.1", 'buffer = "x"', "size = 1"):
        with pytest.raises(ValueError):
            load_config(_write(tmp_path, MINIMAL + f"\n[position]\n{bad}\n"))
    v = load_config(_write(tmp_path, MINIMAL + "\n[verdict]\nobs_min_retention = 0.3\nyear_hit_min = 0.5\n")).verdict
    assert (v.obs_min_retention, v.year_hit_min) == (0.3, 0.5)


def test_verdict_fields(tmp_path):
    v = load_config(_write(tmp_path, MINIMAL)).verdict
    assert (v.z_pass, v.z_weak, v.dedupe_corr, v.extra_lag_t_min) == (2.5, 2.0, 0.7, 2.0)
    v = load_config(_write(tmp_path, MINIMAL + '\n[verdict]\nz_pass = 3\nz_weak = 1.5\n')).verdict
    assert (v.z_pass, v.z_weak) == (3, 1.5)
    for bad in ('mode = "per_factor"', 'fwer_alpha = 0.05', 'fdr_q = 0.1', 'z_pass = 2.0\nz_weak = 2.0',
                'z_pass = 1.5\nz_weak = 2.0', 'z_weak = 0', 'z_weak = -1', 'z_pass = 2.0', 'z_pass = "x"'):
        with pytest.raises(ValueError):
            load_config(_write(tmp_path, MINIMAL + f"\n[verdict]\n{bad}\n"))


@pytest.mark.parametrize("name", ["ty", "eurusd"])
def test_shipped_configs_verdict_fields(name):
    cfg = load_config(CONFIGS / f"{name}.toml")
    v = cfg.verdict
    assert (v.z_pass, v.z_weak) == (2.5, 2.0)
    assert cfg.stats.min_obs == 250


def test_missing_required(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, '[sample]\nis_frac = 0.7\n'))                # no [target]
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, '[target]\nname = "T"\nkind = "spot"\n[lags]\nchain = 1\n'))     # no cost_per_side
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, '[target]\nname="T"\nkind="futures_chain"\ncost_per_side=0.1\n[lags]\nchain=1\n'))  # no root
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, '[target]\nname="T"\nkind="spot"\ncost_per_side=0.1\n[lags]\nchain=1\n'))  # no ticker/field


def test_unknown_section_and_key(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, MINIMAL + "\n[bogus]\nx = 1\n"))
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, MINIMAL + "\n[stats]\nn_bootstrap = 5\n"))


@pytest.mark.parametrize("entry", [
    'class_two = "FX"\ntickers = ["A"]',                      # class + tickers
    'family = "CFTC"\nclass_two = "FX"\nattrs = {a="b"}',     # family + class
    'family = "CFTC"',                                        # family without attrs
    'tickers = ["A"]\nattrs = {a="b"}',                       # tickers + attrs
    'foo = "bar"',                                            # nothing recognised
])
def test_select_form_mixing(tmp_path, entry):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, MINIMAL + f"\n[[universe.select]]\n{entry}\n"))


def test_select_valid_forms(tmp_path):
    txt = MINIMAL + ('\n[[universe.select]]\nclass_two = "FX"\nclass_three = ["A", "B"]\n'
                     '\n[[universe.select]]\nfamily = "CFTC"\nattrs = {direction="Net"}\n'
                     '\n[[universe.select]]\ntickers = ["X"]\n')
    assert len(load_config(_write(tmp_path, txt)).universe.select) == 3


def test_env_override(tmp_path, monkeypatch):
    p = _write(tmp_path, MINIMAL)
    monkeypatch.setenv("DATABANK_TOOLKIT", "/x/toolkit")
    monkeypatch.setenv("DATABANK_CACHE", "/x/cache")
    cfg = load_config(p)
    assert cfg.databank.toolkit_dir == "/x/toolkit" and cfg.databank.cache_dir == "/x/cache"
    monkeypatch.delenv("DATABANK_TOOLKIT")
    monkeypatch.delenv("DATABANK_CACHE")
    assert load_config(p).databank.toolkit_dir != "/x/toolkit"


# ------------------------------------------------------------------ integration
def _db_cfg():
    cfg = load_config(CONFIGS / "ty.toml")
    cur = Path(cfg.databank.cache_dir) / "current.json"
    if not cur.exists() or not Path(cfg.databank.toolkit_dir).exists():
        pytest.skip("databank not available")
    db = json.loads(cur.read_text()).get("database_path", "")
    if not os.path.exists(db):
        pytest.skip("databank file not available")
    return cfg, db


def test_build_panel_ty_integration():
    cfg, db = _db_cfg()
    from tsminer.databank import build_panel

    panel = build_panel(cfg)
    assert panel.X.index.equals(panel.target.index)
    assert list(panel.meta.index) == list(panel.X.columns)
    assert list(panel.meta.columns) == META_COLUMNS and "lag_rule" in panel.meta.columns
    assert "chain:TY" in panel.X.columns
    assert panel.X.columns.is_unique
    allowed = {cfg.lags.chain, *cfg.lags.class_.values(), *cfg.lags.ticker.values()}
    assert set(panel.meta["lag"]) <= allowed
    assert panel.meta["lag_rule"].notna().all() and (panel.meta["lag_rule"] != "").all()
    assert panel.meta.loc["chain:TY", "lag_rule"] == "chain"
    # every non-chain, non-spread series reports the rule it actually hit, consistent with its lag
    for sid, r in panel.meta.iterrows():
        if r["group"] in ("Chain", "Carry"):
            assert r["lag_rule"] == "chain" and r["lag"] == cfg.lags.chain
        elif r["group"] == "Spread":
            legs = r["lag_rule"].split("+")
            assert len(legs) == 2
        elif r["lag_rule"].startswith("ticker:"):
            assert r["lag"] == cfg.lags.ticker[r["lag_rule"][len("ticker:"):]]
        else:
            assert r["lag_rule"].startswith("class:") and r["lag"] == cfg.lags.class_[r["lag_rule"][len("class:"):]]
    # near-month continuous contracts are out of the pool
    assert not any(isinstance(t, str) and "MM_Future" in t for t in panel.meta["group"])

    # lag semantics on real data: X["chain:TY"][t] == TY return_index at calendar row t-lag (lag from the config)
    lag = cfg.lags.chain
    sys.path.insert(0, cfg.databank.toolkit_dir)
    from data_query import DataQuery
    q = DataQuery(db_path=db)
    ch = q.get_futures_chain("TY")
    ch = ch[ch["return_valid"] == 1]
    ri = pd.Series(ch["return_index"].to_numpy(), index=pd.to_datetime(ch["trade_date"]))
    x = panel.X["chain:TY"]
    for t in [lag, lag + 1, lag + 2, 50, 1000, 3000, len(x) - 1]:
        assert x.iloc[t] == pytest.approx(ri.loc[panel.X.index[t - lag]])
    assert x.iloc[:lag].isna().all()


def test_build_panel_rejects_series_without_availability_rule():
    """Remove the class rule of VOL: no VOL series may enter X; they are listed as no_availability_rule."""
    import dataclasses

    cfg, db = _db_cfg()
    from tsminer.databank import build_panel

    sys.path.insert(0, cfg.databank.toolkit_dir)
    from data_query import DataQuery
    q = DataQuery(db_path=db)
    vol = {s["ticker"] for s in q.list_securities(class_one="Market_Data", class_two="VOL")}
    assert vol

    class_ = {k: v for k, v in cfg.lags.class_.items() if k != "Market_Data/VOL"}
    assert len(class_) == len(cfg.lags.class_) - 1
    cfg2 = dataclasses.replace(cfg, lags=dataclasses.replace(cfg.lags, class_=class_))
    panel = build_panel(cfg2)
    assert not (vol & set(panel.X.columns))
    assert not (panel.meta["group"] == "VOL").any()
    rej = panel.excluded[panel.excluded["reason"] == "no_availability_rule"]
    assert len(rej) > 0 and set(rej["ticker"]) & vol
    assert not (set(rej["ticker"]) & set(panel.X.columns))        # rejected series are not in X
    assert not any(r.startswith("class:Market_Data/VOL") for r in panel.meta["lag_rule"])
    assert panel.excluded.groupby("ticker").size().max() == 1


def test_build_panel_spread_leg_without_rule_raises():
    import dataclasses

    cfg, db = _db_cfg()
    from tsminer.databank import build_panel

    class_ = {k: v for k, v in cfg.lags.class_.items() if k != "Market_Data/Government_Bond"}
    cfg2 = dataclasses.replace(cfg, lags=dataclasses.replace(cfg.lags, class_=class_))
    with pytest.raises(ValueError, match="availability"):
        build_panel(cfg2)


# ------------------------------------------------------- formula variables: config
def test_load_variable_and_helper_config(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL + (
        "\n[lags.field]\n\"PX_LAST\" = 0\n[lags.curve]\n\"C\" = 1\n"
        "\n[universe]\nfields = [\"PX_LAST\"]\n"
        "[[universe.helper]]\nid = \"h1\"\nexpr = \"ts('A')\"\n"
        "[[universe.variable]]\nid = \"v1\"\nexpr = \"ref('h1') * 2\"\ngroup = \"G\"\ndescription = \"中文\"\n"
        "[[universe.variable]]\nid = \"v2\"\nexpr = \"ts('B')\"\nkind = \"price\"\n")))
    u = cfg.universe
    assert u.select == [] and [h["id"] for h in u.helper] == ["h1"]
    assert u.variable[0] == {"id": "v1", "expr": "ref('h1') * 2", "kind": "level", "group": "G", "description": "中文"}
    assert u.variable[1]["kind"] == "price" and u.variable[1]["description"] == "v2"
    assert cfg.lags.field_ == {"PX_LAST": 0} and cfg.lags.curve == {"C": 1}
    d = cfg.to_dict()
    assert d["lags"]["field"] == {"PX_LAST": 0} and d["lags"]["curve"] == {"C": 1}
    json.dumps(d)


def _vcfg(tmp_path, body):
    return load_config(_write(tmp_path, MINIMAL + "\n[universe]\nfields = [\"PX_LAST\"]\n" + body))


@pytest.mark.parametrize("body", [
    '[[universe.variable]]\nid = "a"\nexpr = "ts(\'X\')"\n[[universe.variable]]\nid = "a"\nexpr = "ts(\'Y\')"\n',   # dup variable
    '[[universe.helper]]\nid = "a"\nexpr = "ts(\'X\')"\n[[universe.helper]]\nid = "a"\nexpr = "ts(\'Y\')"\n',      # dup helper
    '[[universe.helper]]\nid = "a"\nexpr = "ts(\'X\')"\n[[universe.variable]]\nid = "a"\nexpr = "ts(\'Y\')"\n',    # across
    '[[universe.variable]]\nid = "a"\nexpr = "ts(\'X\')"\nkind = "weird"\n',                                       # bad kind
    '[[universe.variable]]\nid = "a"\n',                                                                           # no expr
    '[[universe.helper]]\nid = "a"\n',
    '[[universe.variable]]\nexpr = "ts(\'X\')"\n',                                                                 # no id
    '[[universe.variable]]\nid = "a-b"\nexpr = "ts(\'X\')"\n',                                                     # bad id
    '[[universe.variable]]\nid = "中文"\nexpr = "ts(\'X\')"\n',
    '[[universe.variable]]\nid = "a"\nexpr = "ts(\'X\') ** 2"\n',                                                  # unsafe expr
    '[[universe.variable]]\nid = "a"\nexpr = "__import__(\'os\')"\n',
    '[[universe.variable]]\nid = "a"\nexpr = "ref(\'zzz\')"\n',                                                    # unknown ref
    '[[universe.variable]]\nid = "a"\nexpr = "ref(\'b\')"\n[[universe.variable]]\nid = "b"\nexpr = "ref(\'a\')"\n', # cycle
    '[[universe.variable]]\nid = "a"\nexpr = "ts(\'X\')"\nbogus = 1\n',                                            # unknown key
    '[[universe.helper]]\nid = "a"\nexpr = "ts(\'X\')"\nkind = "level"\n',
])
def test_bad_variable_config_raises(tmp_path, body):
    with pytest.raises(ValueError):
        _vcfg(tmp_path, body)


def test_bad_lags_field_curve_raise(tmp_path):
    for bad in ('[lags.field]\n"X" = -1\n', '[lags.curve]\n"X" = "a"\n', '[lags.field]\n"X" = 1.5\n'):
        with pytest.raises(ValueError):
            load_config(_write(tmp_path, MINIMAL + "\n" + bad))


def test_ty_vars_config_loads():
    cfg = load_config(CONFIGS / "ty_vars.toml")
    assert len(cfg.universe.variable) == 63 and cfg.universe.select == [] and cfg.universe.chains == []
    assert cfg.universe.spread == [] and cfg.universe.carry_chains == []
    assert cfg.universe.fields == ["PX_LAST", "CFTC_CONTRACTS"]
    ids = [v["id"] for v in cfg.universe.variable]
    assert len(set(ids)) == 63 and {"slope_2s10s", "ff_3m", "ty_oi"} <= set(ids)
    assert cfg.lags.field_["FUND_FLOW"] == 1 and cfg.lags.field_["OPEN_INT"] == 1
    assert "chain:oi_active" not in cfg.lags.field_
    assert cfg.lags.curve == {"USD_SOFR_OIS": 0} and cfg.lags.ticker["NFCIINDX Index"] == 6
    base = load_config(CONFIGS / "ty.toml")
    assert cfg.lags.class_ == base.lags.class_ and cfg.target == base.target and cfg.verdict == base.verdict
    assert cfg.signals == base.signals and cfg.stats == base.stats and cfg.sample == base.sample


@pytest.mark.parametrize("name", ["ty", "eurusd"])
def test_old_configs_have_no_variables_and_unchanged_lags(name):
    cfg = load_config(CONFIGS / f"{name}.toml")
    assert cfg.universe.variable == [] and cfg.universe.helper == []
    # only the documentation-verified fields / curve are present (rules in place for later candidates)
    assert cfg.lags.field_ == {"FUND_FLOW": 1, "1M_CALL_IMP_VOL_25DELTA_DFLT": 1, "1M_PUT_IMP_VOL_25DELTA_DFLT": 1,
                               "OPEN_INT": 1}
    assert cfg.lags.curve == {"USD_SOFR_OIS": 0}
    assert len(cfg.universe.select) > 0 and len(cfg.universe.spread) == 3


FF = [f"FF{i} Comdty" for i in range(1, 25)]
COMMODITY = [f"{c} Index" for c in ("SPGSAGTR SPGSALTR SPGSCITR SPGSCLTR SPGSCRTR SPGSENTR SPGSINTR SPGSNETR "
                                    "SPGSPMTR SPGSRETR SPGSUETR SPGSXNTR SPGSXPTR").split()]
GS = [f"{c} Index" for c in "GSPRHIMO GSPRHVMS GSXUCYCL GSXUDEFS GSXUMFML GSXUMFMS".split()]


@pytest.mark.parametrize("name", ["ty", "eurusd", "ty_vars"])
def test_shipped_configs_final_lag_values_and_provenance(name):
    cfg = load_config(CONFIGS / f"{name}.toml")
    lg = cfg.lags
    assert lg.chain == 0
    assert lg.class_ == {
        "Market_Data/Government_Bond": 0, "Market_Data/Money_Market": 0, "Market_Data/Swap_OIS": 0,
        "Market_Data/Commodity": 0, "Market_Data/Equity": 0, "Market_Data/Inflation_Linked": 0,
        "Market_Data/Credit": 0, "Market_Data/MBS": 0, "Market_Data/FX": 0, "Market_Data/VOL": 1,
        "Macro/Global": 1, "Macro/US": 1, "Positioning/CFTC_COT": 4, "FED/Balance_Sheet": 2,
        "FED/Primary_Dealer": 8}
    t = lg.ticker
    assert t["NFCIINDX Index"] == 6 and t["CESIUSD Index"] == 1 and t["CESIG10 Index"] == 1
    assert t["FEDL01 Index"] == 1 and t["FDTR Index"] == 0 and t["FDTRFTRL Index"] == 0
    for tk in FF + COMMODITY + GS:
        assert t[tk] == 0, tk
    assert t["FF1 Comdty"] == t["FF13 Comdty"] == t["FF24 Comdty"] == 0
    assert t["SPGSINTR Index"] == 0 and t["GSPRHVMS Index"] == 0
    assert len(t) == 49
    assert lg.field_["FUND_FLOW"] == 1 and lg.field_["OPEN_INT"] == 1 and lg.curve == {"USD_SOFR_OIS": 0}
    assert lg.field_["1M_CALL_IMP_VOL_25DELTA_DFLT"] == lg.field_["1M_PUT_IMP_VOL_25DELTA_DFLT"] == 1
    # provenance lists: every key exists, researcher-confirmed = chain + the ten Market_Data classes
    ver = set(lg.verified)
    assert {"field:FUND_FLOW", "field:OPEN_INT", "field:1M_CALL_IMP_VOL_25DELTA_DFLT",
            "field:1M_PUT_IMP_VOL_25DELTA_DFLT", "curve:USD_SOFR_OIS", "ticker:CESIUSD Index",
            "ticker:CESIG10 Index", "ticker:FDTR Index", "ticker:FDTRFTRL Index", "ticker:FEDL01 Index"} <= ver
    assert {f"ticker:{x}" for x in FF + COMMODITY + GS} <= ver
    assert set(lg.confirmed) == {"chain"} | {f"class:{k}" for k in lg.class_ if k.startswith("Market_Data/")}
    assert len(lg.confirmed) == 11 and not (ver & set(lg.confirmed))
    drafts = {"class:Macro/Global", "class:Macro/US", "class:Positioning/CFTC_COT", "class:FED/Balance_Sheet",
              "class:FED/Primary_Dealer", "ticker:NFCIINDX Index"}
    assert not (drafts & (ver | set(lg.confirmed)))
    if name == "ty_vars":
        assert not ({"field:EQY_SH_OUT", "field:FUND_NET_ASSET_VAL", "field:CONVENTIONAL_CTD_FORWARD_FRSK",
                     "field:VOLUME"} & (ver | set(lg.confirmed)))
    assert json.loads(json.dumps(cfg.to_dict()))["lags"]["verified"] == lg.verified


def test_lags_verified_confirmed_loading_and_validation(tmp_path):
    base = MINIMAL.split("[lags]")[0]
    ok = load_config(_write(tmp_path, base + (
        '[lags]\nchain = 1\nverified = ["field:F", "ticker:T", "curve:C"]\nconfirmed = ["chain", "class:A/B"]\n'
        '[lags.class]\n"A/B" = 1\n[lags.ticker]\n"T" = 0\n[lags.field]\n"F" = 1\n[lags.curve]\n"C" = 0\n')))
    assert ok.lags.verified == ["field:F", "ticker:T", "curve:C"] and ok.lags.confirmed == ["chain", "class:A/B"]
    assert load_config(_write(tmp_path, MINIMAL)).lags.verified == []
    assert load_config(_write(tmp_path, MINIMAL)).lags.confirmed == []
    for bad in ('verified = ["ticker:NOPE"]', 'confirmed = ["class:X/Y"]', 'verified = ["field:F"]',
                'confirmed = ["curve:C"]', 'verified = ["NFCIINDX Index"]', 'verified = "chain"', "verified = [1]"):
        with pytest.raises(ValueError):
            load_config(_write(tmp_path, base + f"[lags]\nchain = 1\n{bad}\n[lags.ticker]\n\"T\" = 0\n"))


def test_resolve_lag_rule_field_priority():
    lags = LagsConfig(chain=0, class_={"A/B": 4}, ticker={"T": 9}, field_={"F": 2})
    assert resolve_lag_rule("T", "A", "B", lags, "F") == (9, "ticker:T")
    assert resolve_lag_rule("U", "A", "B", lags, "F") == (2, "field:F")
    assert resolve_lag_rule("U", "A", "B", lags, "G") == (4, "class:A/B")
    assert resolve_lag_rule("U", "A", "B", lags) == (4, "class:A/B")           # no field given: field rules unused
    assert resolve_lag_rule("U", "X", "Y", lags, "G") is None
    assert resolve_lag_rule("U", "X", "Y", {"chain": 0, "field": {"F": 7}}, "F") == (7, "field:F")


# --------------------------------------------------- formula variables: real databank
def test_build_panel_ty_vars_integration():
    cfg0, db = _db_cfg()
    from tsminer.databank import _fetch, build_panel

    cfg = load_config(CONFIGS / "ty_vars.toml")
    panel = build_panel(cfg)
    assert len(panel.X.columns) > 0 and all(c.startswith("var:") for c in panel.X.columns)
    assert panel.X.index.equals(panel.target.index) and list(panel.meta.index) == list(panel.X.columns)
    assert list(panel.meta.columns) == META_COLUMNS
    assert (panel.meta["ticker"] == panel.meta.index).all()
    assert panel.excluded["ticker"].str.startswith("var:").all() if len(panel.excluded) else True
    defs = {v["id"]: v for v in cfg.universe.variable}
    assert set(panel.X.columns) | set(panel.excluded["ticker"]) == {f"var:{i}" for i in defs}
    r = panel.meta.loc["var:slope_2s10s"]
    assert r["field"] == defs["slope_2s10s"]["expr"] and r["description"] == defs["slope_2s10s"]["description"]
    assert r["lag_rule"] == "class:Market_Data/Government_Bond" and r["lag"] == 0 and r["freq"] == "D"
    assert len(panel.X.columns) == 63
    # ticker rule beats the class rule; a formula of two FF legs is made of two ticker rules
    cesi = panel.meta.loc["var:cesi_us"]
    assert cesi["lag_rule"] == "ticker:CESIUSD Index" and cesi["lag"] == 1
    ff = panel.meta.loc["var:ff_3m"]
    assert ff["lag_rule"] == "ticker:FF1 Comdty+ticker:FF4 Comdty" and ff["lag"] == 0

    sys.path.insert(0, cfg.databank.toolkit_dir)
    from data_query import DataQuery
    q = DataQuery(db_path=db)
    cal = panel.X.index

    def al(tk, lag):
        s = _fetch(q, [tk], "PX_LAST")[tk][0]
        return align_to_calendar(s, cal, lag, STALE_LIMIT["D"])

    lag = int(r["lag"])
    exp = al("USGG10YR Index", lag) - al("USGG2YR Index", lag)
    got = panel.X["var:slope_2s10s"]
    m = exp.notna()
    assert m.sum() > 1000
    np.testing.assert_allclose(got[m].to_numpy(), exp[m].to_numpy(), rtol=0, atol=1e-12)
    assert got[~m].isna().all()

    lag = int(panel.meta.at["var:ff_3m", "lag"])
    exp = al("FF1 Comdty", lag) - al("FF4 Comdty", lag)
    got = panel.X["var:ff_3m"]
    m = exp.notna()
    assert m.sum() > 1000
    np.testing.assert_allclose(got[m].to_numpy(), exp[m].to_numpy(), rtol=0, atol=1e-12)
    assert got[~m].isna().all()


def test_build_panel_variable_without_rule_is_excluded_with_leaf_named():
    import dataclasses
    from tsminer.databank import build_panel

    cfg0, db = _db_cfg()
    cfg = load_config(CONFIGS / "ty_vars.toml")
    class_ = {k: v for k, v in cfg.lags.class_.items() if k != "Market_Data/Money_Market"}
    # the FF / policy-rate leaves now have ticker rules (which beat the class rule): drop those as well
    ticker = {k: v for k, v in cfg.lags.ticker.items() if not k.startswith(("FF", "FDTR", "FEDL01"))}
    assert len(ticker) < len(cfg.lags.ticker)
    cfg2 = dataclasses.replace(cfg, lags=dataclasses.replace(cfg.lags, class_=class_, ticker=ticker))
    panel = build_panel(cfg2)
    rej = panel.excluded[panel.excluded["reason"].str.startswith("no_availability_rule")]
    assert "var:ff_3m" in set(rej["ticker"]) and "var:policy_path" in set(rej["ticker"])
    reason = rej.set_index("ticker").at["var:ff_3m", "reason"]
    assert "FF1 Comdty/PX_LAST" in reason and "FF4 Comdty/PX_LAST" in reason
    assert "var:ff_3m" not in panel.X.columns and "var:slope_2s10s" in panel.X.columns
    assert panel.excluded.groupby("ticker").size().max() == 1


def test_futures_oi_matches_manual_sum_on_real_db():
    cfg, db = _db_cfg()
    from tsminer.databank import DBLeaves, _fetch

    cfg = load_config(CONFIGS / "ty_vars.toml")
    sys.path.insert(0, cfg.databank.toolkit_dir)
    from data_query import DataQuery
    q = DataQuery(db_path=db)
    cal = pd.DatetimeIndex(q.get_futures_chain("TY").query("return_valid == 1")["trade_date"].pipe(pd.to_datetime))
    leaf = DBLeaves(q, cfg, cal, {}).futures_oi("TY")
    assert leaf.info.lag == 1 and leaf.info.rule == "field:OPEN_INT"
    recs = q.resolve("BOND_FUTURE", allow_multi=True, root="TY")
    conc = [r["ticker"] for r in recs if "contract" in r["attrs"]]
    assert "TY1 Comdty" not in conc and "TY1 COMB Comdty" not in conc and len(conc) > 50
    got = _fetch(q, conc, "OPEN_INT")
    df = pd.DataFrame({k: v[0] for k, v in got.items()})
    for day in ("2004-11-24", "2004-11-26", "2015-03-12", "2024-11-26"):
        d = pd.Timestamp(day)
        exp = df.loc[d].sum(min_count=1)
        i = cal.get_loc(d)
        assert leaf.series.iloc[i + 1] == pytest.approx(exp)          # lag 1: visible on the next calendar row
        assert 1 <= int(df.loc[d].notna().sum()) <= 5
