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
from tsminer.databank import (STALE_LIMIT, align_to_calendar, infer_freq, infer_kind,
                              resolve_lag)

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


def test_resolve_lag_priority():
    lags = LagsConfig(D=1, W=6, class_={"Positioning/CFTC_COT": 4}, ticker={"ABC Index": 9})
    assert resolve_lag("ABC Index", "Positioning", "CFTC_COT", "W", lags) == 9     # ticker wins
    assert resolve_lag("XYZ Index", "Positioning", "CFTC_COT", "W", lags) == 4     # then class
    assert resolve_lag("XYZ Index", "Market_Data", "VOL", "W", lags) == 6          # then freq default
    assert resolve_lag("XYZ Index", "Market_Data", "VOL", "D", lags) == 1


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
"""


@pytest.mark.parametrize("name", ["ty", "eurusd"])
def test_load_shipped_configs(name):
    cfg = load_config(CONFIGS / f"{name}.toml")
    assert cfg.target.name.lower() == name
    assert cfg.universe.select and cfg.universe.spread
    assert cfg.lags.class_["Positioning/CFTC_COT"] == 4
    d = cfg.to_dict()
    assert d["lags"]["class"]["FED/Primary_Dealer"] == 8
    json.dumps(d)                                       # must be JSON serialisable
    assert load_config(CONFIGS / "eurusd.toml").sample.holdout_frac == 0.3


def test_minimal_config_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.stats.seed == 20261122 and cfg.lags.W == 6 and cfg.verdict.fdr_q == 0.10


def test_verdict_new_fields(tmp_path):
    v = load_config(_write(tmp_path, MINIMAL)).verdict
    assert (v.mode, v.z_pass, v.z_weak) == ("per_factor", 2.5, 2.0)
    v = load_config(_write(tmp_path, MINIMAL + '\n[verdict]\nmode = "family"\nz_pass = 3\nz_weak = 1.5\n')).verdict
    assert (v.mode, v.z_pass, v.z_weak) == ("family", 3, 1.5)
    for bad in ('mode = "bogus"', 'mode = 1', 'z_pass = 2.0\nz_weak = 2.0', 'z_pass = 1.5\nz_weak = 2.0',
                'z_weak = 0', 'z_weak = -1', 'z_pass = 2.0'):
        with pytest.raises(ValueError):
            load_config(_write(tmp_path, MINIMAL + f"\n[verdict]\n{bad}\n"))


@pytest.mark.parametrize("name", ["ty", "eurusd"])
def test_shipped_configs_verdict_fields(name):
    v = load_config(CONFIGS / f"{name}.toml").verdict
    assert (v.mode, v.z_pass, v.z_weak) == ("per_factor", 2.5, 2.0)


def test_missing_required(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, '[sample]\nholdout_frac = 0.2\n'))            # no [target]
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, '[target]\nname = "T"\nkind = "spot"\n'))     # no cost_per_side
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, '[target]\nname="T"\nkind="futures_chain"\ncost_per_side=0.1\n'))  # no root
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, '[target]\nname="T"\nkind="spot"\ncost_per_side=0.1\n'))  # no ticker/field


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
def test_build_panel_ty_integration():
    cfg = load_config(CONFIGS / "ty.toml")
    cur = Path(cfg.databank.cache_dir) / "current.json"
    if not cur.exists() or not Path(cfg.databank.toolkit_dir).exists():
        pytest.skip("databank not available")
    db = json.loads(cur.read_text()).get("database_path", "")
    if not os.path.exists(db):
        pytest.skip("databank file not available")
    from tsminer.databank import build_panel

    panel = build_panel(cfg)
    assert panel.X.index.equals(panel.target.index)
    assert list(panel.meta.index) == list(panel.X.columns)
    assert "chain:TY" in panel.X.columns
    assert panel.X.columns.is_unique
    assert set(panel.meta["lag"]) <= {1, 2, 4, 6, 8}

    # lag semantics on real data: X["chain:TY"][t] == TY return_index at calendar row t-1
    sys.path.insert(0, cfg.databank.toolkit_dir)
    from data_query import DataQuery
    q = DataQuery(db_path=db)
    ch = q.get_futures_chain("TY")
    ch = ch[ch["return_valid"] == 1]
    ri = pd.Series(ch["return_index"].to_numpy(), index=pd.to_datetime(ch["trade_date"]))
    x = panel.X["chain:TY"]
    for t in [1, 2, 50, 1000, 3000, len(x) - 1]:
        assert x.iloc[t] == pytest.approx(ri.loc[panel.X.index[t - 1]])
    assert np.isnan(x.iloc[0])
