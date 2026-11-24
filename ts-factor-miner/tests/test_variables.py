"""Tests for tsminer.variables: whitelist expression engine, flow alignment, availability of leaves."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from tsminer.config import LagsConfig
from tsminer.databank import DBLeaves, align_flow_to_calendar, align_to_calendar
from tsminer.variables import Evaluator, Leaf, LeafInfo, Rejected, parse_expr, validate_definitions

CAL = pd.bdate_range("2020-01-01", periods=60)
nan = np.nan


def S(values, cal=CAL):
    return pd.Series(np.asarray(values, dtype=float), index=cal[:len(values)] if len(values) < len(cal) else cal)


class Fake:
    """Leaf provider that serves prepared series; leaf name -> (series, lag, rule, freq)."""

    def __init__(self, data, calendar=CAL, missing=(), badfreq=()):
        self.calendar = calendar
        self.data = data
        self.missing, self.badfreq = set(missing), set(badfreq)

    def _leaf(self, name, key=None):
        if name in self.missing:
            raise Rejected("no_availability_rule", name)
        if name in self.badfreq:
            raise Rejected("freq_not_daily_or_weekly", name)
        if name not in self.data:
            raise ValueError(f"no data for {name}")
        s, lag, rule, freq = self.data[name]
        s = s.reindex(self.calendar)
        return Leaf(s, LeafInfo(name, lag, rule, freq))

    def ts(self, ticker, field=None):
        return self._leaf(f"{ticker}/{field or 'PX_LAST'}")

    def flow(self, ticker, field):
        return self._leaf(f"flow:{ticker}/{field}")

    def chain(self, root, column):
        return self._leaf(f"chain:{root}/{column}")

    def curve(self, name, measure, a, b):
        return self._leaf(f"curve:{name}/{measure}/{a}/{b}")


def ev(expr, data=None, defs=None, **kw):
    data = data if data is not None else {}
    defs = dict(defs or {})
    defs["v"] = expr
    return Evaluator(Fake(data, **kw), defs).evaluate("v")


A = {"A/PX_LAST": (S([1, 2, 3, 4, 5, 6.0]), 0, "class:X/Y", "D")}


def vals(res):
    return res.series.to_numpy()[:8]


# ------------------------------------------------------------------ 1. safety
@pytest.mark.parametrize("expr", [
    "__import__('os')",
    "ts.__class__",
    "a + b",
    "ts('x')[0]",
    "foo(ts('x'))",
    "'a' + 1",
    "ts('x') + 'a'",
    "ts('x') ** 2",
    "ts('x') % 2",
    "ts('x') // 2",
    "ts('x') > 1",
    "(lambda: 1)()",
    "ts('x', field='PX_LAST')",
    "ts(1)",
    "ts('x') if 1 else 2",
    "[1, 2]",
    "True + 1",
    "lag(ts('x'), -1)",
    "lag(ts('x'), 1.5)",
    "lag(ts('x'), n)",
    "ts('x').lag",
    "ts(*['x'])",
    "ref('a', 'b')",
    "curve('a', 'b')",
    "rvol(ts('x'), 1)",
    "~ts('x')",
    "",
    "ts('x') +",
    "1e999",
])
def test_unsafe_or_invalid_expressions_rejected(expr):
    with pytest.raises(ValueError, match="myvar"):
        parse_expr(expr, "myvar")


def test_error_message_names_the_variable():
    with pytest.raises(ValueError) as e:
        ev("ts('A') + zzz", defs={}, data=A)
    assert "variable 'v'" in str(e.value)


def test_constant_expression_is_not_a_series():
    with pytest.raises(ValueError):
        ev("1 + 2")
    with pytest.raises(ValueError):
        ev("nz(3)")


def test_missing_leaf_data_fails_loudly_with_variable_id_and_leaf():
    with pytest.raises(ValueError) as e:
        ev("ts('NOPE')", data=A)
    assert "variable 'v'" in str(e.value) and "NOPE" in str(e.value)


# -------------------------------------------------- 2. arithmetic and operations
B = {"B/PX_LAST": (S([2, 4, 0, nan, 10, 5.0]), 0, "class:X/Y", "D")}


def test_arithmetic_and_nan_propagation():
    d = {**A, **B}
    r = ev("ts('A') + ts('B')", d)
    np.testing.assert_array_equal(vals(r)[:6], [3, 6, 3, nan, 15, 11])
    r = ev("2*ts('A') - ts('B') / 2 + (-1)", d)
    np.testing.assert_allclose(vals(r)[:6], [2 - 1 - 1, 4 - 2 - 1, 6 - 0 - 1, nan, 10 - 5 - 1, 12 - 2.5 - 1])
    assert np.isnan(vals(ev("ts('A') * ts('B')", d))[3])
    assert ev("+ts('A')", d).series.iloc[0] == 1.0


def test_division_by_zero_and_nonfinite_become_nan():
    d = {**A, **B}
    r = ev("ts('A') / ts('B')", d)
    np.testing.assert_allclose(vals(r)[:6], [0.5, 0.5, nan, nan, 0.5, 1.2], equal_nan=True)
    assert np.isnan(ev("1 / (ts('A') - ts('A'))", d).series.iloc[:6]).all()


def test_nz_sign_lag_diff():
    d = {**A, **B}
    np.testing.assert_array_equal(vals(ev("nz(ts('B'))", d))[:6], [2, 4, 0, 0, 10, 5])
    x = {"C/PX_LAST": (S([-3, 0, 2, nan, 1.0]), 0, "r", "D")}
    np.testing.assert_array_equal(vals(ev("sign(ts('C'))", x))[:5], [-1, 0, 1, nan, 1])
    np.testing.assert_array_equal(vals(ev("lag(ts('A'), 2)", d))[:6], [nan, nan, 1, 2, 3, 4])
    np.testing.assert_array_equal(vals(ev("lag(ts('A'), 0)", d))[:3], [1, 2, 3])
    np.testing.assert_array_equal(vals(ev("diff(ts('B'))", d))[:6], [nan, 2, -4, nan, nan, -5])


def test_dlog_nonpositive_is_nan():
    x = {"P/PX_LAST": (S([1, 2, 0, 3, -1, 4, 8.0]), 0, "r", "D")}
    r = vals(ev("dlog(ts('P'))", x))
    assert np.isnan(r[0]) and r[1] == pytest.approx(np.log(2))
    assert np.isnan(r[2]) and np.isnan(r[3]) and np.isnan(r[4]) and np.isnan(r[5])      # 0 / -1 poison both sides
    assert r[6] == pytest.approx(np.log(2))


# --------------------------------------------------------------- 3. cumsum, rvol
def test_cumsum_nan_before_first_finite_then_nan_as_zero():
    x = {"C/PX_LAST": (S([nan, nan, 1, nan, 2, 3.0]), 0, "r", "D")}
    r = vals(ev("cumsum(ts('C'))", x))[:6]
    np.testing.assert_array_equal(r, [nan, nan, 1, nan, 3, 6])      # NaN row stays NaN, the sum carries on
    allnan = {"C/PX_LAST": (S([nan, nan, nan]), 0, "r", "D")}
    assert np.isnan(ev("cumsum(ts('C'))", allnan).series).all()


def test_rvol_matches_hand_computation_and_window():
    rng = np.random.default_rng(1)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(CAL))))
    x = {"P/PX_LAST": (pd.Series(p, index=CAL), 0, "r", "D")}
    n = 5
    r = ev(f"rvol(ts('P'), {n})", x).series
    lr = np.diff(np.log(p))                       # lr[k] is the log change into row k+1
    assert r.iloc[:n].isna().all()                # needs n dlogs: first finite at row n
    for t in (n, n + 3, 40):
        w = lr[t - n:t]
        assert r.iloc[t] == pytest.approx(np.std(w, ddof=1) * np.sqrt(252) * 100)


# ------------------------------------------------------------------ 4. causality
def _truncate_data(data, t0):
    return {k: (s.iloc[:t0 + 1], lag, rule, freq) for k, (s, lag, rule, freq) in data.items()}


CAUSAL = [
    "cumsum(ts('P') - ts('Q'))",
    "rvol(ts('P'), 10)",
    "lag(ts('P'), 3) / ts('Q')",
    "dlog(ts('P')) * 100",
    "cumsum(lag(sign(dlog(ts('P'))), 1) * dlog(ts('Q')))",
    "ref('h') - rvol(ts('P'), 7)",
    "cumsum(nz(ref('h')) / (nz(ts('Q')) + 1))",
    "diff(ts('P')) + sign(ts('Q') - ts('P'))",
    "clean(ts('P') - ts('Q'), 10, 3)",
    "cumsum(clean(dlog(ts('P')), 8, 4)) + cumsum(ts('Q'))",
    "nz(clean(ts('Q'), 6, 2)) + lag(ts('P'), 2)",
]


@pytest.mark.parametrize("expr", CAUSAL)
def test_causality_truncated_equals_prefix(expr):
    rng = np.random.default_rng(7)
    n = len(CAL)
    p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=CAL)
    q = pd.Series(50 + np.cumsum(rng.normal(0, 1, n)), index=CAL)
    q.iloc[[5, 6, 20]] = nan
    data = {"P/PX_LAST": (p, 0, "a", "D"), "Q/PX_LAST": (q, 1, "b", "D")}
    defs = {"h": "ts('P') - ts('Q')"}
    full = Evaluator(Fake(data), {**defs, "v": expr}).evaluate("v").series
    for t0 in (15, 30, 45):
        cut = Evaluator(Fake(_truncate_data(data, t0), calendar=CAL[:t0 + 1]), {**defs, "v": expr}).evaluate("v").series
        pd.testing.assert_series_equal(cut, full.iloc[:t0 + 1], check_exact=True, check_freq=False)


# --------------------------------------------------------------- 5. flow alignment
def test_flow_non_calendar_day_goes_to_next_calendar_row():
    cal = pd.bdate_range("2020-01-01", periods=10)            # Wed 1 Jan .. ; row 2 = Fri 3 Jan
    sat = cal[2] + pd.Timedelta(days=1)
    assert cal[3].dayofweek == 0 and sat.dayofweek == 5
    x = pd.Series([1.0, 2.0, 4.0], index=[cal[1], sat, cal[5]])
    out = align_flow_to_calendar(x, cal, 0)
    assert np.isnan(out.iloc[0])
    np.testing.assert_array_equal(out.iloc[1:6].to_numpy(), [1, 0, 2, 0, 4])      # Saturday's 2 lands on Monday (row 3)
    np.testing.assert_array_equal(out.iloc[6:].to_numpy(), [0] * 4)               # no forward fill


def test_flow_same_row_sums_and_first_row_nan_before():
    cal = pd.bdate_range("2020-01-01", periods=8)
    x = pd.Series([1.0, 2.0, 3.0, 5.0], index=pd.DatetimeIndex([cal[2], cal[2], cal[3], cal[4]]))
    out = align_flow_to_calendar(x, cal, 0)
    assert out.iloc[:2].isna().all()
    np.testing.assert_array_equal(out.iloc[2:].to_numpy(), [3, 3, 5, 0, 0, 0])
    # a weekend pair lands on one row and is summed
    assert cal[2].dayofweek == 4                            # Friday; the weekend falls on row 3 (Monday)
    sat, sun = cal[2] + pd.Timedelta(days=1), cal[2] + pd.Timedelta(days=2)
    y = pd.Series([1.0, 10.0, 100.0], index=[cal[2], sat, sun])
    o = align_flow_to_calendar(y, cal, 0)
    assert o.iloc[2] == 1.0 and o.iloc[3] == 110.0 and o.iloc[4] == 0.0


def test_flow_lag_and_no_lookahead():
    cal = pd.bdate_range("2020-01-01", periods=30)
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=len(cal)), index=cal)
    base1 = align_flow_to_calendar(x, cal, 1)
    base0 = align_flow_to_calendar(x, cal, 0)
    assert np.isnan(base1.iloc[0])
    np.testing.assert_array_equal(base1.iloc[1:].to_numpy(), base0.iloc[:-1].to_numpy())
    d = 12
    for lag in (0, 1, 3):
        x2 = x.copy()
        x2.iloc[d + 1:] += 100.0                        # change everything after date d
        a = align_flow_to_calendar(x, cal, lag)
        b = align_flow_to_calendar(x2, cal, lag)
        k = d + lag                                      # rows <= index(d)+lag are unaffected
        pd.testing.assert_series_equal(a.iloc[:k + 1], b.iloc[:k + 1])
        assert a.iloc[k + 1] != b.iloc[k + 1]
    with pytest.raises(ValueError):
        align_flow_to_calendar(x, cal, -1)


def test_flow_drops_observations_before_calendar_start():
    cal = pd.bdate_range("2020-01-06", periods=10)
    x = pd.Series([100.0, 200.0, 1.0, 2.0], index=pd.DatetimeIndex(
        ["2019-12-20", "2020-01-02", cal[0], cal[2]]))
    out = align_flow_to_calendar(x, cal, 0)
    assert out.iloc[0] == 1.0 and out.iloc[1] == 0.0 and out.iloc[2] == 2.0      # not 301 on row 0


def test_flow_ends_after_stale_limit_but_gaps_inside_stay_zero():
    cal = pd.bdate_range("2020-01-01", periods=30)
    x = pd.Series([1.0, 1.0, 1.0], index=[cal[2], cal[3], cal[10]])             # gap 4..9 inside; last obs row 10
    out = align_flow_to_calendar(x, cal, 0, stale_limit=5)
    assert (out.iloc[4:10] == 0).all()
    assert (out.iloc[11:16] == 0).all() and out.iloc[10] == 1
    assert out.iloc[16:].isna().all()                                            # more than 5 rows after the last obs
    out2 = align_flow_to_calendar(x, cal, 2, stale_limit=5)
    assert out2.iloc[18:].isna().all() and (out2.iloc[13:18] == 0).all()
    assert (align_flow_to_calendar(x, cal, 0, stale_limit=None).iloc[11:] == 0).all()


def test_flow_cumsum_stops_showing_after_the_flow_ends():
    cal = pd.bdate_range("2020-01-01", periods=20)
    x = pd.Series([1.0, 2.0], index=[cal[2], cal[4]])
    prov = Fake({"flow:E/F": (align_flow_to_calendar(x, cal, 0, 3), 0, "r", "D")}, calendar=cal)
    r = Evaluator(prov, {"v": "cumsum(flow('E','F'))"}).evaluate("v").series
    np.testing.assert_array_equal(r.iloc[:9].to_numpy(), [nan, nan, 1, 1, 3, 3, 3, 3, nan])
    assert r.iloc[8:].isna().all()


def test_flow_all_after_calendar_or_empty_is_nan():
    cal = pd.bdate_range("2020-01-01", periods=5)
    assert align_flow_to_calendar(pd.Series(dtype=float), cal, 0).isna().all()
    late = pd.Series([1.0], index=[cal[-1] + pd.Timedelta(days=3)])
    assert align_flow_to_calendar(late, cal, 0).isna().all()


# ------------------------------------------------------------------------ 6. ref
def test_ref_helper_and_variable_and_chain_of_refs():
    d = {**A, **B}
    defs = {"h": "ts('A') + 1", "w": "ref('h') * 2"}
    r = ev("ref('w') + ref('h')", d, defs=defs)
    np.testing.assert_array_equal(vals(r)[:3], [(2 * 2) + 2, (3 * 2) + 3, (4 * 2) + 4])
    assert r.lag_rule == "class:X/Y" and r.freq == "D"


def test_ref_cycle_and_unknown():
    with pytest.raises(ValueError, match="circular"):
        Evaluator(Fake(A), {"a": "ref('b') + 1", "b": "ref('a') + 1"}).evaluate("a")
    with pytest.raises(ValueError, match="circular"):
        validate_definitions({"a": "ref('a')"})
    with pytest.raises(ValueError, match="unknown"):
        validate_definitions({"a": "ref('nope')"})
    with pytest.raises(ValueError, match="unknown"):
        ev("ref('nope')", A)


# -------------------------------------------------------------- 7. availability
def test_variable_lag_freq_and_rule_composition():
    data = {"A/PX_LAST": (S([1, 2, 3.0]), 2, "ticker:A", "D"),
            "B/PX_LAST": (S([1, 2, 3.0]), 4, "class:P/Q", "W"),
            "C/PX_LAST": (S([1, 2, 3.0]), 0, "ticker:A", "D")}
    r = ev("ts('A') + ts('B') - ts('C') + ts('A')", data)
    assert r.lag == 4 and r.freq == "W" and r.lag_rule == "ticker:A+class:P/Q"
    r = ev("ts('A') / ts('C')", data)
    assert r.lag == 2 and r.freq == "D" and r.lag_rule == "ticker:A"


def test_missing_rule_rejects_variable_and_names_leaves():
    data = {"A/PX_LAST": (S([1, 2, 3.0]), 0, "r", "D"), "B/PX_LAST": (S([1, 2.0]), 0, "r", "D")}
    r = ev("ts('A') - ts('B')", data, missing=["B/PX_LAST"])
    assert r.series is None and r.reason == "no_availability_rule: B/PX_LAST"
    # a rejection upstream in a helper propagates through ref
    r = Evaluator(Fake(data, missing=["B/PX_LAST"]), {"h": "ts('B')", "v": "ref('h') + ts('A')"}).evaluate("v")
    assert r.series is None and "B/PX_LAST" in r.reason
    r = ev("ts('A') + ts('B')", data, badfreq=["A/PX_LAST"])
    assert r.series is None and r.reason == "freq_not_daily_or_weekly: A/PX_LAST"
    r = ev("ts('A') + ts('B')", data, badfreq=["A/PX_LAST"], missing=["B/PX_LAST"])
    assert r.reason.startswith("no_availability_rule: B/PX_LAST")


# ----------------------- availability resolution through the real provider (fake db)
class FakeQ:
    """Just enough of DataQuery for DBLeaves."""

    def __init__(self, ts_data, chain=None, curve=None):
        self.ts_data, self._chain, self._curve = ts_data, chain, curve

    def get_timeseries(self, tickers, fields, as_dataframe=False):
        out = []
        for tk in tickers:
            s = self.ts_data.get((tk, fields[0]))
            if s is not None:
                out += [(tk, d.strftime("%Y-%m-%d"), v, "pct", "price") for d, v in s.items()]
        return out

    def list_fields(self, ticker):
        return [{"field": f} for (t, f) in self.ts_data if t == ticker]

    def get_futures_chain(self, root):
        return self._chain

    def get_curve_history(self, name, measures=None):
        return self._curve


def _daily(n=400, start="2019-01-01", seed=0, level=100.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(level + np.random.default_rng(seed).normal(size=n).cumsum(), index=idx)


def _provider(lags, ts_data=None, chain=None, curve=None, sec=None):
    cal = pd.bdate_range("2019-01-01", periods=400)
    cfg = type("C", (), {"lags": lags, "universe": type("U", (), {"fields": ["PX_LAST", "OTHER"]})()})()
    sec = sec or {"T1": {"class_one": "Market_Data", "class_two": "VOL"}}
    return DBLeaves(FakeQ(ts_data or {}, chain, curve), cfg, cal, sec)


def test_ts_rule_priority_ticker_over_field_over_class():
    s = _daily()
    data = {("T1", "PX_LAST"): s, ("T1", "OTHER"): s}
    lags = LagsConfig(chain=0, class_={"Market_Data/VOL": 1}, ticker={"T1": 5}, field_={"PX_LAST": 3, "OTHER": 2})
    p = _provider(lags, data)
    leaf = p.ts("T1", "PX_LAST")
    assert (leaf.info.lag, leaf.info.rule) == (5, "ticker:T1")
    lags = LagsConfig(chain=0, class_={"Market_Data/VOL": 1}, field_={"PX_LAST": 3})
    p = _provider(lags, data)
    assert (p.ts("T1", "PX_LAST").info.lag, p.ts("T1", "PX_LAST").info.rule) == (3, "field:PX_LAST")
    leaf = p.ts("T1", "OTHER")                         # no field rule for OTHER -> class
    assert (leaf.info.lag, leaf.info.rule) == (1, "class:Market_Data/VOL")
    lags = LagsConfig(chain=0, class_={})
    with pytest.raises(Rejected) as e:
        _provider(lags, data).ts("T1", "PX_LAST")
    assert e.value.reason == "no_availability_rule" and e.value.leaf == "T1/PX_LAST"


def test_ts_default_field_follows_priority_and_values_are_aligned():
    s = _daily()
    lags = LagsConfig(chain=0, class_={"Market_Data/VOL": 2})
    p = _provider(lags, {("T1", "OTHER"): s})
    leaf = p.ts("T1")                                  # PX_LAST missing -> OTHER
    assert leaf.info.name == "T1/OTHER"
    exp = align_to_calendar(s, p.calendar, 2, 5)
    pd.testing.assert_series_equal(leaf.series, exp)
    with pytest.raises(ValueError):
        p.ts("T1", "NOPE")                             # unknown field -> no data -> config error


def test_flow_leaf_uses_flow_alignment_and_field_lag():
    s = _daily() * 0 + 1.0
    lags = LagsConfig(chain=0, field_={"FLOW": 1})
    p = _provider(lags, {("T1", "FLOW"): s})
    leaf = p.flow("T1", "FLOW")
    exp = align_flow_to_calendar(s, p.calendar, 1)
    pd.testing.assert_series_equal(leaf.series, exp)
    assert leaf.info.rule == "field:FLOW" and leaf.info.lag == 1


def _chain_df(n=400):
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"trade_date": idx.strftime("%Y-%m-%d"), "return_valid": 1,
                         "return_index": np.arange(n, dtype=float) + 100, "oi_active": np.arange(n, dtype=float) + 1000,
                         "oi_next": np.nan, "ret": 0.0, "px_active": 110.0, "calendar_spread": 0.1})


def test_chain_rule_field_overrides_chain_default():
    lags = LagsConfig(chain=0, field_={"chain:oi_active": 1})
    p = _provider(lags, chain=_chain_df())
    a = p.chain("TY", "oi_active")
    assert (a.info.lag, a.info.rule) == (1, "field:chain:oi_active")
    b = p.chain("TY", "return_index")
    assert (b.info.lag, b.info.rule) == (0, "chain")
    assert a.series.iloc[1] == 1000.0 and np.isnan(a.series.iloc[0])
    with pytest.raises(ValueError):
        p.chain("TY", "not_a_column")


def test_curve_without_rule_is_rejected_and_with_rule_is_filtered():
    idx = pd.bdate_range("2019-01-01", periods=400)
    rows = []
    for sa, te, off in (("1Y", "1Y", 0.0), ("2Y", "1Y", 10.0)):
        rows.append(pd.DataFrame({"trade_date": idx.strftime("%Y-%m-%d"), "start_tenor": sa, "tenor": te,
                                  "value": np.arange(400.0) + off}))
    df = pd.concat(rows)
    p = _provider(LagsConfig(chain=0), curve=df)
    with pytest.raises(Rejected) as e:
        p.curve("C1", "fwd_start", "1Y", "1Y")
    assert e.value.reason == "no_availability_rule" and "C1" in e.value.leaf
    p = _provider(LagsConfig(chain=0, curve={"C1": 1}), curve=df)
    leaf = p.curve("C1", "fwd_start", "2Y", "1Y")
    assert leaf.info.rule == "curve:C1" and leaf.info.lag == 1
    assert leaf.series.iloc[1] == 10.0 and np.isnan(leaf.series.iloc[0])
    with pytest.raises(ValueError):
        p.curve("C1", "fwd_start", "9Y", "9Y")


def test_variable_through_real_provider_lag_is_max_of_leaves():
    s1, s2 = _daily(seed=1), _daily(seed=2, level=50)
    lags = LagsConfig(chain=0, class_={"Market_Data/VOL": 1}, field_={"OTHER": 3})
    p = _provider(lags, {("T1", "PX_LAST"): s1, ("T1", "OTHER"): s2})
    res = Evaluator(p, {"v": "ts('T1','PX_LAST') - ts('T1','OTHER')"}).evaluate("v")
    assert res.lag == 3 and res.freq == "D" and res.lag_rule == "class:Market_Data/VOL+field:OTHER"
    a = align_to_calendar(s1, p.calendar, 1, 5)
    b = align_to_calendar(s2, p.calendar, 3, 5)
    exp = (a - b).where(np.isfinite(a) & np.isfinite(b))
    pd.testing.assert_series_equal(res.series, exp)


# ----------------------------------------------------------------- clean
def _clean(values, n=10, k=3.0):
    cal = pd.bdate_range("2020-01-01", periods=len(values))
    x = {"X/PX_LAST": (pd.Series(np.asarray(values, dtype=float), index=cal), 0, "r", "D")}
    return ev(f"clean(ts('X'), {n}, {k})", x, calendar=cal).series.to_numpy()


def test_clean_removes_single_spike_only():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 40)
    v[25] = 50.0
    out = _clean(v, n=20, k=8.0)
    assert np.isnan(out[25])
    np.testing.assert_array_equal(np.delete(out, 25)[15:], np.delete(v, 25)[15:])
    assert not np.isnan(out[26:]).any() and not np.isnan(out[:25]).any()


def test_clean_needs_enough_history_and_nonzero_mad():
    v = np.r_[np.random.default_rng(1).normal(0, 1, 3), 100.0, np.random.default_rng(2).normal(0, 1, 10)]
    assert _clean(v, n=10)[3] == 100.0                      # only 3 previous obs < n // 2 = 5: kept
    flat = np.r_[np.arange(12) * 0 + 5.0, 9.0, 5.0, 5.0]
    out = _clean(flat)
    assert not np.isnan(out).any()                           # all identical before -> MAD == 0 -> no filtering


def test_clean_forward_filled_repeats_do_not_collapse_mad():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 40)
    v = np.repeat(base, 5)                                   # a new value every 5 rows, then carried forward
    v[102] = 30.0                                            # a genuine new value that is a spike (start of a block)
    v[103:105] = 30.0
    out = _clean(v, n=30, k=8.0)
    assert np.isnan(out[102:105]).all()                      # the spike and its carried-forward copies are removed
    ok = np.ones(len(v), bool)
    ok[102:105] = False
    # normal blocks survive: repeated values would have made MAD tiny and killed ordinary moves
    assert np.isfinite(out[ok]).all()


def test_clean_is_causal_and_validates_arguments():
    rng = np.random.default_rng(4)
    v = rng.normal(0, 1, 50)
    v[30] = 40.0
    full = _clean(v, k=4.0)
    for t0 in (20, 30, 40):
        np.testing.assert_array_equal(_clean(v[:t0 + 1], k=4.0), full[:t0 + 1])
    for bad in ("clean(ts('X'), 1, 3)", "clean(ts('X'), 10, 0)", "clean(ts('X'), 10)", "clean(ts('X'), 10, -2)",
                "clean(ts('X'), 10.5, 3)", "clean(ts('X'), 10, 'a')"):
        with pytest.raises(ValueError):
            parse_expr(bad, "v")
    parse_expr("clean(ts('X') - ts('Y'), 60, 10)", "v")


# ----------------------------------------------------------------- effective lag
def test_effective_lag_explicit_lag_is_added():
    d = {"A/PX_LAST": (S([1, 2, 3, 4, 5, 6.0]), 1, "class:X/Y", "D"),
         "B/PX_LAST": (S([1, 2, 3, 4, 5, 6.0]), 3, "ticker:B", "D")}
    r = ev("lag(ts('A'), 1)", d)
    assert r.lag == 2 and r.lag_rule == "class:X/Y+expr_lag"
    r = ev("lag(lag(ts('A'), 1), 2)", d)
    assert r.lag == 4 and r.lag_rule.endswith("+expr_lag")
    r = ev("lag(ts('A'), 1) + ts('B')", d)                # max(2, 3) = 3 = leaf max: nothing appended
    assert r.lag == 3 and r.lag_rule == "class:X/Y+ticker:B"
    r = ev("lag(ts('A'), 3) + ts('B')", d)                # 4 > 3
    assert r.lag == 4 and r.lag_rule == "class:X/Y+ticker:B+expr_lag"
    r = ev("cumsum(nz(lag(ts('A'), 0)))", d)
    assert r.lag == 1 and r.lag_rule == "class:X/Y"


def test_effective_lag_propagates_through_ref():
    d = {"A/PX_LAST": (S([1, 2, 3, 4, 5, 6.0]), 1, "class:X/Y", "D"),
         "B/PX_LAST": (S([1, 2, 3, 4, 5, 6.0]), 0, "ticker:B", "D")}
    defs = {"h": "lag(ts('A'), 2)", "w": "ref('h') / ts('B')"}
    r = ev("ref('w') - 1", d, defs=defs)
    assert r.lag == 3 and r.lag_rule == "class:X/Y+ticker:B+expr_lag"
    assert Evaluator(Fake(d), {"h": "lag(ts('B'), 1)", "v": "ref('h') + ts('A')"}).evaluate("v").lag == 1


# ----------------------------------------------------------------- futures_oi
class OIQ(FakeQ):
    def __init__(self, ts_data, contracts):
        super().__init__(ts_data)
        self.contracts = contracts

    def resolve(self, family, allow_multi=False, **attrs):
        assert family == "BOND_FUTURE" and allow_multi and attrs == {"root": "TY"}
        return self.contracts


def _oi_provider(lags, idx_len=300):
    idx = pd.bdate_range("2019-01-01", periods=idx_len)
    a = pd.Series(100.0, index=idx[:200])
    b = pd.Series(10.0, index=idx[150:])
    c = pd.Series(7.0, index=idx)                                   # the generic series must never be summed
    ts_data = {("TYH19 Comdty", "OPEN_INT"): a, ("TYM19 Comdty", "OPEN_INT"): b, ("TY1 Comdty", "OPEN_INT"): c}
    contracts = [{"ticker": "TY1 Comdty", "attrs": {"position": "1", "root": "TY"}},
                 {"ticker": "TY1 COMB Comdty", "attrs": {"root": "TY", "session": "COMB"}},
                 {"ticker": "TYH19 Comdty", "attrs": {"contract": "H19", "root": "TY"}},
                 {"ticker": "TYM19 Comdty", "attrs": {"contract": "M19", "root": "TY"}}]
    cal = idx
    cfg = type("C", (), {"lags": lags, "universe": type("U", (), {"fields": ["PX_LAST"]})()})()
    return DBLeaves(OIQ(ts_data, contracts), cfg, cal, {}), idx


def test_futures_oi_sums_concrete_contracts_with_at_least_one_value():
    p, idx = _oi_provider(LagsConfig(chain=0, field_={"OPEN_INT": 1}))
    leaf = p.futures_oi("TY")
    assert leaf.info.rule == "field:OPEN_INT" and leaf.info.lag == 1 and leaf.info.name.startswith("futures_oi:TY")
    s = leaf.series
    assert np.isnan(s.iloc[0]) and s.iloc[1] == 100.0                 # one contract only (lag 1)
    assert s.iloc[100] == 100.0
    assert s.iloc[151] == 110.0                                      # both contracts (row 150 lagged by one)
    assert s.iloc[160] == 110.0 and s.iloc[201] == 10.0               # first contract expired: still one value
    assert not (s.dropna() == 117.0).any()                           # generic XX1 never included


def test_futures_oi_without_field_rule_is_rejected_ticker_and_class_rules_do_not_apply():
    p, _ = _oi_provider(LagsConfig(chain=0, class_={"Market_Data/VOL": 1}, ticker={"TYH19 Comdty": 1, "TY": 1}))
    with pytest.raises(Rejected) as e:
        p.futures_oi("TY")
    assert e.value.reason == "no_availability_rule" and "futures_oi:TY" in e.value.leaf
    r = Evaluator(p, {"v": "futures_oi('TY')"}).evaluate("v")
    assert r.series is None and r.reason.startswith("no_availability_rule: futures_oi:TY")


def test_futures_oi_expression_in_evaluator():
    p, _ = _oi_provider(LagsConfig(chain=0, field_={"OPEN_INT": 2}))
    r = Evaluator(p, {"v": "futures_oi('TY') / 10"}).evaluate("v")
    assert r.lag == 2 and r.lag_rule == "field:OPEN_INT" and r.series.iloc[2] == 10.0
