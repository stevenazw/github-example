import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsminer.config import load_config
from tsminer.confirm import OosLockedError, confirm
from tsminer.databank import META_COLUMNS, Panel
from tsminer.pipeline import (AC_COLUMNS, CAND_COLUMNS, IC_COLUMNS, SHORT_COLUMNS, assign_verdict, backtest, dedupe,
                              dedupe_tradable, final_verdict, half_life, ic_decay, mine, positions, prepare, signal_autocorr, split_rows,
                              yearly_hit)
from tsminer.transforms import make_signal

ROOT = Path(__file__).resolve().parent.parent
N = 3000
N_SERIES = 30
GROUPS = ["FX", "Equity", "Rates"]

TOML = """
[target]
name = "TEST"
kind = "spot"
ticker = "TEST Curncy"
field = "PX_LAST"
cost_per_side = 0.00005

[lags]
chain = 1

[sample]
is_frac = 0.70
obs_frac = 0.15

[signals]
std_window = 252
clip = 4.0
vol_halflife = 20

[stats]
hac_lags = 5
min_obs = 250
n_boot = 200
min_shift = 60
seed = 7
"""

SHORT_LEN = 700
OLD_FLAG = "cost" + "_negative"          # removed: replaced by the min_net_sharpe hard gate
HIGH_COST = 0.001                        # sig|chg_1: gross Sharpe ~2.3, net ~ -0.9 at this cost


def make_panel(seed, plant=False, short=False, dup=False, n=N, ic_sig=None, flip_at=None, delay=1):
    """Synthetic panel. plant: long series 'sig' (IC 0.15, or ``ic_sig``: scalar / per-row array); short: series
    'short' observed only in the last SHORT_LEN rows (IC 0.2); dup: 'sig2', a near copy of 'sig'.
    flip_at: calendar position from which the 'sig' relation is reversed (IC -> -IC).
    delay: the execution delay the planted signals are built for: x[t] predicts r[t+1+delay], so that after the
    pipeline shifts the signal down by ``execution_delay`` rows (default 1) the held signal at row t predicts
    r[t+1]. ``delay=1`` matches the default config; ``delay=0`` is the old "fill at the signal-day close" world."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    d = rng.normal(0, 1, (n, N_SERIES + 1))
    noise = rng.normal(0, 0.005, n)
    r = noise.copy()
    ic0 = np.zeros(n)
    if plant:
        ic0 = np.full(n, 0.15 if ic_sig is None else float(ic_sig)) if (ic_sig is None or np.isscalar(ic_sig)) \
            else np.asarray(ic_sig, dtype=float)
        if flip_at is not None:
            ic0 = ic0.copy()
            ic0[flip_at:] *= -1
    ics = 0.2 if short else 0.0
    if plant or short:
        sig = ic0[:-1] * d[:-1, 0] / d[:, 0].std() + ics * d[:-1, N_SERIES] / d[:, N_SERIES].std()
        r[1:] = 0.005 * sig + np.sqrt(1 - ic0[:-1] ** 2 - ics**2) * noise[1:]      # x[t] -> r[t+1]
        if delay:                                   # the target lags by `delay` rows: x[t] -> r[t+1+delay]; the held
            r = np.r_[noise[:delay], r[:n - delay]]  # (shifted) signal vs y_fwd is the old relationship exactly
    X = pd.DataFrame(np.cumsum(d[:, :N_SERIES], axis=0), index=idx, columns=[f"s{i}" for i in range(N_SERIES)])
    if plant:
        X = X.rename(columns={"s0": "sig"})
        if dup:
            X["sig2"] = X["sig"] + 0.05 * np.cumsum(rng.normal(0, 1, n))
    if short:
        X["short"] = np.where(np.arange(n) >= n - SHORT_LEN, np.cumsum(d[:, N_SERIES]), np.nan)
    meta = pd.DataFrame({
        "ticker": X.columns, "field": "PX_LAST", "kind": "level", "freq": "D",
        "group": [GROUPS[i % 3] for i in range(X.shape[1])], "lag": 1, "lag_rule": "chain",
        "description": [f"desc {c}" for c in X.columns], "first_obs": idx[0], "n_obs": n,
    }, index=pd.Index(X.columns, name="series_id"))[META_COLUMNS]
    return Panel(target=pd.Series(r, index=idx, name="TEST"), X=X, meta=meta,
                 excluded=pd.DataFrame({"ticker": ["a", "b"], "reason": ["no_data", "too_few_obs"]}),
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


@pytest.fixture(scope="module")
def planted_panel():
    return make_panel(1, plant=True, short=True, dup=True)


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory, cfg, cfg_path, planted_panel):
    d = tmp_path_factory.mktemp("planted") / "run"
    mine(cfg, d, panel=planted_panel, config_path=cfg_path)
    return d


@pytest.fixture(scope="module")
def confirmed_run(tmp_path_factory, planted_run):
    d = tmp_path_factory.mktemp("confirmed") / "run"
    shutil.copytree(planted_run, d)
    return d, confirm(d)


@pytest.fixture(scope="module")
def noise_run(tmp_path_factory, cfg, cfg_path):
    d = tmp_path_factory.mktemp("noise") / "run"
    mine(cfg, d, panel=make_panel(11), config_path=cfg_path)
    return d


# ------------------------------------------------------------------ tests
def test_split_rows():
    rng = np.random.default_rng(0)
    T, N_ = 1000, 6
    valid = rng.random((T, N_)) < 0.9
    valid[:, 1] = False
    valid[int(0.7 * T):, 1] = True                  # only the last 30% of the time axis has data
    valid[:, 2] = False                             # no data at all
    valid[:100, 3] = False
    fi, fo = 0.70, 0.15
    is_, obs, oos, e1, e2 = split_rows(valid, fi, fo)
    assert is_.shape == obs.shape == oos.shape == valid.shape and e1.shape == e2.shape == (N_,)
    assert not (is_ & obs).any() and not (is_ & oos).any() and not (obs & oos).any()
    np.testing.assert_array_equal(is_ | obs | oos, valid)
    assert (e1 % 21 == 0).all() and (e2 % 21 == 0).all() and (e1 <= e2).all()
    rows = np.arange(T)
    for j in range(N_):
        assert (rows[is_[:, j]] < e1[j]).all()
        assert ((rows[obs[:, j]] >= e1[j]) & (rows[obs[:, j]] < e2[j])).all()
        assert (rows[oos[:, j]] >= e2[j]).all()
        n = int(valid[:, j].sum())
        if n:
            tol = 21 / n + 1e-9
            assert abs(is_[:, j].sum() / n - fi) <= tol
            assert abs(obs[:, j].sum() / n - fo) <= tol
            assert abs(oos[:, j].sum() / n - (1 - fi - fo)) <= tol
    assert not is_[:, 2].any() and not obs[:, 2].any() and not oos[:, 2].any()
    # the short column: all three segments live inside its own valid span
    first = int(0.7 * T)
    for m in (is_, obs, oos):
        assert m[:, 1].any() and rows[m[:, 1]].min() >= first
    assert rows[is_[:, 1]].max() < rows[obs[:, 1]].min() and rows[obs[:, 1]].max() < rows[oos[:, 1]].min()
    assert is_[:, 1].sum() + obs[:, 1].sum() + oos[:, 1].sum() == T - first


def test_split_rows_matches_loop():
    rng = np.random.default_rng(5)
    T = 400
    valid = rng.random((T, 8)) < 0.6
    is_, obs, oos, e1, e2 = split_rows(valid, 0.5, 0.25, quantum=21)
    for j in range(8):
        v = np.flatnonzero(valid[:, j])
        n = len(v)
        k1, k2 = int(np.floor(n * 0.5)), int(np.floor(n * (0.5 + 0.25)))
        p1 = v[k1] if k1 < n else T
        p2 = v[k2] if k2 < n else T
        assert e1[j] == (p1 // 21) * 21
        assert e2[j] == max(e1[j], (p2 // 21) * 21)


def test_prepare_shapes(cfg, planted_panel):
    prep = prepare(planted_panel, cfg)
    rows = prep["rows"]
    np.testing.assert_array_equal(rows, np.flatnonzero(np.isfinite(prep["y_fwd"].to_numpy())))
    shape = (len(rows), prep["S"].shape[1])
    assert prep["is_mask"].shape == prep["obs_mask"].shape == prep["oos_mask"].shape == shape
    assert prep["e1"].shape == prep["e2"].shape == (shape[1],)
    assert set(prep) == {"S", "cand_meta", "y_fwd", "sigma", "rows", "is_mask", "obs_mask", "oos_mask", "e1", "e2"}


def _cfg_delay(tmp_path, d):
    p = tmp_path / f"d{d}.toml"
    p.write_text(TOML + f"\n[position]\nexecution_delay = {d}\n")
    return p, load_config(p)


def _is_ic(panel, cfg, cid="sig|chg_1"):
    """Signed in-sample IC of one candidate (corr of the held signal with y_fwd on its own IS rows)."""
    prep = prepare(panel, cfg)
    j = list(prep["S"].columns).index(cid)
    rows = prep["rows"]
    m = prep["is_mask"][:, j]
    s = prep["S"].to_numpy(dtype=float)[rows][m, j]
    y = prep["y_fwd"].to_numpy(dtype=float)[rows][m]
    return float(np.corrcoef(s, y)[0, 1])


def test_execution_delay_means_fill_at_next_close(tmp_path):
    """x[t] predicts r[t+2]: only a fill one close later (execution_delay = 1) can harvest it; with the fill at the
    signal-day close (0) the held signal is 1 day too early. A signal that predicts r[t+1] is unusable at delay 1."""
    p0, cfg0 = _cfg_delay(tmp_path, 0)
    p1, cfg1 = _cfg_delay(tmp_path, 1)
    late = make_panel(61, plant=True, delay=1)          # x[t] -> r[t+2]
    early = make_panel(61, plant=True, delay=0)         # x[t] -> r[t+1]
    ic_late_1, ic_late_0 = _is_ic(late, cfg1), _is_ic(late, cfg0)
    ic_early_1, ic_early_0 = _is_ic(early, cfg1), _is_ic(early, cfg0)
    print("IC x->r[t+2]: delay=1", round(ic_late_1, 4), "delay=0", round(ic_late_0, 4),
          "| IC x->r[t+1]: delay=1", round(ic_early_1, 4), "delay=0", round(ic_early_0, 4))
    assert ic_late_1 > 0.10 and abs(ic_late_0) < 0.05
    assert ic_early_0 > 0.10 and abs(ic_early_1) < 0.05
    mine(cfg1, tmp_path / "late1", panel=late, config_path=p1)
    mine(cfg0, tmp_path / "late0", panel=late, config_path=p0)
    c1 = pd.read_csv(tmp_path / "late1" / "candidates.csv").set_index("id").loc["sig|chg_1"]
    c0 = pd.read_csv(tmp_path / "late0" / "candidates.csv").set_index("id").loc["sig|chg_1"]
    assert c1["verdict"] == "PASS" and c1["z"] > 5 and abs(c0["z"]) < 2.0 and c0["verdict"] == "REJECT"
    mine(cfg1, tmp_path / "early1", panel=early, config_path=p1)
    e1 = pd.read_csv(tmp_path / "early1" / "candidates.csv").set_index("id").loc["sig|chg_1"]
    assert e1["verdict"] == "REJECT" and abs(e1["z"]) < 2.0


@pytest.mark.parametrize("d", [0, 1, 3])
def test_prepare_shifts_candidates_by_execution_delay(tmp_path, planted_panel, d):
    from tsminer.transforms import build_candidates
    _, cfg_d = _cfg_delay(tmp_path, d)
    S0, _ = build_candidates(planted_panel.X, planted_panel.meta, cfg_d.signals.std_window, cfg_d.signals.clip)
    prep = prepare(planted_panel, cfg_d)
    pd.testing.assert_frame_equal(prep["S"], S0.shift(d))
    if d:
        assert prep["S"].iloc[:d].isna().all().all()
    np.testing.assert_array_equal(prep["S"].to_numpy()[d:], S0.to_numpy()[:len(S0) - d])
    first = S0.apply(lambda c: c.first_valid_index())
    first_held = prep["S"].apply(lambda c: c.first_valid_index())
    cal = planted_panel.target.index
    for c in S0.columns[:5]:
        assert cal.get_loc(first_held[c]) == cal.get_loc(first[c]) + d


def test_execution_delay_config(tmp_path):
    base = TOML
    assert load_config(_write_cfg(tmp_path, base)).position.execution_delay == 1
    assert load_config(_write_cfg(tmp_path, base + "\n[position]\nbuffer = 0.2\n")).position.execution_delay == 1
    for v in (0, 1, 5):
        c = load_config(_write_cfg(tmp_path, base + f"\n[position]\nexecution_delay = {v}\n"))
        assert c.position.execution_delay == v and c.to_dict()["position"]["execution_delay"] == v
    for bad in ("-1", "1.0", "0.5", "true", '"1"', "[1]"):
        with pytest.raises(ValueError):
            load_config(_write_cfg(tmp_path, base + f"\n[position]\nexecution_delay = {bad}\n"))
    for name in ("ty", "eurusd", "ty_vars"):
        assert load_config(ROOT / "configs" / f"{name}.toml").position.execution_delay == 1
        assert "execution_delay = 1" in (ROOT / "configs" / f"{name}.toml").read_text(encoding="utf-8")


def test_old_run_without_execution_delay_counts_as_zero(tmp_path, planted_run):
    from tsminer.config import load_run_config
    assert load_run_config(planted_run).position.execution_delay == 1
    old = tmp_path / "old"
    shutil.copytree(planted_run, old)
    rj = json.loads((old / "run.json").read_text())
    del rj["config"]["position"]["execution_delay"]
    (old / "run.json").write_text(json.dumps(rj))
    assert load_run_config(old).position.execution_delay == 0
    assert "成交延迟" not in "".join(_exec_lines(old))


def _exec_lines(d):
    from tsminer.report import write_report
    write_report(d)
    return [l for l in (d / "report.md").read_text(encoding="utf-8").splitlines() if "成交" in l]


def test_rule_provenance_tiers():
    from tsminer.report import PROV_TEXT, rule_provenance
    ver, conf = ["field:F", "ticker:T"], ["chain", "class:A/B"]
    assert rule_provenance("field:F", ver, conf) == 3 and rule_provenance("chain", ver, conf) == 2
    assert rule_provenance("class:X/Y", ver, conf) == 1
    assert rule_provenance("field:F+class:X/Y", ver, conf) == 1                  # verified + draft -> draft
    assert rule_provenance("field:F+class:A/B", ver, conf) == 2                  # verified + confirmed -> confirmed
    assert rule_provenance("field:F+ticker:T", ver, conf) == 3
    assert rule_provenance("field:F+expr_lag", ver, conf) == 3                   # expr_lag takes no part
    assert rule_provenance("class:X/Y+expr_lag", ver, conf) == 1
    assert [PROV_TEXT[k] for k in (3, 2, 1)] == ["已验证", "研究者确认", "起草待确认"]


def test_report_availability_table_has_provenance_column(tmp_path):
    txt = TOML.replace("chain = 1", 'chain = 1\nverified = ["field:F"]\nconfirmed = ["chain"]') + (
        '\n[lags.field]\n"F" = 1\n[lags.class]\n"Rates/Z" = 2\n')
    p = _write_cfg(tmp_path, txt)
    cfg_ = load_config(p)
    panel = make_panel(5)
    panel.meta["lag_rule"] = (["chain", "field:F", "class:Rates/Z", "field:F+class:Rates/Z",
                               "field:F+class:Rates/Z+expr_lag"] * 10)[:panel.meta.shape[0]]
    mine(cfg_, tmp_path / "r", panel=panel, config_path=p)
    sec = (tmp_path / "r" / "report.md").read_text(encoding="utf-8").split("## 可得性输入")[1].split("## 判定标准")[0]
    assert "| 规则键 | lag | 出处 | 命中的序列数 |" in sec
    rows = {l.split("|")[1].strip(): l.split("|")[3].strip() for l in sec.splitlines()
            if l.startswith("| ") and "规则键" not in l}
    assert rows == {"chain": "研究者确认", "field:F": "已验证", "class:Rates/Z": "起草待确认",
                    "field:F+class:Rates/Z": "起草待确认", "field:F+class:Rates/Z+expr_lag": "起草待确认"}
    assert "出处汇总" in sec and "已验证 " in sec and "研究者确认 " in sec and "起草待确认 " in sec


def test_report_states_execution_delay(planted_run, tmp_path):
    rep = (planted_run / "report.md").read_text(encoding="utf-8")
    assert "execution_delay = 1" in rep and "第 1 个标的收盘成交" in rep
    d0 = tmp_path / "r0"
    p, cfg0 = _cfg_delay(tmp_path, 0)
    mine(cfg0, d0, panel=make_panel(3, plant=True, delay=0), config_path=p)
    assert "当日收盘成交" in (d0 / "report.md").read_text(encoding="utf-8")


def _cut_panel(panel, cfg, cid, seed, which):
    """Replace the target and the raw series of candidate cid at every calendar position that the allowed
    rows cannot see: which='e2' -> from rows[e2_j] on (nothing of the OOS may matter); which='e1' -> from
    rows[e1_j] on (nothing of OBS/OOS may matter for the IS statistics). Target values after rows[e]
    (y[t] needs r[t+1]) and the raw series from rows[e] are replaced by random numbers."""
    prep = prepare(panel, cfg)
    j = list(prep["S"].columns).index(cid)
    e = int(prep[which][j])
    assert e < len(prep["rows"])
    c = int(prep["rows"][e])
    rng = np.random.default_rng(seed)
    tgt, X = panel.target.copy(), panel.X.copy()
    tgt.iloc[c + 1:] = rng.normal(0, 0.005, len(tgt) - c - 1)
    sid = cid.split("|")[0]
    fin = np.isfinite(X[sid].to_numpy()) & (np.arange(len(X)) >= c)
    X.loc[fin, sid] = rng.normal(0, 50, int(fin.sum()))
    return Panel(target=tgt, X=X, meta=panel.meta, excluded=panel.excluded, info=panel.info), c


def _same(a, b):
    return a == b or (pd.isna(a) and pd.isna(b))


@pytest.mark.parametrize("cid", ["sig|chg_1", "short|chg_1"])
def test_isolation_nothing_from_oos(tmp_path, cfg, cfg_path, planted_panel, planted_run, cid):
    """(a) data at rows >= e2_j (target and the candidate's raw series) must not change anything about j."""
    p2, c = _cut_panel(planted_panel, cfg, cid, 999, "e2")
    assert c < len(planted_panel.target) - 5            # something is really replaced
    mine(cfg, tmp_path / "b", panel=p2, config_path=cfg_path)
    ca = pd.read_csv(planted_run / "candidates.csv").set_index("id")
    cb = pd.read_csv(tmp_path / "b" / "candidates.csv").set_index("id")
    a, b = ca.loc[cid], cb.loc[cid]
    for k in CAND_COLUMNS:
        if k != "id":
            assert _same(a[k], b[k]), (k, a[k], b[k])
    num = [k for k in CAND_COLUMNS if k != "id" and isinstance(a[k], (int, float, np.number)) and not pd.isna(a[k])]
    assert {"t", "null_sd", "z", "ic", "n_is", "t_h1", "t_h2", "t_xlag", "n_obs", "n_oos", "ic_obs_signed",
            "retention", "t_dev", "sharpe_gross_dev", "sharpe_net_dev"} <= set(num)
    assert a["trade_status"] == "KEEP"
    rep = a["rep"]
    assert rep == cid or rep in set(pd.read_csv(planted_run / "shortlist.csv")["id"])
    sa = pd.read_csv(planted_run / "shortlist.csv").set_index("id")
    sb = pd.read_csv(tmp_path / "b" / "shortlist.csv").set_index("id")
    assert rep in sb.index
    for k in SHORT_COLUMNS:
        if k != "id":
            assert _same(sa.loc[rep, k], sb.loc[rep, k]), (k, sa.loc[rep, k], sb.loc[rep, k])
    assert not sa.loc[rep, IC_COLUMNS + AC_COLUMNS].isna().all()


@pytest.mark.parametrize("cid", ["sig|chg_1", "short|chg_1"])
def test_isolation_is_stats_ignore_obs_and_oos(tmp_path, cfg, cfg_path, planted_panel, planted_run, cid):
    """(b) data at rows >= e1_j must not change the IS statistics of j."""
    p2, c = _cut_panel(planted_panel, cfg, cid, 998, "e1")
    assert c < len(planted_panel.target) - 5
    mine(cfg, tmp_path / "b", panel=p2, config_path=cfg_path)
    a = pd.read_csv(planted_run / "candidates.csv").set_index("id").loc[cid]
    b = pd.read_csv(tmp_path / "b" / "candidates.csv").set_index("id").loc[cid]
    for k in ("t", "null_sd", "z", "ic", "n_is", "t_h1", "t_h2", "t_xlag", "sign", "p", "is_start", "is_end",
              "verdict", "z_verdict", "sharpe_gross_is", "sharpe_net_is"):
        assert _same(a[k], b[k]), (k, a[k], b[k])
    fa = {f for f in str(a["flags"]).split(";") if f in ("half_flip", "xlag_weak")}
    fb = {f for f in str(b["flags"]).split(";") if f in ("half_flip", "xlag_weak")}
    assert fa == fb


def test_short_history_candidate_is_tested(planted_run, planted_panel, cfg):
    cand = pd.read_csv(planted_run / "candidates.csv").set_index("id")
    r = cand.loc["short|chg_1"]
    n_valid = r["n_is"] + r["n_obs"] + r["n_oos"]
    assert n_valid <= SHORT_LEN
    fi, fo = cfg.sample.is_frac, cfg.sample.obs_frac
    assert r["n_is"] >= 250
    assert abs(r["n_is"] / n_valid - fi) <= 21 / n_valid + 1e-9
    assert abs(r["n_obs"] / n_valid - fo) <= 21 / n_valid + 1e-9
    assert r["verdict"] == "PASS"
    cal = planted_panel.target.index
    assert pd.Timestamp(r["is_start"]) >= cal[N - SHORT_LEN]
    print("short|chg_1:", {k: r[k] for k in ("n_is", "n_obs", "n_oos", "z", "ic", "ic_obs_signed", "retention",
                                              "obs_status", "t_dev", "is_start", "is_end")})


@pytest.mark.parametrize("cid", ["sig|chg_1", "short|chg_1"])
def test_planted_signal_survives_screens(planted_run, cid):
    cand = pd.read_csv(planted_run / "candidates.csv").set_index("id")
    short = pd.read_csv(planted_run / "shortlist.csv")
    r = cand.loc[cid]
    assert r["verdict"] == "PASS" and r["obs_status"] == "KEEP"
    assert r["retention"] >= 0.5 and r["ic_obs_signed"] > 0
    assert isinstance(r["rep"], str) and r["rep"] in set(short["id"])
    print(cid, {k: r[k] for k in ("z", "ic", "ic_obs_signed", "retention", "t_dev")})


def test_short_history_candidate_confirmed(confirmed_run, planted_panel):
    d, res = confirmed_run
    cand = pd.read_csv(d / "candidates.csv").set_index("id")
    rep = cand.loc["short|chg_1", "rep"]
    assert rep.startswith("short|")
    row = {c["id"]: c for c in res["candidates"]}[rep]
    assert row["verdict"] == "CONFIRMED"
    cal = planted_panel.target.index
    assert row["oos_start"] > cand.loc[rep, "is_end"]
    assert pd.Timestamp(row["oos_start"]) >= cal[N - SHORT_LEN]
    assert row["oos_end"] == str(cal[-2].date())     # the last row with a finite y_fwd
    assert row["n"] == cand.loc[rep, "n_oos"]
    print("short confirm:", {k: row[k] for k in ("id", "n", "oos_start", "oos_end", "t_signed", "ic_oos_signed",
                                                  "retention_oos", "p_one", "verdict")},
          "z =", cand.loc[rep, "z"])


def test_output_files_and_columns(planted_run):
    d = planted_run
    assert list(pd.read_csv(d / "candidates.csv").columns) == CAND_COLUMNS
    assert list(pd.read_csv(d / "shortlist.csv").columns) == SHORT_COLUMNS
    assert "ac" + "1" not in SHORT_COLUMNS and "ac_1" in SHORT_COLUMNS and "ic_h60" in SHORT_COLUMNS
    for f in ("watchlist.csv", "groups.csv", "exceed.csv"):
        assert not (d / f).exists()
    run = json.loads((d / "run.json").read_text())
    assert set(run) == {"target", "config", "panel_info", "dates", "n_rows", "n_series", "n_candidates",
                        "n_tested", "n_skipped", "n_is_summary", "funnel", "shortlist", "created_at"}
    assert list(run["funnel"]) == ["tested", "is_z_pass", "is_pass", "is_weak", "obs_keep", "obs_insufficient", "obs_drop",
                                   "trade_keep", "trade_drop", "shortlist"]
    assert set(run["dates"]) == {"sample_start", "sample_end"}
    cand = pd.read_csv(d / "candidates.csv")
    f = run["funnel"]
    assert run["n_tested"] == len(cand) == f["tested"] and run["n_candidates"] == run["n_tested"] + run["n_skipped"]
    assert f["is_pass"] == (cand["verdict"] == "PASS").sum() and f["is_weak"] == (cand["verdict"] == "WEAK_PASS").sum()
    assert f["obs_keep"] + f["obs_insufficient"] + f["obs_drop"] == f["is_pass"] + f["is_weak"]
    assert f["is_z_pass"] >= f["is_pass"] + f["is_weak"] and f["is_z_pass"] == (cand["z_verdict"] != "REJECT").sum()
    assert list(cand.columns)[list(cand.columns).index("verdict") + 1:list(cand.columns).index("verdict") + 4] == [
        "z_verdict", "sharpe_gross_is", "sharpe_net_is"]
    assert cand.loc[cand["z_verdict"] == "REJECT", ["sharpe_gross_is", "sharpe_net_is"]].isna().all().all()
    zp = cand[cand["z_verdict"] != "REJECT"]
    assert np.isfinite(zp["sharpe_net_is"]).all()
    assert ((zp["sharpe_net_is"] > 0.3) == (zp["verdict"] != "REJECT")).all()
    assert f["trade_keep"] + f["trade_drop"] == f["obs_keep"] + f["obs_insufficient"]
    assert f["trade_keep"] == (cand["trade_status"] == "KEEP").sum() and f["trade_drop"] == (cand["trade_status"] == "DROP").sum()
    assert f["shortlist"] <= f["trade_keep"]
    assert f["shortlist"] == len(run["shortlist"]) == len(pd.read_csv(d / "shortlist.csv"))
    assert (cand["z"].abs().diff().dropna() <= 1e-12).all()         # sorted by |z| descending
    assert run["n_is_summary"]["min"] == cand["n_is"].min()
    rej = cand[cand["verdict"] == "REJECT"]
    assert rej[["ic_obs_signed", "retention", "t_dev"]].isna().all().all() and rej["obs_status"].isna().all()
    assert (rej["n_obs"] > 0).all()
    drop = cand[cand["obs_status"] == "DROP"]
    assert drop["t_dev"].isna().all() and (drop["rep"].isna() | (drop["rep"] == "")).all()
    # the tradability columns exist only for the step-2 survivors, right after t_dev
    cols = list(cand.columns)
    assert cols[cols.index("t_dev") + 1:cols.index("t_dev") + 4] == ["sharpe_gross_dev", "sharpe_net_dev", "trade_status"]
    nonalive = cand[(cand["verdict"] == "REJECT") | (cand["obs_status"] == "DROP")]
    assert nonalive[["sharpe_gross_dev", "sharpe_net_dev", "trade_status"]].isna().all().all()
    alive = cand.drop(nonalive.index)
    assert alive["trade_status"].isin(["KEEP", "DROP"]).all() and np.isfinite(alive["sharpe_net_dev"]).all()
    keep = alive["trade_status"] == "KEEP"
    assert (alive.loc[keep, "sharpe_net_dev"] >= cfg_min_net_sharpe(d)).all()
    assert (alive.loc[~keep, "sharpe_net_dev"] < cfg_min_net_sharpe(d)).all()
    assert (alive.loc[~keep, "rep"].isna() | (alive.loc[~keep, "rep"] == "")).all()
    assert alive.loc[keep, "rep"].isin(set(alive.loc[keep, "id"])).all()
    assert (d / "panel").is_dir() and (d / "config.toml").exists()


def cfg_min_net_sharpe(d):
    return json.loads((Path(d) / "run.json").read_text())["config"]["verdict"]["min_net_sharpe"]


def test_too_short_candidates_are_skipped(tmp_path, cfg, cfg_path):
    panel = make_panel(2)
    panel.X.loc[panel.X.index[:N - 350], "s1"] = np.nan       # ~220 usable rows -> IS below min_obs
    run = mine(cfg, tmp_path / "r", panel=panel, config_path=cfg_path)
    cand = pd.read_csv(tmp_path / "r" / "candidates.csv")
    assert not cand["id"].str.startswith("s1|").any()
    assert run["n_skipped"] >= 6 and run["n_candidates"] == run["n_tested"] + run["n_skipped"]


def test_planted_signal_confirmed(confirmed_run):
    d, res = confirmed_run
    cand = pd.read_csv(d / "candidates.csv").set_index("id")
    assert cand.loc["sig|chg_1", "verdict"] == "PASS"
    short = pd.read_csv(d / "shortlist.csv")
    rep = cand.loc["sig|chg_1", "rep"]
    assert rep in set(short["id"])
    assert res["status"] == "DONE"
    assert {c["id"]: c for c in res["candidates"]}[rep]["verdict"] == "CONFIRMED"
    assert set(res) == {"status", "forced", "force_reason", "n_tested", "n_confirmed", "n_consistent", "n_failed",
                        "n_insufficient", "candidates", "opened_at"}
    assert res["n_confirmed"] >= 1
    assert res["n_tested"] == len(res["candidates"]) == len(short)
    assert res["n_confirmed"] + res["n_consistent"] + res["n_failed"] + res["n_insufficient"] == len(short)
    assert list(pd.read_csv(d / "confirm.csv").columns) == [
        "id", "sign", "n", "oos_start", "oos_end", "t_signed", "ic_oos_signed", "retention_oos", "p_one",
        "verdict", "sharpe_gross", "sharpe_net"]
    assert (d / "confirm.lock").exists()
    assert "## 样本外确认" in (d / "report.md").read_text(encoding="utf-8")


def test_cluster_size_and_single_representative(planted_run):
    cand = pd.read_csv(planted_run / "candidates.csv").set_index("id")
    short = pd.read_csv(planted_run / "shortlist.csv").set_index("id")
    assert len({"sig|chg_1", "sig2|chg_1"} & set(short.index)) == 1
    rep = cand.loc["sig|chg_1", "rep"]
    assert cand.loc["sig2|chg_1", "rep"] == rep
    assert short.loc[rep, "cluster_size"] == 2
    tradable = cand[cand["trade_status"] == "KEEP"]
    assert short["cluster_size"].sum() == len(tradable)
    assert (short["rep"] == short.index).all()


def test_lock_and_force(confirmed_run):
    d, _ = confirmed_run
    d2 = d.parent / "run_copy"
    shutil.copytree(d, d2)
    before = (d2 / "confirm.json").read_text()
    with pytest.raises(OosLockedError):
        confirm(d2)
    assert (d2 / "confirm.json").read_text() == before
    assert not (d2 / "confirm_log.jsonl").exists()
    res = confirm(d2, force_reason="unit test rerun")
    assert res["status"] == "DONE" and res["forced"] is True
    log = [json.loads(x) for x in (d2 / "confirm_log.jsonl").read_text().splitlines()]
    assert log[-1]["reason"] == "unit test rerun" and "time" in log[-1]


def test_pure_noise_runs(noise_run):
    rep = (noise_run / "report.md").read_text(encoding="utf-8")
    cand = pd.read_csv(noise_run / "candidates.csv")
    assert len(cand) > 0 and "因样本不足未检验" in rep
    run = json.loads((noise_run / "run.json").read_text())
    if run["funnel"]["shortlist"] == 0:
        assert "没有候选留下" in rep
        assert len(pd.read_csv(noise_run / "shortlist.csv")) == 0


def test_nothing_to_confirm(tmp_path, noise_run):
    d = tmp_path / "run"
    shutil.copytree(noise_run, d)
    pd.read_csv(d / "shortlist.csv").iloc[:0].to_csv(d / "shortlist.csv", index=False)
    assert confirm(d) == {"status": "NOTHING_TO_CONFIRM"}
    assert not (d / "confirm.lock").exists()
    assert not (d / "confirm.json").exists()


def test_report_content(planted_run, noise_run, confirmed_run):
    for d in (planted_run, noise_run):
        rep = (d / "report.md").read_text(encoding="utf-8")
        for w in ("噪声预期", "FWER", "omnibus", "holdout", "Holdout", "同批", "多重检验"):
            assert w not in rep
        for h in ("## 结论", "## 数据与样本", "## 可得性输入", "## 判定标准", "## Shortlist", "## IC 衰减与信号自相关",
                  "## 提示与局限", "## 下一步"):
            assert h in rep
    rep = (planted_run / "report.md").read_text(encoding="utf-8")
    assert "5000" in rep and "短样本候选的观察期与样本外也短" in rep and "近月连续合约" in rep
    assert "chain" in rep and "no_availability_rule" not in rep.split("## 可得性输入")[0]
    assert "检验 " in rep and "→ 去重后 shortlist" in rep
    assert "→ 可交易（净 Sharpe ≥ 0.3）保留 " in rep and "可交易性：" in rep and "min_net_sharpe" in rep
    assert rep.index("→ 观察期保留") < rep.index("→ 可交易") < rep.index("→ 去重后 shortlist")
    assert OLD_FLAG not in rep
    full = (confirmed_run[0] / "report.md").read_text(encoding="utf-8")
    assert "holdout" not in full.lower()


def test_report_lists_rejected_for_missing_rule(tmp_path, cfg, cfg_path):
    panel = make_panel(21)
    panel.excluded = pd.DataFrame({"ticker": [f"T{i}" for i in range(25)] + ["u"],
                                   "reason": ["no_availability_rule"] * 25 + ["no_data"]})
    mine(cfg, tmp_path / "r", panel=panel, config_path=cfg_path)
    rep = (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
    sec = rep.split("## 可得性输入")[1].split("## 判定标准")[0]
    assert "25 个" in sec and "T19" in sec and "T20" not in sec and "需研究者确认" in sec


def test_assign_verdict_boundaries():
    nan = float("nan")
    for z, want in [(2.5, "PASS"), (2.49, "WEAK_PASS"), (2.0, "WEAK_PASS"), (1.99, "REJECT"),
                    (-2.5, "PASS"), (-2.49, "WEAK_PASS"), (-2.0, "WEAK_PASS"), (-1.99, "REJECT"),
                    (nan, "REJECT"), (float("inf"), "REJECT"), (3.0, "PASS"), (0.0, "REJECT")]:
        assert assign_verdict(z, 2.5, 2.0) == want, z


def test_dedupe():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    df = pd.DataFrame({"a": a, "b": a + 0.05 * rng.normal(size=500), "c": rng.normal(size=500)})
    out = dedupe(["b", "a", "c"], df, 0.7)  # b is first in priority order, so it is the representative
    assert out == {"b": "b", "a": "b", "c": "c"}
    # too little overlap counts as uncorrelated
    df2 = df.copy()
    df2.loc[df2.index[:450], "a"] = np.nan
    assert dedupe(["b", "a"], df2, 0.7, min_overlap=100) == {"b": "b", "a": "a"}


# ------------------------------------------------------------ positions / backtest
def test_positions_buffer_zero_and_huge():
    rng = np.random.default_rng(1)
    s = rng.normal(size=300)
    s[[5, 77]] = np.nan
    target = 1.0 * np.where(np.isfinite(s), s, 0.0)
    np.testing.assert_array_equal(positions(s, 1.0, 0.0), target)
    np.testing.assert_array_equal(positions(s, -1.0, 0.0), -target)
    big = positions(s, 1.0, 1e9)
    assert (big == target[0]).all()                      # never moves after day one


@pytest.mark.parametrize("buffer", [0.1, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_positions_band_properties(buffer, sign):
    rng = np.random.default_rng(2)
    s = np.clip(rng.normal(size=2000), -4, 4)
    pos = positions(s, sign, buffer)
    target = sign * s
    assert pos[0] == target[0]
    assert (np.abs(pos - target) <= buffer + 1e-12).all()
    moves = np.abs(np.diff(pos))
    assert (moves[moves > 0] > buffer).all()
    assert (moves > 0).any() and (moves == 0).any()


def _bt_inputs():
    rng = np.random.default_rng(0)
    n = 1500
    s = rng.normal(size=n)
    s[:5] = np.nan
    sigma = np.full(n, 0.01)
    y = 0.05 * np.r_[s[1:], 0.0].clip(-4, 4) + rng.normal(size=n)
    return s, np.nan_to_num(y), sigma


def test_backtest():
    s, y, sigma = _bt_inputs()
    b0 = backtest(s, y, sigma, 1.0, 0.0, 0.5)
    assert b0["sharpe_net"] == pytest.approx(b0["sharpe_gross"])
    b1 = backtest(s, y, sigma, 1.0, 1e-4, 0.5)
    assert b0["turnover"] > 0 and b1["turnover"] > 0
    assert b1["sharpe_net"] < b1["sharpe_gross"]
    assert b1["sharpe_gross"] == pytest.approx(b0["sharpe_gross"])
    bm = backtest(s, y, sigma, -1.0, 0.0, 0.5)
    assert bm["sharpe_gross"] == pytest.approx(-b0["sharpe_gross"])
    assert set(b0) == {"sharpe_gross", "sharpe_net", "turnover"}          # the old lag-1 autocorrelation is gone
    # evaluation starts at the first finite s: leading NaN rows do not matter
    s2, y2, sg2 = s[5:], y[5:], sigma[5:]
    assert backtest(s2, y2, sg2, 1.0, 1e-4, 0.5)["sharpe_net"] == pytest.approx(b1["sharpe_net"])
    # buffer = 0: turnover equals the mean |diff| of the target itself
    t0 = backtest(s2, y2, sg2, 1.0, 0.0, 0.0)["turnover"]
    assert t0 == pytest.approx(np.mean(np.abs(np.diff(s2))))


def test_backtest_turnover_not_increasing_in_buffer():
    s, y, sigma = _bt_inputs()
    tv = [backtest(s, y, sigma, 1.0, 1e-4, b)["turnover"] for b in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)]
    print("turnover by buffer:", tv)
    assert all(b <= a + 1e-12 for a, b in zip(tv, tv[1:]))


# ------------------------------------------------------- IC decay / autocorrelation / years
def test_ic_decay_planted_signal_is_one_day(planted_run, cfg, planted_panel):
    short = pd.read_csv(planted_run / "shortlist.csv").set_index("id")
    cand = pd.read_csv(planted_run / "candidates.csv").set_index("id")
    rep = cand.loc["sig|chg_1", "rep"]
    r = short.loc[rep]
    print("ic decay", {c: round(r[c], 4) for c in IC_COLUMNS}, "ac", {c: round(r[c], 3) for c in AC_COLUMNS})
    assert r["ic_h1"] > 0.08
    assert abs(r["ic_h20"]) < 0.5 * r["ic_h1"] and abs(r["ic_h60"]) < 0.5 * r["ic_h1"]
    # h = 1 is the signed IC on the development sample (IS + OBS)
    prep = prepare(planted_panel, cfg)
    j = list(prep["S"].columns).index(rep)
    rows = prep["rows"]
    dev = prep["is_mask"][:, j] | prep["obs_mask"][:, j]
    s = prep["S"].to_numpy()[rows][:, j]
    y = prep["y_fwd"].to_numpy()[rows]
    ic_dev = np.corrcoef(s[dev], y[dev])[0, 1] * r["sign"]
    assert r["ic_h1"] == pytest.approx(ic_dev, abs=1e-9)
    # the planted chg_1 signal is (nearly) white
    assert abs(r["ac_1"]) < 0.15 and r["signal_half_life"] < 1.0 or r["ac_1"] <= 0


def test_autocorr_and_half_life_of_random_walk_zscore():
    rng = np.random.default_rng(3)
    x = pd.Series(np.cumsum(rng.normal(size=4000)))
    z = make_signal(x, "z_252", "level").to_numpy()
    ac = signal_autocorr(z)
    assert ac[0] > 0.9
    assert half_life(ac[0]) > 10
    assert ac[0] > ac[1] > ac[2] > ac[-1]
    assert half_life(0.5) == pytest.approx(1.0)
    assert half_life(-0.2) == 0.0 and half_life(0.0) == 0.0
    assert np.isnan(half_life(1.0)) and np.isnan(half_life(float("nan"))) and np.isnan(half_life(1.2))
    white = signal_autocorr(rng.normal(size=4000))
    assert all(abs(a) < 0.06 for a in white)


def test_ic_decay_uses_only_rows_before_e2():
    rng = np.random.default_rng(4)
    s = rng.normal(size=500)
    y = np.r_[s[1:], 0.0] * 0.5 + rng.normal(size=500) * 0.1
    a = ic_decay(s, y, 400, 1.0)
    assert len(a) == 6
    y2 = y.copy()
    y2[400:] = 1e6
    s2 = s.copy()
    s2[400:] = np.nan
    assert a == ic_decay(s2, y2, 400, 1.0)
    lagged = np.r_[0.0, s[:-1]]                         # y_t = s_{t-1}: s_t lines up with y[t+h-1] at h = 2
    d = ic_decay(s, lagged, 400, 1.0)
    assert abs(d[0]) < 0.15 and d[1] > 0.99
    assert ic_decay(s, lagged, 400, -1.0)[1] < -0.99


def _years_panel(seed, pos_years):
    n = N
    idx = pd.bdate_range("2010-01-01", periods=n)
    ic = np.where(np.isin(idx.year, list(pos_years)), 0.5, -0.15)
    return make_panel(seed, plant=True, ic_sig=ic)


def test_year_hit_stable_signal(planted_run):
    short = pd.read_csv(planted_run / "shortlist.csv").set_index("id")
    cand = pd.read_csv(planted_run / "candidates.csv").set_index("id")
    r = short.loc[cand.loc["sig|chg_1", "rep"]]
    print("stable year_hit", r["year_hit"], r["n_years"])
    assert r["n_years"] >= 8 and r["year_hit"] >= 0.8
    assert "unstable" not in str(r["flags"]).split(";")


def test_year_hit_flips_half_the_years_unstable(tmp_path, cfg, cfg_path):
    panel = _years_panel(31, (2012, 2014, 2017, 2018, 2019))   # strong in 5 years, reversed in the other 5
    mine(cfg, tmp_path / "r", panel=panel, config_path=cfg_path)
    cand = pd.read_csv(tmp_path / "r" / "candidates.csv").set_index("id")
    short = pd.read_csv(tmp_path / "r" / "shortlist.csv").set_index("id")
    r = cand.loc["sig|chg_1"]
    print("year-flip sig|chg_1", {k: r[k] for k in ("z", "obs_status", "retention", "rep")})
    assert r["verdict"] != "REJECT" and r["obs_status"] != "DROP"
    s = short.loc[r["rep"]]
    print("year_hit", s["year_hit"], s["n_years"], s["flags"])
    assert s["n_years"] >= 8 and s["year_hit"] < 0.6
    assert "unstable" in str(s["flags"]).split(";")


def test_yearly_hit_function():
    years = np.repeat([2000, 2001, 2002, 2003], 100)
    rng = np.random.default_rng(0)
    s = rng.normal(size=400)
    y = np.where(np.isin(years, [2000, 2001]), s, -s) + 0.1 * rng.normal(size=400)
    assert yearly_hit(s, y, years, 1.0) == (0.5, 4)
    assert yearly_hit(s, y, years, -1.0) == (0.5, 4)
    s2 = s.copy()
    s2[:50] = np.nan                                  # 2000 drops below 60 usable rows
    assert yearly_hit(s2, y, years, 1.0) == (pytest.approx(1 / 3), 3)


# --------------------------------------------------------------- observation drop
def test_observation_period_drops_decayed_signal(tmp_path, cfg, cfg_path):
    base = make_panel(41, plant=True, ic_sig=0.15)
    prep = prepare(base, cfg)
    j = list(prep["S"].columns).index("sig|chg_1")
    c1 = int(prep["rows"][int(prep["e1"][j])])         # first calendar row after the IS rows of sig|chg_1
    panel = make_panel(41, plant=True, ic_sig=0.15, flip_at=c1)   # relation reversed from there on
    mine(cfg, tmp_path / "r", panel=panel, config_path=cfg_path)
    cand = pd.read_csv(tmp_path / "r" / "candidates.csv").set_index("id")
    short = pd.read_csv(tmp_path / "r" / "shortlist.csv")
    r = cand.loc["sig|chg_1"]
    print("obs drop", {k: r[k] for k in ("z", "ic", "n_is", "n_obs", "ic_obs_signed", "retention", "obs_status")})
    assert r["verdict"] == "PASS"
    assert r["obs_status"] == "DROP" and r["retention"] < 0
    assert "sig|chg_1" not in set(short["id"]) and (pd.isna(r["rep"]) or r["rep"] == "")
    assert pd.isna(r["t_dev"])
    assert r[["sharpe_gross_dev", "sharpe_net_dev"]].isna().all() and pd.isna(r["trade_status"])


def test_observation_period_insufficient_is_kept(tmp_path, cfg_path):
    # a short candidate whose observation period has < 60 rows: INSUFFICIENT -> kept, flagged obs_short
    cfg150 = load_config(_write_cfg(tmp_path, TOML.replace("min_obs = 250", "min_obs = 150")))
    panel = make_panel(51, plant=True, ic_sig=0.3)
    panel.X["tiny"] = np.where(np.arange(N) >= N - 430, panel.X["sig"], np.nan)    # ~300 valid rows
    panel.meta.loc["tiny"] = panel.meta.loc["sig"]
    panel.meta.loc["tiny", ["ticker", "description"]] = ["tiny", "desc tiny"]
    mine(cfg150, tmp_path / "r", panel=panel, config_path=tmp_path / "c.toml")
    cand = pd.read_csv(tmp_path / "r" / "candidates.csv").set_index("id")
    short = pd.read_csv(tmp_path / "r" / "shortlist.csv")
    r = cand.loc["tiny|chg_1"]
    print("tiny", {k: r[k] for k in ("n_is", "n_obs", "n_oos", "z", "obs_status", "flags", "rep")})
    assert r["n_obs"] < 60 and r["verdict"] != "REJECT"
    assert r["obs_status"] == "INSUFFICIENT" and np.isnan(r["retention"]) and np.isnan(r["ic_obs_signed"])
    assert "obs_short" in r["flags"].split(";")
    assert not np.isnan(r["t_dev"]) and r["rep"] in set(short["id"])
    run = json.loads((tmp_path / "r" / "run.json").read_text())
    assert run["funnel"]["obs_insufficient"] >= 1


# ------------------------------------------------------------- tradability screen
def _cfg_text(cost=None, min_net_sharpe=None, is_min=None):
    txt = TOML if cost is None else TOML.replace("cost_per_side = 0.00005", f"cost_per_side = {cost}")
    extra = []
    if min_net_sharpe is not None:
        extra.append(f"min_net_sharpe = {min_net_sharpe}")
    if is_min is not None:
        extra.append(f"is_min_net_sharpe = {is_min}")
    if extra:
        txt += "\n[verdict]\n" + "\n".join(extra) + "\n"
    return txt


@pytest.fixture(scope="module")
def costly_run(tmp_path_factory, planted_panel):
    d = tmp_path_factory.mktemp("costly")
    p = _write_cfg(d, _cfg_text(HIGH_COST, None, "-1e9"))     # step-1 Sharpe bar off: isolates step 3
    mine(load_config(p), d / "run", panel=planted_panel, config_path=p)
    return d / "run"


@pytest.fixture(scope="module")
def is_gated_run(tmp_path_factory, planted_panel):
    d = tmp_path_factory.mktemp("is_gated")
    p = _write_cfg(d, _cfg_text(HIGH_COST))
    mine(load_config(p), d / "run", panel=planted_panel, config_path=p)
    return d / "run"


@pytest.fixture(scope="module")
def costly_off_run(tmp_path_factory, planted_panel):
    d = tmp_path_factory.mktemp("costly_off")
    p = _write_cfg(d, _cfg_text(HIGH_COST, "-1e9", "-1e9"))
    mine(load_config(p), d / "run", panel=planted_panel, config_path=p)
    return d / "run"


def test_high_turnover_signal_fails_tradability(costly_run):
    cand = pd.read_csv(costly_run / "candidates.csv").set_index("id")
    short = pd.read_csv(costly_run / "shortlist.csv")
    for cid in ("sig|chg_1", "sig2|chg_1", "short|chg_1"):
        r = cand.loc[cid]
        print("costly", cid, {k: r[k] for k in ("z", "t_dev", "sharpe_gross_dev", "sharpe_net_dev", "trade_status")})
        assert r["verdict"] == "PASS" and r["obs_status"] == "KEEP"          # the IC evidence is intact
        assert r["sharpe_gross_dev"] > 1.0 and r["sharpe_net_dev"] < 0.1
        assert r["trade_status"] == "DROP"
        assert cid not in set(short["id"]) and (pd.isna(r["rep"]) or r["rep"] == "")
    run = json.loads((costly_run / "run.json").read_text())
    f = run["funnel"]
    assert f["trade_drop"] >= 3 and f["trade_keep"] + f["trade_drop"] == f["obs_keep"] + f["obs_insufficient"]
    assert f["shortlist"] <= f["trade_keep"]
    assert "cluster_size" in short.columns and "sharpe_net" in short.columns
    assert (short["sharpe_net"] >= 0.3).all()                                # every shortlist member passes the bar


def test_gate_is_the_reason_for_the_drop(costly_run, costly_off_run):
    a = pd.read_csv(costly_run / "candidates.csv").set_index("id")
    b = pd.read_csv(costly_off_run / "candidates.csv").set_index("id")
    short = pd.read_csv(costly_off_run / "shortlist.csv").set_index("id")
    # same data, same cost: only the threshold differs, so the readings are identical ...
    for cid in ("sig|chg_1", "sig2|chg_1", "short|chg_1"):
        assert a.loc[cid, "sharpe_net_dev"] == b.loc[cid, "sharpe_net_dev"]
        assert a.loc[cid, "t_dev"] == b.loc[cid, "t_dev"]
        # ... but the gate is open now
        assert b.loc[cid, "trade_status"] == "KEEP" and b.loc[cid, "rep"] in set(short.index)
    print("gate off: sig|chg_1 rep =", b.loc["sig|chg_1", "rep"], "short:", sorted(short.index)[:6],
          "net", b.loc["sig|chg_1", "sharpe_net_dev"])
    assert len({"sig|chg_1", "sig2|chg_1"} & set(short.index)) == 1
    assert (b.loc[np.isfinite(b["sharpe_net_dev"]), "trade_status"] == "KEEP").all()
    fa = json.loads((costly_run / "run.json").read_text())["funnel"]
    fb = json.loads((costly_off_run / "run.json").read_text())["funnel"]
    assert fa["obs_keep"] == fb["obs_keep"] and fa["obs_insufficient"] == fb["obs_insufficient"]
    assert fb["trade_drop"] == 0 and fb["trade_keep"] == fb["obs_keep"] + fb["obs_insufficient"]
    assert fa["trade_drop"] > 0 and fa["shortlist"] < fb["shortlist"]


def test_dedupe_after_screen_promotes_tradable_member():
    """Cluster {A, B} (corr > 0.7); A has the higher t_dev but fails the net-Sharpe bar: B represents the
    cluster instead of the whole cluster vanishing. Screening happens before clustering."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    df = pd.DataFrame({"A": a, "B": a + 0.4 * rng.normal(size=500), "C": rng.normal(size=500)})
    assert abs(np.corrcoef(df["A"], df["B"])[0, 1]) > 0.7
    order = ["A", "B", "C"]                                            # A has the best t_dev
    assert dedupe_tradable(order, {"A": True, "B": True, "C": True}, df, 0.7) == {"A": "A", "B": "A", "C": "C"}
    out = dedupe_tradable(order, {"A": False, "B": True, "C": True}, df, 0.7)
    assert out == {"B": "B", "C": "C"} and "A" not in out
    assert set(out.values()) == {"B", "C"}                             # the cluster survives, represented by B
    # a failing candidate does not count towards cluster_size either
    assert list(out.values()).count("B") == 1
    assert dedupe_tradable(order, {}, df, 0.7) == {}


def test_net_sharpe_gate_boundary_and_nonfinite(tmp_path, planted_panel, planted_run):
    """sharpe_net >= min_net_sharpe keeps; setting the bar just above a candidate's own net Sharpe drops it."""
    cand = pd.read_csv(planted_run / "candidates.csv").set_index("id")
    net = cand.loc["sig|chg_1", "sharpe_net_dev"]
    p = _write_cfg(tmp_path, _cfg_text(None, repr(float(net) + 1e-6)))
    mine(load_config(p), tmp_path / "hi", panel=planted_panel, config_path=p)
    c2 = pd.read_csv(tmp_path / "hi" / "candidates.csv").set_index("id")
    assert c2.loc["sig|chg_1", "sharpe_net_dev"] == net and c2.loc["sig|chg_1", "trade_status"] == "DROP"
    p = _write_cfg(tmp_path, _cfg_text(None, repr(float(net))))
    mine(load_config(p), tmp_path / "eq", panel=planted_panel, config_path=p)
    c3 = pd.read_csv(tmp_path / "eq" / "candidates.csv").set_index("id")
    assert c3.loc["sig|chg_1", "trade_status"] == "KEEP"                # equality passes


def test_final_verdict_boundary():
    nan = float("nan")
    for zv in ("PASS", "WEAK_PASS"):
        assert final_verdict(zv, 0.3 + 1e-9, 0.3) == zv
        assert final_verdict(zv, 0.3, 0.3) == "REJECT"                  # strictly greater
        assert final_verdict(zv, 0.29, 0.3) == "REJECT"
        assert final_verdict(zv, nan, 0.3) == "REJECT" and final_verdict(zv, float("inf"), 0.3) == "REJECT"
        assert final_verdict(zv, nan, -1e9) == "REJECT"                 # non-finite never passes
        assert final_verdict(zv, -5.0, -1e9) == zv
    assert final_verdict("REJECT", 5.0, 0.3) == "REJECT"


def test_in_sample_net_sharpe_gate(is_gated_run, costly_run):
    a = pd.read_csv(is_gated_run / "candidates.csv").set_index("id")
    b = pd.read_csv(costly_run / "candidates.csv").set_index("id")
    short = pd.read_csv(is_gated_run / "shortlist.csv")
    for cid in ("sig|chg_1", "sig2|chg_1", "short|chg_1"):
        r = a.loc[cid]
        print("is gate", cid, {k: r[k] for k in ("z", "z_verdict", "sharpe_gross_is", "sharpe_net_is", "verdict")})
        assert r["z_verdict"] == "PASS" and r["verdict"] == "REJECT"
        assert r["sharpe_gross_is"] > 1.0 and r["sharpe_net_is"] <= 0.3
        assert r[["ic_obs_signed", "retention", "t_dev", "sharpe_net_dev"]].isna().all()
        assert pd.isna(r["obs_status"]) and pd.isna(r["trade_status"]) and (pd.isna(r["rep"]) or r["rep"] == "")
        assert cid not in set(short["id"])
        # gate off (same data and cost): the same readings, but the verdict stands
        assert b.loc[cid, "sharpe_net_is"] == r["sharpe_net_is"] and b.loc[cid, "verdict"] == "PASS"
        assert b.loc[cid, "obs_status"] == "KEEP"
    fa = json.loads((is_gated_run / "run.json").read_text())["funnel"]
    fb = json.loads((costly_run / "run.json").read_text())["funnel"]
    assert fa["is_z_pass"] == fb["is_z_pass"] and fa["is_pass"] + fa["is_weak"] < fb["is_pass"] + fb["is_weak"]
    assert fa["is_z_pass"] >= fa["is_pass"] + fa["is_weak"]
    assert fa["is_pass"] + fa["is_weak"] == fa["obs_keep"] + fa["obs_insufficient"] + fa["obs_drop"]


def test_load_config_is_min_net_sharpe(tmp_path):
    assert load_config(_write_cfg(tmp_path, TOML)).verdict.is_min_net_sharpe == 0.3
    for v, want in (("0.8", 0.8), ("-1e9", -1e9), ("1", 1)):
        c = load_config(_write_cfg(tmp_path, TOML + f"\n[verdict]\nis_min_net_sharpe = {v}\n"))
        assert c.verdict.is_min_net_sharpe == want
    for bad in ('"abc"', "true", "[0.3]", "inf", "nan", "-inf"):
        with pytest.raises(ValueError):
            load_config(_write_cfg(tmp_path, TOML + f"\n[verdict]\nis_min_net_sharpe = {bad}\n"))
    for name in ("ty", "eurusd", "ty_vars"):
        c = load_config(ROOT / "configs" / f"{name}.toml")
        assert c.verdict.is_min_net_sharpe == 0.3 and c.verdict.min_net_sharpe == 0.3
        assert "is_min_net_sharpe = 0.3" in (ROOT / "configs" / f"{name}.toml").read_text(encoding="utf-8")
    assert load_config(_write_cfg(tmp_path, TOML)).to_dict()["verdict"]["is_min_net_sharpe"] == 0.3


def test_load_config_min_net_sharpe(tmp_path):
    assert load_config(_write_cfg(tmp_path, TOML)).verdict.min_net_sharpe == 0.3
    for v, want in (("0.8", 0.8), ("-1e9", -1e9), ("2", 2)):
        assert load_config(_write_cfg(tmp_path, TOML + f"\n[verdict]\nmin_net_sharpe = {v}\n")).verdict.min_net_sharpe == want
    for bad in ('"abc"', "true", "[0.3]", "inf", "nan", "-inf"):
        with pytest.raises(ValueError):
            load_config(_write_cfg(tmp_path, TOML + f"\n[verdict]\nmin_net_sharpe = {bad}\n"))
    for name in ("ty", "eurusd"):
        c = load_config(ROOT / "configs" / f"{name}.toml")
        assert c.verdict.min_net_sharpe == 0.3
        assert "min_net_sharpe = 0.3" in (ROOT / "configs" / f"{name}.toml").read_text(encoding="utf-8")
    assert load_config(_write_cfg(tmp_path, TOML)).to_dict()["verdict"]["min_net_sharpe"] == 0.3


def test_old_flag_is_gone_from_the_repository():
    hits = []
    for f in ROOT.rglob("*"):
        rel = f.relative_to(ROOT).parts
        if (not f.is_file() or rel[0] in ("runs", ".git") or "__pycache__" in rel or ".pytest_cache" in rel
                or f.suffix in (".pyc", ".pkl")):
            continue
        if OLD_FLAG in f.read_text(encoding="utf-8", errors="ignore"):
            hits.append(str(f.relative_to(ROOT)))
    assert hits == []


def _write_cfg(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


def test_cli_locked_exit_code(confirmed_run):
    d, _ = confirmed_run
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, "-m", "tsminer", "confirm", str(d)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 2


def test_report_lists_variables_without_rule_with_their_leaves(tmp_path, cfg, cfg_path):
    panel = make_panel(22)
    panel.excluded = pd.DataFrame({"ticker": ["var:ff_3m", "var:x", "u"],
                                   "reason": ["no_availability_rule: FF1 Comdty/PX_LAST, FF4 Comdty/PX_LAST",
                                              "no_availability_rule: curve:C/fwd_start/1Y/1Y", "no_data"]})
    mine(cfg, tmp_path / "r", panel=panel, config_path=cfg_path)
    rep = (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
    sec = rep.split("## 可得性输入")[1].split("## 判定标准")[0]
    assert "2 个" in sec and "var:ff_3m [FF1 Comdty/PX_LAST, FF4 Comdty/PX_LAST]" in sec and "var:x" in sec
    assert "no_availability_rule: 2" in rep and "no_data: 1" in rep
