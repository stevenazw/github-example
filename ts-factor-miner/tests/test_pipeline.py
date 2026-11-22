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
from tsminer.confirm import HoldoutLockedError, confirm
from tsminer.databank import META_COLUMNS, Panel
from tsminer.pipeline import CAND_COLUMNS, SHORT_COLUMNS, assign_verdict, backtest, dedupe, mine, prepare, split_rows

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

[sample]
holdout_frac = 0.3

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


def make_panel(seed, plant=False, short=False, dup=False, n=N):
    """Synthetic panel. plant: long series 'sig' (IC 0.15); short: series 'short' observed only in the last
    SHORT_LEN rows (IC 0.2); dup: 'sig2', a near copy of 'sig'."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    d = rng.normal(0, 1, (n, N_SERIES + 1))
    noise = rng.normal(0, 0.005, n)
    r = noise.copy()
    ic0 = 0.15 if plant else 0.0
    ics = 0.2 if short else 0.0
    if plant or short:
        sig = ic0 * d[:-1, 0] / d[:, 0].std() + ics * d[:-1, N_SERIES] / d[:, N_SERIES].std()
        r[1:] = 0.005 * sig + np.sqrt(1 - ic0**2 - ics**2) * noise[1:]
    X = pd.DataFrame(np.cumsum(d[:, :N_SERIES], axis=0), index=idx, columns=[f"s{i}" for i in range(N_SERIES)])
    if plant:
        X = X.rename(columns={"s0": "sig"})
        if dup:
            X["sig2"] = X["sig"] + 0.05 * np.cumsum(rng.normal(0, 1, n))
    if short:
        X["short"] = np.where(np.arange(n) >= n - SHORT_LEN, np.cumsum(d[:, N_SERIES]), np.nan)
    meta = pd.DataFrame({
        "ticker": X.columns, "field": "PX_LAST", "kind": "level", "freq": "D",
        "group": [GROUPS[i % 3] for i in range(X.shape[1])], "lag": 1,
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
    T, N_, f = 1000, 6, 0.3
    valid = rng.random((T, N_)) < 0.9
    valid[:, 1] = False
    valid[int(0.7 * T):, 1] = True                  # only the last 30% of the time axis has data
    valid[:, 2] = False                             # no data at all
    valid[:100, 3] = False
    tr, ho, end = split_rows(valid, f)
    assert tr.shape == ho.shape == valid.shape and end.shape == (N_,)
    assert not (tr & ho).any()
    np.testing.assert_array_equal(tr | ho, valid)
    assert (end % 21 == 0).all()
    rows = np.arange(T)
    for j in range(N_):
        assert (rows[tr[:, j]] < end[j]).all() and (rows[ho[:, j]] >= end[j]).all()
        n = int(valid[:, j].sum())
        if n:
            frac = ho[:, j].sum() / n
            assert f - 1e-12 <= frac <= f + 21 / n + 1e-9
    assert not tr[:, 2].any() and not ho[:, 2].any()
    # the short column: train and holdout both live inside its own valid span
    first = int(0.7 * T)
    assert tr[:, 1].any() and ho[:, 1].any()
    assert rows[tr[:, 1]].min() >= first and rows[ho[:, 1]].min() > rows[tr[:, 1]].max()
    assert tr[:, 1].sum() + ho[:, 1].sum() == T - first
    for j in (0, 3):
        assert rows[tr[:, j]].max() < rows[ho[:, j]].min()


def test_split_rows_matches_loop():
    rng = np.random.default_rng(5)
    valid = rng.random((400, 8)) < 0.6
    tr, ho, end = split_rows(valid, 0.25, quantum=21)
    for j in range(8):
        v = np.flatnonzero(valid[:, j])
        n = len(v)
        k = int(np.floor(n * 0.75))
        p = v[k] if k < n else 400
        assert end[j] == (p // 21) * 21


def test_prepare_shapes(cfg, planted_panel):
    prep = prepare(planted_panel, cfg)
    rows = prep["rows"]
    np.testing.assert_array_equal(rows, np.flatnonzero(np.isfinite(prep["y_fwd"].to_numpy())))
    assert prep["train_mask"].shape == prep["holdout_mask"].shape == (len(rows), prep["S"].shape[1])
    assert prep["end_row"].shape == (prep["S"].shape[1],)


def _cut_panel(panel, cfg, cid, seed):
    """Replace the target and the raw series of candidate cid at every calendar position that its own
    train rows cannot see (target after rows[e_j]; raw series from rows[e_j])."""
    prep = prepare(panel, cfg)
    j = list(prep["S"].columns).index(cid)
    e = int(prep["end_row"][j])
    c = int(prep["rows"][e])
    rng = np.random.default_rng(seed)
    tgt, X = panel.target.copy(), panel.X.copy()
    tgt.iloc[c + 1:] = rng.normal(0, 0.005, len(tgt) - c - 1)
    sid = cid.split("|")[0]
    fin = np.isfinite(X[sid].to_numpy()) & (np.arange(len(X)) >= c)
    X.loc[fin, sid] = rng.normal(0, 50, int(fin.sum()))
    return Panel(target=tgt, X=X, meta=panel.meta, excluded=panel.excluded, info=panel.info), e, c


@pytest.mark.parametrize("cid", ["sig|chg_1", "short|chg_1"])
def test_per_candidate_isolation(tmp_path, cfg, cfg_path, planted_panel, planted_run, cid):
    p2, e, c = _cut_panel(planted_panel, cfg, cid, 999)
    assert c < len(planted_panel.target) - 5            # something is really replaced
    mine(cfg, tmp_path / "b", panel=p2, config_path=cfg_path)
    a = pd.read_csv(planted_run / "candidates.csv").set_index("id").loc[cid]
    b = pd.read_csv(tmp_path / "b" / "candidates.csv").set_index("id").loc[cid]
    num = [k for k in CAND_COLUMNS if k not in ("id", "rep") and isinstance(a[k], (int, float, np.number))]
    assert {"t", "null_sd", "z", "ic", "n_train", "t_h1", "t_h2", "t_xlag", "n_holdout"} <= set(num)
    for k in num:
        assert a[k] == b[k], (k, a[k], b[k])
    for k in ("train_start", "train_end", "flags", "verdict"):
        assert a[k] == b[k] or (pd.isna(a[k]) and pd.isna(b[k]))


def test_short_history_candidate_is_tested(planted_run, planted_panel, cfg):
    cand = pd.read_csv(planted_run / "candidates.csv").set_index("id")
    r = cand.loc["short|chg_1"]
    n_valid = r["n_train"] + r["n_holdout"]
    assert n_valid <= SHORT_LEN
    hf = cfg.sample.holdout_frac
    assert r["n_train"] >= 250
    assert (1 - hf) * n_valid - 21 <= r["n_train"] <= (1 - hf) * n_valid + 1
    assert r["verdict"] == "PASS"
    cal = planted_panel.target.index
    assert pd.Timestamp(r["train_start"]) >= cal[N - SHORT_LEN]
    print("short|chg_1:", {k: r[k] for k in ("n_train", "n_holdout", "z", "ic", "train_start", "train_end")})


def test_short_history_candidate_confirmed(confirmed_run, planted_panel):
    d, res = confirmed_run
    cand = pd.read_csv(d / "candidates.csv").set_index("id")
    rep = cand.loc["short|chg_1", "rep"]
    assert rep.startswith("short|")
    row = {c["id"]: c for c in res["candidates"]}[rep]
    assert row["verdict"] == "CONFIRMED"
    cal = planted_panel.target.index
    assert row["holdout_start"] > cand.loc[rep, "train_end"]
    assert pd.Timestamp(row["holdout_start"]) >= cal[N - SHORT_LEN]
    assert row["holdout_end"] == str(cal[-2].date())     # the last row with a finite y_fwd
    assert row["n"] == cand.loc[rep, "n_holdout"]
    print("short confirm:", {k: row[k] for k in ("id", "n", "holdout_start", "holdout_end", "t_signed", "p_one",
                                                  "verdict")}, "z =", cand.loc[rep, "z"])


def test_output_files_and_columns(planted_run):
    d = planted_run
    assert list(pd.read_csv(d / "candidates.csv").columns) == CAND_COLUMNS
    assert list(pd.read_csv(d / "shortlist.csv").columns) == SHORT_COLUMNS
    for f in ("watchlist.csv", "groups.csv", "exceed.csv"):
        assert not (d / f).exists()
    run = json.loads((d / "run.json").read_text())
    assert set(run) == {"target", "config", "panel_info", "dates", "n_rows", "n_series", "n_candidates",
                        "n_tested", "n_skipped", "n_train_summary", "verdict_counts", "shortlist", "created_at"}
    assert set(run["dates"]) == {"sample_start", "sample_end"}
    cand = pd.read_csv(d / "candidates.csv")
    assert run["n_tested"] == len(cand) and run["n_candidates"] == run["n_tested"] + run["n_skipped"]
    assert (cand["z"].abs().diff().dropna() <= 1e-12).all()         # sorted by |z| descending
    assert run["n_train_summary"]["min"] == cand["n_train"].min()
    assert (d / "panel").is_dir() and (d / "config.toml").exists()


def test_too_short_candidates_are_skipped(tmp_path, cfg, cfg_path):
    panel = make_panel(2)
    panel.X.loc[panel.X.index[:N - 350], "s1"] = np.nan       # ~220 usable rows -> train below min_obs
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
    assert set(res) == {"status", "forced", "force_reason", "n_tested", "n_confirmed", "candidates", "opened_at"}
    assert res["n_confirmed"] >= 1
    assert (d / "confirm.lock").exists() and (d / "confirm.csv").exists()
    assert "Holdout 确认" in (d / "report.md").read_text(encoding="utf-8")


def test_cluster_size_and_single_representative(planted_run):
    cand = pd.read_csv(planted_run / "candidates.csv").set_index("id")
    short = pd.read_csv(planted_run / "shortlist.csv").set_index("id")
    assert len({"sig|chg_1", "sig2|chg_1"} & set(short.index)) == 1
    rep = cand.loc["sig|chg_1", "rep"]
    assert cand.loc["sig2|chg_1", "rep"] == rep
    assert short.loc[rep, "cluster_size"] == 2
    nonrej = cand[cand["verdict"] != "REJECT"]
    assert short["cluster_size"].sum() == len(nonrej)
    assert (short["rep"] == short.index).all()


def test_lock_and_force(confirmed_run):
    d, _ = confirmed_run
    d2 = d.parent / "run_copy"
    shutil.copytree(d, d2)
    before = (d2 / "confirm.json").read_text()
    with pytest.raises(HoldoutLockedError):
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
    if (cand["verdict"] != "REJECT").sum() == 0:
        assert "没有候选通过门槛" in rep
        assert len(pd.read_csv(noise_run / "shortlist.csv")) == 0


def test_nothing_to_confirm(tmp_path, noise_run):
    d = tmp_path / "run"
    shutil.copytree(noise_run, d)
    pd.read_csv(d / "shortlist.csv").iloc[:0].to_csv(d / "shortlist.csv", index=False)
    assert confirm(d) == {"status": "NOTHING_TO_CONFIRM"}
    assert not (d / "confirm.lock").exists()
    assert not (d / "confirm.json").exists()


def test_report_content(planted_run, noise_run):
    for d in (planted_run, noise_run):
        rep = (d / "report.md").read_text(encoding="utf-8")
        for w in ("噪声预期", "FWER", "omnibus"):
            assert w not in rep
        for h in ("## 结论", "## 数据与样本", "## 判定标准", "## Shortlist（去重后）", "## 提示与局限", "## 下一步"):
            assert h in rep
    rep = (planted_run / "report.md").read_text(encoding="utf-8")
    assert "5000" in rep and "短样本候选的 holdout 也短" in rep


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
    out = dedupe(["b", "a", "c"], df, 0.7)  # b has the larger |t|, so it is the representative
    assert out == {"b": "b", "a": "b", "c": "c"}
    # too little overlap counts as uncorrelated
    df2 = df.copy()
    df2.loc[df2.index[:450], "a"] = np.nan
    assert dedupe(["b", "a"], df2, 0.7, min_overlap=100) == {"b": "b", "a": "a"}


def test_backtest():
    rng = np.random.default_rng(0)
    n = 1000
    s = rng.normal(size=n)
    s[:5] = np.nan
    sigma = np.full(n, 0.01)
    y = 0.05 * np.r_[s[1:], 0.0].clip(-4, 4) + rng.normal(size=n)
    y = np.nan_to_num(y)
    b0 = backtest(s, y, sigma, 1.0, 0.0)
    assert b0["sharpe_net"] == pytest.approx(b0["sharpe_gross"])
    b1 = backtest(s, y, sigma, 1.0, 1e-4)
    assert b1["sharpe_net"] < b1["sharpe_gross"]
    bm = backtest(s, y, sigma, -1.0, 0.0)
    assert bm["sharpe_gross"] == pytest.approx(-b0["sharpe_gross"])
    assert b0["turnover"] > 0 and abs(b0["ac1"]) < 0.2


def test_cli_locked_exit_code(confirmed_run):
    d, _ = confirmed_run
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, "-m", "tsminer", "confirm", str(d)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 2
