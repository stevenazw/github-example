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
from tsminer.pipeline import assign_verdict, backtest, dedupe, mine, prepare

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
min_obs = 500
n_boot = 200
min_shift = 60
seed = 7
"""


def make_panel(seed, plant=False, n=N):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    d = rng.normal(0, 1, (n, N_SERIES))
    noise = rng.normal(0, 0.005, n)
    r = noise.copy()
    if plant:
        ic = 0.15
        d0 = d[:, 0]
        r[1:] = ic * 0.005 * d0[:-1] / d0.std() + np.sqrt(1 - ic**2) * noise[1:]
    X = pd.DataFrame(np.cumsum(d, axis=0), index=idx, columns=[f"s{i}" for i in range(N_SERIES)])
    if plant:
        X = X.rename(columns={"s0": "sig"})
    meta = pd.DataFrame({
        "ticker": X.columns, "field": "PX_LAST", "kind": "level", "freq": "D",
        "group": [GROUPS[i % 3] for i in range(N_SERIES)], "lag": 1,
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
def planted_run(tmp_path_factory, cfg, cfg_path):
    d = tmp_path_factory.mktemp("planted") / "run"
    mine(cfg, d, panel=make_panel(1, plant=True), config_path=cfg_path)
    return d


@pytest.fixture(scope="module")
def confirmed_run(tmp_path_factory, planted_run):
    d = tmp_path_factory.mktemp("confirmed") / "run"
    shutil.copytree(planted_run, d)
    return d, confirm(d)


# NOTE on the noise seed: with a 5% family-wise error rate, some seeds yield a false PASS
# (or a BH WEAK_PASS) by design; seed 11 was chosen because it yields none.
NOISE_SEED = 11


@pytest.fixture(scope="module")
def noise_run(tmp_path_factory, cfg, cfg_path):
    d = tmp_path_factory.mktemp("noise") / "run"
    mine(cfg, d, panel=make_panel(NOISE_SEED), config_path=cfg_path)
    return d


# ------------------------------------------------------------------ tests
def test_holdout_isolation(tmp_path, cfg, cfg_path):
    panel = make_panel(3, plant=True)
    split = prepare(panel, cfg)["split_idx"]
    rng = np.random.default_rng(999)
    p2 = Panel(target=panel.target.copy(), X=panel.X.copy(), meta=panel.meta, excluded=panel.excluded,
               info=panel.info)
    p2.target.iloc[split:] = rng.normal(0, 0.005, N - split)
    p2.X.iloc[split:] = rng.normal(0, 50, (N - split, N_SERIES))
    a, b = tmp_path / "a", tmp_path / "b"
    mine(cfg, a, panel=panel, config_path=cfg_path)
    mine(cfg, b, panel=p2, config_path=cfg_path)
    for f in ("candidates.csv", "shortlist.csv", "groups.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(a / f), pd.read_csv(b / f), check_exact=True)
    ra, rb = json.loads((a / "run.json").read_text()), json.loads((b / "run.json").read_text())
    ra.pop("created_at")
    rb.pop("created_at")
    assert ra == rb


def test_prepare_split(cfg):
    panel = make_panel(2)
    prep = prepare(panel, cfg)
    tr, ho = prep["train_rows"], prep["holdout_rows"]
    assert len(set(tr) & set(ho)) == 0
    finite = np.flatnonzero(np.isfinite(prep["y_fwd"].to_numpy()))
    assert set(tr) | set(ho) == set(finite)
    assert tr.max() + 1 < prep["split_idx"]
    assert ho.min() == prep["split_idx"] - 1
    assert prep["split_idx"] == int(np.floor(N * 0.7))


def test_planted_signal_confirmed(confirmed_run):
    d, res = confirmed_run
    cand = pd.read_csv(d / "candidates.csv").set_index("id")
    assert cand.loc["sig|chg_1", "verdict"] == "PASS"
    short = pd.read_csv(d / "shortlist.csv")
    assert cand.loc["sig|chg_1", "rep"] in set(short["id"])
    assert res["status"] == "DONE"
    row = pd.read_csv(d / "confirm.csv").set_index("id").loc[cand.loc["sig|chg_1", "rep"]]
    assert row["verdict"] == "CONFIRMED"
    comp = res["composite"]
    assert comp is not None and comp["t"] > 1.645
    assert (d / "confirm.lock").exists()
    assert "Holdout 确认" in (d / "report.md").read_text(encoding="utf-8")


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


def test_pure_noise_no_pass(noise_run):
    cand = pd.read_csv(noise_run / "candidates.csv")
    assert (cand["verdict"] == "PASS").sum() == 0
    assert (cand["verdict"] == "WEAK_PASS").sum() == 0
    assert "没有候选通过多重检验" in (noise_run / "report.md").read_text(encoding="utf-8")


def test_nothing_to_confirm(tmp_path, noise_run):
    d = tmp_path / "run"
    shutil.copytree(noise_run, d)
    run = json.loads((d / "run.json").read_text())
    run["composite_defined"] = False
    run["shortlist"] = []
    (d / "run.json").write_text(json.dumps(run))
    pd.read_csv(d / "shortlist.csv").iloc[:0].to_csv(d / "shortlist.csv", index=False)
    assert confirm(d) == {"status": "NOTHING_TO_CONFIRM"}
    assert not (d / "confirm.lock").exists()
    assert not (d / "confirm.json").exists()


def test_assign_verdict_boundaries():
    assert assign_verdict(0.05, 0.5, 0.05, 0.10) == "PASS"
    assert assign_verdict(0.0501, 0.10, 0.05, 0.10) == "WEAK_PASS"
    assert assign_verdict(0.0501, 0.1001, 0.05, 0.10) == "REJECT"
    assert assign_verdict(float("nan"), float("nan"), 0.05, 0.10) == "REJECT"


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
