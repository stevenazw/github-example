"""Mining pipeline: train-only statistics, multiple-testing verdicts, outputs.

Nothing computed here may depend on the holdout segment; ``confirm`` is the only
place that opens it (and only once).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .databank import Panel, build_panel
from .stats import bh_q, eb_two_group, fwer_p, hac_t, holm, shift_null, two_sided_p
from .transforms import build_candidates, forward_target

WATCHLIST_POOL = 200
WATCHLIST_SIZE = 20
CAND_COLUMNS = ["id", "series", "transform", "group", "kind", "freq", "n", "t", "null_sd", "z", "ic", "sign",
                "p_cal", "q_bh", "p_fwer", "verdict", "p_signal", "shrink", "post_ic", "p_pos",
                "t_h1", "t_h2", "t_xlag", "flags", "rep"]
BT_COLUMNS = ["sharpe_gross", "sharpe_net", "turnover", "ac1"]


# ------------------------------------------------------------------ prepare
def prepare(panel: Panel, cfg) -> dict:
    """Candidate signals on the full (causal) time axis plus the train / holdout row split."""
    S, cand_meta = build_candidates(panel.X, panel.meta, cfg.signals.std_window, cfg.signals.clip)
    y_fwd, sigma = forward_target(panel.target, cfg.signals.vol_halflife)
    n = len(panel.target)
    split_idx = int(np.floor(n * (1.0 - cfg.sample.holdout_frac)))
    pos = np.arange(n)
    fin = np.isfinite(y_fwd.to_numpy(dtype=float))
    # row split_idx-1 carries the return of the first holdout day, so it belongs to holdout
    train_rows = pos[(pos < split_idx - 1) & fin]
    holdout_rows = pos[(pos >= split_idx - 1) & fin]
    return {"S": S, "cand_meta": cand_meta, "y_fwd": y_fwd, "sigma": sigma,
            "split_idx": split_idx, "train_rows": train_rows, "holdout_rows": holdout_rows}


# ------------------------------------------------------------ pure helpers
def assign_verdict(p_fwer, q_bh, fwer_alpha, fdr_q) -> str:
    if np.isfinite(p_fwer) and p_fwer <= fwer_alpha:
        return "PASS"
    if np.isfinite(q_bh) and q_bh <= fdr_q:
        return "WEAK_PASS"
    return "REJECT"


def dedupe(ids, S_train_df, thresh, min_overlap=100) -> dict:
    """Greedy de-redundancy; ids must be ordered by |t| descending. Returns id -> representative id."""
    ids = list(ids)
    if not ids:
        return {}
    X = S_train_df[ids].to_numpy(dtype=float)
    fin = np.isfinite(X)
    reps: list[int] = []
    out: dict = {}
    for j, cid in enumerate(ids):
        hit = None
        for k in reps:
            m = fin[:, j] & fin[:, k]
            if m.sum() < min_overlap:
                continue
            a, b = X[m, j], X[m, k]
            a, b = a - a.mean(), b - b.mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            if den > 0 and abs((a * b).sum() / den) >= thresh:
                hit = k
                break
        if hit is None:
            reps.append(j)
            out[cid] = cid
        else:
            out[cid] = ids[hit]
    return out


def _sharpe(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(252.0)) if sd > 0 else float("nan")


def backtest(s, y_fwd, sigma, sign, cost) -> dict:
    """Descriptive vol-scaled backtest of sign * s on one row set (1-D arrays)."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y_fwd, dtype=float)
    sg = np.asarray(sigma, dtype=float)
    nan = float("nan")
    fin = np.flatnonzero(np.isfinite(s))
    if len(fin) == 0:
        return {"sharpe_gross": nan, "sharpe_net": nan, "turnover": nan, "ac1": nan}
    i0 = fin[0]
    s, y, sg = s[i0:], y[i0:], sg[i0:]
    pos = sign * np.where(np.isfinite(s), s, 0.0)
    gross = pos * y
    notional = pos / sg
    cost_t = cost * np.abs(np.diff(notional, prepend=0.0))
    net = gross - cost_t
    turnover = float(np.mean(np.abs(np.diff(pos)))) if len(pos) > 1 else nan
    a, b = s[1:], s[:-1]
    m = np.isfinite(a) & np.isfinite(b)
    ac1 = nan
    if m.sum() > 2:
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        ac1 = float((a * b).sum() / den) if den > 0 else nan
    return {"sharpe_gross": _sharpe(gross), "sharpe_net": _sharpe(net), "turnover": turnover, "ac1": ac1}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return o


# --------------------------------------------------------------------- mine
def mine(cfg, run_dir, panel=None, config_path=None) -> dict:
    from .report import write_report

    if config_path is None:
        raise ValueError("config_path is required: confirm rebuilds the config from run_dir/config.toml")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if panel is None:
        panel = build_panel(cfg)
    panel.save(run_dir / "panel")
    shutil.copyfile(config_path, run_dir / "config.toml")

    st, vd = cfg.stats, cfg.verdict
    prep = prepare(panel, cfg)
    S, cm, y_fwd, sigma = prep["S"], prep["cand_meta"], prep["y_fwd"], prep["sigma"]
    tr = prep["train_rows"]
    S_tr = S.to_numpy(dtype=float)[tr]
    y_tr = y_fwd.to_numpy(dtype=float)[tr]
    sig_tr = sigma.to_numpy(dtype=float)[tr]

    t_all, ic_all, n_all = hac_t(S_tr, y_tr, st.hac_lags, st.min_obs)
    fam0 = np.flatnonzero(np.isfinite(t_all))
    if len(fam0) == 0:
        raise ValueError("no candidate has a finite t on the train segment")
    gcodes0, guniq0 = pd.factorize(cm.loc[np.array(S.columns)[fam0], "group"].to_numpy(), sort=True)
    null = shift_null(S_tr[:, fam0], y_tr, st.hac_lags, st.min_obs, st.n_boot, st.min_shift, st.seed,
                      groups=gcodes0)
    # studentization needs a valid null sd; candidates without one leave the family
    keep = np.isfinite(null["t_sd"]) & (null["t_sd"] > 0)
    fam = fam0[keep]
    n_dropped = int(len(t_all) - len(fam))
    m = len(fam)
    if m == 0:
        raise ValueError("no candidate has a valid null sd")
    ids = np.array(S.columns)[fam]
    t, ic, n_obs = t_all[fam], ic_all[fam], n_all[fam]
    null_sd = null["t_sd"][keep]
    z = t / null_sd
    S_fam = S_tr[:, fam]
    gcodes, K, guniq = gcodes0[keep], len(guniq0), guniq0
    print(f"[mine] rows={len(S)} train_rows={len(tr)} candidates={S.shape[1]} family={m} dropped={n_dropped}")

    B = st.n_boot
    p_fwer = fwer_p(z, null["max_abs_z"])
    t_var = null["t_var"]
    p_cal = two_sided_p(np.abs(z))
    q_bh = bh_q(p_cal)
    hurdle = float(np.nanquantile(null["max_abs_z"], 1.0 - vd.fwer_alpha))
    obs = float(np.mean(z ** 2))
    p_omni = float((1 + np.sum(null["mean_z2"] >= obs)) / (B + 1))
    present = np.flatnonzero(np.bincount(gcodes, minlength=K) > 0)
    g_obs = np.array([np.mean(z[gcodes == k] ** 2) for k in present])
    g_p = np.array([(1 + np.sum(null["group_mean_z2"][:, k] >= o)) / (B + 1) for k, o in zip(present, g_obs)])
    g_holm = holm(g_p)
    v0 = float(np.nanmean(null["mean_z2"]))
    eb = eb_two_group(z, ic, v0)
    verdict = np.array([assign_verdict(a, b, vd.fwer_alpha, vd.fdr_q) for a, b in zip(p_fwer, q_bh)])

    # flags (informational only)
    h = len(tr) // 2
    mo_half = max(60, st.min_obs // 2)
    t_h1 = hac_t(S_fam[:h], y_tr[:h], st.hac_lags, mo_half)[0]
    t_h2 = hac_t(S_fam[h:], y_tr[h:], st.hac_lags, mo_half)[0]
    S_np = S.to_numpy(dtype=float)
    S_lag = np.vstack([np.full((1, S_np.shape[1]), np.nan), S_np[:-1]])
    t_xlag = hac_t(S_lag[tr][:, fam], y_tr, st.hac_lags, st.min_obs)[0]
    flags = []
    for i in range(m):
        f = []
        if np.isfinite(t_h1[i]) and np.isfinite(t_h2[i]) and t_h1[i] * t_h2[i] < 0:
            f.append("half_flip")
        if not (np.isfinite(t_xlag[i]) and abs(t_xlag[i]) >= vd.extra_lag_t_min and t_xlag[i] * t[i] > 0):
            f.append("xlag_weak")
        flags.append(";".join(f))

    order = np.argsort(-np.abs(z), kind="stable")
    cand = pd.DataFrame({
        "id": ids, "series": cm.loc[ids, "series"].to_numpy(), "transform": cm.loc[ids, "transform"].to_numpy(),
        "group": cm.loc[ids, "group"].to_numpy(), "kind": cm.loc[ids, "kind"].to_numpy(),
        "freq": cm.loc[ids, "freq"].to_numpy(), "n": n_obs, "t": t, "null_sd": null_sd, "z": z, "ic": ic, "sign": np.sign(t),
        "p_cal": p_cal, "q_bh": q_bh, "p_fwer": p_fwer, "verdict": verdict,
        "p_signal": eb["p_signal"], "shrink": eb["shrink"], "post_ic": eb["post_ic"], "p_pos": eb["p_pos"],
        "t_h1": t_h1, "t_h2": t_h2, "t_xlag": t_xlag, "flags": flags,
    }).iloc[order].reset_index(drop=True)

    # de-redundancy
    S_tr_df = pd.DataFrame(S_tr, columns=S.columns)
    passed = cand.loc[cand["verdict"] != "REJECT", "id"].tolist()
    rep_map = dedupe(passed, S_tr_df, vd.dedupe_corr)
    cand["rep"] = [rep_map.get(i, "") for i in cand["id"]]
    short_ids = list(dict.fromkeys(rep_map.values()))
    pool = cand["id"].head(WATCHLIST_POOL).tolist()
    watch_ids = list(dict.fromkeys(dedupe(pool, S_tr_df, vd.dedupe_corr).values()))[:WATCHLIST_SIZE]

    cidx = cand.set_index("id", drop=False)
    desc = panel.meta["description"]

    def table(sel):
        rows = []
        for cid in sel:
            r = cidx.loc[cid]
            bt = backtest(S_tr_df[cid].to_numpy(), y_tr, sig_tr, float(r["sign"]), cfg.target.cost_per_side)
            rows.append({**{c: r[c] for c in CAND_COLUMNS}, **bt, "description": desc.get(r["series"], "")})
        return pd.DataFrame(rows, columns=CAND_COLUMNS + BT_COLUMNS + ["description"])

    cand.to_csv(run_dir / "candidates.csv", index=False)
    table(short_ids).to_csv(run_dir / "shortlist.csv", index=False)
    table(watch_ids).to_csv(run_dir / "watchlist.csv", index=False)
    pd.DataFrame({"group": guniq[present], "n_cand": np.bincount(gcodes, minlength=K)[present],
                  "mean_z2": g_obs, "null_mean_z2": np.nanmean(null["group_mean_z2"], axis=0)[present], "p": g_p,
                  "p_holm": g_holm}).to_csv(run_dir / "groups.csv", index=False)

    cal = panel.target.index
    tau2 = eb["tau2"]
    run = {
        "target": cfg.target.name,
        "config": cfg.to_dict(),
        "panel_info": panel.info,
        "dates": {"sample_start": str(cal[0].date()),
                  "train_end": str(cal[tr[-1] + 1].date()),
                  "holdout_start": str(cal[prep["split_idx"]].date()),
                  "sample_end": str(cal[-1].date())},
        "n_rows": len(cal), "split_idx": prep["split_idx"],
        "n_train_rows": len(tr), "n_holdout_rows": len(prep["holdout_rows"]),
        "n_series": int(panel.X.shape[1]), "n_candidates": int(S.shape[1]),
        "n_dropped": n_dropped, "family_size": m,
        "null": {"t_var": t_var, "hurdle": hurdle, "v0": v0, "n_boot": B,
                 "null_sd_median": float(np.median(null_sd)), "null_sd_min": float(null_sd.min()),
                 "null_sd_max": float(null_sd.max())},
        "omnibus": {"mean_z2": obs, "p": p_omni},
        "eb_tau2": tau2, "eb_pi0": eb["pi0"],
        "composite_defined": bool(tau2 > 0 and eb["pi0"] < 1),
        "verdict_counts": {v: int((verdict == v).sum()) for v in ("PASS", "WEAK_PASS", "REJECT")},
        "shortlist": short_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    run = _jsonable(run)
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    write_report(run_dir)
    print(f"[mine] verdicts={run['verdict_counts']} shortlist={len(short_ids)} -> {run_dir}")
    return run
