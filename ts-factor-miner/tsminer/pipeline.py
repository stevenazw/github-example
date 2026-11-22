"""Mining pipeline: per-candidate single-factor tests on each candidate's own train rows.

Every candidate is tested on the rows where it overlaps the target, split into train / holdout
by its own sample. Nothing computed here may depend on a candidate's holdout rows;
``confirm`` is the only place that opens them (and only once).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .databank import Panel, build_panel
from .stats import hac_t, null_sd, two_sided_p
from .transforms import build_candidates, forward_target

SPLIT_QUANTUM = 21
CAND_COLUMNS = ["id", "series", "transform", "group", "kind", "freq", "n_train", "train_start", "train_end",
                "n_holdout", "t", "null_sd", "z", "ic", "sign", "p", "verdict", "t_h1", "t_h2", "t_xlag",
                "flags", "rep"]
BT_COLUMNS = ["sharpe_gross", "sharpe_net", "turnover", "ac1"]
SHORT_COLUMNS = CAND_COLUMNS + ["cluster_size"] + BT_COLUMNS + ["description"]


# ------------------------------------------------------------------ split
def split_rows(valid, holdout_frac, quantum=SPLIT_QUANTUM):
    """Per-column train / holdout masks on the rows of ``valid`` (T, N bool).

    Column j has n_j valid rows; the holdout starts at the (k_j+1)-th valid row, k_j = floor(n_j * (1 - frac)),
    rounded down to a multiple of ``quantum``: end_row[j] = (p_j // quantum) * quantum (p_j = T if k_j == n_j).
    train = valid & row < end_row, holdout = valid & row >= end_row. Returns (train, holdout, end_row).
    """
    valid = np.asarray(valid, dtype=bool)
    cs = np.cumsum(valid, axis=0)
    n = cs[-1] if len(valid) else np.zeros(valid.shape[1], dtype=int)
    k = np.floor(n * (1.0 - holdout_frac)).astype(int)
    p = (cs < (k + 1)[None, :]).sum(axis=0)   # first row where the running count reaches k+1; T if none
    end_row = (p // quantum) * quantum
    before = np.arange(valid.shape[0])[:, None] < end_row[None, :]
    return valid & before, valid & ~before, end_row


# ------------------------------------------------------------------ prepare
def prepare(panel: Panel, cfg) -> dict:
    """Candidate signals on the full (causal) time axis plus each candidate's own train / holdout rows.

    ``rows`` are the calendar positions where y_fwd is finite; ``train_mask``, ``holdout_mask`` (len(rows), N)
    and ``end_row`` (N,) are defined on those rows.
    """
    S, cand_meta = build_candidates(panel.X, panel.meta, cfg.signals.std_window, cfg.signals.clip)
    y_fwd, sigma = forward_target(panel.target, cfg.signals.vol_halflife)
    rows = np.flatnonzero(np.isfinite(y_fwd.to_numpy(dtype=float)))
    valid = np.isfinite(S.to_numpy(dtype=float)[rows])
    train_mask, holdout_mask, end_row = split_rows(valid, cfg.sample.holdout_frac)
    return {"S": S, "cand_meta": cand_meta, "y_fwd": y_fwd, "sigma": sigma, "rows": rows,
            "train_mask": train_mask, "holdout_mask": holdout_mask, "end_row": end_row}


# ------------------------------------------------------------ pure helpers
def assign_verdict(z, z_pass, z_weak) -> str:
    """Single-factor |z| thresholds; non-finite z is REJECT."""
    if not np.isfinite(z):
        return "REJECT"
    a = abs(float(z))
    return "PASS" if a >= z_pass else "WEAK_PASS" if a >= z_weak else "REJECT"


def dedupe(ids, S_train_df, thresh, min_overlap=100) -> dict:
    """Greedy de-redundancy; ids must be ordered by |z| descending. Returns id -> representative id."""
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
def _edge_dates(mask, rows, cal):
    """First / last calendar date (str) of the True rows of each column; "" for empty columns."""
    any_ = mask.any(axis=0)
    first = rows[np.argmax(mask, axis=0)]
    last = rows[mask.shape[0] - 1 - np.argmax(mask[::-1], axis=0)]
    fmt = lambda pos: np.where(any_, cal[pos].strftime("%Y-%m-%d"), "")
    return fmt(first), fmt(last)


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
    S, cm, y_fwd, sigma, rows = prep["S"], prep["cand_meta"], prep["y_fwd"], prep["sigma"], prep["rows"]
    tr_mask, ho_mask, end_row = prep["train_mask"], prep["holdout_mask"], prep["end_row"]
    cal = panel.target.index
    y = y_fwd.to_numpy(dtype=float)[rows]
    sig = sigma.to_numpy(dtype=float)[rows]
    S_np = S.to_numpy(dtype=float)
    S_tr = np.where(tr_mask, S_np[rows], np.nan)
    N = S.shape[1]

    t_all, ic_all, n_tr_all = hac_t(S_tr, y, st.hac_lags, st.min_obs)

    # per-candidate null scale: candidates sharing a split point share one circular-shift run on y[:e]
    sd_all = np.full(N, np.nan)
    finite_t = np.isfinite(t_all)
    for e in np.unique(end_row[finite_t]):
        if e - 2 * st.min_shift < 1:
            continue
        cols = np.flatnonzero(finite_t & (end_row == e))
        sd_all[cols] = null_sd(S_tr[:e][:, cols], y[:e], st.hac_lags, st.min_obs, st.n_boot, st.min_shift,
                               st.seed)
    ok = finite_t & np.isfinite(sd_all) & (sd_all > 0)
    sel = np.flatnonzero(ok)
    n_tested = int(len(sel))
    n_skipped = int(N - n_tested)
    if n_tested == 0:
        raise ValueError("no candidate has a finite t and a valid null sd on its train rows")
    ids = np.array(S.columns)[sel]
    t, ic, n_tr = t_all[sel], ic_all[sel], n_tr_all[sel]
    null = sd_all[sel]
    z = t / null
    p = two_sided_p(z)
    verdict = np.array([assign_verdict(a, vd.z_pass, vd.z_weak) for a in z])
    print(f"[mine] rows={len(rows)} candidates={N} tested={n_tested} skipped={n_skipped}")

    # flags (informational only), on each candidate's own train rows
    trs = tr_mask[:, sel]
    S_sel = S_tr[:, sel]
    cs = np.cumsum(trs, axis=0)
    half = trs.sum(axis=0) // 2
    mo_half = max(60, st.min_obs // 2)
    t_h1 = hac_t(np.where(trs & (cs <= half), S_sel, np.nan), y, st.hac_lags, mo_half)[0]
    t_h2 = hac_t(np.where(trs & (cs > half), S_sel, np.nan), y, st.hac_lags, mo_half)[0]
    S_lag = np.vstack([np.full((1, N), np.nan), S_np[:-1]])[rows][:, sel]
    t_xlag = hac_t(np.where(trs, S_lag, np.nan), y, st.hac_lags, st.min_obs)[0]
    flags = []
    for i in range(n_tested):
        f = []
        if np.isfinite(t_h1[i]) and np.isfinite(t_h2[i]) and t_h1[i] * t_h2[i] < 0:
            f.append("half_flip")
        if not (np.isfinite(t_xlag[i]) and abs(t_xlag[i]) >= vd.extra_lag_t_min and t_xlag[i] * t[i] > 0):
            f.append("xlag_weak")
        flags.append(";".join(f))

    tr_start, tr_end = _edge_dates(trs, rows, cal)
    order = np.argsort(-np.abs(z), kind="stable")
    cand = pd.DataFrame({
        "id": ids, "series": cm.loc[ids, "series"].to_numpy(), "transform": cm.loc[ids, "transform"].to_numpy(),
        "group": cm.loc[ids, "group"].to_numpy(), "kind": cm.loc[ids, "kind"].to_numpy(),
        "freq": cm.loc[ids, "freq"].to_numpy(), "n_train": n_tr, "train_start": tr_start, "train_end": tr_end,
        "n_holdout": ho_mask[:, sel].sum(axis=0), "t": t, "null_sd": null, "z": z, "ic": ic, "sign": np.sign(t),
        "p": p, "verdict": verdict, "t_h1": t_h1, "t_h2": t_h2, "t_xlag": t_xlag, "flags": flags,
    }).iloc[order].reset_index(drop=True)

    # de-redundancy of the non-REJECT candidates (pairwise correlation on the common train rows)
    S_tr_df = pd.DataFrame(S_tr, columns=S.columns)
    passed = cand.loc[cand["verdict"] != "REJECT", "id"].tolist()
    rep_map = dedupe(passed, S_tr_df, vd.dedupe_corr)
    cand["rep"] = [rep_map.get(i, "") for i in cand["id"]]
    short_ids = list(dict.fromkeys(rep_map.values()))
    cluster = pd.Series(list(rep_map.values())).value_counts()

    col_of = {c: i for i, c in enumerate(S.columns)}
    cidx = cand.set_index("id", drop=False)
    desc = panel.meta["description"]
    rows_out = []
    for cid in short_ids:
        r = cidx.loc[cid]
        m = tr_mask[:, col_of[cid]]
        bt = backtest(S_np[rows][m, col_of[cid]], y[m], sig[m], float(r["sign"]), cfg.target.cost_per_side)
        rows_out.append({**{c: r[c] for c in CAND_COLUMNS}, "cluster_size": int(cluster[cid]), **bt,
                         "description": desc.get(r["series"], "")})
    short = pd.DataFrame(rows_out, columns=SHORT_COLUMNS)

    cand.to_csv(run_dir / "candidates.csv", index=False)
    short.to_csv(run_dir / "shortlist.csv", index=False)

    run = {
        "target": cfg.target.name,
        "config": cfg.to_dict(),
        "panel_info": panel.info,
        "dates": {"sample_start": str(cal[0].date()), "sample_end": str(cal[-1].date())},
        "n_rows": len(cal),
        "n_series": int(panel.X.shape[1]), "n_candidates": int(N),
        "n_tested": n_tested, "n_skipped": n_skipped,
        "n_train_summary": {"min": int(n_tr.min()), "median": float(np.median(n_tr)), "max": int(n_tr.max())},
        "verdict_counts": {v: int((verdict == v).sum()) for v in ("PASS", "WEAK_PASS", "REJECT")},
        "shortlist": short_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    run = _jsonable(run)
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    write_report(run_dir)
    print(f"[mine] verdicts={run['verdict_counts']} shortlist={len(short_ids)} -> {run_dir}")
    return run
