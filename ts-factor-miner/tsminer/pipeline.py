"""Mining pipeline: per-candidate tests on each candidate's own in-sample / observation rows.

Every candidate is tested on the rows where it overlaps the target, split into IS (in-sample), OBS
(observation period) and OOS (out-of-sample) by its own valid rows. ``mine`` never reads a candidate's OOS
values (rows >= e2_j), and its IS statistics never read rows >= e1_j; ``confirm`` is the only place that
opens the OOS (and only once).
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
OBS_MIN_OBS = 60
HORIZONS = (1, 2, 5, 10, 20, 60)
YEAR_MIN_ROWS = 60
CAND_COLUMNS = ["id", "series", "transform", "group", "kind", "freq", "n_is", "is_start", "is_end", "n_obs",
                "n_oos", "t", "null_sd", "z", "ic", "sign", "p", "verdict", "ic_obs_signed", "retention",
                "obs_status", "t_dev", "sharpe_gross_dev", "sharpe_net_dev", "trade_status", "t_h1", "t_h2",
                "t_xlag", "flags", "rep"]
TRADE_COLUMNS = ["sharpe_gross_dev", "sharpe_net_dev", "trade_status"]
SHORT_CAND_COLUMNS = [c for c in CAND_COLUMNS if c not in TRADE_COLUMNS]
BT_COLUMNS = ["sharpe_gross", "sharpe_net", "turnover"]
IC_COLUMNS = [f"ic_h{h}" for h in HORIZONS]
AC_COLUMNS = [f"ac_{h}" for h in HORIZONS]
SHORT_COLUMNS = (SHORT_CAND_COLUMNS + ["cluster_size"] + BT_COLUMNS + IC_COLUMNS + AC_COLUMNS +
                 ["signal_half_life", "year_hit", "n_years", "description"])


# ------------------------------------------------------------------ split
def split_rows(valid, is_frac, obs_frac, quantum=SPLIT_QUANTUM):
    """Per-column IS / OBS / OOS masks on the rows of ``valid`` (T, N bool).

    Column j has n_j valid rows; k1 = floor(n_j * is_frac), k2 = floor(n_j * (is_frac + obs_frac));
    p1 / p2 = row of the (k1+1)-th / (k2+1)-th valid row (T if it does not exist);
    e1 = (p1 // quantum) * quantum, e2 = max(e1, (p2 // quantum) * quantum).
    IS = valid & row < e1; OBS = valid & e1 <= row < e2; OOS = valid & row >= e2.
    Returns (is_mask, obs_mask, oos_mask, e1, e2).
    """
    valid = np.asarray(valid, dtype=bool)
    cs = np.cumsum(valid, axis=0)
    n = cs[-1] if len(valid) else np.zeros(valid.shape[1], dtype=int)
    k1 = np.floor(n * is_frac).astype(int)
    k2 = np.floor(n * (is_frac + obs_frac)).astype(int)
    p1 = (cs < (k1 + 1)[None, :]).sum(axis=0)   # first row where the running count reaches k+1; T if none
    p2 = (cs < (k2 + 1)[None, :]).sum(axis=0)
    e1 = (p1 // quantum) * quantum
    e2 = np.maximum(e1, (p2 // quantum) * quantum)
    row = np.arange(valid.shape[0])[:, None]
    is_mask = valid & (row < e1[None, :])
    obs_mask = valid & (row >= e1[None, :]) & (row < e2[None, :])
    oos_mask = valid & (row >= e2[None, :])
    return is_mask, obs_mask, oos_mask, e1, e2


# ------------------------------------------------------------------ prepare
def prepare(panel: Panel, cfg) -> dict:
    """Candidate signals on the full (causal) time axis plus each candidate's own IS / OBS / OOS rows.

    ``rows`` are the calendar positions where y_fwd is finite; the masks (len(rows), N) and ``e1``, ``e2``
    (N,) are defined on those rows.
    """
    S, cand_meta = build_candidates(panel.X, panel.meta, cfg.signals.std_window, cfg.signals.clip)
    y_fwd, sigma = forward_target(panel.target, cfg.signals.vol_halflife)
    rows = np.flatnonzero(np.isfinite(y_fwd.to_numpy(dtype=float)))
    valid = np.isfinite(S.to_numpy(dtype=float)[rows])
    is_mask, obs_mask, oos_mask, e1, e2 = split_rows(valid, cfg.sample.is_frac, cfg.sample.obs_frac)
    return {"S": S, "cand_meta": cand_meta, "y_fwd": y_fwd, "sigma": sigma, "rows": rows,
            "is_mask": is_mask, "obs_mask": obs_mask, "oos_mask": oos_mask, "e1": e1, "e2": e2}


# ------------------------------------------------------------ pure helpers
def assign_verdict(z, z_pass, z_weak) -> str:
    """Single-factor |z| thresholds; non-finite z is REJECT."""
    if not np.isfinite(z):
        return "REJECT"
    a = abs(float(z))
    return "PASS" if a >= z_pass else "WEAK_PASS" if a >= z_weak else "REJECT"


def dedupe(ids, S_dev_df, thresh, min_overlap=100) -> dict:
    """Greedy de-redundancy; ids must be ordered by priority (best first). Returns id -> representative id."""
    ids = list(ids)
    if not ids:
        return {}
    X = S_dev_df[ids].to_numpy(dtype=float)
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


def dedupe_tradable(ids, trade_ok, S_dev_df, thresh, min_overlap=100) -> dict:
    """De-redundancy after the tradability screen: only ids with ``trade_ok[id]`` true enter ``dedupe`` (order
    kept, best first), so a cluster is represented by its best tradable member instead of vanishing when its
    best-t member fails the net-Sharpe bar. Returns id -> representative id for the tradable ids only."""
    kept = [i for i in ids if trade_ok.get(i, False)]
    return dedupe(kept, S_dev_df, thresh, min_overlap)


def _sharpe(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(252.0)) if sd > 0 else float("nan")


def positions(s, sign, buffer) -> np.ndarray:
    """Continuous position with a no-trade buffer: target = sign * s (NaN -> 0); pos_0 = target_0;
    pos_t = target_t if |target_t - pos_{t-1}| > buffer else pos_{t-1}."""
    s = np.asarray(s, dtype=float)
    target = sign * np.where(np.isfinite(s), s, 0.0)
    pos = np.empty_like(target)
    if len(target) == 0:
        return pos
    pos[0] = target[0]
    for t in range(1, len(target)):
        pos[t] = target[t] if abs(target[t] - pos[t - 1]) > buffer else pos[t - 1]
    return pos


def backtest(s, y_fwd, sigma, sign, cost, buffer) -> dict:
    """Descriptive vol-scaled backtest of the buffered position on one row set (1-D arrays)."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y_fwd, dtype=float)
    sg = np.asarray(sigma, dtype=float)
    nan = float("nan")
    fin = np.flatnonzero(np.isfinite(s))
    if len(fin) == 0:
        return {"sharpe_gross": nan, "sharpe_net": nan, "turnover": nan}
    i0 = fin[0]
    s, y, sg = s[i0:], y[i0:], sg[i0:]
    pos = positions(s, sign, buffer)
    gross = pos * y
    notional = pos / sg
    cost_t = cost * np.abs(np.diff(notional, prepend=0.0))
    net = gross - cost_t
    turnover = float(np.mean(np.abs(np.diff(pos)))) if len(pos) > 1 else nan
    return {"sharpe_gross": _sharpe(gross), "sharpe_net": _sharpe(net), "turnover": turnover}


def _corr(a, b, min_n=3) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def ic_decay(s, y, e2, sign, horizons=HORIZONS) -> list:
    """Signed IC of s_t against y[t+h-1] over pairs with s_t finite and both positions < e2 (rows space)."""
    s = np.asarray(s, dtype=float)[:e2]
    y = np.asarray(y, dtype=float)[:e2]
    out = []
    for h in horizons:
        n = len(s) - (h - 1)
        out.append(sign * _corr(s[:n], y[h - 1:h - 1 + n]) if n > 0 else float("nan"))
    return out


def signal_autocorr(s, horizons=HORIZONS) -> list:
    """corr(s_t, s_{t-h}) over pairs where both are finite (s is NaN outside the rows of interest)."""
    s = np.asarray(s, dtype=float)
    return [_corr(s[h:], s[:-h]) if len(s) > h else float("nan") for h in horizons]


def half_life(ac_1) -> float:
    """ln(0.5)/ln(ac_1) for 0 < ac_1 < 1; 0 for ac_1 <= 0; NaN for ac_1 >= 1 or non-finite."""
    if not np.isfinite(ac_1) or ac_1 >= 1:
        return float("nan")
    if ac_1 <= 0:
        return 0.0
    return float(np.log(0.5) / np.log(ac_1))


def yearly_hit(s, y, years, sign, min_rows=YEAR_MIN_ROWS) -> tuple[float, int]:
    """(share of years with signed IC > 0, number of years counted); a year counts with >= min_rows
    rows where s is finite."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    hits, n = 0, 0
    for yr in np.unique(years):
        m = (years == yr) & np.isfinite(s)
        if m.sum() < min_rows:
            continue
        ic = sign * _corr(s[m], y[m])
        n += 1
        hits += int(np.isfinite(ic) and ic > 0)
    return (hits / n if n else float("nan")), n


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
    is_m, obs_m, oos_m, e1, e2 = (prep[k] for k in ("is_mask", "obs_mask", "oos_mask", "e1", "e2"))
    cal = panel.target.index
    y = y_fwd.to_numpy(dtype=float)[rows]
    sig = sigma.to_numpy(dtype=float)[rows]
    S_np = S.to_numpy(dtype=float)
    S_rows = S_np[rows]
    N = S.shape[1]

    # ---- step 1: in-sample single-factor test (IS rows only)
    S_is = np.where(is_m, S_rows, np.nan)
    t_all, ic_all, n_is_all = hac_t(S_is, y, st.hac_lags, st.min_obs)

    # per-candidate null scale: candidates sharing e1 share one circular-shift run on y[:e1]
    sd_all = np.full(N, np.nan)
    finite_t = np.isfinite(t_all)
    for e in np.unique(e1[finite_t]):
        if e - 2 * st.min_shift < 1:
            continue
        cols = np.flatnonzero(finite_t & (e1 == e))
        sd_all[cols] = null_sd(S_is[:e][:, cols], y[:e], st.hac_lags, st.min_obs, st.n_boot, st.min_shift,
                               st.seed)
    ok = finite_t & np.isfinite(sd_all) & (sd_all > 0)
    sel = np.flatnonzero(ok)
    n_tested = int(len(sel))
    n_skipped = int(N - n_tested)
    if n_tested == 0:
        raise ValueError("no candidate has a finite t and a valid null sd on its in-sample rows")
    ids = np.array(S.columns)[sel]
    t, ic, n_is = t_all[sel], ic_all[sel], n_is_all[sel]
    null = sd_all[sel]
    z = t / null
    p = two_sided_p(z)
    verdict = np.array([assign_verdict(a, vd.z_pass, vd.z_weak) for a in z])
    sign = np.sign(t)
    print(f"[mine] rows={len(rows)} candidates={N} tested={n_tested} skipped={n_skipped}")

    # Every hac_t call below runs on ALL N columns (then subsets), so that a column's float result never
    # depends on which other candidates happen to be tested / alive (BLAS summation order).
    # flags on each candidate's own IS rows (informational only)
    cs = np.cumsum(is_m, axis=0)
    half = is_m.sum(axis=0) // 2
    mo_half = max(60, st.min_obs // 2)
    t_h1 = hac_t(np.where(is_m & (cs <= half), S_is, np.nan), y, st.hac_lags, mo_half)[0][sel]
    t_h2 = hac_t(np.where(is_m & (cs > half), S_is, np.nan), y, st.hac_lags, mo_half)[0][sel]
    S_lag = np.vstack([np.full((1, N), np.nan), S_np[:-1]])[rows]
    t_xlag = hac_t(np.where(is_m, S_lag, np.nan), y, st.hac_lags, st.min_obs)[0][sel]
    del S_lag, cs
    flags: list[list[str]] = []
    for i in range(n_tested):
        f = []
        if np.isfinite(t_h1[i]) and np.isfinite(t_h2[i]) and t_h1[i] * t_h2[i] < 0:
            f.append("half_flip")
        if not (np.isfinite(t_xlag[i]) and abs(t_xlag[i]) >= vd.extra_lag_t_min and t_xlag[i] * t[i] > 0):
            f.append("xlag_weak")
        flags.append(f)

    # ---- step 2: observation-period screen (only candidates that passed step 1)
    ic_obs_signed = np.full(n_tested, np.nan)
    retention = np.full(n_tested, np.nan)
    t_dev = np.full(n_tested, np.nan)
    obs_status = np.array([""] * n_tested, dtype=object)
    nr = np.flatnonzero(verdict != "REJECT")
    if len(nr):
        t_obs_all, ic_obs_all, _ = hac_t(np.where(obs_m, S_rows, np.nan), y, st.hac_lags, OBS_MIN_OBS)
        t_obs, ic_obs = t_obs_all[sel][nr], ic_obs_all[sel][nr]
        ic_obs_signed[nr] = sign[nr] * ic_obs
        with np.errstate(divide="ignore", invalid="ignore"):
            retention[nr] = ic_obs_signed[nr] / np.abs(ic[nr])
        for k, i in enumerate(nr):
            if not np.isfinite(t_obs[k]):
                obs_status[i] = "INSUFFICIENT"
                flags[i].append("obs_short")
            elif retention[i] >= vd.obs_min_retention:
                obs_status[i] = "KEEP"
            else:
                obs_status[i] = "DROP"
        alive = nr[obs_status[nr] != "DROP"]
        if len(alive):
            t_dev_all = hac_t(np.where(is_m | obs_m, S_rows, np.nan), y, st.hac_lags, st.min_obs)[0]
            t_dev[alive] = sign[alive] * t_dev_all[sel][alive]
    alive = nr[obs_status[nr] != "DROP"]

    # ---- step 3: tradability screen on DEV rows (IS + OBS) of every step-2 survivor; the backtest readings
    # are kept and reused by the shortlist
    sharpe_gross_dev = np.full(n_tested, np.nan)
    sharpe_net_dev = np.full(n_tested, np.nan)
    trade_status = np.array([""] * n_tested, dtype=object)
    bt_of: dict = {}
    for i in alive:
        j = sel[i]
        dev = is_m[:, j] | obs_m[:, j]
        bt = backtest(S_rows[dev, j], y[dev], sig[dev], float(sign[i]), cfg.target.cost_per_side,
                      cfg.position.buffer)
        bt_of[ids[i]] = bt
        sharpe_gross_dev[i], sharpe_net_dev[i] = bt["sharpe_gross"], bt["sharpe_net"]
        trade_status[i] = "KEEP" if (np.isfinite(bt["sharpe_net"]) and bt["sharpe_net"] >= vd.min_net_sharpe) \
            else "DROP"

    is_start, is_end = _edge_dates(is_m[:, sel], rows, cal)
    order = np.argsort(-np.abs(z), kind="stable")
    cand = pd.DataFrame({
        "id": ids, "series": cm.loc[ids, "series"].to_numpy(), "transform": cm.loc[ids, "transform"].to_numpy(),
        "group": cm.loc[ids, "group"].to_numpy(), "kind": cm.loc[ids, "kind"].to_numpy(),
        "freq": cm.loc[ids, "freq"].to_numpy(), "n_is": n_is, "is_start": is_start, "is_end": is_end,
        "n_obs": obs_m[:, sel].sum(axis=0), "n_oos": oos_m[:, sel].sum(axis=0), "t": t, "null_sd": null,
        "z": z, "ic": ic, "sign": sign, "p": p, "verdict": verdict, "ic_obs_signed": ic_obs_signed,
        "retention": retention, "obs_status": obs_status, "t_dev": t_dev, "sharpe_gross_dev": sharpe_gross_dev,
        "sharpe_net_dev": sharpe_net_dev, "trade_status": trade_status, "t_h1": t_h1, "t_h2": t_h2,
        "t_xlag": t_xlag, "flags": [";".join(f) for f in flags], "rep": "",
    }).iloc[order].reset_index(drop=True)
    flag_of = {cid: f for cid, f in zip(ids, flags)}

    # ---- step 4: de-redundancy of the tradable survivors on DEV rows, best t_dev first
    survivors = cand[cand["obs_status"].isin(["KEEP", "INSUFFICIENT"])]
    surv_ids = survivors.sort_values("t_dev", ascending=False, na_position="last", kind="stable")["id"].tolist()
    trade_ok = dict(zip(cand["id"], cand["trade_status"] == "KEEP"))
    S_dev_df = pd.DataFrame(np.where(is_m | obs_m, S_rows, np.nan), columns=S.columns)
    rep_map = dedupe_tradable(surv_ids, trade_ok, S_dev_df, vd.dedupe_corr)
    cand["rep"] = [rep_map.get(i, "") for i in cand["id"]]
    short_ids = list(dict.fromkeys(rep_map.values()))
    cluster = pd.Series(list(rep_map.values()), dtype=object).value_counts()

    # ---- shortlist readings, DEV rows only
    col_of = {c: i for i, c in enumerate(S.columns)}
    cidx = cand.set_index("id", drop=False)
    desc = panel.meta["description"]
    years = cal[rows].year.to_numpy()
    rows_out = []
    for cid in short_ids:
        r = cidx.loc[cid]
        j = col_of[cid]
        sgn = float(r["sign"])
        dev = is_m[:, j] | obs_m[:, j]
        s_dev = np.where(dev, S_rows[:, j], np.nan)
        bt = bt_of[cid]
        icd = ic_decay(s_dev, y, int(e2[j]), sgn)
        acs = signal_autocorr(s_dev)
        hit, n_years = yearly_hit(s_dev, y, years, sgn)
        f = list(flag_of[cid])
        if n_years >= 3 and hit < vd.year_hit_min:
            f.append("unstable")
        flag_of[cid] = f
        row = {**{c: r[c] for c in SHORT_CAND_COLUMNS}, "cluster_size": int(cluster[cid]), **bt,
               **dict(zip(IC_COLUMNS, icd)), **dict(zip(AC_COLUMNS, acs)),
               "signal_half_life": half_life(acs[0]), "year_hit": hit, "n_years": n_years,
               "description": desc.get(r["series"], "")}
        row["flags"] = ";".join(f)
        rows_out.append(row)
    short = pd.DataFrame(rows_out, columns=SHORT_COLUMNS)
    cand["flags"] = [";".join(flag_of[i]) for i in cand["id"]]

    cand.to_csv(run_dir / "candidates.csv", index=False)
    short.to_csv(run_dir / "shortlist.csv", index=False)

    funnel = {"tested": n_tested, "is_pass": int((verdict == "PASS").sum()),
              "is_weak": int((verdict == "WEAK_PASS").sum()),
              "obs_keep": int((obs_status == "KEEP").sum()),
              "obs_insufficient": int((obs_status == "INSUFFICIENT").sum()),
              "obs_drop": int((obs_status == "DROP").sum()),
              "trade_keep": int((trade_status == "KEEP").sum()), "trade_drop": int((trade_status == "DROP").sum()),
              "shortlist": len(short_ids)}
    run = {
        "target": cfg.target.name,
        "config": cfg.to_dict(),
        "panel_info": panel.info,
        "dates": {"sample_start": str(cal[0].date()), "sample_end": str(cal[-1].date())},
        "n_rows": len(cal),
        "n_series": int(panel.X.shape[1]), "n_candidates": int(N),
        "n_tested": n_tested, "n_skipped": n_skipped,
        "n_is_summary": {"min": int(n_is.min()), "median": float(np.median(n_is)), "max": int(n_is.max())},
        "funnel": funnel,
        "shortlist": short_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    run = _jsonable(run)
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    write_report(run_dir)
    print(f"[mine] funnel={funnel} -> {run_dir}")
    return run
