"""Walk-forward ("rolling") mining: validate a SELECTION RULE instead of a fixed set of factors.

At each re-mining date D_k the candidates are tested on a trailing window only (the same statistic as ``mine``
step 1: HAC t, circular-shift null sd, z, window net Sharpe), the held set is updated with entry / stay hysteresis,
de-duplicated and weighted, and held until D_{k+1}. Timing is strict T+1: the selection of D_k is computed after
the close of D_k, so its mining rows end at D_k - 1 row (row t needs the return of t+1) and its holding rows start
at D_k + 1 row. The target's last ``holdout_years`` calendar years are sealed: the panels are cut at the cutoff C
before anything is computed, so the holdout is never read.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RollingConfig, load_rolling_config, load_run_config
from .databank import Panel
from .engine import simulate
from .pipeline import _corr, _jsonable, _sharpe, backtest, dedupe, positions, prepare
from .stats import hac_t, null_sd

LAST_ROWS = 5            # a candidate needs a finite value on one of the last LAST_ROWS mining rows
PREFILTER_T = 0.7        # optional speed-up: null_sd only for |t| >= PREFILTER_T ...
PREFILTER_MIN_NULL = 0.75  # ... valid while every computed null_sd >= this (|t| < 0.7 / 0.75 < 1 = stay_z)
SAME_SECTIONS = ("target", "signals", "stats", "verdict", "position")
GRID_FIELDS = ("window_years", "step_months")


# --------------------------------------------------------------------- data
def spec_name(window_years: int, step_months: int) -> str:
    return f"{window_years}y{step_months}m"


def load_sources(run_dirs) -> list:
    """[(name, cfg, panel)] of finished ``mine`` runs; raise unless all share target series and settings."""
    out = [(Path(d).name, load_run_config(d), Panel.load(Path(d) / "panel")) for d in run_dirs]
    names = [n for n, _, _ in out]
    if not out or len(set(names)) != len(names):
        raise ValueError(f"need at least one run and unique run directory names, got {names}")
    n0, c0, p0 = out[0]
    for n, c, p in out[1:]:
        for sec in SAME_SECTIONS:
            if asdict(getattr(c, sec)) != asdict(getattr(c0, sec)):
                raise ValueError(f"[{sec}] of {n} differs from {n0}: pooled sources must share it")
        if not p.target.equals(p0.target):
            raise ValueError(f"target series of {n} differs from {n0}")
    return out


def cutoff_date(cal: pd.DatetimeIndex, holdout_years: int) -> pd.Timestamp:
    """C = last target date minus ``holdout_years`` calendar years (only the date is used, never a value)."""
    return cal[-1] - pd.DateOffset(years=holdout_years) if holdout_years else cal[-1]


def build_pool(sources, last_date) -> dict:
    """Pool the candidates of all sources on the common calendar, every panel cut at ``last_date`` BEFORE
    ``prepare`` (so nothing after it, e.g. the sealed holdout, is read). ``S`` is (rows, N) with unique ids
    ``<source>/<candidate id>``; rows are the calendar positions where y_fwd is finite."""
    cfg = sources[0][1]
    Ss, metas, ref = [], [], None
    for name, c, p in sources:
        p = Panel(target=p.target.loc[:last_date], X=p.X.loc[:last_date], meta=p.meta, excluded=p.excluded,
                  info=p.info)
        prep = prepare(p, c)
        ref = ref or prep
        if not (np.array_equal(prep["rows"], ref["rows"]) and
                np.array_equal(prep["y_fwd"].to_numpy(), ref["y_fwd"].to_numpy(), equal_nan=True)):
            raise ValueError(f"target / calendar of {name} differs from the first source")
        cm = prep["cand_meta"][["series", "transform", "group"]].copy()
        cm["source"] = name
        cm.index = [f"{name}/{i}" for i in cm.index]
        Ss.append(prep["S"].to_numpy(dtype=float)[prep["rows"]])
        metas.append(cm)
    rows = ref["rows"]
    meta = pd.concat(metas)
    return {"S": np.hstack(Ss), "ids": np.array(meta.index), "meta": meta, "cfg": cfg,
            "y": ref["y_fwd"].to_numpy(dtype=float)[rows], "sig": ref["sigma"].to_numpy(dtype=float)[rows],
            "dates": ref["y_fwd"].index[rows]}


def fingerprints(run_dirs) -> list:
    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    out = []
    for d in run_dirs:
        d = Path(d)
        info = json.loads((d / "panel" / "info.json").read_text(encoding="utf-8"))
        out.append({"run": d.name, "panel_pkl_sha256": sha(d / "panel" / "panel.pkl"),
                    "config_sha256": sha(d / "config.toml"), "db_version": info.get("db_version"),
                    "sample_end": info.get("sample_end")})
    return out


# ------------------------------------------------------------------- dates
def remine_rows(dates: pd.DatetimeIndex, first_date, step_months: int, last_row: int) -> list[int]:
    """Row index of each re-mining date: the first row on/after first_date + k*step_months (k = 0, 1, ...),
    kept while it is before ``last_row`` (it needs holding rows)."""
    out, first, k = [], pd.Timestamp(first_date), 0
    while True:
        i = int(dates.searchsorted(first + pd.DateOffset(months=k * step_months), side="left"))
        if i >= last_row:
            return out
        if not out or i > out[-1]:
            out.append(i)
        k += 1


def mining_rows(dates: pd.DatetimeIndex, i: int, window_years: int) -> tuple[int, int]:
    """(lo, hi) inclusive row range mined at re-mining row i: date >= D - W years and row <= i - 1."""
    return int(dates.searchsorted(dates[i] - pd.DateOffset(years=window_years), side="left")), i - 1


# ------------------------------------------------------------- window stats
def entry_threshold(rcfg: RollingConfig, vd) -> float:
    """|z| a NEW factor needs: ``rolling.entry_z``, by default the run configs' ``verdict.z_weak``."""
    return vd.z_weak if rcfg.entry_z is None else rcfg.entry_z


def window_stats(pool, lo, hi, rcfg: RollingConfig, prefilter=False) -> dict:
    """``mine`` step 1 on mining rows lo..hi: eligibility, t, ic, null sd, z, sign and the window net Sharpe of
    every eligible candidate with |z| >= min(z_weak, entry threshold). Arrays over the pool's N columns; NaN where not available."""
    cfg = pool["cfg"]
    st, vd = cfg.stats, cfg.verdict
    Sw, yw, sg = pool["S"][lo:hi + 1], pool["y"][lo:hi + 1], pool["sig"][lo:hi + 1]
    T, N = Sw.shape
    fin = np.isfinite(Sw)
    n = fin.sum(axis=0)
    elig = (n >= rcfg.coverage * T) & (n >= st.min_obs) & fin[-LAST_ROWS:].any(axis=0)
    out = {k: np.full(N, np.nan) for k in ("t", "ic", "null_sd", "z", "sign", "sharpe_net")}
    out["eligible"] = elig
    cols = np.flatnonzero(elig)
    if len(cols):
        t, ic, _ = hac_t(Sw[:, cols], yw, st.hac_lags, st.min_obs)
        out["t"][cols], out["ic"][cols] = t, ic
        use = np.isfinite(t) & ((np.abs(t) >= PREFILTER_T) if prefilter else True)
        if use.any():
            sd = null_sd(Sw[:, cols[use]], yw, st.hac_lags, st.min_obs, st.n_boot, st.min_shift, st.seed)
            if prefilter and (sd < PREFILTER_MIN_NULL).any():
                raise ValueError(f"null_sd < {PREFILTER_MIN_NULL} (min {np.nanmin(sd):.3f}): the |t| >= "
                                 f"{PREFILTER_T} pre-filter is invalid, rerun without it")
            out["null_sd"][cols[use]] = sd
        ok = np.isfinite(out["t"]) & np.isfinite(out["null_sd"]) & (out["null_sd"] > 0)
        out["z"][ok] = out["t"][ok] / out["null_sd"][ok]
        out["sign"][ok] = np.sign(out["ic"][ok])
        # the net Sharpe is needed wherever the entry rule can pass: |z| >= entry threshold (z_weak by default)
        for j in np.flatnonzero(ok & (np.abs(out["z"]) >= min(vd.z_weak, entry_threshold(rcfg, vd)))):
            m = fin[:, j]
            out["sharpe_net"][j] = backtest(Sw[m, j], yw[m], sg[m], float(out["sign"][j]), cfg.target.cost_per_side,
                                            cfg.position.buffer)["sharpe_net"]
    return out


_POOL = {}


def _worker(args):
    lo, hi, rcfg, prefilter = args
    return (lo, hi), window_stats(_POOL["pool"], lo, hi, rcfg, prefilter)


def compute_stats(pool, windows, rcfg, prefilter=False, jobs=1, cache=None, log=print) -> dict:
    """{(lo, hi): window_stats} for the missing windows, in parallel over ``jobs`` forked processes (each
    window is independent, so the result does not depend on ``jobs``)."""
    cache = {} if cache is None else cache
    todo = [w for w in dict.fromkeys(windows) if w not in cache]
    if jobs > 1 and len(todo) > 1:
        _POOL["pool"] = pool
        engine_warmup()
        with mp.get_context("fork").Pool(jobs) as p:
            for k, (key, s) in enumerate(p.imap_unordered(_worker, [(lo, hi, rcfg, prefilter) for lo, hi in todo])):
                cache[key] = s
                log(f"[roll] window stats {k + 1}/{len(todo)}")
    else:
        for k, (lo, hi) in enumerate(todo):
            cache[(lo, hi)] = window_stats(pool, lo, hi, rcfg, prefilter)
            log(f"[roll] window stats {k + 1}/{len(todo)}")
    return cache


def engine_warmup():
    """Trigger the vectorbt import / JIT once, so that forked workers inherit it."""
    simulate(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([0.01, 0.01]), 0.0)


# ------------------------------------------------------------- selection
def group_weights(groups, cap) -> np.ndarray:
    """Weight of each factor: equal, with every group's total <= cap (groups over the cap are cut to it and the
    excess is shared equally per factor by the other groups, iterated). With fewer groups than 1/cap the cap
    cannot hold: equal weight per group, equal within a group. Sums to 1 (empty -> empty)."""
    if len(groups) == 0:
        return np.zeros(0)
    _, inv = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    G, n = len(cnt), len(inv)
    if G * cap < 1 - 1e-12:
        gw = np.full(G, 1.0 / G)
    else:
        gw, capped = cnt / n, np.zeros(G, dtype=bool)
        while True:
            over = ~capped & (gw > cap + 1e-12)
            free = ~(capped | over)
            if not over.any() or not free.any():
                break
            capped |= over
            gw[capped] = cap
            gw[free] = (1.0 - cap * capped.sum()) * cnt[free] / cnt[free].sum()
    return (gw / cnt)[inv]


def entry_mask(stat, rcfg: RollingConfig, vd) -> np.ndarray:
    """The raw entry rule of one date: |z| >= entry threshold and window net Sharpe > is_min_net_sharpe (before
    seasoning, de-duplication and incumbency)."""
    az = np.abs(stat["z"])
    return (np.isfinite(az) & (az >= entry_threshold(rcfg, vd)) & np.isfinite(stat["sharpe_net"]) &
            (stat["sharpe_net"] > vd.is_min_net_sharpe))


def seasoned_mask(stat, entry, history: list, confirm: int) -> np.ndarray:
    """Candidates whose entry rule passed at each of the previous ``confirm - 1`` consecutive re-mining dates with
    the sign they have now. ``history`` = the raw signed entry vectors (sign where the entry rule passed, else 0)
    of the earlier dates of the same spec, oldest first. Fewer than ``confirm - 1`` earlier dates: nobody."""
    if confirm <= 1:
        return np.ones(len(entry), dtype=bool)
    if len(history) < confirm - 1:
        return np.zeros(len(entry), dtype=bool)
    ok = entry.copy()
    for h in history[len(history) - (confirm - 1):]:
        ok &= (h == stat["sign"])          # h is 0 where the rule failed, sign is +-1 where it passed
    return ok


def select(stat, prev: dict, Sw, ids, rcfg: RollingConfig, vd, history: list | None = None) -> dict:
    """The held set at one re-mining date. ``prev`` = {column: sign} of the previous holding; ``Sw`` the mining
    rows of the pool; ``history`` the raw signed entry vectors of the earlier dates (seasoning, see
    ``seasoned_mask``; only used when ``rcfg.confirm_periods > 1``). Returns held {column: status "new" | "stay"}
    (cluster representatives only) and the bookkeeping of the step: exits (column -> reason, detail), counts."""
    z, sign, az = stat["z"], stat["sign"], np.abs(stat["z"])
    tested = np.isfinite(z)
    entry = entry_mask(stat, rcfg, vd)
    seasoned = seasoned_mask(stat, entry, history or [], rcfg.confirm_periods)
    exits, stay = {}, []
    for j, s in prev.items():
        if not tested[j]:
            exits[j] = ("stay_rule", "not_tested")
        elif sign[j] != s:
            exits[j] = ("stay_rule", "sign_flip")
        elif az[j] < rcfg.stay_z:
            exits[j] = ("stay_rule", "low_z")
        else:
            stay.append(j)
    # a failed incumbent is not a new entrant at this date: it may re-enter only at a LATER date, via the entry rule
    new = [j for j in np.flatnonzero(entry & seasoned) if j not in prev]
    n_unseasoned = sum(1 for j in np.flatnonzero(entry & ~seasoned) if j not in prev)
    by_z = lambda js: sorted(js, key=lambda j: (-az[j], j))
    order = by_z(stay) + by_z(new)
    held = {}
    if order:
        rep = dedupe([ids[j] for j in order], pd.DataFrame(Sw[:, order], columns=[ids[j] for j in order]),
                     vd.dedupe_corr)
        for j in order:
            if rep[ids[j]] == ids[j]:
                held[j] = "stay" if j in prev else "new"
            elif j in prev:
                exits[j] = ("dedupe", f"absorbed_by={rep[ids[j]]}")
    return {"held": held, "exits": exits, "n_tested": int(tested.sum()), "n_entry": int(entry.sum()),
            "n_stay": len(stay), "n_new_cand": len(new), "n_unseasoned": n_unseasoned}


# ---------------------------------------------------------------- one spec
def mean_hac_t(x, lags) -> float:
    """Newey-West t of the mean of x (same Bartlett weights as ``stats.hac_t``)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < lags + 2:
        return float("nan")
    u = x - x.mean()
    v = u @ u + 2.0 * sum((1.0 - l / (lags + 1.0)) * (u[l:] @ u[:-l]) for l in range(1, lags + 1))
    return float(x.mean() * n / np.sqrt(v)) if v > 0 else float("nan")


def roll_spec(pool, rcfg: RollingConfig, window_years, step_months, cache, prefilter=False) -> dict:
    """One walk-forward spec; ``cache`` holds the window stats (missing windows are computed)."""
    cfg = pool["cfg"]
    dates, ids, meta, S, y, sig = pool["dates"], pool["ids"], pool["meta"], pool["S"], pool["y"], pool["sig"]
    last = len(dates) - 1                       # last evaluated row (the panels were cut at the cutoff)
    Ds = remine_rows(dates, rcfg.first_date, step_months, last)
    if not Ds:
        raise ValueError("no re-mining date before the cutoff")
    f = rcfg.confirm_periods - 1                # nothing can be held before period f (seasoning needs f earlier dates)
    if f >= len(Ds):
        raise ValueError(f"confirm_periods={rcfg.confirm_periods} needs more than {len(Ds)} re-mining dates")
    win = [mining_rows(dates, i, window_years) for i in Ds]
    compute_stats(pool, win, rcfg, prefilter, cache=cache)
    pos_cache: dict = {}
    pos_of = lambda j: pos_cache.setdefault(j, positions(S[:, j], 1.0, cfg.position.buffer))
    spec = spec_name(window_years, step_months)
    periods, members, exits, comp = [], [], [], []
    prev: dict = {}
    spell: dict = {}
    history: list = []                          # raw signed entry vectors of the earlier dates (seasoning)
    for k, i in enumerate(Ds):
        lo, hi = win[k]
        end = min(Ds[k + 1], last) if k + 1 < len(Ds) else last
        h = slice(i + 1, end + 1)
        stat = cache[(lo, hi)]
        res = select(stat, prev, S[lo:hi + 1], ids, rcfg, cfg.verdict, history)
        history.append(np.where(entry_mask(stat, rcfg, cfg.verdict), stat["sign"], 0.0))
        del history[:-max(f, 1)]                # only the last confirm_periods - 1 dates are ever read
        held = res["held"]
        js = sorted(held, key=lambda j: (-abs(stat["z"][j]), j))
        sgn = np.array([stat["sign"][j] for j in js])
        w = group_weights(meta["group"].to_numpy()[js], rcfg.group_cap)
        P = np.zeros(end - i)
        for wj, sj, j in zip(w, sgn, js):
            P += wj * sj * pos_of(j)[h]
        comp.append(P)
        ic = np.array([_corr(S[h, j], y[h]) for j in js])
        sic = sgn * ic
        for j, wj, sj, a in zip(js, w, sgn, sic):
            m = meta.iloc[j]
            members.append({"spec": spec, "period": k, "date": dates[i], "id": ids[j], "source": m["source"],
                            "series": m["series"], "transform": m["transform"], "group": m["group"], "sign": int(sj),
                            "z": stat["z"][j], "ic": stat["ic"][j], "sharpe_net_window": stat["sharpe_net"][j],
                            "status": held[j], "weight": wj, "ic_signed_next": a})
            spell[j] = spell.get(j, 0) + 1 if held[j] == "stay" else 1
        for j, (why, detail) in res["exits"].items():
            exits.append({"spec": spec, "period": k, "date": dates[i], "id": ids[j], "reason": why, "detail": detail,
                          "sign_prev": prev[j], "sign_now": stat["sign"][j], "z": stat["z"][j],
                          "periods_held": spell.pop(j, np.nan)})
        gw = pd.Series(w).groupby(meta["group"].to_numpy()[js]).sum() if js else pd.Series(dtype=float)
        periods.append({"spec": spec, "period": k, "date": dates[i], "hold_start": dates[i + 1], "hold_end": dates[end],
                        "n_hold_rows": end - i, "n_tested": res["n_tested"], "n_entry_pass": res["n_entry"],
                        "n_unseasoned": res["n_unseasoned"],
                        "n_incumbents": len(prev), "n_stayed": res["n_stay"], "n_exited": len(res["exits"]),
                        "n_new": sum(v == "new" for v in held.values()), "n_held": len(js), "n_groups": len(gw),
                        "group_weights": json.dumps({g: round(float(v), 4) for g, v in gw.items()}),
                        "mean_signed_ic": float(np.nanmean(sic)) if np.isfinite(sic).any() else np.nan,
                        "share_ic_pos": float(np.mean(sic[np.isfinite(sic)] > 0)) if np.isfinite(sic).any() else np.nan})
        prev = {j: int(stat["sign"][j]) for j in js}
    # composite P&L: ONE engine run on the concatenated position, so re-mining switches are charged netted
    first, hold_rows = Ds[0] + 1, np.arange(Ds[0] + 1, last + 1)
    pos = np.concatenate(comp)
    if len(pos) != len(hold_rows):
        raise RuntimeError("holding rows do not partition (D_0, C]")
    gross, net = simulate(pos, y[hold_rows], sig[hold_rows], cfg.target.cost_per_side)
    pnl = pd.DataFrame({"date": dates[hold_rows], "spec": spec, "position": pos, "gross": gross, "net": net})
    per = pd.DataFrame(periods)
    off = 0
    sh = []
    for k, n_ in enumerate(per["n_hold_rows"]):
        sh.append(_sharpe(net[off:off + n_]) if k >= f else np.nan)      # NaN: structurally flat seasoning period
        off += n_
    per["net_sharpe"] = sh
    per = per.drop(columns="n_hold_rows")
    mem, ex = pd.DataFrame(members), pd.DataFrame(exits)
    # evaluated span: from the first date at which holding is possible, D_f + 1 (f = confirm_periods - 1). The
    # seasoning periods 0..f-1 are flat by construction (all-zero position, no costs): they are left out of pnl.csv
    # and of every summary statistic instead of being averaged in as zeros.
    n0 = int(sum(len(c) for c in comp[:f]))
    pnl, hold_rows = pnl.iloc[n0:].reset_index(drop=True), hold_rows[n0:]
    summ = summarise(pool, per.iloc[f:], mem, pnl, hold_rows, y)
    summ.update({"confirm_periods": rcfg.confirm_periods, "eval_start_period": f,
                 "eval_start_date": str(dates[hold_rows[0]].date()), "n_seasoning_periods_excluded": f})
    return {"spec": spec, "window_years": window_years, "step_months": step_months, "periods": per, "members": mem,
            "exits": ex, "pnl": pnl, "summary": summ}


def spells(mem: pd.DataFrame) -> list[tuple[int, bool]]:
    """(length in periods, still held at the end) of every holding spell of every factor."""
    out = []
    if mem.empty:
        return out
    last = mem["period"].max()
    for _, g in mem.groupby("id"):
        p = np.sort(g["period"].to_numpy())
        cut = np.flatnonzero(np.diff(p) > 1) + 1
        for seg in np.split(p, cut):
            out.append((len(seg), bool(seg[-1] == last)))
    return out


def summarise(pool, per, mem, pnl, hold_rows, y) -> dict:
    cfg = pool["cfg"]
    net, gross, pos = pnl["net"].to_numpy(), pnl["gross"].to_numpy(), pnl["position"].to_numpy()
    _, long_net = simulate(np.ones(len(hold_rows)), y[hold_rows], pool["sig"][hold_rows], cfg.target.cost_per_side)
    sd = net.std(ddof=1)
    nav = 1.0 + np.cumsum(net * (0.10 / (sd * np.sqrt(252.0)))) if sd > 0 else np.ones(len(net))
    nav = np.r_[1.0, nav]
    mdd = float(np.max(1.0 - nav / np.maximum.accumulate(nav)))
    yr = pnl["date"].dt.year.to_numpy()
    by_year = {int(a): _sharpe(net[yr == a]) for a in np.unique(yr)}
    sic = per["mean_signed_ic"].dropna().to_numpy()
    sp = spells(mem)
    life = np.array([a for a, _ in sp], dtype=float)
    turn = (per["n_new"] / per["n_held"].where(per["n_held"] > 0)).iloc[1:]
    gw: dict = {}
    for s in per["group_weights"]:
        for g, v in json.loads(s).items():
            gw[g] = gw.get(g, 0.0) + v
    n_held_periods = int((per["n_held"] > 0).sum())
    return {
        "date_from": str(pnl["date"].iloc[0].date()), "date_to": str(pnl["date"].iloc[-1].date()),
        "n_rows": len(net), "n_periods": len(per), "n_flat_periods": int((per["n_held"] == 0).sum()),
        "avg_held": float(per["n_held"].mean()), "name_turnover": float(turn.mean()) if turn.notna().any() else np.nan,
        "lifespan_mean": float(life.mean()) if len(life) else np.nan,
        "lifespan_median": float(np.median(life)) if len(life) else np.nan,
        "lifespan_counts": {int(a): int((life == a).sum()) for a in np.unique(life)},
        "n_spells": len(sp), "n_spells_censored": int(sum(c for _, c in sp)),
        "net_sharpe": _sharpe(net), "gross_sharpe": _sharpe(gross),
        "hac_t_net": mean_hac_t(net, cfg.stats.hac_lags),
        "max_drawdown_10vol": mdd, "cum_pnl_10vol": float(nav[-1] - 1.0),
        "avg_abs_turnover": float(np.mean(np.abs(np.diff(pos)))),
        "corr_long_only": _corr(net, long_net), "long_only_sharpe": _sharpe(y[hold_rows]),
        "net_sharpe_by_year": by_year,
        "mean_signed_ic": float(sic.mean()) if len(sic) else np.nan,
        "t_signed_ic": float(sic.mean() / (sic.std(ddof=1) / np.sqrt(len(sic)))) if len(sic) > 2 and sic.std() > 0
        else np.nan,
        "share_periods_ic_pos": float((sic > 0).mean()) if len(sic) else np.nan,
        "avg_group_weight": {g: v / max(n_held_periods, 1) for g, v in sorted(gw.items())},
    }


# ------------------------------------------------------------------ report
def _md(df: pd.DataFrame) -> str:
    f = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def write_roll_report(out: Path, info: dict, res: list) -> None:
    """Short Chinese ``report.md``: parameters, one summary row per spec, the primary spec's yearly Sharpe and
    group composition."""
    P = info["parameters"]
    rows = []
    for r in res:
        s = r["summary"]
        rows.append({"spec": r["spec"], "区间": f"{s['date_from']}~{s['date_to']}", "期数": s["n_periods"],
                     "平均持有数": s["avg_held"], "名称换手": s["name_turnover"], "平均寿命(期)": s["lifespan_mean"],
                     "净Sharpe": s["net_sharpe"], "毛Sharpe": s["gross_sharpe"], "HAC t": s["hac_t_net"],
                     "回撤@10%vol": s["max_drawdown_10vol"], "与多头相关": s["corr_long_only"],
                     "多头Sharpe": s["long_only_sharpe"], "下期带符号IC": s["mean_signed_ic"],
                     "IC t(跨期)": s["t_signed_ic"], "IC为正期占比": s["share_periods_ic_pos"]})
    prim = next(r for r in res if r["spec"] == info["primary"])
    ps = prim["summary"]
    yr = pd.DataFrame({"年": list(ps["net_sharpe_by_year"]), "净Sharpe": list(ps["net_sharpe_by_year"].values())})
    gw = pd.DataFrame({"group": list(ps["avg_group_weight"]), "平均权重": list(ps["avg_group_weight"].values())})
    pf = info["prefilter_t"]
    seasoning = (f"确认期(seasoning): 前 {P['confirm_periods'] - 1} 个重挖期不可能持仓, 汇总统计从 {ps['eval_start_date']} 起算"
                 f"(主规格; 这些空仓期不计入 Sharpe / HAC t / 回撤 / 期均值, 也不写入 pnl.csv)。\n"
                 if P["confirm_periods"] > 1 else "")
    txt = f"""# 滚动挖掘报告

参数: 窗口 {P['window_years']} 年 / 每 {P['step_months']} 个月重挖 / 首个重挖日 {P['first_date']} / 封存最后 {P['holdout_years']} 年
(评估截止 {info['cutoff']}) / stay_z={P['stay_z']} / group_cap={P['group_cap']} / coverage={P['coverage']} /
entry_z={'z_weak(取自 run 配置)' if P['entry_z'] is None else P['entry_z']} / confirm_periods={P['confirm_periods']}。
{seasoning}候选 {info['n_candidates']} 个; 主规格 `{info['primary']}`; 网格里的规格全部列出, 不挑选。
null_sd 预过滤 (|t| >= {pf['min_t']}): {'启用' if pf['used'] else '未启用'}。

## 各规格汇总

{_md(pd.DataFrame(rows))}

回撤与累计盈亏按"事后缩放到年化 10% 波动率"的单利净值计算, 仅供展示, 不是可交易的杠杆。

## 主规格 {info['primary']} 分年净 Sharpe

{_md(yr)}

## 主规格平均 group 权重 (只对有持仓的期求平均)

{_md(gw)}

明细见 `periods.csv`、`members.csv`、`exits.csv`、`pnl.csv`、`roll.json`。
"""
    (out / "report.md").write_text(txt, encoding="utf-8")


# --------------------------------------------------------------------- run
def specs_of(rcfg: RollingConfig, grid: bool) -> list[tuple[int, int]]:
    out = [(rcfg.window_years, rcfg.step_months)]
    if grid:
        ws = rcfg.grid.get("window_years", [rcfg.window_years])
        ms = rcfg.grid.get("step_months", [rcfg.step_months])
        out += [(w, m) for w in ws for m in ms]
    return list(dict.fromkeys(out))


def roll(rcfg: RollingConfig, run_dirs, out_dir, grid=False, truncate=None, prefilter=False, jobs=1,
         log=print) -> dict:
    """Run the primary spec (and the grid) on the pooled ``run_dirs``; write the outputs to ``out_dir``."""
    import time

    t0 = time.time()
    sources = load_sources(run_dirs)
    cal = sources[0][2].target.index
    if truncate is not None:
        cal = cal[cal <= pd.Timestamp(truncate)]
    C = cutoff_date(cal, rcfg.holdout_years)
    pool = build_pool(sources, C)
    log(f"[roll] pool: {pool['S'].shape[1]} candidates, {len(pool['dates'])} rows up to {pool['dates'][-1].date()} "
        f"(cutoff {C.date()}) in {time.time() - t0:.0f}s")
    specs = specs_of(rcfg, grid)
    # windows of every spec first, so that they are computed once and in parallel
    cache: dict = {}
    wins = [mining_rows(pool["dates"], i, w) for w, m in specs for i in remine_rows(pool["dates"], rcfg.first_date,
                                                                                    m, len(pool["dates"]) - 1)]
    compute_stats(pool, wins, rcfg, prefilter, jobs, cache, log)
    res = [roll_spec(pool, rcfg, w, m, cache, prefilter) for w, m in specs]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("periods", "members", "exits", "pnl"):
        pd.concat([r[name] for r in res], ignore_index=True).to_csv(out / f"{name}.csv", index=False)
    info = {"parameters": asdict(rcfg), "primary": spec_name(rcfg.window_years, rcfg.step_months),
            "specs": [r["spec"] for r in res], "cutoff": str(C.date()), "truncate": truncate,
            "prefilter_t": {"used": bool(prefilter), "min_t": PREFILTER_T, "min_null_sd": PREFILTER_MIN_NULL},
            "sources": fingerprints(run_dirs), "n_candidates": int(pool["S"].shape[1]),
            "run_config": pool["cfg"].to_dict(), "seconds": round(time.time() - t0, 1),
            "summary": {r["spec"]: r["summary"] for r in res}}
    (out / "roll.json").write_text(json.dumps(_jsonable(info), indent=2, ensure_ascii=False))
    write_roll_report(out, info, res)
    log(f"[roll] wrote {out} in {time.time() - t0:.0f}s")
    return info


def roll_cli(config, run_dirs, out, grid=False, holdout_years=None, truncate=None, prefilter=False, jobs=1):
    rcfg = load_rolling_config(config)
    if holdout_years is not None:
        rcfg.holdout_years = holdout_years
    return roll(rcfg, run_dirs, out, grid=grid, truncate=truncate, prefilter=prefilter, jobs=jobs)
