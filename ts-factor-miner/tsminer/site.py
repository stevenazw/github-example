"""Factor tracking dashboard: per-asset data (net-value paths, formulas) and a static site generator.

Isolation: a run directory without ``confirm.json`` has not opened its OOS. In that case no OOS row is read,
computed or written: the net value of every factor stops at the end of its observation period.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_run_config
from .databank import Panel
from .engine import simulate
from .confirm import OOS_MIN_OBS, oos_verdict
from .pipeline import _corr, _jsonable, dedupe, positions, prepare
from .stats import hac_t, norm_sf
from .transforms import chg_std_window

TARGET_VOL = 0.10          # in-sample annualised vol the net value is scaled to
MAX_POINTS = 1500
SEG_NAMES = ("样本内", "观察期", "样本外")
VERDICT_ORDER = {"CONFIRMED": 0, "CONSISTENT": 1, "INSUFFICIENT": 2, "FAILED": 3}
FLAG_TEXT = {
    "xlag_weak": "对数据可得时点敏感",
    "half_flip": "样本内前后两半方向相反",
    "unstable": "逐年方向不稳定",
    "obs_short": "观察期样本不足",
}
RESERVED_PREFIXES = ("settings", "sidebar", "customize", "upgrade", "downloads", "comment-")


# --------------------------------------------------------------- text helpers (pure)
def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    if not s:
        raise ValueError(f"target name {name!r} has no ASCII letters or digits to build a slug from")
    if s == "overview" or s.startswith(RESERVED_PREFIXES):
        raise ValueError(f"slug {s!r} (from target {name!r}) collides with a reserved page name")
    return s


def direction_text(sign, target_name) -> str:
    return f"因子上升 → {'做多' if sign > 0 else '做空'} {target_name}"


def transform_label(transform: str) -> str:
    prefix, _, num = str(transform).partition("_")
    if prefix == "chg":
        return f"{num} 日变化"
    if prefix == "z":
        return f"{num} 日偏离度"
    return str(transform)


def flag_texts(flags) -> list[str]:
    if flags is None or (isinstance(flags, float) and math.isnan(flags)):
        return []
    return [FLAG_TEXT.get(f, f) for f in str(flags).split(";") if f]


def _g(x) -> str:
    return f"{float(x):g}"


def formula_text(meta_row, transform: str, cfg, sign: float = 1.0) -> list[str]:
    """Plain-text formula lines that mirror ``transforms.make_signal`` and ``pipeline.positions``."""
    ticker, field, kind = str(meta_row["ticker"]), str(meta_row["field"]), str(meta_row["kind"])
    lag = int(meta_row["lag"])
    avail = f"（滞后 {lag} 日可得）"
    if ticker.startswith("chain:"):
        first = f"x[t] = {ticker.split(':', 1)[1]} 期货连续合约收益指数{avail}"
    elif ticker.startswith("carry:"):
        first = f"x[t] = {ticker.split(':', 1)[1]} 跨月价差 ÷ 主力合约价格{avail}"
    elif ticker.startswith("var:"):
        first = f"x[t] = {field}"                       # the variable's expression, verbatim
    elif ticker.startswith("spread:"):
        legs = field.split("-")
        pair = f"{legs[0].strip()} − {legs[1].strip()}" if len(legs) == 2 else field
        first = f"x[t] = {pair}{avail}"
    else:
        first = f"x[t] = {ticker} 的 {field}{avail}"
    lines = [first]
    prefix, _, num = str(transform).partition("_")
    clip, win = _g(cfg.signals.clip), int(cfg.signals.std_window)
    if prefix == "chg":
        win = chg_std_window(int(num), win)
        if kind == "price":
            lines.append(f"r[t] = ln(x[t] / x[t−{num}])")
            lines.append(f"s[t] = clip(r[t] / std_{win}(r), ±{clip})")
        else:
            lines.append(f"d[t] = x[t] − x[t−{num}]")
            lines.append(f"s[t] = clip(d[t] / std_{win}(d), ±{clip})")
    elif prefix == "z":
        lines.append(f"s[t] = clip((x[t] − mean_{num}(x)) / std_{num}(x), ±{clip})")
    else:
        lines.append(f"s[t] = {transform}(x)")
    sg = "+1" if sign > 0 else "−1"
    d = int(getattr(cfg.position, "execution_delay", 0) or 0)     # runs from before execution_delay: fill at t
    s_t = f"s[t−{d}]" if d > 0 else "s[t]"
    lines.append(f"pos[t] = {sg} × {s_t} ÷ σ[t]")
    note = (f"σ[t]：标的日收益的 EWMA 波动率（半衰期 {int(cfg.signals.vol_halflife)} 日）；"
            f"目标仓位变化超过缓冲带 {_g(cfg.position.buffer)} 才调仓")
    if d > 0:
        note += f"；信号形成后第 {d} 个收盘成交"
    lines.append(note)
    return lines


# --------------------------------------------------------------- net value (pure)
def _seg_stats(s, y, ret, nav, dates, mask, sign) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"start": None, "end": None, "n": 0, "ic": None, "sharpe": None, "cum_ret": None, "max_dd": None}
    r, v = ret[mask], nav[mask]
    sd = np.std(r, ddof=1) if n > 1 else float("nan")
    sharpe = float(np.mean(r) / sd * np.sqrt(252.0)) if np.isfinite(sd) and sd > 0 else float("nan")
    ic = sign * _corr(s[mask], y[mask])
    dd = float(np.max(np.maximum.accumulate(v) - v))
    d = dates[mask]
    return {"start": str(d[0]), "end": str(d[-1]), "n": n, "ic": float(ic), "sharpe": sharpe,
            "cum_ret": float(r.sum()), "max_dd": dd}


def factor_nav(s, y, sig, seg, dates, sign, buffer, cost, target_vol=TARGET_VOL):
    """Net value of one factor on the given rows (one continuous computation).

    ``s, y, sig``: signal, forward target and sigma on the valid rows; ``seg`` 0/1/2 per row; ``dates``: the
    realisation date of each row. Returns None when the in-sample std of net is zero / not finite.
    """
    s, y, sig = (np.asarray(a, dtype=float) for a in (s, y, sig))
    seg, dates = np.asarray(seg), np.asarray(dates)
    pos = positions(s, sign, buffer)
    _, net = simulate(pos, y, sig, cost)
    ins = seg == 0
    sd = float(np.std(net[ins], ddof=1)) if ins.sum() > 1 else float("nan")
    if not np.isfinite(sd) or sd <= 0:
        return None
    k = target_vol / np.sqrt(252.0) / sd
    ret = k * net
    nav = 1.0 + np.cumsum(ret)
    segs = []
    for i, name in enumerate(SEG_NAMES):
        st = _seg_stats(s, y, ret, nav, dates, seg == i, sign)
        segs.append({"seg": i, "name": name, "opened": True, **st})
    return {"ret": ret, "nav": nav, "k": float(k), "segments": segs}


def downsample_idx(seg, max_points=MAX_POINTS) -> np.ndarray:
    seg = np.asarray(seg)
    n = len(seg)
    if n == 0:
        return np.array([], dtype=int)
    step = max(1, math.ceil(n / max_points))
    keep = set(range(0, n, step)) | {n - 1}
    for k in np.unique(seg):
        w = np.flatnonzero(seg == k)
        keep |= {int(w[0]), int(w[-1])}
    return np.array(sorted(keep), dtype=int)


# --------------------------------------------------------------- per-asset data
STAGE_NAMES = {4: "去重后留下", 3: "去重时被合并", 2: "可交易筛选未通过", 1: "观察期未保留"}
IDENTITY_CORR = 0.999          # |corr| of two signals of the same transform on their common DEV dates: same factor
IDENTITY_MIN_OVERLAP = 250     # fewer common DEV dates than this: never the same factor
DEDUPE_MIN_OVERLAP = 100       # same as pipeline.dedupe


def _read_run(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _common_abs_corr(a: pd.Series, b: pd.Series, min_overlap: int) -> float:
    """|Pearson corr| of two date-indexed series on their common finite dates; NaN if fewer than min_overlap."""
    idx = a.index.intersection(b.index)
    if len(idx) < min_overlap:
        return float("nan")
    x, y = a.loc[idx].to_numpy(dtype=float), b.loc[idx].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_overlap:
        return float("nan")
    x, y = x[m] - x[m].mean(), y[m] - y[m].mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return abs(float((x * y).sum()) / den) if den > 0 else float("nan")


def is_same_factor(transform_a, dev_a: pd.Series, transform_b, dev_b: pd.Series,
                   thresh=IDENTITY_CORR, min_overlap=IDENTITY_MIN_OVERLAP) -> bool:
    """Same factor: same transform name and |corr| >= thresh of the DEV signals on their common dates (the sign
    of the correlation does not matter); fewer than ``min_overlap`` common dates: not the same."""
    if str(transform_a) != str(transform_b):
        return False
    c = _common_abs_corr(dev_a, dev_b, min_overlap)
    return bool(np.isfinite(c) and c >= thresh)


def group_same_factors(entries) -> list[list[int]]:
    """Transitive grouping (union-find) of entries (dicts with ``transform`` and ``dev``) by ``is_same_factor``.
    Returns groups of indices, each in ascending order, groups ordered by their first member."""
    parent = list(range(len(entries)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    by_t: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        by_t.setdefault(str(e["transform"]), []).append(i)
    for idx in by_t.values():
        for x, i in enumerate(idx):
            for j in idx[x + 1:]:
                if find(i) != find(j) and is_same_factor(entries[i]["transform"], entries[i]["dev"],
                                                         entries[j]["transform"], entries[j]["dev"]):
                    parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(entries)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def choose_keeper(members) -> int:
    """Index (into ``members``: dicts with ``opened``, ``run_idx``, ``z``, ``id``) of the entry that represents a
    group of identical factors: a run whose OOS is open first, then the newer run (later in registry order),
    then the larger |z|."""
    return max(range(len(members)), key=lambda k: (bool(members[k]["opened"]), members[k]["run_idx"],
                                                    abs(float(members[k]["z"])), str(members[k]["id"])))


def factor_stage(row, is_representative) -> int:
    """Furthest layer a factor reached: 1 the observation period dropped it, 2 the tradability screen dropped
    it, 3 merged into another representative at de-duplication, 4 kept. Runs from before the tradability
    screen have no ``trade_status``: that layer is then passed by everyone."""
    def get(k):
        v = row.get(k, "")
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
    if get("obs_status") == "DROP":
        return 1
    if get("trade_status") == "DROP":
        return 2
    return 4 if is_representative else 3


def stage_note(row, stage, criteria, rep_title=None) -> str:
    """One sentence naming the gate that stopped the factor, with the numbers (from the CSV row and the config
    of the run it was tested in)."""
    if stage == 1:
        r, lim = row.get("retention"), criteria.get("obs_min_retention")
        if r is None or not np.isfinite(float(r)):
            return f"观察期 IC 保留比例不足（低于 {_g(lim)}）"
        return (f"观察期方向相反（保留比例 {float(r):.2f}）" if float(r) < 0
                else f"观察期 IC 保留比例 {float(r):.2f}，低于 {_g(lim)}")
    if stage == 2:
        sh = row.get("sharpe_net_dev")
        shs = f"{float(sh):.2f}" if sh is not None and np.isfinite(float(sh)) else "无法计算"
        return f"扣成本后净 Sharpe {shs}，低于 {_g(criteria.get('min_net_sharpe'))}"
    if stage == 3:
        return f"与「{rep_title}」高度相关，去重时并入" if rep_title else "去重时并入其他代表"
    return ""


def stage_counts(factors) -> dict:
    """Layered counts of the factors that passed the in-sample bar (a layer includes everything beyond it)."""
    n = lambda k: sum(1 for f in factors if f["stage"] >= k)
    return {"n_is": n(1), "n_obs": n(2), "n_trade": n(3), "n_final": n(4)}


def pool_representatives(order_keys, dev_by_key, thresh) -> dict:
    """De-duplication over the pooled tradable factors (same greedy rule as ``pipeline.dedupe``): ``order_keys``
    best t_dev first; returns key -> representative key."""
    if not order_keys:
        return {}
    df = pd.DataFrame({k: dev_by_key[k] for k in order_keys})
    return dedupe(order_keys, df, thresh, DEDUPE_MIN_OVERLAP)


def _criteria_of(run: dict) -> dict:
    vd = run["config"]["verdict"]
    c = {"z_weak": vd["z_weak"], "obs_min_retention": vd["obs_min_retention"]}
    if "min_net_sharpe" in vd:                       # runs made before the tradability screen lack it
        c["min_net_sharpe"] = vd["min_net_sharpe"]
    if "is_min_net_sharpe" in vd:                    # runs made before the in-sample Sharpe bar lack it
        c["is_min_net_sharpe"] = vd["is_min_net_sharpe"]
    if "dedupe_corr" in vd:
        c["dedupe_corr"] = vd["dedupe_corr"]
    ps = run["config"].get("position", {})
    if "execution_delay" in ps:                      # runs made before the execution delay lack it: nothing shown
        c["execution_delay"] = ps["execution_delay"]
    return c


def _load_run(run_dir, idx: int) -> dict:
    """Everything needed from one run directory. ``confirm.*`` is not read; whether the OOS is open is decided by
    the existence of ``confirm.json`` alone and only then are OOS rows used."""
    d = Path(run_dir)
    run = _read_run(d)
    cfg = load_run_config(d)                         # a run without execution_delay in run.json counts as 0
    panel = Panel.load(d / "panel")
    opened = (d / "confirm.json").exists()
    cand = pd.read_csv(d / "candidates.csv")
    for c in ("flags", "obs_status", "trade_status", "rep"):
        if c in cand.columns:
            cand[c] = cand[c].fillna("").astype(str)
    cand = cand[cand["verdict"] != "REJECT"].reset_index(drop=True)          # everything that passed the IS bar
    prep = prepare(panel, cfg)
    rows = prep["rows"]
    S_np = prep["S"].to_numpy(dtype=float)[rows]
    return {"dir": d, "name": d.resolve().name, "idx": idx, "run": run, "cfg": cfg, "panel": panel,
            "opened": bool(opened), "cand": cand, "prep": prep, "S_np": S_np,
            "y": prep["y_fwd"].to_numpy(dtype=float)[rows], "sig": prep["sigma"].to_numpy(dtype=float)[rows],
            "col_of": {c: i for i, c in enumerate(prep["S"].columns)},
            "cal": panel.target.index, "criteria": _criteria_of(run)}


def _display(ctx, r):
    meta = ctx["panel"].meta
    desc = meta["description"].get(r["series"], "") if "description" in meta.columns else ""
    desc = desc if isinstance(desc, str) and desc else str(r["series"])
    return desc, f"{desc} · {transform_label(r['transform'])}"


def _dev_series(ctx, j) -> pd.Series:
    prep = ctx["prep"]
    idx = np.flatnonzero(prep["is_mask"][:, j] | prep["obs_mask"][:, j])
    return pd.Series(ctx["S_np"][idx, j], index=ctx["cal"][prep["rows"][idx]])


def _factor_nav(ctx, j, sign):
    """Net value and segment stats of column j of one run (all three segments if its OOS is open, else DEV only)."""
    prep, opened = ctx["prep"], ctx["opened"]
    is_m, obs_m, oos_m = prep["is_mask"], prep["obs_mask"], prep["oos_mask"]
    real_dates = np.asarray(ctx["cal"][prep["rows"] + 1].strftime("%Y-%m-%d"))   # return of row p: row p+1
    mask = is_m[:, j] | obs_m[:, j] | (oos_m[:, j] if opened else False)
    idx = np.flatnonzero(mask)
    seg = np.where(is_m[idx, j], 0, np.where(obs_m[idx, j], 1, 2))
    cfg = ctx["cfg"]
    res = factor_nav(ctx["S_np"][idx, j], ctx["y"][idx], ctx["sig"][idx], seg, real_dates[idx], sign,
                     cfg.position.buffer, cfg.target.cost_per_side)
    if res is None:
        return None
    segments = res["segments"]
    if not opened:
        segments[2] = {"seg": 2, "name": SEG_NAMES[2], "opened": False}
    keep = downsample_idx(seg)
    return {"segments": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in sg.items()}
                         for sg in segments],
            "nav": {"dates": [str(x) for x in real_dates[idx][keep]],
                    "values": [round(float(v), 4) for v in res["nav"][keep]],
                    "seg": [int(v) for v in seg[keep]]}}


def _oos_conclusion(ctx, j, sign, ic_is):
    """(verdict, details) of one factor of a run whose OOS is open, by the same rule and statistics as ``confirm``."""
    prep = ctx["prep"]
    m = prep["oos_mask"][:, j]
    t_oos, ic_oos, n = hac_t(ctx["S_np"][m, j][:, None], ctx["y"][m], ctx["cfg"].stats.hac_lags, OOS_MIN_OBS)
    t_signed, ic_signed = sign * t_oos[0], sign * ic_oos[0]
    verdict = oos_verdict(t_signed, ic_signed, ic_is, ctx["cfg"].verdict.obs_min_retention)
    return verdict, _jsonable({"n": int(n[0]), "t_signed": t_signed, "p_one": float(norm_sf(t_signed)),
                               "ic_oos_signed": ic_signed, "retention_oos": ic_signed / abs(float(ic_is))})


def build_asset_data(run_dirs) -> dict:
    """The factor library of one target, pooled over one run directory or a list of them (same target; the order
    is the registry order, later = newer)."""
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    ctxs = [_load_run(p, i) for i, p in enumerate(run_dirs)]
    slugs = {slugify(c["run"]["target"]) for c in ctxs}
    if len(slugs) != 1:
        raise ValueError(f"run directories belong to different targets: {sorted(slugs)}")
    target, slug = ctxs[0]["run"]["target"], slugs.pop()

    # ---- 1. pool the factors that passed the IS bar, with their DEV signals
    entries, skipped = [], []
    for ctx in ctxs:
        for _, r in ctx["cand"].iterrows():
            if r["id"] not in ctx["col_of"]:
                skipped.append({"id": r["id"], "reason": "信号不在候选池中"})
                continue
            j = ctx["col_of"][r["id"]]
            entries.append({"ctx": ctx, "row": r, "j": j, "id": r["id"], "transform": r["transform"],
                            "run_idx": ctx["idx"], "opened": ctx["opened"], "z": float(r["z"]),
                            "dev": _dev_series(ctx, j)})
    n_passed = len(entries)

    # ---- 2. the same factor tested under different ids / in different runs is one factor
    keepers = []
    for g in group_same_factors(entries):
        members = [entries[i] for i in g]
        k = members[choose_keeper(members)]
        k["also_as"] = list(dict.fromkeys(m["id"] for m in members if m is not k and m["id"] != k["id"]))
        keepers.append(k)
    n_merged = n_passed - len(keepers)

    # ---- 3. net value and segment stats of every kept factor (OOS rows only from runs whose OOS is open)
    pool = []
    for e in keepers:
        ctx, r = e["ctx"], e["row"]
        sign = float(r["sign"])
        nav = _factor_nav(ctx, e["j"], sign)
        if nav is None:
            skipped.append({"id": e["id"], "reason": "样本内净收益波动为 0 或无法计算，无法标定净值"})
            continue
        e["nav"], e["sign"] = nav, sign
        e["desc"], e["title"] = _display(ctx, r)
        pool.append(e)

    # ---- 4. layers on the pooled factors; de-duplication is recomputed on the pooled tradable ones
    criteria = ctxs[-1]["criteria"]                                      # gates of the newest run
    criteria_mixed = any(c["criteria"] != criteria for c in ctxs)
    for n_, e in enumerate(pool):
        e["key"] = f"k{n_}"
        e["obs_ok"] = e["row"]["obs_status"] != "DROP"
        e["trade_ok"] = str(e["row"].get("trade_status", "")) != "DROP"
    tradable = [e for e in pool if e["obs_ok"] and e["trade_ok"]]
    t_of = lambda e: float(e["row"]["t_dev"]) if pd.notna(e["row"]["t_dev"]) else float("-inf")
    tradable.sort(key=lambda e: -t_of(e))                                # stable: ties keep the pool order
    thresh = criteria.get("dedupe_corr", 0.7)
    rep_of = pool_representatives([e["key"] for e in tradable], {e["key"]: e["dev"] for e in tradable}, thresh)
    by_key = {e["key"]: e for e in pool}

    factors = []
    for e in pool:
        ctx, r = e["ctx"], e["row"]
        rep = rep_of.get(e["key"])
        stage = factor_stage(r, rep == e["key"])
        verdict, detail = None, None
        if stage == 4 and ctx["opened"]:
            verdict, detail = _oos_conclusion(ctx, e["j"], e["sign"], float(r["ic"]))
        factors.append(_jsonable({
            "id": e["id"], "series": r["series"], "transform": r["transform"],
            "title": e["title"], "description": e["desc"], "also_as": e["also_as"],
            "group": r["group"], "sign": int(e["sign"]), "direction": direction_text(e["sign"], target),
            "formula": formula_text(ctx["panel"].meta.loc[r["series"]], r["transform"], ctx["cfg"], e["sign"]),
            "is_verdict": str(r["verdict"]), "stage": stage,
            "stage_note": stage_note(r, stage, ctx["criteria"], by_key[rep]["title"] if stage == 3 else None),
            "z": float(r["z"]), "t_dev": float(r["t_dev"]) if pd.notna(r["t_dev"]) else None,
            "cluster_size": (sum(1 for v in rep_of.values() if v == e["key"]) if stage == 4 else None),
            "flags": flag_texts(r["flags"]), "oos_opened": ctx["opened"],
            "oos_verdict": verdict, "oos_confirm": detail,
            "segments": e["nav"]["segments"], "nav": e["nav"]["nav"],
        }))

    def sort_key(f):
        if f["stage"] == 4:
            v = VERDICT_ORDER.get(f["oos_verdict"], 4)
            t = f["t_dev"] if f["t_dev"] is not None else float("-inf")
            return (-4, v, -t)
        return (-f["stage"], 0, -abs(f["z"]))
    factors.sort(key=sort_key)

    final = [f for f in factors if f["stage"] == 4]
    vc = pd.Series([f["oos_verdict"] for f in final if f["oos_verdict"]], dtype=object).value_counts()
    oos_counts = {"confirmed": int(vc.get("CONFIRMED", 0)), "consistent": int(vc.get("CONSISTENT", 0)),
                  "failed": int(vc.get("FAILED", 0)), "insufficient": int(vc.get("INSUFFICIENT", 0)),
                  "unopened": sum(1 for f in final if not f["oos_opened"])}
    pub = [c["run"]["panel_info"].get("published_at") for c in ctxs if c["run"]["panel_info"].get("published_at")]
    last = ctxs[-1]["run"]["config"]["sample"]
    return {
        "target": target, "slug": slug,
        "db_published_at": max(pub) if pub else None,
        "sample_start": min(c["run"]["dates"]["sample_start"] for c in ctxs),
        "sample_end": max(c["run"]["dates"]["sample_end"] for c in ctxs),
        "is_frac": last["is_frac"], "obs_frac": last["obs_frac"],
        "funnel": stage_counts(factors), "oos_counts": oos_counts,
        "criteria": criteria, "criteria_mixed": bool(criteria_mixed),
        "n_merged": int(n_merged), "factors": factors, "skipped": skipped,
        "runs": [c["name"] for c in ctxs],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------- site
REGISTRY = "runs.json"


def load_registry(out: Path) -> list[str]:
    p = out / REGISTRY
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"{p} must be a JSON list of run directory paths")
    return data


def build_site(run_dirs, out_dir, reset=False) -> dict:
    """Build the site from the registry of the output directory (``runs.json``) plus ``run_dirs``, which are
    appended to it (de-duplicated, first-seen order). ``reset`` ignores the existing registry."""
    out = Path(out_dir)
    registry = [] if reset else load_registry(out)
    for p in run_dirs:
        ap = str(Path(p).resolve())
        if ap not in registry:
            registry.append(ap)
    if not registry:
        raise ValueError("no run directories: give at least one RUN_DIR (the registry in the output is empty)")
    missing = [p for p in registry if not Path(p).is_dir()]
    if missing:
        raise ValueError("run directories in the registry do not exist: " + ", ".join(missing)
                         + " (use --reset to rebuild from the given RUN_DIRs only)")
    groups: dict[str, list[str]] = {}                  # slug -> run dirs, in registry order
    for p in registry:
        groups.setdefault(slugify(_read_run(Path(p))["target"]), []).append(p)
    built = {slug: build_asset_data(ps) for slug, ps in groups.items()}    # all checks pass before anything is written
    (out / "data").mkdir(parents=True, exist_ok=True)
    for old in (out / "data").glob("*.json"):
        if old.stem not in built:
            old.unlink()                                                    # generated files of targets no longer built
    assets = []
    for slug, data in built.items():
        (out / "data" / f"{slug}.json").write_text(
            json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
        assets.append({"target": data["target"], "slug": slug, "sample_end": data["sample_end"],
                       "db_published_at": data["db_published_at"], "funnel": data["funnel"],
                       "oos_counts": data["oos_counts"], "n_factors": len(data["factors"])})
        print(f"[site] {data['target']}: {len(data['factors'])} factors from {len(data['runs'])} runs "
              f"({data['n_merged']} duplicates merged)")
    head, body = render_page({"assets": assets})
    (out / "artifact.html").write_text(head + body, encoding="utf-8")
    (out / "index.html").write_text(
        "<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        + head + "</head>\n<body>\n" + body + "</body>\n</html>\n", encoding="utf-8")
    (out / REGISTRY).write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"assets": assets}


def render_page(manifest: dict) -> tuple[str, str]:
    """Single template: (head part, body part). artifact.html is their concatenation, index.html wraps them."""
    mj = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    head = ("<title>因子跟踪看板</title>\n"
            "<meta name=\"claude:viewport-fit-sides\" content=\"cover\">\n"
            "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n"
            "<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>\n"
            "<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
            "&family=IBM+Plex+Sans:wght@400;500;600&display=swap\">\n"
            f"<style>\n{CSS}</style>\n")
    body = (BODY_HTML + "<script>\n" + JS.replace("__MANIFEST__", mj) + "</script>\n")
    return head, body


CSS = r"""
:root {
  --bg: #f3f6fb; --surface: #fbfcfe; --fg: #141b27; --fg-2: #435068; --muted: #5d6b82;
  --grid: #e1e7f0; --axis: #b4bfd0; --border: #d2dae6; --hover: #e8eef8; --focus: #1f5fc2;
  --seg-is: #2a78d6; --seg-obs: #1baf7a; --seg-oos: #eb6834;
  --st-good: #0ca30c; --st-warn: #fab219; --st-crit: #d03b3b; --st-na: #7d8aa0;
  --tip-bg: #141b27; --tip-fg: #f3f6fb; --tip-muted: #b4bfd0;
  --font-sans: "IBM Plex Sans", "PingFang SC", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", "PingFang SC", "Microsoft YaHei", "Noto Sans SC", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0c1118; --surface: #131a24; --fg: #e5ebf4; --fg-2: #b2bdcf; --muted: #8f9db3;
    --grid: #202a38; --axis: #3b485c; --border: #2a3545; --hover: #1b2432; --focus: #6aa6ff;
    --seg-is: #3987e5; --seg-obs: #199e70; --seg-oos: #d95926;
    --st-good: #0ca30c; --st-warn: #fab219; --st-crit: #e0504f; --st-na: #8795ab;
    --tip-bg: #e5ebf4; --tip-fg: #0c1118; --tip-muted: #435068;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0c1118; --surface: #131a24; --fg: #e5ebf4; --fg-2: #b2bdcf; --muted: #8f9db3;
  --grid: #202a38; --axis: #3b485c; --border: #2a3545; --hover: #1b2432; --focus: #6aa6ff;
  --seg-is: #3987e5; --seg-obs: #199e70; --seg-oos: #d95926;
  --st-good: #0ca30c; --st-warn: #fab219; --st-crit: #e0504f; --st-na: #8795ab;
  --tip-bg: #e5ebf4; --tip-fg: #0c1118; --tip-muted: #435068;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { background: var(--bg); color: var(--fg); margin: 0; font: 14px/1.55 var(--font-sans); }
.shell { max-width: 1240px; margin: 0 auto;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.mono, .num, pre, td.n { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
a { color: inherit; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 2px; }
.top { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg);
  border-bottom: 1px solid var(--border); }
.top .shell { display: flex; align-items: center; gap: 20px; padding-block: 10px; min-width: 0; }
.brand { font-weight: 600; font-size: 16px; white-space: nowrap; letter-spacing: .02em; }
nav.tabs { display: flex; gap: 2px; overflow-x: auto; min-width: 0; scrollbar-width: thin; }
nav.tabs a { text-decoration: none; padding: 5px 12px; color: var(--fg-2); border-bottom: 2px solid transparent;
  white-space: nowrap; font-weight: 500; }
nav.tabs a:hover { background: var(--hover); }
nav.tabs a[aria-current="page"] { color: var(--fg); border-bottom-color: var(--fg); font-weight: 600; }
main { padding-block: 22px 56px; min-height: 60vh; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 15px; font-weight: 600; margin: 0 0 8px; }
.sub { color: var(--fg-2); margin: 0 0 18px; }
.muted { color: var(--muted); }
.status { padding: 28px 0; color: var(--fg-2); }
.status.err { color: var(--fg); }
button { font: inherit; color: inherit; cursor: pointer; }
button.link { background: none; border: 1px solid var(--border); padding: 4px 12px; margin-left: 10px; }
.scroll { overflow-x: auto; max-width: 100%; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 12px 6px 0; border-bottom: 1px solid var(--grid); white-space: nowrap; vertical-align: top; }
th { font-weight: 500; color: var(--muted); font-size: 12px; }
td.n, th.n { text-align: right; }
td.n { padding-right: 14px; }
th.n { padding-right: 14px; }
table.ov td:first-child a { font-weight: 600; text-decoration: none; border-bottom: 1px solid var(--axis); }
table.ov tr:hover td { background: var(--hover); }
table.ov tr { cursor: pointer; }
.pipe { color: var(--fg-2); }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr)); gap: 14px 28px;
  margin: 14px 0 20px; padding-bottom: 18px; border-bottom: 1px solid var(--border); }
.summary > div { min-width: 0; }
.summary dt { color: var(--muted); font-size: 12px; margin: 0; }
.summary dd { margin: 2px 0 0; overflow-wrap: anywhere; }
.summary .wide { grid-column: 1 / -1; }
.funnel { display: flex; flex-wrap: wrap; gap: 6px 6px; align-items: center; margin: 8px 0 10px; }
.funnel button.fstep { background: transparent; border: 1px solid var(--border); padding: 4px 12px; border-radius: 2px; }
.funnel button.fstep:hover { background: var(--hover); }
.funnel button.fstep[aria-pressed="true"] { border-color: var(--fg); background: var(--fg); color: var(--bg); font-weight: 600; }
.funnel .fend { min-width: 0; overflow-wrap: anywhere; }
.stagesec { margin: 18px 0 8px; }
.stagehead { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.stagedesc { margin: 0 0 10px; font-size: 12.5px; }
.stagenote { margin: 0 0 10px; font-size: 12.5px; color: var(--fg-2); }
article.card.weak { border-style: dashed; background: transparent; }
article.card.weak h3 { font-weight: 500; }
.chartbox { min-height: 230px; }
.funnel b { font-family: var(--font-mono); font-weight: 600; font-variant-numeric: tabular-nums; }
.funnel .arr { color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; align-items: center; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key { display: inline-block; width: 18px; height: 0; border-top: 2px solid currentColor; border-radius: 2px; }
.key.is { color: var(--seg-is); } .key.obs { color: var(--seg-obs); } .key.oos { color: var(--seg-oos); }
.key.off { color: var(--axis); border-top-style: dotted; }
.filters { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 16px; align-items: center; }
.filters button { background: transparent; border: 1px solid var(--border); padding: 4px 12px; border-radius: 2px; }
.filters button:hover { background: var(--hover); }
.filters button[aria-pressed="true"] { border-color: var(--fg); background: var(--fg); color: var(--bg); font-weight: 600; }
.filters .cnt { font-family: var(--font-mono); font-variant-numeric: tabular-nums; margin-left: 6px; opacity: .8; }
.filterbars { margin: 0 0 22px; }
.filterbars .filters { margin-bottom: 8px; }
.glabel { color: var(--muted); font-size: 12px; margin-right: 4px; min-width: 5em; }
.summary.crit { margin: 10px 0 16px; padding-bottom: 0; border-bottom: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 30rem), 1fr)); gap: 16px; }
article.card { border: 1px solid var(--border); background: var(--surface); padding: 14px 16px 12px; min-width: 0; }
.card h3 { font-size: 15px; line-height: 1.35; margin: 0 0 4px; font-weight: 600; overflow-wrap: anywhere; }
.card .idline { font-size: 12px; color: var(--muted); overflow-wrap: anywhere; margin-bottom: 8px; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.badge { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--border); padding: 1px 8px;
  font-size: 12.5px; line-height: 1.6; background: var(--bg); min-width: 0; }
.badge .shape { font-size: 11px; line-height: 1; }
.badge.good .shape { color: var(--st-good); } .badge.warn .shape { color: var(--st-warn); }
.badge.crit .shape { color: var(--st-crit); } .badge.na .shape, .badge.none .shape { color: var(--st-na); }
.badge.none { border-style: dashed; color: var(--fg-2); }
.badge.dir { font-weight: 500; }
pre.formula { margin: 0 0 12px; padding: 8px 10px; background: var(--bg); border-left: 2px solid var(--axis);
  font-size: 12px; line-height: 1.6; overflow-x: auto; max-width: 100%; white-space: pre-wrap;
  overflow-wrap: anywhere; word-break: break-word; }
.chartbox { position: relative; margin-bottom: 8px; min-width: 0; }
.chartbox svg { display: block; width: 100%; touch-action: pan-y; }
.chartbox svg:focus-visible { outline: 2px solid var(--focus); }
.tip { position: absolute; pointer-events: none; background: var(--tip-bg); color: var(--tip-fg); padding: 6px 9px;
  font-size: 12px; line-height: 1.5; white-space: nowrap; z-index: 3; display: none; border-radius: 2px; }
.tip .tv { font-family: var(--font-mono); font-weight: 600; font-size: 13px; }
.tip .tl { color: var(--tip-muted); }
.tip .tk { display: inline-block; width: 12px; border-top: 2px solid currentColor; vertical-align: middle; margin-right: 5px; }
.cardlegend { display: flex; flex-wrap: wrap; gap: 2px 14px; font-size: 12px; color: var(--fg-2); margin-bottom: 4px; }
.cardlegend span { display: inline-flex; align-items: center; gap: 5px; }
.stats { font-size: 12px; }
.stats td, .stats th { padding-top: 4px; padding-bottom: 4px; }
.stats td.iv { color: var(--fg-2); line-height: 1.35; }
.oosnote { font-size: 12px; color: var(--fg-2); margin: 8px 0 0; }
.flags { margin: 8px 0 0; padding: 0; list-style: none; font-size: 12.5px; display: flex; flex-wrap: wrap; gap: 4px 14px; color: var(--fg-2); }
.flags li::before { content: "△ "; color: var(--muted); }
svg text { font-family: var(--font-mono); font-size: 11px; fill: var(--muted); }
svg text.seglab { font-family: var(--font-sans); fill: var(--fg-2); }
svg text.endlab { fill: var(--fg); font-weight: 600; font-size: 12px; }
svg .grid-l { stroke: var(--grid); stroke-width: 1; }
svg .axis-l { stroke: var(--axis); stroke-width: 1; }
svg .div-l { stroke: var(--axis); stroke-width: 1; }
svg .ring { stroke: var(--surface); stroke-width: 2; }
svg .xh { stroke: var(--fg-2); stroke-width: 1; }
svg path.ln { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.s0 { stroke: var(--seg-is); } .s1 { stroke: var(--seg-obs); } .s2 { stroke: var(--seg-oos); }
.f0 { fill: var(--seg-is); } .f1 { fill: var(--seg-obs); } .f2 { fill: var(--seg-oos); }
.empty { padding: 24px 0; color: var(--fg-2); }
@media (max-width: 520px) { body { font-size: 14px; } article.card { padding: 12px 12px 10px; }
  .top .shell { gap: 12px; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; scroll-behavior: auto !important; } }
"""

BODY_HTML = """<div id="app">
<div class="top" role="banner"><div class="shell"><span class="brand">因子跟踪看板</span><nav class="tabs" id="nav" aria-label="资产导航"></nav></div></div>
<div class="shell"><main id="view" aria-live="polite"></main></div>
</div>
"""

JS = r"""
(function () {
'use strict';
var MANIFEST = __MANIFEST__;
var SEGN = ['样本内', '观察期', '样本外'];
var VERD = {
  CONFIRMED: {label: '确认', shape: '●', cls: 'good'},
  CONSISTENT: {label: '一致', shape: '◐', cls: 'warn'},
  FAILED: {label: '失败', shape: '✕', cls: 'crit'},
  INSUFFICIENT: {label: '样本不足', shape: '◇', cls: 'na'},
  NONE: {label: '未打开', shape: '○', cls: 'none'}
};
var cache = {}, redraws = [], filterKey = 'ALL', curSlug = null;
var viewEl = document.getElementById('view'), navEl = document.getElementById('nav');
var SVGNS = 'http://www.w3.org/2000/svg';

function h(tag, attrs, kids) {
  var e = document.createElement(tag), k, v;
  if (attrs) for (k in attrs) {
    v = attrs[k];
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  (kids || []).forEach(function (c) { if (c != null) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
  return e;
}
function sv(tag, attrs, text) {
  var e = document.createElementNS(SVGNS, tag), k;
  if (attrs) for (k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}
function fx(x, d) { return (x == null || !isFinite(x)) ? '–' : Number(x).toFixed(d); }
function fpct(x, d, sign) {
  if (x == null || !isFinite(x)) return '–';
  var s = (x * 100).toFixed(d);
  return (sign && x > 0 ? '+' : '') + s + '%';
}
function fint(x) { return x == null ? '–' : String(x); }

/* ------------------------------------------------------------ views */
function renderNav(active) {
  navEl.textContent = '';
  var items = [{slug: 'overview', label: '总览'}].concat(MANIFEST.assets.map(function (a) { return {slug: a.slug, label: a.target}; }));
  items.forEach(function (it) {
    navEl.appendChild(h('a', {href: '#' + it.slug, 'aria-current': it.slug === active ? 'page' : null, text: it.label}));
  });
}

function oosText(a) {
  var c = a.oos_counts || {}, t = '确认 ' + c.confirmed + ' · 一致 ' + c.consistent + ' · 失败 ' + c.failed;
  if (c.insufficient) t += ' · 样本不足 ' + c.insufficient;
  if (c.unopened) t += ' · 未打开 ' + c.unopened;
  return t;
}

var STAGES = {
  4: {name: '去重后留下', shape: '◆'},
  3: {name: '去重时被合并', shape: '⊕'},
  2: {name: '可交易筛选未通过', shape: '▽'},
  1: {name: '观察期未保留', shape: '✕'}
};

function renderOverview() {
  var root = h('div');
  root.appendChild(h('h1', {text: '因子跟踪看板'}));
  root.appendChild(h('p', {class: 'sub', text: '每个标的资产一个因子库：展示全部通过样本内门槛的因子（同一因子只出现一次），并按漏斗逐层分开。点击一行进入。'}));
  if (!MANIFEST.assets.length) { root.appendChild(h('div', {class: 'empty', text: '没有资产。'})); return root; }
  var head = ['标的', '数据截止日', '样本内过门槛', '观察期保留', '可交易', '去重后留下', '样本外'];
  var tb = h('tbody');
  MANIFEST.assets.forEach(function (a) {
    var f = a.funnel;
    tb.appendChild(h('tr', {onclick: function (e) { if (e.target.tagName !== 'A') location.hash = '#' + a.slug; }}, [
      h('td', null, [h('a', {href: '#' + a.slug, text: a.target})]),
      h('td', {class: 'mono', text: a.sample_end}),
      h('td', {class: 'n', text: fint(f.n_is)}),
      h('td', {class: 'n', text: fint(f.n_obs)}),
      h('td', {class: 'n', text: fint(f.n_trade)}),
      h('td', {class: 'n', text: fint(f.n_final)}),
      h('td', {text: oosText(a)})
    ]));
  });
  var tbl = h('table', {class: 'ov'}, [h('thead', null, [h('tr', null, head.map(function (t, i) { return h('th', {class: (i >= 2 && i <= 5) ? 'n' : null, text: t}); }))]), tb]);
  root.appendChild(h('div', {class: 'scroll'}, [tbl]));
  root.appendChild(h('p', {class: 'muted', style: 'margin-top:14px', text: '漏斗：样本内过门槛 → 观察期保留 → 可交易（扣成本净 Sharpe 达标）→ 去重后留下 → 样本外确认 / 一致 / 失败。每一层的数字包含更靠后的层。样本外只对去重后留下的因子给结论，且只有该因子所在的测试运行过 confirm 之后才显示，否则为「未打开」。'}));
  return root;
}

function dl(label, kids, cls) { return h('div', {class: cls}, [h('dt', {text: label}), h('dd', null, kids)]); }

function criteriaEl(a) {
  var c = a.criteria;
  if (!c || c.z_weak == null || c.obs_min_retention == null) return null;
  var txt = '样本内 |z| ≥ ' + c.z_weak + (c.is_min_net_sharpe != null ? ' 且扣成本后净 Sharpe > ' + c.is_min_net_sharpe : '') + '；观察期方向一致且 IC 不低于样本内的 ' + (Math.round(c.obs_min_retention * 1000) / 10) + '%';
  if (c.min_net_sharpe != null) txt += '；扣成本后净 Sharpe ≥ ' + c.min_net_sharpe + '（样本内+观察期）';
  if (c.dedupe_corr != null) txt += '；去重相关阈值 ' + c.dedupe_corr;
  if (a.criteria_mixed) txt += '。部分因子来自门槛不同的早期测试';
  return dl('入选条件', [txt], 'wide');
}

function execText(c) {
  if (!c || c.execution_delay == null) return '';
  return c.execution_delay > 0 ? '；成交：信号形成后第 ' + c.execution_delay + ' 个收盘' : '；成交：信号形成当日收盘';
}

function stageDesc(k, c) {
  if (k === 4) return '过了样本内门槛、观察期保留' + (c.min_net_sharpe != null ? '、扣成本后净 Sharpe 达标' : '') + '，并在去重后作为代表留下。只有这一层的因子有样本外结论。';
  if (k === 3) return '通过了前几层，但与另一个代表高度相关' + (c.dedupe_corr != null ? '（相关 ≥ ' + c.dedupe_corr + '）' : '') + '，去重时并入该代表。';
  if (k === 2) return '观察期保留，但样本内+观察期扣成本后的净 Sharpe 低于 ' + (c.min_net_sharpe != null ? c.min_net_sharpe : '门槛') + '。';
  return '过了样本内门槛（|z| ≥ ' + c.z_weak + '），但观察期 IC 保留比例低于 ' + c.obs_min_retention + ' 或方向相反。';
}

function legendEl(opened, mixed) {
  return h('div', {class: 'legend'}, [
    h('span', null, [h('i', {class: 'key is'}), '样本内']),
    h('span', null, [h('i', {class: 'key obs'}), '观察期']),
    opened ? h('span', null, [h('i', {class: 'key oos'}), '样本外']) : h('span', {class: 'muted'}, [h('i', {class: 'key off'}), '样本外未打开']),
    mixed ? h('span', {class: 'muted', text: '（部分因子的样本外未打开）'}) : null
  ]);
}

function funnelBar(a, cur, pick) {
  var f = a.funnel, el = h('div', {class: 'funnel', role: 'group', 'aria-label': '漏斗：点击某一步只看到达该步的因子'});
  [['样本内过门槛', f.n_is], ['观察期保留', f.n_obs], ['可交易', f.n_trade], ['去重后留下', f.n_final]].forEach(function (p, i) {
    if (i) el.appendChild(h('span', {class: 'arr', 'aria-hidden': 'true', text: '→'}));
    el.appendChild(h('button', {type: 'button', class: 'fstep', 'aria-pressed': String(cur === i + 1), onclick: function () { pick(i + 1); }}, [p[0] + ' ', h('b', {text: String(p[1])})]));
  });
  el.appendChild(h('span', {class: 'arr', 'aria-hidden': 'true', text: '→'}));
  var c = a.oos_counts, kids = ['样本外：确认 ', h('b', {text: String(c.confirmed)}), ' · 一致 ', h('b', {text: String(c.consistent)}), ' · 失败 ', h('b', {text: String(c.failed)})];
  if (c.insufficient) kids.push(' · 样本不足 ', h('b', {text: String(c.insufficient)}));
  if (c.unopened) kids.push(' · 未打开 ', h('b', {text: String(c.unopened)}), ' 个');
  el.appendChild(h('span', {class: 'fend'}, kids));
  return el;
}

function renderAsset(a) {
  var root = h('div'), cur = 1;
  var nOpen = a.factors.filter(function (f) { return f.oos_opened; }).length;
  var nSkip = a.skipped.length;
  root.appendChild(h('h1', {text: a.target + ' 因子跟踪'}));
  root.appendChild(h('p', {class: 'sub', text: '通过样本内门槛的因子共 ' + a.factors.length + ' 个（同一因子只算一次），其中去重后留下 ' + a.funnel.n_final + ' 个' + (nSkip ? '；另有 ' + nSkip + ' 个无法标定净值而未出图' : '') + '。'}));
  root.appendChild(h('dl', {class: 'summary'}, [
    dl('数据截止日', [h('span', {class: 'mono', text: a.sample_end})]),
    dl('标的样本', [h('span', {class: 'mono', text: a.sample_start + ' 至 ' + a.sample_end})]),
    dl('数据库发布', [h('span', {class: 'mono', text: (a.db_published_at || '–').slice(0, 10)})]),
    dl('净值定义', ['单利累计；vectorbt 模拟损益并扣除交易成本；样本内年化波动标定为 10%' + execText(a.criteria)]),
    dl('三段图例', [legendEl(nOpen > 0, nOpen > 0 && nOpen < a.factors.length)])
  ]));
  var info = [criteriaEl(a),
    dl('切分规则', ['每个因子在自身有效数据内按 ' + Math.round(a.is_frac * 100) + '/' + Math.round(a.obs_frac * 100) + '/' + Math.round((1 - a.is_frac - a.obs_frac) * 100) + ' 切成样本内 / 观察期 / 样本外'], 'wide')];
  if (nSkip) info.push(dl('未出图', [a.skipped.map(function (k) { return k.id; }).join('、')], 'wide'));
  var vbar = h('div', {class: 'filters', role: 'group', 'aria-label': '按样本外结论筛选'});
  var funnelHost = h('div'), secs = h('div', {class: 'sections'});
  var cnt = {ALL: a.funnel.n_final};
  a.factors.forEach(function (f) { if (f.stage === 4) { var k = f.oos_verdict || 'NONE'; cnt[k] = (cnt[k] || 0) + 1; } });
  var vopts = [['ALL', '全部']];
  if (nOpen) vopts.push(['CONFIRMED', '确认'], ['CONSISTENT', '一致'], ['FAILED', '失败'], ['INSUFFICIENT', '样本不足']);
  if (cnt.NONE) vopts.push(['NONE', '未打开']);
  if (!vopts.some(function (o) { return o[0] === filterKey; })) filterKey = 'ALL';
  var items = [], io = null;
  if ('IntersectionObserver' in window) {
    io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        io.unobserve(e.target);
        if (e.target._item) e.target._item.draw();
      });
    }, {rootMargin: '400px 0px'});
  }
  function fill() {
    if (io) io.disconnect();
    items = [];
    redraws = [];
    funnelHost.textContent = '';
    funnelHost.appendChild(funnelBar(a, cur, function (k) { cur = k; fill(); }));
    secs.textContent = '';
    var vis = a.factors.filter(function (f) {
      return f.stage >= cur && (filterKey === 'ALL' || (f.stage === 4 && (f.oos_verdict || 'NONE') === filterKey));
    });
    if (!vis.length) secs.appendChild(h('div', {class: 'empty', text: '当前筛选下没有因子。'}));
    [4, 3, 2, 1].forEach(function (k) {
      var fs = vis.filter(function (f) { return f.stage === k; });
      if (!fs.length) return;
      var list = h('div', {class: 'grid'});
      fs.forEach(function (f) { list.appendChild(card(f, a, items)); });
      secs.appendChild(h('section', {class: 'stagesec st' + k, 'aria-label': STAGES[k].name}, [
        h('h3', {class: 'stagehead'}, [h('span', {class: 'shape', 'aria-hidden': 'true', text: STAGES[k].shape + ' '}), STAGES[k].name, h('span', {class: 'muted', text: ' · ' + fs.length + ' 个'})]),
        h('p', {class: 'stagedesc muted', text: stageDesc(k, a.criteria)}),
        list]));
    });
    items.forEach(function (it) {
      var item = {done: false, draw: function () { item.done = true; drawChart(it.box, it.f, it.f.oos_opened); }};
      it.box._item = item;
      redraws.push(function () { if (item.done) item.draw(); });
      var top = it.box.getBoundingClientRect().top;
      if (!io || top < window.innerHeight * 1.5) item.draw(); else io.observe(it.box);
    });
    Array.prototype.forEach.call(vbar.querySelectorAll('button'), function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-k') === filterKey)); });
  }
  vbar.appendChild(h('span', {class: 'glabel', text: '样本外结论'}));
  vopts.forEach(function (o) {
    vbar.appendChild(h('button', {type: 'button', 'data-k': o[0], onclick: function () { filterKey = o[0]; fill(); }}, [o[1], h('span', {class: 'cnt', text: String(cnt[o[0]] || 0)})]));
  });
  vbar.appendChild(h('span', {class: 'muted', text: nOpen ? '只对「去重后留下」的因子给结论；选了某个结论时其他层隐藏。' : '样本外未打开，暂无结论可筛选。'}));
  root.appendChild(funnelHost);
  root.appendChild(h('dl', {class: 'summary crit'}, info));
  root.appendChild(h('div', {class: 'filterbars'}, [vbar]));
  root.appendChild(secs);
  root._after = fill;
  return root;
}

function badge(cls, shape, text, extra) {
  return h('span', {class: 'badge ' + cls + (extra ? ' ' + extra : '')}, [h('span', {class: 'shape', 'aria-hidden': 'true', text: shape}), text]);
}

function card(f, a, items) {
  var v = VERD[f.oos_verdict || 'NONE'] || VERD.NONE, st = STAGES[f.stage];
  var box = h('div', {class: 'chartbox'});
  var badges = [badge('dir', f.sign > 0 ? '▲' : '▼', f.direction, 'dir'), badge('stage', st.shape, '到达：' + st.name, 'stage')];
  if (f.stage === 4) badges.push(badge(v.cls, v.shape, '样本外：' + v.label));
  var el = h('article', {class: 'card' + (f.stage < 4 ? ' weak' : ''), id: 'card-' + a.slug + '-' + String(f.id).replace(/\s/g, '_'), 'data-stage': String(f.stage)}, [
    h('h3', {text: f.title}),
    h('div', {class: 'idline'}, [h('span', {class: 'mono', text: f.id}), ' · ' + f.group + ' · ' + (f.is_verdict === 'PASS' ? '样本内通过' : '样本内弱通过') + ' · z ' + fx(f.z, 2) + (f.cluster_size > 1 ? ' · 同簇 ' + f.cluster_size + ' 个' : '')]),
    f.also_as && f.also_as.length ? h('div', {class: 'idline'}, ['同一因子的其他写法：', h('span', {class: 'mono', text: f.also_as.join('、')})]) : null,
    h('div', {class: 'badges'}, badges),
    f.stage_note ? h('p', {class: 'stagenote', text: f.stage_note}) : null,
    h('pre', {class: 'formula', tabindex: '0', 'aria-label': '因子公式', text: f.formula.join('\n')}),
    cardLegend(f.oos_opened),
    box,
    statsTable(f),
    oosNote(f),
    f.flags.length ? h('ul', {class: 'flags'}, f.flags.map(function (t) { return h('li', {text: t}); })) : null
  ]);
  items.push({box: box, f: f});
  return el;
}

function cardLegend(opened) {
  return h('div', {class: 'cardlegend'}, [
    h('span', null, [h('i', {class: 'key is'}), '样本内']),
    h('span', null, [h('i', {class: 'key obs'}), '观察期']),
    opened ? h('span', null, [h('i', {class: 'key oos'}), '样本外']) : h('span', {class: 'muted'}, [h('i', {class: 'key off'}), '样本外未打开'])
  ]);
}

function statsTable(f) {
  var rows = f.segments.map(function (s, i) {
    if (s.opened === false) return h('tr', null, [h('td', null, [h('i', {class: 'key off'}), ' ' + s.name]), h('td', {colspan: '6', class: 'muted', text: '未打开'})]);
    if (!s.n) return h('tr', null, [h('td', null, [h('i', {class: 'key ' + ['is', 'obs', 'oos'][i]}), ' ' + s.name]), h('td', {colspan: '6', class: 'muted', text: '无数据'})]);
    return h('tr', null, [
      h('td', null, [h('i', {class: 'key ' + ['is', 'obs', 'oos'][i]}), ' ' + s.name]),
      h('td', {class: 'mono iv'}, [s.start, h('br'), s.end]),
      h('td', {class: 'n', text: fint(s.n)}),
      h('td', {class: 'n', text: fx(s.ic, 3)}),
      h('td', {class: 'n', text: fx(s.sharpe, 2)}),
      h('td', {class: 'n', text: fpct(s.cum_ret, 1, true)}),
      h('td', {class: 'n', text: fpct(s.max_dd, 1, false)})
    ]);
  });
  var heads = ['阶段', '区间', '天数', 'IC', '净 Sharpe', '累计收益', '最大回撤'];
  return h('div', {class: 'scroll'}, [h('table', {class: 'stats'}, [
    h('thead', null, [h('tr', null, heads.map(function (t, i) { return h('th', {class: i >= 2 ? 'n' : null, text: t}); }))]),
    h('tbody', null, rows)])]);
}

function oosNote(f) {
  var c = f.oos_confirm;
  if (!c) return null;
  return h('p', {class: 'oosnote'}, ['样本外统计：t ', h('span', {class: 'num', text: fx(c.t_signed, 2)}), ' · 单边 p ', h('span', {class: 'num', text: fx(c.p_one, 3)}), ' · retention ', h('span', {class: 'num', text: fx(c.retention_oos, 2)})]);
}

/* ------------------------------------------------------------ chart */
function niceTicks(lo, hi, n) {
  var span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
  var step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag, out = [], t = Math.ceil(lo / step - 1e-9) * step;
  for (; t <= hi + 1e-9; t += step) out.push(Math.round(t / step) * step);
  return {ticks: out, step: step};
}

function prep(f) {
  if (f._t) return;
  f._t = f.nav.dates.map(function (d) { return Date.parse(d + 'T00:00:00Z'); });
}

function drawChart(box, f, opened) {
  prep(f);
  var T = f._t, V = f.nav.values, G = f.nav.seg, n = T.length;
  box.textContent = '';
  if (!n) { box.appendChild(h('div', {class: 'empty', text: '没有数据。'})); return; }
  var W = Math.max(240, Math.round(box.clientWidth || 320)), H = 230;
  var ml = 40, mr = 46, mt = 22, mb = 22, pw = W - ml - mr, ph = H - mt - mb;
  var t0 = T[0], t1 = T[n - 1], tEnd = opened ? t1 : t1 + (t1 - t0) * 0.18;
  var lo = Math.min(1, Math.min.apply(null, V)), hi = Math.max(1, Math.max.apply(null, V)), pad = (hi - lo) * 0.08 || 0.05;
  lo -= pad; hi += pad;
  var nt = niceTicks(lo, hi, 4);
  function X(t) { return ml + (t - t0) / (tEnd - t0) * pw; }
  function Y(v) { return mt + (hi - v) / (hi - lo) * ph; }
  var svg = sv('svg', {viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', tabindex: '0',
    'aria-label': f.title + ' 净值走势，末值 ' + fx(V[n - 1], 2) + (opened ? '' : '，样本外未打开')});
  nt.ticks.forEach(function (v) {
    svg.appendChild(sv('line', {x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), class: 'grid-l'}));
    svg.appendChild(sv('text', {x: ml - 6, y: Y(v) + 4, 'text-anchor': 'end'}, v.toFixed(nt.step < 0.1 ? 2 : nt.step < 1 ? 1 : 0)));
  });
  svg.appendChild(sv('line', {x1: ml, x2: W - mr, y1: Y(1), y2: Y(1), class: 'axis-l'}));
  var y0 = new Date(t0).getUTCFullYear(), y1 = new Date(tEnd).getUTCFullYear(), span = y1 - y0 + 1;
  var ystep = 1; [1, 2, 5, 10, 20].some(function (s) { ystep = s; return span / s * 46 <= pw; });
  for (var yr = Math.ceil((y0 + 1) / ystep) * ystep; yr <= y1; yr += ystep) {
    var tt = Date.UTC(yr, 0, 1);
    if (tt < t0 || tt > (opened ? t1 : tEnd)) continue;
    svg.appendChild(sv('line', {x1: X(tt), x2: X(tt), y1: mt + ph, y2: mt + ph + 4, class: 'axis-l'}));
    svg.appendChild(sv('text', {x: X(tt), y: H - 6, 'text-anchor': 'middle'}, String(yr)));
  }
  svg.appendChild(sv('line', {x1: ml, x2: W - mr, y1: mt + ph, y2: mt + ph, class: 'axis-l'}));
  /* segment dividers + names */
  var starts = [0, -1, -1];
  for (var i = 0; i < n; i++) { if (starts[G[i]] < 0) starts[G[i]] = i; }
  var names = [[0, X(t0)], [1, starts[1] >= 0 ? X(T[starts[1]]) : -1], [2, starts[2] >= 0 ? X(T[starts[2]]) : (opened ? -1 : X(t1))]];
  names.forEach(function (p) {
    if (p[1] < 0) return;
    if (p[0] > 0) svg.appendChild(sv('line', {x1: p[1], x2: p[1], y1: mt - 6, y2: mt + ph, class: 'div-l'}));
    var off = (p[0] === 2 && !opened);
    svg.appendChild(sv('text', {x: p[1] + 4, y: mt - 9, class: 'seglab'}, off ? '样本外未打开' : SEGN[p[0]]));
  });
  /* the line, one path per segment (each starts at the previous segment's last point) */
  for (var s = 0; s < 3; s++) {
    var d = '', first = true;
    for (var j = 0; j < n; j++) {
      if (G[j] === s || (G[j] === s - 1 && (j + 1 < n && G[j + 1] === s))) {
        d += (first ? 'M' : 'L') + X(T[j]).toFixed(1) + ' ' + Y(V[j]).toFixed(1); first = false;
      }
    }
    if (d) svg.appendChild(sv('path', {d: d, class: 'ln s' + s}));
  }
  var ex = X(T[n - 1]), ey = Y(V[n - 1]);
  svg.appendChild(sv('circle', {cx: ex, cy: ey, r: 5, class: 'ring f' + G[n - 1]}));
  svg.appendChild(sv('text', {x: ex + 9, y: ey + 4, class: 'endlab'}, fx(V[n - 1], 2)));
  /* hover layer */
  var xh = sv('line', {y1: mt, y2: mt + ph, class: 'xh', visibility: 'hidden'});
  var dot = sv('circle', {r: 4.5, class: 'ring', visibility: 'hidden'});
  svg.appendChild(xh); svg.appendChild(dot);
  var tip = h('div', {class: 'tip', role: 'status'});
  var cur = n - 1;
  function show(i) {
    cur = Math.max(0, Math.min(n - 1, i));
    var x = X(T[cur]), y = Y(V[cur]);
    xh.setAttribute('x1', x); xh.setAttribute('x2', x); xh.setAttribute('visibility', 'visible');
    dot.setAttribute('cx', x); dot.setAttribute('cy', y); dot.setAttribute('class', 'ring f' + G[cur]); dot.setAttribute('visibility', 'visible');
    tip.textContent = '';
    tip.appendChild(h('div', {class: 'tl', text: f.nav.dates[cur]}));
    tip.appendChild(h('div', {class: 'tv', text: V[cur].toFixed(4)}));
    tip.appendChild(h('div', {class: 'tl'}, [h('i', {class: 'tk', style: 'color:var(--seg-' + ['is', 'obs', 'oos'][G[cur]] + ')'}), SEGN[G[cur]]]));
    tip.style.display = 'block';
    var tw = tip.offsetWidth, left = x + 12;
    if (left + tw > W) left = x - 12 - tw;
    tip.style.left = Math.max(0, left) + 'px'; tip.style.top = Math.max(0, y - 54) + 'px';
  }
  function hide() { xh.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); tip.style.display = 'none'; }
  function nearest(clientX) {
    var r = svg.getBoundingClientRect(), tx = t0 + ((clientX - r.left) * (W / r.width) - ml) / pw * (tEnd - t0);
    var a = 0, b = n - 1;
    while (b - a > 1) { var m = (a + b) >> 1; if (T[m] < tx) a = m; else b = m; }
    return Math.abs(T[a] - tx) <= Math.abs(T[b] - tx) ? a : b;
  }
  var hit = sv('rect', {x: ml, y: mt, width: pw, height: ph, fill: 'transparent'});
  svg.appendChild(hit);
  hit.addEventListener('pointermove', function (e) { show(nearest(e.clientX)); });
  hit.addEventListener('pointerdown', function (e) { show(nearest(e.clientX)); });
  hit.addEventListener('pointerleave', hide);
  svg.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft') { show(cur - Math.max(1, Math.round(n / 60))); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { show(cur + Math.max(1, Math.round(n / 60))); e.preventDefault(); }
    else if (e.key === 'Escape') hide();
  });
  svg.addEventListener('blur', hide);
  box.appendChild(svg); box.appendChild(tip);
  box._w = W;
}

/* ------------------------------------------------------------ routing */
function load(slug) {
  if (cache[slug]) return Promise.resolve(cache[slug]);
  return fetch('data/' + slug + '.json').then(function (r) {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }).then(function (d) { cache[slug] = d; return d; });
}

function route() {
  var name = decodeURIComponent((location.hash || '').replace(/^#/, '')) || 'overview';
  var asset = MANIFEST.assets.filter(function (a) { return a.slug === name; })[0];
  if (!asset) name = 'overview';
  curSlug = name;
  renderNav(name);
  viewEl.textContent = '';
  redraws = [];
  window.scrollTo(0, 0);
  if (name === 'overview') { viewEl.appendChild(renderOverview()); return; }
  var status = h('div', {class: 'status', text: '正在加载 ' + asset.target + ' 的因子数据…'});
  viewEl.appendChild(status);
  load(name).then(function (data) {
    if (curSlug !== name) return;
    viewEl.textContent = '';
    var v = renderAsset(data);
    viewEl.appendChild(v);
    v._after();
  }).catch(function (err) {
    if (curSlug !== name) return;
    viewEl.textContent = '';
    viewEl.appendChild(h('div', {class: 'status err', role: 'alert'}, [
      '加载失败：无法读取 data/' + name + '.json（' + (err && err.message ? err.message : err) + '）。请通过 HTTP 服务打开本页（例如 python -m http.server）。',
      h('button', {type: 'button', class: 'link', onclick: route, text: '重试'})]));
  });
}

var rzT = null, lastW = window.innerWidth;
window.addEventListener('resize', function () {
  if (window.innerWidth === lastW) return;
  lastW = window.innerWidth;
  clearTimeout(rzT);
  rzT = setTimeout(function () { redraws.forEach(function (fn) { fn(); }); }, 120);
});
window.addEventListener('hashchange', route);
route();
})();
"""
