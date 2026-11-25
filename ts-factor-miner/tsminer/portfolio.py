"""Portfolio engine: per-factor 有效 / 失效 state machines and an equal-weight portfolio of the valid factors.

Input: the dashboard's final layer of a target (``site.pool_layers``, stage 4), whatever its OOS verdict.
Timing (strict T+1): row t of ``S`` is formed after the close of t-1 and earns ``y_fwd[t]``, known only at the
close of t+1. A state used on row t is decided after the close of t-1, so it may use ``y_fwd`` / signals on rows
<= t-2 only: the machines run on each factor's own valid rows, then the state is shifted ``LAG`` = 2 calendar rows
(``always`` uses no information and is not shifted). An unopened OOS (no ``confirm.json``) is never read: such a
factor only has its IS + OBS rows, exactly as in ``site``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PortfolioConfig, load_portfolio_config
from .engine import simulate
from .pipeline import _corr, _jsonable, _sharpe, positions
from .site import TARGET_VOL, _load_run, _read_run, factor_stage, pool_layers, rolling_ic, slugify
from .transforms import forward_target

LAG = 2
SAMPLES = ("full", "oos")
METHOD_TEXT = {"always": "基线（永远有效）", "slope": "方法一（纯趋势）", "hyst": "方法二（趋势+水平，滞回）",
               "cumic": "方法三（累计 IC 站上均线）"}
POSITION_TEXT = {"cont": "连续版", "binA": "二元·因子级", "binB": "二元·组合级"}
SUMMARY_COLUMNS = ["method", "position_version", "sample", "start", "end", "years", "n_rows", "sharpe_net",
                   "sharpe_gross", "max_dd_10vol", "cum_pnl_10vol", "avg_abs_dpos", "avg_n_valid", "share_flat",
                   "corr_target", "target_sharpe"]


# ------------------------------------------------------------------ state machines (pure; arrays of valid rows)
def state_slope(ric, L) -> np.ndarray:
    """方法一: valid iff ric[k] > ric[k-L] (NaN -> invalid)."""
    ric = np.asarray(ric, dtype=float)
    out = np.zeros(len(ric), dtype=bool)
    if len(ric) > L:
        with np.errstate(invalid="ignore"):
            out[L:] = ric[L:] > ric[:-L]
    return out


def state_hyst(ric, L) -> np.ndarray:
    """方法二: invalid -> valid when ric > 0 and ric > ric[k-L]; valid -> invalid when ric < 0 and ric < ric[k-L];
    otherwise keep. Starts invalid; a NaN input forces invalid."""
    ric = np.asarray(ric, dtype=float)
    out = np.zeros(len(ric), dtype=bool)
    on = False
    for k in range(L, len(ric)):
        a, b = ric[k], ric[k - L]
        if not (np.isfinite(a) and np.isfinite(b)):
            on = False
        elif not on:
            on = bool(a > 0 and a > b)
        else:
            on = not (a < 0 and a < b)
        out[k] = on
    return out


def state_cumic(c, ma) -> np.ndarray:
    """方法三: valid iff C[k] > mean(C[k-ma+1..k]), C = cumsum(c); a full window is required."""
    C = np.cumsum(np.asarray(c, dtype=float))
    out = np.zeros(len(C), dtype=bool)
    if len(C) >= ma:
        avg = np.lib.stride_tricks.sliding_window_view(C, ma).mean(axis=1)
        with np.errstate(invalid="ignore"):
            out[ma - 1:] = C[ma - 1:] > avg
    return out


def raw_states(s, y, sign, pc: PortfolioConfig) -> dict:
    """Raw (unshifted) state of every configured method on one factor's valid rows (``s`` all finite)."""
    s, y = np.asarray(s, dtype=float), np.asarray(y, dtype=float)
    out = {}
    ric = rolling_ic(s, y, sign, pc.ic_window) if {"slope", "hyst"} & set(pc.methods) else None
    for m in pc.methods:
        out[m] = (np.ones(len(s), dtype=bool) if m == "always" else
                  state_slope(ric, pc.slope_lookback) if m == "slope" else
                  state_hyst(ric, pc.slope_lookback) if m == "hyst" else
                  state_cumic(sign * s * y, pc.cumic_ma))
    return out


def applied_states(s_g, y_g, sign, pc: PortfolioConfig) -> dict:
    """State applied on every row (bool, length R) per method. ``s_g``: the factor's signal on the calendar rows
    (NaN where it has none), ``y_g``: ``y_fwd`` there. Machines run on the factor's own valid rows and the result
    is shifted LAG rows, so the state on row t only used rows <= t-2."""
    s_g, y_g = np.asarray(s_g, dtype=float), np.asarray(y_g, dtype=float)
    fin = np.isfinite(s_g)
    valid = np.flatnonzero(fin)
    raw = raw_states(s_g[valid], y_g[valid], sign, pc)
    out = {}
    for m in pc.methods:
        if m == "always":
            out[m] = fin.copy()
            continue
        full = np.zeros(len(s_g), dtype=bool)
        full[valid] = raw[m]
        app = np.zeros(len(s_g), dtype=bool)
        app[LAG:] = full[:-LAG]
        out[m] = app
    return out


# ------------------------------------------------------------------ positions (pure)
def factor_positions(s_g, buffer) -> np.ndarray:
    """Buffered position of sign +1 over the factor's whole row history (from its first finite signal; flat
    before). Sign -1 is the exact mirror image."""
    s_g = np.asarray(s_g, dtype=float)
    p = np.zeros(len(s_g))
    f = np.flatnonzero(np.isfinite(s_g))
    if len(f):
        p[f[0]:] = positions(s_g[f[0]:], 1.0, buffer)
    return p


def portfolio_position(version, V, S, P, sign):
    """Equal-weight position over the valid factors. ``V`` (R, F) bool: valid and finite and eligible; ``S`` signals
    (R, F), ``P`` unsigned buffered positions (R, F), ``sign`` (F,). Returns (position, n_valid)."""
    V = np.asarray(V, dtype=bool)
    if version == "cont":
        votes = sign[None, :] * P
    elif version in ("binA", "binB"):
        votes = sign[None, :] * np.sign(np.nan_to_num(np.asarray(S, dtype=float)))
    else:
        raise ValueError(f"unknown position version {version!r}")
    tot = (votes * V).sum(axis=1)
    n = V.sum(axis=1)
    if version == "binB":
        return np.sign(tot), n                       # integer votes: a tie is exactly 0
    return np.divide(tot, n, out=np.zeros(len(n)), where=n > 0), n


def summarize(gross, net, pos, n_valid, ret, dates) -> dict:
    """Statistics of one variant on one row range (arrays of equal length). ``ret``: the long-only target's raw
    daily return on the same rows; the NAV is the simple-interest net P&L scaled ex post to TARGET_VOL (presentation)."""
    sd = np.std(net, ddof=1) if len(net) > 1 else float("nan")
    if np.isfinite(sd) and sd > 0:
        r = TARGET_VOL / np.sqrt(252.0) / sd * net
        nav = 1.0 + np.cumsum(r)
        dd, cum = float(np.max(np.maximum.accumulate(nav) - nav)), float(r.sum())
    else:
        dd = cum = float("nan")
    return {"start": str(dates[0].date()), "end": str(dates[-1].date()),
            "years": float((dates[-1] - dates[0]).days / 365.25), "n_rows": int(len(net)),
            "sharpe_net": _sharpe(net), "sharpe_gross": _sharpe(gross), "max_dd_10vol": dd, "cum_pnl_10vol": cum,
            "avg_abs_dpos": float(np.mean(np.abs(np.diff(pos)))) if len(pos) > 1 else float("nan"),
            "avg_n_valid": float(np.mean(n_valid)), "share_flat": float(np.mean(pos == 0)),
            "corr_target": _corr(net, ret), "target_sharpe": _sharpe(ret)}


# ------------------------------------------------------------------ loading one target
def master_calendar(ctxs) -> dict:
    """The target's calendar, y_fwd and sigma. All runs must agree on the target series (on their common dates),
    on cost / buffer / vol half-life, and on the row of every y_fwd they use; otherwise fail loud."""
    base = max(ctxs, key=lambda c: (len(c["cal"]), c["cal"][-1]))
    tgt = base["panel"].target
    cfg0 = base["cfg"]
    for c in ctxs:
        t = c["panel"].target
        if not t.index.isin(tgt.index).all():
            raise ValueError(f"run {c['name']}: target dates are not a subset of the longest run's ({base['name']})")
        if not np.allclose(tgt.reindex(t.index).to_numpy(float), t.to_numpy(float), rtol=0, atol=1e-14,
                           equal_nan=True):
            raise ValueError(f"run {c['name']}: target series differs from run {base['name']}")
        for k, a, b in (("cost_per_side", c["cfg"].target.cost_per_side, cfg0.target.cost_per_side),
                        ("buffer", c["cfg"].position.buffer, cfg0.position.buffer),
                        ("vol_halflife", c["cfg"].signals.vol_halflife, cfg0.signals.vol_halflife)):
            if a != b:
                raise ValueError(f"run {c['name']}: {k} = {a} differs from run {base['name']} ({b})")
    y, sigma = forward_target(tgt, cfg0.signals.vol_halflife)
    rows = np.flatnonzero(np.isfinite(y.to_numpy(dtype=float)))
    row_of = np.full(len(tgt), -1)
    row_of[rows] = np.arange(len(rows))
    m = {"cal": tgt.index, "rows": rows, "row_of": row_of, "target": tgt, "dates": tgt.index[rows],
         "y": y.to_numpy(dtype=float)[rows], "sigma": sigma.to_numpy(dtype=float)[rows],
         "cost": float(cfg0.target.cost_per_side), "buffer": float(cfg0.position.buffer),
         "vol_halflife": int(cfg0.signals.vol_halflife)}
    for c in ctxs:
        ridx = place_rows(m, c)
        if not np.allclose(m["y"][ridx], c["y"], rtol=1e-9, atol=0):
            raise ValueError(f"run {c['name']}: y_fwd differs from the longest run's on common rows")
    return m


def place_rows(m, ctx) -> np.ndarray:
    """Master row index of every row of one run (its ``prep["rows"]``)."""
    mpos = m["cal"].get_indexer(ctx["cal"][ctx["prep"]["rows"]])
    ridx = m["row_of"][mpos]
    if (mpos < 0).any() or (ridx < 0).any():
        raise ValueError(f"run {ctx['name']}: a row of the run is missing in the target calendar")
    return ridx


def place_factor(m, ctx, j) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(signal, eligible_full, eligible_oos) of column ``j`` of one run on the master rows. A run whose OOS is not
    open only has its IS + OBS rows: the signal is NaN everywhere else (nothing beyond is read) and there is no
    OOS eligibility."""
    prep, ridx = ctx["prep"], place_rows(m, ctx)
    R = len(m["rows"])
    if ctx["opened"]:
        keep, oos = np.isfinite(ctx["S_np"][:, j]), prep["oos_mask"][:, j]
    else:
        keep, oos = prep["is_mask"][:, j] | prep["obs_mask"][:, j], np.zeros(len(ridx), dtype=bool)
    s = np.full(R, np.nan)
    s[ridx[keep]] = ctx["S_np"][keep, j]
    eo = np.zeros(R, dtype=bool)
    eo[ridx[oos]] = True
    return s, np.isfinite(s), eo & np.isfinite(s)


def load_target(run_dirs) -> dict:
    """Final-layer factors of one target (same pooling and layers as the dashboard) on the master calendar."""
    ctxs = [_load_run(p, i) for i, p in enumerate(run_dirs)]
    slugs = {slugify(c["run"]["target"]) for c in ctxs}
    if len(slugs) != 1:
        raise ValueError(f"run directories belong to different targets: {sorted(slugs)}")
    pool, rep_of, _, skipped, n_merged, _, _ = pool_layers(ctxs)
    final = [e for e in pool if factor_stage(e["row"], rep_of.get(e["key"]) == e["key"]) == 4]
    m = master_calendar(ctxs)
    factors = []
    for e in final:
        ctx, r = e["ctx"], e["row"]
        s, ef, eo = place_factor(m, ctx, e["j"])
        factors.append({"id": e["id"], "title": e["title"], "series": r["series"], "transform": r["transform"],
                        "group": r["group"], "sign": float(e["sign"]), "run": ctx["name"], "opened": ctx["opened"],
                        "s": s, "elig_full": ef, "elig_oos": eo})
    return {"target": ctxs[0]["run"]["target"], "slug": slugs.pop(), "m": m, "factors": factors,
            "n_merged": int(n_merged), "runs": [fingerprint(c, m) for c in ctxs]}


def fingerprint(ctx, m) -> dict:
    d = ctx["dir"]
    return {"run": ctx["name"], "oos_opened": bool(ctx["opened"]), "panel_info": ctx["run"].get("panel_info", {}),
            "candidates_sha256": hashlib.sha256((d / "candidates.csv").read_bytes()).hexdigest(),
            "execution_delay": int(ctx["cfg"].position.execution_delay),
            "target_sha256": hashlib.sha256(ctx["panel"].target.to_numpy(float).tobytes()).hexdigest()}


# ------------------------------------------------------------------ computing one target
def compute_target(td, pc: PortfolioConfig) -> dict:
    """States, positions, P&L and statistics of every method x position x sample of one loaded target."""
    m, fs = td["m"], td["factors"]
    R, F = len(m["y"]), len(fs)
    if F == 0:
        raise ValueError(f"{td['target']}: no final-layer factor")
    S = np.column_stack([f["s"] for f in fs])
    sign = np.array([f["sign"] for f in fs])
    P = np.column_stack([factor_positions(f["s"], m["buffer"]) for f in fs])
    fin = np.isfinite(S)
    app = {k: np.column_stack([applied_states(f["s"], m["y"], f["sign"], pc)[k] for f in fs]) for k in pc.methods}
    ret = m["y"] * m["sigma"]
    E = {"full": np.column_stack([f["elig_full"] for f in fs]), "oos": np.column_stack([f["elig_oos"] for f in fs])}
    variants, summary = {}, []
    for sample in SAMPLES:
        any_ = (E[sample] & fin).any(axis=1)
        if not any_.any():
            continue
        a, b = int(np.argmax(any_)), int(R - 1 - np.argmax(any_[::-1]))
        sl = slice(a, b + 1)
        for k in pc.methods:
            V = app[k] & fin & E[sample]
            for v in pc.positions:
                pos, n = portfolio_position(v, V, S, P, sign)
                gross, net = simulate(pos[sl], m["y"][sl], m["sigma"][sl], m["cost"])
                variants[(k, v, sample)] = {"a": a, "b": b, "position": pos[sl], "n_valid": n[sl], "gross": gross,
                                            "net": net}
                summary.append({"method": k, "position_version": v, "sample": sample,
                                **summarize(gross, net, pos[sl], n[sl], ret[sl], m["dates"][sl])})
    return {"S": S, "P": P, "sign": sign, "applied": app, "elig": E, "variants": variants, "summary": summary,
            "ret": ret, "stats": factor_state_stats(td, app, E, pc)}


def factor_state_stats(td, app, E, pc) -> dict:
    """Per method: state switches per factor-year and share of factor-days valid, on each sample's eligible rows;
    the list of state changes and each factor's latest state."""
    m, fs = td["m"], td["factors"]
    dates = m["dates"]
    out = {"switches_per_year": {}, "valid_share": {}, "changes": [], "current": []}
    for k in pc.methods:
        for sample in SAMPLES:
            n_sw = n_days = n_val = 0
            for i, f in enumerate(fs):
                e = np.flatnonzero(E[sample][:, i])
                st = app[k][e, i]
                n_days += len(e)
                n_val += int(st.sum())
                n_sw += int((st[1:] != st[:-1]).sum())
            out["valid_share"].setdefault(sample, {})[k] = n_val / n_days if n_days else float("nan")
            out["switches_per_year"].setdefault(sample, {})[k] = n_sw / (n_days / 252.0) if n_days else float("nan")
    for i, f in enumerate(fs):
        e = np.flatnonzero(E["full"][:, i])
        valid = np.flatnonzero(np.isfinite(f["s"]))
        ric = rolling_ic(f["s"][valid], m["y"][valid], f["sign"], pc.ic_window)
        cur = {"id": f["id"], "title": f["title"], "series": f["series"], "transform": f["transform"],
               "group": f["group"], "sign": int(f["sign"]), "as_of": str(dates[valid[-1]].date()),
               "latest_ric": float(ric[-1]) if len(ric) else float("nan")}
        for k in pc.methods:
            st = app[k][e, i]
            prev = np.r_[False, st[:-1]]
            ch = np.flatnonzero(st != prev)
            out["changes"] += [{"id": f["id"], "method": k, "date": str(dates[e[c]].date()),
                                "new_state": "有效" if st[c] else "失效"} for c in ch]
            cur[f"{k}_state"] = "有效" if st[-1] else "失效"
            cur[f"{k}_since"] = str(dates[e[ch[-1]]].date()) if len(ch) else str(dates[e[0]].date())
        out["current"].append(cur)
    return out


# ------------------------------------------------------------------ outputs
def pnl_frame(td, res) -> pd.DataFrame:
    dates = td["m"]["dates"]
    parts = []
    for (k, v, sample), d in res["variants"].items():
        parts.append(pd.DataFrame({"date": dates[d["a"]:d["b"] + 1].strftime("%Y-%m-%d"), "method": k,
                                   "position_version": v, "sample": sample, "n_valid": d["n_valid"],
                                   "position": d["position"], "gross": d["gross"], "net": d["net"]}))
    return pd.concat(parts, ignore_index=True)


def _f(x, n=2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def render_report(results: dict, pc: PortfolioConfig) -> str:
    L = ["# 因子组合报告", "",
         "每个标的：最终层因子（与看板「去重后留下」一致，不论样本外结论）→ 每因子逐日 有效 / 失效 状态机 → 只持有效因子的等权组合。"
         "损益由 `engine.simulate`（vectorbt）按组合净仓位一次计算，成本只对净仓位收取。", "",
         "## 口径", "",
         f"- 时点：状态只用行 ≤ t−2 的 `y_fwd`（状态机在因子自己的有效行上算，整体后移 {LAG} 行；基线 always 不后移）。",
         f"- 滚动 IC 窗口 {pc.ic_window} 行（带方向符号）；L = {pc.slope_lookback}；累计 IC 均线 {pc.cumic_ma} 行。"
         "参数事先固定，没有调参。",
         "- 方法一：`ric[t] > ric[t−L]` 则有效。方法二：失效→有效需 `ric>0` 且 `ric[t]>ric[t−L]`；有效→失效需 `ric<0` 且 `ric[t]<ric[t−L]`；"
         "否则保持，初始失效。方法三：累计 IC 曲线站上其 126 行均线则有效。输入缺失一律失效。",
         "- 仓位：连续版 = 各有效因子（带缓冲带的）仓位等权平均；二元·因子级 = 各有效因子 sign(信号) 的等权平均；二元·组合级 = 前者的符号。",
         "- 样本 `full`：每个最终层因子在自己的全部行上都合格，**包含挑选因子所用的数据，有选择偏差（乐观偏高）**；"
         "样本 `oos`：因子只在自己的样本外段合格（纳入与样本外结论无关），状态可用其更早历史；组合从最早的样本外起点开始。",
         "- 未打开样本外（无 `confirm.json`）的 run：与看板一致，其样本外行不读取、不计算；该因子在 `oos` 样本中无贡献，在 `full` 中止于观察期末。",
         "- 区间对同一样本内所有方法相同；预热期（状态机尚无输入）计为空仓，计入统计。回撤与累计收益用净损益按本样本事后标定到年化 10% 波动的单利净值（仅作展示）。"
         "日期为持仓所在行的日期（损益为该日收盘到下一行收盘）。", ""]
    for slug, r in results.items():
        td, res = r["td"], r["res"]
        L += [f"## {td['target']}", "", f"最终层因子 {len(td['factors'])} 个；来源 run：{', '.join(x['run'] for x in td['runs'])}。", ""]
        for sample in SAMPLES:
            rows = [x for x in res["summary"] if x["sample"] == sample]
            if not rows:
                L += [f"### 样本 {sample}", "", "无合格因子行。", ""]
                continue
            x0 = rows[0]
            L += [f"### 样本 {sample}：{x0['start']} 至 {x0['end']}（{x0['years']:.1f} 年，{x0['n_rows']} 行；"
                  f"做多标的 Sharpe {_f(x0['target_sharpe'])}）", "",
                  "| 方法 | 仓位 | 净 Sharpe | 毛 Sharpe | 最大回撤 | 累计收益 | 日均仓位变化 | 平均有效因子数 | 空仓占比 | 与标的相关 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for x in rows:
                L.append(f"| {METHOD_TEXT[x['method']]} | {POSITION_TEXT[x['position_version']]} | {_f(x['sharpe_net'])} | "
                         f"{_f(x['sharpe_gross'])} | {_f(x['max_dd_10vol'], 3)} | {_f(x['cum_pnl_10vol'], 3)} | "
                         f"{_f(x['avg_abs_dpos'], 3)} | {_f(x['avg_n_valid'], 1)} | {_f(x['share_flat'], 2)} | "
                         f"{_f(x['corr_target'], 2)} |")
            L.append("")
        st = res["stats"]
        L += ["### 状态切换与有效占比", "", "| 方法 | 全样本切换次数 / 因子·年 | 样本外切换次数 / 因子·年 | 全样本有效占比 | 样本外有效占比 |", "|---|---|---|---|---|"]
        for k in pc.methods:
            L.append(f"| {METHOD_TEXT[k]} | {_f(st['switches_per_year']['full'][k])} | {_f(st['switches_per_year']['oos'][k])} | "
                     f"{_f(st['valid_share']['full'][k])} | {_f(st['valid_share']['oos'][k])} |")
        L.append("")
    L += ["## 注意", "",
          "- `full` 样本含挑选因子所用的数据，选择偏差使其偏乐观；看效果应以 `oos` 为准。",
          "- 多个方法、多种仓位版本在同一份数据上比较，表现最好的那个不是一次发现，不能据此当作已被证明有效。",
          "- 样本外段较短（各因子约占自己有效行的 15%），夏普的抽样误差大，方法之间的差异多半在噪声范围内。",
          "- 最终层因子由挑选过程选出，`oos` 中每个因子日对该因子的挑选而言是样本外，但不同因子之间的信息有重叠（同一段行情）。",
          "- 基线 always 是对照：状态机方法只有在样本外稳定优于基线时才算增加了价值；表中数字未做任何调整。", ""]
    return "\n".join(L)


def run_portfolio(config_path, run_dirs, out_dir) -> dict:
    pc = load_portfolio_config(config_path)
    groups: dict[str, list[str]] = {}
    for p in run_dirs:
        groups.setdefault(slugify(_read_run(Path(p))["target"]), []).append(str(p))
    results = {}
    for slug, ps in groups.items():                                  # everything is computed before anything is written
        td = load_target(ps)
        results[slug] = {"td": td, "res": compute_target(td, pc)}
        print(f"[portfolio] {td['target']}: {len(td['factors'])} final-layer factors, "
              f"{len(results[slug]['res']['variants'])} variants")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for slug, r in results.items():
        td, res = r["td"], r["res"]
        pnl_frame(td, res).to_csv(out / f"pnl_{slug}.csv", index=False)
        pd.DataFrame(res["summary"], columns=SUMMARY_COLUMNS).to_csv(out / f"summary_{slug}.csv", index=False)
        pd.DataFrame(res["stats"]["changes"], columns=["id", "method", "date", "new_state"]).to_csv(
            out / f"state_changes_{slug}.csv", index=False)
        pd.DataFrame(res["stats"]["current"]).to_csv(out / f"current_{slug}.csv", index=False)
        m = td["m"]
        meta[slug] = {"target": td["target"], "settings": {"cost_per_side": m["cost"], "buffer": m["buffer"],
                                                           "vol_halflife": m["vol_halflife"]},
                      "target_sha256": hashlib.sha256(m["target"].to_numpy(float).tobytes()).hexdigest(),
                      "runs": td["runs"], "n_merged": td["n_merged"],
                      "factors": [{"id": f["id"], "title": f["title"], "sign": int(f["sign"]), "run": f["run"],
                                   "oos_opened": f["opened"], "n_rows": int(f["elig_full"].sum()),
                                   "n_oos_rows": int(f["elig_oos"].sum())} for f in td["factors"]],
                      "summary": res["summary"], "switches_per_factor_year": res["stats"]["switches_per_year"],
                      "valid_share": res["stats"]["valid_share"]}
    doc = {"created_at": datetime.now(timezone.utc).isoformat(), "config": str(config_path),
           "parameters": {**pc.to_dict(), "lag_rows": LAG, "target_vol": TARGET_VOL}, "targets": meta}
    (out / "portfolio.json").write_text(json.dumps(_jsonable(doc), indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(render_report(results, pc), encoding="utf-8")
    return results
