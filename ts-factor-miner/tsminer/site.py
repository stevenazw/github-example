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

from .config import load_config
from .databank import Panel
from .pipeline import _corr, _jsonable, positions, prepare

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
    lines.append(f"pos[t] = {sg} × s[t] ÷ σ[t]")
    lines.append(f"σ[t]：标的日收益的 EWMA 波动率（半衰期 {int(cfg.signals.vol_halflife)} 日）；"
                 f"目标仓位变化超过缓冲带 {_g(cfg.position.buffer)} 才调仓")
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
    notional = pos / sig
    net = pos * y - cost * np.abs(np.diff(notional, prepend=0.0))
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
def _read_run(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def build_asset_data(run_dir) -> dict:
    d = Path(run_dir)
    run = _read_run(d)
    target = run["target"]
    slug = slugify(target)
    cfg = load_config(d / "config.toml")
    panel = Panel.load(d / "panel")
    opened = (d / "confirm.json").exists()

    verdicts, oos_info = {}, {}
    if opened:
        cdf = pd.read_csv(d / "confirm.csv") if (d / "confirm.csv").exists() else pd.DataFrame()
        for _, r in cdf.iterrows():
            verdicts[r["id"]] = str(r["verdict"])
            oos_info[r["id"]] = _jsonable({k: r[k] for k in ("n", "oos_start", "oos_end", "t_signed", "p_one",
                                                                "ic_oos_signed", "retention_oos", "sharpe_net")
                                           if k in r.index})
    short = pd.read_csv(d / "shortlist.csv")
    for c in ("flags", "description"):
        if c in short.columns:
            short[c] = short[c].fillna("").astype(str)

    prep = prepare(panel, cfg)
    S, rows = prep["S"], prep["rows"]
    cal = panel.target.index
    y_all = prep["y_fwd"].to_numpy(dtype=float)[rows]
    sig_all = prep["sigma"].to_numpy(dtype=float)[rows]
    S_np = S.to_numpy(dtype=float)[rows]
    col_of = {c: i for i, c in enumerate(S.columns)}
    is_m, obs_m, oos_m = prep["is_mask"], prep["obs_mask"], prep["oos_mask"]
    real_dates = np.asarray(cal[rows + 1].strftime("%Y-%m-%d"))    # return of row p is realised on row p+1

    factors, skipped = [], []
    for _, r in short.iterrows():
        cid = r["id"]
        if cid not in col_of:
            skipped.append({"id": cid, "reason": "信号不在候选池中"})
            continue
        j = col_of[cid]
        mask = is_m[:, j] | obs_m[:, j] | (oos_m[:, j] if opened else False)
        idx = np.flatnonzero(mask)
        seg = np.where(is_m[idx, j], 0, np.where(obs_m[idx, j], 1, 2))
        sign = float(r["sign"])
        res = factor_nav(S_np[idx, j], y_all[idx], sig_all[idx], seg, real_dates[idx], sign,
                         cfg.position.buffer, cfg.target.cost_per_side)
        if res is None:
            skipped.append({"id": cid, "reason": "样本内净收益波动为 0 或无法计算，无法标定净值"})
            continue
        segments = res["segments"]
        if not opened:
            segments[2] = {"seg": 2, "name": SEG_NAMES[2], "opened": False}
        keep = downsample_idx(seg)
        meta_row = panel.meta.loc[r["series"]]
        desc = r["description"] or str(r["series"])
        verdict = verdicts.get(cid) if opened else None
        factors.append(_jsonable({
            "id": cid, "series": r["series"], "transform": r["transform"],
            "title": f"{desc} · {transform_label(r['transform'])}", "description": desc,
            "group": r["group"], "sign": int(sign), "direction": direction_text(sign, target),
            "formula": formula_text(meta_row, r["transform"], cfg, sign),
            "z": float(r["z"]), "t_dev": float(r["t_dev"]), "cluster_size": int(r["cluster_size"]),
            "flags": flag_texts(r["flags"]), "oos_verdict": verdict,
            "oos_confirm": oos_info.get(cid) if opened else None,
            "segments": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in sg.items()}
                         for sg in segments],
            "nav": {"dates": [str(x) for x in real_dates[idx][keep]],
                    "values": [round(float(v), 4) for v in res["nav"][keep]],
                    "seg": [int(v) for v in seg[keep]]},
        }))

    def sort_key(f):
        v = VERDICT_ORDER.get(f["oos_verdict"], 4) if opened else 4
        t = f["t_dev"] if f["t_dev"] is not None else float("-inf")
        return (v, -t)
    factors.sort(key=sort_key)

    vc = pd.Series([f["oos_verdict"] for f in factors if f["oos_verdict"]], dtype=object).value_counts()
    cfg_d = run["config"]
    vd = cfg_d["verdict"]
    criteria = {"z_weak": vd["z_weak"], "obs_min_retention": vd["obs_min_retention"]}
    if "min_net_sharpe" in vd:                       # runs made before the tradability screen lack it
        criteria["min_net_sharpe"] = vd["min_net_sharpe"]
    return {
        "target": target, "slug": slug,
        "db_published_at": run["panel_info"].get("published_at"),
        "sample_start": run["dates"]["sample_start"], "sample_end": run["dates"]["sample_end"],
        "is_frac": cfg_d["sample"]["is_frac"], "obs_frac": cfg_d["sample"]["obs_frac"],
        "oos_opened": bool(opened), "funnel": run["funnel"], "criteria": criteria,
        "oos_counts": ({"confirmed": int(vc.get("CONFIRMED", 0)), "consistent": int(vc.get("CONSISTENT", 0)),
                        "failed": int(vc.get("FAILED", 0)), "insufficient": int(vc.get("INSUFFICIENT", 0))}
                       if opened else None),
        "skipped": skipped, "factors": factors,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------- site
def build_site(run_dirs, out_dir) -> dict:
    run_dirs = [Path(p) for p in run_dirs]
    seen: dict[str, Path] = {}
    for p in run_dirs:
        t = _read_run(p)["target"]
        slug = slugify(t)
        if slug in seen:
            raise ValueError(f"duplicate target {t!r}: {seen[slug]} and {p}")
        seen[slug] = p
    out = Path(out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    assets = []
    for slug, p in seen.items():
        data = build_asset_data(p)
        (out / "data" / f"{slug}.json").write_text(
            json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
        assets.append({"target": data["target"], "slug": slug, "n_factors": len(data["factors"]),
                       "n_skipped": len(data["skipped"]), "oos_opened": data["oos_opened"],
                       "sample_end": data["sample_end"], "db_published_at": data["db_published_at"],
                       "funnel": data["funnel"], "oos_counts": data["oos_counts"]})
        print(f"[site] {data['target']}: {len(data['factors'])} factors, oos_opened={data['oos_opened']}")
    head, body = render_page({"assets": assets})
    (out / "artifact.html").write_text(head + body, encoding="utf-8")
    (out / "index.html").write_text(
        "<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        + head + "</head>\n<body>\n" + body + "</body>\n</html>\n", encoding="utf-8")
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
.funnel { display: flex; flex-wrap: wrap; gap: 4px 6px; align-items: baseline; }
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
  font-size: 12px; line-height: 1.6; overflow-x: auto; max-width: 100%; white-space: pre; }
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
  if (!a.oos_opened) return '样本外未打开';
  var c = a.oos_counts || {};
  return '确认 ' + c.confirmed + ' · 一致 ' + c.consistent + ' · 失败 ' + c.failed + (c.insufficient ? ' · 样本不足 ' + c.insufficient : '');
}

function renderOverview() {
  var root = h('div');
  root.appendChild(h('h1', {text: '因子跟踪看板'}));
  root.appendChild(h('p', {class: 'sub', text: '每个标的资产一个页面：展示通过检测的因子（shortlist）的净值走势、公式与交易方向。点击一行进入。'}));
  if (!MANIFEST.assets.length) { root.appendChild(h('div', {class: 'empty', text: '没有资产。'})); return root; }
  var hasTrade = MANIFEST.assets.some(function (a) { return a.funnel.trade_keep != null; });
  var head = ['标的', '数据截止日', '检验', '样本内通过', '观察期保留'].concat(hasTrade ? ['可交易'] : [], ['去重后留下', '样本外']);
  var numCols = hasTrade ? 6 : 5;
  var tb = h('tbody');
  MANIFEST.assets.forEach(function (a) {
    var f = a.funnel;
    var tr = h('tr', {onclick: function (e) { if (e.target.tagName !== 'A') location.hash = '#' + a.slug; }}, [
      h('td', null, [h('a', {href: '#' + a.slug, text: a.target})]),
      h('td', {class: 'mono', text: a.sample_end}),
      h('td', {class: 'n', text: fint(f.tested)}),
      h('td', {class: 'n', text: fint(f.is_pass + f.is_weak)}),
      h('td', {class: 'n', text: fint(f.obs_keep + f.obs_insufficient)}),
      hasTrade ? h('td', {class: 'n', text: fint(f.trade_keep)}) : null,
      h('td', {class: 'n', text: fint(f.shortlist)}),
      h('td', {text: oosText(a)})
    ]);
    tb.appendChild(tr);
  });
  var tbl = h('table', {class: 'ov'}, [h('thead', null, [h('tr', null, head.map(function (t, i) { return h('th', {class: (i >= 2 && i <= numCols) ? 'n' : null, text: t}); }))]), tb]);
  root.appendChild(h('div', {class: 'scroll'}, [tbl]));
  root.appendChild(h('p', {class: 'muted', style: 'margin-top:14px', text: '漏斗：检验 → 样本内通过（PASS + WEAK_PASS）→ 观察期保留 → ' + (hasTrade ? '可交易（扣成本净 Sharpe 达标）→ ' : '') + '去重后留下 → 样本外确认 / 一致 / 失败。样本外只在运行 confirm 之后才显示。'}));
  return root;
}

function funnelEl(a) {
  var f = a.funnel, parts = [['检验', f.tested], ['样本内通过', f.is_pass + f.is_weak], ['观察期保留', f.obs_keep + f.obs_insufficient]];
  if (f.trade_keep != null) parts.push(['可交易', f.trade_keep]);
  parts.push(['去重后留下', f.shortlist]);
  var el = h('div', {class: 'funnel'});
  parts.forEach(function (p, i) {
    if (i) el.appendChild(h('span', {class: 'arr', text: '→'}));
    el.appendChild(h('span', null, [p[0] + ' ', h('b', {text: String(p[1])})]));
  });
  el.appendChild(h('span', {class: 'arr', text: '→'}));
  if (a.oos_opened) {
    var c = a.oos_counts;
    el.appendChild(h('span', null, ['确认 ', h('b', {text: String(c.confirmed)}), ' · 一致 ', h('b', {text: String(c.consistent)}), ' · 失败 ', h('b', {text: String(c.failed)}), (c.insufficient ? ' · 样本不足 ' : ''), (c.insufficient ? h('b', {text: String(c.insufficient)}) : null)]));
  } else el.appendChild(h('span', {class: 'muted', text: '样本外未打开'}));
  return el;
}

function criteriaEl(a) {
  var c = a.criteria;
  if (!c || c.z_weak == null || c.obs_min_retention == null) return null;
  var txt = '样本内 |z| ≥ ' + c.z_weak + '；观察期方向一致且 IC 不低于样本内的 ' + (Math.round(c.obs_min_retention * 1000) / 10) + '%';
  if (c.min_net_sharpe != null) txt += '；扣成本后净 Sharpe ≥ ' + c.min_net_sharpe + '（样本内+观察期）';
  return dl('入选条件', [txt], 'wide');
}

function legendEl(opened) {
  return h('div', {class: 'legend'}, [
    h('span', null, [h('i', {class: 'key is'}), '样本内']),
    h('span', null, [h('i', {class: 'key obs'}), '观察期']),
    opened ? h('span', null, [h('i', {class: 'key oos'}), '样本外']) : h('span', {class: 'muted'}, [h('i', {class: 'key off'}), '样本外未打开'])
  ]);
}

function dl(label, kids, cls) { return h('div', {class: cls}, [h('dt', {text: label}), h('dd', null, kids)]); }

function renderAsset(a) {
  var root = h('div');
  root.appendChild(h('h1', {text: a.target + ' 因子跟踪'}));
  root.appendChild(h('p', {class: 'sub', text: '通过检测的因子 ' + a.factors.length + ' 个' + (a.skipped.length ? '，另有 ' + a.skipped.length + ' 个无法标定净值而未出图' : '') + '。'}));
  var sum = h('dl', {class: 'summary'}, [
    dl('数据截止日', [h('span', {class: 'mono', text: a.sample_end})]),
    dl('标的样本', [h('span', {class: 'mono', text: a.sample_start + ' 至 ' + a.sample_end})]),
    dl('数据库发布', [h('span', {class: 'mono', text: (a.db_published_at || '–').slice(0, 10)})]),
    dl('切分规则', ['每个因子在自身有效数据内按 ' + Math.round(a.is_frac * 100) + '/' + Math.round(a.obs_frac * 100) + '/' + Math.round((1 - a.is_frac - a.obs_frac) * 100) + ' 切成样本内 / 观察期 / 样本外']),
    dl('净值定义', ['单利累计；扣除交易成本；样本内年化波动标定为 10%']),
    dl('三段图例', [legendEl(a.oos_opened)]),
    dl('筛选漏斗', [funnelEl(a)], 'wide'),
    criteriaEl(a)
  ]);
  root.appendChild(sum);
  var bar = h('div', {class: 'filters', role: 'group', 'aria-label': '按样本外结论筛选'});
  var list = h('div', {class: 'grid'});
  var cnt = {ALL: a.factors.length};
  a.factors.forEach(function (f) { var k = f.oos_verdict || 'NONE'; cnt[k] = (cnt[k] || 0) + 1; });
  var opts = a.oos_opened ? [['ALL', '全部'], ['CONFIRMED', '确认'], ['CONSISTENT', '一致'], ['FAILED', '失败'], ['INSUFFICIENT', '样本不足']] : [['ALL', '全部']];
  if (!a.oos_opened) filterKey = 'ALL';
  else if (!opts.some(function (o) { return o[0] === filterKey; })) filterKey = 'ALL';
  function fill() {
    redraws = [];
    list.textContent = '';
    var shown = a.factors.filter(function (f) { return filterKey === 'ALL' || f.oos_verdict === filterKey; });
    if (!shown.length) list.appendChild(h('div', {class: 'empty', text: a.factors.length ? '该结论下没有因子。' : '没有因子通过检测。'}));
    shown.forEach(function (f) { list.appendChild(card(f, a)); });
    shown.forEach(function (f, i) { redraws[i](); });
    Array.prototype.forEach.call(bar.children, function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-k') === filterKey)); });
  }
  opts.forEach(function (o) {
    bar.appendChild(h('button', {type: 'button', 'data-k': o[0], 'aria-pressed': String(o[0] === filterKey), onclick: function () { filterKey = o[0]; fill(); }}, [o[1], h('span', {class: 'cnt', text: String(cnt[o[0]] || 0)})]));
  });
  if (!a.oos_opened) bar.appendChild(h('span', {class: 'muted', text: '样本外未打开，暂无结论可筛选。'}));
  root.appendChild(bar);
  root.appendChild(list);
  root._after = fill;
  return root;
}

function badge(cls, shape, text, extra) {
  return h('span', {class: 'badge ' + cls + (extra ? ' ' + extra : '')}, [h('span', {class: 'shape', 'aria-hidden': 'true', text: shape}), text]);
}

function card(f, a) {
  var v = VERD[f.oos_verdict || 'NONE'] || VERD.NONE;
  var box = h('div', {class: 'chartbox'});
  var el = h('article', {class: 'card'}, [
    h('h3', {text: f.title}),
    h('div', {class: 'idline'}, [h('span', {class: 'mono', text: f.id}), ' · ' + f.group + ' · z ' + fx(f.z, 2) + (f.cluster_size > 1 ? ' · 同簇 ' + f.cluster_size + ' 个' : '')]),
    h('div', {class: 'badges'}, [
      badge('dir', f.sign > 0 ? '▲' : '▼', f.direction, 'dir'),
      badge(v.cls, v.shape, '样本外：' + v.label)
    ]),
    h('pre', {class: 'formula', tabindex: '0', 'aria-label': '因子公式', text: f.formula.join('\n')}),
    cardLegend(a.oos_opened),
    box,
    statsTable(f),
    oosNote(f),
    f.flags.length ? h('ul', {class: 'flags'}, f.flags.map(function (t) { return h('li', {text: t}); })) : null
  ]);
  redraws.push(function () { drawChart(box, f, a.oos_opened); });
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
  return h('p', {class: 'oosnote'}, ['样本外检验：t ', h('span', {class: 'num', text: fx(c.t_signed, 2)}), ' · 单边 p ', h('span', {class: 'num', text: fx(c.p_one, 3)}), ' · retention ', h('span', {class: 'num', text: fx(c.retention_oos, 2)})]);
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
