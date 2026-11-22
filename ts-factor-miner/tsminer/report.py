"""Markdown report writer. Reads only files already present in the run directory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _num(x, nd) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{x:.{nd}f}" if np.isfinite(x) else "-"


def _cell(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).replace("|", "\\|").replace("\n", " ")


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    for c in ("flags", "description", "rep"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


def _cand_table(df: pd.DataFrame) -> str:
    headers = ["id", "description", "t", "IC", "p_fwer", "q_bh", "verdict", "post_ic", "P(IC>0)",
               "Sharpe 毛/净", "flags"]
    rows = [[r["id"], r["description"], _num(r["t"], 2), _num(r["ic"], 4), _num(r["p_fwer"], 3),
             _num(r["q_bh"], 3), r["verdict"], _num(r["post_ic"], 4), _num(r["p_pos"], 3),
             f"{_num(r['sharpe_gross'], 2)} / {_num(r['sharpe_net'], 2)}", r["flags"]]
            for _, r in df.iterrows()]
    return _md_table(headers, rows)


def write_report(run_dir) -> None:
    d = Path(run_dir)
    run = json.loads((d / "run.json").read_text())
    short = _read_csv(d / "shortlist.csv")
    watch = _read_csv(d / "watchlist.csv")
    groups = _read_csv(d / "groups.csv")
    excl = _read_csv(d / "panel" / "excluded.csv")
    info, dates, null, omni = run["panel_info"], run["dates"], run["null"], run["omnibus"]
    vc = run["verdict_counts"]
    n_pass, n_weak = vc.get("PASS", 0), vc.get("WEAK_PASS", 0)
    t_var = null["t_var"]

    L = [f"# 因子挖掘报告：{run['target']}", ""]
    L += ["## 结论", ""]
    head = (f"族大小 m = {run['family_size']}，PASS {n_pass} 个，WEAK_PASS {n_weak} 个，"
            f"shortlist {len(run['shortlist'])} 个，族 omnibus p = {_num(omni['p'], 3)}。")
    if n_pass + n_weak == 0:
        head += "没有候选通过多重检验。"
    elif n_pass == 0:
        head += "没有候选通过 FWER 检验（无 PASS），仅有 FDR 层面的 WEAK_PASS。"
    L += [head, ""]

    L += ["## 数据与样本", ""]
    L += [f"- 数据库版本：{info.get('db_version')}，发布时间：{info.get('published_at')}",
          f"- 样本：{dates['sample_start']} 至 {dates['sample_end']}（{run['n_rows']} 行）",
          f"- train 用到的最后一个收益日：{dates['train_end']}；holdout 起点：{dates['holdout_start']}",
          f"- train 行数：{run['n_train_rows']}；holdout 行数：{run['n_holdout_rows']}（holdout 未被打开）",
          f"- 序列数：{run['n_series']}；候选数（序列 x 变换）：{run['n_candidates']}；"
          f"t 无效而未进入族的候选：{run['n_dropped']}"]
    if len(excl):
        cnt = excl["reason"].value_counts()
        L.append(f"- 被剔除的序列：{int(cnt.sum())} 个（" + "，".join(f"{k}: {v}" for k, v in cnt.items()) + "）")
    else:
        L.append("- 被剔除的序列：0 个")
    L.append("")

    L += ["## 多重检验门槛", ""]
    L += [f"- 族大小 m = {run['family_size']}",
          f"- FWER 门槛：|t| >= {_num(null['hurdle'], 2)}（循环平移零分布 single-step maxT，n_boot = {null['n_boot']}）",
          f"- 零分布 t 方差：{_num(t_var, 2)}（p_cal 在其大于 1 时按其平方根保守校准）"]
    if t_var is not None and abs(t_var - 1.0) > 0.1:
        L.append(f"- 警告：零分布 t 方差偏离 1 超过 0.1（{_num(t_var, 2)}），HAC t 的尺度不准，BH 结果请谨慎解读。")
    L += [f"- 族 omnibus：mean t² = {_num(omni['mean_t2'], 2)}，p = {_num(omni['p'], 3)}",
          f"- eb_tau2 = {_num(run['eb_tau2'], 6)}：经验贝叶斯估计的族内平均真实 IC²；为 0 表示 t 的分散度与零分布无法区分，"
          f"不定义组合因子（composite_defined = {run['composite_defined']}）。", ""]

    L += ["## Shortlist", ""]
    L += [_cand_table(short) if len(short) else "没有候选进入 shortlist。", ""]
    L += ["## Watchlist（未必通过多重检验，仅供参考）", ""]
    L += [_cand_table(watch) if len(watch) else "watchlist 为空。", ""]

    L += ["## 分组 omnibus", ""]
    if len(groups):
        g = groups.sort_values("p")
        L.append(_md_table(["group", "n_cand", "mean t²", "null mean t²", "p", "p_holm"],
                           [[r["group"], int(r["n_cand"]), _num(r["mean_t2"], 2), _num(r["null_mean_t2"], 2),
                             _num(r["p"], 3), _num(r["p_holm"], 3)] for _, r in g.iterrows()]))
    L.append("")

    L += ["## 提示与局限", "",
          "- 默认滞后是保守假设：不假设 x 的取值时刻早于标的收盘，真实可用时间可能更早。",
          "- 月频、季频及不规则序列未纳入。",
          "- 数据库的 release_date 未使用（历史值不可靠），可用性完全由配置中的 lag 决定。",
          "- spot 标的的收益不含 carry。",
          "- holdout 只能看一次：confirm 之后不应再据其结果回头调整候选或参数。", ""]
    L += ["## 下一步", "",
          f"确认 shortlist 请运行（只能运行一次）：`python -m tsminer confirm {d}`", ""]
    (d / "report.md").write_text("\n".join(L), encoding="utf-8")


def append_confirm(run_dir) -> None:
    d = Path(run_dir)
    c = json.loads((d / "confirm.json").read_text())
    df = _read_csv(d / "confirm.csv")
    L = ["", "## Holdout 确认", ""]
    L += [f"- holdout 区间：{c['holdout_start']} 至 {c['holdout_end']}，{c['n_holdout_rows']} 行"
          + ("（强制重跑）" if c.get("forced") else ""), ""]
    if len(df):
        L.append(_md_table(["id", "sign_train", "t_hold", "t_signed", "IC", "p_one", "p_holm", "结论",
                            "Sharpe 毛/净"],
                           [[r["id"], int(r["sign_train"]), _num(r["t_hold"], 2), _num(r["t_signed"], 2),
                             _num(r["ic"], 4), _num(r["p_one"], 3), _num(r["p_holm"], 3), r["verdict"],
                             f"{_num(r['sharpe_gross'], 2)} / {_num(r['sharpe_net'], 2)}"]
                            for _, r in df.iterrows()]))
    else:
        L.append("shortlist 为空，没有逐候选的确认。")
    comp = c.get("composite")
    L.append("")
    if comp:
        L.append(f"组合因子（权重为 train 上定死的 post_ic）：holdout t = {_num(comp['t'], 2)}，"
                 f"单边 p = {_num(comp['p'], 3)}，结论 {comp['verdict']}，"
                 f"Sharpe 毛/净 = {_num(comp['sharpe_gross'], 2)} / {_num(comp['sharpe_net'], 2)}。")
    else:
        L.append("组合因子未定义（eb_tau2 = 0）。")
    L.append("")
    with open(d / "report.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))
