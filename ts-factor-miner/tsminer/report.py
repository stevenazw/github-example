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


def _read_meta(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, index_col=0)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _shortlist_table(df: pd.DataFrame) -> str:
    headers = ["id", "description", "样本内区间", "n_is", "z", "IC(样本内)", "IC(观察期)", "retention", "t_dev",
               "Sharpe 毛/净", "turnover", "half-life", "year_hit", "cluster", "flags"]
    rows = [[r["id"], r["description"], f"{r['is_start']} 至 {r['is_end']}", int(r["n_is"]), _num(r["z"], 2),
             _num(r["sign"] * r["ic"], 4), _num(r["ic_obs_signed"], 4), _num(r["retention"], 2),
             _num(r["t_dev"], 2), f"{_num(r['sharpe_gross'], 2)} / {_num(r['sharpe_net'], 2)}",
             _num(r["turnover"], 3), _num(r["signal_half_life"], 1), _num(r["year_hit"], 2),
             int(r["cluster_size"]), r["flags"]] for _, r in df.iterrows()]
    return _md_table(headers, rows)


def write_report(run_dir) -> None:
    d = Path(run_dir)
    run = json.loads((d / "run.json").read_text())
    short = _read_csv(d / "shortlist.csv")
    excl = _read_csv(d / "panel" / "excluded.csv")
    meta = _read_meta(d / "panel" / "meta.csv")
    info, dates, ns = run["panel_info"], run["dates"], run["n_is_summary"]
    fn = run["funnel"]
    cfg = run["config"]
    vd, sm, ps = cfg["verdict"], cfg["sample"], cfg["position"]
    zp, zw = vd["z_pass"], vd["z_weak"]
    n_is_ok = fn["is_pass"] + fn["is_weak"]
    isn = vd["is_min_net_sharpe"]

    L = [f"# 因子筛选报告：{run['target']}", ""]
    L += ["## 结论", ""]
    head = (f"检验 {fn['tested']} 个 → 样本内 |z| 达标 {fn['is_z_pass']} 个"
            f"→ 其中净 Sharpe > {isn:g} 的 {n_is_ok} 个（通过 {fn['is_pass']}、弱通过 {fn['is_weak']}）"
            f"→ 观察期保留 {fn['obs_keep'] + fn['obs_insufficient']} 个（其中观察期样本不足 {fn['obs_insufficient']} 个）"
            f"→ 可交易（净 Sharpe ≥ {vd['min_net_sharpe']:g}）保留 {fn['trade_keep']} 个"
            f"→ 去重后 shortlist {fn['shortlist']} 个。")
    if fn["shortlist"] == 0:
        head += "没有候选留下。"
    L += [head, ""]

    L += ["## 数据与样本", ""]
    L += [f"- 数据库版本：{info.get('db_version')}，发布时间：{info.get('published_at')}",
          f"- 标的样本：{dates['sample_start']} 至 {dates['sample_end']}（{run['n_rows']} 行）",
          f"- 每个候选在自己的有效数据内按 is_frac = {_num(sm['is_frac'], 2)}、obs_frac = {_num(sm['obs_frac'], 2)} "
          f"切成样本内、观察期、样本外（样本外未被打开）。",
          f"- 序列数：{run['n_series']}；候选数（序列 x 变换）：{run['n_candidates']}；"
          f"已检验：{run['n_tested']}；因样本不足未检验：{run['n_skipped']}",
          f"- 已检验候选的样本内长度 n_is：最小 {ns['min']}，中位 {_num(ns['median'], 0)}，最大 {ns['max']}"]
    if len(excl):
        cnt = excl["reason"].astype(str).str.split(":").str[0].value_counts()   # "reason: leaf, ..." -> reason
        L.append(f"- 被剔除的序列：{int(cnt.sum())} 个（" + "，".join(f"{k}: {v}" for k, v in cnt.items()) + "）")
    else:
        L.append("- 被剔除的序列：0 个")
    L.append("")

    L += ["## 可得性输入", ""]
    if len(meta) and "lag_rule" in meta.columns:
        g = meta.groupby("lag_rule").agg(lag=("lag", "max"), n=("lag", "size")).reset_index()
        g = g.sort_values(["n", "lag_rule"], ascending=[False, True])
        L.append(_md_table(["规则键", "lag", "命中的序列数"], [[r["lag_rule"], int(r["lag"]), int(r["n"])]
                                                            for _, r in g.iterrows()]))
    else:
        L.append("没有序列进入候选池。")
    L.append("")
    rej = excl[excl["reason"].astype(str).str.startswith("no_availability_rule")] if len(excl) else excl
    if len(rej):
        # formula variables carry the leaves without a rule after the reason: "var:x [FF1 Comdty/PX_LAST]"
        tk = [str(t) + (f" [{str(r).split(': ', 1)[1]}]" if ": " in str(r) else "")
              for t, r in zip(rej["ticker"], rej["reason"])]
        L.append(f"因没有可得性规则被拒收的序列：{len(tk)} 个。" + ("前 20 个：" if len(tk) > 20 else "") + "、".join(tk[:20]))
    else:
        L.append("因没有可得性规则被拒收的序列：0 个。")
    L += ["", "这些规则是输入，不是系统推断的结果：t 日收盘的信号只能用日期 ≤ t−lag 的观测，需研究者确认。", ""]

    L += ["## 判定标准", "",
          f"- 样本内（两个条件同时满足）：① |z| 达标：PASS：|z| ≥ {_num(zp, 2)}；WEAK_PASS：|z| ≥ {_num(zw, 2)}；"
          "z = HAC t ÷ 该候选自身循环平移零分布的标准差。"
          f"② 样本内行上扣成本后的年化净 Sharpe 严格大于 {isn:g}（is_min_net_sharpe；非有限视为不满足）。"
          "不满足任一条件为 REJECT（candidates.csv 的 z_verdict 记录仅按 z 的结论）。",
          f"- 观察期：对样本内非 REJECT 的候选，要求方向与样本内一致且 retention（按样本内符号的观察期 IC ÷ |样本内 IC|）"
          f"≥ {_num(vd['obs_min_retention'], 2)}；观察期样本不足时不淘汰，保留并标注 obs_short。",
          f"- 可交易性：对未被观察期淘汰的候选，要求开发样本（样本内 + 观察期）上扣成本后的年化净 Sharpe "
          f"≥ {vd['min_net_sharpe']:g}（min_net_sharpe）；在去重之前筛选，不满足（或净 Sharpe 非有限）的候选被淘汰。",
          f"- 去重：对通过可交易性筛选的候选按开发样本（样本内 + 观察期）上的 t_dev 从高到低贪心聚类，"
          f"相关系数绝对值 ≥ {_num(vd['dedupe_corr'], 2)} 归入同一代表。",
          f"- 仓位：连续比例，目标仓位 = 样本内符号 × 信号，按波动率缩放；缓冲带 buffer = {_num(ps['buffer'], 2)}，"
          "目标与当前仓位之差超过 buffer 才调仓。",
          "- 门槛不随样本长度变，同样的 z 在短样本上对应更高的 IC（IC ≈ z ÷ √n_is）：", ""]
    L.append(_md_table(["n_is", f"PASS 门槛 (z = {_num(zp, 2)}) 对应 IC", f"WEAK_PASS 门槛 (z = {_num(zw, 2)}) 对应 IC"],
                       [[n, _num(zp / np.sqrt(n), 4), _num(zw / np.sqrt(n), 4)] for n in (5000, 2000, 1000, 500)]))
    L.append("")

    L += ["## Shortlist", ""]
    if len(short):
        L += [_shortlist_table(short), "",
              "IC 均为按样本内符号的带符号 IC；Sharpe、turnover 在开发样本（样本内 + 观察期）上计算。", ""]
    else:
        L += ["没有候选进入 shortlist。", ""]

    L += ["## IC 衰减与信号自相关", ""]
    if len(short):
        L.append(_md_table(["id", "ic_h1", "ic_h2", "ic_h5", "ic_h10", "ic_h20", "ic_h60", "ac_1", "ac_5", "ac_20"],
                           [[r["id"]] + [_num(r[c], 3) for c in
                                         ("ic_h1", "ic_h2", "ic_h5", "ic_h10", "ic_h20", "ic_h60", "ac_1", "ac_5", "ac_20")]
                            for _, r in short.iterrows()]))
        L.append("")
    L += ["慢信号的 IC 在长 h 上不低，部分只是因为信号本身变化慢（看 ac），不等于信息持续。", ""]

    L += ["## 提示与局限", "",
          "- 可得性规则是输入假设：由研究者确认，系统不推断，也不使用数据库的 release_date。",
          "- 月频、季频及不规则序列未纳入。",
          "- 近月连续合约因换月跳空未纳入。",
          "- spot 标的的收益不含 carry。",
          "- 回测为描述性：成本是固定比例，缓冲带是固定参数，未做 walk-forward。",
          "- 短样本候选的观察期与样本外也短。",
          "- 样本外只能看一次：confirm 之后不应再据其结果回头调整候选或参数。", ""]
    L += ["## 下一步", "",
          f"确认 shortlist 请运行（只能运行一次）：`python -m tsminer confirm {d}`", ""]
    (d / "report.md").write_text("\n".join(L), encoding="utf-8")


def append_confirm(run_dir) -> None:
    d = Path(run_dir)
    c = json.loads((d / "confirm.json").read_text())
    df = _read_csv(d / "confirm.csv")
    L = ["", "## 样本外确认", ""]
    L += [f"- 每个 shortlist 候选在自己的样本外行上按样本内符号做单边 HAC t 检验。"
          f"p_one ≤ 0.10 为 CONFIRMED；否则样本外 IC 同向且 retention_oos ≥ obs_min_retention 为 CONSISTENT；"
          f"其余为 FAILED；样本不足为 INSUFFICIENT" + ("（强制重跑）。" if c.get("forced") else "。"),
          f"- 共检验 {c['n_tested']} 个：CONFIRMED {c['n_confirmed']} 个，CONSISTENT {c['n_consistent']} 个，"
          f"FAILED {c['n_failed']} 个，INSUFFICIENT {c['n_insufficient']} 个。", ""]
    L.append(_md_table(["id", "样本外区间", "n", "t_signed", "IC(样本外)", "retention_oos", "p_one", "verdict",
                        "Sharpe 毛/净"],
                       [[r["id"], f"{r['oos_start']} 至 {r['oos_end']}", int(r["n"]), _num(r["t_signed"], 2),
                         _num(r["ic_oos_signed"], 4), _num(r["retention_oos"], 2), _num(r["p_one"], 3),
                         r["verdict"], f"{_num(r['sharpe_gross'], 2)} / {_num(r['sharpe_net'], 2)}"]
                        for _, r in df.iterrows()]))
    L.append("")
    with open(d / "report.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))
