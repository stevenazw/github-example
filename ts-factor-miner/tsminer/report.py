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
    headers = ["id", "description", "t", "z", "IC", "p_fwer", "q_bh", "verdict", "P(signal)", "post_ic", "P(IC>0)",
               "Sharpe 毛/净", "flags"]
    rows = [[r["id"], r["description"], _num(r["t"], 2), _num(r["z"], 2), _num(r["ic"], 4), _num(r["p_fwer"], 3),
             _num(r["q_bh"], 3), r["verdict"], _num(r["p_signal"], 3), _num(r["post_ic"], 4), _num(r["p_pos"], 3),
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
    vd = run["config"]["verdict"]
    mode = run.get("verdict_mode", vd.get("mode", "family"))
    zp, zw = vd.get("z_pass", 2.5), vd.get("z_weak", 2.0)
    exceed = _read_csv(d / "exceed.csv")

    L = [f"# 因子挖掘报告：{run['target']}", ""]
    L += ["## 结论", ""]
    if mode == "per_factor":
        head = (f"判定模式：单因子门槛（PASS: |z| >= {_num(zp, 2)}，WEAK_PASS: |z| >= {_num(zw, 2)}，不做同批校正，"
                f"成立与否由 holdout 判定）。族大小 m = {run['family_size']}，PASS（|z| >= {_num(zp, 2)}）{n_pass} 个，"
                f"WEAK_PASS {n_weak} 个，shortlist {len(run['shortlist'])} 个。")
        allr = exceed[exceed["scope"] == "ALL"] if len(exceed) else exceed

        def _null_at(thr):
            r = allr[np.isclose(allr["threshold"], thr)]
            return _num(r["null_mean"].iloc[0], 1) if len(r) else "-"
        head += (f"纯噪声下预期有 {_null_at(zp)} 个候选达到 PASS 门槛、{_null_at(zw)} 个达到 WEAK_PASS 门槛以上。")
        if n_pass + n_weak == 0:
            head += "没有候选通过。"
    else:
        head = (f"判定模式：族错误率（FWER / FDR 校正）。族大小 m = {run['family_size']}，PASS {n_pass} 个，"
                f"WEAK_PASS {n_weak} 个，shortlist {len(run['shortlist'])} 个，族 omnibus p = {_num(omni['p'], 3)}。")
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

    L += ["## 判定门槛与多重检验参考", ""]
    if mode == "per_factor":
        sq = np.sqrt(run["n_train_rows"])
        L += [f"- 单因子门槛：PASS 为 |z| >= {_num(zp, 2)}（对应 IC ≈ {_num(zp / sq, 4)}），"
              f"WEAK_PASS 为 |z| >= {_num(zw, 2)}（对应 IC ≈ {_num(zw / sq, 4)}）；"
              f"IC = z ÷ √n_train_rows，n_train_rows = {run['n_train_rows']}。不按同批候选数打折。",
              "- 以下为参考信息，不参与判定："]
    L += [f"- 族大小 m = {run['family_size']}",
          f"- FWER 门槛：|z| >= {_num(null['hurdle'], 2)}（z = t ÷ 该候选自身零分布标准差；"
          f"循环平移零分布 studentized single-step maxT，n_boot = {null['n_boot']}）",
          f"- 各候选零分布 t 标准差 null_sd：中位数 {_num(null['null_sd_median'], 2)}，"
          f"最小 {_num(null['null_sd_min'], 2)}，最大 {_num(null['null_sd_max'], 2)}",
          f"- 诊断：原始 t* 的汇总方差 t_var = {_num(t_var, 2)}（不参与判定）"]
    L += [f"- 族 omnibus：mean z² = {_num(omni['mean_z2'], 2)}，p = {_num(omni['p'], 3)}",
          ("- 经验贝叶斯：两组混合模型未发现非零候选（τ² = 0）：P(signal) 全部为 0，不构造组合因子。"
           if not run["eb_tau2"] else
           f"- 经验贝叶斯（两组混合模型）：零假设占比 pi0 = {_num(run['eb_pi0'], 3)}，非零候选的 IC 离散度 "
           f"tau = {_num(np.sqrt(run['eb_tau2']), 4)}（eb_tau2 = {_num(run['eb_tau2'], 6)}）。"
           f"P(signal) 是该候选非零的后验概率（1 - local fdr），post_ic 是按它加权收缩后的 IC，组合因子按 post_ic 加权"
           f"（composite_defined = {run['composite_defined']}）。"), ""]

    L += ["## 通过数量与噪声预期", ""]
    if len(exceed):
        allr = exceed[exceed["scope"] == "ALL"]
        L.append(_md_table(["阈值 |z| >=", "实际通过数 observed", "纯噪声预期 null_mean", "p"],
                           [[_num(r["threshold"], 2), int(r["observed"]), _num(r["null_mean"], 1), _num(r["p"], 3)]
                            for _, r in allr.iterrows()]))
        L += ["", f"各分组在 z_pass = {_num(zp, 2)} 阈值下（按 p 升序）：", ""]
        gz = exceed[(exceed["scope"] != "ALL") & np.isclose(exceed["threshold"], zp)].sort_values("p", kind="stable")
        L.append(_md_table(["group", "n_cand", "observed", "null_mean", "p"],
                           [[r["scope"], int(r["n_cand"]), int(r["observed"]), _num(r["null_mean"], 1),
                             _num(r["p"], 3)] for _, r in gz.iterrows()]))
    else:
        L.append("没有 exceed.csv。")
    L += ["", "降低门槛不会增加证据。observed 接近 null_mean，说明通过名单大部分来自噪声；observed 明显高于 null_mean 的分组"
          "才值得优先关注。分组内候选高度相关，observed 不等于独立发现的个数。null_mean 由循环平移零分布算出，"
          "已自动考虑候选之间的相关性。", ""]

    L += ["## Shortlist", ""]
    if mode == "per_factor":
        L += ["以下为单因子门槛的筛选结果，未做同批校正；是否成立由 holdout 判定。", ""]
    L += [_cand_table(short) if len(short) else "没有候选进入 shortlist。", ""]
    L += ["## Watchlist（未必通过多重检验，仅供参考）", ""]
    L += [_cand_table(watch) if len(watch) else "watchlist 为空。", ""]

    L += ["## 分组 omnibus", ""]
    if len(groups):
        g = groups.sort_values("p")
        L.append(_md_table(["group", "n_cand", "mean z²", "null mean z²", "p", "p_holm"],
                           [[r["group"], int(r["n_cand"]), _num(r["mean_z2"], 2), _num(r["null_mean_z2"], 2),
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
          + ("（强制重跑）" if c.get("forced") else "")]
    if c.get("verdict_mode") == "per_factor":
        L.append(f"- 判定模式：单因子（未校正单边 p_one：<= 0.05 CONFIRMED，<= 0.10 WEAK）。"
                 f"共检验 {c['n_tested']} 个，按 5% 显著性纯运气预期确认 {_num(c['expected_by_chance'], 2)} 个，"
                 f"实际确认 {c['n_confirmed']} 个。")
    L.append("")
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
