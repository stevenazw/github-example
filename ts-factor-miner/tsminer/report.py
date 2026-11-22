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
    headers = ["id", "description", "group", "train 区间", "n_train", "t", "z", "IC", "verdict", "Sharpe 毛/净",
               "cluster_size", "flags"]
    rows = [[r["id"], r["description"], r["group"], f"{r['train_start']} 至 {r['train_end']}", int(r["n_train"]),
             _num(r["t"], 2), _num(r["z"], 2), _num(r["ic"], 4), r["verdict"],
             f"{_num(r['sharpe_gross'], 2)} / {_num(r['sharpe_net'], 2)}", int(r["cluster_size"]), r["flags"]]
            for _, r in df.iterrows()]
    return _md_table(headers, rows)


def write_report(run_dir) -> None:
    d = Path(run_dir)
    run = json.loads((d / "run.json").read_text())
    short = _read_csv(d / "shortlist.csv")
    excl = _read_csv(d / "panel" / "excluded.csv")
    info, dates, nt = run["panel_info"], run["dates"], run["n_train_summary"]
    vc = run["verdict_counts"]
    n_pass, n_weak = vc.get("PASS", 0), vc.get("WEAK_PASS", 0)
    vd = run["config"]["verdict"]
    hf = run["config"]["sample"]["holdout_frac"]
    zp, zw = vd["z_pass"], vd["z_weak"]

    L = [f"# 因子筛选报告：{run['target']}", ""]
    L += ["## 结论", ""]
    head = (f"逐个检验了 {run['n_tested']} 个候选；PASS {n_pass} 个、WEAK_PASS {n_weak} 个；"
            f"去重后 shortlist {len(run['shortlist'])} 个。")
    if n_pass + n_weak == 0:
        head += "没有候选通过门槛。"
    L += [head, ""]

    L += ["## 数据与样本", ""]
    L += [f"- 数据库版本：{info.get('db_version')}，发布时间：{info.get('published_at')}",
          f"- 标的样本：{dates['sample_start']} 至 {dates['sample_end']}（{run['n_rows']} 行）",
          f"- 每个候选在自己与标的重叠的样本上独立检验，并各自按 holdout_frac = {_num(hf, 2)} 切分 "
          f"train / holdout（holdout 未被打开）。",
          f"- 序列数：{run['n_series']}；候选数（序列 x 变换）：{run['n_candidates']}；"
          f"已检验：{run['n_tested']}；因样本不足未检验：{run['n_skipped']}",
          f"- 已检验候选的 train 长度 n_train：最小 {nt['min']}，中位 {_num(nt['median'], 0)}，最大 {nt['max']}"]
    if len(excl):
        cnt = excl["reason"].value_counts()
        L.append(f"- 被剔除的序列：{int(cnt.sum())} 个（" + "，".join(f"{k}: {v}" for k, v in cnt.items()) + "）")
    else:
        L.append("- 被剔除的序列：0 个")
    L.append("")

    L += ["## 判定标准", "",
          f"- PASS：|z| ≥ {_num(zp, 2)}；WEAK_PASS：|z| ≥ {_num(zw, 2)}；其余 REJECT。",
          "- z = HAC t ÷ 该候选自身循环平移零分布的标准差。",
          "- 门槛不随样本长度变，同样的 z 在短样本上对应更高的 IC（IC ≈ z ÷ √n_train）：", ""]
    L.append(_md_table(["n_train", f"PASS 门槛 (z = {_num(zp, 2)}) 对应 IC", f"WEAK_PASS 门槛 (z = {_num(zw, 2)}) 对应 IC"],
                       [[n, _num(zp / np.sqrt(n), 4), _num(zw / np.sqrt(n), 4)] for n in (5000, 2000, 1000, 500)]))
    L.append("")

    L += ["## Shortlist（去重后）", ""]
    L += [_cand_table(short) if len(short) else "没有候选进入 shortlist。", ""]

    L += ["## 提示与局限", "",
          "- 默认滞后是保守假设：不假设 x 的取值时刻早于标的收盘，真实可用时间可能更早。",
          "- 月频、季频及不规则序列未纳入。",
          "- 数据库的 release_date 未使用（历史值不可靠），可用性完全由配置中的 lag 决定。",
          "- spot 标的的收益不含 carry。",
          "- 短样本候选的 holdout 也短，确认能力有限。",
          "- holdout 只能看一次：confirm 之后不应再据其结果回头调整候选或参数。", ""]
    L += ["## 下一步", "",
          f"确认 shortlist 请运行（只能运行一次）：`python -m tsminer confirm {d}`", ""]
    (d / "report.md").write_text("\n".join(L), encoding="utf-8")


def append_confirm(run_dir) -> None:
    d = Path(run_dir)
    c = json.loads((d / "confirm.json").read_text())
    df = _read_csv(d / "confirm.csv")
    L = ["", "## Holdout 确认", ""]
    L += [f"- 每个 shortlist 候选在自己的 holdout 行上按 train 符号做单边 HAC t 检验"
          f"（p_one ≤ 0.05 为 CONFIRMED，≤ 0.10 为 WEAK）。共检验 {c['n_tested']} 个，确认 {c['n_confirmed']} 个"
          + ("（强制重跑）。" if c.get("forced") else "。"), ""]
    L.append(_md_table(["id", "holdout 区间", "n", "t_signed", "IC", "p_one", "verdict", "Sharpe 毛/净"],
                       [[r["id"], f"{r['holdout_start']} 至 {r['holdout_end']}", int(r["n"]),
                         _num(r["t_signed"], 2), _num(r["ic"], 4), _num(r["p_one"], 3), r["verdict"],
                         f"{_num(r['sharpe_gross'], 2)} / {_num(r['sharpe_net'], 2)}"]
                        for _, r in df.iterrows()]))
    L.append("")
    with open(d / "report.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))
