"""One-shot holdout confirmation. The only code path that opens the holdout segment."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .databank import Panel
from .pipeline import _jsonable, backtest, prepare
from .report import append_confirm
from .stats import hac_t, holm, norm_sf

HOLDOUT_MIN_OBS = 120


class HoldoutLockedError(RuntimeError):
    """The holdout of this run was already opened."""


def _verdict(p, p_ok=0.05, p_weak=0.10) -> str:
    if not np.isfinite(p):
        return "INSUFFICIENT"
    return "CONFIRMED" if p <= p_ok else "WEAK" if p <= p_weak else "NOT_CONFIRMED"


def confirm(run_dir, force_reason=None) -> dict:
    d = Path(run_dir)
    lock = d / "confirm.lock"
    locked = lock.exists()
    if locked and not force_reason:
        raise HoldoutLockedError(f"holdout already opened for {d} (see {lock}); use --force REASON to override")
    if force_reason:
        with open(d / "confirm_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "reason": force_reason},
                                ensure_ascii=False) + "\n")

    run = json.loads((d / "run.json").read_text())
    cands = pd.read_csv(d / "candidates.csv")
    short = pd.read_csv(d / "shortlist.csv")
    if len(short) == 0 and not run["composite_defined"]:
        return {"status": "NOTHING_TO_CONFIRM"}

    cfg = load_config(d / "config.toml")
    panel = Panel.load(d / "panel")
    prep = prepare(panel, cfg)
    S, y_fwd, sigma = prep["S"], prep["y_fwd"], prep["sigma"]
    ho, tr = prep["holdout_rows"], prep["train_rows"]
    y_ho = y_fwd.to_numpy(dtype=float)[ho]
    sig_ho = sigma.to_numpy(dtype=float)[ho]
    hl = cfg.stats.hac_lags
    cost = cfg.target.cost_per_side

    rows = []
    for _, r in short.iterrows():
        s = S[r["id"]].to_numpy(dtype=float)[ho]
        t_hold, ic, n = hac_t(s[:, None], y_ho, hl, HOLDOUT_MIN_OBS)
        sign = float(r["sign"])
        t_signed = sign * t_hold[0]
        bt = backtest(s, y_ho, sig_ho, sign, cost)
        rows.append({"id": r["id"], "sign_train": int(sign), "n": int(n[0]), "t_hold": t_hold[0],
                     "t_signed": t_signed, "ic": ic[0], "p_one": float(norm_sf(t_signed)),
                     "sharpe_gross": bt["sharpe_gross"], "sharpe_net": bt["sharpe_net"]})
    cols = ["id", "sign_train", "n", "t_hold", "t_signed", "ic", "p_one", "p_holm", "verdict",
            "sharpe_gross", "sharpe_net"]
    df = pd.DataFrame(rows, columns=[c for c in cols if c not in ("p_holm", "verdict")])
    df["p_holm"] = holm(df["p_one"].where(np.isfinite(df["t_signed"])).to_numpy(dtype=float))
    df["verdict"] = [_verdict(p) for p in df["p_holm"]]
    df = df[cols]

    composite = None
    if run["composite_defined"]:
        w = cands.set_index("id")["post_ic"].fillna(0.0)
        S_np = S[w.index].to_numpy(dtype=float)
        comp = np.where(np.isfinite(S_np), S_np, 0.0) @ w.to_numpy()
        t_c, _, _ = hac_t(comp[ho][:, None], y_ho, hl, HOLDOUT_MIN_OBS)
        p_c = float(norm_sf(t_c[0])) if np.isfinite(t_c[0]) else float("nan")
        sd = float(np.std(comp[tr]))
        bt = backtest(comp[ho] / sd if sd > 0 else comp[ho] * np.nan, y_ho, sig_ho, 1.0, cost)
        composite = {"t": t_c[0], "p": p_c, "verdict": _verdict(p_c),
                     "sharpe_gross": bt["sharpe_gross"], "sharpe_net": bt["sharpe_net"]}

    cal = panel.target.index
    out = _jsonable({
        "status": "DONE", "forced": bool(locked and force_reason), "force_reason": force_reason,
        "holdout_start": str(cal[ho[0] + 1].date()), "holdout_end": str(cal[-1].date()),
        "n_holdout_rows": len(ho), "n_confirmed": int((df["verdict"] == "CONFIRMED").sum()),
        "composite": composite, "candidates": df.to_dict(orient="records"),
        "opened_at": datetime.now(timezone.utc).isoformat(),
    })
    df.to_csv(d / "confirm.csv", index=False)
    (d / "confirm.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    append_confirm(d)
    lock.write_text(datetime.now(timezone.utc).isoformat())
    return out
