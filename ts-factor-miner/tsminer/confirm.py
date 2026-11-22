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
from .stats import hac_t, norm_sf

HOLDOUT_MIN_OBS = 60


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

    short = pd.read_csv(d / "shortlist.csv")
    if len(short) == 0:
        return {"status": "NOTHING_TO_CONFIRM"}

    cfg = load_config(d / "config.toml")
    panel = Panel.load(d / "panel")
    prep = prepare(panel, cfg)
    S, y_fwd, sigma, rows = prep["S"], prep["y_fwd"], prep["sigma"], prep["rows"]
    ho_mask = prep["holdout_mask"]
    cal = panel.target.index
    y = y_fwd.to_numpy(dtype=float)[rows]
    sig = sigma.to_numpy(dtype=float)[rows]
    S_np = S.to_numpy(dtype=float)[rows]
    col_of = {c: i for i, c in enumerate(S.columns)}
    hl = cfg.stats.hac_lags
    cost = cfg.target.cost_per_side

    out_rows = []
    for _, r in short.iterrows():
        j = col_of[r["id"]]
        m = ho_mask[:, j]
        s, yh = S_np[m, j], y[m]
        t_hold, ic, n = hac_t(s[:, None], yh, hl, HOLDOUT_MIN_OBS)
        sign = float(r["sign"])
        t_signed = sign * t_hold[0]
        p_one = float(norm_sf(t_signed))
        bt = backtest(s, yh, sig[m], sign, cost)
        pos = rows[np.flatnonzero(m)]
        out_rows.append({"id": r["id"], "sign_train": int(sign),
                         "holdout_start": str(cal[pos[0]].date()) if len(pos) else "",
                         "holdout_end": str(cal[pos[-1]].date()) if len(pos) else "",
                         "n": int(n[0]), "t_hold": t_hold[0], "t_signed": t_signed, "ic": ic[0], "p_one": p_one,
                         "verdict": _verdict(p_one), "sharpe_gross": bt["sharpe_gross"],
                         "sharpe_net": bt["sharpe_net"]})
    df = pd.DataFrame(out_rows)

    out = _jsonable({
        "status": "DONE", "forced": bool(locked and force_reason), "force_reason": force_reason,
        "n_tested": int(np.isfinite(df["t_signed"]).sum()), "n_confirmed": int((df["verdict"] == "CONFIRMED").sum()),
        "candidates": df.to_dict(orient="records"), "opened_at": datetime.now(timezone.utc).isoformat(),
    })
    df.to_csv(d / "confirm.csv", index=False)
    (d / "confirm.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    append_confirm(d)
    lock.write_text(datetime.now(timezone.utc).isoformat())
    return out
