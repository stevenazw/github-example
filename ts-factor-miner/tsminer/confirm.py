"""One-shot out-of-sample confirmation. The only code path that opens the OOS segment."""
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

OOS_MIN_OBS = 60
P_CONFIRMED = 0.10


class OosLockedError(RuntimeError):
    """The OOS of this run was already opened."""


def oos_verdict(t_signed, ic_oos_signed, ic_is, min_retention, sufficient=True) -> str:
    """OOS conclusion from the signed OOS t, the signed OOS IC and the in-sample IC (pure; shared with the
    dashboard). CONFIRMED: one-sided p <= P_CONFIRMED; CONSISTENT: right sign and retention >= min_retention;
    INSUFFICIENT: no usable t (too few OOS observations); otherwise FAILED."""
    if not sufficient or not np.isfinite(t_signed):
        return "INSUFFICIENT"
    if float(norm_sf(t_signed)) <= P_CONFIRMED:
        return "CONFIRMED"
    retention = ic_oos_signed / abs(float(ic_is))
    if ic_oos_signed > 0 and retention >= min_retention:
        return "CONSISTENT"
    return "FAILED"


def confirm(run_dir, force_reason=None) -> dict:
    d = Path(run_dir)
    lock = d / "confirm.lock"
    locked = lock.exists()
    if locked and not force_reason:
        raise OosLockedError(f"OOS already opened for {d} (see {lock}); use --force REASON to override")
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
    oos_mask = prep["oos_mask"]
    cal = panel.target.index
    y = y_fwd.to_numpy(dtype=float)[rows]
    sig = sigma.to_numpy(dtype=float)[rows]
    S_np = S.to_numpy(dtype=float)[rows]
    col_of = {c: i for i, c in enumerate(S.columns)}
    hl = cfg.stats.hac_lags
    cost = cfg.target.cost_per_side
    buf = cfg.position.buffer
    min_ret = cfg.verdict.obs_min_retention

    out_rows = []
    for _, r in short.iterrows():
        j = col_of[r["id"]]
        m = oos_mask[:, j]
        s, yo = S_np[m, j], y[m]
        t_oos, ic_oos, n = hac_t(s[:, None], yo, hl, OOS_MIN_OBS)
        sign = float(r["sign"])
        t_signed = sign * t_oos[0]
        ic_signed = sign * ic_oos[0]
        retention = ic_signed / abs(float(r["ic"]))
        p_one = float(norm_sf(t_signed))
        bt = backtest(s, yo, sig[m], sign, cost, buf)
        pos = rows[np.flatnonzero(m)]
        out_rows.append({"id": r["id"], "sign": int(sign), "n": int(n[0]),
                         "oos_start": str(cal[pos[0]].date()) if len(pos) else "",
                         "oos_end": str(cal[pos[-1]].date()) if len(pos) else "",
                         "t_signed": t_signed, "ic_oos_signed": ic_signed, "retention_oos": retention,
                         "p_one": p_one, "verdict": oos_verdict(t_signed, ic_signed, r["ic"], min_ret),
                         "sharpe_gross": bt["sharpe_gross"], "sharpe_net": bt["sharpe_net"]})
    df = pd.DataFrame(out_rows)

    vc = df["verdict"].value_counts()
    out = _jsonable({
        "status": "DONE", "forced": bool(locked and force_reason), "force_reason": force_reason,
        "n_tested": int(np.isfinite(df["t_signed"]).sum()), "n_confirmed": int(vc.get("CONFIRMED", 0)),
        "n_consistent": int(vc.get("CONSISTENT", 0)), "n_failed": int(vc.get("FAILED", 0)),
        "n_insufficient": int(vc.get("INSUFFICIENT", 0)),
        "candidates": df.to_dict(orient="records"), "opened_at": datetime.now(timezone.utc).isoformat(),
    })
    df.to_csv(d / "confirm.csv", index=False)
    (d / "confirm.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    append_confirm(d)
    lock.write_text(datetime.now(timezone.utc).isoformat())
    return out
