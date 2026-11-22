"""Command line: ``python -m tsminer mine|confirm``."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import load_config
from .confirm import HoldoutLockedError, confirm
from .databank import Panel
from .pipeline import mine


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tsminer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("mine", help="mine factors on the train segment")
    pm.add_argument("config")
    pm.add_argument("--run-dir")
    pm.add_argument("--panel", help="saved panel directory to reuse instead of querying the databank")
    pc = sub.add_parser("confirm", help="open the holdout once and confirm the shortlist")
    pc.add_argument("run_dir")
    pc.add_argument("--force", metavar="REASON", help="re-open an already opened holdout (logged)")
    a = ap.parse_args(argv)

    if a.cmd == "mine":
        cfg = load_config(a.config)
        run_dir = a.run_dir or str(Path("runs") / f"{cfg.target.name}-{datetime.now():%Y%m%d-%H%M%S}")
        panel = Panel.load(a.panel) if a.panel else None
        mine(cfg, run_dir, panel=panel, config_path=a.config)
        return 0
    try:
        res = confirm(a.run_dir, force_reason=a.force)
    except HoldoutLockedError as e:
        print(f"HOLDOUT LOCKED: {e}", file=sys.stderr)
        return 2
    print(f"[confirm] status={res['status']}")
    if res["status"] == "DONE":
        print(f"[confirm] confirmed={res['n_confirmed']} composite={res['composite']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
