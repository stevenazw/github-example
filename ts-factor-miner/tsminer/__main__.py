"""Command line: ``python -m tsminer mine|confirm|site``."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import load_config
from .confirm import OosLockedError, confirm
from .databank import Panel
from .pipeline import mine


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tsminer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("mine", help="mine factors on the IS / OBS segments")
    pm.add_argument("config")
    pm.add_argument("--run-dir")
    pm.add_argument("--panel", help="saved panel directory to reuse instead of querying the databank")
    pc = sub.add_parser("confirm", help="open the OOS once and confirm the shortlist")
    pc.add_argument("run_dir")
    pc.add_argument("--force", metavar="REASON", help="re-open an already opened OOS (logged)")
    ps = sub.add_parser("site", help="build the factor-tracking dashboard (static site) from run directories")
    ps.add_argument("run_dirs", nargs="*", metavar="RUN_DIR",
                    help="run directories to add to the registry (OUT/runs.json); the site is rebuilt from all "
                         "registered runs, and the same factor tested twice appears once")
    ps.add_argument("--out", required=True, metavar="DIR", help="output directory of the site")
    ps.add_argument("--reset", action="store_true", help="ignore the existing registry; use only the given RUN_DIRs")
    a = ap.parse_args(argv)

    if a.cmd == "site":
        from .site import build_site
        try:
            build_site(a.run_dirs, a.out, reset=a.reset)
        except (ValueError, FileNotFoundError) as e:
            print(f"site: {e}", file=sys.stderr)
            return 1
        print(f"[site] wrote {a.out}/index.html and {a.out}/artifact.html")
        return 0

    if a.cmd == "mine":
        cfg = load_config(a.config)
        run_dir = a.run_dir or str(Path("runs") / f"{cfg.target.name}-{datetime.now():%Y%m%d-%H%M%S}")
        panel = Panel.load(a.panel) if a.panel else None
        mine(cfg, run_dir, panel=panel, config_path=a.config)
        return 0
    try:
        res = confirm(a.run_dir, force_reason=a.force)
    except OosLockedError as e:
        print(f"OOS LOCKED: {e}", file=sys.stderr)
        return 2
    print(f"[confirm] status={res['status']}")
    if res["status"] == "DONE":
        print(f"[confirm] confirmed={res['n_confirmed']} consistent={res['n_consistent']} "
              f"failed={res['n_failed']} insufficient={res['n_insufficient']} of {res['n_tested']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
