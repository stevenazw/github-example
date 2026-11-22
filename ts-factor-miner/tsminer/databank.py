"""Data adapter: turns the market databank into an aligned, lag-safe ``Panel``.

Look-ahead protection lives entirely in ``align_to_calendar`` + ``resolve_lag``:
the databank's ``release_date`` is NOT used (it is unreliable for history).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STALE_LIMIT = {"D": 5, "W": 10}     # calendar rows an observation may be carried forward
MIN_FINITE_OBS = 250                # series with fewer finite aligned values are excluded
BATCH_SIZE = 200

META_COLUMNS = ["ticker", "field", "kind", "freq", "group", "lag", "description", "first_obs", "n_obs"]
FREQ_RANK = {"D": 0, "W": 1, "other": 2}


@dataclass
class Panel:
    target: pd.Series       # daily return (decimal); DatetimeIndex = target calendar; name = target name
    X: pd.DataFrame         # same calendar; columns = series id; X.loc[t] is known at the close of t
    meta: pd.DataFrame      # index = series id; columns = META_COLUMNS
    excluded: pd.DataFrame  # columns: ticker, reason
    info: dict              # db version, published_at, target name, built_at, sample start/end

    def save(self, directory: str | os.PathLike) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "panel.pkl", "wb") as fh:
            pickle.dump({"target": self.target, "X": self.X, "meta": self.meta,
                         "excluded": self.excluded, "info": self.info}, fh)
        # human-readable companions (panel.pkl is the source of truth)
        self.meta.to_csv(d / "meta.csv")
        self.excluded.to_csv(d / "excluded.csv", index=False)
        (d / "info.json").write_text(json.dumps(self.info, indent=2, default=str, ensure_ascii=False))

    @classmethod
    def load(cls, directory: str | os.PathLike) -> "Panel":
        with open(Path(directory) / "panel.pkl", "rb") as fh:
            d = pickle.load(fh)
        return cls(**d)


# ------------------------------------------------------------ pure functions
def infer_freq(dates: pd.DatetimeIndex) -> str:
    """'D' / 'W' / 'other' from the median gap (calendar days) between observations."""
    dates = pd.DatetimeIndex(dates)
    if len(dates) < 30:
        return "other"
    gaps = np.diff(dates.sort_values().values.astype("datetime64[D]")).astype(float)
    med = float(np.median(gaps))
    if med <= 3:
        return "D"
    if 5 <= med <= 9:
        return "W"
    return "other"


def infer_kind(unit: str, data_type: str) -> str:
    """'price' (take log returns) or 'level' (take differences)."""
    if data_type in {"price", "index_level"} and unit not in {"fwd_pts", "pct"}:
        return "price"
    return "level"


def resolve_lag(ticker: str, class_one: str, class_two: str, freq: str, lags_cfg) -> int:
    """Lag in calendar rows: ticker override > 'class_one/class_two' > frequency default."""
    by_ticker = _lags_attr(lags_cfg, "ticker")
    if ticker in by_ticker:
        return int(by_ticker[ticker])
    by_class = _lags_attr(lags_cfg, "class")
    key = f"{class_one}/{class_two}"
    if key in by_class:
        return int(by_class[key])
    if freq == "D":
        return int(_lags_attr(lags_cfg, "D"))
    if freq == "W":
        return int(_lags_attr(lags_cfg, "W"))
    raise ValueError(f"no default lag for freq {freq!r} (ticker {ticker})")


def _lags_attr(lags_cfg, name: str):
    """Accept a LagsConfig or a plain dict (keys 'D','W','class','ticker')."""
    if isinstance(lags_cfg, dict):
        return lags_cfg.get(name, {} if name in ("class", "ticker") else None)
    return getattr(lags_cfg, "class_" if name == "class" else name)


def _ns(idx: pd.DatetimeIndex) -> np.ndarray:
    return pd.DatetimeIndex(idx).as_unit("ns").values


def align_to_calendar(x: pd.Series, calendar: pd.DatetimeIndex, lag: int, stale_limit: int) -> pd.Series:
    """As-of align ``x`` to ``calendar``, truncate stale values, then shift by ``lag`` rows.

    Result at row t = as-of value of ``x`` at calendar row t-lag, where the as-of value is
    the latest observation dated <= calendar[t-lag], or NaN if that observation is more
    than ``stale_limit`` calendar rows old.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    calendar = pd.DatetimeIndex(calendar)
    x = x.dropna()
    x = x[~x.index.duplicated(keep="last")].sort_index()
    n = len(calendar)
    if len(x) == 0 or n == 0:
        return pd.Series(np.nan, index=calendar, dtype=float)
    cal = _ns(calendar)
    xi = _ns(x.index)
    k = np.searchsorted(xi, cal, side="right") - 1          # last obs with date <= calendar date
    has = k >= 0
    kc = np.where(has, k, 0)
    # calendar rows elapsed since the observation became visible (0 = observed on this row)
    obs_rows = np.searchsorted(cal, xi, side="right")        # rows with date <= obs date
    age = np.arange(1, n + 1) - obs_rows[kc]
    ok = has & (age <= stale_limit)
    vals = np.where(ok, x.to_numpy(dtype=float)[kc], np.nan)
    out = pd.Series(vals, index=calendar, dtype=float)
    return out.shift(lag) if lag else out


# ------------------------------------------------------------------ builders
def _load_db_info(cache_dir: str) -> dict:
    with open(Path(cache_dir) / "current.json") as fh:
        return json.load(fh)


def _open_query(cfg):
    toolkit = cfg.databank.toolkit_dir
    if toolkit not in sys.path:
        sys.path.insert(0, toolkit)
    from data_query import DataQuery  # noqa: E402  (toolkit lives outside the repo)
    cur = _load_db_info(cfg.databank.cache_dir)
    return DataQuery(db_path=cur["database_path"]), cur


def _records_to_series(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=float),
                  index=pd.to_datetime(df["trade_date"]))
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _fetch(q, tickers: list[str], field: str) -> dict[str, tuple[pd.Series, str, str]]:
    """Fetch one field for many tickers: ticker -> (series, unit, data_type)."""
    out = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        recs = q.get_timeseries(batch, [field], as_dataframe=False)
        if not recs:
            continue
        df = pd.DataFrame.from_records(recs, columns=["ticker", "trade_date", "value", "unit", "data_type"])
        for tk, g in df.groupby("ticker", sort=False):
            out[tk] = (_records_to_series(g), str(g["unit"].iloc[0]), str(g["data_type"].iloc[0]))
    return out


def _select_tickers(q, cfg) -> list[str]:
    found: list[str] = []
    for entry in cfg.universe.select:
        if "tickers" in entry:
            found += entry["tickers"]
        elif "family" in entry:
            found += [r["ticker"] for r in q.resolve(entry["family"], allow_multi=True, **entry["attrs"])]
        else:
            c1, c2, c3 = entry.get("class_one"), entry.get("class_two"), entry.get("class_three")
            for third in (c3 if isinstance(c3, list) else [c3]):
                found += [r["ticker"] for r in q.list_securities(class_one=c1, class_two=c2, class_three=third)]
    excl = set(cfg.universe.exclude_tickers)
    return [t for t in dict.fromkeys(found) if t not in excl]    # dedupe, keep order


def _pick_field(q, ticker: str, priority: list[str]) -> str | None:
    have = {f["field"] for f in q.list_fields(ticker)}
    return next((f for f in priority if f in have), None)


def _build_target(q, cfg) -> pd.Series:
    t = cfg.target
    if t.kind == "futures_chain":
        ch = q.get_futures_chain(t.root)
        ch = ch[ch["return_valid"] == 1]
        r = pd.Series(ch["ret"].to_numpy(dtype=float), index=pd.to_datetime(ch["trade_date"]))
    else:
        got = _fetch(q, [t.ticker], t.field)
        if t.ticker not in got:
            raise ValueError(f"target {t.ticker}/{t.field} has no data")
        p = got[t.ticker][0]
        p = p[p > 0]
        r = np.log(p).diff()                                   # first row stays NaN
    r = r[~r.index.duplicated(keep="last")].sort_index()
    if cfg.sample.start:
        r = r[r.index >= pd.Timestamp(cfg.sample.start)]
    r.name = t.name
    return r


def _chain_series(q, root: str, calendar: pd.DatetimeIndex, lag: int, what: str) -> pd.Series:
    ch = q.get_futures_chain(root)
    ch = ch[ch["return_valid"] == 1]
    idx = pd.to_datetime(ch["trade_date"])
    if what == "index":
        v = ch["return_index"].to_numpy(dtype=float)
    else:
        v = (ch["calendar_spread"] / ch["px_active"]).to_numpy(dtype=float)
    s = pd.Series(v, index=idx)
    s = s[np.isfinite(s)]
    return align_to_calendar(s, calendar, lag, STALE_LIMIT["D"])


def build_panel(cfg) -> Panel:
    q, cur = _open_query(cfg)
    r = _build_target(q, cfg)
    calendar = r.index
    lags = cfg.lags
    priority = cfg.universe.fields

    # security lookup (including inactive) for class / description of any ticker
    sec_info = {s["ticker"]: s for s in q.list_securities(include_inactive=True, limit=100000)}

    series: dict[str, pd.Series] = {}
    rows: dict[str, dict] = {}
    excluded: list[dict] = []

    def add_series(sid: str, aligned: pd.Series, **m) -> None:
        series[sid] = aligned
        rows[sid] = m

    def process(ticker: str, s: pd.Series, field: str, unit: str, dtype: str) -> dict | None:
        """Align one databank series; returns the info needed for meta, or None if excluded."""
        freq = infer_freq(s.index)
        if freq == "other":
            excluded.append({"ticker": ticker, "reason": "freq_not_daily_or_weekly"})
            return None
        info = sec_info.get(ticker, {})
        c1, c2 = info.get("class_one", ""), info.get("class_two", "")
        lag = resolve_lag(ticker, c1, c2, freq, lags)
        return dict(aligned=align_to_calendar(s, calendar, lag, STALE_LIMIT[freq]),
                    ticker=ticker, field=field, kind=infer_kind(unit, dtype), freq=freq,
                    group=c2, lag=lag, description=info.get("description", ""))

    # 3. candidate pool and field choice
    pool = _select_tickers(q, cfg)
    by_field: dict[str, list[str]] = {}
    for tk in pool:
        f = _pick_field(q, tk, priority)
        if f is None:
            excluded.append({"ticker": tk, "reason": "no_supported_field"})
        else:
            by_field.setdefault(f, []).append(tk)

    # 4. fetch + align
    fetched: dict[str, tuple] = {}     # ticker -> (series, unit, dtype, field), reused by spreads
    for f, tks in by_field.items():
        for tk, (s, unit, dtype) in _fetch(q, tks, f).items():
            fetched[tk] = (s, unit, dtype, f)
        for tk in tks:
            if tk not in fetched:
                excluded.append({"ticker": tk, "reason": "no_data"})
    for tk in pool:
        if tk not in fetched:
            continue
        s, unit, dtype, f = fetched[tk]
        res = process(tk, s, f, unit, dtype)
        if res is not None:
            add_series(tk, res.pop("aligned"), **res)

    # 5. futures chains and carry
    for root in cfg.universe.chains:
        add_series(f"chain:{root}", _chain_series(q, root, calendar, lags.D, "index"),
                   ticker=f"chain:{root}", field="return_index", kind="price", freq="D",
                   group="Chain", lag=lags.D, description=f"{root} futures chain return index")
    for root in cfg.universe.carry_chains:
        add_series(f"carry:{root}", _chain_series(q, root, calendar, lags.D, "carry"),
                   ticker=f"carry:{root}", field="calendar_spread/px_active", kind="level", freq="D",
                   group="Carry", lag=lags.D, description=f"{root} calendar spread / active price")

    # 6. spreads: legs aligned and lagged on their own, then differenced
    for sp in cfg.universe.spread:
        legs = {}
        for leg in ("a", "b"):
            tk = sp[leg]
            if tk not in fetched:
                try:
                    f = _pick_field(q, tk, priority)
                except Exception as e:     # unknown ticker raises in the databank
                    raise ValueError(f"spread {sp['id']}: leg {tk!r} not available ({e})") from e
                if f is None:
                    raise ValueError(f"spread {sp['id']}: leg {tk!r} has no supported field")
                got = _fetch(q, [tk], f)
                if tk not in got:
                    raise ValueError(f"spread {sp['id']}: leg {tk!r} returned no data")
                fetched[tk] = (*got[tk], f)
            s, unit, dtype, f = fetched[tk]
            res = process(tk, s, f, unit, dtype)
            if res is None:
                # process() logged an exclusion; for a spread leg that is a config error
                excluded.pop()
                raise ValueError(f"spread {sp['id']}: leg {tk!r} is not daily/weekly")
            legs[leg] = res
        a, b = legs["a"], legs["b"]
        diff = (a["aligned"] - b["aligned"]).where(np.isfinite(a["aligned"]) & np.isfinite(b["aligned"]))
        freq = max(a["freq"], b["freq"], key=lambda z: FREQ_RANK[z])
        add_series(f"spread:{sp['id']}", diff, ticker=f"spread:{sp['id']}",
                   field=f"{a['ticker']}-{b['ticker']}", kind="level", freq=freq, group="Spread",
                   lag=max(a["lag"], b["lag"]), description=f"{sp['a']} minus {sp['b']}")

    # 7. minimum observations, then assemble
    keep = []
    for sid in sorted(series):
        if int(np.isfinite(series[sid].to_numpy()).sum()) < MIN_FINITE_OBS:
            excluded.append({"ticker": sid, "reason": "too_few_obs"})
        else:
            keep.append(sid)
    X = pd.DataFrame({sid: series[sid] for sid in keep}, index=calendar, columns=keep, dtype=float)
    meta_rows = []
    for sid in keep:
        col = X[sid]
        fin = col.dropna()
        meta_rows.append({**rows[sid], "first_obs": fin.index[0], "n_obs": int(len(fin))})
    meta = pd.DataFrame(meta_rows, index=pd.Index(keep, name="series_id"), columns=META_COLUMNS)
    exc = pd.DataFrame(excluded, columns=["ticker", "reason"])

    info = {
        "db_version": cur.get("version"),
        "published_at": cur.get("published_at"),
        "db_path": cur.get("database_path"),
        "target": cfg.target.name,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sample_start": str(calendar[0].date()),
        "sample_end": str(calendar[-1].date()),
        "n_rows": int(len(calendar)),
    }
    return Panel(target=r, X=X, meta=meta, excluded=exc, info=info)
