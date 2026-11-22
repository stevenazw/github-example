"""Configuration loading for the single-asset time-series factor miner.

``load_config`` reads a TOML file (stdlib ``tomllib``), validates it strictly
(missing required keys, unknown sections or unknown keys raise ``ValueError``)
and returns a nested dataclass tree that can be dumped with ``Config.to_dict``.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from typing import Any

ENV_TOOLKIT = "DATABANK_TOOLKIT"
ENV_CACHE = "DATABANK_CACHE"

TOP_SECTIONS = {"target", "databank", "sample", "universe", "lags", "signals", "stats", "verdict"}
TARGET_KINDS = {"futures_chain", "spot"}
CLASS_KEYS = {"class_one", "class_two", "class_three"}
VERDICT_MODES = {"per_factor", "family"}


@dataclass
class TargetConfig:
    name: str
    kind: str
    cost_per_side: float
    root: str = ""
    ticker: str = ""
    field: str = ""


@dataclass
class DatabankConfig:
    toolkit_dir: str = "/home/user/databank-work/entry"
    cache_dir: str = "/home/user/databank-work/databank-cache"


@dataclass
class SampleConfig:
    start: str = ""
    holdout_frac: float = 0.2


@dataclass
class UniverseConfig:
    fields: list[str] = field(default_factory=lambda: ["PX_LAST", "CFTC_CONTRACTS"])
    chains: list[str] = field(default_factory=list)
    carry_chains: list[str] = field(default_factory=list)
    exclude_tickers: list[str] = field(default_factory=list)
    select: list[dict] = field(default_factory=list)
    spread: list[dict] = field(default_factory=list)


@dataclass
class LagsConfig:
    D: int = 1
    W: int = 6
    class_: dict[str, int] = field(default_factory=dict)   # TOML key: "class"
    ticker: dict[str, int] = field(default_factory=dict)


@dataclass
class SignalsConfig:
    std_window: int = 252
    clip: float = 4.0
    vol_halflife: int = 20


@dataclass
class StatsConfig:
    hac_lags: int = 5
    min_obs: int = 500
    n_boot: int = 1000
    min_shift: int = 60
    seed: int = 20261122


@dataclass
class VerdictConfig:
    fwer_alpha: float = 0.05
    fdr_q: float = 0.10
    dedupe_corr: float = 0.7
    extra_lag_t_min: float = 2.0
    mode: str = "per_factor"
    z_pass: float = 2.5
    z_weak: float = 2.0


@dataclass
class Config:
    target: TargetConfig
    databank: DatabankConfig = field(default_factory=DatabankConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    lags: LagsConfig = field(default_factory=LagsConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    verdict: VerdictConfig = field(default_factory=VerdictConfig)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable nested dict (``lags.class_`` is emitted as ``class``)."""
        d = asdict(self)
        d["lags"]["class"] = d["lags"].pop("class_")
        return d


# ----------------------------------------------------------------- helpers
def _section(raw: dict, name: str, allowed: set[str], required: set[str] = frozenset()) -> dict:
    sec = raw.get(name, {})
    if not isinstance(sec, dict):
        raise ValueError(f"[{name}] must be a table")
    unknown = set(sec) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) in [{name}]: {sorted(unknown)}")
    missing = required - set(sec)
    if missing:
        raise ValueError(f"missing required key(s) in [{name}]: {sorted(missing)}")
    return sec


def _check_type(value: Any, types: tuple, where: str) -> None:
    # bool is an int subclass; never accept it as a number
    if isinstance(value, bool) and bool not in types:
        raise ValueError(f"{where}: unexpected bool")
    if not isinstance(value, types):
        raise ValueError(f"{where}: expected {'/'.join(t.__name__ for t in types)}, got {type(value).__name__}")


def _str_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where}: expected a list of strings")
    return list(value)


def _int_map(value: Any, where: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a table")
    for k, v in value.items():
        _check_type(v, (int,), f"{where}.{k}")
        if v < 0:
            raise ValueError(f"{where}.{k}: lag must be >= 0")
    return dict(value)


def _fill(cls, sec: dict, where: str, types: dict[str, tuple]):
    """Instantiate dataclass ``cls`` from table ``sec`` with simple type checks."""
    for k, v in sec.items():
        _check_type(v, types[k], f"{where}.{k}")
    return cls(**sec)


def _parse_select(entry: Any, i: int) -> dict:
    where = f"universe.select[{i}]"
    if not isinstance(entry, dict) or not entry:
        raise ValueError(f"{where}: must be a non-empty table")
    keys = set(entry)
    forms = []
    if keys & CLASS_KEYS:
        forms.append("class")
    if keys & {"family", "attrs"}:
        forms.append("family")
    if "tickers" in keys:
        forms.append("tickers")
    if len(forms) != 1:
        raise ValueError(f"{where}: must be exactly one of class / family+attrs / tickers form, got keys {sorted(keys)}")
    form = forms[0]
    if form == "class":
        extra = keys - CLASS_KEYS
        if extra:
            raise ValueError(f"{where}: unknown key(s) {sorted(extra)} in class form")
        out = {}
        for k, v in entry.items():
            out[k] = _str_list(v, f"{where}.{k}") if isinstance(v, list) else v
            if not isinstance(out[k], (str, list)):
                raise ValueError(f"{where}.{k}: expected str or list[str]")
            if isinstance(v, list) and k != "class_three":
                raise ValueError(f"{where}.{k}: list only allowed for class_three")
        return out
    if form == "family":
        if keys != {"family", "attrs"}:
            raise ValueError(f"{where}: family form needs exactly 'family' and 'attrs'")
        _check_type(entry["family"], (str,), f"{where}.family")
        attrs = entry["attrs"]
        if not isinstance(attrs, dict) or not attrs or not all(isinstance(v, str) for v in attrs.values()):
            raise ValueError(f"{where}.attrs: expected a non-empty table of strings")
        return {"family": entry["family"], "attrs": dict(attrs)}
    if keys != {"tickers"}:
        raise ValueError(f"{where}: tickers form takes only 'tickers'")
    return {"tickers": _str_list(entry["tickers"], f"{where}.tickers")}


def _parse_spread(entry: Any, i: int) -> dict:
    where = f"universe.spread[{i}]"
    if not isinstance(entry, dict) or set(entry) != {"id", "a", "b"}:
        raise ValueError(f"{where}: must have exactly keys id, a, b")
    for k, v in entry.items():
        _check_type(v, (str,), f"{where}.{k}")
    return dict(entry)


# -------------------------------------------------------------------- main
def load_config(path: str | os.PathLike) -> Config:
    """Load and validate a TOML config; raise ``ValueError`` on any config error."""
    with open(path, "rb") as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"invalid TOML in {path}: {e}") from e

    unknown = set(raw) - TOP_SECTIONS
    if unknown:
        raise ValueError(f"unknown top-level section(s): {sorted(unknown)}")
    if "target" not in raw:
        raise ValueError("missing required section [target]")

    # target
    sec = _section(raw, "target", {"name", "kind", "root", "ticker", "field", "cost_per_side"},
                   {"name", "kind", "cost_per_side"})
    target = _fill(TargetConfig, sec, "target", {
        "name": (str,), "kind": (str,), "root": (str,), "ticker": (str,), "field": (str,),
        "cost_per_side": (int, float)})
    if target.kind not in TARGET_KINDS:
        raise ValueError(f"target.kind must be one of {sorted(TARGET_KINDS)}, got {target.kind!r}")
    if target.kind == "futures_chain" and not target.root:
        raise ValueError("target.root is required for kind='futures_chain'")
    if target.kind == "spot" and not (target.ticker and target.field):
        raise ValueError("target.ticker and target.field are required for kind='spot'")
    if target.cost_per_side < 0:
        raise ValueError("target.cost_per_side must be >= 0")

    # databank (+ env overrides)
    sec = _section(raw, "databank", {"toolkit_dir", "cache_dir"})
    databank = _fill(DatabankConfig, sec, "databank", {"toolkit_dir": (str,), "cache_dir": (str,)})
    if os.environ.get(ENV_TOOLKIT):
        databank.toolkit_dir = os.environ[ENV_TOOLKIT]
    if os.environ.get(ENV_CACHE):
        databank.cache_dir = os.environ[ENV_CACHE]

    # sample
    sec = _section(raw, "sample", {"start", "holdout_frac"})
    sample = _fill(SampleConfig, sec, "sample", {"start": (str,), "holdout_frac": (int, float)})
    if not 0 < sample.holdout_frac < 1:
        raise ValueError("sample.holdout_frac must be in (0, 1)")

    # universe
    sec = _section(raw, "universe", {"fields", "chains", "carry_chains", "exclude_tickers", "select", "spread"})
    for k in ("fields", "chains", "carry_chains", "exclude_tickers"):
        if k in sec:
            _str_list(sec[k], f"universe.{k}")
    select = [_parse_select(e, i) for i, e in enumerate(sec.pop("select", []))]
    spread = [_parse_spread(e, i) for i, e in enumerate(sec.pop("spread", []))]
    ids = [s["id"] for s in spread]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate universe.spread id")
    universe = UniverseConfig(**sec, select=select, spread=spread)
    if not universe.fields:
        raise ValueError("universe.fields must not be empty")

    # lags
    sec = _section(raw, "lags", {"D", "W", "class", "ticker"})
    lags_kwargs: dict[str, Any] = {}
    for k in ("D", "W"):
        if k in sec:
            _check_type(sec[k], (int,), f"lags.{k}")
            if sec[k] < 0:
                raise ValueError(f"lags.{k} must be >= 0")
            lags_kwargs[k] = sec[k]
    if "class" in sec:
        lags_kwargs["class_"] = _int_map(sec["class"], "lags.class")
        for key in lags_kwargs["class_"]:
            if key.count("/") != 1:
                raise ValueError(f"lags.class key must look like 'class_one/class_two', got {key!r}")
    if "ticker" in sec:
        lags_kwargs["ticker"] = _int_map(sec["ticker"], "lags.ticker")
    lags = LagsConfig(**lags_kwargs)

    # signals / stats / verdict
    sec = _section(raw, "signals", {"std_window", "clip", "vol_halflife"})
    signals = _fill(SignalsConfig, sec, "signals",
                    {"std_window": (int,), "clip": (int, float), "vol_halflife": (int,)})
    sec = _section(raw, "stats", {"hac_lags", "min_obs", "n_boot", "min_shift", "seed"})
    stats = _fill(StatsConfig, sec, "stats", {k: (int,) for k in sec})
    sec = _section(raw, "verdict", {"fwer_alpha", "fdr_q", "dedupe_corr", "extra_lag_t_min", "mode",
                                    "z_pass", "z_weak"})
    verdict = _fill(VerdictConfig, sec, "verdict",
                    {k: ((str,) if k == "mode" else (int, float)) for k in sec})
    if verdict.mode not in VERDICT_MODES:
        raise ValueError(f"verdict.mode must be one of {sorted(VERDICT_MODES)}, got {verdict.mode!r}")
    if not verdict.z_pass > verdict.z_weak > 0:
        raise ValueError("verdict requires z_pass > z_weak > 0")

    return Config(target=target, databank=databank, sample=sample, universe=universe,
                  lags=lags, signals=signals, stats=stats, verdict=verdict)
