"""Configuration loading for the single-asset time-series factor miner.

``load_config`` reads a TOML file (stdlib ``tomllib``), validates it strictly
(missing required keys, unknown sections or unknown keys raise ``ValueError``)
and returns a nested dataclass tree that can be dumped with ``Config.to_dict``.
"""
from __future__ import annotations

import math
import os
import tomllib
from dataclasses import asdict, dataclass, field
from typing import Any

from .variables import ID_RE, validate_definitions

ENV_TOOLKIT = "DATABANK_TOOLKIT"
ENV_CACHE = "DATABANK_CACHE"

TOP_SECTIONS = {"target", "databank", "sample", "universe", "lags", "position", "signals", "stats", "verdict"}
TARGET_KINDS = {"futures_chain", "spot"}
CLASS_KEYS = {"class_one", "class_two", "class_three"}


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
    is_frac: float = 0.70      # in-sample share of each candidate's own valid rows
    obs_frac: float = 0.15     # observation-period share; the rest (1 - is_frac - obs_frac) is out-of-sample


@dataclass
class UniverseConfig:
    fields: list[str] = field(default_factory=lambda: ["PX_LAST", "CFTC_CONTRACTS"])
    chains: list[str] = field(default_factory=list)
    carry_chains: list[str] = field(default_factory=list)
    exclude_tickers: list[str] = field(default_factory=list)
    select: list[dict] = field(default_factory=list)
    spread: list[dict] = field(default_factory=list)
    helper: list[dict] = field(default_factory=list)      # {id, expr}: only reachable through ref(), not candidates
    variable: list[dict] = field(default_factory=list)    # {id, expr, kind, group, description}: panel columns


VARIABLE_KINDS = {"level", "price"}


@dataclass
class LagsConfig:
    """Availability INPUT: a signal at the close of day t may only use observations dated <= t - lag."""
    chain: int                                              # futures chains ``chain:*`` and ``carry:*``
    class_: dict[str, int] = field(default_factory=dict)   # TOML key: "class"; keys "class_one/class_two"
    ticker: dict[str, int] = field(default_factory=dict)
    field_: dict[str, int] = field(default_factory=dict)   # TOML key: "field"; field name or "chain:<column>"
    curve: dict[str, int] = field(default_factory=dict)    # curve name


@dataclass
class PositionConfig:
    buffer: float = 0.5


@dataclass
class SignalsConfig:
    std_window: int = 252
    clip: float = 4.0
    vol_halflife: int = 20


@dataclass
class StatsConfig:
    hac_lags: int = 5
    min_obs: int = 250
    n_boot: int = 1000
    min_shift: int = 60
    seed: int = 20261122


@dataclass
class VerdictConfig:
    z_pass: float = 2.5
    z_weak: float = 2.0
    dedupe_corr: float = 0.7
    extra_lag_t_min: float = 2.0
    obs_min_retention: float = 0.5
    year_hit_min: float = 0.6
    min_net_sharpe: float = 0.3   # DEV (IS+OBS) annualised net Sharpe floor, applied before de-duplication


@dataclass
class Config:
    target: TargetConfig
    lags: LagsConfig
    databank: DatabankConfig = field(default_factory=DatabankConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    verdict: VerdictConfig = field(default_factory=VerdictConfig)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable nested dict (``lags.class_`` is emitted as ``class``)."""
        d = asdict(self)
        d["lags"]["class"] = d["lags"].pop("class_")
        d["lags"]["field"] = d["lags"].pop("field_")
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


def _parse_definition(entry: Any, i: int, section: str, keys: set[str], required: set[str]) -> dict:
    where = f"universe.{section}[{i}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: must be a table")
    unknown = set(entry) - keys
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")
    missing = required - set(entry)
    if missing:
        raise ValueError(f"{where}: missing required key(s) {sorted(missing)}")
    for k, v in entry.items():
        _check_type(v, (str,), f"{where}.{k}")
    if not ID_RE.fullmatch(entry["id"]):
        raise ValueError(f"{where}.id: {entry['id']!r} must be ASCII letters, digits and underscores")
    if not entry["expr"].strip():
        raise ValueError(f"{where}.expr: empty expression for {entry['id']!r}")
    return dict(entry)


def _parse_variable(entry: Any, i: int) -> dict:
    out = _parse_definition(entry, i, "variable", {"id", "expr", "kind", "group", "description"}, {"id", "expr"})
    out.setdefault("kind", "level")
    out.setdefault("group", "Variable")
    out.setdefault("description", out["id"])
    if out["kind"] not in VARIABLE_KINDS:
        raise ValueError(f"universe.variable[{i}].kind must be one of {sorted(VARIABLE_KINDS)}, got {out['kind']!r}")
    return out


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
    sec = _section(raw, "sample", {"start", "is_frac", "obs_frac"})
    sample = _fill(SampleConfig, sec, "sample",
                   {"start": (str,), "is_frac": (int, float), "obs_frac": (int, float)})
    if not (sample.is_frac > 0 and sample.obs_frac > 0 and sample.is_frac + sample.obs_frac < 1):
        raise ValueError("sample requires is_frac > 0, obs_frac > 0 and is_frac + obs_frac < 1")

    # universe
    sec = _section(raw, "universe", {"fields", "chains", "carry_chains", "exclude_tickers", "select", "spread",
                                       "helper", "variable"})
    for k in ("fields", "chains", "carry_chains", "exclude_tickers"):
        if k in sec:
            _str_list(sec[k], f"universe.{k}")
    select = [_parse_select(e, i) for i, e in enumerate(sec.pop("select", []))]
    spread = [_parse_spread(e, i) for i, e in enumerate(sec.pop("spread", []))]
    ids = [s["id"] for s in spread]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate universe.spread id")
    helper = [_parse_definition(e, i, "helper", {"id", "expr"}, {"id", "expr"})
              for i, e in enumerate(sec.pop("helper", []))]
    variable = [_parse_variable(e, i) for i, e in enumerate(sec.pop("variable", []))]
    all_ids = [d["id"] for d in helper + variable]
    if len(all_ids) != len(set(all_ids)):
        dup = sorted({i for i in all_ids if all_ids.count(i) > 1})
        raise ValueError(f"duplicate id among universe.helper / universe.variable: {dup}")
    validate_definitions({d["id"]: d["expr"] for d in helper + variable})
    universe = UniverseConfig(**sec, select=select, spread=spread, helper=helper, variable=variable)
    if not universe.fields:
        raise ValueError("universe.fields must not be empty")

    # lags (availability input: required, no frequency defaults)
    sec = _section(raw, "lags", {"chain", "class", "ticker", "field", "curve"}, {"chain"})
    _check_type(sec["chain"], (int,), "lags.chain")
    if sec["chain"] < 0:
        raise ValueError("lags.chain must be >= 0")
    lags_kwargs: dict[str, Any] = {"chain": sec["chain"]}
    if "class" in sec:
        lags_kwargs["class_"] = _int_map(sec["class"], "lags.class")
        for key in lags_kwargs["class_"]:
            if key.count("/") != 1:
                raise ValueError(f"lags.class key must look like 'class_one/class_two', got {key!r}")
    if "ticker" in sec:
        lags_kwargs["ticker"] = _int_map(sec["ticker"], "lags.ticker")
    if "field" in sec:
        lags_kwargs["field_"] = _int_map(sec["field"], "lags.field")
    if "curve" in sec:
        lags_kwargs["curve"] = _int_map(sec["curve"], "lags.curve")
    lags = LagsConfig(**lags_kwargs)

    # position
    sec = _section(raw, "position", {"buffer"})
    position = _fill(PositionConfig, sec, "position", {"buffer": (int, float)})
    if position.buffer < 0:
        raise ValueError("position.buffer must be >= 0")

    # signals / stats / verdict
    sec = _section(raw, "signals", {"std_window", "clip", "vol_halflife"})
    signals = _fill(SignalsConfig, sec, "signals",
                    {"std_window": (int,), "clip": (int, float), "vol_halflife": (int,)})
    sec = _section(raw, "stats", {"hac_lags", "min_obs", "n_boot", "min_shift", "seed"})
    stats = _fill(StatsConfig, sec, "stats", {k: (int,) for k in sec})
    sec = _section(raw, "verdict", {"z_pass", "z_weak", "dedupe_corr", "extra_lag_t_min", "obs_min_retention",
                                    "year_hit_min", "min_net_sharpe"})
    verdict = _fill(VerdictConfig, sec, "verdict", {k: (int, float) for k in sec})
    if not verdict.z_pass > verdict.z_weak > 0:
        raise ValueError("verdict requires z_pass > z_weak > 0")
    if not math.isfinite(verdict.min_net_sharpe):
        raise ValueError("verdict.min_net_sharpe must be a finite number")

    return Config(target=target, databank=databank, sample=sample, universe=universe,
                  lags=lags, position=position, signals=signals, stats=stats, verdict=verdict)
