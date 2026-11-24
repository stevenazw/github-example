"""Formula variables: a whitelist expression engine over lag-safe leaf series.

A variable is an expression string such as ``ts('USGG10YR Index') - ts('USGG2YR Index')``. The expression is
parsed with ``ast`` and evaluated by a closed-form walker (no ``eval`` / ``exec``): only numeric constants,
``+ - * /``, unary ``+ -``, parentheses and calls of the functions below are allowed.

Leaves (``ts``, ``flow``, ``chain``, ``curve``) are served by a *provider*; each leaf comes back already
aligned to the target calendar and lagged by its own availability rule, so every operation here works on
causal, calendar-aligned series. Operation functions are causal: the output at t only depends on inputs <= t.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FREQ_RANK = {"D": 0, "W": 1, "other": 2}
ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
CHAIN_COLUMNS = ("return_index", "ret", "px_active", "calendar_spread", "oi_active", "oi_next")

# name -> (kind, number of positional args allowed)
_LEAF_ARITY = {"ts": (1, 2), "flow": (2, 2), "chain": (2, 2), "curve": (4, 4), "futures_oi": (1, 1),
               "ref": (1, 1)}
_SERIES_FUNCS = {"lag": 2, "diff": 1, "dlog": 1, "sign": 1, "nz": 1, "cumsum": 1, "rvol": 2, "clean": 3,
                 "rsum": 2, "pos": 1}
_ROWWISE_FUNCS = ("vmean", "vmax", "vmin")        # variadic (>= 2 arguments), row by row, all arguments finite
_BIN_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


class Rejected(Exception):
    """Raised by a provider when a leaf must be rejected (not a configuration error)."""

    def __init__(self, reason: str, leaf: str):
        super().__init__(f"{reason}: {leaf}")
        self.reason, self.leaf = reason, leaf


@dataclass
class LeafInfo:
    name: str
    lag: int
    rule: str
    freq: str


@dataclass
class Leaf:
    """What a provider returns for one leaf call: an aligned + lagged series and its availability info."""
    series: pd.Series
    info: LeafInfo


@dataclass
class VarResult:
    series: pd.Series | None
    lag: int = 0
    freq: str = "D"
    lag_rule: str = ""
    leaves: list = field(default_factory=list)
    rejected: list = field(default_factory=list)      # [(reason, leaf name)]

    @property
    def reason(self) -> str:
        """One exclusion reason string, ``no_availability_rule`` taking priority over the frequency problem."""
        for kind in ("no_availability_rule", "freq_not_daily_or_weekly"):
            names = [n for r, n in self.rejected if r == kind]
            if names:
                return f"{kind}: " + ", ".join(dict.fromkeys(names))
        return ""


# ------------------------------------------------------------------ pure alignment
def _ns(idx) -> np.ndarray:
    return pd.DatetimeIndex(idx).as_unit("ns").values


def align_flow_to_calendar(x: pd.Series, calendar: pd.DatetimeIndex, lag: int,
                           stale_limit: int | None = 5) -> pd.Series:
    """Align a FLOW series (e.g. daily net inflow) to ``calendar``.

    Every observation is added to the first calendar row whose date is >= the observation date (several
    observations landing on one row are summed). Observations dated before the first calendar date are
    dropped; observations after the last calendar date are dropped too. Rows without observations are 0 from
    the row of the first observation on and NaN before it; nothing is carried forward. After the row of the
    last observation, more than ``stale_limit`` rows without a new observation are NaN (the series has ended;
    ``None`` disables this). The result is shifted down by ``lag`` rows.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    calendar = pd.DatetimeIndex(calendar)
    n = len(calendar)
    x = x.dropna()
    nan_out = pd.Series(np.nan, index=calendar, dtype=float)
    if len(x) == 0 or n == 0:
        return nan_out
    cal = _ns(calendar)
    xi = _ns(x.index)
    pos = np.searchsorted(cal, xi, side="left")      # first row with date >= obs date
    keep = (pos < n) & (xi >= cal[0])
    if not keep.any():
        return nan_out
    pos, vals = pos[keep], x.to_numpy(dtype=float)[keep]
    out = np.bincount(pos, weights=vals, minlength=n).astype(float)
    out[:pos.min()] = np.nan
    if stale_limit is not None:
        out[pos.max() + stale_limit + 1:] = np.nan
    s = pd.Series(out, index=calendar, dtype=float)
    return s.shift(lag) if lag else s


# ------------------------------------------------------------------ parsing / validation
def _err(vid: str, msg: str) -> ValueError:
    return ValueError(f"variable {vid!r}: {msg}")


def _check_str_arg(node, vid: str, fn: str) -> None:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        raise _err(vid, f"{fn}() arguments must be string constants")


def _check_int_arg(node, vid: str, fn: str, minimum: int) -> None:
    ok = (isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool)
          and node.value >= minimum)
    if not ok:
        raise _err(vid, f"{fn}() needs an integer constant >= {minimum} as its second argument")


def _check(node, vid: str) -> None:
    """Strict grammar check of a numeric-valued sub-expression."""
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise _err(vid, "only numeric constants are allowed here (strings only as function arguments)")
        if not math.isfinite(float(v)):
            raise _err(vid, "constant must be finite")
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise _err(vid, f"operator {type(node.op).__name__} is not allowed (only + - * /)")
        _check(node.left, vid)
        _check(node.right, vid)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise _err(vid, f"unary operator {type(node.op).__name__} is not allowed")
        _check(node.operand, vid)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise _err(vid, "only direct calls of whitelisted functions are allowed")
        fn = node.func.id
        if node.keywords:
            raise _err(vid, f"{fn}(): keyword arguments are not allowed")
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise _err(vid, f"{fn}(): starred arguments are not allowed")
        args = node.args
        if fn in _LEAF_ARITY:
            lo, hi = _LEAF_ARITY[fn]
            if not lo <= len(args) <= hi:
                raise _err(vid, f"{fn}() takes {lo}{'' if lo == hi else f'-{hi}'} argument(s), got {len(args)}")
            for a in args:
                _check_str_arg(a, vid, fn)
            return
        if fn in _ROWWISE_FUNCS:
            if len(args) < 2:
                raise _err(vid, f"{fn}() takes at least 2 arguments, got {len(args)}")
            for a in args:
                _check(a, vid)
            return
        if fn in _SERIES_FUNCS:
            if len(args) != _SERIES_FUNCS[fn]:
                raise _err(vid, f"{fn}() takes {_SERIES_FUNCS[fn]} argument(s), got {len(args)}")
            _check(args[0], vid)
            if fn == "lag":
                _check_int_arg(args[1], vid, fn, 0)
            elif fn == "rvol":
                _check_int_arg(args[1], vid, fn, 2)
            elif fn == "rsum":
                _check_int_arg(args[1], vid, fn, 1)
            elif fn == "clean":
                _check_int_arg(args[1], vid, fn, 2)
                k = args[2]
                if not (isinstance(k, ast.Constant) and isinstance(k.value, (int, float))
                        and not isinstance(k.value, bool) and math.isfinite(float(k.value)) and k.value > 0):
                    raise _err(vid, "clean() needs a positive numeric constant as its third argument")
            return
        raise _err(vid, f"unknown function {fn!r}")
    raise _err(vid, f"syntax not allowed: {type(node).__name__}")


def parse_expr(expr: str, vid: str = "<expr>") -> ast.Expression:
    """Parse and validate ``expr`` against the whitelist; ``ValueError`` (with the variable id) otherwise."""
    if not isinstance(expr, str) or not expr.strip():
        raise _err(vid, "empty expression")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise _err(vid, f"syntax error: {e.msg}") from e
    _check(tree.body, vid)
    return tree


def expr_refs(tree: ast.Expression) -> list[str]:
    """ids used in ``ref('...')`` calls of a validated tree (in order of appearance)."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ref":
            out.append(node.args[0].value)
    return out


def validate_definitions(defs: dict[str, str]) -> None:
    """Parse every definition, check that every ``ref`` exists and that there is no reference cycle."""
    refs = {}
    for vid, expr in defs.items():
        refs[vid] = expr_refs(parse_expr(expr, vid))
        for r in refs[vid]:
            if r not in defs:
                raise _err(vid, f"ref({r!r}) refers to an unknown id")
    state: dict[str, int] = {}

    def visit(v: str, path: list[str]) -> None:
        if state.get(v) == 2:
            return
        if state.get(v) == 1:
            raise ValueError("circular ref: " + " -> ".join(path[path.index(v):] + [v]))
        state[v] = 1
        for r in refs[v]:
            visit(r, path + [v])
        state[v] = 2

    for v in defs:
        visit(v, [])


# ------------------------------------------------------------------ operations
def _need_series(x, vid: str, fn: str) -> pd.Series:
    if not isinstance(x, pd.Series):
        raise _err(vid, f"{fn}() needs a series, got a constant")
    return x


def _arith(a, b, op: str, vid: str):
    sa, sb = isinstance(a, pd.Series), isinstance(b, pd.Series)
    idx = None
    if sa and sb:
        if not a.index.equals(b.index):
            raise _err(vid, "operands are not on the same calendar")
        idx = a.index
    elif sa or sb:
        idx = (a if sa else b).index
    av = a.to_numpy(dtype=float) if sa else float(a)
    bv = b.to_numpy(dtype=float) if sb else float(b)
    with np.errstate(all="ignore"):
        if op == "+":
            r = av + bv
        elif op == "-":
            r = av - bv
        elif op == "*":
            r = av * bv
        else:
            r = np.where(np.asarray(bv) == 0, np.nan, np.divide(av, np.where(np.asarray(bv) == 0, 1.0, bv)))
        ok = np.isfinite(av) & np.isfinite(bv) & np.isfinite(r)
        r = np.where(ok, r, np.nan)
    if idx is None:
        return float(r)
    return pd.Series(np.asarray(r, dtype=float), index=idx, dtype=float)


def op_dlog(x: pd.Series) -> pd.Series:
    lx = np.log(x.where(x > 0))
    d = lx - lx.shift(1)
    return d.where(np.isfinite(d))


def op_cumsum(x: pd.Series) -> pd.Series:
    """Running sum. NaN before the first finite value; afterwards NaN inputs add 0 but the row itself stays
    NaN (the sum carries on in the next finite row). Causal: row t never looks at later rows."""
    fin = np.isfinite(x.to_numpy(dtype=float))
    if not fin.any():
        return pd.Series(np.nan, index=x.index, dtype=float)
    out = x.where(fin, 0.0).cumsum()
    out.iloc[:int(np.argmax(fin))] = np.nan
    return out.where(fin).astype(float)


def op_clean(x: pd.Series, n: int, k: float) -> pd.Series:
    """Causal outlier filter. For each row t the median ``med`` and MAD (median absolute deviation) of the
    previous ``n`` finite observations (rows < t, raw values: the filter must be able to follow a level shift) are computed, and x[t] becomes NaN
    when |x[t] - med| > k * MAD. No filtering (the value is kept) when MAD == 0 or fewer than n // 2 previous
    observations exist.

    Forward-filled values (a finite value identical to the previous row's value) are not new observations:
    they are left out of the history, so slow series do not get a collapsed MAD, and they inherit the verdict
    of the row they repeat (a flagged bad print stays NaN while it is carried forward).
    """
    v = x.to_numpy(dtype=float)
    out = v.copy()
    hist: list[float] = []
    for t in range(len(v)):
        cur = v[t]
        if not np.isfinite(cur):
            continue
        if t > 0 and np.isfinite(v[t - 1]) and cur == v[t - 1]:
            out[t] = out[t - 1]                              # repeat: same verdict, no new history
            continue
        h = hist[-n:]
        bad = False
        if len(h) >= n // 2:
            med = float(np.median(h))
            mad = float(np.median(np.abs(np.asarray(h) - med)))
            bad = mad > 0 and abs(cur - med) > k * mad
        if bad:
            out[t] = np.nan
        hist.append(cur)                                     # raw history, so the filter adapts to regime shifts
    return pd.Series(out, index=x.index, dtype=float)


def op_rvol(x: pd.Series, n: int) -> pd.Series:
    return op_dlog(x).rolling(n, min_periods=n).std(ddof=1) * np.sqrt(252.0) * 100.0


def op_rsum(x: pd.Series, n: int) -> pd.Series:
    """Sum of the last ``n`` rows (the row itself included); NaN unless all ``n`` rows are finite. Each window is
    summed on its own (no running sum), so there is no drift. Causal."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        win = np.lib.stride_tricks.sliding_window_view(v, n)
        ok = np.isfinite(win).all(axis=1)
        out[n - 1:] = np.where(ok, np.where(ok[:, None], win, 0.0).sum(axis=1), np.nan)
    return pd.Series(out, index=x.index, dtype=float)


def op_pos(x: pd.Series) -> pd.Series:
    """x where x > 0, else NaN (use as a denominator that must be positive)."""
    return x.where(x > 0)


def op_rowwise(fn: str, parts: list) -> pd.Series:
    """Row-wise mean / max / min of several series (constants broadcast); NaN unless every argument is finite."""
    idx = next(p.index for p in parts if isinstance(p, pd.Series))
    for p in parts:
        if isinstance(p, pd.Series) and not p.index.equals(idx):
            raise ValueError("operands are not on the same calendar")
    m = np.vstack([p.to_numpy(dtype=float) if isinstance(p, pd.Series) else np.full(len(idx), float(p))
                   for p in parts])
    ok = np.isfinite(m).all(axis=0)
    with np.errstate(all="ignore"):
        r = {"vmean": np.mean, "vmax": np.max, "vmin": np.min}[fn](m, axis=0)
    return pd.Series(np.where(ok, r, np.nan), index=idx, dtype=float)


# ------------------------------------------------------------------ evaluator
class _Ctx:
    def __init__(self, vid: str):
        self.vid = vid
        self.leaves: list[LeafInfo] = []
        self.rejected: list[tuple[str, str]] = []

    def add(self, infos, rejected) -> None:
        for i in infos:
            if all(i.name != j.name for j in self.leaves):
                self.leaves.append(i)
        for r in rejected:
            if r not in self.rejected:
                self.rejected.append(r)


class Evaluator:
    """Evaluate variable / helper definitions (``defs``: id -> expression) on top of a leaf ``provider``.

    The provider needs a ``calendar`` attribute and the methods ``ts(ticker, field_or_None)``,
    ``flow(ticker, field)``, ``chain(root, column)``, ``curve(name, measure, start_tenor, tenor)``; each returns
    a ``Leaf`` or raises ``Rejected``; any other exception is a configuration error and surfaces as
    ``ValueError`` carrying the variable id and the leaf.
    """

    def __init__(self, provider, defs: dict[str, str]):
        self.p = provider
        self.defs = dict(defs)
        self._trees: dict[str, ast.Expression] = {}
        self._cache: dict[str, VarResult] = {}
        self._stack: list[str] = []

    # -- public
    def evaluate(self, vid: str) -> VarResult:
        if vid in self._cache:
            return self._cache[vid]
        if vid not in self.defs:
            raise ValueError(f"unknown variable or helper id {vid!r}")
        if vid in self._stack:
            raise ValueError("circular ref: " + " -> ".join(self._stack[self._stack.index(vid):] + [vid]))
        if vid not in self._trees:
            self._trees[vid] = parse_expr(self.defs[vid], vid)
        self._stack.append(vid)
        try:
            ctx = _Ctx(vid)
            val, elag = self._ev(self._trees[vid].body, ctx)
        finally:
            self._stack.pop()
        if not isinstance(val, pd.Series):
            raise _err(vid, "expression evaluates to a constant, not a series")
        if ctx.rejected:
            res = VarResult(series=None, leaves=ctx.leaves, rejected=ctx.rejected)
        else:
            freq = max((l.freq for l in ctx.leaves), key=lambda f: FREQ_RANK[f])
            rules = list(dict.fromkeys(l.rule for l in ctx.leaves))
            if elag > max(l.lag for l in ctx.leaves):        # an explicit lag() pushes availability further out
                rules.append("expr_lag")
            res = VarResult(series=val.astype(float), lag=elag, freq=freq, lag_rule="+".join(rules),
                            leaves=ctx.leaves)
        self._cache[vid] = res
        return res

    # -- internals: _ev returns (value, effective lag in rows)
    def _ev(self, node, ctx: _Ctx):
        if isinstance(node, ast.Constant):
            return float(node.value), 0
        if isinstance(node, ast.BinOp):
            a, la = self._ev(node.left, ctx)
            b, lb = self._ev(node.right, ctx)
            return _arith(a, b, _BIN_OPS[type(node.op)], ctx.vid), max(la, lb)
        if isinstance(node, ast.UnaryOp):
            v, lv = self._ev(node.operand, ctx)
            return (v if isinstance(node.op, ast.UAdd) else _arith(-1.0, v, "*", ctx.vid)), lv
        fn = node.func.id                                    # Call (validated by parse_expr)
        if fn == "ref":
            return self._ref(node.args[0].value, ctx)
        if fn in _LEAF_ARITY:
            return self._leaf(fn, [a.value for a in node.args], ctx)
        if fn in _ROWWISE_FUNCS:
            vals = [self._ev(a, ctx) for a in node.args]
            if not any(isinstance(v, pd.Series) for v, _ in vals):
                raise _err(ctx.vid, f"{fn}() needs at least one series argument")
            return op_rowwise(fn, [v for v, _ in vals]), max(l for _, l in vals)
        xv, lx = self._ev(node.args[0], ctx)
        x = _need_series(xv, ctx.vid, fn)
        if fn == "lag":
            n = int(node.args[1].value)
            return x.shift(n), lx + n
        if fn == "diff":
            r = (x - x.shift(1)).where(np.isfinite(x) & np.isfinite(x.shift(1)))
        elif fn == "dlog":
            r = op_dlog(x)
        elif fn == "sign":
            r = pd.Series(np.sign(x.to_numpy(dtype=float)), index=x.index, dtype=float)
        elif fn == "nz":
            r = x.where(np.isfinite(x), 0.0)
        elif fn == "cumsum":
            r = op_cumsum(x)
        elif fn == "rsum":
            r = op_rsum(x, int(node.args[1].value))
        elif fn == "pos":
            r = op_pos(x)
        elif fn == "clean":
            r = op_clean(x, int(node.args[1].value), float(node.args[2].value))
        else:                                                # rvol
            r = op_rvol(x, int(node.args[1].value))
        return r, lx

    def _ref(self, rid: str, ctx: _Ctx):
        if rid not in self.defs:
            raise _err(ctx.vid, f"ref({rid!r}) refers to an unknown id")
        res = self.evaluate(rid)
        ctx.add(res.leaves, res.rejected)
        if res.series is None:                                # rejected upstream: keep evaluating with NaN
            return pd.Series(np.nan, index=self.p.calendar, dtype=float), 0
        return res.series, res.lag

    def _leaf(self, fn: str, args: list[str], ctx: _Ctx):
        what = f"{fn}({', '.join(repr(a) for a in args)})"
        try:
            leaf = getattr(self.p, fn)(*args)
        except Rejected as e:
            ctx.rejected.append((e.reason, e.leaf))
            return pd.Series(np.nan, index=self.p.calendar, dtype=float), 0
        except Exception as e:
            raise _err(ctx.vid, f"leaf {what}: {e}") from e
        ctx.add([leaf.info], [])
        return leaf.series, leaf.info.lag
