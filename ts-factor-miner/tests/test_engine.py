import numpy as np
import pytest

from tsminer.engine import _simulate, simulate


def _old_form(pos, y, sigma, cost):
    """The closed form the engine replaced: pos * y - cost * |diff(pos / sigma)|."""
    gross = pos * y
    return gross, gross - cost * np.abs(np.diff(pos / sigma, prepend=0.0))


def test_hand_computed_with_sign_flip_and_drift_fee():
    # sigma = 0.1 so notional n = pos / sigma = [2, -3, -3, 0.5] (capital 1); r = y * sigma = [1%, -2%, 3%, 1%]
    pos, y, sigma, cost = np.array([0.2, -0.3, -0.3, 0.05]), np.array([0.1, -0.2, 0.3, 0.1]), np.full(4, 0.1), 1e-3
    gross, net = simulate(pos, y, sigma, cost)
    np.testing.assert_allclose(gross, [2 * 0.01, -3 * -0.02, -3 * 0.03, 0.5 * 0.01], atol=1e-12)
    # Traded value per row: 2 (open); 2 * 1.01 = 2.02 -> -3 flips the sign: 5.02; -3 held, but the short drifted to
    # -3 * 0.98 = -2.94 so re-trading to -3 costs 0.06 (the old formula says 0); -3 * 1.03 = -3.09 -> 0.5: 3.59.
    fee = cost * np.array([2.0, 5.02, 0.06, 3.59])
    np.testing.assert_allclose(gross - net, fee, atol=1e-12)
    # the old closed form charged |diff n| = [2, 5, 0, 3.5]: differs exactly where the drift is
    np.testing.assert_allclose(gross - _old_form(pos, y, sigma, cost)[1], cost * np.array([2.0, 5.0, 0.0, 3.5]),
                               atol=1e-12)


def test_large_leverage_long_and_short_are_mirror_images():
    # a futures strategy holds notional of hundreds of times its capital unit; a TargetPercent engine would cap the
    # long side at 1x equity but not the short side, so long and short would no longer be symmetric
    rng = np.random.default_rng(3)
    n = 400
    pos, y, sigma = rng.normal(size=n), rng.normal(size=n), np.full(n, 0.002)       # |notional| up to ~2000
    g_long, _ = simulate(np.abs(pos), y, sigma, 0.0)
    g_short, _ = simulate(-np.abs(pos), y, sigma, 0.0)
    np.testing.assert_allclose(g_long, -g_short, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g_long, np.abs(pos) * y, rtol=1e-9, atol=1e-10)      # and nothing was clipped


def test_insufficient_cash_raises_instead_of_clipping():
    n, r = np.array([5.0, -5.0, 5.0]), np.full(3, 0.01)
    _simulate(n, r, 1e-3, init_cash=1e4)                                              # funded: fine
    with pytest.raises(Exception, match="requested"):                               # vectorbt RejectedOrderError
        _simulate(n, r, 1e-3, init_cash=1.0)


def test_non_finite_or_bad_inputs_raise():
    ok = np.ones(3)
    for bad in (np.array([1.0, np.nan, 1.0]), np.array([1.0, np.inf, 1.0])):
        for args in ((bad, ok, ok), (ok, bad, ok), (ok, ok, bad)):
            with pytest.raises(ValueError):
                simulate(*args, 1e-4)
    with pytest.raises(ValueError):
        simulate(ok, ok, np.array([1.0, 0.0, 1.0]), 1e-4)                           # sigma must be > 0


def test_agrees_with_closed_form_up_to_the_drift_fee():
    rng = np.random.default_rng(7)
    n = 2000
    sigma = rng.uniform(0.002, 0.008, n)                                              # daily vol of a bond future
    y = rng.normal(size=n)
    pos = np.clip(rng.normal(size=n), -4, 4)
    cost = 1e-4
    gross, net = simulate(pos, y, sigma, cost)
    g_old, n_old = _old_form(pos, y, sigma, cost)
    np.testing.assert_allclose(gross, g_old, rtol=0, atol=1e-10)
    note = pos / sigma
    r = y * sigma
    drift_fee = cost * np.abs(note - np.r_[0.0, note[:-1] * (1 + r[:-1])])
    np.testing.assert_allclose(gross - net, drift_fee, rtol=1e-9, atol=1e-12)
    assert np.abs(net - n_old).max() < 0.1 * np.abs(gross).std()                      # small next to the P&L itself


def test_empty_single_row_and_flat():
    g, n = simulate([], [], [], 1e-4)
    assert len(g) == 0 and len(n) == 0
    g, n = simulate([0.1], [0.5], [0.1], 1e-3)                                        # n = 1, r = 5%, buy 1
    assert g == pytest.approx([0.05]) and n == pytest.approx([0.05 - 1e-3])
    g, n = simulate(np.zeros(5), np.ones(5), np.full(5, 0.01), 1e-3)                  # never trades
    assert (g == 0).all() and (n == 0).all()


def test_unchanged_position_on_zero_return_days_is_not_rejected():
    # float noise makes the drift re-trade a ~1e-16 order, which vectorbt's default min_size would reject (and the
    # engine raises on rejects); real data has flat-price days with a held position
    g, n = simulate(np.full(6, 0.1), np.zeros(6), np.full(6, 0.01), 1e-4)
    assert (g == 0).all() and n[0] == pytest.approx(-1e-3) and (n[1:] == 0).all()
