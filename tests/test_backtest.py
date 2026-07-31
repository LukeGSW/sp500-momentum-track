"""
Test del motore di backtest.

Verifica conservazione del capitale, effetto dei costi, gestione dei delistati
e il fatto che il segnale sia letto alla data di decisione e i prezzi a quella
di esecuzione.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from track import backtest as bt
from track import features as ft
from track.config import TrackConfig


# ---------------------------------------------------------------------------
# Mondi sintetici
# ---------------------------------------------------------------------------
def _make_world(
    n_days: int = 1800,
    n_tickers: int = 80,
    seed: int = 3,
    drift_spread: float = 0.0009,
    flat: bool = False,
) -> dict:
    """Pannelli di prezzo. Con `drift_spread` > 0 i ticker hanno drift diversi
    e persistenti, quindi il momentum e' un fenomeno reale in questo mondo."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n_days)
    cols = [f"T{i:03d}" for i in range(n_tickers)]

    if flat:
        close = np.full((n_days, n_tickers), 100.0)
    else:
        drifts = np.linspace(-drift_spread, drift_spread, n_tickers)
        shocks = rng.normal(0.0, 0.014, size=(n_days, n_tickers))
        close = 100.0 * np.exp(np.cumsum(drifts + shocks, axis=0))

    close_adj = pd.DataFrame(close, index=idx, columns=cols)
    # apertura = chiusura precedente (mondo senza gap): tiene il test pulito
    open_adj = close_adj.shift(1)
    open_adj.iloc[0] = close_adj.iloc[0]
    open_raw = open_adj.copy()  # nessun dividendo: grezzo == adjusted

    return {
        "close_adj": close_adj,
        "open_adj": open_adj,
        "open_raw": open_raw,
        "index": idx,
        "cols": cols,
    }


def _make_panel(world: dict, cfg: TrackConfig, membership: pd.DataFrame | None = None) -> bt.Panel:
    close_adj = world["close_adj"]
    if membership is None:
        membership = pd.DataFrame(True, index=close_adj.index, columns=close_adj.columns)

    force = ft.compute_force(close_adj, cfg.horizons)
    velocity = ft.compute_velocity(force, cfg.velocity_window)
    eligible, _ = ft.build_eligibility(
        close_adj,
        world["open_raw"],
        membership,
        min_history_days=cfg.min_history_days,
        max_share_price=cfg.max_share_price,
        sma_filter=cfg.sma_filter,
        sma_window=cfg.sma_window,
    )
    bands = ft.assign_bands(force, eligible, cfg.n_bands)
    rf = pd.Series(0.0, index=close_adj.index)

    return bt.prepare_panel(
        close_adj=close_adj,
        open_adj=world["open_adj"],
        open_raw=world["open_raw"],
        force=force,
        velocity=velocity,
        bands=bands,
        eligible=eligible,
        rf_daily=rf,
        cfg=cfg,
    )


def _cfg(**kw) -> TrackConfig:
    base = dict(backtest_start="2012-01-01", n_names=10, n_bootstrap=20)
    base.update(kw)
    return TrackConfig(**base)


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------
def test_calendario_esecuzione_dopo_decisione():
    days = pd.bdate_range("2015-01-01", periods=800)
    dec, ex = bt.build_rebalance_calendar(days, "2015-06-01")

    assert len(dec) == len(ex)
    assert (ex.to_numpy() > dec.to_numpy()).all(), "l'esecuzione deve seguire la decisione"
    assert (ex >= pd.Timestamp("2015-06-01")).all()
    # la decisione e' l'ultima seduta del mese: il giorno dopo cambia mese
    assert all(d.month != e.month or d.year != e.year for d, e in zip(dec, ex, strict=True))


def test_calendario_una_decisione_al_mese():
    days = pd.bdate_range("2015-01-01", periods=1000)
    dec, _ = bt.build_rebalance_calendar(days, "2015-01-01")
    periods = pd.Series(dec).dt.to_period("M")
    assert periods.is_unique


# ---------------------------------------------------------------------------
# Conservazione del capitale
# ---------------------------------------------------------------------------
def test_mercato_piatto_senza_attriti_conserva_il_capitale():
    cfg = _cfg()
    world = _make_world(flat=True)
    panel = _make_panel(world, cfg)

    fixed = lambda t, p, rng: np.arange(3)  # noqa: E731
    res = bt.run_strategy(panel, fixed, cfg, name="piatto", frictionless=True)

    assert np.allclose(res.equity.to_numpy(), cfg.capital, rtol=0, atol=1e-6)


def test_i_costi_riducono_il_capitale():
    cfg = _cfg()
    world = _make_world(flat=True)
    panel = _make_panel(world, cfg)
    fixed = lambda t, p, rng: np.arange(3)  # noqa: E731

    free = bt.run_strategy(panel, fixed, cfg, frictionless=True)
    paid = bt.run_strategy(panel, fixed, cfg, frictionless=False)

    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert paid.costs.sum() > 0
    assert free.costs.sum() == 0


def test_nessuna_posizione_se_universo_vuoto():
    cfg = _cfg()
    world = _make_world()
    empty = pd.DataFrame(False, index=world["close_adj"].index, columns=world["close_adj"].columns)
    panel = _make_panel(world, cfg, membership=empty)

    res = bt.run_strategy(panel, bt.selector_band(5, cfg.n_names, True), cfg)
    assert (res.n_positions == 0).all()
    assert np.allclose(res.equity.to_numpy(), cfg.capital)


# ---------------------------------------------------------------------------
# Il segnale funziona nel mondo in cui deve funzionare
# ---------------------------------------------------------------------------
def test_fascia_alta_batte_fascia_bassa_con_drift_persistente():
    """In un mondo dove il drift e' costante per titolo, il momentum e' reale.

    Se il motore o il segnale fossero rotti, questo test fallirebbe: e' il
    controllo di sanita' che dice che stiamo misurando qualcosa.
    """
    cfg = _cfg()
    world = _make_world(drift_spread=0.0012, seed=11)
    panel = _make_panel(world, cfg)

    top = bt.run_strategy(panel, bt.selector_band(5, cfg.n_names, True), cfg, name="Testa Corsa")
    bottom = bt.run_strategy(panel, bt.selector_band(1, cfg.n_names, False), cfg, name="Fondo Griglia")

    assert top.equity.iloc[-1] > bottom.equity.iloc[-1]


def test_selettore_velocita_negativa_prende_solo_spinte_negative():
    cfg = _cfg()
    world = _make_world(seed=5)
    panel = _make_panel(world, cfg)
    sel = bt.selector_band_negative_velocity(5, cfg.n_names)
    rng = np.random.default_rng(0)

    checked = 0
    for t in range(panel.n_periods):
        idx = sel(t, panel, rng)
        if idx.size:
            assert (panel.velocity[t, idx] < 0).all()
            assert (panel.bands[t, idx] == 5.0).all()
            checked += 1
    assert checked > 0, "il selettore non ha mai prodotto un paniere"


# ---------------------------------------------------------------------------
# Survivorship
# ---------------------------------------------------------------------------
def test_haircut_delisting_peggiora_il_risultato():
    cfg = _cfg()
    world = _make_world(seed=9)
    close = world["close_adj"]

    # meta' dei titoli sparisce a meta' campione
    dying = close.columns[::2]
    cut = len(close) // 2
    for panel_key in ("close_adj", "open_adj", "open_raw"):
        world[panel_key] = world[panel_key].copy()
        world[panel_key].loc[world[panel_key].index[cut:], dying] = np.nan

    panel = _make_panel(world, cfg)
    sel = bt.selector_band(5, cfg.n_names, True)

    mild = bt.run_strategy(panel, sel, cfg, delisting_haircut=0.0)
    harsh = bt.run_strategy(panel, sel, cfg, delisting_haircut=-1.0)

    assert harsh.diagnostics["liquidazioni_forzate"] > 0
    assert harsh.equity.iloc[-1] < mild.equity.iloc[-1]


def test_buco_temporaneo_non_liquida():
    """Un dato mancante per un periodo non deve chiudere una posizione viva."""
    cfg = _cfg()
    world = _make_world(seed=4)
    panel = _make_panel(world, cfg)

    # tutti i ticker hanno prezzi fino in fondo: nessuno e' morto
    assert (panel.dead_after == panel.n_periods - 1).all()

    res = bt.run_strategy(panel, bt.selector_band(5, cfg.n_names, True), cfg)
    assert res.diagnostics["liquidazioni_forzate"] == 0


# ---------------------------------------------------------------------------
# Vincoli operativi
# ---------------------------------------------------------------------------
def test_cap_prezzo_impedisce_acquisto():
    cfg = _cfg(max_share_price=50.0)  # tutti i titoli partono da 100
    world = _make_world(flat=True)
    panel = _make_panel(world, cfg)

    res = bt.run_strategy(panel, lambda t, p, rng: np.arange(3), cfg)
    assert (res.n_positions == 0).all()


def test_lotti_interi_lasciano_residuo_in_cassa():
    cfg = _cfg(n_names=10)
    world = _make_world(flat=True)
    world["open_raw"] = world["open_raw"] * 0 + 333.0  # prezzo scomodo
    world["open_adj"] = world["open_adj"] * 0 + 333.0
    world["close_adj"] = world["close_adj"] * 0 + 333.0
    panel = _make_panel(world, cfg)

    res = bt.run_strategy(panel, lambda t, p, rng: np.arange(3), cfg, frictionless=True)
    assert (res.cash_weight.iloc[-1] > 0), "l'arrotondamento deve lasciare cassa"
    assert np.allclose(res.equity.to_numpy(), cfg.capital, atol=1e-6)


# ---------------------------------------------------------------------------
# Metriche e inferenza
# ---------------------------------------------------------------------------
def test_newey_west_su_serie_nota():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    t0 = bt.newey_west_tstat(x, lags=0)
    t3 = bt.newey_west_tstat(x, lags=3)
    assert np.isfinite(t0) and np.isfinite(t3)
    assert abs(t0) < 4  # rumore bianco: non deve risultare significativo


def test_newey_west_rileva_media_diversa_da_zero():
    x = np.full(400, 0.01) + np.random.default_rng(1).normal(0, 0.005, 400)
    assert bt.newey_west_tstat(x, lags=3) > 5


def test_max_drawdown():
    eq = pd.Series([100, 120, 60, 80, 130], index=pd.date_range("2020-01-31", periods=5, freq="ME"))
    depth, months = bt.max_drawdown(eq)
    assert depth == pytest.approx(-0.5)
    assert months > 0


def test_bootstrap_produce_le_estrazioni_richieste():
    cfg = _cfg(n_bootstrap=15)
    world = _make_world(seed=2)
    panel = _make_panel(world, cfg)

    null = bt.bootstrap_null(panel, cfg)
    assert len(null) == 15
    assert null["CAGR"].notna().sum() >= 10
    assert null["CAGR"].nunique() > 1, "le estrazioni devono essere diverse tra loro"


def test_pvalue_empirico():
    null = pd.Series(np.linspace(0.0, 1.0, 100))
    assert bt.empirical_pvalue(null, 1.5) < 0.02
    assert bt.empirical_pvalue(null, -1.0) > 0.98
    assert 0.4 < bt.empirical_pvalue(null, 0.5) < 0.6


def test_pnl_a_capitale_fisso_e_additivo():
    cfg = _cfg()
    world = _make_world(seed=6)
    panel = _make_panel(world, cfg)
    res = bt.run_strategy(panel, bt.selector_band(5, cfg.n_names, True), cfg)

    pnl = res.fixed_capital_pnl
    expected = cfg.capital * res.returns.cumsum()
    pd.testing.assert_series_equal(pnl, expected)


def test_metriche_chiavi_presenti():
    cfg = _cfg()
    world = _make_world(seed=8)
    panel = _make_panel(world, cfg)
    res = bt.run_strategy(panel, bt.selector_band(5, cfg.n_names, True), cfg)
    rf = pd.Series(panel.rf_period, index=panel.exec_dates)

    m = bt.performance_metrics(res, rf)
    for key in ("CAGR", "Vol annua", "Sharpe", "Max DD", "Turnover medio",
                "Costo annuo %", "Costi totali $"):
        assert key in m and np.isfinite(m[key]), key


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
