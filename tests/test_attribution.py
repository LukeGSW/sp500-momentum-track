"""
Test dell'attribuzione del P&L e della caccia alle anomalie.

Il caso costruito e' esattamente quello trovato sui dati reali: un titolo con
un prezzo sbagliato che da solo produce il rendimento anomalo di un mese.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from track import analysis
from track import backtest as bt
from track import features as ft
from track.config import TrackConfig

from .test_backtest import _cfg, _make_panel, _make_world


def _panel_con_anomalia(salto: float = 8.0, colonna: int = 0):
    """Mondo normale in cui un titolo triplica di colpo a meta' campione."""
    cfg = _cfg()
    world = _make_world(flat=True)
    idx = world["close_adj"].index
    quando = idx[len(idx) // 2]
    tk = world["close_adj"].columns[colonna]

    for key in ("close_adj", "open_adj", "open_raw"):
        world[key] = world[key].copy()
        world[key].loc[quando:, tk] *= salto

    return cfg, world, tk, quando


# ---------------------------------------------------------------------------
def test_attribuzione_richiede_il_tracciamento():
    cfg = _cfg()
    panel = _make_panel(_make_world(flat=True), cfg)
    res = bt.run_strategy(panel, lambda t, p, rng: np.arange(3), cfg)

    assert res.contributions is None
    with pytest.raises(ValueError, match="track_contributions"):
        analysis.attribution(res, panel.exec_dates[5])


def test_contributi_sommano_al_pnl_del_periodo():
    """Senza attriti la somma dei contributi deve ricostruire il P&L."""
    cfg = _cfg()
    world = _make_world(seed=21)
    panel = _make_panel(world, cfg)
    res = bt.run_strategy(panel, lambda t, p, rng: np.arange(5), cfg,
                          frictionless=True, track_contributions=True)

    eq = res.equity
    contrib = res.contributions.sum(axis=1)
    # dal secondo periodo in poi (il primo non ha posizioni precedenti)
    delta = eq.diff().iloc[2:]
    np.testing.assert_allclose(contrib.iloc[2:].to_numpy(), delta.to_numpy(), atol=1e-6)


def test_attribuzione_individua_il_titolo_anomalo():
    cfg, world, tk, quando = _panel_con_anomalia(salto=8.0, colonna=0)
    panel = _make_panel(world, cfg)
    # selettore fisso che include il titolo malato
    res = bt.run_strategy(panel, lambda t, p, rng: np.arange(4), cfg,
                          frictionless=True, track_contributions=True)

    anom = analysis.anomalous_periods({"test": res}, threshold=0.20)
    assert not anom.empty, "il salto di prezzo deve produrre un mese anomalo"

    mese = anom.iloc[0]["data"]
    attr = analysis.attribution(res, mese, top=5)
    assert attr.iloc[0]["ticker"] == tk, "il titolo col prezzo sbagliato deve essere primo"
    assert attr.iloc[0]["quota_del_periodo"] > 0.9


def test_anomalie_marcano_come_non_verosimile_oltre_il_30():
    cfg, world, tk, _ = _panel_con_anomalia(salto=20.0)
    panel = _make_panel(world, cfg)
    res = bt.run_strategy(panel, lambda t, p, rng: np.arange(4), cfg,
                          frictionless=True, track_contributions=True)

    anom = analysis.anomalous_periods({"test": res}, threshold=0.15)
    grosse = anom[anom["rendimento"].abs() > 0.30]
    assert len(grosse) >= 1
    assert not grosse["verosimile"].any()


def test_nessuna_anomalia_in_un_mondo_tranquillo():
    cfg = _cfg()
    panel = _make_panel(_make_world(seed=33), cfg)
    res = bt.run_strategy(panel, bt.selector_band(5, cfg.n_names, True), cfg,
                          track_contributions=True)
    anom = analysis.anomalous_periods({"test": res}, threshold=0.30)
    assert anom.empty


def test_price_context_rileva_lo_scarto_fra_grezzo_e_adjusted():
    """Split gestito nell'adjusted ma non nel grezzo: lo scarto lo smaschera."""
    idx = pd.bdate_range("2020-01-01", periods=20)
    adj = pd.DataFrame({"AAA": np.full(20, 100.0)}, index=idx)
    raw = adj.copy()
    raw.iloc[10:, 0] = 50.0  # il grezzo dimezza, l'adjusted no

    ctx = analysis.price_context(adj, raw, "AAA", idx[10], window=5)
    assert not ctx.empty
    assert ctx["scarto"].max() > 0.4
    assert ctx["var_grezzo"].min() == pytest.approx(-0.5)
    assert ctx["var_adjusted"].abs().max() == pytest.approx(0.0)


def test_price_context_su_ticker_assente():
    idx = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame({"AAA": np.ones(10)}, index=idx)
    assert analysis.price_context(df, df, "ZZZ", idx[5]).empty


# ---------------------------------------------------------------------------
def test_decomposizione_filtro_ha_entrambe_le_modalita():
    """Il 2x2 deve restituire le due modalita' per ogni paniere."""
    from track import study

    cfg = _cfg()
    world = _make_world(seed=44)
    memb = pd.DataFrame(True, index=world["close_adj"].index, columns=world["close_adj"].columns)
    sectors = pd.Series("Industrials", index=world["close_adj"].columns)
    rf = pd.Series(0.0, index=world["close_adj"].index)

    ds = study.Dataset(
        close_adj=world["close_adj"], open_adj=world["open_adj"], open_raw=world["open_raw"],
        membership=memb, sectors=sectors,
        names=pd.Series(world["close_adj"].columns, index=world["close_adj"].columns),
        risk_free=rf, manifest={},
        force=ft.compute_force(world["close_adj"], cfg.horizons),
        velocity=ft.compute_velocity(ft.compute_force(world["close_adj"], cfg.horizons),
                                     cfg.velocity_window),
    )

    dec = study.filter_decomposition(ds, cfg)
    assert set(dec["Filtro media mobile"]) == {"attivo", "spento"}
    assert "Spread Q5 − Q1" in set(dec["Paniere"])

    # spegnendo il filtro l'universo eleggibile non puo' che crescere
    u_on = dec[dec["Filtro media mobile"] == "attivo"]["Universo medio"].iloc[0]
    u_off = dec[dec["Filtro media mobile"] == "spento"]["Universo medio"].iloc[0]
    assert u_off > u_on


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
