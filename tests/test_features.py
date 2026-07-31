"""
Test del modulo segnale.

Il test che conta davvero e' `test_no_lookahead`: se fallisce, ogni numero
prodotto dal backtest e' privo di significato.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from track import features as ft


def _panel(n_days: int = 900, n_tickers: int = 60, seed: int = 7) -> pd.DataFrame:
    """Pannello di prezzi sintetico: random walk geometrici indipendenti."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.018, size=(n_days, n_tickers))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    cols = [f"T{i:03d}" for i in range(n_tickers)]
    return pd.DataFrame(prices, index=idx, columns=cols)


# ---------------------------------------------------------------- z-score
def test_zscore_invariante_a_shift_comune():
    """Aggiungere una costante a TUTTA la riga non deve muovere lo z-score.

    E' la proprieta' che fa sparire il benchmark da F: il rendimento di
    mercato e' identico per tutti i titoli alla stessa data, quindi si
    cancella. Se questo test passa, cambiare SPY con RSP non cambia nulla.
    """
    df = _panel(300, 50)
    shift = pd.Series(np.linspace(-0.5, 0.5, len(df)), index=df.index)

    z_a = ft.robust_zscore(df)
    z_b = ft.robust_zscore(df.add(shift, axis=0))

    pd.testing.assert_frame_equal(z_a, z_b, atol=1e-12, rtol=0)


def test_zscore_invariante_a_scala_comune():
    df = _panel(300, 50)
    z_a = ft.robust_zscore(df)
    z_b = ft.robust_zscore(df * 3.0)
    pd.testing.assert_frame_equal(z_a, z_b, atol=1e-12, rtol=0)


def test_zscore_nan_se_troppi_pochi_titoli():
    df = _panel(50, 5)
    z = ft.robust_zscore(df, min_count=20)
    assert z.isna().all().all()


def test_zscore_winsorizzato():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(40, 80)), index=pd.bdate_range("2020-01-01", periods=40))
    df.iloc[:, 0] = 500.0  # outlier estremo
    z = ft.robust_zscore(df, winsor=3.0)
    assert z.max().max() <= 3.0 + 1e-12
    assert z.min().min() >= -3.0 - 1e-12


# ------------------------------------------------------------- pendenza
def test_pendenza_su_retta_esatta():
    """Su una serie perfettamente lineare la pendenza OLS e' il coefficiente."""
    n, w = 200, 63
    idx = pd.bdate_range("2020-01-01", periods=n)
    slopes = {"a": 0.5, "b": -1.25, "c": 0.0}
    df = pd.DataFrame({k: 10.0 + m * np.arange(n) for k, m in slopes.items()}, index=idx)

    out = ft.rolling_ols_slope(df, w)

    assert out.iloc[: w - 1].isna().all().all()
    for k, m in slopes.items():
        np.testing.assert_allclose(out[k].iloc[w - 1 :].to_numpy(), m, atol=1e-9)


def test_pendenza_propaga_nan():
    n, w = 100, 10
    df = pd.DataFrame({"a": np.arange(n, dtype="float64")}, index=pd.bdate_range("2020-01-01", periods=n))
    df.iloc[50, 0] = np.nan
    out = ft.rolling_ols_slope(df, w)
    # il NaN in posizione 50 contamina le finestre che lo contengono
    assert out["a"].iloc[50 : 50 + w].isna().all()
    assert out["a"].iloc[50 + w :].notna().all()


# ---------------------------------------------------------------- FORZA
def test_force_richiede_tutti_gli_orizzonti():
    df = _panel(400, 40)
    f = ft.compute_force(df, horizons=(63, 126, 252))
    # prima di 252 sedute nessun titolo puo' avere il punteggio completo
    assert f.iloc[:252].isna().all().all()
    assert f.iloc[300:].notna().any().any()


def test_no_lookahead():
    """IL test critico.

    F e V calcolate su un pannello troncato al giorno k devono coincidere
    esattamente con quelle calcolate sul pannello completo, per tutte le date
    fino a k. Se differiscono, da qualche parte stiamo guardando nel futuro.
    """
    df = _panel(800, 50)
    horizons = (63, 126, 252)
    window = 63

    f_full = ft.compute_force(df, horizons)
    v_full = ft.compute_velocity(f_full, window)

    for k in (400, 600, 799):
        truncated = df.iloc[: k + 1]
        f_tr = ft.compute_force(truncated, horizons)
        v_tr = ft.compute_velocity(f_tr, window)

        pd.testing.assert_frame_equal(f_tr, f_full.iloc[: k + 1], atol=1e-12, rtol=0)
        pd.testing.assert_frame_equal(v_tr, v_full.iloc[: k + 1], atol=1e-12, rtol=0)


def test_no_lookahead_su_eleggibilita():
    df = _panel(800, 50)
    raw = df.copy()
    memb = pd.DataFrame(True, index=df.index, columns=df.columns)

    kwargs = dict(min_history_days=273, max_share_price=1e9, sma_filter=True, sma_window=200)
    elig_full, _ = ft.build_eligibility(df, raw, memb, **kwargs)

    k = 600
    elig_tr, _ = ft.build_eligibility(df.iloc[: k + 1], raw.iloc[: k + 1], memb.iloc[: k + 1], **kwargs)
    pd.testing.assert_frame_equal(elig_tr, elig_full.iloc[: k + 1])


# ---------------------------------------------------------------- FASCE
def test_fasce_quintili_bilanciate():
    df = _panel(600, 100)
    f = ft.compute_force(df, (63, 126, 252))
    elig = f.notna()
    bands = ft.assign_bands(f, elig, n_bands=5)

    last = bands.iloc[-1].dropna()
    counts = last.value_counts()
    assert set(counts.index) == {1.0, 2.0, 3.0, 4.0, 5.0}
    # con 100 titoli ogni quintile deve avere 20 nomi +/- arrotondamento
    assert counts.min() >= 15 and counts.max() <= 25


def test_fascia_1_e_la_piu_debole():
    df = _panel(600, 100)
    f = ft.compute_force(df, (63, 126, 252))
    elig = f.notna()
    bands = ft.assign_bands(f, elig, n_bands=5)

    row_f, row_b = f.iloc[-1], bands.iloc[-1]
    f_low = row_f[row_b == 1].max()
    f_high = row_f[row_b == 5].min()
    assert f_low < f_high


def test_fasce_ignorano_i_non_eleggibili():
    df = _panel(600, 100)
    f = ft.compute_force(df, (63, 126, 252))
    elig = f.notna()
    elig.iloc[:, :30] = False  # i primi 30 titoli non sono selezionabili

    bands = ft.assign_bands(f, elig, n_bands=5)
    assert bands.iloc[-1, :30].isna().all()
    assert bands.iloc[-1, 30:].notna().any()


def test_sector_neutral_bilancia_dentro_i_settori():
    df = _panel(600, 90)
    f = ft.compute_force(df, (63, 126, 252))
    elig = f.notna()
    sectors = pd.Series(["Alpha"] * 30 + ["Beta"] * 30 + ["Gamma"] * 30, index=df.columns)

    bands = ft.assign_bands(f, elig, n_bands=5, sectors=sectors, sector_neutral=True)
    last = bands.iloc[-1]
    for sec in ("Alpha", "Beta", "Gamma"):
        cols = sectors[sectors == sec].index
        vals = last[cols].dropna()
        # ogni settore deve contenere titoli di ogni fascia
        assert vals.nunique() == 5


# ------------------------------------------------------------- ELEGGIBILITA'
def test_cap_prezzo_esclude():
    df = _panel(400, 10)
    raw = df.copy()
    raw.iloc[:, 0] = 5000.0
    memb = pd.DataFrame(True, index=df.index, columns=df.columns)

    elig, crit = ft.build_eligibility(
        df, raw, memb, min_history_days=273, max_share_price=1500.0,
        sma_filter=False, sma_window=200,
    )
    assert not elig.iloc[-1, 0]
    assert not crit["sotto_cap_prezzo"].iloc[-1, 0]


def test_storia_insufficiente_esclude():
    df = _panel(400, 10)
    df.iloc[:200, 0] = np.nan  # titolo quotato a meta' campione
    raw = df.copy()
    memb = pd.DataFrame(True, index=df.index, columns=df.columns)

    elig, _ = ft.build_eligibility(
        df, raw, memb, min_history_days=273, max_share_price=1e9,
        sma_filter=False, sma_window=200,
    )
    assert not elig.iloc[-1, 0]
    assert elig.iloc[-1, 1]


def test_sma_filter_riduce_universo():
    df = _panel(600, 40)
    raw = df.copy()
    memb = pd.DataFrame(True, index=df.index, columns=df.columns)
    base = dict(min_history_days=273, max_share_price=1e9, sma_window=200)

    off, _ = ft.build_eligibility(df, raw, memb, sma_filter=False, **base)
    on, _ = ft.build_eligibility(df, raw, memb, sma_filter=True, **base)

    assert on.sum().sum() < off.sum().sum()
    assert (on & ~off).sum().sum() == 0  # il filtro puo' solo togliere


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
