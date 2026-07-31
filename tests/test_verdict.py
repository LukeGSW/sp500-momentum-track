"""
Test del verdetto.

Il requisito piu' importante non e' che il verdetto sia positivo: e' che sia
ONESTO. Deve dichiarare i propri limiti anche quando i numeri sono belli, e
deve cambiare direzione se i dati cambiano direzione.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from track import verdict as vd

TOP, BOT, UNI = "Q5 Leader", "Q1 Laggard", "Universo eleggibile (senza costi)"


def _serie(mu_mensile: float, sigma: float, n: int = 316, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-04-30", periods=n, freq="ME")
    return pd.Series(rng.normal(mu_mensile, sigma, n), index=idx)


def _scenario(delta: float = 0.005, seed: int = 1) -> tuple[dict, pd.DataFrame]:
    """Mondo in cui TOP batte BOT di `delta` al mese."""
    returns = {
        TOP: _serie(0.010 + delta, 0.055, seed=seed),
        BOT: _serie(0.010, 0.050, seed=seed + 1),
        UNI: _serie(0.009, 0.045, seed=seed + 2),
    }
    metrics = pd.DataFrame({
        "CAGR": {TOP: 0.14, BOT: 0.08, UNI: 0.11},
        "Sharpe": {TOP: 0.63, BOT: 0.47, UNI: 0.60},
        "Max DD": {TOP: -0.54, BOT: -0.57, UNI: -0.44},
        "Costo annuo %": {TOP: 0.0035, BOT: 0.0067, UNI: 0.0},
    })
    return returns, metrics


def _build(returns, metrics, **kw):
    return vd.build_verdict(returns, metrics, vincitore=TOP, perdente=BOT,
                            riferimento=UNI, **kw)


# ---------------------------------------------------------------------------
def test_struttura_di_base():
    v = _build(*_scenario())

    assert v.vincitore == TOP and v.perdente == BOT
    assert v.n_mesi == 316
    assert v.valutabili >= 6, "servono abbastanza indicatori per un giudizio"
    assert v.affermazioni, "deve produrre affermazioni"
    assert v.non_affermabili, "deve SEMPRE dichiarare cosa non puo' affermare"


def test_dichiara_sempre_i_limiti_anche_quando_vince():
    """Il requisito centrale: nessun verdetto senza limiti dichiarati."""
    v = _build(*_scenario(delta=0.02))  # vantaggio enorme

    assert v.concordi >= v.valutabili - 1
    testo = " ".join(v.non_affermabili).lower()
    assert "ripeter" in testo, "deve sempre avvertire che il passato non si ripete"


def test_segnala_la_non_significativita():
    v = _build(*_scenario(delta=0.002))  # vantaggio piccolo, t basso

    assert not v.significativo
    testo = " ".join(v.non_affermabili).lower()
    assert "significativ" in testo
    assert any("t-stat" in s.lower() for s in v.non_affermabili)


def test_segnala_intervallo_che_comprende_lo_zero():
    v = _build(*_scenario(delta=0.001))
    lo, hi = v.ic95
    if lo < 0 < hi:
        assert any("comprende lo zero" in s for s in v.non_affermabili)


def test_intervallo_di_confidenza_coerente():
    v = _build(*_scenario())
    lo, hi = v.ic95
    assert lo < v.differenza_annua < hi
    assert hi - lo > 0


# ---------------------------------------------------------------------------
def test_evidenze_cambiano_verso_se_i_dati_cambiano_verso():
    """Il verdetto non deve essere cablato: deve seguire i dati."""
    returns, metrics = _scenario(delta=-0.006)  # ora TOP perde
    v = _build(returns, metrics)

    segno = [e for e in v.evidenze if e.nome == "Segno della stima"][0]
    assert segno.favorevole is False
    assert v.differenza_annua < 0
    assert v.forza in ("contraddittoria", "moderata")


def test_costo_inferiore_e_evidenza_a_favore():
    returns, metrics = _scenario()
    v = _build(returns, metrics)
    costo = [e for e in v.evidenze if e.nome == "Costo di esercizio"][0]
    assert costo.favorevole is True  # TOP costa 0.35% contro 0.67%


def test_costo_superiore_e_evidenza_contraria():
    returns, metrics = _scenario()
    metrics.loc[TOP, "Costo annuo %"] = 0.02  # ora il vincitore costa di piu'
    v = _build(returns, metrics)
    costo = [e for e in v.evidenze if e.nome == "Costo di esercizio"][0]
    assert costo.favorevole is False


def test_stress_sui_delistati_entra_fra_le_evidenze():
    returns, metrics = _scenario()
    stress = pd.DataFrame({"+0%": {TOP: 0.14, BOT: 0.08}, "-100%": {TOP: 0.10, BOT: 0.07}})
    v = _build(returns, metrics, stress=stress)

    e = [x for x in v.evidenze if "stress" in x.nome.lower()][0]
    assert e.favorevole is True  # 0.10 > 0.07: il divario regge


def test_stress_che_ribalta_e_evidenza_contraria():
    returns, metrics = _scenario()
    stress = pd.DataFrame({"+0%": {TOP: 0.14, BOT: 0.08}, "-100%": {TOP: 0.05, BOT: 0.09}})
    v = _build(returns, metrics, stress=stress)
    e = [x for x in v.evidenze if "stress" in x.nome.lower()][0]
    assert e.favorevole is False


def test_persistenza_entra_fra_le_evidenze():
    returns, metrics = _scenario()
    pers = pd.DataFrame(
        [[0.44, 0.30, 0.16, 0.07, 0.03], [0.24, 0.33, 0.26, 0.13, 0.04],
         [0.12, 0.23, 0.31, 0.25, 0.09], [0.05, 0.12, 0.24, 0.37, 0.22],
         [0.01, 0.03, 0.08, 0.22, 0.66]],
        index=["Q1 Laggard", "Q2 Sottoperformanti", "Q3 In linea", "Q4 Sovraperformanti", "Q5 Leader"],
        columns=["Q1 Laggard", "Q2 Sottoperformanti", "Q3 In linea", "Q4 Sovraperformanti", "Q5 Leader"],
    )
    v = _build(returns, metrics, persistenza=pers)
    e = [x for x in v.evidenze if "persistenza" in x.nome.lower()][0]
    assert e.favorevole is True  # 66% contro 44%


def test_argomenti_assenti_non_rompono_nulla():
    """Ogni argomento opzionale mancante toglie un'evidenza, non solleva."""
    v = _build(*_scenario())
    assert v.valutabili >= 6
    assert all(e.dettaglio for e in v.evidenze)


# ---------------------------------------------------------------------------
def test_forza_convergente_quando_quasi_tutto_concorda():
    returns, metrics = _scenario(delta=0.015)
    stress = pd.DataFrame({"+0%": {TOP: 0.18, BOT: 0.08}, "-100%": {TOP: 0.14, BOT: 0.07}})
    v = _build(returns, metrics, stress=stress,
               null_pvalues={"Sharpe": {TOP: 0.03}})
    assert v.forza == "convergente"
    assert v.quota_concordi >= vd.SOGLIA_CONVERGENTE


def test_capitale_finale_calcolato_su_tutti_i_panieri():
    returns, metrics = _scenario()
    v = _build(returns, metrics, capitale=100_000.0)
    assert set(v.capitale_finale) == {TOP, BOT, UNI}
    assert all(x > 0 for x in v.capitale_finale.values())
    assert v.capitale_finale[TOP] > v.capitale_finale[BOT]


# ---------------------------------------------------------------------------
def test_serializzazione_completa_e_con_avvertenza():
    v = _build(*_scenario())
    d = vd.to_dict(v)

    for chiave in ("vincitore", "differenza_annua", "intervallo_confidenza_95",
                   "t_stat_newey_west", "statisticamente_significativo",
                   "forza_evidenza", "evidenze", "possiamo_affermare",
                   "non_possiamo_affermare", "avvertenza"):
        assert chiave in d, f"manca {chiave}"

    assert "NON e' una dimostrazione statistica" in d["avvertenza"]
    assert d["non_possiamo_affermare"], "l'export deve portarsi dietro i limiti"

    import json
    json.loads(json.dumps(d))  # deve essere JSON valido


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
