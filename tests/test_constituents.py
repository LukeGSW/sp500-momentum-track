"""
Test della gestione dei costituenti storici.

Non toccano la rete: usano dati costruiti a mano che riproducono le insidie
reali della fonte (ticker riassegnati, periodi disgiunti, nomi mancanti).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from track import universe


def _spells() -> pd.DataFrame:
    """Riproduce la struttura di sp500_ticker_start_end.csv.

    AAL compare due volte in intervalli disgiunti: nella fonte reale sono due
    societa' diverse, ma i nomi non ci sono e non e' possibile saperlo.
    """
    return pd.DataFrame([
        {"code": "AAPL", "name": "Apple Inc.", "sector": "Information Technology",
         "start_date": "1996-01-02", "end_date": None},
        {"code": "AAL", "name": None, "sector": None,
         "start_date": "1996-01-02", "end_date": "1997-01-15"},
        {"code": "AAL", "name": None, "sector": None,
         "start_date": "2015-03-23", "end_date": "2024-09-23"},
        {"code": "ENRNQ", "name": None, "sector": None,
         "start_date": "1996-01-02", "end_date": "2001-11-29"},
        {"code": "MMM", "name": "3M", "sector": "Industrials",
         "start_date": "1996-01-02", "end_date": None},
    ])


def test_periodi_disgiunti_marcati_come_sospetti():
    """Senza i nomi, l'unico segnale e' il doppio periodo. Va rilevato lo stesso."""
    const = universe.normalize_constituents(_spells(), today=pd.Timestamp("2026-07-31"))

    aal = const[const["code"] == "AAL"]
    assert len(aal) == 2
    assert aal["ticker_reuse_suspect"].all(), "AAL ha due periodi: va marcato"
    assert not const.loc[const["code"] == "AAPL", "ticker_reuse_suspect"].any()


def test_solo_il_periodo_piu_recente_sopravvive():
    const = universe.normalize_constituents(_spells(), today=pd.Timestamp("2026-07-31"))
    cal = pd.bdate_range("1996-01-01", "2026-07-31")
    memb = universe.build_membership(const, cal, drop_reused_prior=True)

    # il periodo 1996-1997 di AAL viene scartato, quello 2015-2024 resta
    assert not memb.loc["1996-06-03", "AAL"]
    assert memb.loc["2016-06-01", "AAL"]
    # i codici senza ambiguita' non vengono toccati
    assert memb.loc["1996-06-03", "AAPL"]


def test_tenendo_i_periodi_precedenti_la_membership_cresce():
    const = universe.normalize_constituents(_spells(), today=pd.Timestamp("2026-07-31"))
    cal = pd.bdate_range("1996-01-01", "2026-07-31")

    strict = universe.build_membership(const, cal, drop_reused_prior=True)
    loose = universe.build_membership(const, cal, drop_reused_prior=False)
    assert loose.sum().sum() > strict.sum().sum()
    assert loose.loc["1996-06-03", "AAL"]


def test_end_date_nulla_significa_ancora_nell_indice():
    const = universe.normalize_constituents(_spells(), today=pd.Timestamp("2026-07-31"))
    cal = pd.bdate_range("2026-07-01", "2026-07-31")
    memb = universe.build_membership(const, cal)

    assert memb["AAPL"].all()
    assert memb["MMM"].all()
    assert not memb["ENRNQ"].any(), "uscito nel 2001, non puo' essere nell'indice oggi"


def test_titolo_uscito_esce_dalla_membership_alla_data_giusta():
    const = universe.normalize_constituents(_spells(), today=pd.Timestamp("2026-07-31"))
    cal = pd.bdate_range("2001-01-01", "2002-06-30")
    memb = universe.build_membership(const, cal)

    assert memb.loc["2001-11-28", "ENRNQ"]
    assert not memb.loc["2001-12-03", "ENRNQ"]


def test_settori_mancanti_diventano_non_classificato():
    const = universe.normalize_constituents(_spells(), today=pd.Timestamp("2026-07-31"))
    smap = universe.sector_map(const)

    assert smap["AAPL"] == "Information Technology"
    assert smap["MMM"] == "Industrials"
    assert smap["ENRNQ"] == "Non classificato"


def test_normalizza_i_nomi_dei_settori():
    """La fonte usa nomi in stile Morningstar, la mappa usa le 11 colonne settoriali GICS."""
    assert universe.normalize_sector("Technology") == "Information Technology"
    assert universe.normalize_sector("Consumer Cyclical") == "Consumer Discretionary"
    assert universe.normalize_sector("Consumer Defensive") == "Consumer Staples"
    assert universe.normalize_sector("Financial Services") == "Financials"
    assert universe.normalize_sector("Healthcare") == "Health Care"
    assert universe.normalize_sector("Basic Materials") == "Materials"
    # gia' in forma GICS: invariati
    assert universe.normalize_sector("Industrials") == "Industrials"
    assert universe.normalize_sector("Real Estate") == "Real Estate"
    # ignoti e vuoti
    assert universe.normalize_sector(None) == "Non classificato"
    assert universe.normalize_sector("") == "Non classificato"
    assert universe.normalize_sector("Qualcosa Di Strano") == "Non classificato"


def test_periodi_invertiti_vengono_scartati():
    bad = pd.DataFrame([
        {"code": "XXX", "name": None, "sector": None,
         "start_date": "2010-01-01", "end_date": "2005-01-01"},
        {"code": "YYY", "name": None, "sector": None,
         "start_date": "2010-01-01", "end_date": "2015-01-01"},
    ])
    const = universe.normalize_constituents(bad, today=pd.Timestamp("2026-07-31"))
    assert "XXX" not in set(const["code"])
    assert "YYY" in set(const["code"])


def test_coverage_report_rileva_i_prezzi_mancanti():
    cal = pd.bdate_range("2020-01-01", "2020-06-30")
    memb = pd.DataFrame(True, index=cal, columns=["A", "B", "C", "D"])
    prices = pd.DataFrame(100.0, index=cal, columns=["A", "B", "C", "D"])
    prices["D"] = np.nan  # un titolo su quattro senza prezzo

    cov = universe.coverage_report(memb, prices)
    # pytest.approx confronta un array intero; `Series == approx` no
    assert cov["coverage"].dropna().to_numpy() == pytest.approx(0.75)
    assert (cov["mancanti"].dropna() == 1).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
