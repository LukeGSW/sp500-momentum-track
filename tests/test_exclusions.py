"""
Test dell'esclusione delle serie compromesse.

Il caso riprodotto e' quello reale: RAI, fusione con concambio non gestito nel
fattore di rettifica, che produce un salto fittizio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from track import exclusions as exc


def _panels(n: int = 400) -> dict[str, pd.DataFrame]:
    idx = pd.bdate_range("2004-01-01", periods=n)
    cols = ["AAA", "RAI", "CCC"]
    base = pd.DataFrame(100.0, index=idx, columns=cols)
    return {"close_adj": base.copy(), "open_adj": base.copy(),
            "open_raw": base.copy(), "close_raw": base.copy()}


def _scrivi(tmp_path: Path, righe: str) -> Path:
    p = tmp_path / "exclusions.csv"
    p.write_text("ticker,start_date,end_date,reason\n" + righe, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
def test_file_assente_non_e_un_errore(tmp_path: Path):
    assert exc.load_exclusions(tmp_path / "inesistente.csv") == []


def test_none_esplicito_disattiva_le_esclusioni():
    assert exc.load_exclusions("NONE") == []
    assert exc.load_exclusions("none") == []


def test_default_senza_argomenti_non_e_scambiato_per_NONE():
    """str(None).upper() vale 'NONE': senza guardia il default salterebbe tutto."""
    assert exc.load_exclusions(None), \
        "il percorso predefinito deve caricare exclusions.csv, non disattivarle"


def test_lettura_con_finestra(tmp_path: Path):
    p = _scrivi(tmp_path, 'RAI,2004-06-01,2004-12-31,"fusione non gestita"\n')
    lista = exc.load_exclusions(p)

    assert len(lista) == 1
    e = lista[0]
    assert e.ticker == "RAI"
    assert e.start == pd.Timestamp("2004-06-01")
    assert e.end == pd.Timestamp("2004-12-31")
    assert not e.whole_series
    assert "fusione" in e.reason


def test_lettura_intera_serie(tmp_path: Path):
    p = _scrivi(tmp_path, 'XXX,,,"serie irrecuperabile"\n')
    e = exc.load_exclusions(p)[0]
    assert e.whole_series and e.start is None and e.end is None


def test_commenti_e_righe_vuote_ignorati(tmp_path: Path):
    p = tmp_path / "e.csv"
    p.write_text(
        "# un commento\n"
        "ticker,start_date,end_date,reason\n"
        'RAI,2004-06-01,2004-12-31,"motivo"\n',
        encoding="utf-8",
    )
    assert len(exc.load_exclusions(p)) == 1


def test_colonne_mancanti_sollevano(tmp_path: Path):
    p = tmp_path / "e.csv"
    p.write_text("ticker,start_date\nRAI,2004-01-01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="colonne mancanti"):
        exc.load_exclusions(p)


# ---------------------------------------------------------------------------
def test_applicazione_finestra_azzera_solo_il_periodo():
    panels = _panels()
    e = exc.Exclusion("RAI", pd.Timestamp("2004-06-01"), pd.Timestamp("2004-12-31"), "test")

    out, rep = exc.apply_exclusions(panels, [e])
    col = out["close_adj"]["RAI"]

    dentro = (col.index >= e.start) & (col.index <= e.end)
    assert col[dentro].isna().all(), "la finestra deve essere azzerata"
    assert col[~dentro].notna().all(), "fuori dalla finestra i dati restano"
    # gli altri titoli non vengono toccati
    assert out["close_adj"]["AAA"].notna().all()
    assert rep[0]["applicata"] and rep[0]["osservazioni_rimosse"] > 0


def test_applicazione_a_tutti_i_pannelli_prezzi():
    panels = _panels()
    e = exc.Exclusion("RAI", None, None, "test")
    out, _ = exc.apply_exclusions(panels, [e])

    for nome, df in out.items():
        assert df["RAI"].isna().all(), f"{nome} non e' stato ripulito"


def test_originali_non_modificati():
    """apply_exclusions non deve mutare i pannelli passati."""
    panels = _panels()
    prima = panels["close_adj"]["RAI"].copy()
    exc.apply_exclusions(panels, [exc.Exclusion("RAI", None, None, "test")])
    pd.testing.assert_series_equal(panels["close_adj"]["RAI"], prima)


def test_ticker_assente_viene_registrato_non_ignorato():
    panels = _panels()
    out, rep = exc.apply_exclusions(panels, [exc.Exclusion("ZZZ", None, None, "test")])

    assert len(rep) == 1
    assert rep[0]["applicata"] is False
    assert "assente" in rep[0]["nota"]


def test_nessuna_esclusione_restituisce_gli_stessi_pannelli():
    panels = _panels()
    out, rep = exc.apply_exclusions(panels, [])
    assert rep == []
    assert out is panels


# ---------------------------------------------------------------------------
def test_rileva_divergenza_fra_rettificato_e_grezzo():
    """La firma dello split non gestito: i due prezzi si muovono diversamente."""
    idx = pd.bdate_range("2004-01-01", periods=100)
    adj = pd.DataFrame({"AAA": np.full(100, 100.0), "RAI": np.full(100, 100.0)}, index=idx)
    raw = adj.copy()
    raw.iloc[50:, 1] = 50.0  # il grezzo dimezza, il rettificato no

    det = exc.detect_broken_series(adj, raw, return_threshold=0.60, divergence_threshold=0.25)
    assert "RAI" in set(det["ticker"])
    riga = det[det["ticker"] == "RAI"].iloc[0]
    assert riga["divergenti"] >= 1
    assert bool(riga["quasi_certo"])
    assert "AAA" not in set(det["ticker"])


def test_rileva_salto_che_rientra():
    idx = pd.bdate_range("2004-01-01", periods=100)
    px = np.full(100, 100.0)
    px[50] = 300.0  # prezzo sbagliato per un giorno solo
    adj = pd.DataFrame({"AAA": px}, index=idx)

    det = exc.detect_broken_series(adj, None, return_threshold=0.60)
    riga = det[det["ticker"] == "AAA"].iloc[0]
    assert riga["rimbalzi"] >= 1
    assert bool(riga["quasi_certo"])


def test_serie_pulita_non_viene_segnalata():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2004-01-01", periods=500)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 500)))
    adj = pd.DataFrame({"AAA": px}, index=idx)
    assert exc.detect_broken_series(adj, adj).empty


def test_proposte_hanno_finestra_stretta_non_intera_serie():
    idx = pd.bdate_range("2004-01-01", periods=1000)
    px = np.full(1000, 100.0)
    px[500] = 400.0
    adj = pd.DataFrame({"RAI": px}, index=idx)

    det = exc.detect_broken_series(adj, None)
    prop = exc.suggest_exclusions(det, padding_days=45)

    assert len(prop) == 1
    r = prop.iloc[0]
    durata = (pd.Timestamp(r["end_date"]) - pd.Timestamp(r["start_date"])).days
    assert durata < 200, "la finestra proposta deve essere stretta, non l'intera serie"
    assert r["reason"].startswith("AUTO:")
    assert list(prop.columns) == list(exc.COLUMNS), "deve essere incollabile in exclusions.csv"


def test_proposte_vuote_se_nessun_caso_certo():
    assert exc.suggest_exclusions(pd.DataFrame(), only_certain=True).empty


# ---------------------------------------------------------------------------
def test_manifest_distingue_file_assente_da_file_vuoto(tmp_path: Path):
    """Senza questa distinzione, 'non caricato' e 'caricato ma vuoto' sono
    indistinguibili a posteriori — ed e' esattamente il caso che si verifica."""
    assente = exc.exclusions_manifest(tmp_path / "non-esiste.csv")
    assert assente["esiste"] is False and assente["righe"] == 0
    assert assente["impronta"] is None

    vuoto = _scrivi(tmp_path, "")
    m = exc.exclusions_manifest(vuoto)
    assert m["esiste"] is True and m["righe"] == 0
    assert m["impronta"] is not None, "un file esistente ha sempre un'impronta"


def test_manifest_elenca_i_ticker_dichiarati(tmp_path: Path):
    p = _scrivi(tmp_path, 'RAI,2004-06-01,2004-12-31,"a"\nRRD,2001-10-01,2002-04-30,"b"\n')
    m = exc.exclusions_manifest(p)

    assert m["righe"] == 2
    assert m["ticker"] == ["RAI", "RRD"], "servono i nomi, non solo il conteggio"
    assert len(m["impronta"]) == 12


def test_manifest_cambia_impronta_se_cambia_il_file(tmp_path: Path):
    a = exc.exclusions_manifest(_scrivi(tmp_path, 'RAI,,,"uno"\n'))
    b = exc.exclusions_manifest(_scrivi(tmp_path, 'RAI,,,"uno"\nRRD,,,"due"\n'))
    assert a["impronta"] != b["impronta"]


def test_manifest_segnala_disattivazione_esplicita():
    m = exc.exclusions_manifest("NONE")
    assert m["disattivate"] is True and m["righe"] == 0


def test_il_file_del_progetto_e_leggibile_e_documentato():
    """exclusions.csv fa parte del metodo: deve essere valido e motivato."""
    lista = exc.load_exclusions()
    assert lista, "il file del progetto dovrebbe contenere almeno RAI"
    for e in lista:
        assert e.reason and e.reason != "non specificato", \
            f"{e.ticker} escluso senza motivo: inaccettabile"
        assert len(e.reason) > 20, f"{e.ticker}: il motivo deve essere verificabile"

    rai = [e for e in lista if e.ticker == "RAI"]
    assert rai, "RAI e' l'errore accertato di riferimento"
    assert not rai[0].whole_series, "RAI va escluso solo intorno alla fusione del 2004"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
