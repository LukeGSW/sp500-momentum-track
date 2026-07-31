"""
Test sul riconoscimento del dataset caricato.

Il banner "dati sintetici" e' l'unica difesa contro il caso peggiore: leggere
numeri finti credendoli reali. Deve scattare quando serve e tacere quando non
serve, senza eccezioni.
"""
from __future__ import annotations

import pandas as pd
import pytest

from track import study

_VUOTO = pd.DataFrame()


def _ds(manifest: dict) -> study.Dataset:
    return study.Dataset(
        close_adj=_VUOTO, open_adj=_VUOTO, open_raw=_VUOTO, membership=_VUOTO,
        sectors=pd.Series(dtype="object"), names=pd.Series(dtype="object"),
        risk_free=pd.Series(dtype="float64"), manifest=manifest,
        force=_VUOTO, velocity=_VUOTO,
    )


def test_dataset_demo_riconosciuto():
    assert _ds({"demo_data": True, "source": "DATI SINTETICI (demo)"}).is_demo


def test_dataset_reale_non_marcato():
    assert not _ds({"demo_data": False, "source": "EODHD (prezzi)"}).is_demo


def test_campo_assente_significa_reale():
    """Un manifest prodotto da una versione precedente non deve allarmare."""
    assert not _ds({"source": "EODHD (prezzi)"}).is_demo
    assert not _ds({}).is_demo


@pytest.mark.parametrize("valore", ["false", "False", "FALSE", "0", "no", ""])
def test_stringhe_negative_non_marcano_come_demo(valore):
    """In Python la stringa 'false' e' truthy: senza la conversione esplicita
    un manifest con demo_data testuale marcherebbe come finti dei dati reali."""
    assert not _ds({"demo_data": valore}).is_demo


@pytest.mark.parametrize("valore", ["true", "True", "1", "si"])
def test_stringhe_positive_marcano_come_demo(valore):
    assert _ds({"demo_data": valore}).is_demo


def test_zero_e_none_non_marcano():
    assert not _ds({"demo_data": 0}).is_demo
    assert not _ds({"demo_data": None}).is_demo


def test_fonte_e_data_sempre_leggibili():
    """Il banner mostra fonte e data: non devono mai sollevare."""
    ds = _ds({"source": "EODHD (prezzi)", "built_at": "2026-07-31T16:31:13Z"})
    assert ds.source == "EODHD (prezzi)"
    assert ds.built_at == "2026-07-31T16:31:13Z"

    vuoto = _ds({})
    assert vuoto.source == "sconosciuta"
    assert vuoto.built_at == "data sconosciuta"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
