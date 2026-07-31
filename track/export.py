"""
Export JSON pensato per essere letto da un LLM.

Non un dump: uno schema. Il campo piu' importante non sono i risultati ma
`caveats`, che elenca ogni limite noto con la DIREZIONE del bias. Senza quello,
chi legge prende i numeri per oro colato.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import TrackConfig
from . import __version__

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
def _clean(obj: Any, ndigits: int = 6) -> Any:
    """NaN/inf -> None, float arrotondati, tipi numpy -> tipi python, date ISO."""
    if obj is None:
        return None
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else round(v, ndigits)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, np.ndarray):
        return [_clean(x, ndigits) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _clean(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(x, ndigits) for x in obj]
    if isinstance(obj, pd.Series):
        return {str(k): _clean(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [_clean(rec, ndigits) for rec in obj.to_dict(orient="records")]
    return obj


def _series_to_pairs(s: pd.Series, ndigits: int = 6) -> list[list]:
    s = s.dropna()
    return [[d.strftime("%Y-%m-%d"), round(float(v), ndigits)] for d, v in s.items()]


# ---------------------------------------------------------------------------
CAVEAT_LIBRARY: list[dict[str, str]] = [
    {
        "id": "delisted_prices",
        "description": (
            "I prezzi dei titoli usciti dall'indice possono mancare. I titoli che "
            "spariscono si concentrano nel paniere piu' debole, quindi ogni buco "
            "rimuove dal backtest proprio i peggiori risultati di quel paniere."
        ),
        "direction_of_bias": "favorevole al paniere Fondo Griglia (tesi contrarian)",
        "mitigation": "stress test con rendimento terminale imposto a -30%, -50%, -100%",
    },
    {
        "id": "constituents_are_a_reconstruction",
        "description": (
            "La lista dei costituenti storici NON e' un dato ufficiale S&P: e' la "
            "ricostruzione di fja05680/sp500 (licenza MIT), costruita dal dataset di "
            "'Trading Evolved' (Clenow) per il 1996-2019 e dal tracciamento delle "
            "variazioni su Wikipedia da allora. Le ricostruzioni contengono errori: "
            "variazioni non registrate, cambi di ticker scambiati per uscita piu' "
            "ingresso. L'errore si accumula andando indietro nel tempo. Il conteggio "
            "dei titoli per anno (in diagnostics) mostra un sottoconteggio nei primi "
            "anni: in quel periodo il backtest misura un sottoinsieme dell'indice."
        ),
        "direction_of_bias": "incerta; peggiora andando indietro nel tempo",
        "mitigation": "conteggio annuo dei costituenti esposto nella diagnostica; "
                      "leggere con cautela i risultati precedenti al 2005",
    },
    {
        "id": "price_coverage_start",
        "description": (
            "I prezzi US di EODHD sono disponibili prevalentemente da gennaio 2000, "
            "anche se la lista dei costituenti parte dal 1996. Il backtest puo' "
            "partire prima solo se il coverage report lo conferma, e comunque serve "
            "circa un anno e mezzo di warm-up per gli orizzonti a 252 sedute."
        ),
        "direction_of_bias": "nessuna direzione, riduce il campione",
        "mitigation": "verify_data misura da che anno esistono davvero i prezzi",
    },
    {
        "id": "sectors_only_for_current_members",
        "description": (
            "I settori GICS provengono dalla lista CORRENTE dell'indice: le societa' "
            "uscite restano 'Non classificato' (circa il 58% dei periodi storici), a "
            "meno di arricchirli dai Fundamentals EODHD per singolo titolo."
        ),
        "direction_of_bias": "rende inaffidabile l'opzione sector-neutral sui backtest "
                             "storici lunghi; nessun effetto sulla vista corrente",
        "mitigation": "eseguire build_dataset con --enrich-sectors se il piano lo consente",
    },
    {
        "id": "todays_costs_applied_historically",
        "description": (
            "Commissioni (1,5$ per lato) e spread (3 bps) di oggi sono applicati a "
            "tutto lo storico, per scelta esplicita. Nel 2001 le commissioni retail "
            "erano 10-30$ per eseguito: a quei livelli la strategia sarebbe stata "
            "impraticabile con 100.000$ di capitale."
        ),
        "direction_of_bias": "favorevole a tutte le strategie ad alto turnover nei primi anni",
        "mitigation": "curva di break-even sui costi; lo studio risponde a 'cosa avrebbe "
                      "prodotto il segnale con gli attriti di oggi', non a 'cosa potevi fare nel 2001'",
    },
    {
        "id": "sma200_filter_not_neutral",
        "description": (
            "Il filtro 'prezzo sopra la media a 200 sedute' e' esso stesso una "
            "scommessa sul momentum. Con il filtro attivo la domanda diventa 'leader "
            "o ritracciamento dentro un trend', non 'momentum o debolezza'."
        ),
        "direction_of_bias": "confonde l'effetto del filtro con quello della selezione",
        "mitigation": "backtest eseguibile con filtro acceso e spento; confronto 2x2",
    },
    {
        "id": "price_cap_excludes_compounders",
        "description": (
            "I titoli con prezzo unitario sopra 1.500$ sono esclusi perche' non "
            "entrerebbero in uno slot da 3.333$ con lotti interi. Non sono nomi "
            "casuali: tipicamente sono compounder che non hanno mai splittato."
        ),
        "direction_of_bias": "sfavorevole al paniere Testa Corsa (compensa in parte il bias sui delistati)",
        "mitigation": "conteggio mensile degli esclusi nella diagnostica",
    },
    {
        "id": "ticker_reuse",
        "description": (
            "Simboli come C, GM, K sono stati riassegnati a societa' diverse. Le "
            "occorrenze non piu' recenti dei codici riassegnati vengono scartate per "
            "evitare serie che incollano due aziende."
        ),
        "direction_of_bias": "riduce marginalmente l'universo storico",
        "mitigation": "conteggio dei codici sospetti nella diagnostica",
    },
    {
        "id": "recent_ipos_excluded",
        "description": (
            "Serve una storia minima (273 sedute) per calcolare il punteggio: IPO e "
            "spin-off recenti sono esclusi finche' non maturano storia sufficiente."
        ),
        "direction_of_bias": "esclude titoli tipicamente volatili, in entrambe le direzioni",
        "mitigation": "nessuna: senza storia non c'e' segnale",
    },
    {
        "id": "index_addition_effect",
        "description": (
            "I titoli entrano nell'S&P 500 dopo aver sovraperformato, quindi "
            "atterrano meccanicamente nella fascia alta subito dopo l'ingresso. E' "
            "un artefatto parziale del disegno dell'indice."
        ),
        "direction_of_bias": "favorevole al paniere Testa Corsa",
        "mitigation": "verificare se StartDate e' la data di annuncio o di efficacia",
    },
    {
        "id": "multiple_testing",
        "description": (
            "La dashboard consente di esplorare molte combinazioni di parametri. "
            "Con venti test, un p-value sotto 0.05 su uno di essi e' il risultato "
            "atteso dal puro caso."
        ),
        "direction_of_bias": "sovrastima la significativita' se si sceglie la configurazione a posteriori",
        "mitigation": "configurazione preregistrata dichiarata; griglia completa dei risultati",
    },
    {
        "id": "long_short_not_implementable",
        "description": (
            "Lo spread Testa Corsa meno Fondo Griglia e' un costrutto analitico: i "
            "panieri sono simulati long-only. Non include costi e vincoli dello "
            "short selling."
        ),
        "direction_of_bias": "sovrastima la redditivita' di un'implementazione long-short",
        "mitigation": "usarlo solo per isolare l'effetto della selezione dal beta di mercato",
    },
]


# ---------------------------------------------------------------------------
def build_export(
    *,
    cfg: TrackConfig,
    provenance: dict,
    snapshot: pd.DataFrame,
    band_names: tuple[str, ...],
    backtest_metrics: dict[str, dict[str, float]],
    backtest_returns: dict[str, pd.Series] | None = None,
    null_summary: dict | None = None,
    subperiods: pd.DataFrame | None = None,
    transition_matrix: pd.DataFrame | None = None,
    diagnostics: dict | None = None,
    stress: dict | None = None,
    level: str = "compact",
) -> dict:
    """Costruisce il payload JSON completo.

    level='compact' esclude le serie mensili (adatto a una conversazione con un
    LLM); level='full' le include (adatto a rianalisi programmatica).
    """
    diagnostics = diagnostics or {}
    is_prereg = cfg.hash() == TrackConfig().hash()

    snap_cols = [c for c in ("ticker", "name", "sector", "F", "V", "band", "band_label",
                             "band_prev_label", "giorni_in_fascia", "sopra_media_mobile",
                             "tradable", "in_portfolio") if c in snapshot.columns]
    snap = snapshot[snap_cols].copy()
    for c in ("F", "V"):
        if c in snap.columns:
            snap[c] = snap[c].round(3)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": {
            "name": "La Pista",
            "version": __version__,
            "question": (
                "Conviene comprare azioni in forte momentum relativo oppure in "
                "debolezza momentanea, all'interno dell'S&P 500?"
            ),
            "method_note": (
                "Studio originale. Asse verticale = Forza relativa F (z-score "
                "trasversale robusto di log-rendimenti multi-orizzonte); il momentum "
                "non e' un secondo asse ma la pendenza della traiettoria. Nessuna "
                "relazione con schemi rotazionali commerciali."
            ),
        },
        "config": {**cfg.to_dict(), "is_preregistered": is_prereg,
                   "config_hash": cfg.hash(),
                   "band_names_bottom_to_top": list(band_names)},
        "data_provenance": provenance,
        "snapshot": {
            "as_of": provenance.get("last_data_date"),
            "universe_size": int(len(snap)),
            "titles": _clean(snap),
        },
        "distribution_by_band": _clean(
            snapshot.groupby("band_label", dropna=True)["ticker"].count().to_dict()
        ) if "band_label" in snapshot.columns else {},
        "distribution_by_band_sector": _clean(
            snapshot.pivot_table(index="band_label", columns="sector",
                                 values="ticker", aggfunc="count").fillna(0).astype(int)
            .to_dict() if "band_label" in snapshot.columns else {}
        ),
        "backtest": {
            "portfolios": _clean(backtest_metrics),
            "null_distribution": _clean(null_summary or {}),
            "subperiods": _clean(subperiods) if subperiods is not None else [],
            "delisting_stress": _clean(stress or {}),
        },
        "transition_matrix": (
            {"labels": list(transition_matrix.index),
             "rows_to_columns_probabilities": _clean(transition_matrix.to_numpy())}
            if transition_matrix is not None else {}
        ),
        "diagnostics": _clean(diagnostics),
        "caveats": CAVEAT_LIBRARY,
    }

    if level == "full" and backtest_returns:
        payload["backtest"]["monthly_returns"] = {
            name: _series_to_pairs(s) for name, s in backtest_returns.items()
        }

    payload["export_level"] = level
    return payload


def to_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# ---------------------------------------------------------------------------
SCHEMA_README = """# Schema del file JSON — La Pista

Studio sulla domanda: *conviene comprare azioni in forte momentum relativo o in
debolezza momentanea, dentro l'S&P 500?*

## Come leggere questo file

Leggi **`caveats` per primo**. Ogni voce ha `direction_of_bias`: dice verso
quale conclusione ciascun limite metodologico spinge i risultati. Numeri letti
senza quei limiti sono fuorvianti.

## Chiavi principali

| chiave | contenuto |
|---|---|
| `config` | tutti i parametri dello studio. `is_preregistered` = true significa che e' la configurazione dichiarata prima di guardare i risultati; false significa esplorazione a posteriori, da trattare con piu' cautela |
| `data_provenance` | fonte, data dello snapshot, copertura dei costituenti storici, coverage ratio dei prezzi per anno |
| `snapshot.titles` | un record per titolo eleggibile alla data piu' recente |
| `backtest.portfolios` | metriche per ciascun paniere |
| `backtest.null_distribution` | percentili della distribuzione ottenuta con 30 titoli estratti a caso, piu' i p-value empirici |
| `backtest.subperiods` | le stesse metriche per regime di mercato |
| `backtest.delisting_stress` | risultati imponendo -30/-50/-100% ai delistati senza prezzo |
| `transition_matrix` | probabilita' di passare da una fascia all'altra in un mese |

## Definizioni

- **F (Forza)**: z-score trasversale robusto (mediana + MAD, winsorizzato a ±3)
  della media dei log-rendimenti su piu' orizzonti. Misura *relativa*: F = 0 e'
  il titolo mediano della giornata. Non e' un rendimento.
- **V (Spinta)**: pendenza OLS di F sulle ultime N sedute, ri-standardizzata.
  E' la velocita' lungo la pista.
- **Fasce**: quintili trasversali di F, dal basso `Fondo Griglia`, `Rimonta`,
  `Gruppo`, `Scia`, `Testa Corsa`. Sono un ordinamento, non soglie assolute.
- **Panieri**: 30 titoli per paniere, 100.000$ per paniere, holding 3 mesi con
  tranche mensili sfalsate, lotti interi, cassa remunerata al T-bill 3 mesi.

## Cosa NON dedurre

1. Un CAGR piu' alto **non** significa segnale migliore se la volatilita' e' piu'
   alta: guarda `Sharpe` e la distribuzione nulla.
2. Battere SPY **non** e' rilevante: il confronto corretto e' la distribuzione
   nulla (estrazione casuale dallo stesso universo eleggibile).
3. Un `p_value` sotto 0.05 su una configurazione scelta dopo aver visto i
   risultati **non** e' significativo. Controlla `config.is_preregistered`.
4. I risultati precedenti al 2000 non esistono: la fonte dati non arriva prima.
"""
