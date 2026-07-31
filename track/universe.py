"""
Universo storico S&P 500: appartenenze, settori, copertura dati.

Qui vive la difesa contro il survivorship bias. Due meta' del problema:

  1. sapere CHI era nell'indice a una certa data  -> risolta dalla lista storica
  2. avere i PREZZI di chi poi e' sparito         -> risolta solo in parte

La seconda meta' e' quella pericolosa: i titoli che spariscono si concentrano
nel paniere dei piu' deboli, quindi ogni buco spinge il risultato a favore
della tesi contrarian. Per questo `coverage_report()` non e' un accessorio di
diagnostica: e' il numero che dice quanto vale tutto il resto.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import GICS_SECTORS, SECTOR_UNKNOWN

log = logging.getLogger(__name__)

# EODHD usa la tassonomia stile Morningstar/Yahoo, non i nomi GICS ufficiali.
# Mappiamo sulle 11 corsie della pista.
SECTOR_ALIASES: dict[str, str] = {
    "technology": "Information Technology",
    "information technology": "Information Technology",
    "consumer cyclical": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "consumer staples": "Consumer Staples",
    "healthcare": "Health Care",
    "health care": "Health Care",
    "financial services": "Financials",
    "financial": "Financials",
    "financials": "Financials",
    "basic materials": "Materials",
    "materials": "Materials",
    "industrials": "Industrials",
    "industrial goods": "Industrials",
    "energy": "Energy",
    "utilities": "Utilities",
    "real estate": "Real Estate",
    "communication services": "Communication Services",
    "communication": "Communication Services",
}


def normalize_sector(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return SECTOR_UNKNOWN
    key = raw.strip().lower()
    if key in SECTOR_ALIASES:
        return SECTOR_ALIASES[key]
    for gics in GICS_SECTORS:
        if key == gics.lower():
            return gics
    return SECTOR_UNKNOWN


# ---------------------------------------------------------------------------
# Costituenti
# ---------------------------------------------------------------------------
def normalize_constituents(df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Pulisce la lista storica e marca i ticker riassegnati.

    Simboli come C, GM, K sono stati riassegnati a societa' diverse. Se
    facessimo il match sul solo codice otterremmo serie Frankenstein che
    incollano due aziende. Marchiamo il sospetto e, di default, teniamo solo
    l'occorrenza piu' recente: e' una perdita piccola e dichiarata, molto
    meglio di un dato sbagliato.
    """
    today = today or pd.Timestamp.today().normalize()
    out = df.copy()

    out["code"] = out["code"].astype("string").str.strip().str.upper()
    out["name"] = out["name"].astype("string").str.strip()
    out["sector"] = out["sector"].map(normalize_sector)
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")

    out = out.dropna(subset=["code", "start_date"])
    out["end_date"] = out["end_date"].fillna(today)
    out = out[out["end_date"] >= out["start_date"]]

    out = out.sort_values(["code", "start_date"]).reset_index(drop=True)

    # Sospetto riassegnazione: un codice con piu' NOMI distinti, oppure con
    # piu' PERIODI di appartenenza separati.
    #
    # Il secondo criterio serve per le fonti che non forniscono i nomi delle
    # societa' uscite dall'indice: li' l'unico segnale disponibile e' che lo
    # stesso simbolo compare in intervalli disgiunti. Puo' trattarsi di una
    # societa' uscita e rientrata (caso benigno) o di un simbolo riassegnato a
    # un'azienda diversa (caso velenoso: la serie prezzi incollerebbe due
    # aziende). Senza i nomi non e' possibile distinguerli, quindi trattiamo
    # entrambi come sospetti e teniamo solo il periodo piu' recente.
    names_per_code = out.groupby("code")["name"].nunique(dropna=True)
    spells_per_code = out.groupby("code").size()
    reused = set(names_per_code[names_per_code > 1].index) | set(spells_per_code[spells_per_code > 1].index)
    out["ticker_reuse_suspect"] = out["code"].isin(reused)

    out["occurrence"] = out.groupby("code").cumcount()
    last_occ = out.groupby("code")["occurrence"].transform("max")
    out["is_latest_occurrence"] = out["occurrence"] == last_occ

    if reused:
        log.warning(
            "%d codici con possibile riassegnazione del ticker: %s",
            len(reused),
            ", ".join(sorted(reused)[:15]),
        )
    return out


def build_membership(
    constituents: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    drop_reused_prior: bool = True,
) -> pd.DataFrame:
    """Pannello booleano date x ticker: True se il titolo era nell'indice.

    `drop_reused_prior` scarta le occorrenze non piu' recenti dei codici
    riassegnati. Il conteggio degli scartati finisce nella diagnostica.
    """
    df = constituents
    if drop_reused_prior:
        df = df[~df["ticker_reuse_suspect"] | df["is_latest_occurrence"]]

    codes = sorted(df["code"].dropna().unique().tolist())
    memb = pd.DataFrame(False, index=dates, columns=codes)

    date_vals = dates.to_numpy()
    for code, start, end in zip(df["code"], df["start_date"], df["end_date"], strict=False):
        if code not in memb.columns:
            continue
        mask = (date_vals >= np.datetime64(start)) & (date_vals <= np.datetime64(end))
        if mask.any():
            memb.loc[mask, code] = True
    return memb


def sector_map(constituents: pd.DataFrame) -> pd.Series:
    """Codice -> settore GICS. Si tiene il settore dell'occorrenza piu' recente."""
    latest = constituents[constituents["is_latest_occurrence"]]
    s = latest.set_index("code")["sector"]
    s = s[~s.index.duplicated(keep="last")]
    s.name = "sector"
    return s


# ---------------------------------------------------------------------------
# Copertura dati - il numero che dice quanto vale tutto il resto
# ---------------------------------------------------------------------------
def coverage_report(
    membership: pd.DataFrame,
    prices_adj: pd.DataFrame,
    freq: str = "ME",
) -> pd.DataFrame:
    """Per ogni periodo: quanti titoli in indice, quanti con prezzo, il rapporto.

    Un coverage sotto il 100% significa che qualche titolo dell'indice sparisce
    in silenzio dal backtest. Il tasso e' per definizione peggiore sui titoli
    poi delistati, che si concentrano nella fascia piu' debole.
    """
    cols = membership.columns.union(prices_adj.columns)
    memb = membership.reindex(columns=cols).fillna(False).astype(bool)
    have = prices_adj.reindex(index=membership.index, columns=cols).notna()

    in_index = memb.sum(axis=1)
    with_price = (memb & have).sum(axis=1)

    daily = pd.DataFrame({"in_indice": in_index, "con_prezzo": with_price})
    out = daily.resample(freq).last()
    out["coverage"] = (out["con_prezzo"] / out["in_indice"].replace(0, np.nan)).astype("float64")
    out["mancanti"] = out["in_indice"] - out["con_prezzo"]
    return out


def coverage_by_band(
    membership: pd.DataFrame,
    prices_adj: pd.DataFrame,
    bands: pd.DataFrame,
    band_names: tuple[str, ...],
    freq: str = "ME",
) -> pd.DataFrame:
    """Coverage separato per fascia: mostra dove si concentrano i buchi."""
    memb = membership.reindex(columns=bands.columns).fillna(False).astype(bool)
    have = prices_adj.reindex(index=bands.index, columns=bands.columns).notna()

    frames = {}
    for i, label in enumerate(band_names, start=1):
        sel = memb & (bands == float(i))
        tot = sel.sum(axis=1).resample(freq).last()
        ok = (sel & have).sum(axis=1).resample(freq).last()
        frames[label] = (ok / tot.replace(0, np.nan)).astype("float64")
    return pd.DataFrame(frames)


def price_anomalies(prices_adj: pd.DataFrame, threshold: float = 0.60) -> pd.DataFrame:
    """Rendimenti giornalieri oltre soglia: quasi sempre split non gestiti.

    Non li correggiamo automaticamente (rischieremmo di cancellare eventi
    reali); li contiamo e li mostriamo, cosi' si puo' decidere caso per caso.
    """
    rets = prices_adj.pct_change()
    hits = rets.abs() > threshold
    per_ticker = hits.sum(axis=0)
    per_ticker = per_ticker[per_ticker > 0].sort_values(ascending=False)
    return pd.DataFrame({"anomalie": per_ticker})


def eligible_universe_size(eligible: pd.DataFrame, freq: str = "ME") -> pd.Series:
    """Numero di titoli eleggibili nel tempo. Crolla nei bear market."""
    return eligible.sum(axis=1).resample(freq).last().astype("int64")
