"""
Costituenti storici dell'S&P 500: fonti alternative e intercambiabili.

Perche' questo modulo esiste
----------------------------
L'endpoint EODHD `fundamentals/GSPC.INDX` (che restituisce
`HistoricalTickerComponents`) e' incluso solo in alcuni piani: senza di esso
risponde 403. Serviva un piano B, e il piano B si e' rivelato migliore
dell'originale: parte dal **1996** invece che dal 2000, quindi rende
analizzabile anche la SALITA della bolla dot-com, non solo lo scoppio.

Le fonti
--------
`github`  (predefinita) — ricostruzione di fja05680/sp500, licenza MIT.
          `sp500_ticker_start_end.csv` da' un record per ogni periodo di
          appartenenza (ticker, inizio, fine). I settori arrivano da
          `sp500.csv`, che e' la lista corrente di Wikipedia gia' normalizzata.

`eodhd`   — l'endpoint originale, se il piano lo include.

ATTENZIONE alla qualita' del dato
---------------------------------
La fonte `github` e' una **ricostruzione di terze parti**, non un dato
ufficiale S&P: il tratto 1996-2019 deriva dal dataset del libro "Trading
Evolved" di Andreas Clenow, quello successivo dal tracciamento delle
variazioni su Wikipedia. Le ricostruzioni sbagliano: variazioni non
registrate, cambi di ticker scambiati per uscita+ingresso. L'errore si
accumula andando indietro nel tempo, quindi i primi anni sono i meno
affidabili — proprio quelli piu' interessanti per la domanda dello studio.
La cosa va dichiarata nei risultati, ed e' nei `caveats` dell'export.

I settori esistono solo per i membri ATTUALI. Per le societa' uscite
dall'indice restano "Non classificato", a meno di arricchirli via EODHD
(vedi `enrich_sectors_from_eodhd`). Di conseguenza l'opzione sector-neutral
e' poco affidabile sui backtest storici lunghi.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from .storage import data_dir

log = logging.getLogger(__name__)

GITHUB_BASE = "https://raw.githubusercontent.com/fja05680/sp500/master/"
SPELLS_FILE = "sp500_ticker_start_end.csv"
CURRENT_FILE = "sp500.csv"

SOURCE_GITHUB = "github"
SOURCE_EODHD = "eodhd"


# ---------------------------------------------------------------------------
def _download(name: str, cache: Path | None, force: bool = False) -> bytes:
    """Scarica con cache su disco: le esecuzioni ripetute non ribattono su GitHub."""
    if cache is not None:
        local = cache / f"_src_{name}"
        if local.exists() and not force:
            log.info("uso la copia locale di %s", name)
            return local.read_bytes()

    url = GITHUB_BASE + requests.utils.quote(name)
    log.info("scarico %s", url)
    resp = requests.get(url, timeout=90, headers={"User-Agent": "la-mappa/1.0"})
    resp.raise_for_status()

    if cache is not None:
        (cache / f"_src_{name}").write_bytes(resp.content)
    return resp.content


def fetch_github_spells(cache_dir=None, force: bool = False) -> pd.DataFrame:
    """Un record per periodo di appartenenza: ticker, start_date, end_date.

    Un ticker con piu' periodi distinti significa o una societa' uscita e
    rientrata, o un simbolo riassegnato a una societa' diversa. Senza i nomi
    non e' possibile distinguerli, quindi li trattiamo tutti come sospetti
    (vedi `universe.normalize_constituents`).
    """
    raw = _download(SPELLS_FILE, data_dir(cache_dir), force)
    df = pd.read_csv(io.BytesIO(raw))

    expected = {"ticker", "start_date", "end_date"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{SPELLS_FILE}: colonne inattese {list(df.columns)}")

    df = df.rename(columns={"ticker": "code"})
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df.dropna(subset=["code", "start_date"]).reset_index(drop=True)


def fetch_github_current(cache_dir=None, force: bool = False) -> pd.DataFrame:
    """Lista corrente con settori GICS (Wikipedia, gia' normalizzata)."""
    raw = _download(CURRENT_FILE, data_dir(cache_dir), force)
    df = pd.read_csv(io.BytesIO(raw))

    ren = {"Symbol": "code", "Security": "name",
           "GICS Sector": "sector", "GICS Sub-Industry": "industry"}
    missing = [c for c in ren if c not in df.columns]
    if missing:
        raise ValueError(f"{CURRENT_FILE}: colonne mancanti {missing}")

    out = df.rename(columns=ren)[list(ren.values())].copy()
    out["code"] = out["code"].astype("string").str.strip().str.upper()
    return out


def load_from_github(cache_dir=None, force: bool = False) -> pd.DataFrame:
    """Costituenti storici nello schema atteso da `universe.normalize_constituents`."""
    spells = fetch_github_spells(cache_dir, force)
    current = fetch_github_current(cache_dir, force)

    meta = current.set_index("code")[["name", "sector", "industry"]]
    meta = meta[~meta.index.duplicated(keep="last")]

    out = spells.copy()
    out["code"] = out["code"].astype("string").str.strip().str.upper()
    out = out.join(meta, on="code")

    out["exchange"] = "US"
    out["is_active_now"] = out["end_date"].isna().astype("int64")
    # La fonte non distingue delisting da semplice uscita dall'indice.
    # Lasciamo 0 e non ci basiamo su questo campo da nessuna parte.
    out["is_delisted"] = 0

    n_sector = int(out["sector"].notna().sum())
    log.info(
        "costituenti da GitHub: %d periodi, %d ticker distinti, %s → %s. "
        "Settore noto per %d/%d periodi (%.0f%%): le societa' uscite "
        "dall'indice restano non classificate.",
        len(out), out["code"].nunique(),
        out["start_date"].min().date(), out["end_date"].max().date(),
        n_sector, len(out), 100.0 * n_sector / max(len(out), 1),
    )
    return out


# ---------------------------------------------------------------------------
def load_from_eodhd(client) -> pd.DataFrame:
    """Fonte originale. Richiede i Fundamentals sugli indici nel piano."""
    return client.historical_constituents()


def load_constituents(
    source: str = SOURCE_GITHUB,
    *,
    client=None,
    cache_dir=None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Punto di ingresso unico. `source` in {'github', 'eodhd'}."""
    if source == SOURCE_GITHUB:
        return load_from_github(cache_dir, force_download)
    if source == SOURCE_EODHD:
        if client is None:
            raise ValueError("la fonte 'eodhd' richiede un client")
        return load_from_eodhd(client)
    raise ValueError(f"fonte sconosciuta: {source!r}")


# ---------------------------------------------------------------------------
def enrich_sectors_from_eodhd(
    constituents: pd.DataFrame,
    client,
    *,
    only_missing: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """Recupera i settori mancanti dai Fundamentals per singolo titolo.

    E' un'entitlement diversa da quella sugli indici: puo' funzionare anche
    se `fundamentals/GSPC.INDX` risponde 403. Degrada con grazia — se anche
    questa e' bloccata, i settori restano vuoti e la dashboard lo dichiara.

    Costa una chiamata API per ticker: con ~700 societa' uscite dall'indice
    non e' gratis, quindi si esegue una volta sola e si tiene il risultato.
    """
    out = constituents.copy()
    if "sector" not in out.columns:
        out["sector"] = pd.NA

    codes = out.loc[out["sector"].isna(), "code"].dropna().unique().tolist() \
        if only_missing else out["code"].dropna().unique().tolist()
    if limit:
        codes = codes[:limit]
    if not codes:
        return out

    log.info("arricchimento settori per %d ticker via EODHD…", len(codes))
    found: dict[str, str] = {}
    for i, code in enumerate(codes, 1):
        try:
            payload = client._get(f"fundamentals/{code}.US", {"filter": "General::Sector"})
        except Exception as exc:  # noqa: BLE001 - una chiamata fallita non deve fermare tutto
            if i == 1:
                log.warning("Fundamentals per singolo titolo non disponibili (%s): "
                            "i settori storici resteranno vuoti.", exc)
                break
            continue
        if isinstance(payload, str) and payload.strip():
            found[code] = payload.strip()
        if i % 100 == 0:
            log.info("   %d/%d", i, len(codes))

    if found:
        mask = out["code"].isin(found)
        out.loc[mask, "sector"] = out.loc[mask, "code"].map(found)
        log.info("settori recuperati per %d ticker", len(found))
    return out
