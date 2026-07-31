"""
Costruisce i pannelli letti dall'app.

    python -m pipeline.build_dataset

Scarica i costituenti storici S&P 500, i prezzi di TUTTI i ticker che hanno
fatto parte dell'indice (inclusi i delistati), calcola Forza, Spinta e fasce, e
salva tutto in data/*.parquet.

Costo indicativo: ~1.100-1.300 chiamate EOD la prima volta. Da eseguire
offline o in GitHub Actions, MAI dall'app.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from track import constituents as ct
from track import exclusions as exc
from track import features as ft
from track import storage, universe
from track.config import PREREGISTERED, TrackConfig
from track.eodhd import EODHDClient, resolve_api_key

log = logging.getLogger("pipeline.build_dataset")

CALENDAR_SYMBOL = "SPY.US"


# ---------------------------------------------------------------------------
def assemble_price_panels(
    series: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Da {simbolo: OHLC} a tre pannelli allineati sul calendario di borsa.

    close_adj : chiusura adjusted  -> segnale e contabilita' (total return)
    open_adj  : apertura adjusted  -> valorizzazione all'esecuzione
    open_raw  : apertura grezza    -> lotti interi e cap sul prezzo

    Il prezzo grezzo serve SOLO a decidere quante azioni compri; ogni
    rendimento passa dall'adjusted. Mischiarli gonfia i titoli ad alto dividendo.
    """
    close_adj, open_adj, open_raw, close_raw = {}, {}, {}, {}

    for sym, df in series.items():
        code = sym.split(".")[0]
        if "adjusted_close" not in df.columns or "close" not in df.columns:
            continue
        d = df.reindex(calendar)
        adj_close = d["adjusted_close"].astype("float64")
        raw_close = d["close"].astype("float64")
        factor = (adj_close / raw_close.where(raw_close > 0)).astype("float64")

        o_raw = d["open"].astype("float64") if "open" in d.columns else raw_close
        # apertura mancante o nulla: si ripiega sulla chiusura precedente
        o_raw = o_raw.where(o_raw > 0)

        close_adj[code] = adj_close
        close_raw[code] = raw_close.where(raw_close > 0)
        open_raw[code] = o_raw
        open_adj[code] = o_raw * factor

    return (
        pd.DataFrame(close_adj).sort_index(axis=1),
        pd.DataFrame(open_adj).sort_index(axis=1),
        pd.DataFrame(open_raw).sort_index(axis=1),
        pd.DataFrame(close_raw).sort_index(axis=1),
    )


def compute_signal_panels(
    close_adj: pd.DataFrame,
    open_raw: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: TrackConfig,
    sectors: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """Forza, Spinta, eleggibilita', fasce."""
    log.info("calcolo Forza su orizzonti %s ...", cfg.horizons)
    force = ft.compute_force(close_adj, cfg.horizons, cfg.winsor)

    log.info("calcolo Spinta su finestra %d ...", cfg.velocity_window)
    velocity = ft.compute_velocity(force, cfg.velocity_window, cfg.winsor)

    log.info("costruzione eleggibilita' (filtro media mobile: %s) ...", cfg.sma_filter)
    eligible, criteria = ft.build_eligibility(
        close_adj, open_raw, membership,
        min_history_days=cfg.min_history_days,
        max_share_price=cfg.max_share_price,
        sma_filter=cfg.sma_filter,
        sma_window=cfg.sma_window,
    )

    bands = ft.assign_bands(force, eligible, cfg.n_bands, sectors, cfg.sector_neutral)

    return {
        "force": force,
        "velocity": velocity,
        "eligible": eligible,
        "bands": bands,
        "_criteria": criteria,
    }


def save_dataset(
    *,
    close_adj: pd.DataFrame,
    open_adj: pd.DataFrame,
    open_raw: pd.DataFrame,
    membership: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    sectors: pd.Series,
    names: pd.Series,
    risk_free: pd.Series,
    provenance: dict,
    directory=None,
    close_raw: pd.DataFrame | None = None,
) -> None:
    storage.save_panel(close_adj, "close_adj", directory)
    storage.save_panel(open_adj, "open_adj", directory)
    storage.save_panel(open_raw, "open_raw", directory)
    if close_raw is not None:
        # non e' fra i pannelli obbligatori: serve solo alla diagnostica, che
        # confronta rendimento rettificato e grezzo per smascherare gli split
        storage.save_panel(close_raw, "close_raw", directory)
    storage.save_panel(membership.astype(bool), "membership", directory)
    storage.save_panel(signals["force"], "force", directory)
    storage.save_panel(signals["velocity"], "velocity", directory)
    storage.save_panel(signals["bands"], "bands", directory)
    storage.save_panel(signals["eligible"].astype(bool), "eligible", directory)

    meta = pd.DataFrame({"sector": sectors, "name": names})
    meta.index.name = "ticker"
    storage.save_table(meta.reset_index(), "meta", directory)
    storage.save_table(risk_free.rename("rf").to_frame().reset_index(names="date"),
                       "risk_free", directory)
    storage.save_manifest(provenance, directory)
    log.info("dataset scritto in %s", storage.data_dir(directory))


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Costruisce il dataset de La Pista")
    parser.add_argument("--limit", type=int, default=0,
                        help="scarica solo i primi N ticker (per prove rapide)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--constituents-source", choices=[ct.SOURCE_GITHUB, ct.SOURCE_EODHD],
                        default=ct.SOURCE_GITHUB,
                        help="'github' (predefinita): ricostruzione MIT dal 1996, nessuna "
                             "entitlement richiesta. 'eodhd': endpoint sugli indici, dal 2000, "
                             "incluso solo in alcuni piani.")
    parser.add_argument("--exclusions", default=None,
                        help="file CSV delle serie compromesse da escludere "
                             "(default: exclusions.csv nella radice del progetto). "
                             "Usa --exclusions NONE per disattivarle.")
    parser.add_argument("--enrich-sectors", action="store_true",
                        help="recupera i settori delle societa' uscite dall'indice dai "
                             "Fundamentals EODHD per singolo titolo (~700 chiamate in piu', "
                             "una volta sola). Senza, restano 'Non classificato'.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    cfg = PREREGISTERED

    client = EODHDClient(resolve_api_key(), max_workers=args.workers)

    log.info("1/6  costituenti storici (fonte: %s) ...", args.constituents_source)
    raw_const = ct.load_constituents(args.constituents_source, client=client,
                                     cache_dir=args.data_dir)
    if args.enrich_sectors:
        raw_const = ct.enrich_sectors_from_eodhd(raw_const, client)
    const = universe.normalize_constituents(raw_const)
    log.info("     %d record, %d codici distinti, dal %s",
             len(const), const["code"].nunique(), const["start_date"].min().date())

    log.info("2/6  calendario di borsa da %s ...", CALENDAR_SYMBOL)
    spy = client.eod(CALENDAR_SYMBOL, cfg.download_start)
    if spy is None or spy.empty:
        log.error("impossibile scaricare %s: calendario non determinabile", CALENDAR_SYMBOL)
        return 2
    calendar = pd.DatetimeIndex(spy.index)

    codes = sorted(const["code"].dropna().unique().tolist())
    if args.limit:
        codes = codes[: args.limit]
    symbols = [f"{c}.US" for c in codes]
    log.info("3/6  download prezzi per %d ticker (inclusi i delistati) ...", len(symbols))

    def _tick(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            log.info("     %d/%d", done, total)

    series = client.eod_many(symbols, cfg.download_start, progress=_tick)
    log.info("     ottenuti %d/%d (%.1f%%)", len(series), len(symbols),
             100.0 * len(series) / max(len(symbols), 1))

    log.info("4/6  assemblaggio pannelli ...")
    close_adj, open_adj, open_raw, close_raw = assemble_price_panels(series, calendar)

    # Le esclusioni si applicano QUI, prima di qualunque calcolo: cosi' sono
    # parte del metodo e non un ritocco a valle sui risultati.
    excl = exc.load_exclusions(args.exclusions)
    panels, excl_report = exc.apply_exclusions(
        {"close_adj": close_adj, "open_adj": open_adj,
         "open_raw": open_raw, "close_raw": close_raw},
        excl,
    )
    close_adj, open_adj = panels["close_adj"], panels["open_adj"]
    open_raw, close_raw = panels["open_raw"], panels["close_raw"]
    if excl_report:
        applicate = sum(1 for r in excl_report if r.get("applicata"))
        log.info("     %d esclusioni applicate su %d dichiarate", applicate, len(excl_report))
    else:
        log.warning("     NESSUNA esclusione applicata: il dataset conterra' anche le "
                    "serie con errori di dato. Esegui  python -m pipeline.find_bad_series  "
                    "dopo la costruzione per individuarle.")

    membership = universe.build_membership(const, calendar)
    membership = membership.reindex(columns=close_adj.columns, fill_value=False)
    sectors = universe.sector_map(const).reindex(close_adj.columns)
    names = (const[const["is_latest_occurrence"]].set_index("code")["name"]
             .reindex(close_adj.columns))

    log.info("5/6  segnali ...")
    signals = compute_signal_panels(close_adj, open_raw, membership, cfg, sectors)

    log.info("6/6  risk-free e diagnostica ...")
    rf = client.risk_free_daily(cfg.download_start)
    rf = rf.reindex(calendar).ffill().fillna(0.0)

    cov = universe.coverage_report(membership, close_adj)
    anomalies = universe.price_anomalies(close_adj)
    reused = sorted(const.loc[const["ticker_reuse_suspect"], "code"].unique().tolist())

    n_unclassified = int((sectors == "Non classificato").sum())
    provenance = {
        "exclusions_applied": excl_report,
        "exclusions_count": int(sum(1 for r in excl_report if r.get("applicata"))),
        "source": "EODHD (prezzi)",
        "constituents_source": args.constituents_source,
        "constituents_source_note": (
            "fja05680/sp500, licenza MIT: RICOSTRUZIONE di terze parti, non dato "
            "ufficiale S&P. Tratto 1996-2019 dal dataset di 'Trading Evolved' "
            "(Clenow), successivo dal tracciamento Wikipedia."
            if args.constituents_source == ct.SOURCE_GITHUB
            else "EODHD fundamentals/GSPC.INDX HistoricalTickerComponents"
        ),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_hash": cfg.hash(),
        "calendar_symbol": CALENDAR_SYMBOL,
        "sectors_known": int(len(sectors) - n_unclassified),
        "sectors_unclassified": n_unclassified,
        "first_data_date": str(calendar.min().date()),
        "last_data_date": str(calendar.max().date()),
        "constituents_records": int(len(const)),
        "constituents_distinct_codes": int(const["code"].nunique()),
        "constituents_coverage_start": str(const["start_date"].min().date()),
        "symbols_requested": int(len(symbols)),
        "symbols_downloaded": int(len(series)),
        "download_success_rate": round(len(series) / max(len(symbols), 1), 4),
        "coverage_mean": float(cov["coverage"].mean(skipna=True)),
        "coverage_min": float(cov["coverage"].min(skipna=True)),
        "coverage_by_year": {
            str(y): round(float(v), 4)
            for y, v in cov["coverage"].groupby(cov.index.year).mean().items()
        },
        "risk_free_source": rf.attrs.get("source", "sconosciuto"),
        "ticker_reuse_suspects": reused,
        "price_anomaly_tickers": int(len(anomalies)),
        "demo_data": False,
    }

    save_dataset(
        close_adj=close_adj, open_adj=open_adj, open_raw=open_raw,
        membership=membership, signals=signals, sectors=sectors, names=names,
        risk_free=rf, provenance=provenance, directory=args.data_dir,
        close_raw=close_raw,
    )

    log.info("FATTO. Coverage medio %.2f%%, minimo %.2f%%",
             100 * provenance["coverage_mean"], 100 * provenance["coverage_min"])
    if provenance["coverage_mean"] < 0.95:
        log.warning("Coverage sotto il 95%%: leggere i risultati insieme allo stress test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
