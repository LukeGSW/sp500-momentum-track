"""
Dataset sintetico per provare la dashboard SENZA chiave API.

    python -m pipeline.make_demo_data

Genera un universo finto con settori, rotazioni settoriali, ingressi e uscite
dall'indice, delistamenti e un titolo a prezzo molto alto (per esercitare il cap).

I dati NON sono reali: servono solo a verificare che l'interfaccia funzioni.
Il manifest marca `demo_data: true` e l'app mostra un banner di avviso.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from track import exclusions as exc
from track import universe
from track.config import GICS_SECTORS, PREREGISTERED
from pipeline.build_dataset import assemble_price_panels, compute_signal_panels, save_dataset

log = logging.getLogger("pipeline.make_demo_data")

N_PER_SECTOR = 13
START = "2013-01-02"
END = "2026-07-31"

# Prefissi espliciti: troncare il nome del settore produrrebbe collisioni
# (Consumer Discretionary / Consumer Staples / Communication -> tutti "CO").
SECTOR_PREFIX = {
    "Energy": "ENR",
    "Materials": "MAT",
    "Industrials": "IND",
    "Consumer Discretionary": "CDI",
    "Consumer Staples": "CST",
    "Health Care": "HLT",
    "Financials": "FIN",
    "Information Technology": "TEC",
    "Communication Services": "CMS",
    "Utilities": "UTL",
    "Real Estate": "RES",
}


def _generate(seed: int = 20260731) -> dict:
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range(START, END)
    n = len(calendar)

    # fattore di mercato + fattori settoriali lenti (creano rotazione)
    market = rng.normal(0.0003, 0.010, n)
    sector_factor = {
        s: np.cumsum(rng.normal(0.0, 0.0016, n)) * 0.02 + rng.normal(0.0, 0.006, n)
        for s in GICS_SECTORS
    }

    series: dict[str, pd.DataFrame] = {}
    rows = []
    today = pd.Timestamp(END)

    for si, sector in enumerate(GICS_SECTORS):
        for k in range(N_PER_SECTOR):
            code = f"{SECTOR_PREFIX[sector]}{k:02d}"
            beta = rng.uniform(0.6, 1.5)
            drift = rng.normal(0.0002, 0.00035)
            idio = rng.normal(0.0, rng.uniform(0.010, 0.024), n)
            ret = drift + beta * market + sector_factor[sector] + idio

            price = 40.0 * np.exp(np.cumsum(ret))
            if si == 0 and k == 0:
                price = price * 60.0  # un titolo oltre il cap dei 1.500$

            start_i, end_i = 0, n
            entry = calendar[0]
            exit_ = today

            if k == 1:  # entra tardi (IPO / ingresso nell'indice)
                start_i = int(n * rng.uniform(0.25, 0.45))
                entry = calendar[start_i]
            if k == 2:  # esce e sparisce: delistato
                end_i = int(n * rng.uniform(0.45, 0.85))
                exit_ = calendar[end_i - 1]

            idx = calendar[start_i:end_i]
            px = price[start_i:end_i]
            div_drag = np.linspace(1.0, rng.uniform(1.00, 1.09), len(idx))

            series[f"{code}.US"] = pd.DataFrame(
                {
                    "open": px * rng.normal(1.0, 0.003, len(idx)),
                    "close": px,
                    "adjusted_close": px * div_drag,
                },
                index=idx,
            )
            rows.append(
                {
                    "code": code,
                    "name": f"Demo {sector} {k}",
                    "sector": sector,
                    "industry": sector,
                    "start_date": entry,
                    "end_date": exit_ if k == 2 else pd.NaT,
                    "is_active_now": 0 if k == 2 else 1,
                    "is_delisted": 1 if k == 2 else 0,
                }
            )

    return {"series": series, "calendar": calendar, "constituents": pd.DataFrame(rows)}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    cfg = PREREGISTERED

    log.info("generazione universo sintetico ...")
    world = _generate()
    calendar = world["calendar"]

    const = universe.normalize_constituents(world["constituents"], today=pd.Timestamp(END))
    close_adj, open_adj, open_raw, close_raw = assemble_price_panels(world["series"], calendar)

    # stessa catena della pipeline reale, cosi' la demo esercita anche questa
    excl = exc.load_exclusions()
    panels, excl_report = exc.apply_exclusions(
        {"close_adj": close_adj, "open_adj": open_adj,
         "open_raw": open_raw, "close_raw": close_raw},
        excl,
    )
    close_adj, open_adj = panels["close_adj"], panels["open_adj"]
    open_raw, close_raw = panels["open_raw"], panels["close_raw"]

    membership = universe.build_membership(const, calendar)
    membership = membership.reindex(columns=close_adj.columns, fill_value=False)
    sectors = universe.sector_map(const).reindex(close_adj.columns)
    names = (const[const["is_latest_occurrence"]].set_index("code")["name"]
             .reindex(close_adj.columns))

    log.info("calcolo segnali su %d titoli x %d sedute ...", close_adj.shape[1], close_adj.shape[0])
    signals = compute_signal_panels(close_adj, open_raw, membership, cfg, sectors)

    rf_annual = 0.02 + 0.02 * np.sin(np.linspace(0, 6, len(calendar)))
    rf = pd.Series((1.0 + rf_annual) ** (1 / 252) - 1.0, index=calendar, name="rf")

    cov = universe.coverage_report(membership, close_adj)
    provenance = {
        "source": "DATI SINTETICI (demo)",
        "exclusions_applied": excl_report,
        "exclusions_count": int(sum(1 for r in excl_report if r.get("applicata"))),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_hash": cfg.hash(),
        "first_data_date": str(calendar.min().date()),
        "last_data_date": str(calendar.max().date()),
        "constituents_distinct_codes": int(const["code"].nunique()),
        "constituents_coverage_start": str(const["start_date"].min().date()),
        "coverage_mean": float(cov["coverage"].mean(skipna=True)),
        "coverage_min": float(cov["coverage"].min(skipna=True)),
        "coverage_by_year": {
            str(y): round(float(v), 4)
            for y, v in cov["coverage"].groupby(cov.index.year).mean().items()
        },
        "risk_free_source": "sintetico",
        "ticker_reuse_suspects": [],
        "demo_data": True,
    }

    save_dataset(
        close_adj=close_adj, open_adj=open_adj, open_raw=open_raw,
        membership=membership, signals=signals, sectors=sectors, names=names,
        risk_free=rf, provenance=provenance, close_raw=close_raw,
    )
    log.info("dataset demo pronto: %d titoli, %s -> %s",
             close_adj.shape[1], calendar.min().date(), calendar.max().date())
    return 0


if __name__ == "__main__":
    sys.exit(main())
