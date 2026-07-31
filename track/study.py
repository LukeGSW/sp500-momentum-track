"""
Orchestrazione dello studio: dai pannelli salvati ai risultati.

Usato sia dall'app (con caching Streamlit) sia dalla pipeline. Diviso in due
livelli di ricalcolo, perche' hanno costi molto diversi:

  livello LENTO  - cambia `holding_months` -> cambiano gli orizzonti, quindi
                   Forza e Spinta vanno ricalcolate su tutto il pannello
  livello VELOCE - cambiano filtro, cap, numero di fasce, sector-neutral,
                   costi, numero di titoli -> bastano eleggibilita', fasce e
                   una nuova passata del motore (millisecondi)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import backtest as bt
from . import features as ft
from . import storage
from .config import TrackConfig

log = logging.getLogger(__name__)

# Nomi dei panieri: usati ovunque come chiavi.
P_TOP = "Testa Corsa"
P_BOTTOM = "Fondo Griglia"
P_PULLBACK = "Testa Corsa in ritracciamento"
P_MID = "Gruppo"
P_UNIVERSE = "Universo eleggibile (senza costi)"

PORTFOLIO_ORDER = (P_TOP, P_BOTTOM, P_PULLBACK, P_MID, P_UNIVERSE)


# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    close_adj: pd.DataFrame
    open_adj: pd.DataFrame
    open_raw: pd.DataFrame
    membership: pd.DataFrame
    sectors: pd.Series
    names: pd.Series
    risk_free: pd.Series
    manifest: dict
    # segnali della configurazione preregistrata, gia' calcolati dalla pipeline
    force: pd.DataFrame
    velocity: pd.DataFrame

    @property
    def is_demo(self) -> bool:
        return bool(self.manifest.get("demo_data", False))


def load_dataset(directory=None) -> Dataset:
    meta = storage.load_table("meta", directory).set_index("ticker")
    rf_tbl = storage.load_table("risk_free", directory)
    rf = pd.Series(
        rf_tbl["rf"].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(rf_tbl["date"]),
        name="rf",
    )
    return Dataset(
        close_adj=storage.load_panel("close_adj", directory),
        open_adj=storage.load_panel("open_adj", directory),
        open_raw=storage.load_panel("open_raw", directory),
        membership=storage.load_panel("membership", directory).astype(bool),
        sectors=meta["sector"],
        names=meta["name"],
        risk_free=rf,
        manifest=storage.load_manifest(directory),
        force=storage.load_panel("force", directory),
        velocity=storage.load_panel("velocity", directory),
    )


# ---------------------------------------------------------------------------
@dataclass
class Signals:
    force: pd.DataFrame
    velocity: pd.DataFrame
    eligible: pd.DataFrame
    bands: pd.DataFrame
    criteria: dict[str, pd.DataFrame] = field(default_factory=dict)


def compute_force_velocity(ds: Dataset, cfg: TrackConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Livello LENTO. Se la configurazione usa gli orizzonti della pipeline,
    riusa i pannelli gia' salvati invece di ricalcolarli."""
    default = TrackConfig()
    if cfg.horizons == default.horizons and cfg.velocity_window == default.velocity_window \
            and cfg.winsor == default.winsor:
        return ds.force, ds.velocity

    log.info("ricalcolo Forza/Spinta per orizzonti %s", cfg.horizons)
    force = ft.compute_force(ds.close_adj, cfg.horizons, cfg.winsor)
    velocity = ft.compute_velocity(force, cfg.velocity_window, cfg.winsor)
    return force, velocity


def compute_signals(ds: Dataset, cfg: TrackConfig) -> Signals:
    """Livello VELOCE (piu' l'eventuale livello lento a monte)."""
    force, velocity = compute_force_velocity(ds, cfg)
    eligible, criteria = ft.build_eligibility(
        ds.close_adj, ds.open_raw, ds.membership,
        min_history_days=cfg.min_history_days,
        max_share_price=cfg.max_share_price,
        sma_filter=cfg.sma_filter,
        sma_window=cfg.sma_window,
    )
    bands = ft.assign_bands(force, eligible, cfg.n_bands, ds.sectors, cfg.sector_neutral)
    return Signals(force, velocity, eligible, bands, criteria)


# ---------------------------------------------------------------------------
@dataclass
class StudyResult:
    cfg: TrackConfig
    panel: bt.Panel
    results: dict[str, bt.BacktestResult]
    metrics: pd.DataFrame
    rf_period: pd.Series
    spread: pd.Series
    spread_tstat: float
    diagnostics: dict


def build_panel(ds: Dataset, sig: Signals, cfg: TrackConfig) -> bt.Panel:
    return bt.prepare_panel(
        close_adj=ds.close_adj,
        open_adj=ds.open_adj,
        open_raw=ds.open_raw,
        force=sig.force,
        velocity=sig.velocity,
        bands=sig.bands,
        eligible=sig.eligible,
        rf_daily=ds.risk_free,
        cfg=cfg,
    )


def portfolio_selectors(cfg: TrackConfig) -> dict[str, bt.Selector]:
    top_band = cfg.n_bands
    mid_band = (cfg.n_bands + 1) // 2
    return {
        P_TOP: bt.selector_band(top_band, cfg.n_names, take_highest=True),
        P_BOTTOM: bt.selector_band(1, cfg.n_names, take_highest=False),
        P_PULLBACK: bt.selector_band_negative_velocity(top_band, cfg.n_names),
        P_MID: bt.selector_band(mid_band, cfg.n_names, take_highest=True),
        P_UNIVERSE: bt.selector_all_eligible(),
    }


def portfolio_members_at(
    sig: Signals, cfg: TrackConfig, as_of: pd.Timestamp
) -> dict[str, list[str]]:
    """Chi comprerebbe ciascun paniere se si ribilanciasse a questa data.

    Stessa logica dei selettori del motore, applicata a una singola data per
    la vista corrente. Serve a evidenziare i titoli sulla pista.
    """
    if as_of not in sig.force.index:
        pos = sig.force.index.searchsorted(as_of, side="right") - 1
        if pos < 0:
            return {}
        as_of = sig.force.index[pos]

    f = sig.force.loc[as_of]
    v = sig.velocity.loc[as_of]
    b = sig.bands.loc[as_of]
    e = sig.eligible.loc[as_of].astype(bool)

    top_band, mid_band = float(cfg.n_bands), float((cfg.n_bands + 1) // 2)

    def _pick(mask: pd.Series, by: pd.Series, ascending: bool) -> list[str]:
        sub = by[mask & by.notna()]
        return sub.sort_values(ascending=ascending).head(cfg.n_names).index.tolist()

    return {
        P_TOP: _pick(e & (b == top_band), f, ascending=False),
        P_BOTTOM: _pick(e & (b == 1.0), f, ascending=True),
        P_PULLBACK: _pick(e & (b == top_band) & (v < 0), v, ascending=True),
        P_MID: _pick(e & (b == mid_band), f, ascending=False),
    }


def run_study(
    ds: Dataset,
    cfg: TrackConfig,
    *,
    signals: Signals | None = None,
    delisting_haircut: float = 0.0,
) -> StudyResult:
    sig = signals or compute_signals(ds, cfg)
    panel = build_panel(ds, sig, cfg)
    rf_period = pd.Series(panel.rf_period, index=panel.exec_dates, name="rf")

    results: dict[str, bt.BacktestResult] = {}
    for name, sel in portfolio_selectors(cfg).items():
        # L'universo eleggibile e' un RIFERIMENTO, non un portafoglio
        # implementabile: centinaia di nomi con commissione fissa sarebbero
        # improponibili. Lo simuliamo senza attriti e lo dichiariamo tale.
        frictionless = name == P_UNIVERSE
        results[name] = bt.run_strategy(
            panel, sel, cfg, name=name,
            frictionless=frictionless,
            delisting_haircut=delisting_haircut,
        )

    rows = {}
    for name in PORTFOLIO_ORDER:
        if name in results:
            rows[name] = bt.performance_metrics(results[name], rf_period)
    metrics = pd.DataFrame(rows).T

    top_r = results[P_TOP].returns
    bot_r = results[P_BOTTOM].returns
    common = top_r.index.intersection(bot_r.index)
    spread = (top_r.loc[common] - bot_r.loc[common]).rename("Spread Testa - Fondo")

    # L'universo va misurato SOLO alle date di decisione e SOLO dentro il
    # periodo di backtest: mediarlo su tutte le sedute includerebbe il warm-up
    # iniziale, dove nessun titolo ha ancora storia sufficiente e il conteggio
    # e' zero per costruzione.
    elig_at_decision = sig.eligible.reindex(panel.decision_dates).sum(axis=1)

    diagnostics = {
        "periodi": int(panel.n_periods),
        "primo_ribilanciamento": str(panel.exec_dates.min().date()) if panel.n_periods else None,
        "ultimo_ribilanciamento": str(panel.exec_dates.max().date()) if panel.n_periods else None,
        "universo_eleggibile_medio": float(elig_at_decision.mean()),
        "universo_eleggibile_minimo": int(elig_at_decision.min()),
        "universo_eleggibile_mediano": float(elig_at_decision.median()),
        "ribilanciamenti_sotto_soglia_minima": int((elig_at_decision < cfg.min_eligible).sum()),
        "titoli_per_paniere_configurati": int(cfg.n_names),
        "posizioni_massime_teoriche": int(cfg.n_names * cfg.n_tranches),
        **{f"{k} — {kk}": vv for k, r in results.items()
           for kk, vv in r.diagnostics.items()
           if kk in ("mesi_paniere_incompleto", "liquidazioni_forzate", "costi_totali")},
    }

    return StudyResult(
        cfg=cfg, panel=panel, results=results, metrics=metrics,
        rf_period=rf_period, spread=spread,
        spread_tstat=bt.newey_west_tstat(spread, lags=3),
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
def cost_sensitivity(
    panel: bt.Panel,
    cfg: TrackConfig,
    bps_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """CAGR dei due panieri estremi al variare del costo di transazione.

    Il costo e' espresso in bps totali per rotazione (andata e ritorno) e
    imputato interamente allo spread, azzerando la commissione fissa: cosi'
    la griglia resta leggibile su un asse solo.

    Attenzione a come si legge il risultato. Il costo abbatte fortemente il
    rendimento di CIASCUN paniere, ma la loro DIFFERENZA e' quasi insensibile,
    perche' entrambi pagano. La curva della differenza e' piatta quando i
    turnover sono simili: e' il comportamento atteso, non un errore. Il
    break-even esiste solo se un paniere ruota molto piu' dell'altro.
    """
    if bps_grid is None:
        bps_grid = np.array([0, 5, 10, 15, 20, 30, 40, 60, 80, 120, 200], dtype="float64")

    top_sel = bt.selector_band(cfg.n_bands, cfg.n_names, True)
    bot_sel = bt.selector_band(1, cfg.n_names, False)

    rows = []
    for bps in bps_grid:
        c = replace(cfg, commission_per_side=0.0, spread_bps=float(bps) / 2.0)
        mt = bt.performance_metrics(bt.run_strategy(panel, top_sel, c, name="top"))
        mb = bt.performance_metrics(bt.run_strategy(panel, bot_sel, c, name="bot"))
        rows.append({
            "bps": float(bps),
            P_TOP: mt.get("CAGR", np.nan),
            P_BOTTOM: mb.get("CAGR", np.nan),
            "Differenza": mt.get("CAGR", np.nan) - mb.get("CAGR", np.nan),
        })
    return pd.DataFrame(rows)


def current_cost_bps(cfg: TrackConfig) -> float:
    """Costo per rotazione completa, in bps sullo slot."""
    slot = cfg.slot_value
    if slot <= 0:
        return float("nan")
    commission_bps = 2.0 * cfg.commission_per_side / slot * 1e4
    return commission_bps + 2.0 * cfg.spread_bps
