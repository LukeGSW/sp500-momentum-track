"""
Export JSON per analisi esterna con un LLM.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from track import analysis, backtest as bt
from track import didactics, export, study, ui
from track.config import BAND_NAMES, PREREGISTERED

ui.page_config("Export")

if not ui.require_dataset():
    st.stop()

cfg = ui.sidebar_config()
ds = ui.get_dataset()
sig = ui.get_signals(cfg)
res = ui.get_study(cfg)

ui.header("Export JSON", ds)

st.markdown(
    """
Scarica lo studio completo in un formato pensato per essere incollato in una
conversazione con un modello linguistico. Il campo piu' importante non sono i
risultati: e' **`caveats`**, che elenca ogni limite metodologico noto insieme
alla **direzione** verso cui distorce le conclusioni.
"""
)

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    level = st.radio("Livello di dettaglio", ["compact", "full"], index=0,
                     help="compact esclude le serie mensili complete: piu' leggero, "
                          "adatto a una conversazione. full le include.")
with c2:
    include_null = st.toggle("Includi l'ipotesi nulla", value=True,
                             help="Aggiunge i percentili della distribuzione ottenuta "
                                  "con 30 titoli estratti a caso e i p-value empirici.")
with c3:
    n_draws = st.select_slider("Estrazioni per l'ipotesi nulla",
                               options=[100, 250, 500, 1000], value=250,
                               disabled=not include_null)

as_of = sig.force.dropna(how="all").index[-1]
members = study.portfolio_members_at(sig, cfg, as_of)
snapshot = analysis.build_snapshot(
    as_of, force=sig.force, velocity=sig.velocity, bands=sig.bands,
    eligible=sig.eligible, sectors=ds.sectors, names=ds.names,
    band_names=BAND_NAMES, prices_raw=ds.open_raw,
    max_share_price=cfg.max_share_price, portfolios=members,
)

metrics = {n: bt.performance_metrics(r, res.rf_period) for n, r in res.results.items()}

null_summary = {}
if include_null:
    null = ui.get_bootstrap(cfg, int(n_draws))
    null_summary = {
        "n_draws": int(len(null)),
        "method": (
            "Stesso motore di backtest, stessi costi, stesso arrotondamento a lotti "
            "interi; l'unica differenza e' che i 30 titoli sono estratti a caso "
            "dall'universo eleggibile alla stessa data."
        ),
        "percentiles": {
            metric: {f"p{int(q*100)}": float(null[metric].quantile(q))
                     for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
            for metric in ("CAGR", "Sharpe") if metric in null.columns
        },
        "p_values": {
            metric: {name: bt.empirical_pvalue(null[metric], metrics[name].get(metric))
                     for name in metrics}
            for metric in ("CAGR", "Sharpe") if metric in null.columns
        },
    }

stress_df = ui.get_delisting_stress(cfg)
stress = {col: stress_df[col].to_dict() for col in stress_df.columns}

provenance = dict(ds.manifest)
provenance["snapshot_as_of"] = str(pd.Timestamp(as_of).date())

payload = export.build_export(
    cfg=cfg,
    provenance=provenance,
    snapshot=snapshot,
    band_names=BAND_NAMES,
    backtest_metrics=metrics,
    backtest_returns={n: r.returns for n, r in res.results.items()},
    null_summary=null_summary,
    subperiods=bt.subperiod_table(res.results, res.rf_period),
    transition_matrix=analysis.transition_matrix(sig.bands, cfg.n_bands, BAND_NAMES, 21),
    diagnostics={**res.diagnostics,
                 "spread_tstat_newey_west": res.spread_tstat,
                 "costo_per_rotazione_bps": study.current_cost_bps(cfg)},
    stress=stress,
    level=level,
)

raw = export.to_json_bytes(payload)
stamp = pd.Timestamp(as_of).strftime("%Y%m%d")

st.divider()
d1, d2, d3 = st.columns(3)
d1.metric("Dimensione", f"{len(raw) / 1024:,.0f} KB")
d2.metric("Titoli nello snapshot", f"{len(snapshot)}")
d3.metric("Configurazione", "preregistrata" if cfg.hash() == PREREGISTERED.hash()
          else "personalizzata")

b1, b2 = st.columns(2)
with b1:
    st.download_button(
        f"⬇ Scarica lo studio ({level})",
        data=raw,
        file_name=f"la_pista_{level}_{stamp}_{cfg.hash()}.json",
        mime="application/json",
        width="stretch",
        type="primary",
    )
with b2:
    st.download_button(
        "⬇ Scarica la descrizione dello schema",
        data=export.SCHEMA_README.encode("utf-8"),
        file_name="README_SCHEMA.md",
        mime="text/markdown",
        width="stretch",
        help="Descrive ogni campo del JSON: scaricalo insieme al file cosi' il "
             "modello non deve indovinare il significato delle chiavi.",
    )

didactics.render("json_export", expanded=True)

st.divider()
st.subheader("Anteprima")
tab1, tab2, tab3 = st.tabs(["Struttura", "Limiti metodologici", "JSON grezzo"])

with tab1:
    st.json({k: (f"<{type(v).__name__}, {len(v)} elementi>"
                 if isinstance(v, (list, dict)) and len(str(v)) > 400 else v)
             for k, v in payload.items()})

with tab2:
    st.caption("Il contenuto del campo `caveats`. E' la parte del file che va letta per prima.")
    for cav in payload["caveats"]:
        with st.expander(f"**{cav['id']}** — {cav['direction_of_bias']}"):
            st.markdown(f"{cav['description']}\n\n**Mitigazione:** {cav['mitigation']}")

with tab3:
    st.code(json.dumps(payload, ensure_ascii=False, indent=2)[:20000] +
            ("\n\n… troncato nell'anteprima, il file scaricato e' completo"
             if len(raw) > 20000 else ""),
            language="json")

st.info(
    "**Suggerimento per l'uso con un LLM.** Allega sia il JSON sia "
    "`README_SCHEMA.md`, e chiedi esplicitamente di leggere `caveats` prima di "
    "interpretare i numeri. Senza quel passaggio la risposta piu' probabile e' una "
    "lettura ingenua del CAGR piu' alto.",
    icon="💡",
)
