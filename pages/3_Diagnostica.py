"""
Diagnostica dei dati.

Questa pagina determina quanto vale tutto il resto della dashboard. Un
backtest su dati bucati non e' un backtest prudente: e' un backtest sbagliato,
e sbagliato in una direzione precisa.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from track import didactics, plotting, ui, universe
from track.config import BAND_NAMES

ui.page_config("Diagnostica")

if not ui.require_dataset():
    st.stop()

cfg = ui.sidebar_config()
ds = ui.get_dataset()
sig = ui.get_signals(cfg)

ui.header("Diagnostica dei dati", ds)

man = ds.manifest

# ---------------------------------------------------------------------------
st.subheader("Provenienza")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Fonte", str(man.get("source", "?")))
p2.metric("Periodo dati", f"{man.get('first_data_date', '?')} → {man.get('last_data_date', '?')}")
p3.metric("Ticker nel dataset", f"{ds.close_adj.shape[1]:,}")
p4.metric("Costituenti storici da", str(man.get("constituents_coverage_start", "?")))

if man.get("download_success_rate") is not None:
    st.caption(
        f"Scaricati {man.get('symbols_downloaded', '?')} su "
        f"{man.get('symbols_requested', '?')} simboli richiesti "
        f"({man.get('download_success_rate', 0):.1%}). "
        f"Risk-free: `{man.get('risk_free_source', '?')}`."
    )

if str(man.get("risk_free_source", "")).startswith("FALLBACK"):
    st.warning(
        "La serie del tasso privo di rischio **non e' stata trovata** ed e' stata "
        "sostituita con una costante al 2% annuo. Gli Sharpe ratio e la remunerazione "
        "della liquidita' sono quindi approssimati.", icon="⚠️",
    )

st.divider()

# ---------------------------------------------------------------------------
st.subheader("Copertura dei prezzi — il numero che conta")
cov = universe.coverage_report(ds.membership, ds.close_adj)
cov = cov[cov["in_indice"] > 0]

c1, c2, c3 = st.columns(3)
c1.metric("Coverage medio", f"{cov['coverage'].mean():.2%}")
c2.metric("Coverage minimo", f"{cov['coverage'].min():.2%}")
c3.metric("Mesi sotto il 95%", f"{int((cov['coverage'] < 0.95).sum())}")

ui.chart(plotting.plot_coverage(cov), key="cov")

with st.expander("Coverage per fascia — dove si concentrano i buchi"):
    cov_band = universe.coverage_by_band(ds.membership, ds.close_adj, sig.bands, BAND_NAMES)
    cov_band = cov_band.dropna(how="all")
    if not cov_band.empty:
        ui.chart(plotting.plot_series(
            {c: cov_band[c] for c in cov_band.columns},
            "Coverage per fascia", "% con prezzo", percent=True, height=340),
            key="covband")
        st.caption(
            "Se il coverage e' peggiore in **Fondo Griglia** — e in genere lo e' — "
            "il bias spinge il risultato a favore della tesi contrarian: i peggiori "
            "risultati di quel paniere spariscono dal backtest."
        )
    else:
        st.info("Nessun dato di copertura per fascia in questo dataset.")

didactics.render("coverage", expanded=True)

st.divider()

# ---------------------------------------------------------------------------
st.subheader("Universo eleggibile")
size = universe.eligible_universe_size(sig.eligible)
size = size[size.index >= pd.Timestamp(cfg.backtest_start)]
ui.chart(plotting.plot_universe_size(size, cfg.min_eligible), key="usize")

u1, u2, u3 = st.columns(3)
u1.metric("Media", f"{size.mean():.0f}")
u2.metric("Minimo", f"{size.min():.0f}")
u3.metric("Mesi sotto la soglia", f"{int((size < cfg.min_eligible).sum())}")

st.markdown("**Quanti titoli cadono per ciascun criterio** (media sul periodo)")
crit_rows = []
for name, mask in sig.criteria.items():
    sub = mask.loc[mask.index >= pd.Timestamp(cfg.backtest_start)]
    crit_rows.append({
        "Criterio": name.replace("_", " "),
        "Titoli che lo superano (media)": float(sub.sum(axis=1).mean()),
    })
st.dataframe(pd.DataFrame(crit_rows).style.format({"Titoli che lo superano (media)": "{:.0f}"}),
             width="stretch", hide_index=True)

didactics.render("universe_size", expanded=True)

st.divider()

# ---------------------------------------------------------------------------
st.subheader("Titoli esclusi dal cap sul prezzo")
over_cap = (ds.open_raw > cfg.max_share_price) & ds.membership
per_month = over_cap.sum(axis=1).resample("ME").last()
per_month = per_month[per_month.index >= pd.Timestamp(cfg.backtest_start)]

e1, e2 = st.columns([2, 1])
with e1:
    ui.chart(plotting.plot_series({"esclusi dal cap": per_month},
                                  f"Titoli sopra {cfg.max_share_price:,.0f}$ per azione",
                                  "numero di titoli", height=320), key="cap")
with e2:
    names_over = over_cap.any(axis=0)
    lst = sorted(names_over[names_over].index.tolist())
    st.metric("Titoli mai acquistabili", f"{len(lst)}")
    if lst:
        st.dataframe(pd.DataFrame({"ticker": lst, "settore": [ds.sectors.get(t) for t in lst]}),
                     width="stretch", hide_index=True, height=240)

didactics.render("price_cap_exclusions")

st.divider()

# ---------------------------------------------------------------------------
st.subheader("Qualita' delle serie e ticker riassegnati")
q1, q2 = st.columns(2)

with q1:
    st.markdown("**Rendimenti giornalieri anomali (|r| > 60%)**")
    anom = universe.price_anomalies(ds.close_adj)
    if anom.empty:
        st.success("Nessuna anomalia rilevata.", icon="✅")
    else:
        st.dataframe(anom.head(30), width="stretch", height=280)
        st.caption(f"{len(anom)} titoli con almeno un'anomalia. "
                   "Quasi sempre split non gestiti nella serie adjusted.")

with q2:
    st.markdown("**Codici con possibile riassegnazione del ticker**")
    reused = man.get("ticker_reuse_suspects", [])
    if not reused:
        st.success("Nessun codice sospetto.", icon="✅")
    else:
        st.dataframe(pd.DataFrame({"codice": reused}), width="stretch", height=280)
        st.caption(
            f"{len(reused)} codici usati da societa' diverse in periodi diversi. "
            "Le occorrenze non piu' recenti sono scartate per non incollare due "
            "aziende nella stessa serie."
        )

didactics.render("data_quality")

st.divider()

# ---------------------------------------------------------------------------
st.subheader("Turnover e costi realizzati")
res = ui.get_study(cfg)
turn = {n: r.turnover[r.turnover.index >= pd.Timestamp(cfg.backtest_start)]
        for n, r in res.results.items() if n != "Universo eleggibile (senza costi)"}
ui.chart(plotting.plot_series(turn, "Turnover per ribilanciamento",
                              "frazione del portafoglio", percent=True, height=340),
         key="turnover")

cost_tab = res.metrics[["Turnover medio", "Posizioni medie", "Costo annuo %", "Costi totali $"]]
st.dataframe(ui.format_metrics(cost_tab), width="stretch")
didactics.render("turnover", expanded=True)

with st.expander("Manifest completo del dataset"):
    st.json(man)
