"""
Verdetto: cosa si puo' onestamente affermare con i dati a disposizione.

E' la pagina che risponde alla domanda dello studio in modo leggibile da chi
non ha seguito tutta la costruzione — senza per questo nascondere i limiti.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from track import analysis, backtest as bt
from track import didactics, plotting, study, ui, verdict
from track.config import BAND_NAMES

ui.page_config("Verdetto")

if not ui.require_dataset():
    st.stop()

cfg = ui.sidebar_config()
ds = ui.get_dataset()
res = ui.get_study(cfg)
sig = ui.get_signals(cfg)

ui.header("Verdetto", ds)

if res.panel.n_periods < 24:
    st.error("Campione troppo corto per un verdetto.")
    st.stop()

returns = {n: r.returns for n, r in res.results.items()}

# Il vincitore non e' deciso a priori: si sceglie il paniere estremo con il
# CAGR piu' alto, cosi' il verdetto resta valido anche se i dati cambiano segno.
estremi = [study.P_TOP, study.P_BOTTOM]
cagr = {n: res.metrics.loc[n, "CAGR"] for n in estremi if n in res.metrics.index}
vincitore = max(cagr, key=cagr.get)
perdente = min(cagr, key=cagr.get)

with st.spinner("Raccolgo le evidenze…"):
    stress = ui.get_delisting_stress(cfg)
    persistenza = analysis.transition_matrix(sig.bands, cfg.n_bands, BAND_NAMES, 21)
    null = ui.get_bootstrap(cfg, min(cfg.n_bootstrap, 500))
    pv = {
        "Sharpe": {
            n: bt.empirical_pvalue(null["Sharpe"], res.metrics.loc[n, "Sharpe"])
            for n in res.metrics.index if "Sharpe" in res.metrics.columns
        }
    }

v = verdict.build_verdict(
    returns, res.metrics,
    vincitore=vincitore, perdente=perdente, riferimento=study.P_UNIVERSE,
    capitale=cfg.capital, stress=stress, persistenza=persistenza, null_pvalues=pv,
)

# ---------------------------------------------------------------------------
st.markdown("## Con i dati a disposizione possiamo affermare che…")
for frase in v.affermazioni:
    st.markdown(f"- {frase}")

colori = {"convergente": "green", "moderata": "orange", "contraddittoria": "red"}
st.markdown("")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vantaggio annuo", f"{v.differenza_annua:+.2%}",
          help="Differenza media di rendimento annualizzata fra i due panieri.")
c2.metric("Intervallo 95%", f"{v.ic95[0]:+.1%} … {v.ic95[1]:+.1%}",
          help="Se comprende lo zero, un vantaggio nullo resta compatibile con i dati.")
c3.metric("Evidenze concordi", f"{v.concordi}/{v.valutabili}")
c4.metric("Forza dell'evidenza", v.forza.capitalize())
st.badge(f"Evidenza {v.forza}", color=colori.get(v.forza, "gray"))

# ---------------------------------------------------------------------------
st.divider()
st.subheader("Non possiamo invece affermare che…")
for frase in v.non_affermabili:
    st.markdown(f"- {frase}")

st.error(verdict.avvertenza(v), icon="⚠️")

# ---------------------------------------------------------------------------
st.divider()
st.subheader("Le evidenze, una per una")
st.caption(
    "Nessuna di queste basta da sola. Il punto e' se puntano tutte nella stessa "
    "direzione: e' cosi' che si decide quando la statistica non chiude la partita."
)

righe = []
for e in v.evidenze:
    righe.append({
        "Indicatore": e.nome,
        "Verso": {True: "a favore", False: "contro", None: "neutro"}[e.favorevole],
        "Dettaglio": e.dettaglio,
        "Peso": e.peso,
    })
tab = pd.DataFrame(righe)


def _colora(row):
    colore = {"a favore": "rgba(63,158,106,0.30)",
              "contro": "rgba(193,68,75,0.30)"}.get(row["Verso"], "")
    return [f"background-color: {colore}" if colore else ""] * len(row)


st.dataframe(tab.style.apply(_colora, axis=1), width="stretch", hide_index=True,
             column_config={"Dettaglio": st.column_config.TextColumn(width="large")})

# ---------------------------------------------------------------------------
st.divider()
st.subheader(f"Avresti fatto meglio a… — {v.capitale_iniziale:,.0f} USD investiti da {v.da}")

ordine = [n for n in study.PORTFOLIO_ORDER if n in v.capitale_finale]
finale = pd.Series({n: v.capitale_finale[n] for n in ordine}).sort_values(ascending=False)

f1, f2 = st.columns([3, 2])
with f1:
    ui.chart(plotting.plot_equity(
        {n: res.results[n].equity for n in ordine},
        title=f"Evoluzione di {v.capitale_iniziale:,.0f} USD per paniere"), key="verdetto_eq")
with f2:
    st.markdown("**Capitale finale**")
    st.dataframe(
        pd.DataFrame({
            "Paniere": finale.index,
            "Finale (USD)": finale.to_numpy(),
            "Multiplo": finale.to_numpy() / v.capitale_iniziale,
        }),
        width="stretch", hide_index=True,
        column_config={
            "Finale (USD)": st.column_config.NumberColumn(format="%.0f"),
            "Multiplo": st.column_config.NumberColumn(format="%.1fx"),
        },
    )
    st.caption(
        "Il paniere **universo eleggibile** e' simulato **senza costi** e con "
        "centinaia di posizioni: e' un riferimento teorico, non un'alternativa "
        "realmente attuabile con questo capitale."
    )

didactics.render("verdetto", expanded=True)

# ---------------------------------------------------------------------------
st.divider()
st.subheader("Se dovessi tradurlo in una regola operativa")

peggiore_3a = v.evidenze[3].dettaglio if len(v.evidenze) > 3 else ""
st.markdown(
    f"""
1. **Scegli {v.vincitore}, non {v.perdente}.** È la parte più solida del
   verdetto: {v.concordi} indicatori su {v.valutabili} concordano, e la scelta
   costa meno da mantenere ({res.metrics.loc[v.vincitore, 'Costo annuo %']:.2%}
   contro {res.metrics.loc[v.perdente, 'Costo annuo %']:.2%} l'anno di
   commissioni e spread).

2. **Aspettati periodi morti lunghi.** {peggiore_3a.capitalize() if peggiore_3a else ''}
   Su tre anni il risultato peggiore osservato è pesantemente negativo: chi non
   può reggerlo senza cambiare strategia non dovrebbe iniziarla.

3. **Non aspettarti il {v.differenza_annua:+.1%} annuo.** È la stima centrale,
   ma l'intervallo va da {v.ic95[0]:+.1%} a {v.ic95[1]:+.1%}. Dimensiona le
   attese sul limite inferiore, non su quello centrale.

4. **Il drawdown è la vera valuta.** {v.vincitore} ha toccato
   {res.metrics.loc[v.vincitore, 'Max DD']:.0%} contro
   {res.metrics.loc[study.P_UNIVERSE, 'Max DD']:.0%} del riferimento passivo.
   Il rendimento aggiuntivo si paga in oscillazione, non è gratis.
"""
)

st.info(
    "Per capire **come** si arriva a questi numeri vai alla pagina Backtest; per "
    "sapere **quanto valgono i dati sottostanti**, alla pagina Diagnostica. "
    "Il verdetto non sostituisce nessuna delle due.",
    icon="➡️",
)
