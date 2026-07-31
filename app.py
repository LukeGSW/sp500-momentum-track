"""
Monitor di Forza Relativa — pagina principale.

Dove si trova oggi ogni titolo dell'S&P 500 lungo un asse verticale di forza
relativa, e in che direzione si sta muovendo.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from track import analysis, didactics, plotting, study, ui
from track.config import BAND_NAMES, GICS_SECTORS

ui.page_config("Monitor di Forza Relativa")

if not ui.require_dataset():
    st.stop()

cfg = ui.sidebar_config()
ds = ui.get_dataset()
sig = ui.get_signals(cfg)

ui.header("Monitor di Forza Relativa", ds)

st.markdown(
    """
Ogni pallino e' un titolo. **In alto i forti, in basso i deboli** — misurati in
modo relativo rispetto a tutti gli altri, non in valore assoluto. Le undici
colonne sono i settori GICS. Il momentum **non e' un secondo asse**: e' la
pendenza della traiettoria, cioe' il movimento verticale del titolo nelle settimane
precedenti.
"""
)

# ---------------------------------------------------------------------------
# Data di riferimento
# ---------------------------------------------------------------------------
valid_dates = sig.force.dropna(how="all").index
if len(valid_dates) == 0:
    st.error("Nessuna data con segnale calcolabile. Storia insufficiente nel dataset.")
    st.stop()

c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
with c1:
    as_of = st.select_slider(
        "Data di riferimento",
        options=list(valid_dates),
        value=valid_dates[-1],
        format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"),
        help="Trascina per far scorrere la mappa nel tempo e vedere come si "
             "riordinano i titoli.",
    )
with c2:
    x_mode_label = st.segmented_control(
        "Asse orizzontale", ["Settori", "RS Slope"], default="Settori",
        help="'Settori' mostra 11 colonne settoriali GICS e rende visibile la rotazione "
             "settoriale. 'RS Slope' mette la velocita' sull'asse x.",
    ) or "Settori"
x_mode = "sector" if x_mode_label == "Settori" else "velocity"

members = study.portfolio_members_at(sig, cfg, as_of)

snapshot = analysis.build_snapshot(
    as_of,
    force=sig.force, velocity=sig.velocity, bands=sig.bands, eligible=sig.eligible,
    sectors=ds.sectors, names=ds.names, band_names=BAND_NAMES,
    prices_raw=ds.open_raw, max_share_price=cfg.max_share_price,
    portfolios=members,
)
# La colonna `paniere` resta completa (utile in tabella), ma sul grafico
# evidenziamo un paniere alla volta: quattro anelli sovrapposti su meta' del
# campo non comunicano nulla.
snapshot["paniere_completo"] = snapshot["paniere"]

if snapshot.empty:
    st.warning("Nessun titolo eleggibile a questa data. "
               "Prova a spegnere il filtro sulla media mobile nella barra laterale.")
    st.stop()

# ---------------------------------------------------------------------------
# Indicatori
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Titoli eleggibili", f"{len(snapshot)}",
          help="Nell'indice, con storia sufficiente, sotto il cap di prezzo e "
               "(se attivo) sopra la media a 200 sedute.")
k2.metric("In Q5 Leader", f"{int((snapshot['band_label'] == 'Q5 Leader').sum())}",
          help="Il quintile piu' forte alla data selezionata.")
k3.metric("In Q1 Laggard", f"{int((snapshot['band_label'] == 'Q1 Laggard').sum())}",
          help="Il quintile piu' debole fra gli eleggibili.")
salgono = int((snapshot["movimento"] == "sale").sum())
k4.metric("In risalita sulla mappa", f"{salgono}",
          delta=f"{salgono / max(len(snapshot), 1):.0%} del campo",
          help="Titoli che hanno guadagnato almeno una fascia nell'ultimo mese.")
k5.metric("RS Slope mediano", f"{snapshot['V'].median():+.2f}",
          help="Segno positivo: il campo nel suo insieme sta accelerando.")

if len(snapshot) < cfg.min_eligible:
    st.warning(
        f"Solo **{len(snapshot)}** titoli eleggibili, sotto la soglia minima di "
        f"{cfg.min_eligible}. I quintili contengono pochi nomi e la classificazione "
        "e' instabile: e' quello che succede nei bear market, quando il filtro sulla "
        "media mobile svuota l'universo.",
        icon="⚠️",
    )

st.divider()

# ---------------------------------------------------------------------------
# Il grafico
# ---------------------------------------------------------------------------
f1, f2, f3 = st.columns([2, 2, 2])
with f1:
    sectors_present = [s for s in GICS_SECTORS if (snapshot["sector"] == s).any()]
    sel_sectors = st.multiselect("Filtra settori", sectors_present, default=[],
                                 placeholder="tutti i settori")
with f2:
    highlight_pf = st.selectbox(
        "Evidenzia il paniere",
        ["nessuno", study.P_TOP, study.P_BOTTOM, study.P_PULLBACK, study.P_MID],
        index=1,
        help="Cerchia sulla mappa i titoli che quel paniere comprerebbe oggi, e ne "
             "disegna le traiettorie. Un paniere alla volta: quattro anelli sovrapposti "
             "non comunicano nulla.",
    )
with f3:
    trail_choice = st.multiselect(
        "Aggiungi la traiettoria di…", sorted(snapshot["ticker"].tolist()), default=[],
        placeholder="titoli da tracciare singolarmente",
        help="Con centinaia di titoli tutte le traiettorie insieme sono illeggibili.",
    )

# solo il paniere selezionato viene cerchiato sul grafico
selected_members = members.get(highlight_pf, []) if highlight_pf != "nessuno" else []
snapshot["in_portfolio"] = snapshot["ticker"].isin(selected_members)

trail_tickers = sorted(set(list(trail_choice) + list(selected_members)))

view = snapshot if not sel_sectors else snapshot[snapshot["sector"].isin(sel_sectors)]
trails = analysis.build_trails(
    [t for t in trail_tickers if t in set(view["ticker"])],
    as_of, force=sig.force, velocity=sig.velocity, weeks=cfg.trail_weeks,
)

ui.chart(
    plotting.plot_track(
        view, trails, band_names=BAND_NAMES, n_bands=cfg.n_bands,
        x_mode=x_mode, highlight=set(trail_choice),
        portfolio_label=highlight_pf if highlight_pf != "nessuno" else "In portafoglio",
        title=f"Monitor di Forza Relativa — {pd.Timestamp(as_of).strftime('%d %B %Y')}",
    ),
    key="track",
)
didactics.render("mappa", expanded=True)
didactics.render("velocita")

st.divider()

# ---------------------------------------------------------------------------
# Composizione
# ---------------------------------------------------------------------------
st.subheader("Chi sta dove")
ui.chart(plotting.plot_band_sector_matrix(snapshot, BAND_NAMES), key="bandsec")
didactics.render("band_distribution")

# ---------------------------------------------------------------------------
# Tabella
# ---------------------------------------------------------------------------
st.subheader("Tabella titoli")

only_pf = st.checkbox(
    "Mostra solo i titoli che uno dei panieri comprerebbe oggi", value=False,
    help="La colonna 'Paniere' indica quale. Un titolo puo' appartenere a un solo "
         "paniere, perche' i panieri prendono fasce diverse.",
)
table = snapshot[snapshot["paniere_completo"].notna()] if only_pf else snapshot
table = table.assign(paniere=table["paniere_completo"])

st.dataframe(
    table[["ticker", "name", "sector", "F", "V", "band_label", "band_prev_label",
           "movimento", "giorni_in_fascia", "prezzo", "tradable", "paniere"]],
    width="stretch", hide_index=True, height=430,
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "name": st.column_config.TextColumn("Societa'"),
        "sector": st.column_config.TextColumn("Settore"),
        "F": st.column_config.NumberColumn("RS Score", format="%.2f",
                                           help="z-score trasversale: 0 = titolo mediano"),
        "V": st.column_config.NumberColumn("RS Slope", format="%.2f",
                                           help="pendenza della traiettoria, standardizzata"),
        "band_label": st.column_config.TextColumn("Fascia"),
        "band_prev_label": st.column_config.TextColumn("Fascia 1 mese fa"),
        "movimento": st.column_config.TextColumn("Movimento", width="small"),
        "giorni_in_fascia": st.column_config.NumberColumn("Sedute in fascia", format="%d"),
        "prezzo": st.column_config.NumberColumn("Prezzo", format="$%.2f"),
        "tradable": st.column_config.CheckboxColumn("Sotto il cap"),
        "paniere": st.column_config.TextColumn("Paniere"),
    },
)
didactics.render("snapshot_table")

st.divider()

# ---------------------------------------------------------------------------
# Dinamica storica
# ---------------------------------------------------------------------------
st.subheader("Come si muovono i titoli lungo la mappa")
st.caption(
    "Queste due viste rispondono a meta' della domanda dello studio **prima** del "
    "backtest: misurano se la forza relativa e' un fenomeno persistente o se le "
    "posizioni ruotano continuamente."
)

t1, t2 = st.columns([3, 2])
with t1:
    tm = analysis.transition_matrix(sig.bands, cfg.n_bands, BAND_NAMES, horizon_days=21)
    ui.chart(plotting.plot_transition_matrix(tm), key="transitions")
with t2:
    dw = analysis.dwell_time(sig.bands, BAND_NAMES)
    st.markdown("**Permanenza per fascia**")
    st.dataframe(dw, width="stretch", hide_index=True,
                 column_config={
                     "Permanenza media (mesi)": st.column_config.NumberColumn(format="%.1f"),
                     "Permanenza mediana (mesi)": st.column_config.NumberColumn(format="%.1f"),
                     "Permanenza massima (mesi)": st.column_config.NumberColumn(format="%.0f"),
                     "Episodi": st.column_config.NumberColumn(format="%d"),
                 })
    st.caption(f"Holding period configurato: **{cfg.holding_months} mesi**. "
               "Confronta con la permanenza mediana.")

didactics.render("transition_matrix", expanded=True)
didactics.render("dwell_time")

st.divider()
st.info(
    "Questa pagina descrive **dove** si trovano i titoli. Per sapere se conviene "
    "comprarli vai alla pagina **Backtest**: e' li' che la domanda riceve una "
    "risposta misurata, con l'ipotesi nulla e lo stress test sul survivorship bias.",
    icon="➡️",
)
