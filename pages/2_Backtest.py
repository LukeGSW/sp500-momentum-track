"""
Backtest — qui la domanda riceve una risposta misurata.

Quattro panieri da 100.000$ ciascuno, indipendenti, piu' un riferimento senza
attriti e un'ipotesi nulla costruita con lo stesso motore.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from track import backtest as bt
from track import didactics, plotting, study, ui
from track.config import PREREGISTERED

ui.page_config("Backtest")

if not ui.require_dataset():
    st.stop()

cfg = ui.sidebar_config()
ds = ui.get_dataset()
res = ui.get_study(cfg)

ui.header("Backtest", ds)

if res.panel.n_periods < 24:
    st.error(
        f"Solo {res.panel.n_periods} ribilanciamenti disponibili: troppo pochi per "
        "qualunque conclusione. Controlla la data di inizio o la copertura dei dati."
    )
    st.stop()

st.caption(didactics.escape_markdown(
    f"{res.panel.n_periods} ribilanciamenti mensili · "
    f"{res.diagnostics['primo_ribilanciamento']} → {res.diagnostics['ultimo_ribilanciamento']} · "
    f"holding {cfg.holding_months} mesi su {cfg.n_tranches} tranche sfalsate · "
    f"{cfg.n_names} titoli per paniere · slot {cfg.slot_value:,.0f}$ · "
    f"costo {study.current_cost_bps(cfg):.1f} bps per rotazione"
))

equities = {name: r.equity for name, r in res.results.items()}
returns = {name: r.returns for name, r in res.results.items()}

# ---------------------------------------------------------------------------
# La risposta, in una riga
# ---------------------------------------------------------------------------
m = res.metrics
top_c = m.loc[study.P_TOP, "CAGR"]
bot_c = m.loc[study.P_BOTTOM, "CAGR"]
pull_c = m.loc[study.P_PULLBACK, "CAGR"] if study.P_PULLBACK in m.index else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Testa Corsa (momentum forte)", f"{top_c:.1%}",
          delta=f"{top_c - bot_c:+.1%} vs Fondo Griglia")
k2.metric("Fondo Griglia (debolezza relativa)", f"{bot_c:.1%}")
k3.metric("Leader in ritracciamento", "—" if pd.isna(pull_c) else f"{pull_c:.1%}",
          help="Titoli della fascia alta con Spinta negativa: la 'debolezza "
               "momentanea di un titolo forte'.")
k4.metric("t-stat dello spread (Newey-West)", f"{res.spread_tstat:.2f}",
          help="|t| > 2 e' la soglia convenzionale; su dati finanziari con scelta "
               "di parametri molti ricercatori chiedono |t| > 3.")

if cfg.hash() != PREREGISTERED.hash():
    st.warning(
        "Configurazione **personalizzata**: questi risultati sono esplorazione, non "
        "una conclusione. Confrontali con la configurazione preregistrata prima di "
        "trarre inferenze.", icon="🧪",
    )

tab_res, tab_rob, tab_dati = st.tabs(
    ["Risultato", "Robustezza e inferenza", "Costi, dati e orizzonti"]
)

# ===========================================================================
with tab_res:
    log_scale = st.toggle("Scala logaritmica", value=True,
                          help="Su un periodo lungo la scala lineare rende invisibile "
                               "tutto cio' che accade nei primi anni.")
    ui.chart(plotting.plot_equity(equities, log_scale=log_scale), key="equity")
    didactics.render("equity_curves", expanded=True)

    ui.chart(plotting.plot_fixed_capital(
        {n: r.fixed_capital_pnl for n, r in res.results.items()}), key="fixedcap")
    didactics.render("fixed_capital_pnl")

    st.subheader("Metriche")
    st.dataframe(ui.format_metrics(m), width="stretch")
    didactics.render("metrics_table", expanded=True)

    ui.chart(plotting.plot_drawdown(equities), key="dd")
    didactics.render("drawdown")

# ===========================================================================
with tab_rob:
    st.subheader("Ipotesi nulla: 30 titoli estratti a caso")
    st.caption(
        "Stesso motore, stessi costi, stesso arrotondamento a lotti interi, stesso "
        "universo eleggibile. L'unica differenza e' che la selezione e' casuale."
    )

    n_draws = st.select_slider("Estrazioni", options=[100, 250, 500, 1000, 2000],
                               value=min(cfg.n_bootstrap, 500))
    null = ui.get_bootstrap(cfg, int(n_draws))

    observed = {n: m.loc[n, "CAGR"] for n in (study.P_TOP, study.P_BOTTOM, study.P_PULLBACK)
                if n in m.index}
    observed_sharpe = {n: m.loc[n, "Sharpe"] for n in observed}

    n1, n2 = st.columns(2)
    with n1:
        ui.chart(plotting.plot_null_distribution(null["CAGR"], observed, "CAGR"), key="null_cagr")
    with n2:
        ui.chart(plotting.plot_null_distribution(null["Sharpe"], observed_sharpe, "Sharpe"),
                 key="null_sharpe")

    pvals = pd.DataFrame({
        "p-value CAGR": {n: bt.empirical_pvalue(null["CAGR"], v) for n, v in observed.items()},
        "p-value Sharpe": {n: bt.empirical_pvalue(null["Sharpe"], v)
                           for n, v in observed_sharpe.items()},
        "CAGR mediano casuale": {n: float(null["CAGR"].median()) for n in observed},
    })
    st.dataframe(pvals.style.format({"p-value CAGR": "{:.3f}", "p-value Sharpe": "{:.3f}",
                                     "CAGR mediano casuale": "{:.2%}"}), width="stretch")
    didactics.render("null_distribution", expanded=True)

    st.divider()
    st.subheader("Spread Testa Corsa − Fondo Griglia")
    s1, s2, s3 = st.columns(3)
    s1.metric("Media mensile", f"{res.spread.mean():+.2%}")
    s2.metric("Annualizzata", f"{res.spread.mean() * 12:+.1%}")
    s3.metric("t-stat (NW, 3 lag)", f"{res.spread_tstat:.2f}")
    ui.chart(plotting.plot_series({"Spread Testa - Fondo": res.spread.cumsum()},
                                  "Spread cumulato (somma dei rendimenti mensili)",
                                  "somma cumulata", percent=True), key="spread")
    didactics.render("longshort")

    st.divider()
    st.subheader("Tenuta nei diversi regimi di mercato")
    sub = bt.subperiod_table(res.results, res.rf_period)
    if not sub.empty:
        pivot = sub.pivot_table(index="Periodo", columns="Paniere", values="CAGR")
        order = [p for p, _, _ in bt.SUBPERIODS if p in pivot.index]
        st.dataframe(ui.heat_table(pivot.reindex(order), "{:.1%}"), width="stretch")
    didactics.render("subperiods", expanded=True)

    st.divider()
    st.subheader("Quanto dipende da pochi mesi")
    hm1, hm2 = st.columns([3, 2])
    with hm1:
        which = st.selectbox("Paniere", list(res.results), index=0, key="hm_pick")
        ui.chart(plotting.plot_monthly_heatmap(returns[which], f"Rendimenti mensili — {which}"),
                 key="heatmap")
    with hm2:
        st.markdown("**Togliendo i mesi migliori**")
        rows = []
        for name, r in returns.items():
            for k in (1, 3, 5, 10):
                d = bt.drop_best_months(r, k)
                if d:
                    rows.append({"Paniere": name, "Mesi tolti": k,
                                 "Rendimento totale": d[f"senza i {k} mesi migliori"]})
        if rows:
            tab = pd.DataFrame(rows).pivot_table(index="Paniere", columns="Mesi tolti",
                                                 values="Rendimento totale")
            full = {n: float((1 + r).prod()) - 1 for n, r in returns.items()}
            tab.insert(0, 0, pd.Series(full))
            st.dataframe(tab.style.format("{:.0%}", na_rep="—"), width="stretch")
            st.caption("Colonna 0 = periodo intero. Se il vantaggio sparisce togliendo "
                       "cinque mesi, non e' un edge ripetibile.")
    didactics.render("monthly_heatmap")

# ===========================================================================
with tab_dati:
    st.subheader("Sensibilita' ai costi di transazione")
    sens = ui.get_cost_sensitivity(cfg)
    ui.chart(plotting.plot_cost_sensitivity(sens, study.current_cost_bps(cfg)), key="costs")
    didactics.render("cost_breakeven", expanded=True)

    st.divider()
    st.subheader("Stress test sul survivorship bias")
    st.caption(
        "Il backtest ripetuto imponendo un rendimento terminale ai titoli che "
        "spariscono dai dati. E' il test piu' importante di questa pagina."
    )
    stress = ui.get_delisting_stress(cfg)
    st.dataframe(ui.heat_table(stress, "{:.2%}"), width="stretch")
    liq = {k.split(" — ")[0]: v for k, v in res.diagnostics.items()
           if k.endswith("liquidazioni_forzate")}
    if liq:
        st.caption("Liquidazioni forzate per titolo sparito: " +
                   " · ".join(f"{k}: {int(v)}" for k, v in liq.items()))
    didactics.render("delisting_stress", expanded=True)

    st.divider()
    st.subheader("Griglia degli orizzonti")
    st.markdown(
        "Lo stesso studio con holding di 1, 3 e 6 mesi, **ciascuno con il lookback "
        "congruente al proprio holding**. Un segnale reale produce risultati della "
        "stessa direzione su tutti e tre; uno che funziona solo a un orizzonte e si "
        "inverte sugli altri e' un artefatto."
    )
    if st.button("Calcola la griglia (ricalcola Forza e Spinta: puo' richiedere tempo)"):
        rows = []
        prog = st.progress(0.0, text="…")
        for i, h in enumerate((1, 3, 6)):
            prog.progress(i / 3, text=f"holding {h} mesi…")
            c = replace(cfg, holding_months=h)
            r = ui.get_study(c)
            for name in (study.P_TOP, study.P_BOTTOM, study.P_PULLBACK):
                if name in r.metrics.index:
                    rows.append({
                        "Holding (mesi)": h,
                        "Orizzonti F": ", ".join(str(x) for x in c.horizons),
                        "Paniere": name,
                        "CAGR": r.metrics.loc[name, "CAGR"],
                        "Sharpe": r.metrics.loc[name, "Sharpe"],
                        "Turnover": r.metrics.loc[name, "Turnover medio"],
                        "Costo annuo %": r.metrics.loc[name, "Costo annuo %"],
                    })
            rows.append({
                "Holding (mesi)": h, "Orizzonti F": ", ".join(str(x) for x in c.horizons),
                "Paniere": "Spread Testa − Fondo",
                "CAGR": r.metrics.loc[study.P_TOP, "CAGR"] - r.metrics.loc[study.P_BOTTOM, "CAGR"],
                "Sharpe": np.nan, "Turnover": np.nan, "Costo annuo %": np.nan,
            })
        prog.empty()
        grid = pd.DataFrame(rows)
        st.session_state["horizon_grid"] = grid

    if "horizon_grid" in st.session_state:
        grid = st.session_state["horizon_grid"]
        st.dataframe(
            grid.style.format({"CAGR": "{:.2%}", "Sharpe": "{:.2f}",
                               "Turnover": "{:.1%}", "Costo annuo %": "{:.2%}"}, na_rep="—"),
            width="stretch", hide_index=True,
        )
        st.caption(f"La configurazione preregistrata e' holding **{PREREGISTERED.holding_months} "
                   f"mesi** con orizzonti {', '.join(str(h) for h in PREREGISTERED.horizons)}.")
    didactics.render("horizon_grid", expanded=True)

    st.divider()
    with st.expander("Diagnostica del run"):
        st.json(res.diagnostics)
