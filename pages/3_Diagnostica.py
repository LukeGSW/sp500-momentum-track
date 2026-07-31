"""
Diagnostica dei dati.

Questa pagina determina quanto vale tutto il resto della dashboard. Un
backtest su dati bucati non e' un backtest prudente: e' un backtest sbagliato,
e sbagliato in una direzione precisa.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from track import analysis, didactics, plotting, storage, ui, universe
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

# Il dataset viene scaricato una volta sola per container: senza un modo
# esplicito di forzarlo, una Release nuova non arriverebbe mai all'app e si
# leggerebbero risultati vecchi credendoli aggiornati.
r1, r2 = st.columns([3, 1], vertical_alignment="bottom")
with r1:
    costruito = man.get("built_at", "sconosciuto")
    st.caption(
        f"Dataset in uso costruito il **{costruito}** (UTC). Se hai rilanciato la "
        "pipeline dopo questo momento, l'app sta ancora servendo la versione "
        "precedente: usa il pulsante per riscaricarla."
    )
with r2:
    if st.button("Riscarica dataset", width="stretch"):
        url, repo = ui._secret("DATA_URL"), ui._secret("DATA_REPO")
        if not url and not repo:
            st.error("Nessun `DATA_URL` o `DATA_REPO` configurato nei secrets.")
        else:
            try:
                with st.spinner("Scarico l'ultima Release…"):
                    ok = storage.ensure_dataset(
                        url=url, repo=repo,
                        token=ui._secret("DATA_TOKEN") or ui._secret("GITHUB_TOKEN"),
                        force=True,
                    )
                if ok:
                    st.cache_resource.clear()
                    st.success("Dataset aggiornato. Ricarico…", icon="✅")
                    st.rerun()
                else:
                    st.error("Archivio scaricato ma incompleto.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"**{type(exc).__name__}**: {exc}")

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

st.divider()

# ---------------------------------------------------------------------------
st.subheader("Serie escluse dallo studio")

fonte = man.get("exclusions_source")
if fonte:
    if not fonte.get("esiste"):
        st.error(
            f"**Il file di esclusioni non e' stato trovato** durante la costruzione "
            f"del dataset (`{fonte.get('file')}`). Nessuna serie compromessa e' stata "
            "esclusa: i risultati contengono i mesi anomali.",
            icon="🚨",
        )
    else:
        st.caption(
            f"File letto dalla pipeline: `{fonte.get('file')}` — "
            f"**{fonte.get('righe')} righe**, ticker dichiarati: "
            f"`{'`, `'.join(fonte.get('ticker') or []) or 'nessuno'}` "
            f"(impronta `{fonte.get('impronta')}`)."
        )
        st.caption(
            "Se un ticker che hai aggiunto non compare in questo elenco, il file "
            "usato per costruire il dataset **non e' quello aggiornato**: la "
            "modifica non e' arrivata nel repository prima del run."
        )

escl = man.get("exclusions_applied") or []
if not escl:
    st.info(
        "Nessuna esclusione attiva. Se il dataset e' stato costruito con una versione "
        "precedente della pipeline, ricostruiscilo per applicare `exclusions.csv`.",
        icon="ℹ️",
    )
else:
    applicate = [e for e in escl if e.get("applicata")]
    st.caption(
        f"{len(applicate)} serie escluse su {len(escl)} dichiarate, **prima di "
        "qualunque calcolo**. La lista e' versionata in `exclusions.csv` e finisce "
        "nei caveats dell'export: l'esclusione e' parte del metodo, non un ritocco "
        "sui risultati."
    )
    st.dataframe(
        pd.DataFrame(escl), width="stretch", hide_index=True,
        column_config={
            "osservazioni_rimosse": st.column_config.NumberColumn("Sedute rimosse", format="%d"),
            "applicata": st.column_config.CheckboxColumn("Applicata"),
            "reason": st.column_config.TextColumn("Motivo", width="large"),
        },
    )
    st.warning(
        "Le esclusioni sono state individuate **guardando i risultati**. E' pulizia "
        "legittima quando l'errore e' verificabile in modo indipendente (una fusione "
        "documentata, non 'peggiora la performance'), ma resta una scelta post-hoc: "
        "un risultato ottenuto dopo aver rimosso dati scelti a posteriori vale meno "
        "di uno preregistrato. Riporta sempre i numeri con e senza.",
        icon="⚠️",
    )

st.divider()

# ---------------------------------------------------------------------------
st.subheader("Caccia alle anomalie: dal mese sospetto al titolo colpevole")
st.markdown(
    "Un paniere equipesato di 30 titoli **non fa ±40% in un mese**. Quando succede "
    "e' quasi sempre uno split non gestito o un prezzo sbagliato in una singola "
    "posizione. Qui si risale dal mese al titolo."
)

# lo slider lavora in punti percentuali interi: passandogli una frazione,
# il formato "%.0f%%" arrotonderebbe 0,20 a "0%"
soglia = st.slider("Soglia di sospetto sul rendimento mensile (%)", 10, 50, 20, 5,
                   help="Sopra il 30% l'ipotesi 'evento di mercato' e' da scartare "
                        "quasi sempre. Tra 15% e 30% va guardato caso per caso: "
                        "marzo 2020 e ottobre 2008 sono esistiti davvero.") / 100.0

if st.button("Cerca i mesi anomali (esegue il backtest con l'attribuzione attiva)"):
    st.session_state["attrib_study"] = ui.get_study_with_attribution(cfg)

if "attrib_study" in st.session_state:
    res_attr = st.session_state["attrib_study"]
    anom = analysis.anomalous_periods(res_attr.results, threshold=soglia)

    if anom.empty:
        st.success(f"Nessun mese oltre il {soglia:.0%} su nessun paniere.", icon="✅")
    else:
        gravi = anom[~anom["verosimile"]]
        if not gravi.empty:
            st.error(
                f"**{len(gravi)} mesi oltre il 30%**: per un paniere diversificato "
                "sono implausibili. Vanno indagati prima di dare peso ai risultati.",
                icon="🚨",
            )
        st.dataframe(
            anom.assign(data=anom["data"].dt.strftime("%Y-%m-%d")),
            width="stretch", hide_index=True,
            column_config={
                "rendimento": st.column_config.NumberColumn("Rendimento", format="%.2f%%"),
                "verosimile": st.column_config.CheckboxColumn(
                    "Plausibile", help="Falso se oltre il 30%: quasi certamente un errore di dato"),
            },
        )

        st.markdown("**Attribuzione di un mese**")
        a1, a2 = st.columns(2)
        with a1:
            scelta_pf = st.selectbox("Paniere", sorted(anom["paniere"].unique()))
        with a2:
            mesi = anom[anom["paniere"] == scelta_pf]["data"].sort_values()
            scelta_mese = st.selectbox("Mese", list(mesi),
                                       format_func=lambda d: pd.Timestamp(d).strftime("%B %Y"))

        try:
            attr = analysis.attribution(res_attr.results[scelta_pf], pd.Timestamp(scelta_mese),
                                        top=12, names=ds.names, sectors=ds.sectors)
        except ValueError as exc:
            st.error(str(exc))
            attr = pd.DataFrame()

        if attr.empty:
            st.info("Nessuna posizione con P&L in questo periodo.")
        else:
            st.dataframe(
                attr, width="stretch", hide_index=True,
                column_config={
                    "pnl_usd": st.column_config.NumberColumn("P&L (USD)", format="%.0f"),
                    "quota_del_periodo": st.column_config.NumberColumn(
                        "Quota del mese", format="%.1f%%"),
                },
            )
            dominante = attr.iloc[0]
            if abs(dominante["quota_del_periodo"]) > 0.5:
                st.error(
                    f"**{dominante['ticker']}** da solo spiega il "
                    f"{dominante['quota_del_periodo']:.0%} del mese. "
                    "Con ogni probabilita' e' il dato sbagliato.", icon="🎯",
                )

            st.markdown(f"**Prezzi intorno alla data — {dominante['ticker']}**")
            ctx = analysis.price_context(ds.close_adj, ds.open_raw,
                                         str(dominante["ticker"]),
                                         pd.Timestamp(scelta_mese), window=8)
            if ctx.empty:
                st.info("Serie prezzi non disponibile per questo titolo.")
            else:
                st.dataframe(
                    ctx.assign(data=ctx.index.strftime("%Y-%m-%d")).reset_index(drop=True),
                    width="stretch", hide_index=True,
                    column_config={
                        "adjusted": st.column_config.NumberColumn(format="%.2f"),
                        "grezzo": st.column_config.NumberColumn(format="%.2f"),
                        "var_adjusted": st.column_config.NumberColumn("Δ adjusted", format="%.1f%%"),
                        "var_grezzo": st.column_config.NumberColumn("Δ grezzo", format="%.1f%%"),
                        "scarto": st.column_config.NumberColumn(
                            "Scarto", format="%.1f%%",
                            help="Divergenza fra i due rendimenti: se e' grande, "
                                 "il fattore di rettifica e' sbagliato"),
                    },
                )
                if ctx["scarto"].max() > 0.15:
                    st.warning(
                        "Prezzo grezzo e adjusted si muovono in modo molto diverso: "
                        "e' la firma di uno **split non gestito** nella serie rettificata.",
                        icon="⚠️",
                    )

didactics.render("anomaly_hunt", expanded=True)

st.divider()
with st.expander("Manifest completo del dataset"):
    st.json(man)
