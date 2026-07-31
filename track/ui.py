"""
Livello condiviso dell'interfaccia: configurazione, cache, banner, formattazione.

Le funzioni con cache usano `st.cache_resource` e non `st.cache_data` perche'
i pannelli sono grandi (fino a ~150 MB per configurazione) e cache_data li
serializzerebbe a ogni accesso. cache_resource restituisce lo stesso oggetto:
nessuna delle funzioni a valle deve mutarlo.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from . import backtest as bt
from . import didactics, storage, study
from .config import PREREGISTERED, TrackConfig

APP_TITLE = "La Pista"
SUBTITLE = "Conviene comprare azioni in forte momentum, o in debolezza momentanea?"


# ---------------------------------------------------------------------------
def page_config(page: str) -> None:
    title = APP_TITLE if page == APP_TITLE else f"{APP_TITLE} — {page}"
    st.set_page_config(
        page_title=title,
        page_icon="🏁",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carico il dataset…", max_entries=2)
def get_dataset(data_dir: str | None = None) -> study.Dataset:
    return study.load_dataset(data_dir)


@st.cache_resource(show_spinner="Calcolo Forza, Spinta e fasce…", max_entries=4)
def get_signals(cfg: TrackConfig, data_dir: str | None = None) -> study.Signals:
    return study.compute_signals(get_dataset(data_dir), cfg)


@st.cache_resource(show_spinner="Eseguo il backtest…", max_entries=4)
def get_study(cfg: TrackConfig, data_dir: str | None = None,
              haircut: float = 0.0) -> study.StudyResult:
    return study.run_study(get_dataset(data_dir), cfg,
                           signals=get_signals(cfg, data_dir),
                           delisting_haircut=haircut)


@st.cache_resource(show_spinner="Estrazioni casuali (ipotesi nulla)…", max_entries=3)
def get_bootstrap(cfg: TrackConfig, n_draws: int, data_dir: str | None = None) -> pd.DataFrame:
    res = get_study(cfg, data_dir)
    return bt.bootstrap_null(res.panel, cfg, n_draws=n_draws)


@st.cache_resource(show_spinner="Griglia di sensibilita' ai costi…", max_entries=3)
def get_cost_sensitivity(cfg: TrackConfig, data_dir: str | None = None) -> pd.DataFrame:
    return study.cost_sensitivity(get_study(cfg, data_dir).panel, cfg)


@st.cache_resource(show_spinner="Attribuzione del P&L per titolo…", max_entries=2)
def get_study_with_attribution(cfg: TrackConfig, data_dir: str | None = None):
    """Studio con l'attribuzione attiva: piu' pesante, serve solo alla diagnostica."""
    return study.run_study(get_dataset(data_dir), cfg,
                           signals=get_signals(cfg, data_dir),
                           track_contributions=True)


@st.cache_resource(show_spinner="Confronto con e senza filtro…", max_entries=2)
def get_filter_decomposition(cfg: TrackConfig, data_dir: str | None = None) -> pd.DataFrame:
    return study.filter_decomposition(get_dataset(data_dir), cfg)


@st.cache_resource(show_spinner="Stress test sul survivorship bias…", max_entries=2)
def get_delisting_stress(cfg: TrackConfig, data_dir: str | None = None) -> pd.DataFrame:
    ds = get_dataset(data_dir)
    sig = get_signals(cfg, data_dir)
    out: dict[str, dict[str, float]] = {}
    for h in cfg.delisting_haircuts:
        r = study.run_study(ds, cfg, signals=sig, delisting_haircut=h)
        out[f"{h:+.0%}"] = {
            name: bt.performance_metrics(res, r.rf_period).get("CAGR", float("nan"))
            for name, res in r.results.items()
        }
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
_SECRETS_ERROR: str | None = None


def _secret(name: str) -> str | None:
    """Legge un valore dai secrets Streamlit, dall'ambiente o dal file locale.

    Su Streamlit Cloud vince `st.secrets` (la piattaforma li inietta). In
    locale funziona anche lanciando l'app da un'altra cartella, perche'
    `storage.read_secret` guarda pure nel secrets.toml del progetto.

    Un eventuale errore di `st.secrets` (tipicamente un secrets.toml malformato)
    viene memorizzato invece che ingoiato: senza, l'utente vedrebbe solo
    "dataset non trovato" senza alcun indizio sulla causa.
    """
    global _SECRETS_ERROR
    try:
        val = st.secrets.get(name)  # type: ignore[union-attr]
        if val:
            return str(val).strip()
    except Exception as exc:  # noqa: BLE001 - assente in locale: caso normale
        _SECRETS_ERROR = f"{type(exc).__name__}: {exc}"
    return storage.read_secret(name)


def _mask(value: str | None) -> str:
    if not value:
        return "— non impostato —"
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:4]}…{value[-4:]}  ({len(value)} caratteri)"


def secrets_report() -> dict:
    """Cosa vede davvero l'app. Non espone mai il valore dei segreti veri."""
    report: dict = {"chiavi_in_st_secrets": None, "errore_st_secrets": None, "valori": {}}

    try:
        report["chiavi_in_st_secrets"] = sorted(st.secrets.keys())  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        report["errore_st_secrets"] = f"{type(exc).__name__}: {exc}"

    for name in ("DATA_REPO", "DATA_URL"):
        report["valori"][name] = _secret(name) or "— non impostato —"
    for name in ("DATA_TOKEN", "GITHUB_TOKEN", "EODHD_API_KEY"):
        report["valori"][name] = _mask(_secret(name))

    if _SECRETS_ERROR and not report["errore_st_secrets"]:
        report["errore_st_secrets"] = _SECRETS_ERROR
    return report


@st.cache_resource(show_spinner="Scarico il dataset pubblicato…", max_entries=1)
def _fetch_published_dataset(url: str | None, repo: str | None, token: str | None) -> bool:
    """Un solo tentativo per container: il risultato resta in cache."""
    return storage.ensure_dataset(url=url, repo=repo, token=token)


def require_dataset() -> bool:
    """Assicura che i dati ci siano, altrimenti blocca la pagina con istruzioni.

    Sul deploy i pannelli non sono nel repository (sono grandi e vengono
    rigenerati): la pipeline li pubblica come asset di una Release e l'app li
    recupera da li' al primo avvio.
    """
    if storage.dataset_available():
        return True

    url = _secret("DATA_URL")
    repo = _secret("DATA_REPO")
    token = _secret("DATA_TOKEN") or _secret("GITHUB_TOKEN")

    if url or repo:
        try:
            if _fetch_published_dataset(url, repo, token):
                st.success("Dataset scaricato dalla Release.", icon="✅")
                return True
            st.error("Archivio scaricato ma incompleto: mancano ancora dei pannelli.")
        except Exception as exc:  # noqa: BLE001 - va mostrato all'utente, non nei log
            st.error(f"**Download del dataset fallito.**\n\n`{type(exc).__name__}: {exc}`")
            _fetch_published_dataset.clear()

    st.error("**Dataset non trovato.** L'app legge solo artefatti gia' calcolati, "
             "non li costruisce mai da sola.")
    st.markdown(f"Mancano: `{'`, `'.join(storage.missing_panels())}`")

    tab_deploy, tab_locale = st.tabs(["Sei sul deploy", "Sei in locale"])

    with tab_deploy:
        st.markdown(
            "La pipeline pubblica il dataset come asset di una **Release** su GitHub, "
            "ma l'app non sa dove cercarlo. Indicaglielo nei *secrets* "
            "**dell'applicazione Streamlit** (share.streamlit.io → la tua app → "
            "**Settings → Secrets**):"
        )
        st.code('DATA_REPO = "tuo-utente/tuo-repository"', language="toml")
        st.warning(
            "**Non e' lo stesso posto dove hai messo `EODHD_API_KEY`.** Quella va nei "
            "secret del *repository GitHub* (Settings → Secrets and variables → "
            "Actions) e serve al workflow. `DATA_REPO` va nei secret "
            "dell'*applicazione Streamlit* e serve all'app. Sono due pannelli diversi.",
            icon="⚠️",
        )
        st.caption(
            "In alternativa puoi indicare l'URL diretto di un archivio con "
            "`DATA_URL = \"https://…/la-pista-data.tar.gz\"`. Se il repository e' "
            "privato aggiungi anche `DATA_TOKEN = \"ghp_…\"` con permesso di lettura."
        )

        # --- prova immediata, senza aspettare un redeploy -------------------
        st.divider()
        st.markdown("**Prova subito, senza toccare i secrets**")
        st.caption("Serve a capire se il problema e' la configurazione o l'archivio. "
                   "Il dataset scaricato cosi' resta finche' il container e' vivo.")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        with c1:
            manual = st.text_input(
                "Repository (owner/repo) oppure URL completo dell'archivio",
                placeholder="tuo-utente/tuo-repository",
                label_visibility="visible",
            )
        with c2:
            go = st.button("Scarica", width="stretch", type="primary")

        if go and manual.strip():
            value = manual.strip()
            is_url = value.startswith(("http://", "https://"))
            target = value if is_url else storage.release_asset_url(value)
            st.caption(f"Scarico da: `{target}`")
            try:
                with st.spinner("Download in corso…"):
                    ok = storage.ensure_dataset(
                        url=target,
                        token=_secret("DATA_TOKEN") or _secret("GITHUB_TOKEN"),
                    )
                if ok:
                    _fetch_published_dataset.clear()
                    st.success("Fatto. Ricarica la pagina.", icon="✅")
                    st.rerun()
                else:
                    st.error(f"Archivio scaricato ma incompleto. "
                             f"Mancano: `{'`, `'.join(storage.missing_panels())}`")
            except Exception as exc:  # noqa: BLE001 - va mostrato, non loggato
                st.error(f"**{type(exc).__name__}**: {exc}")

        # --- cosa vede l'app -----------------------------------------------
        st.divider()
        with st.expander("Diagnostica: cosa vede l'app", expanded=True):
            rep = secrets_report()
            if rep["errore_st_secrets"]:
                st.error(
                    "Streamlit non riesce a leggere i secrets:\n\n"
                    f"`{rep['errore_st_secrets']}`\n\n"
                    "Se il messaggio parla di sintassi, controlla che ogni riga sia "
                    "nella forma `CHIAVE = \"valore\"`, virgolette comprese."
                )
            keys = rep["chiavi_in_st_secrets"]
            if keys is None:
                st.caption("Nessun secret configurato (normale in locale).")
            elif not keys:
                st.warning("Streamlit legge i secrets ma **sono vuoti**: la sezione "
                           "Settings → Secrets dell'app non contiene nulla.", icon="⚠️")
            else:
                st.caption(f"Chiavi viste dall'app: `{'`, `'.join(keys)}`")

            st.dataframe(
                pd.DataFrame(
                    [{"secret": k, "valore risolto": v} for k, v in rep["valori"].items()]
                ),
                width="stretch", hide_index=True,
            )
            st.caption("I token sono mascherati di proposito. `DATA_REPO` e `DATA_URL` "
                       "non sono segreti, quindi si vedono per intero: se qui sono "
                       "'non impostato', l'app non li sta ricevendo.")

        st.info(
            "Verifica anche che il workflow **Aggiorna dataset** sia arrivato in fondo "
            "e abbia creato una **Release**. Se lo step finale e' fallito, l'archivio "
            "esiste solo fra gli *artifacts* del run, che non sono scaricabili "
            "dall'esterno: in quel caso scaricalo a mano dal run e ripubblicalo, "
            "oppure rilancia il workflow.",
            icon="ℹ️",
        )

    with tab_locale:
        st.markdown("Genera i dati dalla cartella del progetto:")
        st.code("python -m pipeline.make_demo_data", language="bash")
        st.caption("↑ dati **sintetici**, nessuna chiave API: serve solo a vedere "
                   "come funziona l'interfaccia. I numeri non sono reali.")
        st.code("python -m pipeline.build_dataset", language="bash")
        st.caption("↑ dati **reali**. Richiede la chiave in `.streamlit/secrets.toml` "
                   "o nella variabile d'ambiente `EODHD_API_KEY`. "
                   "Circa 1.200 chiamate API, alcuni minuti.")
        st.markdown("Oppure scarica quello gia' pubblicato dalla pipeline:")
        st.code("python -m pipeline.fetch_dataset --repo tuo-utente/tuo-repository",
                language="bash")

    return False


def demo_banner(ds: study.Dataset) -> None:
    if ds.is_demo:
        st.warning(
            "**Dati sintetici.** Questo dataset e' generato casualmente per provare "
            "l'interfaccia: ogni numero che vedi e' finto. Esegui "
            "`python -m pipeline.build_dataset` per i dati reali EODHD.",
            icon="⚠️",
        )


def header(page_name: str, ds: study.Dataset | None = None) -> None:
    label = APP_TITLE if page_name == APP_TITLE else f"{APP_TITLE} — {page_name}"
    st.title(f"🏁 {label}")
    st.caption(SUBTITLE)
    if ds is not None:
        demo_banner(ds)


# ---------------------------------------------------------------------------
def sidebar_config() -> TrackConfig:
    """Costruisce la configurazione e dichiara se e' quella preregistrata."""
    sb = st.sidebar
    sb.header("Configurazione")

    mode = sb.radio(
        "Impostazioni",
        ["Preregistrata", "Personalizzata"],
        index=0,
        help=(
            "La configurazione preregistrata e' stata dichiarata PRIMA di guardare "
            "i risultati. E' la difesa contro l'overfitting: ogni altra "
            "combinazione va letta come esplorazione, non come conclusione."
        ),
    )

    if mode == "Preregistrata":
        cfg = PREREGISTERED
        sb.success("Configurazione preregistrata", icon="🔒")
        _config_recap(sb, cfg)
        return cfg

    sb.warning(
        "Stai esplorando. Con molte combinazioni provate, un p-value sotto 0,05 "
        "e' il risultato atteso dal puro caso.",
        icon="🧪",
    )

    with sb.expander("Segnale", expanded=True):
        holding = st.select_slider(
            "Holding period (mesi)", options=[1, 3, 6], value=PREREGISTERED.holding_months,
            help="Determina anche gli orizzonti di calcolo: il lookback deve essere "
                 "congruo all'holding, altrimenti il segnale muore prima che la "
                 "posizione si chiuda.",
        )
        sector_neutral = st.toggle(
            "Fasce calcolate dentro ciascun settore", value=PREREGISTERED.sector_neutral,
            help="Neutralizza la rotazione settoriale: il paniere non diventa una "
                 "scommessa su un solo settore.",
        )

    with sb.expander("Universo", expanded=True):
        sma_filter = st.toggle(
            "Filtro: prezzo sopra la media a 200 sedute", value=PREREGISTERED.sma_filter,
            help="ATTENZIONE: il filtro non e' neutrale, e' gia' una scommessa sul "
                 "momentum. Con il filtro la domanda diventa 'leader o ritracciamento "
                 "dentro un trend'. Spegnilo e confronta.",
        )
        max_price = st.number_input(
            "Cap sul prezzo per azione ($)", 100.0, 10000.0,
            float(PREREGISTERED.max_share_price), 100.0,
            help="Sopra questa soglia il titolo non entra in uno slot con lotti interi.",
        )

    with sb.expander("Portafoglio", expanded=True):
        capital = st.number_input("Capitale per paniere (USD)", 10_000.0, 10_000_000.0,
                                  float(PREREGISTERED.capital), 10_000.0)
        n_names = st.slider("Titoli per paniere", 10, 60, PREREGISTERED.n_names, 5)
        st.caption(f"Slot per titolo: **{capital / n_names:,.0f} USD**")
        commission = st.number_input("Commissione per lato ($)", 0.0, 30.0,
                                     float(PREREGISTERED.commission_per_side), 0.5)
        spread = st.number_input("Spread per lato (bps)", 0.0, 100.0,
                                 float(PREREGISTERED.spread_bps), 0.5)

    with sb.expander("Periodo e inferenza"):
        start = st.text_input("Inizio backtest (AAAA-MM-GG)", PREREGISTERED.backtest_start)
        n_boot = st.select_slider("Estrazioni per l'ipotesi nulla",
                                  options=[100, 250, 500, 1000, 2000],
                                  value=PREREGISTERED.n_bootstrap)

    cfg = replace(
        PREREGISTERED,
        holding_months=int(holding),
        sector_neutral=bool(sector_neutral),
        sma_filter=bool(sma_filter),
        max_share_price=float(max_price),
        capital=float(capital),
        n_names=int(n_names),
        commission_per_side=float(commission),
        spread_bps=float(spread),
        backtest_start=str(start),
        n_bootstrap=int(n_boot),
    )
    _config_recap(sb, cfg)
    return cfg


def _config_recap(sb, cfg: TrackConfig) -> None:
    sb.divider()
    sb.caption(didactics.escape_markdown(
        f"**Orizzonti F**: {', '.join(str(h) for h in cfg.horizons)} sedute  \n"
        f"**Finestra V**: {cfg.velocity_window} sedute  \n"
        f"**Tranche sfalsate**: {cfg.n_tranches}  \n"
        f"**Slot**: {cfg.slot_value:,.0f}$ per titolo  \n"
        f"**Costo per rotazione**: {study.current_cost_bps(cfg):.1f} bps  \n"
        f"`config_hash = {cfg.hash()}`"
    ))


# ---------------------------------------------------------------------------
PERCENT_METRICS = {"CAGR", "Vol annua", "Max DD", "Hit rate", "Turnover medio",
                   "Cash medio", "Costo annuo %"}


def format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Formatta la tabella metriche per la visualizzazione."""
    out = metrics.copy()
    for col in out.columns:
        if col in PERCENT_METRICS:
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}")
        elif col in ("Costi totali $", "Valore finale $"):
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}")
        elif col in ("Mesi", "Mesi recupero", "Posizioni medie"):
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}")
        else:
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
    return out


_NEUTRAL = (91, 101, 112)
_GOOD = (63, 158, 106)
_BAD = (193, 68, 75)


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def heat_table(df: pd.DataFrame, fmt: str = "{:.1%}", center: float = 0.0,
               na_rep: str = "—"):
    """Tabella con celle colorate, senza matplotlib.

    `Styler.background_gradient` di pandas richiede matplotlib: ~30 MB di
    dipendenza in piu' sul deploy solo per colorare delle celle. Qui la scala
    divergente e' calcolata a mano e applicata come CSS inline.
    """
    import numpy as np

    vals = df.to_numpy(dtype="float64", na_value=np.nan)
    finite = vals[np.isfinite(vals)]
    span = float(np.abs(finite - center).max()) if finite.size else 1.0
    span = max(span, 1e-9)

    def _style(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(x):
            return ""
        t = float(np.clip((x - center) / span, -1.0, 1.0))
        r, g, b = _lerp(_NEUTRAL, _GOOD, t) if t >= 0 else _lerp(_NEUTRAL, _BAD, -t)
        return f"background-color: rgba({r},{g},{b},0.55)"

    return df.style.map(_style).format(fmt, na_rep=na_rep)


def chart(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
