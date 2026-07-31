"""
Il segnale: Forza F, Spinta V, fasce.

Modulo PURO: nessun I/O, nessuna chiamata di rete, nessuna dipendenza da
Streamlit. Tutto qui dentro e' testabile e deve superare il test di
non-anticipazione (`tests/test_features.py`).

Convenzioni sui pannelli
------------------------
Tutti i DataFrame hanno:  righe = date (DatetimeIndex crescente)
                          colonne = ticker
Un valore NaN significa "dato non disponibile", mai "zero".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "robust_zscore",
    "log_return_horizon",
    "compute_force",
    "rolling_ols_slope",
    "compute_velocity",
    "above_sma",
    "sufficient_history",
    "build_eligibility",
    "assign_bands",
    "band_label",
]


# ---------------------------------------------------------------------------
# Standardizzazione trasversale robusta
# ---------------------------------------------------------------------------
def robust_zscore(
    df: pd.DataFrame,
    winsor: float = 3.0,
    min_count: int = 20,
) -> pd.DataFrame:
    """z-score TRASVERSALE (riga per riga) con mediana e MAD.

    Mediana e MAD invece di media e deviazione standard perche' su 500 titoli
    bastano due outlier (una biotech che raddoppia) per spostare la media e
    schiacciare tutti gli altri verso lo zero.

    Proprieta' che sfruttiamo altrove: essendo la standardizzazione fatta
    lungo le colonne, qualunque costante additiva comune a tutti i titoli
    della stessa data si cancella. E' per questo che il rendimento del
    benchmark sparisce da F e la scelta SPY/RSP diventa irrilevante.
    """
    med = df.median(axis=1, skipna=True)
    dev = df.sub(med, axis=0)
    mad = dev.abs().median(axis=1, skipna=True) * 1.4826

    # righe con troppi pochi titoli o dispersione nulla -> non standardizzabili
    valid = (df.count(axis=1) >= min_count) & (mad > 0)
    mad = mad.where(valid)

    z = dev.div(mad, axis=0)
    z = z.clip(lower=-winsor, upper=winsor)
    return z.where(valid, other=np.nan)


# ---------------------------------------------------------------------------
# Forza F
# ---------------------------------------------------------------------------
def log_return_horizon(prices: pd.DataFrame, h: int) -> pd.DataFrame:
    """Log-rendimento su h sedute. Usa SEMPRE prezzi adjusted."""
    logp = np.log(prices.where(prices > 0))
    return logp - logp.shift(h)


def compute_force(
    prices_adj: pd.DataFrame,
    horizons: tuple[int, ...],
    winsor: float = 3.0,
    min_count: int = 20,
) -> pd.DataFrame:
    """Forza relativa F: la coordinata verticale sulla pista.

    Media degli z-score trasversali dei log-rendimenti sugli orizzonti dati,
    ri-standardizzata. Un titolo entra nel calcolo solo se ha TUTTI gli
    orizzonti disponibili: la copertura parziale produrrebbe punteggi non
    confrontabili tra titoli.

    NB: F e' calcolata sull'universo *misurabile* (tutti i titoli dell'indice
    con storia sufficiente), non sull'universo *eleggibile*. Il filtro sulla
    media mobile agisce dopo, in fase di selezione. Cosi' F non cambia quando
    accendi o spegni il filtro, e il confronto 2x2 resta interpretabile.
    """
    if not horizons:
        raise ValueError("serve almeno un orizzonte")

    z_sum: pd.DataFrame | None = None
    z_cnt: pd.DataFrame | None = None
    for h in horizons:
        z = robust_zscore(log_return_horizon(prices_adj, h), winsor, min_count)
        filled = z.fillna(0.0)
        present = z.notna().astype("float64")
        z_sum = filled if z_sum is None else z_sum + filled
        z_cnt = present if z_cnt is None else z_cnt + present

    assert z_sum is not None and z_cnt is not None
    complete = z_cnt == float(len(horizons))
    f_raw = (z_sum / z_cnt).where(complete)
    return robust_zscore(f_raw, winsor, min_count)


# ---------------------------------------------------------------------------
# Spinta V
# ---------------------------------------------------------------------------
def rolling_ols_slope(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Pendenza della regressione lineare su finestra mobile.

    Implementata come somma pesata con pesi fissi: con ascisse equispaziate
    la pendenza OLS e' una combinazione lineare dei valori nella finestra,
    quindi non serve risolvere nulla. Molto piu' veloce di rolling().apply()
    su un pannello con centinaia di colonne.

        slope_i = sum_j w_j * x_{i-W+1+j},   w_j = (j - jbar) / sum_k (k-jbar)^2
    """
    if window < 3:
        raise ValueError("finestra troppo corta per una pendenza")

    j = np.arange(window, dtype="float64")
    w = (j - j.mean()) / ((j - j.mean()) ** 2).sum()

    arr = df.to_numpy(dtype="float64", copy=False)
    n = arr.shape[0]
    if n < window:
        return pd.DataFrame(np.nan, index=df.index, columns=df.columns)

    out = np.full(arr.shape, np.nan, dtype="float64")
    acc = np.zeros((n - window + 1, arr.shape[1]), dtype="float64")
    for k in range(window):
        # NaN nella finestra propagano naturalmente -> pendenza NaN
        acc += w[k] * arr[k : n - window + 1 + k]
    out[window - 1 :] = acc
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def compute_velocity(
    force: pd.DataFrame,
    window: int,
    winsor: float = 3.0,
    min_count: int = 20,
) -> pd.DataFrame:
    """Spinta V: la pendenza della scia, ri-standardizzata trasversalmente."""
    return robust_zscore(rolling_ols_slope(force, window), winsor, min_count)


# ---------------------------------------------------------------------------
# Eleggibilita'
# ---------------------------------------------------------------------------
def above_sma(prices_adj: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """True dove il prezzo e' sopra la propria media mobile semplice.

    Calcolata su prezzi adjusted per coerenza con il resto della contabilita'.
    ATTENZIONE: questo filtro non e' neutrale, e' gia' una scommessa sul
    momentum. Va acceso e spento e i due casi vanno confrontati.
    """
    sma = prices_adj.rolling(window, min_periods=window).mean()
    return (prices_adj > sma).where(sma.notna() & prices_adj.notna())


def sufficient_history(prices_adj: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """True dove il titolo ha almeno `min_days` osservazioni valide alle spalle.

    Esclude meccanicamente le IPO e gli spin-off recenti. E' un bias noto e
    dichiarato: senza storia non c'e' segnale.
    """
    valid = prices_adj.notna().astype("int64")
    return valid.rolling(min_days, min_periods=min_days).sum() >= min_days


def build_eligibility(
    prices_adj: pd.DataFrame,
    prices_raw: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    min_history_days: int,
    max_share_price: float,
    sma_filter: bool,
    sma_window: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Maschera booleana dei titoli selezionabili, piu' i singoli criteri.

    I criteri vengono restituiti separatamente perche' la pagina Diagnostica
    deve poter mostrare quanti titoli cadono per ciascun motivo: e' l'unico
    modo per accorgersi che il filtro sulla media mobile sta svuotando
    l'universo proprio nei mesi che decidono il risultato.
    """
    cols = prices_adj.columns
    idx = prices_adj.index

    memb = membership.reindex(index=idx, columns=cols).fillna(False).astype(bool)
    has_price = prices_adj.notna()
    hist = sufficient_history(prices_adj, min_history_days).reindex(index=idx, columns=cols).fillna(False)
    under_cap = (prices_raw.reindex(index=idx, columns=cols) <= max_share_price).fillna(False)

    criteria = {
        "in_indice": memb,
        "prezzo_disponibile": has_price,
        "storia_sufficiente": hist.astype(bool),
        "sotto_cap_prezzo": under_cap.astype(bool),
    }

    eligible = memb & has_price & criteria["storia_sufficiente"] & criteria["sotto_cap_prezzo"]

    if sma_filter:
        sma_ok = above_sma(prices_adj, sma_window).reindex(index=idx, columns=cols).fillna(False).astype(bool)
        criteria["sopra_media_mobile"] = sma_ok
        eligible = eligible & sma_ok

    return eligible, criteria


# ---------------------------------------------------------------------------
# Fasce
# ---------------------------------------------------------------------------
def assign_bands(
    force: pd.DataFrame,
    eligible: pd.DataFrame,
    n_bands: int = 5,
    sectors: pd.Series | None = None,
    sector_neutral: bool = False,
) -> pd.DataFrame:
    """Assegna ogni titolo eleggibile a una fascia (1 = piu' debole).

    Quantili e non soglie assolute in z: le soglie fisse producono panieri
    vuoti o squilibrati quando la distribuzione trasversale si sposta (2008,
    2020), e un backtest con panieri di dimensione variabile nel tempo non e'
    confrontabile con se' stesso.

    Con `sector_neutral` i quantili sono calcolati DENTRO ciascun settore
    GICS: serve a distinguere la selezione titoli dalla rotazione settoriale.
    """
    masked = force.where(eligible)

    def _rank_to_band(sub: pd.DataFrame) -> pd.DataFrame:
        pct = sub.rank(axis=1, pct=True, method="average")
        band = np.ceil(pct * n_bands)
        return band.clip(lower=1, upper=n_bands)

    if not sector_neutral or sectors is None:
        return _rank_to_band(masked)

    out = pd.DataFrame(np.nan, index=force.index, columns=force.columns)
    aligned = sectors.reindex(force.columns)
    for _, cols in aligned.groupby(aligned, dropna=True).groups.items():
        cols = list(cols)
        if not cols:
            continue
        out[cols] = _rank_to_band(masked[cols])
    return out


def band_label(band_number: float, band_names: tuple[str, ...]) -> str | None:
    """Da numero di fascia (1-based) a nome leggibile."""
    if band_number is None or not np.isfinite(band_number):
        return None
    i = int(band_number) - 1
    if 0 <= i < len(band_names):
        return band_names[i]
    return None
