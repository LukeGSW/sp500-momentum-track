"""
Analitiche descrittive sulla pista: transizioni, permanenza, snapshot, scie.

NB: la matrice di transizione guarda deliberatamente in avanti (dove sara' il
titolo fra un mese). E' statistica descrittiva sul passato, non un segnale
operativo: nessuna di queste funzioni alimenta il backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
def transition_matrix(
    bands: pd.DataFrame,
    n_bands: int,
    band_names: tuple[str, ...],
    horizon_days: int = 21,
) -> pd.DataFrame:
    """Probabilita' di passare da una fascia all'altra in `horizon_days`.

    Le righe sommano a 1. La diagonale misura la persistenza: alta = il
    momentum e' appiccicoso, bassa = le classifiche ruotano.
    """
    a = bands.to_numpy(dtype="float64").ravel()
    b = bands.shift(-horizon_days).to_numpy(dtype="float64").ravel()
    m = np.isfinite(a) & np.isfinite(b)

    counts = np.zeros((n_bands, n_bands), dtype="float64")
    if m.any():
        np.add.at(counts, (a[m].astype(int) - 1, b[m].astype(int) - 1), 1.0)

    totals = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)

    labels = list(band_names[:n_bands])
    return pd.DataFrame(probs, index=labels, columns=labels)


def dwell_time(
    bands: pd.DataFrame,
    band_names: tuple[str, ...],
    monthly: bool = True,
) -> pd.DataFrame:
    """Durata media e mediana delle permanenze consecutive in ciascuna fascia.

    Campionata a fine mese (`monthly=True`) e non giorno per giorno. Su dati
    giornalieri il quintile di un titolo sul confine oscilla continuamente e
    la mediana delle permanenze crolla a 2-3 sedute: un numero vero ma
    inutile, perche' non e' la frequenza a cui si decide. La permanenza va
    confrontata con l'holding period, che e' mensile.
    """
    if monthly:
        bands = bands.resample("ME").last()
    unit = 1.0 if monthly else 21.0

    runs: dict[int, list[int]] = {i + 1: [] for i in range(len(band_names))}

    for col in bands.columns:
        s = bands[col].to_numpy(dtype="float64")
        ok = np.isfinite(s)
        if not ok.any():
            continue
        v = s[ok]
        # run-length encoding
        change = np.flatnonzero(np.diff(v) != 0) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(v)]))
        for st, en in zip(starts, ends, strict=True):
            band = int(v[st])
            if band in runs:
                runs[band].append(en - st)

    rows = []
    for i, label in enumerate(band_names, start=1):
        arr = np.array(runs.get(i, []), dtype="float64")
        if arr.size == 0:
            rows.append({"Fascia": label, "Permanenza media (mesi)": np.nan,
                         "Permanenza mediana (mesi)": np.nan,
                         "Permanenza massima (mesi)": np.nan, "Episodi": 0})
            continue
        rows.append({
            "Fascia": label,
            "Permanenza media (mesi)": float(arr.mean() * unit),
            "Permanenza mediana (mesi)": float(np.median(arr) * unit),
            "Permanenza massima (mesi)": float(arr.max() * unit),
            "Episodi": int(arr.size),
        })
    return pd.DataFrame(rows)


def days_in_current_band(bands: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Da quante sedute consecutive ogni titolo si trova nella fascia attuale."""
    upto = bands.loc[:as_of]
    if upto.empty:
        return pd.Series(dtype="float64")

    current = upto.iloc[-1]
    out = pd.Series(0.0, index=bands.columns, dtype="float64")
    arr = upto.to_numpy(dtype="float64")

    for j, col in enumerate(bands.columns):
        cur = current.iloc[j]
        if not np.isfinite(cur):
            out.iloc[j] = np.nan
            continue
        k = 0
        for i in range(arr.shape[0] - 1, -1, -1):
            if arr[i, j] == cur:
                k += 1
            elif np.isfinite(arr[i, j]):
                break
        out.iloc[j] = float(k)
    return out


# ---------------------------------------------------------------------------
def build_snapshot(
    as_of: pd.Timestamp,
    *,
    force: pd.DataFrame,
    velocity: pd.DataFrame,
    bands: pd.DataFrame,
    eligible: pd.DataFrame,
    sectors: pd.Series,
    names: pd.Series | None = None,
    band_names: tuple[str, ...],
    prices_raw: pd.DataFrame | None = None,
    max_share_price: float = 1500.0,
    lookback_prev_days: int = 21,
    portfolios: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Fotografia della pista a una data: una riga per titolo eleggibile."""
    if as_of not in force.index:
        pos = force.index.searchsorted(as_of, side="right") - 1
        if pos < 0:
            return pd.DataFrame()
        as_of = force.index[pos]

    elig_row = eligible.loc[as_of].astype(bool)
    tickers = elig_row[elig_row].index

    prev_pos = max(force.index.searchsorted(as_of) - lookback_prev_days, 0)
    prev_date = force.index[prev_pos]

    df = pd.DataFrame({"ticker": list(tickers)})
    df["name"] = df["ticker"].map(names) if names is not None else df["ticker"]
    df["sector"] = df["ticker"].map(sectors)
    df["F"] = df["ticker"].map(force.loc[as_of])
    df["V"] = df["ticker"].map(velocity.loc[as_of])
    df["band"] = df["ticker"].map(bands.loc[as_of])
    df["band_prev"] = df["ticker"].map(bands.loc[prev_date])

    def _lab(x):
        if x is None or not np.isfinite(x):
            return None
        i = int(x) - 1
        return band_names[i] if 0 <= i < len(band_names) else None

    df["band_label"] = df["band"].map(_lab)
    df["band_prev_label"] = df["band_prev"].map(_lab)
    df["movimento"] = np.where(
        df["band"].notna() & df["band_prev"].notna(),
        np.select(
            [df["band"] > df["band_prev"], df["band"] < df["band_prev"]],
            ["sale", "scende"], default="fermo",
        ),
        None,
    )

    dib = days_in_current_band(bands.loc[:as_of], as_of)
    df["giorni_in_fascia"] = df["ticker"].map(dib)

    if prices_raw is not None and as_of in prices_raw.index:
        px = prices_raw.loc[as_of]
        df["prezzo"] = df["ticker"].map(px)
        df["tradable"] = df["prezzo"].le(max_share_price).fillna(False)
    else:
        df["prezzo"] = np.nan
        df["tradable"] = True

    df["in_portfolio"] = False
    df["paniere"] = None
    for label, members in (portfolios or {}).items():
        mask = df["ticker"].isin(members)
        df.loc[mask, "in_portfolio"] = True
        df.loc[mask, "paniere"] = label

    df["sector"] = df["sector"].fillna("Non classificato")
    return df.sort_values("F", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Caccia alle anomalie di dato
# ---------------------------------------------------------------------------
def anomalous_periods(
    results: dict, threshold: float = 0.20
) -> pd.DataFrame:
    """Periodi con un rendimento implausibile per un paniere diversificato.

    Un paniere equipesato di 30 titoli che fa +40% in un mese non ha vissuto
    un evento di mercato: quasi sempre e' uno split non gestito o un prezzo
    sbagliato in una singola posizione. Sopra il 30% l'ipotesi 'evento reale'
    e' da scartare quasi sempre; tra il 15% e il 30% va guardata caso per caso
    (marzo 2020 e novembre 2008 esistono davvero).
    """
    rows = []
    for name, res in results.items():
        r = res.returns.dropna()
        for dt, val in r[r.abs() > threshold].items():
            rows.append({
                "data": dt,
                "paniere": name,
                "rendimento": float(val),
                "verosimile": bool(abs(val) <= 0.30),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("rendimento", key=lambda s: s.abs(), ascending=False)


def attribution(
    result,
    period: pd.Timestamp,
    top: int = 12,
    names: pd.Series | None = None,
    sectors: pd.Series | None = None,
) -> pd.DataFrame:
    """Chi ha prodotto il P&L di quel periodo, in dollari e in quota.

    Richiede che il backtest sia stato eseguito con `track_contributions=True`.
    """
    if result.contributions is None:
        raise ValueError(
            "Attribuzione non disponibile: eseguire run_strategy con "
            "track_contributions=True"
        )
    if period not in result.contributions.index:
        pos = result.contributions.index.searchsorted(period)
        if pos >= len(result.contributions.index):
            raise KeyError(f"periodo {period} fuori dal campione")
        period = result.contributions.index[pos]

    row = result.contributions.loc[period]
    row = row[row != 0.0]
    if row.empty:
        return pd.DataFrame()

    totale = float(row.sum())
    df = pd.DataFrame({"ticker": row.index, "pnl_usd": row.to_numpy()})
    df["quota_del_periodo"] = df["pnl_usd"] / totale if totale != 0 else np.nan
    if names is not None:
        df["nome"] = df["ticker"].map(names)
    if sectors is not None:
        df["settore"] = df["ticker"].map(sectors)

    df = df.reindex(df["pnl_usd"].abs().sort_values(ascending=False).index)
    return df.head(top).reset_index(drop=True)


def price_context(
    prices_adj: pd.DataFrame,
    prices_raw: pd.DataFrame,
    ticker: str,
    around: pd.Timestamp,
    window: int = 8,
) -> pd.DataFrame:
    """Prezzi grezzi e adjusted intorno a una data, con i rendimenti.

    Serve a distinguere un evento reale da uno split non gestito: se il prezzo
    grezzo dimezza e l'adjusted no (o viceversa), il fattore di rettifica e'
    sbagliato.
    """
    if ticker not in prices_adj.columns:
        return pd.DataFrame()

    idx = prices_adj.index
    pos = idx.searchsorted(pd.Timestamp(around))
    lo, hi = max(pos - window, 0), min(pos + window + 1, len(idx))
    win = idx[lo:hi]

    out = pd.DataFrame({
        "adjusted": prices_adj.loc[win, ticker],
        "grezzo": prices_raw.loc[win, ticker] if ticker in prices_raw.columns else np.nan,
    })
    out["var_adjusted"] = out["adjusted"].pct_change()
    out["var_grezzo"] = out["grezzo"].pct_change()
    # se i due rendimenti divergono, il fattore di rettifica non torna
    out["scarto"] = (out["var_adjusted"] - out["var_grezzo"]).abs()
    return out


def build_trails(
    tickers: list[str],
    as_of: pd.Timestamp,
    *,
    force: pd.DataFrame,
    velocity: pd.DataFrame | None = None,
    weeks: int = 13,
    step_days: int = 3,
) -> pd.DataFrame:
    """Formato lungo (date, ticker, F, V) per disegnare le scie.

    Sottocampionato ogni `step_days` sedute: una scia di 13 settimane ha ~65
    punti, che moltiplicati per 300 titoli renderebbero il grafico illeggibile
    e lento senza alcun guadagno informativo.
    """
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "F", "V"])

    end = force.index.searchsorted(as_of, side="right")
    start = max(end - int(weeks * 5), 0)
    window = force.index[start:end][::max(step_days, 1)]

    cols = [t for t in tickers if t in force.columns]
    if not cols:
        return pd.DataFrame(columns=["date", "ticker", "F", "V"])

    f = force.loc[window, cols].reset_index(names="date").melt(
        id_vars="date", var_name="ticker", value_name="F"
    )
    if velocity is not None:
        v = velocity.loc[window, cols].reset_index(names="date").melt(
            id_vars="date", var_name="ticker", value_name="V"
        )
        f = f.merge(v, on=["date", "ticker"], how="left")
    else:
        f["V"] = np.nan
    return f.dropna(subset=["F"])
