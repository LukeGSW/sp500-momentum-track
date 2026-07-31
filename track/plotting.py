"""
Grafici. Tema scuro, Plotly, nessuna dipendenza da Streamlit.

Il grafico centrale e' `plot_track()`: la pista. Asse verticale = Forza F,
asse orizzontale = corsia settoriale. Il momentum NON e' un secondo asse: e'
la pendenza della scia, cioe' il movimento verticale del titolo nel tempo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import BAND_COLORS, GICS_SECTORS, SECTOR_UNKNOWN

BG = "#11151c"
GRID = "#2a313d"
FG = "#d6dce6"
MUTED = "#7d8899"

POS = "#3f9e6a"   # spinta positiva
NEG = "#c1444b"   # spinta negativa

LANE_HALF_WIDTH = 0.38


def _base_layout(fig: go.Figure, height: int = 700, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FG, size=12),
        title=dict(text=title, font=dict(size=17)) if title else None,
        margin=dict(l=60, r=90, t=60 if title else 30, b=60),
        height=height,
        hoverlabel=dict(bgcolor="#1b2029", font_size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# ===========================================================================
# LA PISTA
# ===========================================================================
def _lane_positions(snapshot: pd.DataFrame, sectors_order: list[str]) -> pd.Series:
    """Coordinata x stabile: corsia del settore + scostamento deterministico.

    Lo scostamento dipende dall'ordine alfabetico del ticker dentro il settore,
    non da F: cosi' resta costante nel tempo e le scie risultano verticali,
    che e' esattamente la metafora della pista dritta.
    """
    x = pd.Series(np.nan, index=snapshot.index, dtype="float64")
    lane_of = {s: i for i, s in enumerate(sectors_order)}

    for sector, grp in snapshot.groupby("sector", sort=False):
        lane = lane_of.get(sector)
        if lane is None:
            continue
        order = grp.sort_values("ticker").index
        k = len(order)
        offsets = np.zeros(1) if k == 1 else np.linspace(-LANE_HALF_WIDTH, LANE_HALF_WIDTH, k)
        x.loc[order] = lane + offsets
    return x


def _band_edges(force_values: np.ndarray, n_bands: int) -> np.ndarray:
    """Confini delle fasce alla data corrente (quantili effettivi)."""
    v = force_values[np.isfinite(force_values)]
    if v.size < n_bands:
        return np.array([])
    qs = np.linspace(0, 1, n_bands + 1)
    return np.nanquantile(v, qs)


def plot_track(
    snapshot: pd.DataFrame,
    trails: pd.DataFrame | None = None,
    *,
    band_names: tuple[str, ...],
    n_bands: int = 5,
    x_mode: str = "sector",
    highlight: set[str] | None = None,
    portfolio_label: str = "In portafoglio",
    title: str = "",
    height: int = 760,
) -> go.Figure:
    """Disegna la pista.

    snapshot: colonne ticker, name, sector, F, V, band, in_portfolio, tradable
    trails:   formato lungo con colonne date, ticker, F  (solo i ticker scelti)
    x_mode:   'sector' -> 11 corsie GICS ; 'velocity' -> x = Spinta V
    """
    snap = snapshot.dropna(subset=["F"]).copy()
    highlight = highlight or set()
    fig = go.Figure()

    if snap.empty:
        _base_layout(fig, height, title or "La Pista")
        fig.add_annotation(text="Nessun titolo eleggibile a questa data",
                           showarrow=False, font=dict(size=16, color=MUTED))
        return fig

    sectors_order = [s for s in GICS_SECTORS if (snap["sector"] == s).any()]
    if (snap["sector"] == SECTOR_UNKNOWN).any():
        sectors_order.append(SECTOR_UNKNOWN)

    if x_mode == "velocity":
        snap["x"] = snap["V"].fillna(0.0)
        x_title = "Spinta V (z-score)"
    else:
        snap["x"] = _lane_positions(snap, sectors_order)
        x_title = ""

    # --- fasce di sfondo ---------------------------------------------------
    edges = _band_edges(snap["F"].to_numpy(dtype="float64"), n_bands)
    y_lo = float(np.nanmin(snap["F"])) - 0.35
    y_hi = float(np.nanmax(snap["F"])) + 0.35
    if edges.size:
        for i in range(n_bands):
            lo = y_lo if i == 0 else float(edges[i])
            hi = y_hi if i == n_bands - 1 else float(edges[i + 1])
            label = band_names[i] if i < len(band_names) else f"Fascia {i+1}"
            fig.add_hrect(
                y0=lo, y1=hi, layer="below", line_width=0,
                fillcolor=BAND_COLORS.get(label, "#333a45"), opacity=0.12,
                annotation_text=label, annotation_position="right",
                annotation_font=dict(size=11, color=MUTED),
            )
            if i < n_bands - 1:
                fig.add_hline(y=float(edges[i + 1]), line=dict(color=GRID, width=1, dash="dot"))

    # --- separatori di corsia ---------------------------------------------
    if x_mode == "sector":
        for i in range(1, len(sectors_order)):
            fig.add_vline(x=i - 0.5, line=dict(color=GRID, width=1))

    # --- scie --------------------------------------------------------------
    if trails is not None and not trails.empty:
        xs: list[float] = []
        ys: list[float] = []
        xmap = snap.set_index("ticker")["x"].to_dict()
        for tk, grp in trails.sort_values("date").groupby("ticker", sort=False):
            if tk not in xmap:
                continue
            g = grp.dropna(subset=["F"])
            if len(g) < 2:
                continue
            if x_mode == "velocity" and "V" in g.columns:
                xs.extend(g["V"].fillna(0.0).tolist())
            else:
                xs.extend([xmap[tk]] * len(g))
            ys.extend(g["F"].tolist())
            xs.append(None)
            ys.append(None)
        if xs:
            fig.add_trace(go.Scattergl(
                x=xs, y=ys, mode="lines", name="Scia",
                line=dict(color="rgba(150,165,185,0.55)", width=1.4),
                hoverinfo="skip", showlegend=False,
            ))

    # --- titoli ------------------------------------------------------------
    v = snap["V"].fillna(0.0).to_numpy(dtype="float64")
    sizes = 7.0 + 9.0 * np.clip(np.abs(v) / 2.0, 0, 1)
    in_pf = snap.get("in_portfolio", pd.Series(False, index=snap.index)).fillna(False)

    fig.add_trace(go.Scattergl(
        x=snap["x"], y=snap["F"], mode="markers", name="Titoli",
        marker=dict(
            size=sizes,
            color=v, colorscale=[[0, NEG], [0.5, "#5b6570"], [1, POS]],
            cmid=0, cmin=-2.5, cmax=2.5,
            line=dict(width=0.6, color="rgba(255,255,255,0.35)"),
            colorbar=dict(title=dict(text="Spinta V", side="right"), thickness=12, len=0.45, y=0.22),
        ),
        customdata=np.column_stack([
            snap["ticker"].to_numpy(),
            snap.get("name", snap["ticker"]).to_numpy(),
            snap["sector"].to_numpy(),
            snap["F"].round(2).to_numpy(),
            snap["V"].round(2).to_numpy(),
            snap.get("band_label", pd.Series("", index=snap.index)).to_numpy(),
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
            "Settore: %{customdata[2]}<br>"
            "Forza F: %{customdata[3]}<br>"
            "Spinta V: %{customdata[4]}<br>"
            "Fascia: %{customdata[5]}<extra></extra>"
        ),
        showlegend=False,
    ))

    # titoli in portafoglio: anello bianco
    if in_pf.any():
        sel = snap[in_pf.astype(bool)]
        fig.add_trace(go.Scattergl(
            x=sel["x"], y=sel["F"], mode="markers", name=f"{portfolio_label} ({len(sel)})",
            marker=dict(size=16, color="rgba(0,0,0,0)",
                        line=dict(width=1.8, color="#f0f4fa")),
            hoverinfo="skip",
        ))

    # titoli evidenziati dall'utente: etichetta
    if highlight:
        sel = snap[snap["ticker"].isin(highlight)]
        if not sel.empty:
            fig.add_trace(go.Scattergl(
                x=sel["x"], y=sel["F"], mode="markers+text", name="Selezionati",
                text=sel["ticker"], textposition="top center",
                textfont=dict(size=11, color="#ffd479"),
                marker=dict(size=13, color="rgba(0,0,0,0)",
                            line=dict(width=2, color="#ffd479")),
                hoverinfo="skip",
            ))

    _base_layout(fig, height, title or "La Pista")
    fig.add_hline(y=0.0, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_yaxes(title="Forza relativa F  (z-score trasversale)", range=[y_lo, y_hi])

    if x_mode == "sector":
        fig.update_xaxes(
            title=x_title,
            tickmode="array",
            tickvals=list(range(len(sectors_order))),
            ticktext=[s.replace(" ", "<br>") for s in sectors_order],
            range=[-0.6, len(sectors_order) - 0.4],
            showgrid=False,
        )
    else:
        fig.update_xaxes(title=x_title, zeroline=True, zerolinecolor=MUTED)
    return fig


# ===========================================================================
# Distribuzioni e matrici
# ===========================================================================
def plot_band_sector_matrix(
    snapshot: pd.DataFrame, band_names: tuple[str, ...]
) -> go.Figure:
    tab = (
        snapshot.dropna(subset=["band_label"])
        .pivot_table(index="band_label", columns="sector", values="ticker", aggfunc="count")
        .reindex(index=list(band_names)[::-1])
        .fillna(0)
    )
    fig = go.Figure(go.Heatmap(
        z=tab.to_numpy(), x=list(tab.columns), y=list(tab.index),
        colorscale="Teal", showscale=False,
        text=tab.to_numpy().astype(int), texttemplate="%{text}",
        hovertemplate="%{y} — %{x}<br>%{z} titoli<extra></extra>",
    ))
    _base_layout(fig, 420, "Composizione settoriale per fascia")
    fig.update_xaxes(tickangle=-35)
    return fig


def plot_transition_matrix(matrix: pd.DataFrame, title: str = "") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy() * 100.0,
        x=list(matrix.columns), y=list(matrix.index),
        colorscale="Blues", zmin=0, zmax=float(np.nanmax(matrix.to_numpy()) * 100.0),
        text=(matrix.to_numpy() * 100.0).round(1), texttemplate="%{text}%",
        hovertemplate="da %{y}<br>a %{x}<br>%{z:.1f}%<extra></extra>",
        colorbar=dict(title="%", thickness=12),
    ))
    _base_layout(fig, 480, title or "Probabilita' di transizione a un mese")
    fig.update_xaxes(title="fascia dopo un mese")
    fig.update_yaxes(title="fascia di partenza", autorange="reversed")
    return fig


# ===========================================================================
# Backtest
# ===========================================================================
PALETTE = {
    "Testa Corsa": "#3f9e6a",
    "Fondo Griglia": "#c1444b",
    "Testa Corsa in ritracciamento": "#e0a13c",
    "Gruppo": "#8b95a5",
    "Universo eleggibile (senza costi)": "#4aa3c7",
    "Spread Testa - Fondo": "#a97bd6",
}
_FALLBACK_COLORS = ["#3f9e6a", "#c1444b", "#e0a13c", "#8b95a5", "#4aa3c7", "#a97bd6"]


def _color_for(name: str, i: int) -> str:
    return PALETTE.get(name, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])


def plot_equity(
    series: dict[str, pd.Series],
    *,
    log_scale: bool = True,
    title: str = "Capitale (100.000$ per paniere)",
    ylabel: str = "Valore del conto ($)",
) -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna()
        if s.empty:
            continue
        dashed = "dot" if "senza costi" in name or "Universo" in name else "solid"
        fig.add_trace(go.Scatter(
            x=s.index, y=s.to_numpy(), name=name, mode="lines",
            line=dict(color=_color_for(name, i), width=2.0, dash=dashed),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b %Y}}<br>%{{y:,.0f}}$<extra></extra>",
        ))
    _base_layout(fig, 560, title)
    fig.update_yaxes(title=ylabel, type="log" if log_scale else "linear")
    return fig


def plot_fixed_capital(series: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.to_numpy(), name=name, mode="lines",
            line=dict(color=_color_for(name, i), width=2.0),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b %Y}}<br>%{{y:,.0f}}$<extra></extra>",
        ))
    _base_layout(fig, 470, "P&L cumulato a capitale fisso (ogni mese pesa uguale)")
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_yaxes(title="P&L cumulato ($)")
    return fig


def plot_drawdown(series: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna()
        if s.empty:
            continue
        dd = (s / s.cummax() - 1.0) * 100.0
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.to_numpy(), name=name, mode="lines",
            line=dict(color=_color_for(name, i), width=1.6),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>",
        ))
    _base_layout(fig, 400, "Drawdown: distanza dal massimo precedente")
    fig.update_yaxes(title="%", ticksuffix="%")
    return fig


def plot_null_distribution(
    null_values: pd.Series, observed: dict[str, float], metric: str = "CAGR"
) -> go.Figure:
    v = pd.Series(null_values).dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=v.to_numpy() * (100.0 if metric == "CAGR" else 1.0),
        nbinsx=60, marker=dict(color="#3a4655", line=dict(width=0)),
        name="estrazioni casuali", hovertemplate="%{x:.2f}<br>%{y} estrazioni<extra></extra>",
    ))
    for i, (name, val) in enumerate(observed.items()):
        if val is None or not np.isfinite(val):
            continue
        x = val * (100.0 if metric == "CAGR" else 1.0)
        fig.add_vline(x=x, line=dict(color=_color_for(name, i), width=2.4),
                      annotation_text=name, annotation_position="top",
                      annotation_font=dict(size=11, color=_color_for(name, i)))
    _base_layout(fig, 420, f"Ipotesi nulla: 30 titoli estratti a caso — {metric}")
    fig.update_xaxes(title=f"{metric}" + (" (%)" if metric == "CAGR" else ""))
    fig.update_yaxes(title="numero di estrazioni")
    return fig


def plot_monthly_heatmap(returns: pd.Series, title: str = "") -> go.Figure:
    r = returns.dropna() * 100.0
    df = pd.DataFrame({"anno": r.index.year, "mese": r.index.month, "r": r.to_numpy()})
    piv = df.pivot_table(index="anno", columns="mese", values="r", aggfunc="mean")
    piv = piv.reindex(columns=range(1, 13))
    lim = float(np.nanpercentile(np.abs(piv.to_numpy()), 97)) if piv.notna().any().any() else 1.0

    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=["G", "F", "M", "A", "M", "G", "L", "A", "S", "O", "N", "D"],
        y=[str(a) for a in piv.index],
        colorscale=[[0, NEG], [0.5, "#1b2029"], [1, POS]], zmid=0, zmin=-lim, zmax=lim,
        hovertemplate="%{y} — mese %{x}<br>%{z:.2f}%<extra></extra>",
        colorbar=dict(title="%", thickness=12),
    ))
    _base_layout(fig, max(320, 22 * len(piv) + 120), title or "Rendimenti mensili")
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_cost_sensitivity(sens: pd.DataFrame, current_bps: float) -> go.Figure:
    """Livelli dei due panieri E la loro differenza al variare del costo.

    Servono entrambi: i livelli mostrano quanto il costo distrugge in assoluto,
    la differenza mostra quanto influisce sulla scelta tra i due panieri. La
    seconda e' quasi sempre molto piu' piatta della prima.
    """
    fig = go.Figure()
    for i, col in enumerate([c for c in sens.columns if c not in ("bps", "Differenza")]):
        fig.add_trace(go.Scatter(
            x=sens["bps"], y=sens[col] * 100.0, mode="lines+markers", name=col,
            line=dict(color=_color_for(col, i), width=2.2),
            hovertemplate=f"<b>{col}</b><br>costo %{{x:.0f}} bps<br>CAGR %{{y:.2f}}%<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=sens["bps"], y=sens["Differenza"] * 100.0, mode="lines+markers",
        name="Differenza (Testa - Fondo)", line=dict(color="#a97bd6", width=2.6, dash="dash"),
        hovertemplate="costo %{x:.0f} bps<br>differenza %{y:.2f}%/anno<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1.4, dash="dot"))
    fig.add_vline(x=current_bps, line=dict(color="#ffd479", width=2),
                  annotation_text=f"costo assunto ({current_bps:.0f} bps)",
                  annotation_position="top", annotation_font=dict(color="#ffd479", size=11))
    _base_layout(fig, 430, "Sensibilita' ai costi di transazione")
    fig.update_xaxes(title="costo per rotazione completa (bps, andata e ritorno)")
    fig.update_yaxes(title="CAGR annuo (%)", ticksuffix="%")
    return fig


# ===========================================================================
# Diagnostica dati
# ===========================================================================
def plot_coverage(cov: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cov.index, y=cov["coverage"] * 100.0, mode="lines",
        line=dict(color="#4aa3c7", width=2), name="coverage",
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra></extra>",
    ))
    for lvl, color, label in ((98, POS, "98% — affidabile"), (90, NEG, "90% — inutilizzabile")):
        fig.add_hline(y=lvl, line=dict(color=color, width=1, dash="dot"),
                      annotation_text=label, annotation_position="right",
                      annotation_font=dict(size=10, color=color))
    _base_layout(fig, 380, "Copertura prezzi dei costituenti storici")
    fig.update_yaxes(title="% costituenti con serie prezzi", range=[80, 101], ticksuffix="%")
    return fig


def plot_universe_size(size: pd.Series, min_eligible: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=size.index, y=size.to_numpy(), mode="lines", fill="tozeroy",
        line=dict(color="#4aa3c7", width=1.6), fillcolor="rgba(74,163,199,0.18)",
        name="titoli eleggibili",
        hovertemplate="%{x|%b %Y}<br>%{y} titoli<extra></extra>",
    ))
    fig.add_hline(y=min_eligible, line=dict(color=NEG, width=1.2, dash="dot"),
                  annotation_text=f"soglia minima ({min_eligible})",
                  annotation_position="right", annotation_font=dict(size=10, color=NEG))
    _base_layout(fig, 360, "Universo eleggibile nel tempo")
    fig.update_yaxes(title="numero di titoli")
    return fig


def plot_series(series: dict[str, pd.Series], title: str, ylabel: str,
                percent: bool = False, height: int = 360) -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna()
        if s.empty:
            continue
        y = s.to_numpy() * (100.0 if percent else 1.0)
        fig.add_trace(go.Scatter(x=s.index, y=y, mode="lines", name=name,
                                 line=dict(color=_color_for(name, i), width=1.8)))
    _base_layout(fig, height, title)
    fig.update_yaxes(title=ylabel, ticksuffix="%" if percent else "")
    return fig
