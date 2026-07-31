"""
Motore di backtest a tranche sfalsate.

Scelte contabili, tutte deliberate:

* **Segnale a t, esecuzione a t+1.** F, V, fasce ed eleggibilita' sono lette
  all'ultima seduta del mese; gli scambi avvengono all'APERTURA della prima
  seduta del mese successivo. Calcolare ed eseguire sullo stesso prezzo di
  chiusura sarebbe anticipazione mascherata.

* **Tranche sfalsate.** Con holding di 3 mesi ribilanciamo 1/3 del portafoglio
  ogni mese. Costi identici al trimestrale puro, ma 12 decisioni l'anno invece
  di 4: conserviamo ~316 osservazioni mensili invece di ~105, e sparisce il
  timing luck di quale trimestre si sceglie come partenza.

* **Contabilita' in prezzi adjusted, dimensionamento in prezzi grezzi.** Il
  valore di una posizione evolve con il rendimento total-return; quante azioni
  compri dipende dal prezzo che paghi davvero. Mischiare i due e' l'errore
  classico che gonfia i titoli ad alto dividendo.

* **Ribilanciamento conservativo sui costi.** Vendiamo solo cio' che esce dal
  target, compriamo solo cio' che entra, non ripesiamo i sopravvissuti. Un
  ribilanciamento completo a equal weight costerebbe sensibilmente di piu'.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .config import TrackConfig

Selector = Callable[[int, "Panel", np.random.Generator], np.ndarray]


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------
def build_rebalance_calendar(
    trading_days: pd.DatetimeIndex,
    start: str | pd.Timestamp,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """(date di decisione, date di esecuzione).

    Decisione = ultima seduta del mese. Esecuzione = prima seduta successiva.
    """
    td = pd.DatetimeIndex(trading_days).sort_values()
    decision = td.to_series().resample("ME").last().dropna()
    decision = pd.DatetimeIndex(decision.to_numpy())

    pos = td.searchsorted(decision.to_numpy(), side="right")
    keep = pos < len(td)
    decision, exec_dates = decision[keep], td[pos[keep]]

    start = pd.Timestamp(start)
    keep2 = exec_dates >= start
    return decision[keep2], exec_dates[keep2]


# ---------------------------------------------------------------------------
# Pannello pronto per il motore
# ---------------------------------------------------------------------------
@dataclass
class Panel:
    """Tutto cio' che serve al motore, gia' allineato e in numpy."""

    decision_dates: pd.DatetimeIndex
    exec_dates: pd.DatetimeIndex
    tickers: list[str]

    open_adj: np.ndarray      # (T, N) prezzo adjusted di apertura, NaN se morto
    open_adj_ff: np.ndarray   # (T, N) stesso, forward-filled (per liquidazioni)
    open_raw: np.ndarray      # (T, N) prezzo grezzo di apertura (lotti + cap)
    alive: np.ndarray         # (T, N) bool: prezzo realmente disponibile
    dead_after: np.ndarray    # (N,) ultimo periodo con prezzo; oltre, il titolo e' morto

    force: np.ndarray         # (T, N) F alla data di DECISIONE
    velocity: np.ndarray      # (T, N) V alla data di DECISIONE
    bands: np.ndarray         # (T, N) fascia alla data di DECISIONE (1..n, NaN)
    eligible: np.ndarray      # (T, N) bool, alla data di DECISIONE

    rf_period: np.ndarray     # (T,) rendimento risk-free del periodo che finisce a t

    @property
    def n_periods(self) -> int:
        return len(self.exec_dates)


def prepare_panel(
    *,
    close_adj: pd.DataFrame,
    open_adj: pd.DataFrame,
    open_raw: pd.DataFrame,
    force: pd.DataFrame,
    velocity: pd.DataFrame,
    bands: pd.DataFrame,
    eligible: pd.DataFrame,
    rf_daily: pd.Series,
    cfg: TrackConfig,
) -> Panel:
    """Campiona i segnali alle date di decisione e i prezzi a quelle di esecuzione."""
    decision, exec_dates = build_rebalance_calendar(close_adj.index, cfg.backtest_start)
    tickers = list(close_adj.columns)

    def at(df: pd.DataFrame, when: pd.DatetimeIndex) -> np.ndarray:
        return df.reindex(index=when, columns=tickers).to_numpy(dtype="float64")

    oa = at(open_adj, exec_dates)
    alive = np.isfinite(oa)

    # forward-fill esplicito per poter valorizzare e liquidare chi sparisce
    oa_ff = pd.DataFrame(oa, index=exec_dates, columns=tickers)
    oa_ff = oa_ff.ffill().to_numpy(dtype="float64")

    # Ultimo periodo con un prezzo vero: oltre questo il titolo e' morto per
    # davvero. Distinguerlo da un buco temporaneo di dato evita di liquidare
    # una posizione sana solo perche' manca una quotazione.
    T = len(exec_dates)
    has_any = alive.any(axis=0)
    dead_after = np.where(has_any, (T - 1) - np.argmax(alive[::-1], axis=0), -1)

    rf = rf_daily.reindex(pd.DatetimeIndex(close_adj.index)).fillna(0.0)
    compounded = (1.0 + rf).cumprod()
    cp = compounded.reindex(exec_dates).to_numpy(dtype="float64")
    rf_period = np.zeros(len(exec_dates), dtype="float64")
    rf_period[1:] = cp[1:] / cp[:-1] - 1.0
    rf_period = np.nan_to_num(rf_period, nan=0.0, posinf=0.0, neginf=0.0)

    return Panel(
        decision_dates=decision,
        exec_dates=exec_dates,
        tickers=tickers,
        open_adj=oa,
        open_adj_ff=oa_ff,
        open_raw=at(open_raw, exec_dates),
        alive=alive,
        dead_after=dead_after,
        force=at(force, decision),
        velocity=at(velocity, decision),
        bands=at(bands, decision),
        eligible=at(eligible.astype("float64"), decision) > 0.5,
        rf_period=rf_period,
    )


# ---------------------------------------------------------------------------
# Selettori
# ---------------------------------------------------------------------------
def _extreme_within_band(
    panel: Panel, t: int, band: int, n: int, take_highest: bool
) -> np.ndarray:
    mask = panel.eligible[t] & (panel.bands[t] == float(band)) & np.isfinite(panel.force[t])
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx
    vals = panel.force[t, idx]
    order = np.argsort(-vals if take_highest else vals, kind="stable")
    return idx[order[:n]]


def selector_band(band: int, n: int, take_highest: bool) -> Selector:
    """I n titoli con F piu' estremo dentro una fascia."""
    def _sel(t: int, panel: Panel, rng: np.random.Generator) -> np.ndarray:
        return _extreme_within_band(panel, t, band, n, take_highest)
    return _sel


def selector_band_negative_velocity(band: int, n: int) -> Selector:
    """Leader che stanno ritracciando: fascia alta con Spinta negativa.

    E' la definizione piu' pura di 'comprare la debolezza momentanea di un
    titolo forte', e non coincide con il paniere della fascia piu' bassa.
    """
    def _sel(t: int, panel: Panel, rng: np.random.Generator) -> np.ndarray:
        mask = (
            panel.eligible[t]
            & (panel.bands[t] == float(band))
            & np.isfinite(panel.velocity[t])
            & (panel.velocity[t] < 0.0)
        )
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return idx
        order = np.argsort(panel.velocity[t, idx], kind="stable")  # piu' negativi primi
        return idx[order[:n]]
    return _sel


def selector_random(n: int) -> Selector:
    """Estrazione casuale dall'universo eleggibile: e' l'ipotesi nulla."""
    def _sel(t: int, panel: Panel, rng: np.random.Generator) -> np.ndarray:
        idx = np.flatnonzero(panel.eligible[t] & np.isfinite(panel.force[t]))
        if idx.size <= n:
            return idx
        return rng.choice(idx, size=n, replace=False)
    return _sel


def selector_all_eligible() -> Selector:
    def _sel(t: int, panel: Panel, rng: np.random.Generator) -> np.ndarray:
        return np.flatnonzero(panel.eligible[t] & np.isfinite(panel.force[t]))
    return _sel


# ---------------------------------------------------------------------------
# Risultato
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    name: str
    dates: pd.DatetimeIndex
    equity: pd.Series           # capitale composto
    returns: pd.Series          # rendimenti periodali (mensili)
    n_positions: pd.Series
    turnover: pd.Series
    costs: pd.Series            # dollari pagati nel periodo
    cash_weight: pd.Series
    contributions: pd.DataFrame | None = None
    """P&L in dollari per titolo e per periodo, se richiesto.

    Serve a risalire dal rendimento anomalo di un mese al titolo che l'ha
    causato: un +40% mensile su un paniere equipesato non e' un evento di
    mercato, e' quasi sempre un prezzo sbagliato, ma senza attribuzione non
    c'e' modo di sapere quale."""
    diagnostics: dict = field(default_factory=dict)

    @property
    def fixed_capital_pnl(self) -> pd.Series:
        """P&L cumulato a capitale costante: ogni mese pesa uguale.

        Derivato dalla stessa serie di rendimenti (capitale x somma cumulata),
        non da una simulazione separata. E' la vista corretta per confrontare
        periodi con livelli di capitale molto diversi.
        """
        base = self.diagnostics.get("capital", 100_000.0)
        return base * self.returns.cumsum()


# ---------------------------------------------------------------------------
# Motore
# ---------------------------------------------------------------------------
def run_strategy(
    panel: Panel,
    selector: Selector,
    cfg: TrackConfig,
    *,
    name: str = "strategia",
    frictionless: bool = False,
    delisting_haircut: float = 0.0,
    seed: int | None = None,
    track_contributions: bool = False,
) -> BacktestResult:
    """Esegue una strategia sul pannello.

    `delisting_haircut` e' il rendimento imposto a una posizione il cui titolo
    smette di avere prezzi (-1.0 = azzeramento). Serve allo stress test sul
    survivorship bias: se la conclusione regge a -1.0 e' robusta.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    T, N = panel.n_periods, len(panel.tickers)
    n_tr = cfg.n_tranches

    qty = np.zeros((n_tr, N), dtype="float64")   # quantita' "adjusted"
    cash = np.full(n_tr, cfg.capital / n_tr, dtype="float64")

    commission = 0.0 if frictionless else cfg.commission_per_side
    spread = 0.0 if frictionless else cfg.spread_bps / 1e4

    values = np.zeros(T)
    n_pos = np.zeros(T, dtype="int64")
    turn = np.zeros(T)
    cost_paid = np.zeros(T)
    cash_w = np.zeros(T)
    thin_months = 0
    forced_liquidations = 0

    # Attribuzione: opzionale perche' il bootstrap esegue il motore centinaia
    # di volte e non ha alcun bisogno di sapere chi ha guadagnato cosa.
    contrib = np.zeros((T, N), dtype="float64") if track_contributions else None
    prev_px: np.ndarray | None = None

    for t in range(T):
        px = panel.open_adj[t]
        px_ff = np.nan_to_num(panel.open_adj_ff[t], nan=0.0)
        raw = panel.open_raw[t]
        alive = panel.alive[t]
        is_dead = t > panel.dead_after  # morto per davvero, non buco temporaneo

        # --- 0. attribuzione del P&L maturato dal periodo precedente --------
        # Va fatta PRIMA di qualunque movimento: qui `qty` e' ancora quello
        # detenuto durante l'intervallo [t-1, t].
        if contrib is not None and prev_px is not None:
            contrib[t] = qty.sum(axis=0) * (px_ff - prev_px)

        # --- 1. il risk-free matura sulla liquidita' di tutte le tranche ----
        cash *= 1.0 + panel.rf_period[t]

        # --- 2. liquidazione forzata dei titoli spariti --------------------
        for i in range(n_tr):
            dead = (qty[i] > 0) & is_dead
            if dead.any():
                proceeds = qty[i, dead] * px_ff[dead] * (1.0 + delisting_haircut)
                cash[i] += float(proceeds.sum())
                forced_liquidations += int(dead.sum())
                qty[i, dead] = 0.0

        # --- 3. la tranche di turno ribilancia -----------------------------
        i = t % n_tr
        target = selector(t, panel, rng)
        target = target[np.isfinite(raw[target]) & np.isfinite(px[target]) & (raw[target] > 0)]

        if target.size and target.size < cfg.n_names:
            thin_months += 1

        held = np.flatnonzero(qty[i] > 0)
        target_set = set(target.tolist())
        to_sell = np.array([j for j in held if j not in target_set], dtype="int64")
        to_buy = np.array([j for j in target if qty[i, j] <= 0], dtype="int64")

        traded_notional = 0.0
        period_cost = 0.0

        # vendite
        for j in to_sell:
            gross = qty[i, j] * px_ff[j]
            fee = commission + spread * gross
            cash[i] += gross - fee
            qty[i, j] = 0.0
            traded_notional += gross
            period_cost += fee

        # acquisti: budget per nome, mai oltre lo slot teorico
        if to_buy.size:
            invested = float(np.nansum(qty[i] * px_ff))
            tranche_value = cash[i] + invested
            n_target = max(int(target.size), 1)
            slot = tranche_value / n_target
            budget = min(slot, cash[i] / to_buy.size) if cash[i] > 0 else 0.0

            for j in to_buy:
                if budget <= 0 or raw[j] <= 0 or raw[j] > cfg.max_share_price:
                    continue
                shares = np.floor(budget / raw[j])
                if shares < 1:
                    continue
                notional = float(shares * raw[j])
                fee = commission + spread * notional
                if notional + fee > cash[i]:
                    continue
                cash[i] -= notional + fee
                qty[i, j] = notional / px[j]
                traded_notional += notional
                period_cost += fee

        # --- 4. valorizzazione del portafoglio complessivo ------------------
        # Le posizioni si valutano all'ultimo prezzo noto: un buco temporaneo
        # di dato non deve far sparire il valore di una posizione viva.
        invested_all = float(np.nansum(qty * px_ff))
        total = invested_all + float(cash.sum())
        values[t] = total
        n_pos[t] = int((qty > 0).sum())
        cost_paid[t] = period_cost
        cash_w[t] = float(cash.sum()) / total if total > 0 else 1.0
        turn[t] = traded_notional / total if total > 0 else 0.0
        prev_px = px_ff

    equity = pd.Series(values, index=panel.exec_dates, name=name)

    # I primi n_tr periodi sono la rampa di ingresso: le tranche non sono
    # ancora tutte investite. Li scartiamo dalle statistiche.
    warmup = min(n_tr, T)
    equity_eff = equity.iloc[warmup - 1 :] if warmup > 0 else equity
    returns = equity_eff.pct_change().dropna()

    return BacktestResult(
        name=name,
        dates=panel.exec_dates,
        equity=equity,
        returns=returns,
        n_positions=pd.Series(n_pos, index=panel.exec_dates),
        turnover=pd.Series(turn, index=panel.exec_dates),
        costs=pd.Series(cost_paid, index=panel.exec_dates),
        cash_weight=pd.Series(cash_w, index=panel.exec_dates),
        contributions=(
            pd.DataFrame(contrib, index=panel.exec_dates, columns=panel.tickers)
            if contrib is not None else None
        ),
        diagnostics={
            "capital": cfg.capital,
            "warmup_periods": warmup,
            "mesi_paniere_incompleto": thin_months,
            "liquidazioni_forzate": forced_liquidations,
            "costi_totali": float(cost_paid.sum()),
            "delisting_haircut": delisting_haircut,
            "frictionless": frictionless,
        },
    )


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------
def newey_west_tstat(x: np.ndarray | pd.Series, lags: int = 3) -> float:
    """t-stat corretto per autocorrelazione.

    Con holding di 3 mesi i rendimenti mensili sono autocorrelati per
    costruzione: senza correzione il t-stat risulta gonfiato.
    """
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    n = a.size
    if n < 8:
        return float("nan")
    mu = a.mean()
    e = a - mu
    s = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float(e[lag:] @ e[:-lag]) / n
    if s <= 0:
        return float("nan")
    return float(mu / np.sqrt(s / n))


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """(profondita' massima, mesi trascorsi sott'acqua nel periodo peggiore)."""
    eq = equity.dropna()
    if eq.empty:
        return float("nan"), 0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    trough = dd.idxmin()
    depth = float(dd.min())

    peak_date = eq.loc[:trough].idxmax()
    after = eq.loc[trough:]
    recovered = after[after >= eq.loc[peak_date]]
    end = recovered.index[0] if len(recovered) else eq.index[-1]
    months = int(round((end - peak_date).days / 30.44))
    return depth, months


def performance_metrics(
    result: BacktestResult,
    rf_period: pd.Series | None = None,
    periods_per_year: int = 12,
) -> dict[str, float]:
    r = result.returns.dropna()
    if r.empty:
        return {}

    years = len(r) / periods_per_year
    total_growth = float((1.0 + r).prod())
    cagr = total_growth ** (1.0 / years) - 1.0 if years > 0 and total_growth > 0 else float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))

    if rf_period is not None:
        rf = rf_period.reindex(r.index).fillna(0.0)
        excess = r - rf
    else:
        excess = r
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)) if excess.std(ddof=1) > 0 else float("nan")

    downside = excess[excess < 0]
    sortino = (
        float(excess.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else float("nan")
    )

    depth, months = max_drawdown(result.equity)
    warm = result.diagnostics.get("warmup_periods", 0)

    # I costi in dollari crescono col patrimonio: su una curva composta il
    # totale assoluto e' illeggibile. Il numero interpretabile e' il drag
    # annuo in percentuale del patrimonio.
    eq = result.equity.replace(0.0, np.nan)
    cost_drag = float((result.costs / eq).iloc[warm:].mean() * periods_per_year)

    return {
        "CAGR": cagr,
        "Vol annua": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": depth,
        "Mesi recupero": float(months),
        "Hit rate": float((r > 0).mean()),
        "Mesi": float(len(r)),
        "Turnover medio": float(result.turnover.iloc[warm:].mean()),
        "Posizioni medie": float(result.n_positions.iloc[warm:].mean()),
        "Costo annuo %": cost_drag,
        "Costi totali $": float(result.costs.sum()),
        "Cash medio": float(result.cash_weight.iloc[warm:].mean()),
        "Valore finale $": float(result.equity.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Ipotesi nulla
# ---------------------------------------------------------------------------
def bootstrap_null(
    panel: Panel,
    cfg: TrackConfig,
    n_draws: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Estrazioni casuali di n_names titoli dall'universo eleggibile.

    Usa lo STESSO motore della strategia reale: stessi costi, stesso
    arrotondamento a lotti interi, stesso trattamento dei delistati. E'
    l'unico confronto che isola l'effetto della selezione per fascia da
    quello dell'equal weighting, del filtro e della composizione dell'universo.
    """
    n_draws = n_draws or cfg.n_bootstrap
    sel = selector_random(cfg.n_names)
    rf = pd.Series(panel.rf_period, index=panel.exec_dates)

    rows = []
    for b in range(n_draws):
        res = run_strategy(panel, sel, cfg, name=f"null_{b}", seed=cfg.seed + 1000 + b)
        m = performance_metrics(res, rf)
        rows.append({"draw": b, "CAGR": m.get("CAGR"), "Sharpe": m.get("Sharpe"),
                     "Max DD": m.get("Max DD"), "Vol annua": m.get("Vol annua")})
        if progress is not None:
            progress(b + 1, n_draws)
    return pd.DataFrame(rows)


def empirical_pvalue(null_values: pd.Series, observed: float) -> float:
    """Frazione di estrazioni casuali che hanno battuto il valore osservato."""
    v = pd.Series(null_values).dropna()
    if v.empty or not np.isfinite(observed):
        return float("nan")
    return float(((v >= observed).sum() + 1) / (len(v) + 1))


# ---------------------------------------------------------------------------
# Sottoperiodi
# ---------------------------------------------------------------------------
SUBPERIODS: tuple[tuple[str, str, str], ...] = (
    ("Scoppio dot-com", "2000-01-01", "2002-12-31"),
    ("Ripresa 2003-2007", "2003-01-01", "2007-10-31"),
    ("Crisi finanziaria", "2007-11-01", "2009-06-30"),
    ("Toro 2009-2019", "2009-07-01", "2019-12-31"),
    ("Covid 2020", "2020-01-01", "2020-12-31"),
    ("Tassi 2021-2022", "2021-01-01", "2022-12-31"),
    ("Recente", "2023-01-01", "2100-01-01"),
)


def subperiod_table(
    results: dict[str, BacktestResult],
    rf_period: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    for label, start, end in SUBPERIODS:
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        for name, res in results.items():
            r = res.returns.loc[(res.returns.index >= lo) & (res.returns.index <= hi)]
            if len(r) < 6:
                continue
            growth = float((1.0 + r).prod())
            years = len(r) / 12.0
            rows.append(
                {
                    "Periodo": label,
                    "Paniere": name,
                    "Rend. totale": growth - 1.0,
                    "CAGR": growth ** (1.0 / years) - 1.0 if growth > 0 else float("nan"),
                    "Vol": float(r.std(ddof=1) * np.sqrt(12)),
                    "Mesi": len(r),
                }
            )
    return pd.DataFrame(rows)


def drop_best_months(returns: pd.Series, k: int = 5) -> dict[str, float]:
    """Quanto del risultato dipende dai k mesi migliori.

    Se togliendone cinque il vantaggio sparisce, non e' un edge ripetibile:
    e' un premio per il rischio di coda concentrato in pochi eventi.
    """
    r = returns.dropna().sort_values()
    if len(r) <= k:
        return {}
    full = float((1.0 + r).prod()) - 1.0
    without = float((1.0 + r.iloc[:-k]).prod()) - 1.0
    return {"rendimento totale": full, f"senza i {k} mesi migliori": without,
            "quota dai top mesi": (full - without)}
