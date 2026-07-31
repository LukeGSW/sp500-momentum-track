"""
Configurazione centrale del Monitor di Forza Relativa.

Tutti i parametri dello studio vivono qui. La configurazione PREREGISTRATA
(vedi `PREREGISTERED`) e' quella dichiarata prima di guardare i risultati:
serve come difesa contro l'overfitting. Ogni altra combinazione va letta come
esplorazione, non come conclusione.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# Nomenclatura delle fasce (quintili di RS Score, dal piu' debole al piu' forte)
# ---------------------------------------------------------------------------
# Notazione ibrida: la sigla del quintile per chi legge da quant, l'etichetta
# descrittiva per tutti gli altri. E' la convenzione della ricerca fattoriale.
#
# NB: "Relative Strength" e' vocabolario finanziario generico. Non usiamo
# "Relative Rotation Graph", "RRG", "JdK RS-Ratio" ne' "JdK RS-Momentum":
# quelli sono marchi di RRG Research BV.
BAND_NAMES: tuple[str, ...] = (
    "Q1 Laggard",            # RS Score piu' basso
    "Q2 Sottoperformanti",
    "Q3 In linea",           # centro, portafoglio di controllo
    "Q4 Sovraperformanti",
    "Q5 Leader",             # RS Score piu' alto
)

BAND_COLORS: dict[str, str] = {
    "Q1 Laggard": "#c1444b",
    "Q2 Sottoperformanti": "#c98a3d",
    "Q3 In linea": "#5b6570",
    "Q4 Sovraperformanti": "#3d8fb0",
    "Q5 Leader": "#3f9e6a",
}

# Nomi delle due metriche, usati ovunque nei testi e nelle etichette.
METRIC_LEVEL = "RS Score"
"""Forza relativa: z-score trasversale robusto di log-rendimenti multi-orizzonte."""

METRIC_SLOPE = "RS Slope"
"""Pendenza dell'RS Score: la velocita' con cui la forza relativa cambia."""

# 11 settori GICS = le 11 colonne settoriali della mappa.
GICS_SECTORS: tuple[str, ...] = (
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
)

SECTOR_UNKNOWN = "Non classificato"

# ---------------------------------------------------------------------------
# Congruenza lookback <-> holding period
# ---------------------------------------------------------------------------
# Principio: un segnale la cui informazione decade in 1 mese non ha senso su un
# holding di 3 mesi (la posizione resta aperta dopo che il segnale e' morto).
# Per ogni holding period definiamo gli orizzonti di calcolo congrui.
HORIZONS_BY_HOLDING: dict[int, tuple[int, ...]] = {
    1: (21, 63, 126),
    3: (63, 126, 252),
    6: (126, 252),
}

# Finestra su cui misuriamo l'RS Slope (pendenza dell'RS Score).
VELOCITY_WINDOW_BY_HOLDING: dict[int, int] = {1: 21, 3: 63, 6: 126}

# Lunghezza della traiettoria disegnata sul grafico, in settimane.
TRAIL_WEEKS_BY_HOLDING: dict[int, int] = {1: 4, 3: 13, 6: 26}


@dataclass(frozen=True)
class TrackConfig:
    """Parametri completi di uno studio. Immutabile: l'hash identifica il run."""

    # --- segnale -----------------------------------------------------------
    holding_months: int = 3
    """Mesi di permanenza in portafoglio. Determina orizzonti e finestra dello Slope."""

    n_bands: int = 5
    """Numero di quintili di RS Score."""

    sector_neutral: bool = False
    """Se True i quintili sono calcolati DENTRO ciascun settore GICS."""

    winsor: float = 3.0
    """Clipping degli z-score robusti, in deviazioni standard robuste."""

    # --- universo ----------------------------------------------------------
    sma_filter: bool = True
    """Filtro 'prezzo sopra la media mobile': esclude i titoli in depressione."""

    sma_window: int = 200

    min_history_days: int = 273
    """Barre minime richieste (252 per r_252 + margine). Esclude le IPO recenti."""

    min_eligible: int = 50
    """Sotto questa soglia l'universo eleggibile e' troppo sottile: si segnala."""

    max_share_price: float = 1500.0
    """Cap sul prezzo per azione: sopra questa soglia il titolo non e' tradabile
    con 100k divisi su n_names slot. NB: penalizza leggermente i compounder che
    non hanno mai splittato, quindi il paniere Q5 Leader. Il conteggio degli
    esclusi finisce nella diagnostica."""

    # --- portafoglio -------------------------------------------------------
    capital: float = 100_000.0
    n_names: int = 30
    """Titoli per paniere. 100k / 30 = 3.333$ per slot."""

    commission_per_side: float = 1.5
    """Dollari per eseguito, per lato. Costo di OGGI applicato a tutto lo storico."""

    spread_bps: float = 3.0
    """Meta' spread + impatto, in bps sul nozionale tradato, per lato."""

    # --- periodo -----------------------------------------------------------
    download_start: str = "1996-01-01"
    """I dati partono prima del backtest per riempire il warm-up dei lookback.

    La lista dei costituenti ricostruita parte dal 1996-01-02, quindi tanto
    vale chiedere i prezzi da subito: non costa chiamate API in piu'.
    """

    backtest_start: str = "2000-01-01"
    """Default prudente: l'EOD US di EODHD copre bene dal 2000 in poi.

    Se il coverage report mostra dati solidi anche prima, si puo' abbassare
    fino al 1998 e includere la SALITA della bolla dot-com, non solo lo
    scoppio. Serve pero' il warm-up: con orizzonti fino a 252 sedute piu' la
    finestra dell'RS Slope, il primo segnale valido arriva ~15 mesi dopo
    l'inizio dei prezzi.
    """

    # --- inferenza ---------------------------------------------------------
    n_bootstrap: int = 1000
    seed: int = 12345

    # --- stress test survivorship -----------------------------------------
    delisting_haircuts: tuple[float, ...] = (0.0, -0.30, -0.50, -1.00)
    """Rendimento imposto ai titoli che spariscono senza prezzo di uscita.
    0.0 = liquidazione all'ultimo prezzo noto (ipotesi ottimistica).
    -1.00 = azzeramento. Se la conclusione regge a -1.00 e' robusta."""

    # ---------------------------------------------------------------- utils
    @property
    def horizons(self) -> tuple[int, ...]:
        return HORIZONS_BY_HOLDING[self.holding_months]

    @property
    def velocity_window(self) -> int:
        return VELOCITY_WINDOW_BY_HOLDING[self.holding_months]

    @property
    def trail_weeks(self) -> int:
        return TRAIL_WEEKS_BY_HOLDING[self.holding_months]

    @property
    def n_tranches(self) -> int:
        """Tranche sfalsate: una per mese di holding.

        Con holding 3 mesi ribilanciamo 1/3 del portafoglio ogni mese. Costi
        identici al ribilanciamento trimestrale puro, ma conserviamo 12
        decisioni all'anno (~316 osservazioni mensili invece di ~105) e
        annulliamo il timing luck del trimestre di partenza.
        """
        return self.holding_months

    @property
    def slot_value(self) -> float:
        return self.capital / self.n_names

    @property
    def band_names(self) -> tuple[str, ...]:
        if self.n_bands == 5:
            return BAND_NAMES
        return tuple(f"Q{i + 1}" for i in range(self.n_bands))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["horizons"] = list(self.horizons)
        d["velocity_window"] = self.velocity_window
        d["n_tranches"] = self.n_tranches
        d["slot_value"] = self.slot_value
        return d

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


# La configurazione dichiarata PRIMA di guardare i risultati.
PREREGISTERED = TrackConfig()
