"""
Configurazione centrale de "La Pista".

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
# Nomenclatura delle fasce (dal basso verso l'alto della pista)
# ---------------------------------------------------------------------------
# NB: nomenclatura interamente nostra. Non usiamo "Relative Rotation Graph",
# "RRG", "JdK RS-Ratio" ne' "JdK RS-Momentum": sono marchi di RRG Research BV.
BAND_NAMES: tuple[str, ...] = (
    "Fondo Griglia",  # banda 1 - forza relativa piu' bassa
    "Rimonta",        # banda 2
    "Gruppo",         # banda 3 - centro, portafoglio di controllo
    "Scia",           # banda 4
    "Testa Corsa",    # banda 5 - forza relativa piu' alta
)

BAND_COLORS: dict[str, str] = {
    "Fondo Griglia": "#c1444b",
    "Rimonta": "#c98a3d",
    "Gruppo": "#5b6570",
    "Scia": "#3d8fb0",
    "Testa Corsa": "#3f9e6a",
}

# 11 settori GICS = le 11 corsie della pista.
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

# Finestra su cui misuriamo la Spinta V (pendenza della scia).
VELOCITY_WINDOW_BY_HOLDING: dict[int, int] = {1: 21, 3: 63, 6: 126}

# Lunghezza della scia disegnata sul grafico, in settimane.
TRAIL_WEEKS_BY_HOLDING: dict[int, int] = {1: 4, 3: 13, 6: 26}


@dataclass(frozen=True)
class TrackConfig:
    """Parametri completi di uno studio. Immutabile: l'hash identifica il run."""

    # --- segnale -----------------------------------------------------------
    holding_months: int = 3
    """Mesi di permanenza in portafoglio. Determina orizzonti e finestra di V."""

    n_bands: int = 5
    """Numero di fasce (quintili) sulla pista."""

    sector_neutral: bool = False
    """Se True le fasce sono calcolate DENTRO ciascun settore GICS."""

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
    non hanno mai splittato, quindi il paniere Testa Corsa. Il conteggio degli
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
    download_start: str = "1997-01-01"
    """I dati partono prima del backtest per riempire il warm-up dei lookback."""

    backtest_start: str = "2000-01-01"

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
        return tuple(f"Fascia {i + 1}" for i in range(self.n_bands))

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
