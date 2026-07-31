"""
Esclusione di serie prezzi compromesse.

Perche' esiste
--------------
Alcune serie contengono errori accertati: fusioni con concambio non gestito,
split assenti dal fattore di rettifica, prezzi palesemente sbagliati. Un solo
titolo puo' produrre un mese a +40% su un paniere equipesato e falsare la
conclusione dell'intero studio.

Il punto delicato e' **quando** si esclude. Togliere dati dopo aver visto i
risultati e' una scelta post-hoc: legittima se l'errore e' accertato, ma da
dichiarare. Per questo la lista vive in un file **versionato nel repository**
(`exclusions.csv`), viene applicata dalla PIPELINE prima di qualunque calcolo,
e finisce nel manifest e nei caveats dell'export. Cosi' l'esclusione e' parte
del metodo, riproducibile e ispezionabile, non un ritocco silenzioso.

Formato del file
----------------
    ticker,start_date,end_date,reason

`start_date` e `end_date` vuoti significano "tutta la serie". Preferire sempre
una finestra stretta intorno all'evento: scartare vent'anni di dati validi per
un errore di un mese introduce un bias peggiore di quello che corregge.

Cosa succede ai dati esclusi
----------------------------
Le quotazioni nella finestra diventano NaN. Il titolo risulta quindi privo di
prezzo, e la macchina esistente lo tratta come non eleggibile: non viene
comprato. Se era gia' in portafoglio, resta valorizzato all'ultimo prezzo noto
fino alla fine della finestra — rendimento zero invece di un rendimento falso.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = REPO_ROOT / "exclusions.csv"

COLUMNS = ("ticker", "start_date", "end_date", "reason")


@dataclass(frozen=True)
class Exclusion:
    ticker: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    reason: str

    @property
    def whole_series(self) -> bool:
        return self.start is None and self.end is None

    def describe(self) -> str:
        if self.whole_series:
            return f"{self.ticker}: intera serie — {self.reason}"
        a = self.start.date() if self.start is not None else "inizio"
        b = self.end.date() if self.end is not None else "fine"
        return f"{self.ticker}: {a} → {b} — {self.reason}"


# ---------------------------------------------------------------------------
def load_exclusions(path: str | Path | None = None) -> list[Exclusion]:
    """Legge la lista. Assenza del file non e' un errore: significa nessuna esclusione."""
    # `path is not None` e' indispensabile: str(None).upper() vale "NONE" e
    # senza la guardia il caso predefinito verrebbe scambiato per la
    # disattivazione esplicita, saltando in silenzio tutte le esclusioni.
    if path is not None and str(path).strip().upper() == "NONE":
        log.warning("ESCLUSIONI DISATTIVATE su richiesta esplicita (--exclusions NONE): "
                    "le serie con errori accertati resteranno nello studio.")
        return []

    p = Path(path) if path else DEFAULT_FILE
    if not p.exists():
        log.warning(
            "File di esclusioni NON TROVATO in %s. Il dataset verra' costruito con "
            "tutte le serie, comprese quelle con errori accertati: i risultati "
            "conterranno i mesi anomali. Se non e' voluto, verifica che "
            "exclusions.csv sia stato caricato nella radice del progetto.", p,
        )
        return []

    df = pd.read_csv(p, dtype="string", comment="#", skip_blank_lines=True)
    mancanti = [c for c in ("ticker", "reason") if c not in df.columns]
    if mancanti:
        raise ValueError(f"{p}: colonne mancanti {mancanti}. Attese: {', '.join(COLUMNS)}")

    out: list[Exclusion] = []
    for _, row in df.iterrows():
        tk = str(row["ticker"]).strip().upper()
        if not tk or tk.lower() == "nan":
            continue
        out.append(Exclusion(
            ticker=tk,
            start=pd.to_datetime(row.get("start_date"), errors="coerce")
            if pd.notna(row.get("start_date")) else None,
            end=pd.to_datetime(row.get("end_date"), errors="coerce")
            if pd.notna(row.get("end_date")) else None,
            reason=str(row.get("reason") or "non specificato").strip(),
        ))
    log.info("caricate %d esclusioni da %s", len(out), p)
    return out


def exclusions_manifest(path: str | Path | None = None) -> dict:
    """Cosa la pipeline ha REALMENTE letto: percorso, righe, ticker, impronta.

    Senza questo, un file non caricato e un file caricato ma vuoto producono
    lo stesso identico manifest, e non c'e' modo di distinguerli a posteriori.
    """
    import hashlib

    if path is not None and str(path).strip().upper() == "NONE":
        return {"file": "NONE", "esiste": False, "disattivate": True,
                "righe": 0, "ticker": [], "impronta": None}

    p = Path(path) if path else DEFAULT_FILE
    if not p.exists():
        return {"file": str(p), "esiste": False, "disattivate": False,
                "righe": 0, "ticker": [], "impronta": None}

    raw = p.read_bytes()
    lista = load_exclusions(p)
    return {
        "file": str(p),
        "esiste": True,
        "disattivate": False,
        "righe": len(lista),
        "ticker": sorted({e.ticker for e in lista}),
        "impronta": hashlib.sha256(raw).hexdigest()[:12],
    }


def apply_exclusions(
    panels: dict[str, pd.DataFrame],
    exclusions: list[Exclusion],
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Azzera (NaN) le quotazioni escluse in tutti i pannelli prezzi.

    Ritorna i pannelli modificati e un resoconto di cosa e' stato tolto, che
    va nel manifest: un'esclusione non registrata e' un dato manipolato.
    """
    if not exclusions:
        return panels, []

    out = {k: v.copy() for k, v in panels.items()}
    resoconto: list[dict] = []

    for ex in exclusions:
        presente = any(ex.ticker in df.columns for df in out.values())
        if not presente:
            log.warning("esclusione ignorata, ticker assente dal dataset: %s", ex.ticker)
            resoconto.append({"ticker": ex.ticker, "reason": ex.reason,
                              "applicata": False, "osservazioni_rimosse": 0,
                              "nota": "ticker assente dal dataset"})
            continue

        rimosse = 0
        for df in out.values():
            if ex.ticker not in df.columns:
                continue
            col = df[ex.ticker]
            mask = col.notna()
            if ex.start is not None:
                mask &= df.index >= ex.start
            if ex.end is not None:
                mask &= df.index <= ex.end
            rimosse = max(rimosse, int(mask.sum()))
            df.loc[mask, ex.ticker] = np.nan

        resoconto.append({
            "ticker": ex.ticker,
            "start_date": str(ex.start.date()) if ex.start is not None else None,
            "end_date": str(ex.end.date()) if ex.end is not None else None,
            "reason": ex.reason,
            "applicata": True,
            "osservazioni_rimosse": rimosse,
        })
        log.info("escluso %s (%d osservazioni)", ex.describe(), rimosse)

    return out, resoconto


# ---------------------------------------------------------------------------
def detect_broken_series(
    close_adj: pd.DataFrame,
    close_raw: pd.DataFrame | None = None,
    *,
    return_threshold: float = 0.60,
    divergence_threshold: float = 0.25,
    reversal_tolerance: float = 0.5,
) -> pd.DataFrame:
    """Individua i giorni con la firma tipica di un errore di dato.

    Tre segnali, ciascuno con il suo significato:

    `estremo`    — |rendimento| oltre soglia. Da solo non basta: esistono
                   giornate vere a +60% (fusioni annunciate, biotech).
    `divergenza` — il rendimento su prezzo rettificato e quello su prezzo
                   grezzo si discostano troppo. E' la firma del **fattore di
                   rettifica sbagliato**: split o concambio non gestiti.
    `rimbalzo`   — un salto che il giorno dopo si annulla quasi del tutto.
                   Nessun evento societario si comporta cosi': e' un prezzo
                   sbagliato per un giorno.

    Un giorno estremo che e' ANCHE divergente o rimbalzato e' quasi
    certamente un errore. Un estremo isolato va guardato a mano.
    """
    r_adj = close_adj.pct_change()
    estremo = r_adj.abs() > return_threshold

    if close_raw is not None:
        r_raw = close_raw.reindex_like(close_adj).pct_change()
        divergenza = (r_adj - r_raw).abs() > divergence_threshold
    else:
        divergenza = pd.DataFrame(False, index=close_adj.index, columns=close_adj.columns)

    # Un salto seguito dal suo (quasi) opposto. Il test va fatto sui rendimenti
    # LOGARITMICI: quelli semplici sono asimmetrici, e un +300% annullato da un
    # -75% sommerebbe +225% invece di zero, mancando ogni rimbalzo.
    lr = np.log(close_adj.where(close_adj > 0)).diff()
    rimbalzo = estremo & ((lr + lr.shift(-1)).abs() < lr.abs() * reversal_tolerance)

    sospetto = estremo | divergenza
    righe = []
    for tk in close_adj.columns:
        hit = sospetto[tk]
        if not hit.any():
            continue
        date = close_adj.index[hit.fillna(False)]
        righe.append({
            "ticker": tk,
            "giorni_sospetti": int(len(date)),
            "estremi": int(estremo[tk].sum()),
            "divergenti": int(divergenza[tk].sum()),
            "rimbalzi": int(rimbalzo[tk].fillna(False).sum()),
            "prima_data": date.min(),
            "ultima_data": date.max(),
            "peggior_rendimento": float(r_adj[tk].reindex(date).abs().max()),
        })

    out = pd.DataFrame(righe)
    if out.empty:
        return out
    # priorita' a chi ha divergenze o rimbalzi: sono i casi quasi certi
    out["quasi_certo"] = (out["divergenti"] > 0) | (out["rimbalzi"] > 0)
    return out.sort_values(["quasi_certo", "peggior_rendimento"], ascending=False).reset_index(drop=True)


def suggest_exclusions(
    detected: pd.DataFrame,
    padding_days: int = 45,
    only_certain: bool = True,
) -> pd.DataFrame:
    """Da serie sospette a righe pronte per exclusions.csv.

    Propone una **finestra stretta** intorno alle date incriminate, non
    l'intera serie: scartare vent'anni di dati validi per un errore di un mese
    introduce un bias peggiore di quello che corregge.
    """
    if detected.empty:
        return pd.DataFrame(columns=list(COLUMNS))

    df = detected[detected["quasi_certo"]] if only_certain else detected
    righe = []
    for _, r in df.iterrows():
        motivo = []
        if r["divergenti"]:
            motivo.append(f"{int(r['divergenti'])} giorni con rettifica divergente")
        if r["rimbalzi"]:
            motivo.append(f"{int(r['rimbalzi'])} salti rientrati il giorno dopo")
        if r["estremi"]:
            motivo.append(f"{int(r['estremi'])} rendimenti oltre soglia")
        righe.append({
            "ticker": r["ticker"],
            "start_date": (r["prima_data"] - pd.Timedelta(days=padding_days)).date(),
            "end_date": (r["ultima_data"] + pd.Timedelta(days=padding_days)).date(),
            "reason": "AUTO: " + "; ".join(motivo),
        })
    return pd.DataFrame(righe, columns=list(COLUMNS))
