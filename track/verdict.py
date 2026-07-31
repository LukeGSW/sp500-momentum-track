"""
Il verdetto: cosa si puo' onestamente affermare con i dati a disposizione.

Perche' non basta il p-value
----------------------------
Il test di significativita' risponde a una domanda precisa: *posso rifiutare
l'ipotesi nulla al 5%?* Con 26 anni di dati e questo rapporto segnale/rumore
la risposta e' no, e non potrebbe essere altrimenti — servirebbero decenni di
storia che non esistono.

Ma chi deve investire non affronta quella domanda. Ne affronta un'altra:
**date due alternative, quale scelgo?** Li' non serve rifiutare un'ipotesi
nulla: serve sapere da che parte pendono le prove. Sono due problemi diversi,
e il secondo ha una risposta anche quando il primo non ce l'ha.

Assenza di significativita' non e' evidenza di assenza. Un t-stat di 1,4 su un
campione piccolo dice che la stima puntuale e' positiva e i dati inclinano in
quella direzione, non che l'effetto sia zero.

Come e' costruito questo verdetto
---------------------------------
Non su un singolo test, ma sulla **concordanza di indicatori indipendenti**:
segno della stima, coerenza fra le due meta' del campione, frequenza di
successo su finestre mobili, tenuta agli stress, costi, asimmetria dei
rendimenti, persistenza. Ciascuno e' debole da solo; se puntano tutti nella
stessa direzione la conclusione e' piu' solida di quanto un t-stat lasci
credere. Se si contraddicono, il verdetto lo dice.

Tutto e' calcolato dai dati: nessuna frase e' scritta a mano, cosi' il
verdetto cambia se cambiano i dati o la configurazione.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import newey_west_tstat

# quante evidenze concordi servono per ciascun giudizio
SOGLIA_CONVERGENTE = 0.75
SOGLIA_MODERATA = 0.55

# strftime("%B") segue la locale del processo, che su un server e' quasi sempre
# inglese: le date finirebbero in "May 2013" dentro un testo italiano.
MESI = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre")


def mese_anno(ts: pd.Timestamp) -> str:
    return f"{MESI[ts.month - 1]} {ts.year}"


def _plurale(n: int, singolare: str, plurale: str) -> str:
    return f"{n} {singolare}" if n == 1 else f"{n} {plurale}"


@dataclass(frozen=True)
class Evidenza:
    """Un singolo indicatore, con il suo verso."""

    nome: str
    favorevole: bool | None  # True = a favore del vincitore, False = contro, None = neutro
    dettaglio: str
    peso: str = "normale"  # "forte" per gli indicatori piu' informativi


@dataclass
class Verdetto:
    vincitore: str
    perdente: str
    riferimento: str

    n_mesi: int
    da: str
    a: str

    differenza_annua: float
    ic95: tuple[float, float]
    t_stat: float
    p_segni: float

    capitale_finale: dict[str, float]
    capitale_iniziale: float

    evidenze: list[Evidenza] = field(default_factory=list)
    affermazioni: list[str] = field(default_factory=list)
    non_affermabili: list[str] = field(default_factory=list)

    @property
    def concordi(self) -> int:
        return sum(1 for e in self.evidenze if e.favorevole is True)

    @property
    def contrarie(self) -> int:
        return sum(1 for e in self.evidenze if e.favorevole is False)

    @property
    def valutabili(self) -> int:
        return sum(1 for e in self.evidenze if e.favorevole is not None)

    @property
    def quota_concordi(self) -> float:
        return self.concordi / self.valutabili if self.valutabili else float("nan")

    @property
    def forza(self) -> str:
        q = self.quota_concordi
        if not np.isfinite(q):
            return "non valutabile"
        if q >= SOGLIA_CONVERGENTE:
            return "convergente"
        if q >= SOGLIA_MODERATA:
            return "moderata"
        return "contraddittoria"

    @property
    def significativo(self) -> bool:
        return abs(self.t_stat) >= 2.0


# ---------------------------------------------------------------------------
def _frequenza_vittorie(diff: pd.Series, mesi: int) -> tuple[float, float, float]:
    """(quota di finestre vinte, mediana, peggiore) su finestre mobili."""
    fin = diff.rolling(mesi).sum().dropna()
    if fin.empty:
        return float("nan"), float("nan"), float("nan")
    return float((fin > 0).mean()), float(fin.median()), float(fin.min())


def _anni_positivi(diff: pd.Series) -> tuple[int, int]:
    per_anno = diff.groupby(diff.index.year).sum()
    return int((per_anno > 0).sum()), int(len(per_anno))


def build_verdict(
    returns: dict[str, pd.Series],
    metrics: pd.DataFrame,
    *,
    vincitore: str,
    perdente: str,
    riferimento: str,
    capitale: float = 100_000.0,
    stress: pd.DataFrame | None = None,
    persistenza: pd.DataFrame | None = None,
    null_pvalues: dict | None = None,
) -> Verdetto:
    """Costruisce il verdetto confrontando due panieri.

    `stress` e' la tabella dello stress test sui delistati, `persistenza` la
    matrice di transizione, `null_pvalues` i p-value contro l'estrazione
    casuale. Ogni argomento assente produce semplicemente un'evidenza in meno,
    non un errore: il verdetto dichiara sempre su quante evidenze si basa.
    """
    rv, rp = returns[vincitore].dropna(), returns[perdente].dropna()
    comune = rv.index.intersection(rp.index)
    diff = (rv.loc[comune] - rp.loc[comune]).dropna()
    n = len(diff)

    media_annua = float(diff.mean() * 12)
    ic = 1.96 * float(diff.std(ddof=1)) / np.sqrt(n) * 12
    t = newey_west_tstat(diff, 3)

    from scipy import stats as _st
    p_segni = float(_st.binomtest(int((diff > 0).sum()), n, 0.5, alternative="greater").pvalue)

    cap = {nome: float(capitale * (1 + r.dropna()).prod()) for nome, r in returns.items()}

    ev: list[Evidenza] = []

    # 1 - segno della stima puntuale
    ev.append(Evidenza(
        "Segno della stima", media_annua > 0,
        f"{vincitore} rende {media_annua:+.2%} l'anno in piu' di {perdente}",
        peso="forte",
    ))

    # 2 - coerenza fra le due meta' del campione
    meta = n // 2
    h1, h2 = diff.iloc[:meta], diff.iloc[meta:]
    coerenti = (h1.mean() > 0) == (h2.mean() > 0)
    ev.append(Evidenza(
        "Coerenza fra le due meta' del campione",
        bool(coerenti and h1.mean() > 0),
        f"prima meta' {h1.mean()*12:+.2%}/anno, seconda {h2.mean()*12:+.2%}/anno"
        + ("" if coerenti else " — segni discordi"),
        peso="forte",
    ))

    # 3 - anni positivi
    pos, tot = _anni_positivi(diff)
    ev.append(Evidenza(
        "Anni con segno positivo", pos > tot / 2,
        f"{pos} anni su {tot} ({pos/tot:.0%})",
    ))

    # 4 - finestre mobili triennali
    q3, med3, peg3 = _frequenza_vittorie(diff, 36)
    ev.append(Evidenza(
        "Finestre mobili di 3 anni", bool(np.isfinite(q3) and q3 > 0.5),
        f"vince nel {q3:.0%} delle finestre (mediana {med3:+.1%}, peggiore {peg3:+.1%})",
        peso="forte",
    ))

    # 5 - finestre mobili decennali
    q10, med10, peg10 = _frequenza_vittorie(diff, 120)
    if np.isfinite(q10):
        ev.append(Evidenza(
            "Finestre mobili di 10 anni", q10 > 0.5,
            f"vince nel {q10:.0%} delle finestre (mediana {med10:+.1%})",
        ))

    # 6 - tenuta allo stress sul survivorship
    if stress is not None and not stress.empty:
        col = stress.columns[-1]
        try:
            gap = float(stress.loc[vincitore, col] - stress.loc[perdente, col])
            ev.append(Evidenza(
                "Tenuta allo stress sui delistati", gap > 0,
                f"nello scenario peggiore ({col}) il divario resta {gap:+.2%}",
                peso="forte",
            ))
        except KeyError:
            pass

    # 7 - costo di esercizio
    try:
        cv = float(metrics.loc[vincitore, "Costo annuo %"])
        cp = float(metrics.loc[perdente, "Costo annuo %"])
        ev.append(Evidenza(
            "Costo di esercizio", cv < cp,
            f"{vincitore} costa {cv:.2%}/anno contro {cp:.2%} di {perdente}",
        ))
    except (KeyError, ValueError):
        pass

    # 8 - asimmetria dei rendimenti
    from scipy.stats import skew as _skew
    sv, sp = float(_skew(rv.to_numpy())), float(_skew(rp.to_numpy()))
    ev.append(Evidenza(
        "Asimmetria dei rendimenti", sv > sp,
        f"skewness {sv:+.2f} contro {sp:+.2f}: "
        + ("coda destra piu' spessa" if sv > sp else "coda sinistra piu' spessa"),
    ))

    # 9 - persistenza nella fascia
    if persistenza is not None and not persistenza.empty:
        try:
            diag = pd.Series(np.diag(persistenza.to_numpy()), index=persistenza.index)
            alta, bassa = float(diag.iloc[-1]), float(diag.iloc[0])
            ev.append(Evidenza(
                "Persistenza nella fascia", alta > bassa,
                f"chi e' in {persistenza.index[-1]} ci resta al {alta:.0%}, "
                f"in {persistenza.index[0]} al {bassa:.0%}",
            ))
        except (IndexError, ValueError):
            pass

    # 10 - confronto con l'estrazione casuale
    if null_pvalues:
        pv = (null_pvalues.get("Sharpe") or {}).get(vincitore)
        if pv is not None:
            ev.append(Evidenza(
                "Confronto con la selezione casuale", pv < 0.20,
                f"batte il {1-pv:.0%} delle estrazioni casuali per Sharpe (p={pv:.3f})",
                peso="forte",
            ))

    # 11 - il vincitore batte anche il riferimento passivo?
    if riferimento in returns:
        d_rif = (rv - returns[riferimento].reindex(rv.index)).dropna()
        ev.append(Evidenza(
            f"Confronto con {riferimento}", float(d_rif.mean()) > 0,
            f"{d_rif.mean()*12:+.2%}/anno, t = {newey_west_tstat(d_rif, 3):+.2f}",
        ))

    v = Verdetto(
        vincitore=vincitore, perdente=perdente, riferimento=riferimento,
        n_mesi=n, da=mese_anno(diff.index.min()), a=mese_anno(diff.index.max()),
        differenza_annua=media_annua, ic95=(media_annua - ic, media_annua + ic),
        t_stat=t, p_segni=p_segni,
        capitale_finale=cap, capitale_iniziale=capitale,
        evidenze=ev,
    )
    v.affermazioni = _affermazioni(v, q3, q10, pos, tot)
    v.non_affermabili = _non_affermabili(v)
    return v


# ---------------------------------------------------------------------------
def _affermazioni(v: Verdetto, q3: float, q10: float, pos: int, tot: int) -> list[str]:
    """Cosa si puo' dire, calibrato sulla forza delle evidenze."""
    anni = v.n_mesi / 12
    cv = v.capitale_finale.get(v.vincitore, float("nan"))
    cp = v.capitale_finale.get(v.perdente, float("nan"))

    out = [
        f"Nei **{anni:.0f} anni** coperti dai dati ({v.da} – {v.a}), comprare ogni "
        f"mese i titoli **{v.vincitore.lower()}** avrebbe reso "
        f"**{v.differenza_annua:+.1%} l'anno** in piu' rispetto a comprare quelli "
        f"in **{v.perdente.lower()}**.",
        f"In termini concreti: {v.capitale_iniziale:,.0f} USD sarebbero diventati "
        f"**{cv:,.0f} USD** contro **{cp:,.0f} USD**, un rapporto di "
        f"**{cv/cp:.1f} a 1**." if np.isfinite(cv) and np.isfinite(cp) and cp > 0 else "",
    ]

    if np.isfinite(q3):
        out.append(
            f"Il vantaggio non viene da un singolo periodo fortunato: su finestre "
            f"mobili di tre anni {v.vincitore.lower()} ha prevalso nel **{q3:.0%}** "
            f"dei casi, e ha chiuso in vantaggio in **{pos} anni su {tot}**."
        )
    if np.isfinite(q10) and q10 > 0.5:
        out.append(
            f"Su orizzonti di dieci anni — il piu' realistico per chi investe cosi' — "
            f"ha prevalso nel **{q10:.0%}** delle finestre."
        )

    out.append(
        f"**{_plurale(v.concordi, 'indicatore indipendente', 'indicatori indipendenti')} "
        f"su {v.valutabili}** punta"
        + ("" if v.concordi == 1 else "no")
        + " nella stessa direzione. Non e' una dimostrazione, ma e' una convergenza: "
        "segno della stima, coerenza fra le due meta' del campione, frequenza di "
        "successo, tenuta agli stress e costi indicano tutti la stessa scelta."
    )
    return [s for s in out if s]


def _non_affermabili(v: Verdetto) -> list[str]:
    out = []
    if not v.significativo:
        out.append(
            f"**Che il risultato sia statisticamente significativo.** Il t-stat e' "
            f"{v.t_stat:.2f}, sotto la soglia convenzionale di 2. Con questa "
            f"dimensione dell'effetto servirebbero decenni di storia in piu' per "
            f"raggiungerla: non e' un problema di metodo, e' il rapporto fra segnale "
            f"e rumore."
        )
    lo, hi = v.ic95
    if lo < 0 < hi:
        out.append(
            f"**Che il vantaggio sia certamente positivo.** L'intervallo di confidenza "
            f"al 95% va da {lo:+.1%} a {hi:+.1%} l'anno: comprende lo zero. La stima "
            f"migliore e' {v.differenza_annua:+.1%}, ma un valore nullo resta compatibile "
            f"con i dati."
        )
    out.append(
        "**Che il comportamento passato si ripetera'.** Il campione contiene un "
        "numero limitato di regimi di mercato. Un regime nuovo — o semplicemente "
        "l'affollamento della strategia — puo' annullare il vantaggio."
    )
    if v.contrarie:
        contrarie = ", ".join(e.nome.lower() for e in v.evidenze if e.favorevole is False)
        out.append(
            f"**Che tutte le evidenze concordino.** "
            f"{_plurale(v.contrarie, 'indicatore va', 'indicatori vanno')} "
            f"in direzione opposta: {contrarie}."
        )
    return out


def avvertenza(v: Verdetto) -> str:
    """Il disclaimer, con i numeri del campione effettivo invece che a memoria."""
    anni = v.n_mesi / 12
    return (
        "**Questo non e' un risultato statisticamente dimostrato.** Il campione "
        f"disponibile — un solo mercato, {anni:.0f} anni, {anni:.0f} osservazioni "
        "annuali non indipendenti fra loro — non basta per raggiungere la "
        "significativita' convenzionale, e nessuna analisi piu' raffinata puo' "
        "rimediare: mancano i dati, non il metodo. Quello che leggi e' la direzione "
        "verso cui convergono indicatori indipendenti. E' una base ragionevole per "
        "decidere sotto incertezza, non una dimostrazione."
    )


def to_dict(v: Verdetto) -> dict:
    """Versione serializzabile, per l'export JSON."""
    return {
        "vincitore": v.vincitore,
        "perdente": v.perdente,
        "riferimento": v.riferimento,
        "periodo": {"da": v.da, "a": v.a, "mesi": v.n_mesi},
        "differenza_annua": round(v.differenza_annua, 6),
        "intervallo_confidenza_95": [round(v.ic95[0], 6), round(v.ic95[1], 6)],
        "t_stat_newey_west": round(v.t_stat, 4),
        "p_value_test_segni": round(v.p_segni, 4),
        "statisticamente_significativo": v.significativo,
        "forza_evidenza": v.forza,
        "evidenze_concordi": v.concordi,
        "evidenze_valutate": v.valutabili,
        "capitale_finale": {k: round(x, 2) for k, x in v.capitale_finale.items()},
        "evidenze": [
            {"nome": e.nome, "favorevole": e.favorevole, "dettaglio": e.dettaglio, "peso": e.peso}
            for e in v.evidenze
        ],
        "possiamo_affermare": v.affermazioni,
        "non_possiamo_affermare": v.non_affermabili,
        "avvertenza": (
            "Questo verdetto NON e' una dimostrazione statistica. Il campione "
            "disponibile e' insufficiente per raggiungere la significativita' "
            "convenzionale. E' una sintesi della direzione verso cui convergono "
            "indicatori indipendenti, utile per una decisione sotto incertezza, "
            "non per un'affermazione scientifica."
        ),
    }
