"""Metodologia: come e' costruito lo studio e perche'."""
from __future__ import annotations

import streamlit as st

from track import didactics, ui
from track.config import BAND_NAMES, HORIZONS_BY_HOLDING, PREREGISTERED

ui.page_config("Metodologia")
st.title("🏁 La Pista — Metodologia")
st.caption(ui.SUBTITLE)

st.markdown(
    """
## La domanda, e perche' e' ambigua

*"Conviene comprare azioni in forte momentum, o in debolezza momentanea?"*
contiene in realta' tre domande diverse:

1. **Momentum trasversale a 12 mesi** — comprare chi ha battuto il mercato
   nell'ultimo anno. Premio storicamente documentato, ma con crolli violenti.
2. **Inversione di brevissimo termine** — comprare chi ha appena perso.
   Anomalia reale, storicamente mangiata dai costi.
3. **Ritracciamento dentro un trend** — comprare chi e' strutturalmente in
   salita ma temporaneamente debole.

Il filtro *prezzo sopra la media a 200 sedute* trasforma automaticamente la
domanda nella terza. Non e' un errore, ma **il filtro non e' neutrale: e' gia'
una scommessa sul momentum**. Per questo e' un interruttore e non una costante:
il confronto acceso/spento isola l'effetto del filtro da quello della selezione.

---

## Perche' non un grafico rotazionale

Uno schema rotazionale classico mette la forza relativa su un asse e la sua
derivata sull'altro, e i titoli ruotano tra quattro quadranti. Noi abbiamo
smontato quella geometria per due ragioni.

**La prima e' tecnica.** La formula classica normalizza in serie storica *per
singolo titolo*: due titoli con lo stesso punteggio non sono confrontabili tra
loro, perche' ciascuno e' normalizzato sulla propria storia. Su undici ETF
settoriali si tollera; su cinquecento azioni rende le fasce prive di
significato. Noi standardizziamo **trasversalmente**, cioe' confrontando i
titoli tra loro alla stessa data.

**La seconda e' di lettura.** Su una pista dritta il momentum non ha bisogno di
un asse suo: *e'* la pendenza della traiettoria. Chi sale sta migliorando, chi
scende sta peggiorando. In cambio otteniamo che la fascia piu' bassa e quella
piu' alta corrispondono letteralmente alla domanda dello studio — cosa che in
uno schema rotazionale non accade, perche' li' i due estremi del ciclo sono
"debolezza profonda" e "forza che sfuma".

Non usiamo la nomenclatura ne' le formule degli strumenti rotazionali
commerciali, che sono marchi registrati dei rispettivi titolari.

---

## Il segnale

### Forza F — la posizione sulla pista

Per ogni orizzonte h si calcola il log-rendimento su h sedute, lo si
standardizza trasversalmente con **mediana e MAD** (non media e deviazione
standard: su cinquecento titoli bastano due outlier per schiacciare tutti gli
altri verso lo zero), lo si winsorizza a ±3. F e' la media di questi z-score,
ri-standardizzata.

Un titolo entra nel calcolo solo se ha **tutti** gli orizzonti disponibili: la
copertura parziale produrrebbe punteggi non confrontabili.

> **Il benchmark si cancella da solo.** Il rendimento di mercato e' identico per
> tutti i titoli alla stessa data, quindi sparisce nella standardizzazione
> trasversale. Cambiare SPY con RSP non muove F di un decimale — e' verificato
> da un test automatico. Un parametro in meno, una dipendenza dati in meno, una
> fonte di anticipazione in meno.

### Spinta V — la velocita' lungo la pista

Pendenza della regressione lineare di F sulle ultime N sedute, ri-standardizzata.
Implementata come somma pesata con pesi fissi: con ascisse equispaziate la
pendenza e' una combinazione lineare dei valori nella finestra.

### Le fasce

Quintili **trasversali** di F fra i titoli eleggibili, dal basso verso l'alto:
"""
)

st.markdown("\n".join(f"{i}. **{n}**" for i, n in enumerate(BAND_NAMES, 1)))

st.markdown(
    """
Quantili e non soglie assolute in z: le soglie fisse producono panieri vuoti o
squilibrati quando la distribuzione trasversale si sposta (2008, 2020), e un
backtest con panieri di dimensione variabile nel tempo non e' confrontabile con
se stesso.

Combinando fascia e segno della Spinta si ottengono **dieci stati** invece di
quattro quadranti, e la velocita' resta una variabile continua invece di essere
dicotomizzata da un confine.

---

## La congruenza lookback / holding

Un segnale la cui informazione decade in un mese non ha senso su un holding di
tre: la posizione resta aperta dopo che il segnale e' morto, e si paga turnover
per nulla. Per ogni holding definiamo gli orizzonti congrui:
"""
)

st.dataframe(
    {
        "Holding (mesi)": list(HORIZONS_BY_HOLDING),
        "Orizzonti di F (sedute)": [", ".join(str(x) for x in v)
                                    for v in HORIZONS_BY_HOLDING.values()],
        "Finestra di V (sedute)": [60 if k == 3 else (21 if k == 1 else 126)
                                   for k in HORIZONS_BY_HOLDING],
    },
    width="stretch", hide_index=True,
)

st.markdown(didactics.escape_markdown(
    f"""
La configurazione preregistrata usa **holding {PREREGISTERED.holding_months} mesi**.
L'orizzonte a 1 mese resta nella griglia perche' e' l'unico su cui l'inversione
di brevissimo termine puo' manifestarsi: scartarlo a priori escluderebbe per
costruzione meta' della domanda.

---

## Il portafoglio

| Scelta | Valore | Perche' |
|---|---|---|
| Capitale | {PREREGISTERED.capital:,.0f}$ **per paniere** | Panieri indipendenti, non un capitale diviso: mescolandoli i due effetti si mediano e la domanda resta senza risposta |
| Titoli per paniere | {PREREGISTERED.n_names} | Slot da {PREREGISTERED.slot_value:,.0f}$: la commissione fissa pesa 4,5 bps per lato invece dei 9,8 che peserebbe con 65 titoli |
| Holding | {PREREGISTERED.holding_months} mesi su {PREREGISTERED.n_tranches} tranche sfalsate | Costi da ribilanciamento trimestrale, ma 12 decisioni l'anno invece di 4: conserviamo ~316 osservazioni mensili invece di ~105, e sparisce il timing luck del trimestre di partenza |
| Commissione | {PREREGISTERED.commission_per_side:.2f}$ per lato | Costo di oggi, applicato a tutto lo storico |
| Spread | {PREREGISTERED.spread_bps:.1f} bps per lato | Un conto da 100.000$ su large cap e' price taker: non muove il mercato |
| Cap sul prezzo | {PREREGISTERED.max_share_price:,.0f}$ per azione | Sopra questa soglia il titolo non entra in uno slot con lotti interi |
| Lotti | interi | Nessun anacronismo: le frazionarie retail non esistevano prima del 2019 |
| Liquidita' | remunerata al T-bill 3 mesi | Nel 2000-2007 il tasso era 4-5%: assumere zero penalizzerebbe le fasi in cui il filtro svuota l'universo, cioe' i bear market |

### Segnale a t, esecuzione a t+1

F, V, fasce ed eleggibilita' si leggono all'ultima seduta del mese; gli scambi
avvengono all'**apertura della prima seduta del mese successivo**. Calcolare ed
eseguire sullo stesso prezzo di chiusura sarebbe anticipazione mascherata.

Un test automatico verifica la proprieta' in forma forte: **il segnale calcolato
su dati troncati alla data t deve coincidere esattamente con quello calcolato
sul dataset completo**. Se quel test fallisce, ogni numero prodotto dal backtest
e' privo di significato.

### Contabilita' in adjusted, dimensionamento in grezzo

Il valore di una posizione evolve con il rendimento total-return (dividendi
reinvestiti, split gestiti); quante azioni compri dipende dal prezzo che paghi
davvero. Mischiare i due e' l'errore classico che gonfia i titoli ad alto
dividendo.

### Ribilanciamento conservativo sui costi

Vendiamo solo cio' che esce dal target, compriamo solo cio' che entra, non
ripesiamo i sopravvissuti. Un ribilanciamento completo a equal weight
costerebbe sensibilmente di piu'.

---

## L'ipotesi nulla

Il confronto corretto **non e' SPY**. Battere l'indice puo' dipendere
interamente dall'equal weighting o dal filtro sulla media mobile, non dalla
selezione dei titoli.

Il nulla e': **estrarre a caso {PREREGISTERED.n_names} titoli dallo stesso
universo eleggibile, alle stesse date**, con lo stesso motore, gli stessi costi
e lo stesso arrotondamento a lotti interi. Ripetuto centinaia di volte. Il
p-value empirico e' la frazione di estrazioni casuali che hanno battuto il
paniere reale.

Solo cosi' si isola l'effetto della **selezione per fascia** da quello
dell'equal weighting, del filtro e della composizione dell'universo.

A questo si aggiungono il t-stat di Newey-West sullo spread (i rendimenti
mensili di portafogli con holding a 3 mesi sono autocorrelati per costruzione:
senza correzione il t-stat risulta gonfiato) e la scomposizione per regime di
mercato.

---

## I limiti, con la loro direzione

Il limite piu' serio non e' statistico ma di dato.

> **I prezzi dei titoli delistati.** Sapere chi era nell'indice a una certa data
> risolve meta' del problema di survivorship. L'altra meta' sono i prezzi di chi
> poi e' sparito. I titoli che spariscono si concentrano **nel paniere piu'
> debole**: ogni buco rimuove dal backtest proprio i peggiori risultati di quel
> paniere. **Direzione del bias: a favore della tesi contrarian.** Senza
> correzione, "compra la debolezza" potrebbe sembrare vincente solo perche' i
> fallimenti sono invisibili.
>
> Per questo la pagina Backtest ripete lo studio imponendo un rendimento
> terminale di −30%, −50% e −100% ai titoli che spariscono senza prezzo di
> uscita. **Se la conclusione regge a −100%, e' robusta.**

Gli altri limiti noti sono elencati per esteso, ciascuno con la propria
direzione di distorsione, nel campo `caveats` dell'export JSON.

---

## Cosa questo studio non e'

- **Non e' un consiglio di investimento.** E' uno strumento di misura.
- **Non risponde a "cosa potevi fare nel 2001"**, perche' applica i costi di
  oggi a tutto lo storico. Risponde a *"cosa avrebbe prodotto questo segnale,
  con gli attriti di oggi"*.
- **Non copre la salita della bolla dot-com.** La fonte dati parte dal 2000: lo
  scoppio si', la salita no.
- **Lo spread Testa Corsa − Fondo Griglia non e' implementabile**: i panieri
  sono simulati long-only, e lo short selling ha costi e vincoli propri —
  compresa l'impossibilita' di shortare i nomi piu' stressati proprio quando
  servirebbe.
"""
))

st.divider()
st.subheader("Indice dei riquadri esplicativi")
st.caption("Ogni grafico e ogni tabella della dashboard ha il proprio riquadro. "
           "Qui sotto sono raccolti tutti.")

for key in didactics.BLOCKS:
    didactics.render(key)
