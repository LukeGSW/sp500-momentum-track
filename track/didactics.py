"""
Riquadri esplicativi per OGNI grafico e OGNI tabella della dashboard.

Tutto il testo interpretativo vive qui, centralizzato, cosi' che le pagine
Streamlit restino codice e non prosa. Ogni blocco risponde a quattro domande:

    cosa      -> cosa sto guardando
    lettura   -> come si legge, passo per passo
    attenzione-> cosa NON si puo' dedurre (la parte che di solito manca)
    formula   -> la matematica, quando serve

Uso nelle pagine:  didactics.render("pista")
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Block:
    title: str
    cosa: str
    lettura: tuple[str, ...]
    attenzione: tuple[str, ...]
    formula: str = ""


BLOCKS: dict[str, Block] = {}


def _add(key: str, blk: Block) -> None:
    BLOCKS[key] = blk


# ===========================================================================
# PAGINA 1 - LA PISTA
# ===========================================================================

_add("pista", Block(
    title="La Pista",
    cosa=(
        "Ogni pallino e' un titolo dell'S&P 500. La posizione **verticale** e' la "
        "Forza relativa **F**: quanto quel titolo sta andando meglio o peggio "
        "della media del mercato, misurata su piu' orizzonti temporali insieme. "
        "La posizione **orizzontale** dice soltanto a quale settore GICS "
        "appartiene: sono 11 corsie parallele, non un secondo asse numerico."
    ),
    lettura=(
        "**Alto = forte, basso = debole.** F e' uno z-score trasversale: F = 0 e' "
        "il titolo mediano della giornata, F = +2 significa due deviazioni "
        "standard robuste sopra la mediana. E' una misura *relativa*: in un "
        "mercato che sale tutto, meta' dei titoli sta comunque sotto zero.",
        "**La scia e' il momentum.** La coda dietro ogni pallino mostra dove si "
        "trovava il titolo nelle settimane precedenti. Scia che punta verso "
        "l'alto = forza in aumento. Scia verso il basso = forza in calo. Su una "
        "pista dritta il momentum non ha bisogno di un asse suo: e' la pendenza.",
        "**Il colore** codifica il segno della Spinta V (la pendenza misurata "
        "formalmente), la **dimensione** del pallino la sua intensita'. Un "
        "pallino grande e verde sta accelerando forte verso l'alto.",
        "**Le cinque bande orizzontali** sono i quintili della giornata: sotto "
        "sta Fondo Griglia (il 20% piu' debole), sopra Testa Corsa (il 20% piu' "
        "forte). Essendo quintili, ogni banda contiene sempre lo stesso numero "
        "di titoli: e' un ordinamento, non una soglia assoluta.",
        "**Titoli concentrati in una corsia** significa rotazione settoriale in "
        "corso. Se Testa Corsa e' per meta' Information Technology, il 'momentum' "
        "che stai guardando e' in buona parte una scommessa su un settore.",
    ),
    attenzione=(
        "F **non e' un prezzo e non e' un rendimento**. Un titolo in Testa Corsa "
        "puo' avere perso il 10% in valore assoluto: significa solo che gli altri "
        "hanno perso di piu'.",
        "F **non predice nulla da solo**. Dice dove sei, non dove vai. La domanda "
        "'conviene comprare qui?' ha risposta solo nella pagina Backtest.",
        "Le posizioni vicine allo zero sono **rumore**. Titoli con |F| < 0.3 sono "
        "statisticamente indistinguibili dalla mediana: la banda in cui cadono "
        "dipende da frazioni di percentuale ed e' instabile giorno per giorno.",
        "Il grafico mostra solo i titoli **eleggibili** (sopra la media mobile a "
        "200 sedute, se il filtro e' attivo). I titoli in depressione strutturale "
        "sono gia' stati esclusi: quello che vedi in fondo alla pista e' "
        "debolezza *dentro un trend rialzista*, non un titolo che sta morendo.",
    ),
    formula=(
        "F = z( media pesata degli z-score trasversali dei log-rendimenti su "
        "63/126/252 sedute ).  Standardizzazione robusta (mediana + MAD), "
        "winsorizzazione a +/-3.  \n\n"
        "Nota tecnica: essendo standardizzato *trasversalmente*, il benchmark si "
        "cancella algebricamente — il termine di mercato e' identico per tutti i "
        "titoli alla stessa data, quindi sparisce nello z-score. Cambiare SPY con "
        "RSP non muove F di un decimale."
    ),
))

_add("velocita", Block(
    title="Spinta V",
    cosa=(
        "V e' la pendenza della retta di regressione di F sulle ultime N sedute "
        "(N = 63 con holding 3 mesi). E' la velocita' con cui il titolo sta "
        "risalendo o scendendo lungo la pista."
    ),
    lettura=(
        "**V > 0**: il titolo sta guadagnando forza relativa. **V < 0**: la sta "
        "perdendo.",
        "La combinazione (fascia, segno di V) genera **10 stati** invece dei 4 "
        "quadranti di uno schema rotazionale: piu' informazione, non meno.",
        "Lo stato piu' interessante per la domanda di questo studio e' "
        "**Testa Corsa con V negativo**: un leader che sta ritracciando. E' la "
        "definizione piu' pura di 'comprare la debolezza momentanea di un titolo "
        "forte', e non coincide con Fondo Griglia.",
    ),
    attenzione=(
        "V e' una **derivata**: e' strutturalmente piu' rumorosa di F. Piccoli "
        "cambi di prezzo producono cambi di segno.",
        "V misurato su 63 sedute non dice nulla su cosa e' successo la settimana "
        "scorsa. La finestra e' scelta congruente all'holding period, non "
        "all'ultima notizia.",
    ),
    formula="V = pendenza OLS di F sulle ultime 63 sedute, ri-standardizzata trasversalmente.",
))

_add("snapshot_table", Block(
    title="Tabella titoli (situazione a oggi)",
    cosa=(
        "L'elenco completo dei titoli eleggibili con la loro posizione sulla "
        "pista alla data piu' recente disponibile."
    ),
    lettura=(
        "**F** e **V** sono z-score: confrontabili tra titoli alla stessa data, "
        "non confrontabili tra date diverse in valore assoluto.",
        "**Fascia** e' il quintile del giorno. **Fascia prec.** e' quello di un "
        "mese fa: se differiscono, il titolo si sta muovendo lungo la pista.",
        "**Giorni in fascia** misura la persistenza. Valori molto alti indicano "
        "un titolo stabile; valori bassi un titolo che oscilla sul confine tra "
        "due bande e la cui classificazione va presa con cautela.",
        "**In portafoglio** segnala i 30 titoli che la strategia comprerebbe "
        "oggi, che sono un sottoinsieme della fascia (i 30 con F piu' estremo).",
    ),
    attenzione=(
        "Questa e' una **fotografia**, non un segnale operativo datato. La "
        "strategia del backtest ribilancia a inizio mese: la tabella di meta' "
        "mese mostra dove sarebbero i titoli *se* si ribilanciasse oggi.",
        "I titoli esclusi dal cap sul prezzo per azione (sopra 1.500$) compaiono "
        "nella pista ma sono marcati come non tradabili.",
    ),
))

_add("band_distribution", Block(
    title="Distribuzione per fascia e settore",
    cosa=(
        "Quanti titoli, e di quale settore, occupano ciascuna delle cinque bande "
        "della pista in questo momento."
    ),
    lettura=(
        "Essendo **quintili**, il conteggio totale per banda e' per costruzione "
        "quasi identico. Cio' che conta e' la **composizione settoriale**, non "
        "il numero.",
        "Una banda dominata da 2-3 settori segnala che il momentum in questo "
        "momento e' un fenomeno settoriale, non di selezione titoli.",
        "Confronta Testa Corsa e Fondo Griglia: se i settori sono speculari (es. "
        "tech in alto, utilities in basso) stai osservando una rotazione classica "
        "risk-on / risk-off.",
    ),
    attenzione=(
        "Se la concentrazione settoriale ti preoccupa, attiva l'opzione "
        "**sector-neutral** nella sidebar: le fasce vengono ricalcolate dentro "
        "ciascun settore, cosi' il paniere e' bilanciato per costruzione. Il "
        "confronto tra le due versioni ti dice quanta parte del risultato del "
        "backtest e' rotazione settoriale mascherata da stock selection.",
    ),
))

_add("transition_matrix", Block(
    title="Matrice di transizione tra fasce",
    cosa=(
        "La probabilita' storica che un titolo che oggi si trova nella fascia di "
        "riga si trovi, dopo un mese, nella fascia di colonna. Stimata su tutti "
        "i titoli e tutte le date del campione."
    ),
    lettura=(
        "**Leggi per riga.** Ogni riga somma a 100%. La cella (Testa Corsa, Testa "
        "Corsa) e' la probabilita' che un leader resti leader il mese prossimo.",
        "**La diagonale misura la persistenza.** Diagonale alta = il momentum e' "
        "appiccicoso, chi e' forte resta forte. Diagonale bassa = mean reversion, "
        "le posizioni ruotano.",
        "**Le celle agli angoli** (Testa Corsa -> Fondo Griglia e viceversa) sono "
        "i ribaltamenti violenti. Se sono > 2-3% hai un mercato in cui le "
        "classifiche si rovesciano, e una strategia momentum ci lascia le penne.",
        "Questa matrice risponde a meta' della domanda dello studio **prima** del "
        "backtest: se la persistenza in Testa Corsa e' alta e quella in Fondo "
        "Griglia bassa, il momentum e' un fenomeno reale in questo campione.",
    ),
    attenzione=(
        "Persistenza **non significa profitto**. Un titolo puo' restare in Testa "
        "Corsa per sei mesi generando un rendimento mediocre: la fascia misura "
        "la posizione *relativa*, non il guadagno.",
        "La matrice e' calcolata **su tutto il campione**: aggrega regimi molto "
        "diversi. Il 2001 e il 2017 finiscono nella stessa cella. Usa il filtro "
        "per sottoperiodo per vedere se la struttura e' stabile.",
        "Le transizioni sono misurate a un mese anche quando l'holding e' di tre. "
        "E' una scelta di leggibilita': serve a descrivere la dinamica, non a "
        "replicare la strategia.",
    ),
))

_add("dwell_time", Block(
    title="Tempo di permanenza per fascia",
    cosa=(
        "Per quanti mesi consecutivi, in media e in mediana, un titolo resta "
        "nella stessa banda prima di cambiarla."
    ),
    lettura=(
        "Confronta la permanenza con il tuo **holding period** (3 mesi). Se i "
        "titoli restano in Testa Corsa mediamente 5 mesi, un holding di 3 mesi "
        "sta uscendo troppo presto e paga turnover inutile.",
        "Se la permanenza mediana e' inferiore all'holding, stai tenendo in "
        "portafoglio titoli che il segnale ha gia' abbandonato.",
        "La differenza tra **media** e **mediana** e' informativa: media molto "
        "sopra la mediana significa pochi titoli che restano incollati a lungo e "
        "molti che entrano ed escono subito.",
    ),
    attenzione=(
        "La permanenza e' misurata su **fasce campionate a fine mese**, non "
        "giorno per giorno. Su dati giornalieri il quintile di un titolo sul "
        "confine oscilla in continuazione e la mediana crolla a due o tre "
        "sedute: un numero vero ma inutile, perche' non e' la frequenza a cui "
        "si decide.",
        "Anche cosi' la permanenza e' calcolata su blocchi consecutivi stretti, "
        "senza tolleranza: un titolo che esce dalla fascia per un solo mese e "
        "poi rientra conta come due episodi distinti, non uno lungo.",
    ),
))

# ===========================================================================
# PAGINA 2 - BACKTEST
# ===========================================================================

_add("equity_curves", Block(
    title="Curve di capitale (regime composto)",
    cosa=(
        "L'evoluzione di 100.000$ investiti in ciascun paniere, ribilanciati con "
        "tranche sfalsate e al netto di tutti i costi. Ogni paniere e' un conto "
        "separato da 100.000$: non stai dividendo un capitale, stai confrontando "
        "quattro conti paralleli."
    ),
    lettura=(
        "**Testa Corsa** = i 30 titoli piu' forti. **Fondo Griglia** = i 30 piu' "
        "deboli fra quelli comunque sopra la media a 200 sedute. Sono le due "
        "risposte candidate alla domanda dello studio.",
        "**Testa Corsa in ritracciamento** = leader con Spinta negativa. E' la "
        "terza risposta, quella che molti trader intendono davvero con "
        "'comprare la debolezza'.",
        "**Gruppo** e' il quintile centrale: serve da controllo. Se Testa Corsa e "
        "Fondo Griglia battono entrambi il Gruppo, il segnale sta premiando gli "
        "estremi, non una direzione.",
        "**Universo eleggibile EW** e' la linea che conta davvero. E' l'equal "
        "weight di *tutti* i titoli eleggibili: comprende gia' l'effetto del "
        "filtro a 200 sedute e dell'equal weighting. Batterla significa che la "
        "**selezione per fascia** aggiunge valore. Batterla e' molto piu' "
        "difficile che battere l'indice.",
    ),
    attenzione=(
        "**Non confrontare con SPY.** SPY e' cap-weighted e non filtrato: "
        "batterlo puo' dipendere interamente dall'equal weighting o dal filtro, "
        "non dalla selezione. La linea grigia dell'universo eleggibile e' il "
        "confronto corretto.",
        "La scala logaritmica e' attiva di default: su 26 anni la scala lineare "
        "rende invisibile tutto cio' che accade prima del 2015.",
        "Le curve **compongono**, quindi gli anni finali pesano molto di piu' "
        "nelle statistiche. Per un confronto in cui ogni mese pesa uguale usa il "
        "grafico a capitale fisso qui sotto.",
    ),
))

_add("fixed_capital_pnl", Block(
    title="P&L cumulato a capitale fisso (100.000$)",
    cosa=(
        "Lo stesso backtest, ma con il capitale riportato a 100.000$ a ogni "
        "periodo e il profitto accantonato a parte. La curva e' la somma dei P&L "
        "mensili in dollari, non un montante che cresce."
    ),
    lettura=(
        "**Serve per confrontare, non per stimare la ricchezza finale.** Qui ogni "
        "mese vale esattamente quanto ogni altro mese: il 2001 pesa quanto il "
        "2024. E' la contabilita' corretta per l'inferenza statistica.",
        "La pendenza della curva e' il **P&L medio mensile in dollari**. Un "
        "tratto piatto significa che in quel periodo la strategia non ha prodotto "
        "nulla, indipendentemente da quanto capitale avesse accumulato.",
        "Confronta questa curva con quella composta: se una strategia sembra "
        "ottima nel composto ma piatta qui, tutta la sua performance viene da "
        "pochi anni recenti con capitale gia' grande.",
    ),
    attenzione=(
        "Questa curva **non e' una simulazione separata**: e' derivata dalla "
        "stessa serie di rendimenti mensili (100.000$ x somma cumulata dei "
        "rendimenti). Non e' l'esito di un conto che preleva utili ogni mese, "
        "che avrebbe implicazioni fiscali diverse.",
        "L'arrotondamento a lotti interi e' simulato sul capitale corrente, "
        "quindi il suo peso relativo cala man mano che il capitale composto "
        "cresce. Effetto stimato ~13 bps/anno all'inizio, meno dopo.",
    ),
))

_add("metrics_table", Block(
    title="Tabella metriche",
    cosa="Le statistiche riassuntive di ciascun paniere sul periodo selezionato.",
    lettura=(
        "**CAGR** e' il tasso composto annuo. **Vol** e' la deviazione standard "
        "annualizzata dei rendimenti mensili. **Sharpe** usa il tasso T-bill a 3 "
        "mesi come risk-free, non zero.",
        "**Guarda Sharpe e vol prima del CAGR.** Fondo Griglia contiene titoli "
        "sotto pressione, quindi ha volatilita' strutturalmente piu' alta: un "
        "CAGR superiore con vol doppia non e' una vittoria, e' piu' rischio.",
        "**Max DD** e' la perdita massima da un massimo precedente. **Mesi di "
        "recupero** dice quanto ci e' voluto a tornare in pari: e' la metrica che "
        "determina se una strategia e' tenibile psicologicamente.",
        "**Posizioni medie** conta i titoli in portafoglio su **tutte le tranche "
        "insieme**: con 30 titoli e 3 tranche sfalsate il massimo teorico e' 90, "
        "non 30. Un valore molto sotto quel massimo segnala che la fascia non "
        "conteneva abbastanza nomi — succede sistematicamente al paniere dei "
        "leader in ritracciamento, che e' quindi molto meno diversificato degli "
        "altri e va letto con piu' cautela.",
        "**Turnover** e' la frazione di portafoglio sostituita per "
        "ribilanciamento. **Costo annuo %** e' il drag da commissioni e spread "
        "in percentuale del patrimonio: e' il numero interpretabile. **Costi "
        "totali $** e' il cumulato assoluto e va letto con cautela — su una "
        "curva composta i dollari pagati crescono col patrimonio, quindi un "
        "totale enorme su un conto diventato enorme puo' valere pochi punti "
        "base l'anno.",
        "**Hit rate** e' la percentuale di mesi positivi. Utile per capire se il "
        "rendimento viene da tanti piccoli guadagni o da pochi mesi eccezionali.",
    ),
    attenzione=(
        "Il **Sharpe di una strategia long-only azionaria** contiene quasi "
        "interamente il premio azionario di mercato. Un Sharpe di 0.6 non "
        "significa che il segnale funziona: significa che sei stato investito in "
        "azioni. Il numero che risponde alla domanda e' la **differenza** tra "
        "panieri, e il suo p-value.",
        "Tutte le metriche sono **al netto dei costi** con il modello dichiarato "
        "(1,5$ per lato + 3 bps di spread, costi di oggi applicati a tutto lo "
        "storico).",
    ),
))

_add("null_distribution", Block(
    title="Distribuzione nulla (bootstrap)",
    cosa=(
        "L'istogramma mostra cosa sarebbe successo estraendo **a caso** lo stesso "
        "numero di titoli dallo stesso universo eleggibile, alle stesse date, con "
        "lo stesso holding, gli stessi costi e lo stesso arrotondamento a lotti "
        "interi. Ripetuto centinaia di volte — il numero e' regolabile qui sopra. "
        "Le linee verticali sono i panieri reali."
    ),
    lettura=(
        "**Questo e' il test che conta.** Confrontando con l'estrazione casuale "
        "isoli l'effetto della *selezione per fascia* da tutto il resto: equal "
        "weighting, filtro a 200 sedute, composizione dell'universo, costi.",
        "Se la linea di Testa Corsa cade **dentro** il corpo dell'istogramma, la "
        "selezione per momentum non ha aggiunto nulla di distinguibile dal caso.",
        "Il **p-value empirico** e' la frazione di estrazioni casuali che hanno "
        "fatto meglio del paniere reale. Sotto 0.05 il risultato e' difficile da "
        "attribuire alla fortuna. Sopra 0.20 e' probabilmente rumore.",
        "Guarda **sia CAGR sia Sharpe**. Un paniere puo' avere un CAGR "
        "significativo solo perche' ha piu' rischio: nel pannello Sharpe questo "
        "vantaggio sparisce.",
    ),
    attenzione=(
        "Il bootstrap testa **una** ipotesi alla volta. Se provi venti "
        "configurazioni diverse e ne trovi una con p = 0.04, non hai trovato "
        "niente: con venti test un p < 0.05 e' atteso per puro caso. Guarda il "
        "pannello 'griglia orizzonti' prima di concludere.",
        "L'estrazione casuale campiona **lo stesso universo eleggibile**, non "
        "l'intero S&P 500. Se il filtro a 200 sedute e' spento, cambia anche il "
        "nulla: il confronto resta coerente ma misura una cosa diversa.",
    ),
))

_add("subperiods", Block(
    title="Tabella sottoperiodi",
    cosa=(
        "Le stesse metriche calcolate su regimi di mercato separati: scoppio "
        "dot-com, ripresa 2003-2007, crisi finanziaria, decennio del toro, 2020, "
        "il ciclo dei tassi 2021-2022, il periodo recente."
    ),
    lettura=(
        "**E' qui che si risponde davvero alla domanda.** Un risultato che vale "
        "in tutti i regimi e' un risultato. Un risultato che viene tutto da un "
        "singolo periodo e' un aneddoto.",
        "Il periodo **2000-2002** e' il piu' informativo per la domanda dello "
        "studio: e' lo scoppio della bolla, quando comprare la debolezza ha "
        "distrutto capitale. Se Fondo Griglia vince nel campione completo ma "
        "perde catastroficamente qui, sai che stai comprando una lotteria.",
        "**2009 e 2020-2021** sono i mesi classici dei crolli di momentum: "
        "rimbalzi violenti dai minimi in cui i titoli piu' massacrati "
        "sovraperformano brutalmente. Verifica quanto pesano.",
    ),
    attenzione=(
        "I confini dei sottoperiodi sono **scelti da noi con il senno di poi**. "
        "Non sono una segmentazione neutrale: sono etichette utili per la lettura, "
        "non un test statistico.",
        "Su periodi di 2-3 anni le stime di Sharpe hanno un errore standard "
        "enorme. Leggi la direzione, non il decimale.",
    ),
))

_add("drawdown", Block(
    title="Curva di drawdown",
    cosa="La distanza percentuale dal massimo precedente, giorno per giorno.",
    lettura=(
        "**La profondita'** dice quanto avresti perso nel momento peggiore. "
        "**La larghezza** dice per quanto tempo saresti rimasto sott'acqua, ed e' "
        "la dimensione che fa abbandonare le strategie.",
        "Confronta i drawdown dei panieri **nello stesso momento**: se Testa "
        "Corsa e Fondo Griglia crollano insieme, non stai diversificando nulla, "
        "stai solo prendendo beta di mercato in due modi diversi.",
    ),
    attenzione=(
        "Il drawdown e' calcolato sulla curva **composta** e su dati mensili di "
        "fine periodo. Il drawdown reale intra-mese e' stato peggiore.",
        "Un max drawdown misurato su 26 anni e' **una singola osservazione**. "
        "Non e' una stima del peggio possibile: e' il peggio capitato.",
    ),
))

_add("cost_breakeven", Block(
    title="Sensibilita' ai costi di transazione",
    cosa=(
        "Come cambiano il rendimento di **ciascun** paniere e la loro "
        "**differenza** al variare del costo di transazione. Le due cose si "
        "comportano in modo molto diverso e vanno lette separatamente."
    ),
    lettura=(
        "**Le due linee piene** (i livelli) scendono ripidamente: il costo "
        "distrugge rendimento in assoluto, ed e' la ragione per cui l'holding e' "
        "di 3 mesi e non di uno.",
        "**La linea tratteggiata** (la differenza) e' quasi sempre molto piu' "
        "piatta. Non e' un errore: e' il risultato. Se i due panieri hanno "
        "turnover simile pagano entrambi, e il costo **si cancella nel "
        "confronto**. Quindi la risposta a 'quale dei due conviene' e' robusta "
        "ai costi anche quando la redditivita' assoluta non lo e'.",
        "La differenza si inclina solo se un paniere ruota molto piu' "
        "dell'altro: guarda la colonna Turnover nella tabella metriche. Se "
        "Fondo Griglia ruota il 50% e Testa Corsa il 32%, alzando i costi il "
        "vantaggio del primo si erode piu' in fretta.",
        "La linea verticale segna il **costo assunto** nel backtest. Il costo "
        "per rotazione dipende dallo **slot**: con 30 titoli e 100.000$ lo slot "
        "e' 3.333$, e 1,5$ di commissione valgono 4,5 bps per lato. Raddoppiando "
        "il numero di titoli il peso relativo della commissione raddoppia.",
    ),
    attenzione=(
        "Applichiamo i **costi di oggi a tutto lo storico**, per scelta esplicita. "
        "Nel 2001 le commissioni retail erano 10-30$ per eseguito: a quei livelli "
        "questa strategia sarebbe stata impraticabile. Il backtest risponde a "
        "'cosa avrebbe prodotto il segnale, con gli attriti di oggi', non a "
        "'cosa potevi fare nel 2001'.",
        "Lo spread di 3 bps e' realistico per un conto da 100.000$ su large cap "
        "S&P 500, che e' price taker e non muove il mercato. Non e' estendibile a "
        "capitali istituzionali.",
    ),
))

_add("monthly_heatmap", Block(
    title="Heatmap dei rendimenti mensili",
    cosa="Ogni cella e' il rendimento di un mese; le righe sono gli anni.",
    lettura=(
        "Cerca i **cluster di rosso**: le perdite si concentrano o sono sparse? "
        "Perdite concentrate significano un fattore di rischio specifico che si "
        "attiva in certi regimi.",
        "Le colonne evidenziano eventuali **effetti stagionali**. Trattali con "
        "sospetto: su 26 osservazioni per mese, quasi ogni pattern e' rumore.",
        "I singoli mesi estremi vanno annotati e confrontati con la cronologia di "
        "mercato: aprile-maggio 2009 e novembre 2020 sono i crolli di momentum "
        "piu' noti.",
    ),
    attenzione=(
        "Se togliendo i 5 mesi migliori il vantaggio sparisce, **il vantaggio non "
        "esiste**: e' un premio per il rischio di coda concentrato in pochi "
        "eventi, non un edge ripetibile. Il pannello riporta questo calcolo.",
    ),
))

_add("horizon_grid", Block(
    title="Griglia orizzonti (controllo di overfitting)",
    cosa=(
        "Lo stesso studio ripetuto con holding di 1, 3 e 6 mesi, ciascuno con il "
        "lookback congruente al proprio holding. Piu' le altre combinazioni di "
        "parametri esplorate nella sidebar."
    ),
    lettura=(
        "**Guarda la dispersione, non il massimo.** Se la configurazione "
        "preregistrata (holding 3 mesi, 30 nomi, filtro attivo) si trova nella "
        "parte centrale della distribuzione dei risultati, il risultato e' "
        "onesto. Se e' il massimo assoluto della griglia, hai fatto data mining.",
        "Un segnale reale produce risultati **della stessa direzione** su tutti e "
        "tre gli orizzonti, con intensita' diversa. Un segnale che funziona solo "
        "a 3 mesi e si inverte a 1 e a 6 mesi e' un artefatto.",
        "L'holding a **1 mese** e' l'unico su cui l'inversione di brevissimo "
        "termine (comprare chi ha appena perso) puo' manifestarsi. Se la risposta "
        "cambia segno tra 1 e 3 mesi, hai trovato la cosa piu' interessante di "
        "tutto lo studio.",
    ),
    attenzione=(
        "La configurazione **preregistrata** e' evidenziata. E' quella dichiarata "
        "prima di guardare i risultati: le altre sono esplorazione e vanno "
        "riportate come tali.",
        "Ogni parametro che muovi moltiplica i test impliciti. Con 20 "
        "combinazioni provate, un p-value di 0.05 su una di esse e' il risultato "
        "atteso dal puro caso.",
    ),
))

_add("delisting_stress", Block(
    title="Stress test sul survivorship bias",
    cosa=(
        "Il backtest ripetuto imponendo un rendimento terminale di -30%, -50% e "
        "-100% ai titoli che escono dall'indice e per i quali non abbiamo un "
        "prezzo di uscita affidabile."
    ),
    lettura=(
        "**Questo e' il test piu' importante della pagina.** I titoli che "
        "spariscono sono sistematicamente concentrati nel paniere Fondo Griglia: "
        "se i loro prezzi mancano, il backtest sovrastima la strategia contrarian "
        "e falsifica proprio la risposta che stiamo cercando.",
        "**Direzione del bias: a favore della tesi contrarian.** Senza questo "
        "stress test, 'compra la debolezza' potrebbe sembrare vincente solo "
        "perche' i fallimenti sono invisibili.",
        "Se la conclusione **regge anche a -100%**, e' robusta. Se si ribalta gia' "
        "a -30%, il risultato dipende interamente dalla qualita' dei dati sui "
        "delistati e va dichiarato come non conclusivo.",
        "Il **coverage ratio** nella pagina Diagnostica dice quanti titoli sono "
        "effettivamente interessati. Se e' sopra il 98%, lo stress test e' quasi "
        "irrilevante; se scende sotto il 90% in qualche anno, e' decisivo.",
    ),
    attenzione=(
        "Lo scenario a 0% (liquidazione all'ultimo prezzo noto) e' **ottimistico**: "
        "assume che l'ultimo prezzo disponibile sia un prezzo a cui avresti "
        "davvero potuto vendere. Per un titolo sospeso prima del fallimento non e' "
        "cosi'.",
        "Le uscite per fusione o acquisizione avvengono in genere **a premio**: "
        "trattarle come -30% le penalizza ingiustamente. Non distinguiamo le due "
        "cause, quindi lo stress e' volutamente conservativo.",
    ),
))

_add("longshort", Block(
    title="Spread Testa Corsa - Fondo Griglia",
    cosa=(
        "La differenza di rendimento mensile tra il paniere forte e quello "
        "debole, con il suo t-statistic corretto per autocorrelazione."
    ),
    lettura=(
        "Lo spread **rimuove il beta di mercato**: se entrambi i panieri salgono "
        "del 20% perche' sale tutto, lo spread e' zero. Quello che resta e' "
        "l'effetto puro della selezione.",
        "Il **t-stat Newey-West** corregge per il fatto che i rendimenti mensili "
        "di portafogli con holding a 3 mesi sono autocorrelati per costruzione: "
        "senza correzione il t-stat e' gonfiato.",
        "|t| > 2 e' l'indicazione convenzionale di significativita'. Su dati "
        "finanziari con selezione di parametri, molti ricercatori chiedono |t| > 3.",
    ),
    attenzione=(
        "Lo spread e' un **costrutto long-short**, ma i due panieri sono "
        "simulati long-only. Non e' un portafoglio implementabile: non include "
        "costi e vincoli dello short selling (prestito titoli, richiami, "
        "impossibilita' di shortare i nomi piu' stressati proprio quando conta).",
    ),
))

# ===========================================================================
# PAGINA 3 - DIAGNOSTICA DATI
# ===========================================================================

_add("coverage", Block(
    title="Coverage ratio dei prezzi",
    cosa=(
        "Per ogni mese: la frazione dei titoli presenti nella lista storica dei "
        "costituenti S&P 500 per i quali abbiamo effettivamente una serie prezzi."
    ),
    lettura=(
        "**Questo grafico determina quanto vale tutto il resto della dashboard.** "
        "Un coverage del 100% significa survivorship bias risolto; ogni punto "
        "sotto il 100% e' un titolo che scompare in silenzio dal backtest.",
        "**Sopra il 98%**: il backtest e' affidabile. **95-98%**: leggi i "
        "risultati con lo stress test a fianco. **Sotto il 90%**: il periodo non "
        "e' utilizzabile per conclusioni.",
        "Il pannello per fascia mostra dove si concentrano i buchi. Se il "
        "coverage e' peggiore in Fondo Griglia — e lo sara' — il bias spinge il "
        "risultato a favore della tesi contrarian.",
    ),
    attenzione=(
        "EODHD copre i costituenti storici dell'S&P 500 **da gennaio 2000** e i "
        "prezzi US prevalentemente dalla stessa data. Prima del 2000 non c'e' "
        "backtest possibile con questa fonte: la salita della bolla dot-com "
        "(1995-1999) e' fuori portata, lo scoppio (2000-2002) no.",
        "Un coverage alto non garantisce prezzi **corretti**. La pagina riporta "
        "anche il conteggio di rendimenti giornalieri anomali (|r| > 60%), che "
        "segnalano split non gestiti o errori di dato.",
    ),
))

_add("universe_size", Block(
    title="Dimensione dell'universo eleggibile",
    cosa=(
        "Quanti titoli superano ogni mese tutti i filtri: presenza nell'indice, "
        "serie prezzi disponibile, storia sufficiente, prezzo sotto il cap, e "
        "(se attivo) prezzo sopra la media a 200 sedute."
    ),
    lettura=(
        "**Il filtro a 200 sedute e' prociclico.** Nei bear market l'universo "
        "collassa: a marzo 2009 o a ottobre 2002 pochissimi titoli erano sopra la "
        "loro media a 200 sedute.",
        "Quando l'universo scende sotto ~150 titoli, i quintili contengono meno "
        "di 30 nomi e il portafoglio prende **l'intera fascia**: la selettivita' "
        "sparisce proprio nei mesi che decidono il risultato.",
        "Sotto la soglia minima (50 titoli) il mese e' marcato e va letto con "
        "sospetto: il portafoglio diventa concentrato e idiosincratico.",
    ),
    attenzione=(
        "**Il filtro a 200 sedute non e' neutrale: e' gia' una scommessa sul "
        "momentum.** Applicandolo, la domanda dello studio smette di essere "
        "'momentum o debolezza' e diventa 'leader o ritracciamento dentro un "
        "trend'. Usa il toggle per eseguire il backtest anche senza filtro: la "
        "differenza tra i due e' l'effetto del filtro, isolato.",
    ),
))

_add("price_cap_exclusions", Block(
    title="Titoli esclusi dal cap sul prezzo",
    cosa=(
        "Quanti e quali titoli vengono scartati ogni mese perche' il prezzo per "
        "azione supera 1.500$ e non entrerebbero in uno slot da 3.333$ con lotti "
        "interi."
    ),
    lettura=(
        "Il numero e' normalmente piccolo (pochi nomi su 500), ma **non sono nomi "
        "casuali**: i titoli a prezzo unitario molto alto sono tipicamente "
        "compounder di lungo periodo che non hanno mai splittato.",
        "**Direzione del bias: contro il paniere Testa Corsa.** E' l'opposto del "
        "bias dei delistati, quindi i due si compensano parzialmente.",
    ),
    attenzione=(
        "Il cap e' una conseguenza diretta della scelta di capitale (100.000$) e "
        "di numero di titoli (30). Con piu' capitale o meno titoli il cap "
        "morderebbe meno.",
    ),
))

_add("turnover", Block(
    title="Turnover e costi realizzati",
    cosa=(
        "La frazione di portafoglio effettivamente sostituita a ogni "
        "ribilanciamento e i dollari pagati in commissioni e spread."
    ),
    lettura=(
        "Il turnover **misurato** sostituisce le stime a priori. Serve a "
        "verificare che l'ipotesi di costo usata nel dimensionamento (70% "
        "mensile) fosse realistica.",
        "Confronta il turnover di Testa Corsa e Fondo Griglia: se sono molto "
        "diversi, i costi **non si cancellano** nel confronto tra i due panieri e "
        "una parte della differenza di rendimento e' solo attrito.",
        "I costi totali in dollari su 26 anni, rapportati a 100.000$ di capitale, "
        "sono il numero piu' comunicativo della tabella.",
    ),
    attenzione=(
        "La regola di ribilanciamento e' **conservativa sui costi**: vendiamo solo "
        "i titoli usciti dal target e compriamo solo i nuovi entranti, senza "
        "ripesare le posizioni sopravvissute. Un ribilanciamento completo a equal "
        "weight costerebbe sensibilmente di piu'.",
    ),
))

_add("data_quality", Block(
    title="Anomalie di dato",
    cosa=(
        "Conteggio dei rendimenti giornalieri superiori al 60% in valore "
        "assoluto, per anno e per titolo."
    ),
    lettura=(
        "Un rendimento del +200% in una seduta e' quasi sempre uno **split non "
        "gestito** nella serie adjusted, non un evento reale.",
        "Anomalie concentrate su un titolo indicano una serie da scartare. "
        "Anomalie concentrate su un anno indicano un problema sistematico di "
        "fonte in quel periodo.",
    ),
    attenzione=(
        "Facciamo tutta la contabilita' su **adjusted close** (dividendi "
        "reinvestiti, split gestiti) e usiamo il prezzo **grezzo** solo per "
        "decidere quante azioni comprare e per applicare il cap. Mischiare i due "
        "e' l'errore classico che gonfia i rendimenti dei titoli ad alto "
        "dividendo.",
    ),
))

# ===========================================================================
# EXPORT
# ===========================================================================

_add("json_export", Block(
    title="Export JSON per analisi esterna",
    cosa=(
        "Un file JSON strutturato con configurazione, provenienza dei dati, "
        "snapshot corrente, risultati del backtest, distribuzione nulla e — "
        "soprattutto — l'elenco esplicito dei limiti metodologici con la loro "
        "direzione."
    ),
    lettura=(
        "**compact** (~200-500 KB) contiene tutto tranne le serie mensili "
        "complete: e' il formato da incollare in una conversazione con un LLM.",
        "**full** contiene anche tutte le serie storiche: serve per rianalisi "
        "programmatica.",
        "Il campo `caveats` elenca ogni bias noto con `direction_of_bias` e una "
        "stima di magnitudine. E' il campo piu' importante del file: costringe "
        "chi legge a considerare i limiti invece di prendere i numeri per oro "
        "colato.",
        "Scarica anche `README_SCHEMA.md`: descrive ogni campo, cosi' un LLM non "
        "deve indovinare il significato delle chiavi.",
    ),
    attenzione=(
        "Il JSON contiene i risultati della **configurazione attualmente "
        "selezionata**, non della preregistrata, a meno che coincidano. Il campo "
        "`config.is_preregistered` lo dichiara esplicitamente.",
    ),
))


# ===========================================================================
# Rendering
# ===========================================================================

def escape_markdown(text: str) -> str:
    """Neutralizza il simbolo di dollaro.

    Streamlit interpreta `$...$` come LaTeX: una frase con due importi in
    dollari perde tutto il testo compreso fra i due simboli. Con l'escape
    resta testo normale.
    """
    return text.replace("$", r"\$")


def to_markdown(key: str, escape: bool = True) -> str:
    """Costruisce il markdown del riquadro (usabile anche fuori da Streamlit)."""
    blk = BLOCKS[key]
    parts = [f"**Cosa mostra.** {blk.cosa}", "", "**Come si legge.**"]
    parts += [f"- {x}" for x in blk.lettura]
    parts += ["", "**Attenzione.**"]
    parts += [f"- {x}" for x in blk.attenzione]
    if blk.formula:
        parts += ["", "**Formula.**", "", blk.formula]
    out = "\n".join(parts)
    return escape_markdown(out) if escape else out


def render(key: str, expanded: bool = False) -> None:
    """Disegna il riquadro esplicativo in Streamlit."""
    import streamlit as st

    blk = BLOCKS[key]
    with st.expander(f"Come leggere: {blk.title}", expanded=expanded):
        st.markdown(to_markdown(key))


def render_inline(key: str) -> None:
    """Variante sempre aperta, per i grafici piu' delicati."""
    render(key, expanded=True)
