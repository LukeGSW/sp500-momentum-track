# 🏁 La Pista

**Conviene comprare azioni in forte momentum, o in debolezza momentanea?**

Uno strumento di misura sull'S&P 500. Non un indicatore: uno studio con
un'ipotesi nulla esplicita, uno stress test sul survivorship bias e l'elenco
dichiarato dei propri limiti.

---

## L'idea

I titoli vengono disposti lungo un **asse verticale di forza relativa**: in alto
i forti, in basso i deboli, misurati confrontandoli fra loro e non in valore
assoluto. Le undici colonne sono i settori GICS.

Il momentum **non è un secondo asse**: è la pendenza della scia, cioè il
movimento verticale del titolo nelle settimane precedenti. Su una pista dritta
chi sale sta migliorando e chi scende sta peggiorando — non serve altro.

Da questa geometria discende il punto che rende lo studio possibile: la fascia
più bassa e quella più alta corrispondono **letteralmente** alla domanda, e i
due panieri si possono confrontare.

Cinque fasce, dal basso: **Fondo Griglia · Rimonta · Gruppo · Scia · Testa
Corsa**. Sono quintili trasversali, cioè un ordinamento, non soglie assolute.

---

## Avvio rapido

```bash
git clone <questo-repo> && cd sp500-momentum-track
python -m venv .venv && .venv/Scripts/activate      # su Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Provare l'interfaccia senza chiave API

```bash
python -m pipeline.make_demo_data
streamlit run app.py
```

Genera un universo **sintetico**: serve solo a vedere come funziona la
dashboard. Ogni numero è finto e l'app lo dichiara con un banner.

### Dati reali EODHD

Metti la chiave in `.streamlit/secrets.toml` (parti da
`.streamlit/secrets.toml.example`) oppure nella variabile d'ambiente
`EODHD_API_KEY`, poi:

```bash
python -m pipeline.verify_data
```

**Esegui sempre questo per primo.** Verifica in pochi minuti se il tuo piano
EODHD rende lo studio fattibile — soprattutto se esistono i prezzi dei titoli
*delistati*, che è la condizione senza la quale il backtest non può rispondere
alla domanda. Poi:

```bash
python -m pipeline.build_dataset
streamlit run app.py
```

Il download completo è di circa 1.100–1.300 chiamate API e richiede alcuni
minuti.

---

## Architettura

```
pipeline/  (offline, con chiave API)  ──►  data/*.parquet  ──►  app.py (nessuna rete)
```

L'app **non chiama mai EODHD a runtime**. Due conseguenze pratiche: il deploy
pubblico funziona senza chiave, e nessun visitatore consuma la tua quota.

| | |
|---|---|
| `track/features.py` | Forza F, Spinta V, fasce. Modulo puro, nessun I/O |
| `track/backtest.py` | Motore a tranche sfalsate, costi, ipotesi nulla |
| `track/universe.py` | Costituenti storici, coverage, ticker riassegnati |
| `track/didactics.py` | I riquadri esplicativi di ogni grafico e tabella |
| `track/study.py` | Orchestrazione, condivisa fra app e pipeline |
| `pipeline/verify_data.py` | La verifica di fattibilità da eseguire per prima |

Gli artefatti Parquet **non vanno committati** (bloat + limite di 100 MB per
file di GitHub): il workflow in `.github/workflows/refresh.yml` li pubblica
come asset di una Release.

---

## Le scelte metodologiche, in breve

| Scelta | Perché |
|---|---|
| Standardizzazione **trasversale** (mediana + MAD) | Confronta i titoli fra loro alla stessa data. La normalizzazione in serie storica per singolo titolo non è confrontabile su 500 nomi |
| Il benchmark **si cancella** | Il rendimento di mercato è identico per tutti alla stessa data, quindi sparisce nello z-score. SPY o RSP è indifferente — c'è un test che lo verifica |
| Holding 3 mesi su **3 tranche sfalsate** | Costi da ribilanciamento trimestrale, ma 12 decisioni l'anno: ~316 osservazioni mensili invece di ~105, e nessun timing luck |
| 30 titoli, slot da 3.333 USD | La commissione fissa pesa 4,5 bps per lato invece dei 9,8 che peserebbe con 65 titoli |
| Segnale a *t*, esecuzione a *t+1* in apertura | Calcolare ed eseguire sulla stessa chiusura è anticipazione mascherata |
| Contabilità in *adjusted*, lotti sul prezzo *grezzo* | Mischiarli gonfia i titoli ad alto dividendo |
| Ipotesi nulla = **estrazione casuale**, non SPY | Isola la selezione per fascia dall'equal weighting, dal filtro e dalla composizione dell'universo |

Il dettaglio completo è nella pagina **Metodologia** della dashboard.

---

## Il limite che conta

Sapere chi era nell'indice a una certa data risolve **metà** del problema di
survivorship. L'altra metà sono i prezzi di chi poi è sparito — e i titoli che
spariscono si concentrano nel paniere più debole.

> **Direzione del bias: a favore della tesi contrarian.** Senza correzione,
> "compra la debolezza" potrebbe sembrare vincente solo perché i fallimenti sono
> invisibili.

Per questo la pagina Backtest ripete lo studio imponendo un rendimento
terminale di −30%, −50% e −100% ai titoli che spariscono senza prezzo di uscita.
**Se la conclusione regge a −100%, è robusta.** In caso contrario va dichiarata
non conclusiva.

Gli altri limiti noti, ciascuno con la propria direzione di distorsione, sono
nel campo `caveats` dell'export JSON e nella pagina Metodologia.

Due vincoli di dato da tenere presenti:

- La copertura EODHD dei costituenti storici parte da **gennaio 2000**. Lo
  scoppio della bolla dot-com è coperto, la salita 1995-1999 no.
- I costi di **oggi** sono applicati a tutto lo storico, per scelta esplicita.
  Lo studio risponde a *"cosa avrebbe prodotto questo segnale con gli attriti di
  oggi"*, non a *"cosa potevi fare nel 2001"* — quando le commissioni retail
  erano di 10-30 USD per eseguito e questa strategia sarebbe stata
  impraticabile con 100.000 USD.

---

## Test

```bash
python -m pytest tests -q
```

Il test che conta è `test_no_lookahead`: verifica che il segnale calcolato su
dati troncati alla data *t* coincida **esattamente** con quello calcolato sul
dataset completo. Se fallisce, ogni numero prodotto dal backtest è privo di
significato.

---

## Deploy su Streamlit Cloud

1. Push del repository su GitHub.
2. Su [share.streamlit.io](https://share.streamlit.io) punta a `app.py`.
3. Nessun secret necessario: l'app non chiama EODHD.
4. Rendi disponibili i dati con una di queste due strade:
   - il workflow GitHub Actions pubblica una Release, e un piccolo passo di
     avvio la scarica in `data/`;
   - oppure, per un primo giro, esegui `make_demo_data` in locale e committa
     temporaneamente `data/` rimuovendolo da `.gitignore`.

---

## Nota su proprietà intellettuale

Nomenclatura, metrica e geometria sono originali. Non vengono usati né i nomi
né le formule degli strumenti rotazionali commerciali, che sono marchi
registrati dei rispettivi titolari.

---

## Avvertenza

Questo è uno strumento di misura e di ricerca. **Non è un consiglio di
investimento.** I risultati di un backtest, per quanto curati, non predicono
rendimenti futuri.
