"""
Verifica preliminare dei dati EODHD. ESEGUIRE QUESTO PER PRIMO.

    python -m pipeline.verify_data

Risponde alle domande che possono far fallire l'intero progetto, PRIMA di
scaricare 1.300 serie storiche:

  1. Il piano EODHD include i costituenti storici dell'S&P 500?
  2. Da che data partono davvero?
  3. Esistono i prezzi dei titoli DELISTATI, o solo dei sopravvissuti?
  4. `StartDate` e' la data di annuncio o quella di efficacia?
  5. Il simbolo del tasso privo di rischio e' disponibile?

La domanda 3 e' la piu' importante: i titoli che spariscono si concentrano nel
paniere piu' debole, quindi ogni buco spinge il risultato a favore della tesi
contrarian. Se la copertura sui delistati e' scarsa, il backtest non puo'
rispondere alla domanda dello studio e va detto subito.
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from track import universe
from track.config import PREREGISTERED
from track.eodhd import EODHDClient, EODHDError, resolve_api_key

log = logging.getLogger("verify")

SEP = "=" * 78


def _title(text: str) -> None:
    print(f"\n{SEP}\n  {text}\n{SEP}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verifica la fattibilita' dei dati EODHD")
    parser.add_argument("--sample", type=int, default=40,
                        help="quanti delistati campionare per il test dei prezzi")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
    cfg = PREREGISTERED

    try:
        client = EODHDClient(resolve_api_key(), max_workers=6)
    except RuntimeError as exc:
        print(f"\n✗ {exc}")
        return 2

    # -----------------------------------------------------------------------
    _title("1. Costituenti storici S&P 500")
    try:
        raw = client.historical_constituents()
    except EODHDError as exc:
        print(f"✗ FALLITO: {exc}")
        print("\n  Senza i costituenti storici non e' possibile correggere il")
        print("  survivorship bias. Verifica che il piano includa i Fundamentals")
        print("  per gli indici, oppure procurati la lista da un'altra fonte.")
        return 3

    const = universe.normalize_constituents(raw)
    print(f"✓ {len(const):,} record, {const['code'].nunique():,} codici distinti")
    print(f"  Prima data di ingresso : {const['start_date'].min().date()}")
    print(f"  Ultima data di uscita  : {const['end_date'].max().date()}")
    print(f"  Membri attuali         : {int(const['is_active_now'].sum()):,}")
    print(f"  Marcati come delistati : {int(const['is_delisted'].sum()):,}")

    first = const["start_date"].min()
    if first > pd.Timestamp("2000-06-30"):
        print(f"\n  ⚠ La copertura parte dal {first.date()}: piu' tardi di quanto")
        print("    atteso. Il backtest va fatto partire di conseguenza.")
    else:
        print("\n  → Backtest possibile dal 2000. Lo SCOPPIO della bolla dot-com")
        print("    (mar 2000 – ott 2002) e' coperto; la salita 1995-1999 no.")

    reused = sorted(const.loc[const["ticker_reuse_suspect"], "code"].unique())
    print(f"\n  Codici con possibile riassegnazione: {len(reused)}")
    if reused:
        print(f"    {', '.join(reused[:20])}{' …' if len(reused) > 20 else ''}")

    # -----------------------------------------------------------------------
    _title("2. Distribuzione temporale delle uscite dall'indice")
    exits = const[const["end_date"] < pd.Timestamp.today().normalize() - pd.Timedelta(days=5)]
    by_year = exits.groupby(exits["end_date"].dt.year).size()
    print("  Uscite per anno (le prime 15 righe):")
    for year, n in by_year.head(15).items():
        print(f"    {year}: {'█' * min(int(n), 50)} {n}")
    print(f"\n  Totale titoli usciti: {len(exits):,}")
    print("  Sono questi i titoli il cui prezzo DEVE esistere per non avere")
    print("  survivorship bias.")

    # -----------------------------------------------------------------------
    _title("3. Prezzi dei titoli usciti — LA VERIFICA CRITICA")
    sample = exits.sample(min(args.sample, len(exits)), random_state=cfg.seed)
    symbols = [f"{c}.US" for c in sample["code"]]
    print(f"  Campiono {len(symbols)} titoli usciti dall'indice…\n")

    series = client.eod_many(symbols, cfg.download_start)

    rows = []
    for code, name, start, end in zip(sample["code"], sample["name"],
                                      sample["start_date"], sample["end_date"], strict=True):
        df = series.get(f"{code}.US")
        if df is None or df.empty:
            rows.append({"code": code, "name": name, "esito": "NESSUN DATO",
                         "ultimo_prezzo": None, "copre_uscita": False})
            continue
        last = df.index.max()
        # il dato deve arrivare almeno fino a poco prima dell'uscita dall'indice
        covers = last >= (end - pd.Timedelta(days=30))
        rows.append({"code": code, "name": name,
                     "esito": "ok" if covers else "SERIE TRONCATA",
                     "ultimo_prezzo": last.date(), "copre_uscita": covers})

    check = pd.DataFrame(rows)
    ok = int(check["copre_uscita"].sum())
    rate = ok / max(len(check), 1)

    print(f"\n  Serie utilizzabili: {ok}/{len(check)}  ({rate:.1%})")
    bad = check[~check["copre_uscita"]]
    if not bad.empty:
        print("\n  Titoli problematici:")
        for _, r in bad.head(20).iterrows():
            print(f"    {r['code']:<8} {str(r['name'])[:36]:<36} {r['esito']}")

    print()
    if rate >= 0.98:
        print("  ✓✓ COPERTURA ECCELLENTE. Il survivorship bias e' sotto controllo.")
    elif rate >= 0.90:
        print("  ✓ COPERTURA ACCETTABILE. Leggere sempre i risultati insieme allo")
        print("    stress test sui delistati: il bias residuo favorisce la tesi")
        print("    contrarian (il paniere Fondo Griglia perde i suoi peggiori nomi).")
    else:
        print("  ✗✗ COPERTURA INSUFFICIENTE.")
        print("     Con questo tasso di buchi il backtest NON puo' rispondere alla")
        print("     domanda dello studio: i titoli mancanti si concentrano nel")
        print("     paniere debole e ne gonfiano artificialmente il rendimento.")
        print("     Opzioni: verificare il piano EODHD per i dati sui delistati,")
        print("     oppure limitare le conclusioni allo scenario di stress a -100%.")

    # -----------------------------------------------------------------------
    _title("4. StartDate: annuncio o efficacia?")
    print("  S&P annuncia le aggiunte ~1 settimana prima dell'efficacia. Se le date")
    print("  fossero quelle di ANNUNCIO, il backtest guadagnerebbe gratis il rialzo")
    print("  da inclusione nell'indice: un vantaggio non replicabile.\n")
    recent = const[const["start_date"] > pd.Timestamp("2015-01-01")].nlargest(5, "start_date")
    for _, r in recent.iterrows():
        print(f"    {r['code']:<8} {str(r['name'])[:40]:<40} ingresso {r['start_date'].date()}")
    print("\n  → Confronta queste date con i comunicati S&P Dow Jones Indices.")
    print("    Se coincidono con l'annuncio e non con l'efficacia, ritarda")
    print("    l'ingresso di 5 sedute in universe.build_membership().")

    # -----------------------------------------------------------------------
    _title("5. Tasso privo di rischio")
    rf = client.risk_free_daily(cfg.download_start)
    src = rf.attrs.get("source", "?")
    if str(src).startswith("FALLBACK"):
        print("  ⚠ Nessun simbolo risk-free trovato: verra' usata una costante al 2%.")
        print("    Sharpe ratio e remunerazione della liquidita' saranno approssimati.")
    else:
        annual = ((1 + rf) ** 252 - 1) * 100
        print(f"  ✓ Simbolo utilizzato: {src}")
        print(f"    {len(rf):,} osservazioni, da {rf.index.min().date()} a {rf.index.max().date()}")
        print(f"    Tasso annuo implicito: min {annual.min():.2f}%  "
              f"mediana {annual.median():.2f}%  max {annual.max():.2f}%")

    # -----------------------------------------------------------------------
    _title("VERDETTO")
    blockers, warnings_ = [], []
    if first > pd.Timestamp("2000-06-30"):
        warnings_.append(f"costituenti disponibili solo dal {first.date()}")
    if rate < 0.90:
        blockers.append(f"copertura prezzi sui delistati troppo bassa ({rate:.0%})")
    elif rate < 0.98:
        warnings_.append(f"copertura prezzi sui delistati al {rate:.0%}")
    if str(src).startswith("FALLBACK"):
        warnings_.append("tasso privo di rischio non disponibile")
    if len(reused) > 40:
        warnings_.append(f"{len(reused)} codici riassegnati da scartare")

    if blockers:
        print("  ✗ PROBLEMI BLOCCANTI:")
        for b in blockers:
            print(f"      • {b}")
    if warnings_:
        print("  ⚠ DA TENERE PRESENTE:")
        for w in warnings_:
            print(f"      • {w}")
    if not blockers and not warnings_:
        print("  ✓ Nessun problema rilevato. Procedi con:")
        print("      python -m pipeline.build_dataset")
    elif not blockers:
        print("\n  Puoi procedere. Esegui:")
        print("      python -m pipeline.build_dataset")
    print()
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
