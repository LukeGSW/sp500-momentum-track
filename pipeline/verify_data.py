"""
Verifica preliminare dei dati. ESEGUIRE QUESTO PER PRIMO.

    python -m pipeline.verify_data

Risponde alle domande che possono far fallire l'intero progetto, PRIMA di
scaricare 1.200 serie storiche:

  1. La lista dei costituenti storici e' ottenibile? Da quando?
  2. Esistono i prezzi dei titoli usciti dall'indice, o solo dei sopravvissuti?
  3. Quanto indietro arrivano davvero i prezzi?
  4. I settori sono recuperabili per le societa' uscite?
  5. Il simbolo del tasso privo di rischio e' disponibile?

La domanda 2 e' la piu' importante: i titoli che spariscono si concentrano nel
paniere piu' debole, quindi ogni buco spinge il risultato a favore della tesi
contrarian. Se la copertura sui delistati e' scarsa, il backtest non puo'
rispondere alla domanda dello studio e va detto subito.

NOTA: la fonte predefinita dei costituenti NON e' EODHD. L'endpoint
`fundamentals/GSPC.INDX` e' incluso solo in alcuni piani; la ricostruzione MIT
di fja05680/sp500 non richiede entitlement e parte dal 1996 invece che dal 2000.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from track import constituents as ct
from track import universe
from track.config import PREREGISTERED
from track.eodhd import EODHDClient, EODHDError, resolve_api_key

log = logging.getLogger("verify")

SEP = "=" * 78


def _title(text: str) -> None:
    print(f"\n{SEP}\n  {text}\n{SEP}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verifica la fattibilita' dei dati")
    parser.add_argument("--sample", type=int, default=40,
                        help="quanti titoli usciti campionare per il test dei prezzi")
    parser.add_argument("--constituents-source", choices=[ct.SOURCE_GITHUB, ct.SOURCE_EODHD],
                        default=ct.SOURCE_GITHUB)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
    cfg = PREREGISTERED

    try:
        client = EODHDClient(resolve_api_key(), max_workers=6)
    except RuntimeError as exc:
        print(f"\n✗ {exc}")
        return 2

    blockers: list[str] = []
    warnings_: list[str] = []

    # =======================================================================
    _title("1. Costituenti storici S&P 500")
    try:
        raw = ct.load_constituents(args.constituents_source, client=client)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ FALLITO ({args.constituents_source}): {exc}")
        if args.constituents_source == ct.SOURCE_EODHD:
            print("\n  L'endpoint sugli indici e' incluso solo in alcuni piani EODHD.")
            print("  Riprova con la fonte predefinita, che non richiede entitlement:")
            print("      python -m pipeline.verify_data --constituents-source github")
        return 3

    const = universe.normalize_constituents(raw)
    print(f"✓ fonte '{args.constituents_source}': {len(const):,} periodi di appartenenza, "
          f"{const['code'].nunique():,} ticker distinti")
    print(f"  Copertura              : {const['start_date'].min().date()} → "
          f"{const['end_date'].max().date()}")
    print(f"  Membri attuali         : {int(const['is_active_now'].sum()):,}")

    if args.constituents_source == ct.SOURCE_GITHUB:
        print("\n  ⚠ E' una RICOSTRUZIONE di terze parti (fja05680/sp500, MIT), non un")
        print("    dato ufficiale S&P. Le ricostruzioni sbagliano, e l'errore si")
        print("    accumula andando indietro: i primi anni sono i meno affidabili.")
        warnings_.append("costituenti da ricostruzione non ufficiale")

    # completezza per anno: quanti titoli risultano nell'indice
    cal = pd.bdate_range(const["start_date"].min(), const["end_date"].max())
    memb = universe.build_membership(const, cal)
    per_year = memb.sum(axis=1).groupby(memb.index.year).median().astype(int)
    print("\n  Titoli nell'indice per anno (mediana) — dovrebbero essere ~500:")
    for year, n in per_year.items():
        flag = "  ← sottoconteggio" if n < 480 else ""
        print(f"    {year}: {'█' * (n // 12)} {n}{flag}")

    thin = per_year[per_year < 480]
    if len(thin):
        print(f"\n  ⚠ {len(thin)} anni sotto i 480 titoli ({thin.index.min()}-{thin.index.max()}).")
        print("    In quegli anni l'universo e' incompleto: il backtest resta valido")
        print("    ma misura un sottoinsieme dell'indice, non l'indice.")
        warnings_.append(f"universo incompleto {thin.index.min()}-{thin.index.max()}")

    reused = sorted(const.loc[const["ticker_reuse_suspect"], "code"].unique())
    dropped = len(const) - len(const[~const["ticker_reuse_suspect"] | const["is_latest_occurrence"]])
    print(f"\n  Codici con piu' periodi o piu' nomi: {len(reused)} "
          f"({dropped} periodi scartati per non incollare due aziende)")

    # =======================================================================
    _title("2. Prezzi dei titoli usciti dall'indice — LA VERIFICA CRITICA")
    exits = const[(const["is_active_now"] == 0) & const["is_latest_occurrence"]]
    print(f"  Titoli usciti dall'indice: {len(exits):,}")
    print("  Sono questi quelli il cui prezzo DEVE esistere per non avere")
    print("  survivorship bias.\n")

    sample = exits.sample(min(args.sample, len(exits)), random_state=cfg.seed)
    symbols = [f"{c}.US" for c in sample["code"]]
    print(f"  Campiono {len(symbols)} titoli…")
    series = client.eod_many(symbols, cfg.download_start)

    rows = []
    for code, start, end in zip(sample["code"], sample["start_date"],
                                sample["end_date"], strict=True):
        df = series.get(f"{code}.US")
        if df is None or df.empty:
            rows.append({"code": code, "esito": "NESSUN DATO", "ultimo": None, "ok": False})
            continue
        last = df.index.max()
        covers = last >= (end - pd.Timedelta(days=30))
        rows.append({"code": code, "esito": "ok" if covers else "SERIE TRONCATA",
                     "ultimo": last.date(), "ok": covers})

    check = pd.DataFrame(rows)
    ok = int(check["ok"].sum())
    rate = ok / max(len(check), 1)

    print(f"\n  Serie utilizzabili: {ok}/{len(check)}  ({rate:.1%})")
    bad = check[~check["ok"]]
    if not bad.empty:
        print("\n  Titoli problematici:")
        for _, r in bad.head(20).iterrows():
            print(f"    {r['code']:<8} {r['esito']:<16} ultimo dato: {r['ultimo']}")

    print()
    if rate >= 0.98:
        print("  ✓✓ COPERTURA ECCELLENTE. Il survivorship bias e' sotto controllo.")
    elif rate >= 0.90:
        print("  ✓ COPERTURA ACCETTABILE. Leggere sempre i risultati insieme allo")
        print("    stress test sui delistati: il bias residuo favorisce la tesi")
        print("    contrarian (il paniere Q1 Laggard perde i suoi peggiori nomi).")
        warnings_.append(f"copertura prezzi sui delistati al {rate:.0%}")
    else:
        print("  ✗✗ COPERTURA INSUFFICIENTE.")
        print("     I titoli mancanti si concentrano nel paniere debole e ne gonfiano")
        print("     artificialmente il rendimento: il backtest NON puo' rispondere")
        print("     alla domanda dello studio. Limitare le conclusioni allo scenario")
        print("     di stress a -100%, oppure procurarsi i dati sui delistati.")
        blockers.append(f"copertura prezzi sui delistati troppo bassa ({rate:.0%})")

    # =======================================================================
    _title("3. Quanto indietro arrivano i prezzi?")
    print("  Verifico da che anno esistono davvero i dati, su un campione di")
    print("  titoli tuttora nell'indice.\n")
    alive = const[const["is_active_now"] == 1].sample(12, random_state=cfg.seed)
    alive_series = client.eod_many([f"{c}.US" for c in alive["code"]], cfg.download_start)

    firsts = []
    for code in alive["code"]:
        df = alive_series.get(f"{code}.US")
        if df is not None and not df.empty:
            firsts.append((code, df.index.min().date()))
    for code, d in sorted(firsts, key=lambda x: x[1])[:12]:
        print(f"    {code:<8} primo dato: {d}")

    if firsts:
        earliest = min(d for _, d in firsts)
        print(f"\n  Dato piu' vecchio nel campione: {earliest}")
        if earliest.year <= 1997:
            print("  → Si puo' provare a far partire il backtest dal 1998-1999 e")
            print("    includere la SALITA della bolla dot-com, non solo lo scoppio.")
            print("    Serve il warm-up: il primo segnale valido arriva ~15 mesi dopo.")
        else:
            print(f"  → Il backtest non puo' partire prima del {earliest.year + 2} circa")
            print("    (servono ~15 mesi di warm-up per gli orizzonti a 252 sedute).")

    # =======================================================================
    _title("4. Settori")
    known = int(const["sector"].notna().sum())
    print(f"  Settore noto per {known}/{len(const)} periodi ({known/max(len(const),1):.0%}).")
    if known / max(len(const), 1) < 0.9:
        print("\n  I settori arrivano dalla lista CORRENTE: le societa' uscite")
        print("  dall'indice restano 'Non classificato'. Conseguenze:")
        print("    • sulla mappa la colonna settoriale 'Non classificato' sara' affollata nelle")
        print("      viste storiche (la vista di oggi e' completa)")
        print("    • l'opzione sector-neutral e' inaffidabile sui backtest lunghi")
        print("\n  Provo a recuperarli dai Fundamentals per singolo titolo…")
        probe = const.loc[const["sector"].isna(), "code"].dropna().unique()[:3]
        got = 0
        for code in probe:
            try:
                val = client._get(f"fundamentals/{code}.US", {"filter": "General::Sector"})
                if isinstance(val, str) and val.strip():
                    print(f"    ✓ {code}: {val.strip()}")
                    got += 1
            except EODHDError as exc:
                print(f"    ✗ {code}: {exc}")
                break
        if got:
            print("\n  → Disponibili. Aggiungi --enrich-sectors a build_dataset")
            print("    (~700 chiamate in piu', una volta sola).")
        else:
            print("\n  → Non disponibili con questo piano: i settori storici restano vuoti.")
            warnings_.append("settori assenti per le societa' uscite dall'indice")

    # =======================================================================
    _title("5. Tasso privo di rischio")
    rf = client.risk_free_daily(cfg.download_start)
    src = rf.attrs.get("source", "?")
    if str(src).startswith("FALLBACK"):
        print("  ⚠ Nessun simbolo risk-free trovato: verra' usata una costante al 2%.")
        print("    Sharpe ratio e remunerazione della liquidita' saranno approssimati.")
        warnings_.append("tasso privo di rischio non disponibile")
    else:
        annual = ((1 + rf) ** 252 - 1) * 100
        print(f"  ✓ Simbolo utilizzato: {src}")
        print(f"    {len(rf):,} osservazioni, da {rf.index.min().date()} a {rf.index.max().date()}")
        print(f"    Tasso annuo implicito: min {annual.min():.2f}%  "
              f"mediana {annual.median():.2f}%  max {annual.max():.2f}%")

    # =======================================================================
    _title("VERDETTO")
    if blockers:
        print("  ✗ PROBLEMI BLOCCANTI:")
        for b in blockers:
            print(f"      • {b}")
    if warnings_:
        print("  ⚠ DA TENERE PRESENTE (finiscono nei caveats dell'export):")
        for w in warnings_:
            print(f"      • {w}")
    if not blockers:
        print("\n  Puoi procedere:")
        print(f"      python -m pipeline.build_dataset --constituents-source {args.constituents_source}")
    print()
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
