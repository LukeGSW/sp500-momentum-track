"""
Trova le serie prezzi compromesse e propone le righe per exclusions.csv.

    python -m pipeline.find_bad_series                 # solo diagnosi
    python -m pipeline.find_bad_series --write proposte.csv

Lavora sul dataset gia' costruito: nessuna chiamata API.

Le proposte NON vengono applicate da sole. Vanno lette, verificate una per una
e copiate a mano in `exclusions.csv` con un motivo scritto da te. Un'esclusione
automatica non verificata e' un dato manipolato: il file deve restare qualcosa
che puoi difendere riga per riga.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from track import exclusions as exc
from track import storage

log = logging.getLogger("find_bad_series")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Individua serie prezzi compromesse")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="soglia sul rendimento giornaliero assoluto")
    parser.add_argument("--divergence", type=float, default=0.25,
                        help="scarto massimo tollerato fra rendimento rettificato e grezzo")
    parser.add_argument("--padding", type=int, default=45,
                        help="giorni di margine intorno alle date incriminate")
    parser.add_argument("--all", action="store_true",
                        help="proponi anche i casi incerti, non solo quelli quasi certi")
    parser.add_argument("--write", default=None, help="salva le proposte in questo CSV")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if not storage.dataset_available(args.data_dir):
        log.error("dataset non trovato: esegui prima  python -m pipeline.build_dataset")
        return 2

    close_adj = storage.load_panel("close_adj", args.data_dir)
    try:
        close_raw = storage.load_panel("close_raw", args.data_dir)
        log.info("prezzo grezzo disponibile: il test di divergenza e' attivo")
    except FileNotFoundError:
        close_raw = None
        log.warning("close_raw assente (dataset costruito con una versione precedente): "
                    "il test piu' affidabile, quello sulla divergenza fra prezzo "
                    "rettificato e grezzo, non puo' essere eseguito. Ricostruisci il "
                    "dataset per attivarlo.")

    det = exc.detect_broken_series(
        close_adj, close_raw,
        return_threshold=args.threshold,
        divergence_threshold=args.divergence,
    )

    if det.empty:
        print("\nNessuna serie sospetta con queste soglie.")
        return 0

    certi = det[det["quasi_certo"]]
    print(f"\n{'=' * 78}")
    print(f"  {len(det)} serie sospette, di cui {len(certi)} quasi certamente rotte")
    print(f"{'=' * 78}\n")
    print("  'quasi certo' = presenta divergenza fra prezzo rettificato e grezzo,")
    print("  oppure salti che rientrano il giorno dopo. Entrambi sono firme di")
    print("  errori di dato, non di eventi di mercato.\n")

    mostra = det.head(args.top).copy()
    mostra["prima_data"] = mostra["prima_data"].dt.date
    mostra["ultima_data"] = mostra["ultima_data"].dt.date
    mostra["peggior_rendimento"] = mostra["peggior_rendimento"].map("{:.0%}".format)
    print(mostra.to_string(index=False))

    prop = exc.suggest_exclusions(det, padding_days=args.padding, only_certain=not args.all)
    if prop.empty:
        print("\nNessuna proposta di esclusione: i casi trovati sono tutti incerti.")
        print("Rilancia con --all per vederli comunque.")
        return 0

    print(f"\n{'=' * 78}")
    print(f"  PROPOSTE ({len(prop)} righe) — da verificare a mano prima di usarle")
    print(f"{'=' * 78}\n")
    print(prop.to_csv(index=False))

    gia_escluse = {e.ticker for e in exc.load_exclusions()}
    nuove = prop[~prop["ticker"].isin(gia_escluse)]
    if len(nuove) < len(prop):
        print(f"  ({len(prop) - len(nuove)} gia' presenti in exclusions.csv)")

    if args.write:
        prop.to_csv(args.write, index=False)
        print(f"\n  Proposte salvate in {args.write}")

    print("\n  PROSSIMO PASSO: verifica ogni riga (la pagina Diagnostica mostra i")
    print("  prezzi intorno alla data), poi copia in exclusions.csv SOLO quelle che")
    print("  sai difendere, con un motivo scritto da te. Infine ricostruisci:")
    print("      python -m pipeline.build_dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
