"""
Scarica in locale il dataset gia' costruito dalla pipeline e pubblicato come
asset di una Release GitHub.

    python -m pipeline.fetch_dataset --repo tuo-utente/tuo-repository

Utile per lavorare in locale sui dati reali senza rifare il download da EODHD
(~1.200 chiamate API). Nessuna chiave necessaria se il repository e' pubblico.
"""
from __future__ import annotations

import argparse
import logging
import sys

from track import storage

log = logging.getLogger("fetch_dataset")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scarica il dataset pubblicato")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="owner/repository: prende l'asset dell'ultima Release")
    src.add_argument("--url", help="URL diretto di un archivio .tar.gz")
    parser.add_argument("--asset", default=storage.DEFAULT_ASSET)
    parser.add_argument("--token", default=None, help="serve solo per repository privati")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force", action="store_true",
                        help="riscarica anche se i pannelli sono gia' presenti")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")

    if storage.dataset_available(args.data_dir) and not args.force:
        log.info("dataset gia' presente in %s (usa --force per riscaricarlo)",
                 storage.data_dir(args.data_dir))
        return 0

    url = args.url or storage.release_asset_url(args.repo, args.asset)
    try:
        names = storage.download_and_extract(url, args.data_dir, args.token)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 3
    except Exception as exc:  # noqa: BLE001
        log.error("download fallito: %s: %s", type(exc).__name__, exc)
        return 2

    log.info("estratti: %s", ", ".join(sorted(names)))

    missing = storage.missing_panels(args.data_dir)
    if missing:
        log.error("archivio incompleto, mancano ancora: %s", ", ".join(missing))
        return 4

    man = storage.load_manifest(args.data_dir)
    log.info("dataset pronto — fonte: %s, periodo %s → %s",
             man.get("source", "?"), man.get("first_data_date", "?"),
             man.get("last_data_date", "?"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
