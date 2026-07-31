"""
Persistenza degli artefatti.

L'app Streamlit legge SOLO da qui: nessuna chiamata di rete a runtime. La
pipeline offline scrive i Parquet, l'app li carica. I file non vanno
committati nel repository (bloat + limite 100 MB per file di GitHub): si
pubblicano come asset di una Release e si scaricano al primo avvio.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

MANIFEST_NAME = "manifest.json"

# Tutto cio' che `study.load_dataset` si aspetta di trovare. Serve anche a
# riconoscere un archivio troncato: senza `meta` o `risk_free` l'app
# fallirebbe piu' avanti con un errore molto meno chiaro.
PANELS = (
    "close_adj",
    "open_adj",
    "open_raw",
    "membership",
    "force",
    "velocity",
    "bands",
    "eligible",
    "meta",
    "risk_free",
)


def data_dir(override: str | Path | None = None) -> Path:
    p = Path(override) if override else DEFAULT_DATA_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
def read_secret(name: str) -> str | None:
    """Legge un segreto da ambiente o da un secrets.toml, senza Streamlit.

    Streamlit cerca `secrets.toml` a partire dalla directory di LAVORO, non da
    quella dello script: lanciando `streamlit run /percorso/assoluto/app.py` da
    un'altra cartella il file del progetto viene ignorato senza avvisi. Qui
    guardiamo entrambe le posizioni, cosi' il comportamento in locale non
    dipende da dove ci si trova.

    Su Streamlit Cloud i segreti arrivano dalla piattaforma e questa funzione
    non serve: e' `st.secrets` a risolverli, e viene provata per prima.
    """
    import os
    import tomllib

    val = os.environ.get(name, "").strip()
    if val:
        return val

    for candidate in (
        Path.cwd() / ".streamlit" / "secrets.toml",
        REPO_ROOT / ".streamlit" / "secrets.toml",
    ):
        try:
            if candidate.exists():
                # utf-8-sig, non utf-8: su Windows molti editor (e PowerShell
                # con -Encoding utf8) scrivono il BOM, che tomllib rifiuta con
                # un "Invalid statement at line 1, column 1" incomprensibile.
                data = tomllib.loads(candidate.read_text(encoding="utf-8-sig"))
                val = str(data.get(name, "")).strip()
                if val:
                    return val
        except (OSError, ValueError) as exc:
            log.warning("secrets.toml illeggibile in %s: %s", candidate, exc)
    return None


# ---------------------------------------------------------------------------
def save_panel(df: pd.DataFrame, name: str, directory: str | Path | None = None) -> Path:
    """Salva un pannello date x ticker in Parquet compresso."""
    path = data_dir(directory) / f"{name}.parquet"
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    # i booleani occupano meno come tali; i float li portiamo a 32 bit
    if out.dtypes.map(lambda d: d == bool).all():
        pass
    else:
        for c in out.columns:
            if str(out[c].dtype).startswith("float"):
                out[c] = out[c].astype("float32")
    out.to_parquet(path, compression="zstd")
    log.info("scritto %s (%s righe x %s colonne)", path.name, len(out), out.shape[1])
    return path


def load_panel(name: str, directory: str | Path | None = None) -> pd.DataFrame:
    path = data_dir(directory) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Pannello '{name}' non trovato in {path.parent}. "
            "Esegui prima  python -m pipeline.build_dataset"
        )
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def save_table(df: pd.DataFrame, name: str, directory: str | Path | None = None) -> Path:
    path = data_dir(directory) / f"{name}.parquet"
    df.to_parquet(path, compression="zstd")
    return path


def load_table(name: str, directory: str | Path | None = None) -> pd.DataFrame:
    path = data_dir(directory) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Tabella '{name}' non trovata in {path.parent}")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
def save_manifest(payload: dict, directory: str | Path | None = None) -> Path:
    path = data_dir(directory) / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_manifest(directory: str | Path | None = None) -> dict:
    path = data_dir(directory) / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_available(directory: str | Path | None = None) -> bool:
    d = data_dir(directory)
    return all((d / f"{n}.parquet").exists() for n in PANELS)


def missing_panels(directory: str | Path | None = None) -> list[str]:
    d = data_dir(directory)
    return [n for n in PANELS if not (d / f"{n}.parquet").exists()]


# ---------------------------------------------------------------------------
# Recupero del dataset pubblicato come Release
# ---------------------------------------------------------------------------
DEFAULT_ASSET = "la-mappa-data.tar.gz"


def release_asset_url(repo: str, asset: str = DEFAULT_ASSET) -> str:
    """URL stabile dell'asset dell'ultima Release.

    Questa forma viene redirezionata da GitHub alla Release piu' recente senza
    passare dalle API: evita il limite di 60 richieste/ora per IP non
    autenticate, che su un hosting condiviso come Streamlit Cloud si esaurisce
    in fretta.
    """
    return f"https://github.com/{repo.strip('/')}/releases/latest/download/{asset}"


def download_and_extract(
    url: str,
    directory: str | Path | None = None,
    token: str | None = None,
    timeout: int = 300,
) -> list[str]:
    """Scarica un .tar.gz e lo estrae nella cartella dati. Ritorna i file estratti."""
    import tarfile
    import tempfile

    import requests

    dest = data_dir(directory)
    headers = {"User-Agent": "la-mappa/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    log.info("scarico il dataset da %s", url)
    resp = requests.get(url, headers=headers, timeout=timeout,
                        allow_redirects=True, stream=True)
    if resp.status_code == 404:
        raise FileNotFoundError(
            f"Nessun asset trovato a {url}\n\n"
            "GitHub risponde 404 in quattro casi, tutti verificabili in un minuto:\n"
            "  1. il repository non ha ancora nessuna Release pubblicata\n"
            "  2. la Release esiste ma e' una bozza (draft): va pubblicata\n"
            "  3. il nome dell'asset e' diverso da quello atteso\n"
            "  4. il repository e' privato e manca il token di lettura"
        )
    if resp.status_code in (401, 403):
        raise PermissionError(
            f"Accesso negato ({resp.status_code}) a {url}. Se il repository e' "
            "privato serve un token con permesso di lettura nel secret DATA_TOKEN."
        )
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        with tarfile.open(tmp_path, "r:gz") as tar:
            # filter="data" blocca path traversal e link simbolici: l'archivio
            # arriva dalla rete e non va estratto alla cieca.
            tar.extractall(dest, filter="data")
            names = tar.getnames()
    finally:
        tmp_path.unlink(missing_ok=True)

    log.info("estratti %d file in %s", len(names), dest)
    return names


def ensure_dataset(
    directory: str | Path | None = None,
    *,
    url: str | None = None,
    repo: str | None = None,
    asset: str = DEFAULT_ASSET,
    token: str | None = None,
    force: bool = False,
) -> bool:
    """Se i pannelli mancano, prova a scaricarli. Ritorna True se ci sono.

    Serve al deploy: l'app non costruisce mai il dataset (richiederebbe la
    chiave API e minuti di download), ma puo' recuperare quello gia' costruito
    dalla pipeline e pubblicato come Release.

    `force=True` riscarica anche se i dati ci sono gia'. Senza questa opzione
    un container che ha gia' scaricato una Release continuerebbe a servirla per
    sempre: pubblicata una versione nuova, l'app non se ne accorgerebbe mai, e
    l'utente vedrebbe risultati vecchi credendoli aggiornati.
    """
    if dataset_available(directory) and not force:
        return True
    if not url and not repo:
        return False

    target = url or release_asset_url(repo, asset)  # type: ignore[arg-type]
    download_and_extract(target, directory, token)
    return dataset_available(directory)
