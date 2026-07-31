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

# Pannelli attesi dall'app.
PANELS = (
    "close_adj",
    "open_adj",
    "open_raw",
    "membership",
    "force",
    "velocity",
    "bands",
    "eligible",
)


def data_dir(override: str | Path | None = None) -> Path:
    p = Path(override) if override else DEFAULT_DATA_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


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
