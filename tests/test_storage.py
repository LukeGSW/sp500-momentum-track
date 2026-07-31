"""
Test del recupero dataset.

Il download viene provato davvero, contro un server HTTP locale: e' l'unico
modo per verificare la catena richiesta → tar → estrazione senza dipendere
dalla rete esterna.
"""
from __future__ import annotations

import http.server
import io
import tarfile
import threading
from pathlib import Path

import pandas as pd
import pytest

from track import storage


# ---------------------------------------------------------------------------
def _make_archive(panels: list[str], extra: dict[str, bytes] | None = None) -> bytes:
    """Archivio con la stessa struttura di quello prodotto dal workflow:
    file alla radice dell'archivio, non dentro una cartella."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in panels:
            df = pd.DataFrame(
                {"AAA": [1.0, 2.0], "BBB": [3.0, 4.0]},
                index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
            )
            raw = io.BytesIO()
            df.to_parquet(raw)
            data = raw.getvalue()
            info = tarfile.TarInfo(f"{name}.parquet")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for fname, data in (extra or {}).items():
            info = tarfile.TarInfo(fname)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _Server:
    """Server HTTP effimero che restituisce un singolo payload."""

    def __init__(self, payload: bytes | None):
        self.payload = payload
        body = payload
        status = 200 if payload is not None else 404

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(status)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def log_message(self, *a):  # silenzia il logging del server
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]

    def __enter__(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}/la-pista-data.tar.gz"

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------------
def test_url_asset_release():
    url = storage.release_asset_url("utente/repo")
    assert url == "https://github.com/utente/repo/releases/latest/download/la-pista-data.tar.gz"
    # tollera lo slash finale
    assert storage.release_asset_url("utente/repo/") == url
    assert storage.release_asset_url("utente/repo", "altro.tar.gz").endswith("/altro.tar.gz")


def test_download_ed_estrazione(tmp_path: Path):
    archive = _make_archive(list(storage.PANELS), {"manifest.json": b'{"source":"test"}'})

    assert not storage.dataset_available(tmp_path)
    with _Server(archive) as url:
        names = storage.download_and_extract(url, tmp_path)

    assert storage.dataset_available(tmp_path)
    assert "manifest.json" in names
    assert storage.load_manifest(tmp_path)["source"] == "test"
    assert len(storage.missing_panels(tmp_path)) == 0

    df = storage.load_panel("close_adj", tmp_path)
    assert list(df.columns) == ["AAA", "BBB"]


def test_archivio_incompleto_viene_rilevato(tmp_path: Path):
    """Un archivio a cui manca un pannello non deve risultare 'pronto'."""
    parziale = [p for p in storage.PANELS if p != "bands"]
    with _Server(_make_archive(parziale)) as url:
        storage.download_and_extract(url, tmp_path)

    assert not storage.dataset_available(tmp_path)
    assert storage.missing_panels(tmp_path) == ["bands"]


def test_404_messaggio_esplicito(tmp_path: Path):
    with _Server(None) as url:
        with pytest.raises(FileNotFoundError, match="Nessun asset trovato"):
            storage.download_and_extract(url, tmp_path)


def test_ensure_dataset_non_scarica_se_gia_presente(tmp_path: Path):
    with _Server(_make_archive(list(storage.PANELS))) as url:
        assert storage.ensure_dataset(tmp_path, url=url)
        # il server e' chiuso subito dopo: se riprovasse a scaricare, fallirebbe
    assert storage.ensure_dataset(tmp_path, url="http://127.0.0.1:1/non-esiste")


def test_ensure_dataset_senza_sorgente_non_fa_nulla(tmp_path: Path):
    assert storage.ensure_dataset(tmp_path) is False


def test_force_riscarica_anche_se_i_dati_ci_sono(tmp_path: Path):
    """Senza force, una Release nuova non arriverebbe mai a un container che ha
    gia' scaricato: l'app servirebbe dati vecchi credendoli aggiornati."""
    vecchio = _make_archive(list(storage.PANELS), {"manifest.json": b'{"built_at":"vecchio"}'})
    with _Server(vecchio) as url:
        assert storage.ensure_dataset(tmp_path, url=url)
    assert storage.load_manifest(tmp_path)["built_at"] == "vecchio"

    nuovo = _make_archive(list(storage.PANELS), {"manifest.json": b'{"built_at":"nuovo"}'})
    with _Server(nuovo) as url:
        # senza force il manifest resta quello vecchio
        assert storage.ensure_dataset(tmp_path, url=url)
        assert storage.load_manifest(tmp_path)["built_at"] == "vecchio"
        # con force viene sostituito
        assert storage.ensure_dataset(tmp_path, url=url, force=True)
        assert storage.load_manifest(tmp_path)["built_at"] == "nuovo"


def test_secrets_toml_con_bom(tmp_path: Path, monkeypatch):
    """Su Windows molti editor scrivono il BOM: non deve rompere la lettura.

    Senza tolleranza al BOM, tomllib fallisce con 'Invalid statement at line 1,
    column 1' — un messaggio che non suggerisce minimamente la causa reale.
    """
    from track import storage as st_mod

    conf = tmp_path / ".streamlit"
    conf.mkdir()
    (conf / "secrets.toml").write_text(
        'DATA_URL = "https://esempio/archivio.tar.gz"\n', encoding="utf-8-sig"
    )
    assert (conf / "secrets.toml").read_bytes().startswith(b"\xef\xbb\xbf"), "il test deve avere il BOM"

    monkeypatch.delenv("DATA_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    assert st_mod.read_secret("DATA_URL") == "https://esempio/archivio.tar.gz"


def test_variabile_ambiente_ha_la_precedenza(tmp_path: Path, monkeypatch):
    from track import storage as st_mod

    conf = tmp_path / ".streamlit"
    conf.mkdir()
    (conf / "secrets.toml").write_text('DATA_URL = "dal-file"\n', encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_URL", "dall-ambiente")
    assert st_mod.read_secret("DATA_URL") == "dall-ambiente"


def test_secret_assente_ritorna_none(tmp_path: Path, monkeypatch):
    from track import storage as st_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHIAVE_INESISTENTE", raising=False)
    assert st_mod.read_secret("CHIAVE_INESISTENTE") is None


def test_estrazione_blocca_il_path_traversal(tmp_path: Path):
    """Un archivio che tenta di scrivere fuori dalla cartella dati va respinto.

    L'archivio arriva dalla rete: non va estratto alla cieca.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"payload"
        info = tarfile.TarInfo("../../evil.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with _Server(buf.getvalue()) as url:
        with pytest.raises(Exception):  # noqa: B017 - tarfile solleva vari tipi
            storage.download_and_extract(url, tmp_path)

    assert not (tmp_path.parent / "evil.txt").exists()
    assert not (tmp_path / ".." / ".." / "evil.txt").resolve().exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
