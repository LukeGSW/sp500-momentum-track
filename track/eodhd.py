"""
Client EODHD.

Usato SOLO dalla pipeline offline (cartella `pipeline/`). L'app Streamlit non
scarica nulla a runtime: legge gli artefatti Parquet prodotti dalla pipeline.
L'unica eccezione ammessa e' `bulk_last_day()`, che restituisce l'intero
mercato US in una singola chiamata.

Endpoint usati (verificare la disponibilita' sul proprio piano):
  fundamentals/GSPC.INDX?filter=HistoricalTickerComponents  -> costituenti storici
  eod/{SYM}.US                                              -> serie OHLC + adjusted
  eod-bulk-last-day/US                                      -> ultimo giorno, tutto il mercato
  exchange-symbol-list/US?delisted=1                        -> elenco delistati
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://eodhd.com/api"


def resolve_api_key() -> str:
    """Cerca la chiave in ambiente e nei secrets.toml (progetto e cartella corrente).

    La chiave non compare MAI negli artefatti ne' nei log. In GitHub Actions
    va passata come repository secret `EODHD_API_KEY`.
    """
    from .storage import read_secret

    key = read_secret("EODHD_API_KEY")
    if key:
        return key

    raise RuntimeError(
        "Chiave EODHD non trovata. Imposta la variabile d'ambiente EODHD_API_KEY "
        "oppure crea .streamlit/secrets.toml con la riga  EODHD_API_KEY = \"...\""
    )


class _RateLimiter:
    """Token bucket semplice, thread-safe: al massimo `per_minute` richieste/minuto."""

    def __init__(self, per_minute: int = 600):
        self._min_interval = 60.0 / max(per_minute, 1)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class EODHDError(RuntimeError):
    pass


class EODHDClient:
    def __init__(
        self,
        api_key: str,
        *,
        max_workers: int = 8,
        per_minute: int = 600,
        max_retries: int = 4,
        timeout: int = 45,
    ):
        if not api_key:
            raise ValueError("API key EODHD mancante")
        self.api_key = api_key
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self._limiter = _RateLimiter(per_minute)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "la-pista/1.0"})

    # ------------------------------------------------------------------ core
    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        params.setdefault("fmt", "json")
        params["api_token"] = self.api_key
        url = f"{BASE_URL}/{path.lstrip('/')}"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._limiter.acquire()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # rete
                last_exc = exc
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise EODHDError(f"risposta non JSON da {path}") from exc
            if resp.status_code == 404:
                return None  # simbolo inesistente: non e' un errore fatale
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt + 1)
                last_exc = EODHDError(f"HTTP {resp.status_code} su {path}")
                continue
            raise EODHDError(f"HTTP {resp.status_code} su {path}: {resp.text[:200]}")

        raise EODHDError(f"{path}: esauriti i tentativi") from last_exc

    # --------------------------------------------------------- costituenti
    def historical_constituents(self, index_symbol: str = "GSPC.INDX") -> pd.DataFrame:
        """Storico completo delle appartenenze all'indice.

        Colonne restituite: code, name, sector, industry, start_date, end_date,
        is_active_now, is_delisted.

        ATTENZIONE: la copertura EODHD per GSPC.INDX parte da **gennaio 2000**.
        Serie precedenti non sono ricostruibili con questa fonte.
        """
        raw = self._get(
            f"fundamentals/{index_symbol}",
            {"filter": "HistoricalTickerComponents"},
        )
        if not raw:
            raise EODHDError(
                "HistoricalTickerComponents vuoto: il piano EODHD potrebbe non "
                "includere i costituenti storici degli indici."
            )
        rows = raw.values() if isinstance(raw, dict) else raw
        recs = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            recs.append(
                {
                    "code": r.get("Code"),
                    "name": r.get("Name"),
                    "exchange": r.get("Exchange"),
                    "sector": r.get("Sector"),
                    "industry": r.get("Industry"),
                    "start_date": r.get("StartDate"),
                    "end_date": r.get("EndDate"),
                    "is_active_now": int(r.get("IsActiveNow") or 0),
                    "is_delisted": int(r.get("IsDelisted") or 0),
                }
            )
        df = pd.DataFrame.from_records(recs).dropna(subset=["code"])
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        return df.reset_index(drop=True)

    def current_constituents(self, index_symbol: str = "GSPC.INDX") -> pd.DataFrame:
        raw = self._get(f"fundamentals/{index_symbol}", {"filter": "Components"})
        rows = (raw or {}).values() if isinstance(raw, dict) else (raw or [])
        recs = [
            {
                "code": r.get("Code"),
                "name": r.get("Name"),
                "sector": r.get("Sector"),
                "industry": r.get("Industry"),
            }
            for r in rows
            if isinstance(r, dict)
        ]
        return pd.DataFrame.from_records(recs)

    # ---------------------------------------------------------------- prezzi
    def eod(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame | None:
        """Serie giornaliera OHLC + adjusted_close per un simbolo (es. 'AAPL.US')."""
        params = {"from": start, "period": "d", "order": "a"}
        if end:
            params["to"] = end
        raw = self._get(f"eod/{symbol}", params)
        if not raw:
            return None
        df = pd.DataFrame(raw)
        if df.empty or "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        keep = [c for c in ("date", "open", "high", "low", "close", "adjusted_close", "volume") if c in df.columns]
        df = df[keep].set_index("date").sort_index()
        return df.astype("float64", errors="ignore")

    def eod_many(
        self,
        symbols: Iterable[str],
        start: str,
        end: str | None = None,
        progress: Any | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Download parallelo. `progress` puo' essere una callable(done, total)."""
        symbols = list(symbols)
        out: dict[str, pd.DataFrame] = {}
        done = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.eod, s, start, end): s for s in symbols}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    df = fut.result()
                except Exception as exc:  # noqa: BLE001 - un simbolo non deve fermare tutto
                    log.warning("download fallito per %s: %s", sym, exc)
                    df = None
                if df is not None and not df.empty:
                    out[sym] = df
                done += 1
                if progress is not None:
                    progress(done, len(symbols))
        return out

    def bulk_last_day(self, exchange: str = "US") -> pd.DataFrame:
        """Ultimo giorno di contrattazione per l'intero mercato, in UNA chiamata."""
        raw = self._get(f"eod-bulk-last-day/{exchange}")
        df = pd.DataFrame(raw or [])
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def delisted_symbols(self, exchange: str = "US") -> pd.DataFrame:
        raw = self._get(f"exchange-symbol-list/{exchange}", {"delisted": 1})
        return pd.DataFrame(raw or [])

    # ------------------------------------------------------------- risk free
    def risk_free_daily(self, start: str, end: str | None = None) -> pd.Series:
        """Rendimento giornaliero del T-bill 3 mesi.

        Prova alcuni simboli candidati e degrada con grazia. Il fallback e' una
        serie costante a 2% annuo, che va considerata un tappabuchi, non un dato:
        se viene usato, la pipeline lo registra nel manifest.
        """
        for sym in ("US3M.GBOND", "IRX.INDX", "US3M.INDX"):
            try:
                df = self.eod(sym, start, end)
            except EODHDError:
                df = None
            if df is not None and not df.empty and "close" in df.columns:
                annual_pct = df["close"].astype(float)
                # I simboli sopra quotano il rendimento in percentuale annua.
                daily = (1.0 + annual_pct / 100.0) ** (1.0 / 252.0) - 1.0
                daily.name = "rf"
                daily.attrs["source"] = sym
                return daily
        log.warning("nessun simbolo risk-free disponibile: fallback costante 2%%")
        idx = pd.date_range(start, end or pd.Timestamp.today(), freq="B")
        s = pd.Series((1.02) ** (1 / 252) - 1.0, index=idx, name="rf")
        s.attrs["source"] = "FALLBACK_COSTANTE_2PCT"
        return s
