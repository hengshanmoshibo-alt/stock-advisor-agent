from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .quote_lookup import ticker_for_source


@dataclass(frozen=True, slots=True)
class MarketCandle:
    timestamp: int
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataClient:
    def __init__(
        self,
        *,
        provider: str = "finnhub",
        api_key: str | None = None,
        timeout: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.provider = provider.strip().lower()
        self.api_key = api_key
        self.timeout = timeout
        self.client = client
        self._cache: dict[tuple[str, str, int], tuple[datetime, list[MarketCandle]]] = {}

    async def get_daily_candles(self, source_key: str, *, lookback_days: int = 420) -> list[MarketCandle]:
        if self.provider not in {"finnhub", "massive"} or not self.api_key:
            return []
        ticker = ticker_for_source(source_key)
        cache_key = (self.provider, ticker, lookback_days)
        cached = self._cache.get(cache_key)
        if cached is not None and datetime.now(tz=timezone.utc) - cached[0] < timedelta(minutes=30):
            return list(cached[1])
        now = datetime.now(tz=timezone.utc)
        start = now - timedelta(days=lookback_days)
        if self.provider == "massive":
            candles = await self._get_massive_daily_candles(
                ticker,
                start_date=start.date().isoformat(),
                end_date=now.date().isoformat(),
            )
        else:
            candles = await self._get_finnhub_daily_candles(
                ticker,
                start_timestamp=int(start.timestamp()),
                end_timestamp=int(now.timestamp()),
            )
        if candles:
            self._cache[cache_key] = (datetime.now(tz=timezone.utc), list(candles))
        return candles

    async def _get_finnhub_daily_candles(
        self,
        ticker: str,
        *,
        start_timestamp: int,
        end_timestamp: int,
    ) -> list[MarketCandle]:
        close_client = self.client is None
        http_client = self.client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await http_client.get(
                "https://finnhub.io/api/v1/stock/candle",
                params={
                    "symbol": ticker,
                    "resolution": "D",
                    "from": str(start_timestamp),
                    "to": str(end_timestamp),
                    "token": self.api_key,
                },
            )
            response.raise_for_status()
            return parse_finnhub_candles(response.json())
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return []
        finally:
            if close_client:
                await http_client.aclose()

    async def _get_massive_daily_candles(
        self,
        ticker: str,
        *,
        start_date: str,
        end_date: str,
    ) -> list[MarketCandle]:
        close_client = self.client is None
        http_client = self.client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await http_client.get(
                f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}",
                params={
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": "50000",
                    "apiKey": self.api_key,
                },
            )
            response.raise_for_status()
            return parse_massive_candles(response.json())
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return []
        finally:
            if close_client:
                await http_client.aclose()


def parse_finnhub_candles(payload: dict[str, object]) -> list[MarketCandle]:
    if payload.get("s") != "ok":
        return []
    opens = _float_list(payload.get("o"))
    highs = _float_list(payload.get("h"))
    lows = _float_list(payload.get("l"))
    closes = _float_list(payload.get("c"))
    volumes = _float_list(payload.get("v"))
    timestamps = _int_list(payload.get("t"))
    row_count = min(
        len(opens),
        len(highs),
        len(lows),
        len(closes),
        len(volumes),
        len(timestamps),
    )
    candles = [
        MarketCandle(
            timestamp=timestamps[index],
            date=datetime.fromtimestamp(timestamps[index], tz=timezone.utc).date().isoformat(),
            open=opens[index],
            high=highs[index],
            low=lows[index],
            close=closes[index],
            volume=volumes[index],
        )
        for index in range(row_count)
    ]
    candles.sort(key=lambda item: item.timestamp)
    return candles


def parse_massive_candles(payload: dict[str, object]) -> list[MarketCandle]:
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    candles: list[MarketCandle] = []
    for row in results:
        if not isinstance(row, dict):
            return []
        candle = _parse_massive_row(row)
        if candle is None:
            return []
        candles.append(candle)
    candles.sort(key=lambda item: item.timestamp)
    return candles


def _parse_massive_row(row: dict[str, Any]) -> MarketCandle | None:
    timestamp_ms = _first_int(row.get("t"))
    open_price = _first_float(row.get("o"))
    high_price = _first_float(row.get("h"))
    low_price = _first_float(row.get("l"))
    close_price = _first_float(row.get("c"))
    volume = _first_float(row.get("v"))
    if None in {timestamp_ms, open_price, high_price, low_price, close_price, volume}:
        return None
    timestamp = int(timestamp_ms // 1000)
    return MarketCandle(
        timestamp=timestamp,
        date=datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat(),
        open=float(open_price),
        high=float(high_price),
        low=float(low_price),
        close=float(close_price),
        volume=float(volume),
    )


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return []
    return result


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return []
    return result


def _first_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
