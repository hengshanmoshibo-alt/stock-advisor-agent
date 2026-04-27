from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import httpx

from .stock_nodes import StockNodeRecord


YAHOO_TICKER_BY_SOURCE = {
    "brk": "BRK-B",
    "costco": "COST",
    "googl": "GOOGL",
    "ko": "KO",
    "li": "LI",
    "v": "V",
}

QUOTE_INTENT_KEYWORDS = (
    "当前",
    "现在",
    "实时",
    "最新",
    "股价",
    "价格",
    "报价",
    "离",
    "距离",
    "多远",
    "差多少",
    "接近",
    "高多少",
    "低多少",
    "current",
    "price",
    "quote",
    "distance",
)


@dataclass(frozen=True, slots=True)
class QuoteSnapshot:
    source_key: str
    ticker: str
    price: float
    previous_close: float | None = None
    currency: str = "USD"
    market_time: str | None = None
    provider: str = "yahoo_finance"

    @property
    def change(self) -> float | None:
        if self.previous_close is None:
            return None
        return self.price - self.previous_close

    @property
    def change_percent(self) -> float | None:
        if self.previous_close in (None, 0):
            return None
        return (self.price - self.previous_close) / self.previous_close * 100


@dataclass(frozen=True, slots=True)
class NodeDistance:
    label: str
    price: float
    distance_percent: float
    date: str
    article_file: str


_QUOTE_CACHE: dict[tuple[str, str], tuple[datetime, QuoteSnapshot | None]] = {}


def quote_lookup_requested(query: str) -> bool:
    normalized = query.strip().lower()
    return any(keyword in normalized for keyword in QUOTE_INTENT_KEYWORDS)


def ticker_for_source(source_key: str) -> str:
    key = source_key.strip().lower()
    return YAHOO_TICKER_BY_SOURCE.get(key, key.upper())


async def lookup_quote(
    source_key: str,
    *,
    timeout: float = 8.0,
    provider: str = "yahoo",
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> QuoteSnapshot | None:
    ticker = ticker_for_source(source_key)
    normalized_provider = provider.strip().lower()
    cache_key = (normalized_provider, ticker)
    if client is None:
        cached = _QUOTE_CACHE.get(cache_key)
        if cached is not None and datetime.now(tz=timezone.utc) - cached[0] < timedelta(minutes=2):
            return cached[1]
    if normalized_provider == "finnhub":
        quote = await _lookup_finnhub_quote(
            source_key,
            ticker=ticker,
            api_key=api_key,
            timeout=timeout,
            client=client,
        )
    else:
        quote = await _lookup_yahoo_quote(
            source_key,
            ticker=ticker,
            timeout=timeout,
            client=client,
        )
    if client is None:
        _QUOTE_CACHE[cache_key] = (datetime.now(tz=timezone.utc), quote)
    return quote


async def _lookup_yahoo_quote(
    source_key: str,
    *,
    ticker: str,
    timeout: float,
    client: httpx.AsyncClient | None,
) -> QuoteSnapshot | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "1d", "interval": "1m", "includePrePost": "false"}
    close_client = client is None
    http_client = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await http_client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        result = payload["chart"]["result"][0]
        meta = result["meta"]
        price = _first_float(
            meta.get("regularMarketPrice"),
            meta.get("postMarketPrice"),
            meta.get("preMarketPrice"),
            meta.get("chartPreviousClose"),
        )
        if price is None:
            return None
        previous_close = _first_float(
            meta.get("previousClose"),
            meta.get("regularMarketPreviousClose"),
            meta.get("chartPreviousClose"),
        )
        market_time = _market_time(meta.get("regularMarketTime"))
        return QuoteSnapshot(
            source_key=source_key,
            ticker=ticker,
            price=price,
            previous_close=previous_close,
            currency=str(meta.get("currency") or "USD"),
            market_time=market_time,
        )
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return None
    finally:
        if close_client:
            await http_client.aclose()


async def _lookup_finnhub_quote(
    source_key: str,
    *,
    ticker: str,
    api_key: str | None,
    timeout: float,
    client: httpx.AsyncClient | None,
) -> QuoteSnapshot | None:
    if not api_key:
        return None
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": ticker, "token": api_key}
    close_client = client is None
    http_client = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await http_client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        price = _first_float(payload.get("c"))
        if price is None or price <= 0:
            return None
        previous_close = _first_float(payload.get("pc"))
        market_time = _market_time(payload.get("t"))
        return QuoteSnapshot(
            source_key=source_key,
            ticker=ticker,
            price=price,
            previous_close=previous_close,
            currency="USD",
            market_time=market_time,
            provider="finnhub",
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return None
    finally:
        if close_client:
            await http_client.aclose()


def parse_node_prices(nodes: Iterable[str]) -> list[float]:
    prices: list[float] = []
    for node in nodes:
        for match in re.findall(r"\d+(?:\.\d+)?", node):
            price = float(match)
            if 0 < price < 10000:
                prices.append(price)
    return prices


def build_quote_distance_summary(
    quote: QuoteSnapshot,
    records: list[StockNodeRecord],
    *,
    max_nodes: int = 8,
) -> str | None:
    distances = build_node_distances(quote, records, max_nodes=max_nodes)
    if not distances:
        return None

    change_text = ""
    if quote.change is not None and quote.change_percent is not None:
        change_text = f"，较前收盘 {quote.change:+.2f}（{quote.change_percent:+.2f}%）"
    time_text = f"，时间 {quote.market_time}" if quote.market_time else ""
    lines = [
        f"当前 {quote.ticker} 参考价格约 {quote.price:.2f} {quote.currency}{change_text}{time_text}。",
        "相对历史节点的距离：",
    ]
    for item in distances:
        relation = "高于" if item.distance_percent >= 0 else "低于"
        lines.append(
            f"- 距 {item.date} 的 {item.label}（{item.price:g}）约 {abs(item.distance_percent):.2f}% {relation}；{item.article_file}"
        )
    provider_name = "Finnhub" if quote.provider == "finnhub" else "Yahoo Finance"
    lines.append(f"实时行情来自 {provider_name}，仅用于和文章里的历史节点做距离对比。")
    return "\n".join(lines)


def build_node_distances(
    quote: QuoteSnapshot,
    records: list[StockNodeRecord],
    *,
    max_nodes: int = 8,
) -> list[NodeDistance]:
    distances: list[NodeDistance] = []
    seen: set[tuple[str, float]] = set()
    for record in records:
        for node in record.nodes:
            for price in parse_node_prices([node]):
                key = (record.date, round(price, 4))
                if key in seen or price == 0:
                    continue
                seen.add(key)
                distances.append(
                    NodeDistance(
                        label=node,
                        price=price,
                        distance_percent=(quote.price - price) / price * 100,
                        date=record.date,
                        article_file=record.article_file,
                    )
                )
                if len(distances) >= max_nodes:
                    return distances
    return distances


def _first_float(*values: object) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _market_time(value: object) -> str | None:
    try:
        timestamp = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")
