from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

from .fundamental_data import FundamentalDataClient, FundamentalSnapshot
from .market_context import MarketContext, build_market_context
from .market_data import MarketDataClient, MarketCandle
from .quote_lookup import QuoteSnapshot, lookup_quote
from .trade_scoring import sector_etfs_for_source


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ToolResult(Generic[T]):
    name: str
    ok: bool
    data: T | None
    source: str
    timestamp: float
    elapsed_ms: int
    error: str = ""


@dataclass(frozen=True, slots=True)
class TradePlanToolBundle:
    quote: ToolResult[QuoteSnapshot]
    candles: ToolResult[list[MarketCandle]]
    fundamentals: ToolResult[FundamentalSnapshot]
    market_context: ToolResult[MarketContext]

    def as_log_facts(self) -> dict[str, object]:
        return {
            "quote": _tool_log(self.quote),
            "candles": _tool_log(self.candles, data_size=len(self.candles.data or [])),
            "fundamentals": _tool_log(self.fundamentals),
            "market_context": _tool_log(self.market_context),
        }


class TradePlanToolRunner:
    """Runs the bounded tool set used by the controlled stock-advisor Agent."""

    def __init__(
        self,
        *,
        market_data: MarketDataClient,
        fundamentals: FundamentalDataClient,
        quote_timeout: float,
        finnhub_api_key: str | None,
        quote_lookup_fn: Callable[..., Awaitable[QuoteSnapshot | None]] = lookup_quote,
    ) -> None:
        self.market_data = market_data
        self.fundamentals = fundamentals
        self.quote_timeout = quote_timeout
        self.finnhub_api_key = finnhub_api_key
        self.quote_lookup = quote_lookup_fn

    async def run(self, source_key: str) -> TradePlanToolBundle:
        quote, candles, fundamentals, market_context = await asyncio.gather(
            _run_tool(
                "quote_lookup",
                source=self._quote_source(),
                call=lambda: self.quote_lookup(
                    source_key,
                    timeout=self.quote_timeout,
                    provider="finnhub",
                    api_key=self.finnhub_api_key,
                ),
            ),
            _run_tool(
                "market_candles",
                source=self.market_data.provider,
                call=lambda: self.market_data.get_daily_candles(source_key, lookback_days=420),
                ok_when=lambda value: bool(value),
            ),
            _run_tool(
                "fundamentals",
                source="finnhub",
                call=lambda: self.fundamentals.get_snapshot(source_key),
            ),
            _run_tool(
                "market_context",
                source=self.market_data.provider,
                call=lambda: build_market_context(
                    self.market_data,
                    extra_tickers=sector_etfs_for_source(source_key),
                ),
            ),
        )
        return TradePlanToolBundle(
            quote=quote,
            candles=candles,
            fundamentals=fundamentals,
            market_context=market_context,
        )

    def _quote_source(self) -> str:
        return "finnhub" if self.finnhub_api_key else "unavailable"


async def _run_tool(
    name: str,
    *,
    source: str,
    call: Callable[[], Awaitable[T | None]],
    ok_when: Callable[[T | None], bool] | None = None,
) -> ToolResult[T]:
    started = time.perf_counter()
    try:
        data = await call()
        ok = ok_when(data) if ok_when is not None else data is not None
        return ToolResult(
            name=name,
            ok=ok,
            data=data,
            source=source,
            timestamp=time.time(),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            error="" if ok else "empty_result",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            name=name,
            ok=False,
            data=None,
            source=source,
            timestamp=time.time(),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            error=type(exc).__name__,
        )


def _tool_log(result: ToolResult[Any], *, data_size: int | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": result.ok,
        "source": result.source,
        "elapsed_ms": result.elapsed_ms,
    }
    if result.error:
        payload["error"] = result.error
    if data_size is not None:
        payload["data_size"] = data_size
    return payload
